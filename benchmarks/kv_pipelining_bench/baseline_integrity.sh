#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Is arm A a real baseline, or a handicapped one?
#
# THE PROBLEM. The paper reports ratios against "our fork with tiering off".
# Our fork adds ~3,900 lines across 22 files in vllm/. If any of it costs time
# on the DISABLED path -- a per-step check, an attribute lookup in the runner's
# hot loop, a branch in attention -- then arm A is slower than the stock engine
# it stands in for, and part of every reported win is self-inflicted. A
# handicapped baseline produces perfectly plausible numbers, so this needs a
# measurement rather than a reading of the code.
#
# Not hypothetical here: four loop-invariant computations were found inside the
# per-layer loop in 2026-09, worth ~21 ms/step, and only an operator trace
# found them. The same mistake on the disabled path would be invisible.
#
# WHY NOT pd_decode_goodput.py. Its "no_riders" arm is NOT a stock engine: it
# passes --enable-hybrid-kv-tiers, --enable-kv-pipelining and the hybrid env
# vars for BOTH arms, differing only in b_hbm_max. That is the right control
# for "what do riders cost" -- both arms carry identical machinery -- but it
# cannot answer "what does our fork cost when off", and against a stock tree it
# does not even start, because those flags do not parse. (Discovered by running
# it; the arms silently returned nan.)
#
# THE TEST. `main` is this fork immediately before the tiering work. Our
# changes are pure Python, so the two revisions swap in place with no rebuild.
# Both arms run a PLAIN OpenAI server with no hybrid flags at all, measured
# with `vllm bench serve`, which ships in both revisions and reports the same
# goodput metric the evaluation uses.
#
#   STOCK   vllm/ at main          -- the engine we claim to compare against
#   OFF     vllm/ at HEAD          -- our fork, feature paths never enabled
#
# Usage:  bash benchmarks/kv_pipelining_bench/baseline_integrity.sh
#         MODEL=... NUM_PROMPTS=... REPEATS=2 RATE=... bash .../baseline_integrity.sh
#
# Exit 0 = arm A is sound. 1 = handicapped. 2 = inconclusive.

set -u
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-.venv/bin/python}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
NUM_PROMPTS=${NUM_PROMPTS:-64}
RATE=${RATE:-8}                 # requests/s, open loop
IN_LEN=${IN_LEN:-512}
OUT_LEN=${OUT_LEN:-64}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.30}
REPEATS=${REPEATS:-2}
TOL=${TOL:-3}
PORT=${PORT:-9971}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/mnt/beegfs/benishay/huggingface}
export HF_TOKEN="$(cat hf_token 2>/dev/null | tr -d '\n')"
OUTDIR=${OUTDIR:-/mnt/beegfs/benishay/paper_runs/20260912_validate}
mkdir -p "$OUTDIR"

HEAD_REF=$(git rev-parse --abbrev-ref HEAD)
BASE_REF=${BASE_REF:-main}

if [ -n "$(git status --porcelain -- vllm/)" ]; then
  echo "REFUSING: vllm/ has uncommitted changes; this script swaps vllm/ in place."
  exit 2
fi

SERVER_PID=""
cleanup() {
  [ -n "$SERVER_PID" ] && kill -9 "$SERVER_PID" 2>/dev/null
  pkill -9 -f "api_server.*--port $PORT" 2>/dev/null
  echo ""; echo "restoring vllm/ to $HEAD_REF ..."
  git checkout "$HEAD_REF" -- vllm/ 2>/dev/null
  echo "restored: $([ -z "$(git status --porcelain -- vllm/)" ] && echo clean || echo DIRTY)"
}
trap cleanup EXIT INT TERM

run_arm() {  # run_arm <label> -> echoes goodput-ish "req_s tok_s"
  local label="$1"
  local res="$OUTDIR/bi_${label}.json"
  rm -f "$res"
  # PLAIN server. No hybrid flags -- that is the whole point, and it is also
  # the only way the stock revision will start at all.
  $PY -m vllm.entrypoints.openai.api_server --model "$MODEL" \
      --port "$PORT" --max-model-len $((IN_LEN + OUT_LEN + 64)) \
      --gpu-memory-utilization "$GPU_MEM_UTIL" --dtype float16 \
      --enforce-eager --disable-log-requests --no-enable-prefix-caching \
      > "$OUTDIR/server_${label}.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 120); do
    curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 2
  done
  $PY -m vllm.entrypoints.cli.main bench serve --backend openai \
      --model "$MODEL" --port "$PORT" \
      --dataset-name random --random-input-len "$IN_LEN" \
      --random-output-len "$OUT_LEN" --num-prompts "$NUM_PROMPTS" \
      --request-rate "$RATE" --ignore-eos \
      --save-result --result-filename "$res" \
      > "$OUTDIR/bench_${label}.log" 2>&1
  kill -9 "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""
  sleep 5
  $PY - "$res" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(f"{d['request_throughput']:.3f} {d['output_throughput']:.1f}")
except Exception:
    print("nan nan")
EOF
}

echo "============================================================"
echo " BASELINE INTEGRITY   $MODEL"
echo " STOCK = vllm/ at $BASE_REF   OFF = vllm/ at $HEAD_REF (no hybrid flags)"
echo " $NUM_PROMPTS prompts at $RATE req/s, $REPEATS repeats, interleaved"
echo "============================================================"

declare -a S_R O_R
for i in $(seq 1 "$REPEATS"); do
  # Interleaved, so drift cannot be read as a code difference.
  git checkout "$BASE_REF" -- vllm/ 2>/dev/null
  read -r s_req s_tok <<<"$(run_arm stock)"
  git checkout "$HEAD_REF" -- vllm/ 2>/dev/null
  read -r o_req o_tok <<<"$(run_arm off)"
  S_R+=("$s_req"); O_R+=("$o_req")
  printf "  rep %d   STOCK %8s req/s (%s tok/s)   OFF %8s req/s (%s tok/s)\n" \
         "$i" "$s_req" "$s_tok" "$o_req" "$o_tok"
done

$PY - "$TOL" "${S_R[@]}" -- "${O_R[@]}" <<'EOF'
import sys
tol = float(sys.argv[1]); rest = sys.argv[2:]
i = rest.index("--")
def nums(xs):
    out = []
    for x in xs:
        try: out.append(float(x))
        except ValueError: pass
    return [v for v in out if v == v]
stock, off = nums(rest[:i]), nums(rest[i+1:])
if not stock or not off:
    print("\nINCONCLUSIVE: an arm produced no number. Check the server logs.")
    sys.exit(2)
ms, mo = sum(stock)/len(stock), sum(off)/len(off)
d = (mo/ms - 1) * 100
print(f"\n  STOCK mean {ms:7.3f} req/s")
print(f"  OFF   mean {mo:7.3f} req/s")
print(f"  disabled-path cost: {d:+.2f}%\n")
if abs(d) <= tol:
    print(f"PASS -- arm A is stock within {tol:.0f}%. Ratios against it are honest.")
    sys.exit(0)
print(f"FAIL -- {abs(d):.1f}% {'SLOWER' if d < 0 else 'FASTER'} with the feature off.")
print("Either find the cost on the disabled path, or make the paper's baseline")
print("the STOCK tree and say so.")
sys.exit(1)
EOF
