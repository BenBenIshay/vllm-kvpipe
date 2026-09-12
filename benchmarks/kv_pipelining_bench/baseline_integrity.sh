#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Is arm A a real baseline, or a handicapped one?
#
# THE PROBLEM. Experiment 4.0/4.1's arm A is "our fork with the tiers turned
# off". Every number the paper reports is a ratio against it. But our fork adds
# ~3,900 lines across 22 files in vllm/, and if any of that costs time even
# when the feature is disabled -- a per-step check, an extra attribute lookup
# in the model runner's hot loop, a branch in attention -- then arm A is slower
# than the stock engine we claim to be comparing against, and part of our
# reported win is self-inflicted. Nobody would catch it from the output: a
# handicapped baseline produces perfectly plausible numbers.
#
# This is not hypothetical for this codebase. Four loop-invariant computations
# were found inside the per-layer loop in 2026-09, worth ~21 ms/step, and they
# were only found by an operator trace. The same species of mistake sitting on
# the disabled path would be invisible.
#
# THE TEST. `main` (f417a0e25) is this fork immediately before the tiering work
# began. Our changes are pure Python, so the two versions can be swapped in
# place without a rebuild:
#
#   arm STOCK   vllm/ at main                      -- the real baseline
#   arm OFF     vllm/ at HEAD, all features off    -- what 4.0 actually uses
#
# Same model, same workload, same GPU, back to back. If they differ by more
# than noise, arm A is not stock vLLM and the plan must either fix the overhead
# or report against STOCK instead.
#
# WHY THIS MATTERS FOR COMPARABILITY. Pie's headline is 1.9x over vLLM and Pie
# is built on vLLM, so their baseline has the same shape as ours; DirectKV
# instead uses SGLang as its HBM-resident baseline. If our arm A is quietly
# slower than stock vLLM, our ratio is not comparable to either of theirs.
#
# Usage:
#   bash benchmarks/kv_pipelining_bench/baseline_integrity.sh
#   MODEL=... CTX=... REPEATS=3 bash .../baseline_integrity.sh
#
# Exit 0 = arm A is sound. 1 = it is handicapped; read the delta.

set -u
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-.venv/bin/python}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
CTX=${CTX:-1024}
OUT_LEN=${OUT_LEN:-32}
OFFERED=${OFFERED:-16}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.30}
REPEATS=${REPEATS:-3}
TOL=${TOL:-3}          # percent; above this arm A is handicapped
export HF_HUB_CACHE=${HF_HUB_CACHE:-/mnt/beegfs/benishay/huggingface}
export HF_TOKEN="$(cat hf_token 2>/dev/null | tr -d '\n')"

HEAD_REF=$(git rev-parse --abbrev-ref HEAD)
BASE_REF=${BASE_REF:-main}

if [ -n "$(git status --porcelain -- vllm/)" ]; then
  echo "REFUSING: vllm/ has uncommitted changes. This script swaps vllm/"
  echo "between revisions in place and would destroy them."
  exit 2
fi

# The tree MUST come back, whatever happens -- including Ctrl-C. A run that
# dies halfway through leaves the engine importing a half-old vllm/, and every
# subsequent measurement on this machine would be silently wrong.
restore() {
  echo ""
  echo "restoring vllm/ to $HEAD_REF ..."
  git checkout "$HEAD_REF" -- vllm/ 2>/dev/null
  git status --porcelain -- vllm/ | head -3
  echo "restored."
}
trap restore EXIT INT TERM

run_arm() {  # run_arm <label> ; echoes "tok_s itl_ms"
  # NOT one `local` statement: bash declares every name in a `local` first and
  # only then assigns, so ${label} is still unbound when json is evaluated --
  # which under `set -u` aborts the function and yields an empty measurement.
  local label="$1"
  local json="/tmp/bi_${label}_$$.json"
  $PY benchmarks/kv_pipelining_bench/pd_decode_goodput.py \
      --model "$MODEL" --ctx "$CTX" --out-len "$OUT_LEN" \
      --offered "$OFFERED" --gpu-mem-util "$GPU_MEM_UTIL" \
      --arms no_riders --out "$json" >/dev/null 2>&1
  $PY - "$json" <<'EOF'
import json, sys
try:
    r = json.load(open(sys.argv[1]))["rows"][0]
    print(f"{r['decode_tok_s']:.2f} {r['itl_median_ms']:.2f}")
except Exception:
    print("nan nan")
EOF
}

echo "============================================================"
echo " BASELINE INTEGRITY   $MODEL  ctx=$CTX offered=$OFFERED"
echo " STOCK = vllm/ at $BASE_REF      OFF = vllm/ at $HEAD_REF, features off"
echo " $REPEATS repeats each, interleaved"
echo "============================================================"

declare -a STOCK_T OFF_T
for i in $(seq 1 "$REPEATS"); do
  # Interleaved, not blocked: a thermal or contention drift that happens to
  # land on one arm would otherwise be read as a code difference.
  git checkout "$BASE_REF" -- vllm/ 2>/dev/null
  read -r s_tok s_itl <<<"$(run_arm stock)"
  git checkout "$HEAD_REF" -- vllm/ 2>/dev/null
  read -r o_tok o_itl <<<"$(run_arm off)"
  STOCK_T+=("$s_tok"); OFF_T+=("$o_tok")
  printf "  rep %d   STOCK %8s tok/s (%s ms)   OFF %8s tok/s (%s ms)\n" \
         "$i" "$s_tok" "$s_itl" "$o_tok" "$o_itl"
done

$PY - "$TOL" "${STOCK_T[@]}" -- "${OFF_T[@]}" <<'EOF'
import sys
tol = float(sys.argv[1])
rest = sys.argv[2:]
i = rest.index("--")
stock = [float(x) for x in rest[:i] if x == x]
off   = [float(x) for x in rest[i+1:] if x == x]
if not stock or not off or any(x != x for x in stock + off):
    print("\nINCONCLUSIVE: an arm failed to produce a number.")
    sys.exit(2)
ms, mo = sum(stock)/len(stock), sum(off)/len(off)
delta = (mo/ms - 1) * 100
print(f"\n  STOCK mean {ms:8.2f} tok/s   (spread {max(stock)-min(stock):.2f})")
print(f"  OFF   mean {mo:8.2f} tok/s   (spread {max(off)-min(off):.2f})")
print(f"  disabled-path cost: {delta:+.2f}%")
print()
if abs(delta) <= tol:
    print(f"PASS -- arm A is stock within {tol:.0f}%. Ratios against it are honest.")
    sys.exit(0)
if delta < 0:
    print(f"FAIL -- our fork is {abs(delta):.1f}% SLOWER with every feature off.")
    print("Arm A is handicapped, so part of any reported win is self-inflicted.")
    print("Either find the overhead on the disabled path, or make arm A the")
    print("STOCK tree and say so in the paper.")
    sys.exit(1)
print(f"FAIL -- our fork is {delta:.1f}% FASTER with every feature off, which")
print("should not happen and means the two arms are not running the same")
print("workload. Check that the feature really was disabled in both.")
sys.exit(1)
EOF
rc=$?
exit $rc
