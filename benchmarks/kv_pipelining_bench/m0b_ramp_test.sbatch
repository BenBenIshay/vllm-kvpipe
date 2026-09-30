#!/bin/bash
# M0b -- is the paper's "not affine" rise real, or admission-ramp residue?
#
# THE QUESTION. Section 6.5 reports dT per rider RISING (9.06, 11.01, 15.56 ms at
# r = 2, 4, 6) and concludes from it that dT = F + cr "fits a negative fixed cost
# and the fixed-versus-slope split is not a meaningful decomposition". Arm A of
# M0 reproduced those numbers to 0.35%, and the purity gate then showed why they
# might not mean that: admission STAGGERS, queueing max_waiting = r-1, and the
# step count grows 136 -> 138 -> 142 -> 146, i.e. about 2(r-1) extra steps.
#
# The estimator is (wall_long - wall_short) / (long_len - short_len) and assumes
# the ramp is identical in both runs so it cancels. With --short-len 8 and a ramp
# of ~10 steps at r=6, the SHORT run never reaches steady state: its ramp is
# truncated, R_long - R_short lands in the numerator, and the step is
# overestimated by an amount that GROWS WITH r -- which is exactly a rising
# per-rider cost.
#
# THE TEST. Raise --short-len until both runs clear the ramp and see whether the
# rise survives. LONG_LEN IS HELD FIXED AT 136, so short_len is the only thing
# that varies:
#
#   short 8   long 136   <- M0 arm A, already measured: the published shape
#   short 24  long 136   <- clears a ~10-step ramp
#   short 48  long 136   <- clears it with a large margin
#
# The first attempt at this held the DIFFERENCE at 128 instead (48/176, 96/224)
# and that was wrong, for a reason worth recording. The KV pool is
# "GPU KV cache size: 35,472 tokens". Arm A's 16 residents at ctx 2048 +
# long_len 136 need 16 * 2184 = 34,944 -- it fits with 1.5% to spare. At
# long_len 176 they need 16 * 2224 = 35,584, which does NOT fit, and at 224 it is
# 36,352. The engine then evicts and RECOMPUTES: the purity gate caught "prefill
# leaked: 2208 tokens over 1 steps" (one request re-prefilling 2048 prompt plus
# ~160 generated), 15 extra steps at r=0, a baseline moved 53.45 -> 64.21 ms, and
# reps ranging 45 to 90 ms at r=2. All artifact, none of it about riders.
#
# So the published configuration sits at 98.5% pool occupancy and anything that
# lengthens a sequence tips it over. Holding long_len fixed keeps the pool state
# identical across arms, which also makes this a one-variable experiment. The
# cost is a smaller token difference (128 / 112 / 88), i.e. slightly more noise
# in the wall-clock difference -- cheap next to measuring a different regime.
#
# If dT stabilises as short_len rises, the original was biased and the rise is an
# artifact. If it persists at short 48, it is real and the paper's kink model
# stands. I am not assuming either.
#
# No delay command anywhere: this cluster's submit filter rejects any batch
# script containing one.
#
#SBATCH --job-name=ovh-m0b
#SBATCH --partition=supernova
#SBATCH --nodelist=nova07
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m0b_ramp_%j.out

set -u
TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/00b_ramp
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

echo "========= M0b ramp test: $(hostname) $(date -Is) ========="
VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "tree OK" ;;
  *) echo "FATAL: wrong tree ($VLLM_PATH)"; exit 10 ;;
esac

# Full output to a file; the console tail is only a convenience. M0 piped arm A
# through `tail -45` and threw away every purity verdict except the last one,
# which had to be recovered from the JSON afterwards.
run () {   # run <tag> <short> <long>
    local tag=$1 s=$2 l=$3
    echo
    echo "=== $tag  short_len=$s long_len=$l  (difference $((l - s)))"
    "$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
        --model Qwen/Qwen2.5-32B-Instruct --ctx 2048 --batch 16 \
        --riders 0,2,4,6 --connector --repeats 3 \
        --short-len "$s" --long-len "$l" \
        --gpu-mem-util 0.90 --link-gbps 25.1 \
        --outdir "$OUT" --out "${tag}.json" > "$OUT/${tag}.log" 2>&1
    echo "exit=$?  full log: $OUT/${tag}.log"
    grep -E "PURITY|CROSS-RUN|DISCARDED|IMPURE|^ *[0-9]+ +[0-9]+\.[0-9]+" \
        "$OUT/${tag}.log" | tail -30
}

run ramp_s24 24 136
run ramp_s48b 48 136

# ---- the comparison, side by side ------------------------------------------
echo
echo "=== CONVERGENCE: dT per rider as the short run clears the ramp"
"$PY" - <<'PYEOF'
import glob
import json
import os

OUT = "/mnt/beegfs/benishay/paper_runs/20260930/overhead"
files = [("s8  (M0 arm A, published shape)",
          f"{OUT}/00_baseline/A_decode_node.json"),
         ("s24 (clears ~10-step ramp)", f"{OUT}/00b_ramp/ramp_s24.json"),
         ("s48 (clears with margin)", f"{OUT}/00b_ramp/ramp_s48b.json")]
print(f"{'arm':<34} {'r':>3} {'step_ms':>9} {'dT_ms':>8} "
      f"{'dT/rider':>9} {'resid':>7} {'usable':>7} {'ramp_end':>9}")
for label, path in files:
    if not os.path.exists(path):
        print(f"{label:<34}  (missing: {os.path.basename(path)})")
        continue
    d = json.load(open(path))
    rows = {x["riders"]: x for x in d["rows"]}
    base = rows.get(0, {}).get("step_ms")
    for r in sorted(rows):
        x = rows[r]
        dt = (x["step_ms"] - base) if base else float("nan")
        # The kink model's residual: step - max(compute, transfer). This is the
        # quantity the paper reports as "overhead 8.6-18.1 ms", and it is what
        # the projection rests on, so it matters more than dT/rider.
        resid = x["step_ms"] - max(base or 0.0, x.get("link_ms") or 0.0)
        rep0 = (x.get("reps") or [{}])[0]
        pur = rep0.get("purity") or {}
        re_ = (pur.get("long") or {}).get("ramp_end")
        print(f"{label:<34} {r:>3} {x['step_ms']:>9.2f} {dt:>8.2f} "
              f"{(dt / r if r else 0):>9.2f} {resid:>7.2f} "
              f"{str(x.get('usable')):>7} {str(re_):>9}")
print()
print("Read it like this. dT/rider RISES in the published curve (8.96 / 10.97 /")
print("15.49 at r = 2/4/6), which is what makes an affine fit return F = -23.4.")
print("If it flattens as short_len rises, that rise was ramp residue. If it")
print("survives at s48, it is a property of the mechanism and the kink model")
print("stands. The `resid` column is the paper's own overhead figure (18.1 /")
print("8.6 / 13.5, reproduced today as 17.91 / 8.49 / 11.75); watch whether it")
print("becomes flat in r, which is what a genuine fixed cost would look like.")
PYEOF

echo
echo "--- teardown"
"$PY" -c "
import sys; sys.path.insert(0,'benchmarks/kv_pipelining_bench')
try:
    from bench_teardown import terminate_engine; terminate_engine(); print('ok')
except Exception as e: print('teardown helper unavailable:', e)
" 2>&1 | tail -3
ps -u "$USER" -o args= | grep -E "[.]venv/bin/python.*kv_pipelining_bench|^VLLM::EngineCore" | head -5
echo "========= M0b done $(date -Is) ========="
