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
# THE TEST. Raise --short-len until both runs clear the ramp, holding
# (long_len - short_len) = 128 fixed so the arms are comparable, and see whether
# the rise survives. Three points, so this is a convergence check rather than a
# single comparison:
#
#   short 8   long 136   <- M0 arm A, already measured: the published shape
#   short 48  long 176   <- clears a ~10-step ramp
#   short 96  long 224   <- clears it with a large margin
#
# If dT stabilises as short_len rises, the original was biased and the rise is an
# artifact. If the rise persists at short 96, it is real and the paper's kink
# model stands. I am not assuming either.
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

run ramp_s48 48 176
run ramp_s96 96 224

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
         ("s48 (clears ~10-step ramp)", f"{OUT}/00b_ramp/ramp_s48.json"),
         ("s96 (clears with margin)", f"{OUT}/00b_ramp/ramp_s96.json")]
print(f"{'arm':<34} {'r':>3} {'step_ms':>9} {'dT_ms':>8} "
      f"{'dT/rider':>9} {'usable':>7} {'ramp_end':>9}")
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
        rep0 = (x.get("reps") or [{}])[0]
        pur = rep0.get("purity") or {}
        re_ = (pur.get("long") or {}).get("ramp_end")
        print(f"{label:<34} {r:>3} {x['step_ms']:>9.2f} {dt:>8.2f} "
              f"{(dt / r if r else 0):>9.2f} {str(x.get('usable')):>7} "
              f"{str(re_):>9}")
print()
print("Read it like this: if dT/rider flattens as short_len rises, the rise in")
print("the published curve was ramp residue. If it still rises at s96, it is a")
print("property of the mechanism and the kink model stands.")
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
