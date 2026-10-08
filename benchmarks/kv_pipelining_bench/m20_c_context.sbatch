#!/bin/bash
# M20 -- does `c` have a context term, and is it resolvable?
#
#   sbatch benchmarks/kv_pipelining_bench/m20_c_context.sbatch
#
# THIS IS THE EXPERIMENT THAT DECIDES TABLE 2. The paper charges c = 0.52 ms,
# measured at ctx 128, at ctx 8192 -- six doublings away. Three things in the
# archive say that is not safe, and all three were found by a reviewer reading our
# own JSONs:
#
#   1. c across TWELVE post-optimisation arms at ONE configuration runs 0.414 to
#      0.538, a spread of 26% of the mean. Section 6.3 claims it "repeats to 1.5%".
#      That claim is false and this job measures the real figure.
#   2. The marginal cost of a rider rises with context EVEN WITH THE BYTES REMOVED:
#      MB_c128 1.072 -> MB_c256 1.190 ms, +11%, same campaign, same job, l/T0 =
#      0.007 in both. So something per-rider scales with blocks per request and
#      survives the extent allocator that was supposed to remove it.
#   3. Our own falsification table in section 6.3 mixed two campaigns whose F
#      differs by 2x, and its replicate spread (0.285 ms) exceeds the effect it
#      claimed to exclude. It has to be rebuilt from one job or cut.
#
# DESIGN. Differencing dT(2) - dT(1) cancels F identically, so it measures the
# marginal cost of one rider with no fit and no intercept. Sweep context at FIXED
# depth, batch and link, with THREE replicates per cell in ONE job so the spread is
# measured rather than assumed, and repeat the whole ladder with the bytes removed
# (xfer-mode=minbytes) so transfer and host work are separated at every context.
#
#   l/T0 at r=2 stays at or under 0.25 across the whole ladder, inside the paper's
#   own below-the-knee gate, so exposure cannot explain a rise.
#
# PREDICTIONS, stated before the run:
#   flat c          0% across 128 -> 512
#   the MB data    ~+25%
#   the withdrawn law c = 0.205 + 1.84e-3*S:  +161%
#
# No delay command anywhere: the submit filter rejects any batch script with one.
#
#SBATCH --job-name=opt-m20
#SBATCH --partition=supernova
#SBATCH --nodelist=nova07
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=05:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m20_%j.out

set -u
TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/22_c_context
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct
RIDERS=0,1,2,3,4,5,6,8

echo "========= M20 : $(hostname) $(date -Is) ========="
echo "head: $(git rev-parse --short HEAD)   uncommitted: $(git status --porcelain | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "tree OK: $VLLM_PATH" ;;
  *) echo "FATAL: wrong tree ($VLLM_PATH)"; exit 10 ;;
esac

# One arm. Everything fixed except context and whether the bytes are sent.
run () {
    local tag=$1 ctx=$2; shift 2
    echo
    echo "=== $tag  ctx=$ctx $* $(date -Is)"
    "$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
        --model "$Q32" --ctx "$ctx" --batch 16 --riders "$RIDERS" \
        --connector --repeats 5 --short-len 8 --long-len 48 \
        --gpu-mem-util 0.90 --link-gbps 25.1 --k-window 4 \
        "$@" \
        --outdir "$OUT" --out "${tag}.json" 2>&1 | tail -10
    echo "--- $tag done $(date -Is)"
    echo "    ctx in artifact: $(grep -o '"ctx": [0-9]*' "$OUT/${tag}.json" 2>/dev/null | head -1)"
    echo "    xfer_mode:       $(grep -o '"xfer_mode": "[a-z]*"' "$OUT/${tag}.json" 2>/dev/null | head -1)"
}

# THREE replicates per cell, interleaved across contexts rather than blocked, so a
# slow drift in the machine cannot masquerade as a context effect.
for rep in a b c; do
    run "CC_c128_$rep"  128
    run "CC_c256_$rep"  256
    run "CC_c512_$rep"  512
done

# The same ladder with the bytes removed. If the rise survives this, it is host
# work and not transfer -- which is the finding that decides how Table 2 must be
# restated.
for rep in a b; do
    run "MB_c128_$rep"  128 --xfer-mode minbytes
    run "MB_c256_$rep"  256 --xfer-mode minbytes
    run "MB_c512_$rep"  512 --xfer-mode minbytes
done

echo
echo "========= M20 complete $(date -Is) ========="
ls -la "$OUT"
