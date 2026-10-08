#!/bin/bash
# M21 -- measure c's context dependence AT THE TARGET CONTEXT, by removing the knee.
#
#   sbatch benchmarks/kv_pipelining_bench/m21_c_nobus_ladder.sbatch
#
# BEN'S IDEA, and it is the right instrument. The obstacle to testing c's context
# dependence is the knee: on a PCIe part at ctx 512 only two rider points clear
# l/T0 <= 1/3, at 4096 one, at 8192 none -- a single request's transfer exceeds the
# whole step. So c cannot be fitted below the knee at the contexts Table 2 uses, and
# M20's attempt to do it at ctx 512 silently admitted transfer-bound points.
#
# REMOVE THE BYTES AND THE KNEE GOES AWAY. `--xfer-mode minbytes` issues every
# launch, every gather, every block-table entry and every bookkeeping step, and
# sends ~no bytes. l/T0 is then ~0.007 at ANY context, so every rider point is below
# the knee by construction and c can be fitted at 8192 tokens exactly as at 128.
#
# WHAT THAT MEASURES, precisely: the HOST component of c, c_host. That is the
# quantity the paper's claim is actually about -- "below the knee the transfer is
# hidden, so c is launch and bookkeeping work". If c_host is flat from 128 to 4096,
# the claim is tested at the target rather than assumed there. If it rises, we have
# found the context term and can put it in the model with a measured slope.
#
# AND IT IS THE RIGHT PROXY FOR THE GH200. At ctx 8192 on C2C the knee is 6.4
# riders, so the transfer IS hidden there and c = c_host. Measuring c_host on an
# A100 therefore measures the coefficient Table 2 needs for a machine we do not
# have.
#
# RESIDENT BATCH IS 4 AND FIXED. Qwen-32B fp16 leaves ~6.5 GB for KV after weights,
# and a request at ctx 4096 needs 1.07 GB, so batch 4 is what fits at the top of the
# ladder. It is held constant across every rung: varying it would change T0 and
# confound the context axis, which is the whole point of the run.
#
# No delay command anywhere: the submit filter rejects any batch script with one.
#
#SBATCH --job-name=opt-m21
#SBATCH --partition=supernova
#SBATCH --nodelist=nova07
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m21_%j.out

set -u
TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/23_c_nobus_ladder
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct
RIDERS=0,1,2,3,4,5,6,8
BATCH=4

echo "========= M21 : $(hostname) $(date -Is) ========="
echo "head: $(git rev-parse --short HEAD)   uncommitted: $(git status --porcelain | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "tree OK: $VLLM_PATH" ;;
  *) echo "FATAL: wrong tree ($VLLM_PATH)"; exit 10 ;;
esac

run () {
    local tag=$1 ctx=$2; shift 2
    echo
    echo "=== $tag  ctx=$ctx batch=$BATCH $* $(date -Is)"
    "$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
        --model "$Q32" --ctx "$ctx" --batch "$BATCH" --riders "$RIDERS" \
        --connector --repeats 5 --short-len 8 --long-len 48 \
        --gpu-mem-util 0.90 --link-gbps 25.1 --k-window 4 \
        "$@" \
        --outdir "$OUT" --out "${tag}.json" 2>&1 | tail -10
    echo "--- $tag done $(date -Is)"
    echo "    ctx:       $(grep -o '"ctx": [0-9]*' "$OUT/${tag}.json" 2>/dev/null | head -1)"
    echo "    xfer_mode: $(grep -o '"xfer_mode": "[a-z]*"' "$OUT/${tag}.json" 2>/dev/null | head -1)"
    echo "    valid:     $(grep -o '"numerically_valid": [a-z]*' "$OUT/${tag}.json" 2>/dev/null | head -1)"
}

# THE LADDER, bytes removed so every point is below the knee at every context.
# Interleaved across contexts rather than blocked, so a drift in the machine cannot
# be read as a context effect.
for rep in a b c; do
    run "NB_c128_$rep"   128  --xfer-mode minbytes
    run "NB_c512_$rep"   512  --xfer-mode minbytes
    run "NB_c2048_$rep" 2048  --xfer-mode minbytes
    run "NB_c4096_$rep" 4096  --xfer-mode minbytes
done

# Real bytes at the two contexts where they can still be fitted below the knee at
# this batch, to anchor c_host against the full c and confirm the two agree where
# both are measurable.
for rep in a b; do
    run "RB_c128_$rep"  128
    run "RB_c512_$rep"  512
done

echo
echo "========= M21 complete $(date -Is) ========="
ls -la "$OUT"
