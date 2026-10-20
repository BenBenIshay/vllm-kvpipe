#!/bin/bash
# M22b -- WHERE c's context slope comes from. M22's first repeat (2026-10-20)
# measured, bytes removed, 1 resident: c = 0.60 / 0.93 / 1.34 / 2.23 ms at ctx
# 512 / 2048 / 4096 / 8192 (ours), F flat at ~4.8. The per-block table adds only
# 0.13 -> 1.36 ms on top, so most of the slope is NOT the table. Hypothesis: it
# is the riders' attention kernel itself, reading S keys per layer from the slot
# -- work any request pays, resident or rider. Test: the same ladder with the
# second call's flash kernel skipped (--attend-mode nokernel; gather and scatter
# kept), interleaved with the default arm in the SAME job so drift cannot read as
# context. If c_ours - c_nokernel carries the slope and c_nokernel is flat, the
# mechanism's own per-rider cost is context-independent.
#
#SBATCH --job-name=opt-m22b
#SBATCH --partition=supernova
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20261019/m22b/m22b_%j.out

set -u
TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20261019/m22b
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct

echo "========= M22b : $(hostname) $(date -Is) ========="
echo "head: $(git rev-parse --short HEAD)   uncommitted: $(git status --porcelain -- vllm benchmarks | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "co-tenants on $(hostname):"; squeue -w "$(hostname)" -o "%i %u %C %b" 2>/dev/null

VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "tree OK: $VLLM_PATH" ;;
  *) echo "FATAL: wrong tree ($VLLM_PATH)"; exit 10 ;;
esac

# run <tag> <ctx> <batch> <riders> [flags...]
run () {
    local tag=$1 ctx=$2 batch=$3 riders=$4; shift 4
    echo
    echo "=== $tag  ctx=$ctx batch=$batch riders=$riders $* $(date -Is)"
    "$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
        --model "$Q32" --ctx "$ctx" --batch "$batch" --riders "$riders" \
        --connector --repeats 3 --short-len 8 --long-len 48 \
        --gpu-mem-util 0.90 --link-gbps 25.1 \
        "$@" \
        --outdir "$OUT" --out "${tag}.json" 2>&1 | tail -6
    echo "--- $tag done $(date -Is)"
    echo "    valid: $(grep -o '"numerically_valid": [a-z]*' "$OUT/${tag}.json" 2>/dev/null | head -1)"
}

LADDER_R=0,1,2,3,4,5

LADDER_R=0,1,2,3,4,5
for rep in a b; do
  for ctx in 512 2048 4096 8192 14724; do
    run "NK_ours_c${ctx}_$rep" "$ctx" 1 "$LADDER_R" --k-window 4 --xfer-mode minbytes
    run "NK_nokr_c${ctx}_$rep" "$ctx" 1 "$LADDER_R" --k-window 4 --xfer-mode minbytes --attend-mode nokernel
  done
done

echo
echo "========= M22b complete $(date -Is) ========="
ls -la "$OUT"
