#!/bin/bash
# M22c -- k = 2 vs 4 vs 8 NEAR THE KNEE (Ben, 2026-10-20: "let's add the test to
# M22b"). M22's k sweep ran at ctx 128 with the link ~17% loaded, so it says
# nothing about whether a deeper window absorbs jitter when per-layer transfer
# approaches per-layer compute. Submitted as a separate job after M22b
# (afterany) because the queued M22b script is already frozen by SLURM.
#
#   sbatch --dependency=afterany:2190252 benchmarks/kv_pipelining_bench/m22c_k_near_knee.sbatch
#
#SBATCH --job-name=opt-m22c
#SBATCH --partition=supernova
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20261019/m22c/m22c_%j.out

set -u
TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20261019/m22c
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct

echo "========= M22c : $(hostname) $(date -Is) ========="
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

# k = 2 / 4 / 8 NEAR THE KNEE, real bytes. ctx 1024 moves 268 MB per rider,
# 10.7 ms at 25.1 GB/s against a ~46 ms step, so riders 0..6 cross the knee
# (~4.3 riders). Interleaved by k, two repeats.
for rep in a b; do
  for k in 2 4 8; do
    run "KK_k${k}_c1024_$rep" 1024 1 0,1,2,3,4,5,6 --k-window "$k"
  done
done

echo
echo "========= M22c complete $(date -Is) ========="
ls -la "$OUT"
