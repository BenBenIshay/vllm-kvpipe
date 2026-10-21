#!/bin/bash
# E1 -- decode GEMMs under controlled HBM write traffic (Ben, 2026-10-21: "yes,
# start E1"). Off-engine microbenchmark; decides whether M23's real-bytes cost
# grows with link/HBM bandwidth share (GH200 C2C ~10.5% vs A100 PCIe ~1.3%).
#
#   sbatch -w nova07 benchmarks/kv_pipelining_bench/e1_gemm_write_contention.sbatch
#
#SBATCH --job-name=opt-e1
#SBATCH --partition=supernova
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=0:45:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20261021/e1/e1_%j.out
set -u
TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20261021/e1
cd "$TREE" || exit 1
echo "========= E1 : $(hostname) $(date -Is)  head $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem --format=csv,noheader
echo "co-tenants:"; squeue -w "$(hostname)" -o "%i %u %C %b" 2>/dev/null
for rep in a b; do
  "$PY" benchmarks/kv_pipelining_bench/e1_gemm_write_contention.py \
      --out "$OUT/E1_$rep.json"
done
echo "========= E1 complete $(date -Is)"
