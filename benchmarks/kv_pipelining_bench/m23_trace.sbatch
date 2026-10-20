#!/bin/bash
# M23 -- the post-M22 trace (Ben, 2026-10-20: "run the nsys trace after M22c").
# nsys is not installed on nova07, so the timeline comes from torch.profiler
# (CUPTI; same CUDA activity with stream ids). rider_cost_curve.py --trace-dir
# runs one extra long generation per traced rider count AFTER the measurement.
#
# Questions, each with the arm that answers it:
#   Q1 does the write-back D2H (non-blocking, on the COMPUTE stream) sit on the
#      critical path?                                  -> TR_c1024_real r=3
#   Q2 host headroom: does GPU idle appear between kernels as riders grow?
#                                                      -> TR_c512_real r=0,1,3,6
#   Q3 what is c_mech's residual slope (~0.06 ms per 1k tokens, M22b)?
#      Per-step kernel time by name, nokernel arm, ctx 2048 vs 14724.
#                                                      -> TR_nk_c2048 / c14724
#   Q4 what is the ~7 ms beyond the knee (M22c: step = link + 6.8-7.6 ms)?
#                                                      -> TR_c1024_real r=4,5
#   Q5 why is c 1.55 ms with real bytes at ctx 1024 (M22c) when the minbytes
#      ladder interpolates to ~0.72?                   -> TR_c1024_minb r=3,
#      plus a TIMED real-vs-minbytes ladder at ctx 1024, interleaved, 2 reps.
#
#   sbatch -w nova07 benchmarks/kv_pipelining_bench/m23_trace.sbatch
#
#SBATCH --job-name=opt-m23
#SBATCH --partition=supernova
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=3:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20261020/m23/m23_%j.out

set -u
TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20261020/m23
TR="$OUT/traces"
mkdir -p "$OUT" "$TR"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct

echo "========= M23 : $(hostname) $(date -Is) ========="
echo "head: $(git rev-parse --short HEAD)   uncommitted: $(git status --porcelain -- vllm benchmarks | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "co-tenants on $(hostname):"; squeue -w "$(hostname)" -o "%i %u %C %b" 2>/dev/null

VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "tree OK: $VLLM_PATH" ;;
  *) echo "FATAL: wrong tree ($VLLM_PATH)"; exit 10 ;;
esac

# run <tag> <ctx> <batch> <riders> <repeats> [flags...]
run () {
    local tag=$1 ctx=$2 batch=$3 riders=$4 reps=$5; shift 5
    echo
    echo "=== $tag  ctx=$ctx batch=$batch riders=$riders reps=$reps $* $(date -Is)"
    "$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
        --model "$Q32" --ctx "$ctx" --batch "$batch" --riders "$riders" \
        --connector --repeats "$reps" --short-len 8 --long-len 48 \
        --gpu-mem-util 0.90 --link-gbps 25.1 \
        "$@" \
        --outdir "$OUT" --out "${tag}.json" 2>&1 | grep -E "trace ->|r= ?[0-9]+:|wrote|Error|error" | tail -12
    echo "--- $tag done $(date -Is)"
}

# ---- Q5 first: the TIMED arm. No profiler scopes, interleaved real/minbytes.
for rep in a b; do
  run "RB_real_c1024_$rep" 1024 1 0,1,2,3,4 3
  run "RB_minb_c1024_$rep" 1024 1 0,1,2,3,4 3 --xfer-mode minbytes
done

# ---- Traces. vLLM's step-phase scopes (preprocess/forward/sample/...) on.
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
run TR_c1024_real 1024 1 0,3,4,5 1 --trace-dir "$TR"                      # Q1 Q4
run TR_c1024_minb 1024 1 0,3     1 --trace-dir "$TR" --xfer-mode minbytes # Q5
run TR_c512_real  512  1 0,1,3,6 1 --trace-dir "$TR"                      # Q2
run TR_nk_c2048   2048  1 0,3    1 --trace-dir "$TR" --xfer-mode minbytes --attend-mode nokernel  # Q3
run TR_nk_c14724  14724 1 0,3    1 --trace-dir "$TR" --xfer-mode minbytes --attend-mode nokernel  # Q3
run TR_ours_c14724 14724 1 0,3   1 --trace-dir "$TR" --xfer-mode minbytes # Q3 (kernel on)

echo
echo "========= M23 complete $(date -Is) ========="
ls -la "$OUT" "$TR"
