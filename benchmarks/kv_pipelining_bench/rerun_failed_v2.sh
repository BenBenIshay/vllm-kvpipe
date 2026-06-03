#!/bin/bash
# Re-run the three Llama-8B experiments that failed due to InferenceMode bug.
# The bug has been fixed in vllm/v1/kv_cache_pipelining.py.
set -euo pipefail

export HF_HOME=/home/ben/hf_cache
EVAL=/home/ben/vllm/benchmarks/kv_pipelining_bench/eval_decode_isolation.py
PYTHON=/usr/bin/python3
OUTDIR=/tmp/kv_full_eval_v2
GPU_MON=/home/ben/vllm/benchmarks/kv_pipelining_bench/gpu_monitor.sh

cleanup_gpu() {
    echo "[$(date '+%F %T')] Cleaning up GPU & port 8199 ..."
    pkill -f "vllm.entrypoints.*8199" 2>/dev/null || true
    sleep 3
    pkill -9 -f "vllm.entrypoints.*8199" 2>/dev/null || true
    sleep 2
    local mem
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}')
    echo "[$(date '+%F %T')] GPU memory: ${mem:-?} MiB"
}

run_phase() {
    local label="$1"; shift
    mkdir -p "$OUTDIR/$label"
    local gpu_csv="$OUTDIR/$label/gpu_util.csv"
    echo "[$(date '+%F %T')] ━━━ Starting: $label ━━━"
    echo "[$(date '+%F %T')]   Args: $*"
    echo "[$(date '+%F %T')]   Log : $OUTDIR/$label/eval.log"
    echo "[$(date '+%F %T')]   GPU : $gpu_csv"
    cleanup_gpu
    # Start GPU monitor
    bash "$GPU_MON" "$gpu_csv" &
    local mon_pid=$!
    echo "[$(date '+%F %T')]   GPU monitor PID: $mon_pid"
    # Run eval
    $PYTHON "$EVAL" "$@" --result-dir "$OUTDIR/$label" 2>&1 | tee "$OUTDIR/$label/eval.log"
    local rc=${PIPESTATUS[0]}
    # Stop GPU monitor
    kill "$mon_pid" 2>/dev/null || true
    wait "$mon_pid" 2>/dev/null || true
    if [ "$rc" -eq 0 ]; then
        echo "[$(date '+%F %T')] ✓ Completed: $label"
    else
        echo "[$(date '+%F %T')] ✗ FAILED: $label (exit $rc)"
    fi
    cleanup_gpu
}

echo "================================================================"
echo "  RE-RUN FAILED LLAMA-8B EXPERIMENTS (InferenceMode bug fixed)"
echo "  $(date '+%F %T')"
echo "================================================================"

# Phase 1: Re-run K=8 K-sweep
run_phase llama8b_ksweep_k8 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --workloads random \
    --modes pipelined_inflated \
    --k-window 8 \
    --concurrency 64 128 256 \
    --num-prompts 150 \
    --random-input-len 512 \
    --random-output-len 128 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90

# Phase 2: Re-run K=10 K-sweep
run_phase llama8b_ksweep_k10 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --workloads random \
    --modes pipelined_inflated \
    --k-window 10 \
    --concurrency 64 128 256 \
    --num-prompts 150 \
    --random-input-len 512 \
    --random-output-len 128 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90

# Phase 3: Re-run capacity sweep with K=6
run_phase llama8b_capacity_k6 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --workloads random \
    --modes pipelined_inflated \
    --k-window 6 \
    --concurrency 64 128 256 384 512 \
    --num-prompts 100 \
    --random-input-len 512 \
    --random-output-len 128 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90

# Phase 4: Regenerate graphs and report
echo "[$(date '+%F %T')] Regenerating graphs and report ..."
$PYTHON /home/ben/vllm/benchmarks/kv_pipelining_bench/generate_graphs.py \
    --result-dir "$OUTDIR" 2>&1 | tee "$OUTDIR/graphs_rerun.log"

echo "================================================================"
echo "  ALL RE-RUNS DONE. $(date '+%F %T')"
echo "  Results : $OUTDIR/"
echo "  Graphs  : $OUTDIR/graphs/"
echo "  Report  : $OUTDIR/REPORT_v2.txt"
echo "================================================================"
