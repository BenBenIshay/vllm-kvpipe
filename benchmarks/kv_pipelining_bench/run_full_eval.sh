#!/usr/bin/env bash
# ============================================================================
# run_full_eval.sh — Unattended KV Pipelining Evaluation
# ============================================================================
#
# Usage (run detached, survives SSH disconnect):
#   nohup bash /home/ben/vllm/benchmarks/kv_pipelining_bench/run_full_eval.sh \
#       > /tmp/kv_full_eval/master.log 2>&1 &
#
# Monitor:
#   tail -f /tmp/kv_full_eval/master.log
#   ls /tmp/kv_full_eval/
#   cat /tmp/kv_full_eval/REPORT.txt   # when done
# ============================================================================

export HF_HOME=/home/ben/hf_cache
VLLM_DIR="/home/ben/vllm"
EVAL_SCRIPT="${VLLM_DIR}/benchmarks/kv_pipelining_bench/eval_decode_isolation.py"
REPORT_SCRIPT="${VLLM_DIR}/benchmarks/kv_pipelining_bench/generate_report.py"
RESULT_ROOT="/tmp/kv_full_eval"
PYTHON="/usr/bin/python3"
PORT=8199
K_WINDOW=4
GPU_MEM_UTIL=0.90
STARTUP_TIMEOUT=600
COOLDOWN=12

mkdir -p "${RESULT_ROOT}"

MYPID=$$

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cleanup_gpu() {
    log "Cleaning up GPU & port ${PORT} ..."
    # Kill vllm server processes (exclude our own PID and our children)
    ps aux 2>/dev/null | grep -E "vllm.entrypoints|api_server" | grep -v grep | grep -v "$$" | awk '{print $2}' | while read -r p; do
        kill -9 "$p" 2>/dev/null || true
    done
    ps aux 2>/dev/null | grep -E "multiprocessing" | grep -v grep | grep -v "$$" | awk '{print $2}' | while read -r p; do
        kill -9 "$p" 2>/dev/null || true
    done
    sleep 5
    # Kill leftover GPU processes
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r pid; do
        if [ -n "$pid" ]; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    sleep 3
    local mem=""
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null) || mem="unknown"
    log "GPU memory: ${mem}"
}

run_eval() {
    local label="$1"; shift
    local result_dir="${RESULT_ROOT}/${label}"
    local logfile="${result_dir}/eval.log"
    mkdir -p "${result_dir}"

    log "━━━ Starting: ${label} ━━━"
    log "  Args: $*"
    log "  Log : ${logfile}"

    cleanup_gpu

    cd "${VLLM_DIR}"
    local rc=0
    ${PYTHON} "${EVAL_SCRIPT}" \
        --port "${PORT}" \
        --startup-timeout "${STARTUP_TIMEOUT}" \
        --cooldown "${COOLDOWN}" \
        --result-dir "${result_dir}" \
        "$@" \
        > "${logfile}" 2>&1 || rc=$?

    if [ ${rc} -eq 0 ]; then
        log "✓ Completed: ${label}"
    else
        log "✗ Failed: ${label} (exit=${rc}). See ${logfile}"
    fi

    cleanup_gpu
}

# ============================================================================
log "╔══════════════════════════════════════════════════════════════╗"
log "║  FULL KV PIPELINING EVALUATION — PID ${MYPID}              ║"
log "╚══════════════════════════════════════════════════════════════╝"

# ── Phase 1: Llama-3.1-8B-Instruct (32 layers, inflation=6.4×) ──
MODEL1="meta-llama/Llama-3.1-8B-Instruct"
log "PHASE 1: ${MODEL1}"

run_eval "llama8b_random" \
    --model "${MODEL1}" \
    --workloads random \
    --modes baseline pipelined pipelined_inflated \
    --k-window ${K_WINDOW} \
    --concurrency 1 4 16 64 \
    --num-prompts 150 \
    --random-input-len 512 \
    --random-output-len 128 \
    --max-model-len 4096 \
    --gpu-memory-utilization ${GPU_MEM_UTIL}

run_eval "llama8b_multi_round" \
    --model "${MODEL1}" \
    --workloads multi_round \
    --modes baseline pipelined \
    --k-window ${K_WINDOW} \
    --concurrency 1 4 16 \
    --num-prompts 100 \
    --max-model-len 4096 \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --multi-round-turns 4 \
    --multi-round-sessions 20 \
    --multi-round-input-len 256 \
    --multi-round-output-len 128

run_eval "llama8b_random_long" \
    --model "${MODEL1}" \
    --workloads random \
    --modes baseline pipelined pipelined_inflated \
    --k-window ${K_WINDOW} \
    --concurrency 1 4 16 64 \
    --num-prompts 100 \
    --random-input-len 2048 \
    --random-output-len 256 \
    --max-model-len 4096 \
    --gpu-memory-utilization ${GPU_MEM_UTIL}

run_eval "llama8b_offloading" \
    --model "${MODEL1}" \
    --workloads random \
    --modes baseline offloading \
    --k-window ${K_WINDOW} \
    --concurrency 1 4 16 64 \
    --num-prompts 150 \
    --random-input-len 512 \
    --random-output-len 128 \
    --max-model-len 4096 \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --offloading-size-gb 40

# ── Phase 2: Qwen2.5-14B-Instruct (48 layers, inflation=9.6×) ──
MODEL2="Qwen/Qwen2.5-14B-Instruct"
log "PHASE 2: ${MODEL2}"

run_eval "qwen14b_random" \
    --model "${MODEL2}" \
    --workloads random \
    --modes baseline pipelined pipelined_inflated \
    --k-window ${K_WINDOW} \
    --concurrency 1 4 16 64 \
    --num-prompts 150 \
    --random-input-len 512 \
    --random-output-len 128 \
    --max-model-len 4096 \
    --gpu-memory-utilization ${GPU_MEM_UTIL}

run_eval "qwen14b_multi_round" \
    --model "${MODEL2}" \
    --workloads multi_round \
    --modes baseline pipelined \
    --k-window ${K_WINDOW} \
    --concurrency 1 4 16 \
    --num-prompts 100 \
    --max-model-len 4096 \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --multi-round-turns 4 \
    --multi-round-sessions 20 \
    --multi-round-input-len 256 \
    --multi-round-output-len 128

run_eval "qwen14b_random_long" \
    --model "${MODEL2}" \
    --workloads random \
    --modes baseline pipelined pipelined_inflated \
    --k-window ${K_WINDOW} \
    --concurrency 1 4 16 64 \
    --num-prompts 100 \
    --random-input-len 2048 \
    --random-output-len 256 \
    --max-model-len 4096 \
    --gpu-memory-utilization ${GPU_MEM_UTIL}

# ── Phase 3: Generate report ──
log "Generating final report ..."
${PYTHON} "${REPORT_SCRIPT}" --result-root "${RESULT_ROOT}" \
    > "${RESULT_ROOT}/REPORT.txt" 2>&1 || true

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "ALL DONE."
log "  Results : ${RESULT_ROOT}/"
log "  Report  : ${RESULT_ROOT}/REPORT.txt"
log "  Master  : ${RESULT_ROOT}/master.log"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
