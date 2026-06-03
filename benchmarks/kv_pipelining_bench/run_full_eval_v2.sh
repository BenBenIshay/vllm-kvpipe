#!/usr/bin/env bash
# ============================================================================
# run_full_eval_v2.sh — K-sweep + LMCache + Metrics Collection
# ============================================================================
#
# Usage (run detached, survives SSH disconnect):
#   nohup bash /home/ben/vllm/benchmarks/kv_pipelining_bench/run_full_eval_v2.sh \
#       > /tmp/kv_full_eval_v2/master.log 2>&1 &
#   echo $! > /tmp/kv_full_eval_v2/master.pid
#
# Monitor:
#   tail -f /tmp/kv_full_eval_v2/master.log
#   cat /tmp/kv_full_eval_v2/REPORT.txt    # when done
#   ls /tmp/kv_full_eval_v2/graphs/        # publication plots
# ============================================================================

set -o pipefail
export HF_HOME=/home/ben/hf_cache

VLLM_DIR="/home/ben/vllm"
EVAL_SCRIPT="${VLLM_DIR}/benchmarks/kv_pipelining_bench/eval_decode_isolation.py"
GPU_MONITOR="${VLLM_DIR}/benchmarks/kv_pipelining_bench/gpu_monitor.sh"
GRAPH_SCRIPT="${VLLM_DIR}/benchmarks/kv_pipelining_bench/generate_graphs.py"
RESULT_ROOT="/tmp/kv_full_eval_v2"
PREV_RESULTS="/tmp/kv_full_eval"
PYTHON="/usr/bin/python3"
PORT=8199
GPU_MEM_UTIL=0.90
STARTUP_TIMEOUT=600
COOLDOWN=15

mkdir -p "${RESULT_ROOT}"

MYPID=$$
echo "${MYPID}" > "${RESULT_ROOT}/master.pid"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cleanup_gpu() {
    log "Cleaning up GPU & port ${PORT} ..."
    ps aux 2>/dev/null | grep -E "vllm.entrypoints|api_server" | grep -v grep | awk '{print $2}' | while read -r p; do
        kill -9 "$p" 2>/dev/null || true
    done
    ps aux 2>/dev/null | grep -E "multiprocessing" | grep -v grep | awk '{print $2}' | while read -r p; do
        kill -9 "$p" 2>/dev/null || true
    done
    sleep 5
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r pid; do
        [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
    done
    sleep 3
    log "GPU memory: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
}

start_gpu_monitor() {
    local csv_file="$1"
    bash "${GPU_MONITOR}" "$csv_file" 1 </dev/null >/dev/null 2>&1 &
    local mon_pid=$!
    echo "$mon_pid"
}

stop_gpu_monitor() {
    local csv_file="$1"
    touch "${csv_file}.stop"
    sleep 2
    rm -f "${csv_file}.stop"
}

run_eval() {
    local label="$1"; shift
    local result_dir="${RESULT_ROOT}/${label}"
    local logfile="${result_dir}/eval.log"
    local gpu_csv="${result_dir}/gpu_util.csv"
    mkdir -p "${result_dir}"

    log "━━━ Starting: ${label} ━━━"
    log "  Args: $*"
    log "  Log : ${logfile}"
    log "  GPU : ${gpu_csv}"

    cleanup_gpu

    # Start GPU monitor
    local mon_pid
    mon_pid=$(start_gpu_monitor "$gpu_csv")
    log "  GPU monitor PID: ${mon_pid}"

    cd "${VLLM_DIR}"
    local rc=0
    ${PYTHON} "${EVAL_SCRIPT}" \
        --port "${PORT}" \
        --startup-timeout "${STARTUP_TIMEOUT}" \
        --cooldown "${COOLDOWN}" \
        --result-dir "${result_dir}" \
        "$@" \
        > "${logfile}" 2>&1 || rc=$?

    # Stop GPU monitor
    stop_gpu_monitor "$gpu_csv"

    if [ ${rc} -eq 0 ]; then
        log "✓ Completed: ${label}"
    else
        log "✗ Failed: ${label} (exit=${rc}). See ${logfile}"
    fi

    cleanup_gpu
}

# ============================================================================
log "╔══════════════════════════════════════════════════════════════╗"
log "║  FULL KV PIPELINING EVALUATION v2 — PID ${MYPID}          ║"
log "║  K-sweep + LMCache + Metrics                               ║"
log "╚══════════════════════════════════════════════════════════════╝"

# Copy existing baseline results so everything is in one place
log "Copying existing baseline results from ${PREV_RESULTS} ..."
for model_dir in llama8b_random qwen14b_random; do
    src="${PREV_RESULTS}/${model_dir}"
    if [ -d "$src" ]; then
        dst="${RESULT_ROOT}/${model_dir}_baseline"
        mkdir -p "$dst"
        cp -r "${src}"/* "$dst/" 2>/dev/null || true
        log "  Copied: ${src} -> ${dst}"
    fi
done

# ============================================================================
# PHASE 1: K-sweep on Llama-3.1-8B (K=6,8,10 × conc=64,128,256)
# ============================================================================
MODEL1="meta-llama/Llama-3.1-8B-Instruct"
log "PHASE 1: K-SWEEP on ${MODEL1}"

for K in 6 8 10; do
    run_eval "llama8b_ksweep_k${K}" \
        --model "${MODEL1}" \
        --workloads random \
        --modes pipelined_inflated \
        --k-window ${K} \
        --concurrency 64 128 256 \
        --num-prompts 150 \
        --random-input-len 512 \
        --random-output-len 128 \
        --max-model-len 4096 \
        --gpu-memory-utilization ${GPU_MEM_UTIL}
done

# Determine best K from Phase 1
log "Phase 1 complete. Analyzing K-sweep results..."
BEST_K=$(${PYTHON} -c "
import json, os, sys
results = {}
for k in [6, 8, 10]:
    d = '${RESULT_ROOT}/llama8b_ksweep_k%d/random' % k
    try:
        with open(os.path.join(d, 'results_pipelined_inflated_conc64.json')) as f:
            r = json.load(f)
            itl = r.get('mean_itl_ms', 999)
            results[k] = itl
            print(f'  K={k}: ITL@conc64 = {itl:.2f}ms', file=sys.stderr)
    except:
        results[k] = 999
        print(f'  K={k}: FAILED', file=sys.stderr)

# Find K with ITL closest to baseline (8.65ms)
baseline_itl = 8.65
best = min(results, key=lambda k: abs(results[k] - baseline_itl))
print(best)
" 2>&1 | tee /dev/stderr | tail -1)

log "Best K for Llama-8B: ${BEST_K}"
echo "${BEST_K}" > "${RESULT_ROOT}/best_k_llama8b.txt"

# ============================================================================
# PHASE 2: LMCache on Llama-3.1-8B
# ============================================================================
log "PHASE 2: LMCache on ${MODEL1}"

run_eval "llama8b_lmcache" \
    --model "${MODEL1}" \
    --workloads random \
    --modes lmcache \
    --k-window 4 \
    --concurrency 64 128 \
    --num-prompts 150 \
    --random-input-len 512 \
    --random-output-len 128 \
    --max-model-len 4096 \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --lmcache-cpu-size-gb 40

# ============================================================================
# PHASE 3: Capacity sweep with best K on Llama-3.1-8B
# ============================================================================
log "PHASE 3: Capacity sweep with K=${BEST_K} on ${MODEL1}"

run_eval "llama8b_capacity_k${BEST_K}" \
    --model "${MODEL1}" \
    --workloads random \
    --modes pipelined_inflated \
    --k-window ${BEST_K} \
    --concurrency 64 128 256 384 512 \
    --num-prompts 100 \
    --random-input-len 512 \
    --random-output-len 128 \
    --max-model-len 4096 \
    --gpu-memory-utilization ${GPU_MEM_UTIL}

# ============================================================================
# PHASE 4: K-sweep on Qwen2.5-14B (K=6,8,10 × conc=64,128,256)
# ============================================================================
MODEL2="Qwen/Qwen2.5-14B-Instruct"
log "PHASE 4: K-SWEEP on ${MODEL2}"

for K in 6 8 10; do
    run_eval "qwen14b_ksweep_k${K}" \
        --model "${MODEL2}" \
        --workloads random \
        --modes pipelined_inflated \
        --k-window ${K} \
        --concurrency 64 128 256 \
        --num-prompts 150 \
        --random-input-len 512 \
        --random-output-len 128 \
        --max-model-len 4096 \
        --gpu-memory-utilization ${GPU_MEM_UTIL}
done

# Determine best K for Qwen
BEST_K_QWEN=$(${PYTHON} -c "
import json, os, sys
results = {}
for k in [6, 8, 10]:
    d = '${RESULT_ROOT}/qwen14b_ksweep_k%d/random' % k
    try:
        with open(os.path.join(d, 'results_pipelined_inflated_conc64.json')) as f:
            r = json.load(f)
            itl = r.get('mean_itl_ms', 999)
            results[k] = itl
            print(f'  K={k}: ITL@conc64 = {itl:.2f}ms', file=sys.stderr)
    except:
        results[k] = 999
        print(f'  K={k}: FAILED', file=sys.stderr)

baseline_itl = 15.03  # from previous eval
best = min(results, key=lambda k: abs(results[k] - baseline_itl))
print(best)
" 2>&1 | tee /dev/stderr | tail -1)

log "Best K for Qwen-14B: ${BEST_K_QWEN}"
echo "${BEST_K_QWEN}" > "${RESULT_ROOT}/best_k_qwen14b.txt"

# ============================================================================
# PHASE 5: LMCache on Qwen2.5-14B
# ============================================================================
log "PHASE 5: LMCache on ${MODEL2}"

run_eval "qwen14b_lmcache" \
    --model "${MODEL2}" \
    --workloads random \
    --modes lmcache \
    --k-window 4 \
    --concurrency 64 128 \
    --num-prompts 150 \
    --random-input-len 512 \
    --random-output-len 128 \
    --max-model-len 4096 \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --lmcache-cpu-size-gb 40

# ============================================================================
# PHASE 6: Capacity sweep with best K on Qwen2.5-14B
# ============================================================================
log "PHASE 6: Capacity sweep with K=${BEST_K_QWEN} on ${MODEL2}"

run_eval "qwen14b_capacity_k${BEST_K_QWEN}" \
    --model "${MODEL2}" \
    --workloads random \
    --modes pipelined_inflated \
    --k-window ${BEST_K_QWEN} \
    --concurrency 64 128 256 384 512 \
    --num-prompts 100 \
    --random-input-len 512 \
    --random-output-len 128 \
    --max-model-len 4096 \
    --gpu-memory-utilization ${GPU_MEM_UTIL}

# ============================================================================
# PHASE 7: Generate graphs and final report
# ============================================================================
log "PHASE 7: Generating graphs and report ..."

${PYTHON} "${GRAPH_SCRIPT}" \
    --result-root "${RESULT_ROOT}" \
    --prev-results "${PREV_RESULTS}" \
    --output-dir "${RESULT_ROOT}/graphs" \
    > "${RESULT_ROOT}/graphs.log" 2>&1 || log "Graph generation had errors, check graphs.log"

# ============================================================================
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "ALL DONE."
log "  Results : ${RESULT_ROOT}/"
log "  Graphs  : ${RESULT_ROOT}/graphs/"
log "  Master  : ${RESULT_ROOT}/master.log"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
