#!/usr/bin/env bash
# Rerun short c512 R0+R3 with max-model-len=1024 (matches original matrix protocol).
# Fixes CpuKvTable slot exhaustion (383 slots @ max_model_len=9216 vs 512 concurrent).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench/pipelined_decode_fast_bench.py"
OUT="${1:-/mnt/beegfs/benishay/paper_runs/20260703/orchestration_tax/r0_r3_prefix_on_v1}"
PORT_BASE="${2:-9400}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/beegfs/benishay/huggingface}"
export VLLM_HF_TOKEN_FILE="${VLLM_HF_TOKEN_FILE:-${ROOT}/hf_token}"
export PYTHONUNBUFFERED=1
export VLLM_KVPIPE_BATCHED_D2H=1 VLLM_KVPIPE_STEP_PLAN=1
export VLLM_KVPIPE_SKIP_DELTA_STAGE=0 VLLM_KVPIPE_PROFILE=0 VLLM_KVPIPE_PROFILE_JSON=0
GPU_UTIL="${GPU_UTIL:-0.90}"
COOLDOWN_S="${COOLDOWN_S:-20}"

kill_gpu() {
  local port="$1"
  local pids
  pids=$(pgrep -f "vllm.entrypoints.openai.api_server.*--port ${port}" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null || true
    sleep 3
  fi
  local eng
  eng=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
  if [[ -n "$eng" ]]; then
    kill -9 $eng 2>/dev/null || true
    sleep 3
  fi
}

run_cell() {
  local arm="$1" port="$2"
  local subdir="${OUT}/short/conc512/${arm}"
  mkdir -p "$subdir"
  echo ""
  echo "######################################################################"
  echo "# short conc=512 arm=${arm} port=${port} max_model_len=1024"
  echo "######################################################################"
  kill_gpu "$port"
  {
    echo "shape=short"
    echo "conc=512"
    echo "arm=${arm}"
    echo "input_len=512"
    echo "max_model_len=1024"
    echo "port=${port}"
    echo "rerun_reason=CpuKvTable slot fix (1024 vs 9216)"
    date -Is
  } >"${subdir}/cell_meta.txt"

  local -a extra=(
    --model "$MODEL"
    --port "$port"
    --max-model-len 1024
    --input-len 512
    --output-len 128
    --concurrency 512
    --k-window 4
    --no-auto-gpu-memory
    --no-pipelined-gpu-mem-cap
    --gpu-memory-utilization "$GPU_UTIL"
    --enforce-eager
    --prefix-caching-on
    --skip-isolated-latency
    --itl-steady-skip 1
    --startup-timeout 900
    --request-timeout 900
    --result-dir "$subdir"
  )

  case "$arm" in
    R0)
      export VLLM_KVPIPE_FAKE_TRANSFER=0
      "$PY" "$BENCH" "${extra[@]}" --modes baseline 2>&1 | tee "${subdir}/run.log"
      ;;
    R3)
      export VLLM_KVPIPE_FAKE_TRANSFER=1
      "$PY" "$BENCH" "${extra[@]}" --modes pipelined 2>&1 | tee "${subdir}/run.log"
      ;;
  esac
  if [[ "$COOLDOWN_S" -gt 0 ]]; then
    sleep "$COOLDOWN_S"
  fi
}

p="$PORT_BASE"
run_cell R0 "$((p++))"
run_cell R3 "$((p++))"

echo ""
echo "=== Summarizing ==="
"$PY" "${ROOT}/benchmarks/kv_pipelining_bench/summarize_decode_matrix.py" "$OUT"
