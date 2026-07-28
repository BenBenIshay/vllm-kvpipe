#!/usr/bin/env bash
# Phase 3: Hybrid vs R0 vs R1 validation (Llama-8B short S=512).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench/pipelined_decode_fast_bench.py"
OUT="${1:-/mnt/beegfs/benishay/paper_runs/20260705/hybrid_kv/validation_llama8b}"
PORT_BASE="${2:-9700}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/beegfs/benishay/huggingface}"
export VLLM_HF_TOKEN_FILE="${VLLM_HF_TOKEN_FILE:-${ROOT}/hf_token}"
export PYTHONUNBUFFERED=1
export VLLM_KVPIPE_BATCHED_D2H=1 VLLM_KVPIPE_STEP_PLAN=1
export VLLM_KVPIPE_SKIP_DELTA_STAGE=0 VLLM_KVPIPE_PROFILE=0
GPU_UTIL="${GPU_UTIL:-0.90}"
COOLDOWN_S="${COOLDOWN_S:-25}"

mkdir -p "$OUT"
{
  echo "start=$(date -Is)"
  echo "OUT=$OUT"
  echo "MODEL=$MODEL"
} >"${OUT}/MANIFEST.txt"

kill_gpu() {
  local port="$1"
  local pids
  pids=$(pgrep -f "vllm.entrypoints.openai.api_server.*--port ${port}" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null || true
    sleep 3
  fi
}

run_cell() {
  local label="$1" mode="$2" conc="$3" port="$4" fake="${5:-0}"
  local subdir="${OUT}/${label}/conc${conc}/${mode}"
  mkdir -p "$subdir"
  echo ""
  echo "######################################################################"
  echo "# ${label} mode=${mode} conc=${conc} fake=${fake}"
  echo "######################################################################"
  kill_gpu "$port"
  export VLLM_KVPIPE_FAKE_TRANSFER="${fake}"
  "$PY" "$BENCH" \
    --model "$MODEL" \
    --port "$port" \
    --max-model-len 1024 \
    --input-len 512 \
    --output-len 128 \
    --concurrency "$conc" \
    --k-window 4 \
    --hybrid-b-hbm-max 49 \
    --hybrid-seq-tokens 512 \
    --no-auto-gpu-memory \
    --no-pipelined-gpu-mem-cap \
    --gpu-memory-utilization "$GPU_UTIL" \
    --enforce-eager \
    --prefix-caching-on \
    --skip-isolated-latency \
    --itl-steady-skip 1 \
    --startup-timeout 900 \
    --request-timeout 900 \
    --result-dir "$subdir" \
    --modes "$mode" 2>&1 | tee "${subdir}/run.log"
  if [[ "$COOLDOWN_S" -gt 0 ]]; then
    sleep "$COOLDOWN_S"
  fi
}

p="$PORT_BASE"
# Low load: hybrid should match R0 (all HBM tier).
run_cell "low_load" baseline 16 "$((p++))" 0
run_cell "low_load" hybrid 16 "$((p++))" 0

# Mid load: hybrid over HBM cap (49+6 roofline @ S=512).
run_cell "mid_load" baseline 49 "$((p++))" 0
run_cell "mid_load" hybrid 55 "$((p++))" 0
run_cell "mid_load" pipelined 55 "$((p++))" 0

echo "end=$(date -Is)" >>"${OUT}/MANIFEST.txt"
echo "Done. Results under: ${OUT}/"
