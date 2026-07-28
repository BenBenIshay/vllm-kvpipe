#!/usr/bin/env bash
# Phase 1: R1 hide-boundary sweep (all-pipelined, real PCIe).
# Finds max concurrency where R1 ITL stays within SLO of R0 at same batch.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench/pipelined_decode_fast_bench.py"
OUT="${1:-/mnt/beegfs/benishay/paper_runs/20260705/hybrid_kv/hide_boundary}"
PORT_BASE="${2:-9500}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/beegfs/benishay/huggingface}"
export VLLM_HF_TOKEN_FILE="${VLLM_HF_TOKEN_FILE:-${ROOT}/hf_token}"
export PYTHONUNBUFFERED=1
export VLLM_KVPIPE_BATCHED_D2H=1 VLLM_KVPIPE_STEP_PLAN=1
export VLLM_KVPIPE_SKIP_DELTA_STAGE=0 VLLM_KVPIPE_FAKE_TRANSFER=0
export VLLM_KVPIPE_PROFILE=0 VLLM_KVPIPE_PROFILE_JSON=0
GPU_UTIL="${GPU_UTIL:-0.90}"
COOLDOWN_S="${COOLDOWN_S:-20}"

# Short context hide-boundary grid (S=512 input, out=128, max_model_len=1024).
CONC_LIST="${CONC_LIST:-1 4 8 12 16 20 24 28 32 40 48 64}"

mkdir -p "$OUT"
{
  echo "start=$(date -Is)"
  echo "OUT=$OUT"
  echo "MODEL=$MODEL"
  echo "X_eff=25.1 GB/s (see pcie_microbench)"
  echo "CONC_LIST=$CONC_LIST"
} >"${OUT}/MANIFEST.txt"

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

run_r1_cell() {
  local conc="$1" port="$2"
  local subdir="${OUT}/short/conc${conc}/R1"
  mkdir -p "$subdir"
  echo ""
  echo "######################################################################"
  echo "# R1 hide-boundary conc=${conc} port=${port}"
  echo "######################################################################"
  kill_gpu "$port"
  date -Is >"${subdir}/cell_meta.txt"

  "$PY" "$BENCH" \
    --model "$MODEL" \
    --port "$port" \
    --max-model-len 1024 \
    --input-len 512 \
    --output-len 128 \
    --concurrency "$conc" \
    --k-window 4 \
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
    --modes pipelined 2>&1 | tee "${subdir}/run.log"

  if [[ "$COOLDOWN_S" -gt 0 ]]; then
    sleep "$COOLDOWN_S"
  fi
}

p="$PORT_BASE"
for c in $CONC_LIST; do
  run_r1_cell "$c" "$((p++))"
done

echo "end=$(date -Is)" >>"${OUT}/MANIFEST.txt"
echo ""
echo "=== Summarizing ==="
"$PY" "${ROOT}/benchmarks/kv_pipelining_bench/summarize_hide_boundary.py" "$OUT"
