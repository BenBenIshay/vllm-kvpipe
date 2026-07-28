#!/usr/bin/env bash
# R0 vs R3 matrix rerun (prefix caching ON both arms).
# Grid matches DECODE_ENGINE_MATRIX.txt (Llama-3.1-8B, k_window=4, out=128).
# Uses current pipelined_decode_fast_bench.py (--itl-steady-skip 1 only;
# legacy setup-seed / untimed-prefix flags omitted).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench/pipelined_decode_fast_bench.py"
OUT="${1:-/mnt/beegfs/benishay/paper_runs/20260703/orchestration_tax/r0_r3_prefix_on_v1}"
PORT_BASE="${2:-9300}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/beegfs/benishay/huggingface}"
export VLLM_HF_TOKEN_FILE="${VLLM_HF_TOKEN_FILE:-${ROOT}/hf_token}"
export PYTHONUNBUFFERED=1
export VLLM_KVPIPE_BATCHED_D2H=1 VLLM_KVPIPE_STEP_PLAN=1
export VLLM_KVPIPE_SKIP_DELTA_STAGE=0 VLLM_KVPIPE_PROFILE=0 VLLM_KVPIPE_PROFILE_JSON=0

GPU_UTIL="${GPU_UTIL:-0.90}"
COOLDOWN_S="${COOLDOWN_S:-20}"

mkdir -p "$OUT"
{
  echo "start=$(date -Is)"
  echo "OUT=$OUT"
  echo "MODEL=$MODEL"
  echo "prefix_caching=ON (both arms)"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true
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

run_cell() {
  local shape="$1" conc="$2" arm="$3" input_len="$4" port="$5" max_model_len="$6"
  local subdir="${OUT}/${shape}/conc${conc}/${arm}"
  mkdir -p "$subdir"
  echo ""
  echo "######################################################################"
  echo "# ${shape} conc=${conc} arm=${arm} port=${port} max_model_len=${max_model_len}"
  echo "######################################################################"
  kill_gpu "$port"
  {
    echo "shape=${shape}"
    echo "conc=${conc}"
    echo "arm=${arm}"
    echo "input_len=${input_len}"
    echo "max_model_len=${max_model_len}"
    echo "port=${port}"
    date -Is
  } >"${subdir}/cell_meta.txt"

  local -a extra=(
    --model "$MODEL"
    --port "$port"
    --max-model-len "$max_model_len"
    --input-len "$input_len"
    --output-len 128
    --concurrency "$conc"
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
    *)
      echo "unknown arm $arm" >&2
      exit 1
      ;;
  esac
  if [[ "$COOLDOWN_S" -gt 0 ]]; then
    sleep "$COOLDOWN_S"
  fi
}

p="$PORT_BASE"
# long S=8192 — max_model_len=9216 (8192+128+margin; CpuKvTable slots ~398)
for c in 1 4 8 16 32 64; do
  run_cell long "$c" R0 8192 "$((p++))" 9216
  run_cell long "$c" R3 8192 "$((p++))" 9216
done
# short S=512 — max_model_len=1024 (512+128+margin; ~3452 CpuKvTable slots @ c512)
for c in 16 64 256 512; do
  run_cell short "$c" R0 512 "$((p++))" 1024
  run_cell short "$c" R3 512 "$((p++))" 1024
done

echo "end=$(date -Is)" >>"${OUT}/MANIFEST.txt"
echo ""
echo "=== Summarizing ==="
"$PY" "${ROOT}/benchmarks/kv_pipelining_bench/summarize_decode_matrix.py" "$OUT"
