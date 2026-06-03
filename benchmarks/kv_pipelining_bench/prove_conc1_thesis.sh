#!/usr/bin/env bash
# Orchestration-only thesis (conc=1): R0 baseline | R1 optimized pipelined | R3 fake-transfer
#
# Fake mode: full staging/prefetch CUDA DAG (streams, events, wait_event on instant-ready
# events); zero-byte memcpy — simulates infinite CPU<->GPU bandwidth.
#
# Usage:
#   TRIALS=5 GPU_UTIL=0.60 COOLDOWN_S=30 bash prove_conc1_thesis.sh [/tmp/out] [port_base]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench/pipelined_decode_fast_bench.py"
OUT="${1:-/tmp/thesis_orchestration_proof}"
PORT_BASE="${2:-9000}"
TRIALS="${TRIALS:-5}"
COOLDOWN_S="${COOLDOWN_S:-30}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/ben/huggingface}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT"
GPU_UTIL="${GPU_UTIL:-0.60}"

{
  echo "suite_start=$(date -Is)"
  echo "GPU_UTIL=${GPU_UTIL}"
  echo "TRIALS=${TRIALS}"
  echo "COOLDOWN_S=${COOLDOWN_S}"
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null || true
} >"${OUT}/gpu_state_at_start.txt"

_bench_common() {
  "$PY" "$BENCH" \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --k-window 4 \
    --concurrency 1 \
    --input-len 256 \
    --output-len 128 \
    --no-inflate-kv-pool \
    --no-auto-gpu-memory \
    --no-pipelined-gpu-mem-cap \
    --gpu-memory-utilization "$GPU_UTIL" \
    --enforce-eager \
    --startup-timeout 600 \
    --result-dir "$1" \
    --port "$2" \
    "${@:3}"
}

run_baseline() {
  local dir="$1" port="$2"
  (
    export VLLM_KVPIPE_BATCHED_D2H=0 VLLM_KVPIPE_STEP_PLAN=0
    export VLLM_KVPIPE_SKIP_DELTA_STAGE=0 VLLM_KVPIPE_FAKE_TRANSFER=0
    export VLLM_KVPIPE_PROFILE=0 VLLM_KVPIPE_PROFILE_JSON=0
    _bench_common "$dir" "$port" --modes baseline --with-baseline
  ) 2>&1 | tee "${dir}/run.log"
}

run_pipelined() {
  local dir="$1" port="$2" profile="${3:-0}"
  (
    export VLLM_KVPIPE_BATCHED_D2H=1 VLLM_KVPIPE_STEP_PLAN=1
    export VLLM_KVPIPE_SKIP_DELTA_STAGE=0 VLLM_KVPIPE_FAKE_TRANSFER=0
    export VLLM_KVPIPE_PROFILE="$profile" VLLM_KVPIPE_PROFILE_JSON="$profile"
    _bench_common "$dir" "$port" --modes pipelined
  ) 2>&1 | tee "${dir}/run.log"
}

run_fake() {
  local dir="$1" port="$2" profile="${3:-0}"
  (
    export VLLM_KVPIPE_BATCHED_D2H=1 VLLM_KVPIPE_STEP_PLAN=1
    export VLLM_KVPIPE_SKIP_DELTA_STAGE=0 VLLM_KVPIPE_FAKE_TRANSFER=1
    export VLLM_KVPIPE_PROFILE="$profile" VLLM_KVPIPE_PROFILE_JSON="$profile"
    _bench_common "$dir" "$port" --modes pipelined
  ) 2>&1 | tee "${dir}/run.log"
}

run_config() {
  local cfg="$1" dir="$2" port="$3"
  case "$cfg" in
    R0) run_baseline "$dir" "$port" ;;
    R1) run_pipelined "$dir" "$port" 0 ;;
    R3) run_fake "$dir" "$port" 0 ;;
    *) echo "unknown config $cfg" >&2; exit 1 ;;
  esac
}

ORDER_FOR_TRIAL=(
  "R0 R1 R3"
  "R0 R3 R1"
  "R1 R0 R3"
  "R1 R3 R0"
  "R3 R0 R1"
  "R3 R1 R0"
)

p="$PORT_BASE"
for t in $(seq 1 "$TRIALS"); do
  idx=$(( (t - 1) % ${#ORDER_FOR_TRIAL[@]} ))
  order="${ORDER_FOR_TRIAL[$idx]}"
  echo "${order}" >"${OUT}/trial_${t}_order.txt"
  echo ""
  echo "######################################################################"
  echo "# TRIAL ${t}/${TRIALS}  order=${order}"
  echo "######################################################################"
  for cfg in $order; do
    case "$cfg" in
      R0) subdir="R0_baseline/trial_${t}" ;;
      R1) subdir="R1_pipelined/trial_${t}" ;;
      R3) subdir="R3_fake_transfer/trial_${t}" ;;
    esac
    mkdir -p "${OUT}/${subdir}"
    {
      echo "trial=${t}"
      echo "config_order=${order}"
      echo "cfg=${cfg}"
      date -Is
    } >"${OUT}/${subdir}/config_meta.txt"
    run_config "$cfg" "${OUT}/${subdir}" "$((p++))"
    if [[ "$COOLDOWN_S" -gt 0 ]]; then
      sleep "$COOLDOWN_S"
    fi
  done
done

echo ""
echo "######################################################################"
echo "# PROFILE (optional, single pass)"
echo "######################################################################"
mkdir -p "${OUT}/profile_R1/trial_1" "${OUT}/profile_R3/trial_1"
run_pipelined "${OUT}/profile_R1/trial_1" "$((p++))" 1
[[ "$COOLDOWN_S" -gt 0 ]] && sleep "$COOLDOWN_S"
run_fake "${OUT}/profile_R3/trial_1" "$((p++))" 1

{
  echo "suite_end=$(date -Is)"
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null || true
} >>"${OUT}/gpu_state_at_start.txt"

echo ""
echo "======================================================================"
echo "SUMMARY (${TRIALS} trials, conc=1)"
echo "======================================================================"
"$PY" "${ROOT}/benchmarks/kv_pipelining_bench/summarize_thesis_conc1.py" "$OUT"
