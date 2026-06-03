#!/usr/bin/env bash
# Claim 2: layer-KV feature overhead (R1 vs R0) amortizes with concurrency.
#
# R0 = DecodeBenchConnector (GPU KV, no-op hooks)
# R1 = PipelinedDecodeBenchConnector (B1+B2 production path)
#
# Usage:
#   TRIALS=3 GPU_UTIL=0.60 COOLDOWN_S=20 bash prove_claim2_conc_sweep.sh [/tmp/out] [port_base]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench/pipelined_decode_fast_bench.py"
OUT="${1:-/tmp/thesis_claim2_conc}"
PORT_BASE="${2:-9100}"
TRIALS="${TRIALS:-3}"
COOLDOWN_S="${COOLDOWN_S:-20}"
CONC="${CONC:-1 2 4 8 16 32}"
GPU_UTIL="${GPU_UTIL:-0.60}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/ben/huggingface}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT"
mkdir -p "$(dirname "${OUT}/suite.log")"

{
  echo "suite_start=$(date -Is)"
  echo "GPU_UTIL=${GPU_UTIL}"
  echo "TRIALS=${TRIALS}"
  echo "COOLDOWN_S=${COOLDOWN_S}"
  echo "CONC=${CONC}"
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null || true
} >"${OUT}/gpu_state_at_start.txt"

_bench_common() {
  "$PY" "$BENCH" \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --k-window 4 \
    --concurrency $CONC \
    --input-len 256 \
    --output-len 128 \
    --no-inflate-kv-pool \
    --no-auto-gpu-memory \
    --no-pipelined-gpu-mem-cap \
    --gpu-memory-utilization "$GPU_UTIL" \
    --enforce-eager \
    --skip-isolated-latency \
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
    _bench_common "$dir" "$port" --modes baseline
  ) 2>&1 | tee "${dir}/run.log"
}

run_pipelined() {
  local dir="$1" port="$2"
  (
    export VLLM_KVPIPE_BATCHED_D2H=1 VLLM_KVPIPE_STEP_PLAN=1
    export VLLM_KVPIPE_SKIP_DELTA_STAGE=0 VLLM_KVPIPE_FAKE_TRANSFER=0
    export VLLM_KVPIPE_PROFILE=0 VLLM_KVPIPE_PROFILE_JSON=0
    _bench_common "$dir" "$port" --modes pipelined
  ) 2>&1 | tee "${dir}/run.log"
}

_run_ok() {
  local dir="$1" mode="$2"
  local summary="${dir}/results_${mode}.json"
  [[ -f "$summary" ]] || return 1
  "$PY" - <<'PY' "$summary" "$mode"
import json, sys
path, mode = sys.argv[1], sys.argv[2]
data = json.load(open(path))
if data.get("error") or data.get("aborted"):
    sys.exit(1)
thr = data.get("throughput") or []
if not thr:
    sys.exit(1)
for row in thr:
    if row.get("failures", 0) != 0:
        sys.exit(1)
    if row.get("concurrent_mean_itl_steady_s_mean") is None and \
       row.get("concurrent_mean_itl_s_mean") is None:
        sys.exit(1)
sys.exit(0)
PY
}

run_config_with_retry() {
  local cfg="$1" dir="$2" port="$3"
  local util="$GPU_UTIL"
  local attempt
  for attempt in 1 2 3; do
    GPU_UTIL="$util"
    case "$cfg" in
      R0) run_baseline "$dir" "$port" ;;
      R1) run_pipelined "$dir" "$port" ;;
      *) echo "unknown config $cfg" >&2; return 1 ;;
    esac
    local mode="baseline"
    [[ "$cfg" == "R1" ]] && mode="pipelined"
    if _run_ok "$dir" "$mode"; then
      return 0
    fi
    echo "[retry] ${cfg} attempt ${attempt} failed; cooling down..." >&2
    sleep 30
    util="$("$PY" - <<PY
u = float("${util}")
print(f"{max(0.40, u - 0.10):.2f}")
PY
)"
    echo "[retry] next GPU_UTIL=${util}" >&2
    rm -rf "${dir:?}"/*
  done
  return 1
}

ORDER_FOR_TRIAL=(
  "R0 R1"
  "R1 R0"
  "R0 R1"
)

p="$PORT_BASE"
for t in $(seq 1 "$TRIALS"); do
  idx=$(( (t - 1) % ${#ORDER_FOR_TRIAL[@]} ))
  order="${ORDER_FOR_TRIAL[$idx]}"
  echo "${order}" >"${OUT}/trial_${t}_order.txt"
  echo ""
  echo "######################################################################"
  echo "# TRIAL ${t}/${TRIALS}  order=${order}  GPU_UTIL=${GPU_UTIL}"
  echo "######################################################################"
  for cfg in $order; do
    case "$cfg" in
      R0) subdir="R0_baseline/trial_${t}" ;;
      R1) subdir="R1_pipelined/trial_${t}" ;;
    esac
    mkdir -p "${OUT}/${subdir}"
    {
      echo "trial=${t}"
      echo "config_order=${order}"
      echo "cfg=${cfg}"
      echo "GPU_UTIL=${GPU_UTIL}"
      date -Is
    } >"${OUT}/${subdir}/config_meta.txt"
    run_config_with_retry "$cfg" "${OUT}/${subdir}" "$((p++))" || {
      echo "[FAIL] ${cfg} trial ${t} after retries" >&2
      echo "fail cfg=${cfg} trial=${t} $(date -Is)" >>"${OUT}/failures.log"
    }
    if [[ "$COOLDOWN_S" -gt 0 ]]; then
      sleep "$COOLDOWN_S"
    fi
  done
done

{
  echo "suite_end=$(date -Is)"
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null || true
} >>"${OUT}/gpu_state_at_start.txt"

echo ""
echo "======================================================================"
echo "SUMMARY (claim 2: R0 vs R1 conc sweep)"
echo "======================================================================"
"$PY" "${ROOT}/benchmarks/kv_pipelining_bench/summarize_thesis_conc_sweep.py" "$OUT"
