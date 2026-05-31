#!/usr/bin/env bash
# A/B sweep: B1 (batched D2H) and B2 (step plan) at conc 8 and 1.
# Env: VLLM_KVPIPE_BATCHED_D2H, VLLM_KVPIPE_STEP_PLAN (default both 1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench/pipelined_decode_fast_bench.py"
OUT="${1:-/tmp/kv_b1_b2_sweep}"
PORT_BASE="${2:-8800}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/ben/huggingface}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT"
CONC="${CONCURRENCY:-8 1}"
GPU_UTIL="${GPU_UTIL:-0.72}"

run_cfg() {
  local name="$1"
  local batched="$2"
  local step_plan="$3"
  local port="$4"
  export VLLM_KVPIPE_BATCHED_D2H="$batched"
  export VLLM_KVPIPE_STEP_PLAN="$step_plan"
  local dir="${OUT}/${name}"
  mkdir -p "$dir"
  echo "=== ${name} BATCHED_D2H=${batched} STEP_PLAN=${step_plan} port=${port} ==="
  "$PY" "$BENCH" \
    --model Qwen/Qwen2.5-0.5B-Instruct --k-window 4 \
    --concurrency $CONC --input-len 256 --output-len 128 \
    --modes pipelined --skip-isolated-latency \
    --no-inflate-kv-pool --no-auto-gpu-memory --no-pipelined-gpu-mem-cap \
    --gpu-memory-utilization "$GPU_UTIL" --enforce-eager \
    --startup-timeout 600 --port "$port" \
    --result-dir "$dir" 2>&1 | tee "${dir}/run.log"
}

p="$PORT_BASE"
run_cfg "neither" 0 0 "$((p++))"
run_cfg "b1_only" 1 0 "$((p++))"
run_cfg "b2_only" 0 1 "$((p++))"
run_cfg "both" 1 1 "$((p++))"

echo "Done. Summaries:"
for d in neither b1_only b2_only both; do
  f="${OUT}/${d}/summary_all.json"
  if [[ -f "$f" ]]; then
    echo "--- $d ---"
    "$PY" -c "
import json,sys
p=json.load(open(sys.argv[1]))['pipelined']['throughput']
for row in p:
    c=row['concurrency']
    print(f'  conc={c}: steady_ITL={row[\"concurrent_mean_itl_steady_s_mean\"]*1000:.2f}ms tok/s={row[\"throughput_tok_per_s\"]:.1f}')
" "$f"
  fi
done
