#!/usr/bin/env bash
# B1/B2 validation — aligned with K4 sweep (/tmp/kv_profile_v2/sweep/K4):
#   Qwen2.5-0.5B, K=4, conc 1 4 8, input/output 256/128, enforce_eager,
#   --no-inflate-kv-pool, --no-pipelined-gpu-mem-cap,
#   gpu_memory_utilization=0.90 when GPU is idle; default 0.72 if ~12GiB is in use.
#   isolated latency ON (no --skip-isolated-latency), concurrent ITL throughput ON.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench/pipelined_decode_fast_bench.py"
OUT="${1:-/tmp/kv_b1_b2_aligned}"
PORT_BASE="${2:-8820}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/ben/huggingface}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT"
# Bench sorts concurrency ascending → 1, 4, 8 (same order as K4 sweep).
CONC="1 4 8"
GPU_UTIL="${GPU_UTIL:-0.72}"

BENCH_COMMON=(
  "$BENCH"
  --model Qwen/Qwen2.5-0.5B-Instruct
  --k-window 4
  --concurrency $CONC
  --input-len 256
  --output-len 128
  --no-inflate-kv-pool
  --no-auto-gpu-memory
  --no-pipelined-gpu-mem-cap
  --gpu-memory-utilization "$GPU_UTIL"
  --enforce-eager
  --startup-timeout 600
)

run_cfg() {
  local name="$1"
  local batched="$2"
  local step_plan="$3"
  local port="$4"
  export VLLM_KVPIPE_BATCHED_D2H="$batched"
  export VLLM_KVPIPE_STEP_PLAN="$step_plan"
  local dir="${OUT}/${name}"
  mkdir -p "$dir"
  echo ""
  echo "######################################################################"
  echo "# ${name}  BATCHED_D2H=${batched}  STEP_PLAN=${step_plan}  port=${port}"
  echo "######################################################################"
  "$PY" "${BENCH_COMMON[@]}" \
    --modes pipelined \
    --port "$port" \
    --result-dir "$dir" 2>&1 | tee "${dir}/run.log"
}

p="$PORT_BASE"
run_cfg "neither" 0 0 "$((p++))"
run_cfg "b1_only" 1 0 "$((p++))"
run_cfg "b2_only" 0 1 "$((p++))"
run_cfg "both" 1 1 "$((p++))"

# Optional baseline reference (same bench flags, no pipelining flags).
echo ""
echo "######################################################################"
echo "# baseline_reference (DecodeBenchConnector)"
echo "######################################################################"
BASE_DIR="${OUT}/baseline_reference"
mkdir -p "$BASE_DIR"
unset VLLM_KVPIPE_BATCHED_D2H VLLM_KVPIPE_STEP_PLAN
"$PY" "${BENCH_COMMON[@]}" \
  --modes baseline \
  --with-baseline \
  --port "$((p++))" \
  --result-dir "$BASE_DIR" 2>&1 | tee "${BASE_DIR}/run.log"

echo ""
echo "======================================================================"
echo "SUMMARY (aligned with K4: isolated steady + concurrent steady ITL, ms)"
echo "Reference K4 pipelined: conc1 ~8.1ms / conc8 ~17.8ms (throughput steady)"
echo "======================================================================"
"$PY" <<PY
import json
from pathlib import Path

out = Path("${OUT}")
ref_k4 = Path("/tmp/kv_profile_v2/sweep/K4/summary_all.json")
if ref_k4.is_file():
    k4 = json.loads(ref_k4.read_text()).get("pipelined", {})
    lat = k4.get("latency", {})
    print(f"K4 reference isolated steady ITL: {lat.get('mean_itl_steady_s_mean', 0)*1000:.2f} ms")
    for row in k4.get("throughput", []):
        c = row["concurrency"]
        print(f"K4 reference conc={c} throughput steady: {row['concurrent_mean_itl_steady_s_mean']*1000:.2f} ms")

print()
for name in ["baseline_reference", "neither", "b1_only", "b2_only", "both"]:
    f = out / name / "summary_all.json"
    if not f.is_file():
        print(f"--- {name}: MISSING ---")
        continue
    data = json.loads(f.read_text())
    mode = "baseline" if name == "baseline_reference" else "pipelined"
    block = data.get(mode, {})
    lat = block.get("latency", {})
    iso = lat.get("mean_itl_steady_s_mean")
    if lat.get("skipped"):
        iso_s = "n/a"
    elif iso is not None:
        iso_s = f"{iso*1000:.2f}"
    else:
        iso_s = "n/a"
    print(f"--- {name} (isolated steady={iso_s} ms) ---")
    for row in block.get("throughput", []):
        c = row["concurrency"]
        t = row["concurrent_mean_itl_steady_s_mean"] * 1000
        tok = row["throughput_tok_per_s"]
        print(f"  conc={c}: steady_ITL={t:.2f} ms  tok/s={tok:.1f}")
PY

echo ""
echo "Results under: ${OUT}/"
