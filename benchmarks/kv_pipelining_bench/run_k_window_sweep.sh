#!/usr/bin/env bash
# P3: Sweep kv-k-window for baseline vs real_prefill (default: Qwen-0.5B).
#
# Usage:
#   bash benchmarks/kv_pipelining_bench/run_k_window_sweep.sh
#   K_WINDOWS="2 4 8" CONC="1" MODEL=... bash ...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench"
OUT="${OUT:-/data/ben/kv_phase_a_internal/open_loop_eval_v1/k_window_sweep_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
# Llama-8B: MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct bash run_k_window_sweep.sh
K_WINDOWS="${K_WINDOWS:-2 4 8 16}"
CONC="${CONC:-1 16}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
ITL_STEADY_SKIP="${ITL_STEADY_SKIP:-1}"
PORT_BASE="${PORT_BASE:-8240}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/ben/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUT"
SUMMARY="$OUT/k_window_sweep_summary.csv"
echo "model,k_window,mode,K,mean_itl_ms,mean_itl_steady_ms,throughput_tok_s,p99_global_ms,p99_global_steady_ms" >"$SUMMARY"

MIN_FREE="${MIN_FREE:-0.85}"
GPU_UTIL="${GPU_UTIL:-0.85}"
echo "=== GPU preflight (need >= ${MIN_FREE} free, vLLM util ${GPU_UTIL}) ==="
"$PY" "$BENCH/gpu_preflight.py" \
  --min-free-fraction "$MIN_FREE" \
  --gpu-memory-utilization "$GPU_UTIL"

port="$PORT_BASE"
for kw in $K_WINDOWS; do
  # Llama-3.1-8B has 32 layers; k_window must be < 32
  if [[ "$kw" -ge 32 ]]; then
    echo "Skip k_window=$kw (need kw < num_layers)"
    continue
  fi
  kw_dir="$OUT/kw${kw}"
  mkdir -p "$kw_dir"
  echo ""
  echo "########################################"
  echo "# k_window=$kw  port=$port"
  echo "########################################"
  "$PY" "$BENCH/decode_only_bench.py" \
    --model "$MODEL" \
    --modes baseline real_prefill \
    --concurrency $CONC \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --k-window "$kw" \
    --streaming-throughput \
    --gpu-memory-utilization "$GPU_UTIL" \
    --require-gpu-free-fraction "$MIN_FREE" \
    --result-dir "$kw_dir" \
    --port "$port" \
    --warmup-requests "$WARMUP_REQUESTS" \
    --itl-steady-skip "$ITL_STEADY_SKIP" \
    2>&1 | tee "$kw_dir/run.log"

  for mode in baseline real_prefill; do
  json="$kw_dir/results_${mode}.json"
  [[ -f "$json" ]] || continue
  "$PY" - "$json" "$MODEL" "$kw" "$mode" <<'PY' >>"$SUMMARY"
import json, sys
path, model, kw, mode = sys.argv[1:5]
data = json.load(open(path))
for cell in data.get("throughput") or []:
    K = int(cell["concurrency"])
    m = cell.get("concurrent_mean_itl_s_mean")
    ms = cell.get("concurrent_mean_itl_steady_s_mean")
    g = cell.get("concurrent_global_p99_itl_s")
    gs = cell.get("concurrent_global_p99_itl_steady_s")
    t = cell.get("throughput_tok_per_s")
    print(
        f"{model},{kw},{mode},{K},"
        f"{float(m)*1000 if m else ''},"
        f"{float(ms)*1000 if ms else ''},"
        f"{t if t is not None else ''},"
        f"{float(g)*1000 if g else ''},"
        f"{float(gs)*1000 if gs else ''}"
    )
PY
  done
  port=$((port + 1))
  sleep 5
done

echo ""
echo "Wrote $SUMMARY"
"$PY" "$BENCH/plot_k_window_sweep.py" --csv "$SUMMARY" --out "$OUT/k_window_sweep.png" || true
