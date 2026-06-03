#!/usr/bin/env bash
# M3: Compare vanilla baseline vs feature-ON real_prefill at fixed concurrencies.
# Produces ITL/throughput for "compute + memory path" separation (paper table).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench/decode_only_bench.py"
OUT="${OUT:-/data/ben/kv_phase_a_internal/open_loop_eval_v1/m3_baseline_vs_feature}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
CONC="${CONC:-16 32}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/ben/huggingface}"

mkdir -p "$OUT"
for MODE in baseline real_prefill; do
  echo "=== mode=$MODE model=$MODEL ==="
  "$PY" "$BENCH" \
    --model "$MODEL" \
    --modes "$MODE" \
    --concurrency $CONC \
    --input-len 1024 \
    --output-len 256 \
    --k-window 4 \
    --streaming-throughput \
    --gpu-memory-utilization 0.85 \
    --result-dir "$OUT/${MODE//\//_}" \
    --port 8210 \
    2>&1 | tee "$OUT/run_${MODE}.log"
done
echo "Done. Results under $OUT"
