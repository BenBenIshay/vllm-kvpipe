#!/usr/bin/env bash
# Sanity: transfer budget, pipeline debug probe, small-model ITL, QA correctness.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench"
OUT="${OUT:-/data/ben/kv_phase_a_internal/open_loop_eval_v1/sanity_$(date +%Y%m%d_%H%M%S)}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
LLAMA="${LLAMA:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
K_WINDOW="${K_WINDOW:-4}"
CONC="${CONC:-1 16}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/ben/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUT"

echo "=== Output directory: $OUT ==="
echo ""

echo "=== 1) Transfer budget (theory) ==="
for model in "$SMALL_MODEL" "$LLAMA"; do
  echo "--- $model conc=1 ---"
  "$PY" "$BENCH/kv_transfer_budget.py" \
    --model "$model" --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" \
    --k-window "$K_WINDOW" --concurrency 1
  echo "--- $model conc=16 ---"
  "$PY" "$BENCH/kv_transfer_budget.py" \
    --model "$model" --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" \
    --k-window "$K_WINDOW" --concurrency 16
  echo ""
done | tee "$OUT/transfer_budget.txt"

echo "=== 2) Pipeline debug probe (one decode, $SMALL_MODEL) ==="
PROBE_LOG="$OUT/probe_small_model.log"
"$PY" "$BENCH/probe_one_decode.py" \
  --model "$SMALL_MODEL" \
  --input-len "$INPUT_LEN" \
  --output-len 64 \
  --k-window "$K_WINDOW" \
  --pipeline-debug-log \
  --log "$PROBE_LOG" \
  --port 8225 \
  2>&1 | tee "$OUT/probe_run.txt"
"$PY" "$BENCH/parse_pipeline_debug_log.py" "$PROBE_LOG" \
  2>&1 | tee "$OUT/probe_parse.txt"

echo "=== 3) decode_only_bench: baseline vs real_prefill ($SMALL_MODEL) ==="
"$PY" "$BENCH/decode_only_bench.py" \
  --model "$SMALL_MODEL" \
  --modes baseline real_prefill \
  --concurrency $CONC \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --k-window "$K_WINDOW" \
  --streaming-throughput \
  --gpu-memory-utilization 0.85 \
  --result-dir "$OUT/decode_bench_small" \
  --port 8220 \
  2>&1 | tee "$OUT/decode_bench_small.log"

echo "=== 4) validate_qa conc=1 ($SMALL_MODEL) ==="
"$PY" "$BENCH/validate_qa.py" \
  --model "$SMALL_MODEL" \
  --k-window "$K_WINDOW" \
  --concurrency 1 \
  --max-tokens "$OUTPUT_LEN" \
  --result-dir "$OUT/validate_qa_small" \
  --port 8221 \
  2>&1 | tee "$OUT/validate_qa_small.log"

echo ""
echo "Done. Summarize ITL from: $OUT/decode_bench_small/results_*.json"
echo "QA summary: $OUT/validate_qa_small/validation_summary.json"
