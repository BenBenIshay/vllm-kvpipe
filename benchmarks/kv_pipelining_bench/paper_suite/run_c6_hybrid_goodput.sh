#!/usr/bin/env bash
# Phase 5: SLO-aware goodput — stock vs predictor policy vs hybrid tiers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench/bench_admission_policy_e2e.py"
OUT="${1:-/mnt/beegfs/benishay/paper_runs/20260705/hybrid_kv/c6_hybrid_goodput}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/beegfs/benishay/huggingface}"
export VLLM_HF_TOKEN_FILE="${VLLM_HF_TOKEN_FILE:-${ROOT}/hf_token}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_KVPIPE_BATCHED_D2H=1 VLLM_KVPIPE_STEP_PLAN=1 VLLM_KVPIPE_FAKE_TRANSFER=0

SMOKE="${SMOKE:-1}"
mkdir -p "$OUT"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SLO="${SLO:-200}"
RATE="${RATE:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
PREDICTOR_CSV="${PREDICTOR_CSV:-${OUT}/synthetic_predictor.csv}"

if [[ ! -f "$PREDICTOR_CSV" ]]; then
  {
    echo "model,K,input_len,output_len,p99_itl_ms"
    echo "${MODEL},1,${INPUT_LEN},${OUTPUT_LEN},20"
    echo "${MODEL},16,${INPUT_LEN},${OUTPUT_LEN},80"
    echo "${MODEL},32,${INPUT_LEN},${OUTPUT_LEN},150"
    echo "${MODEL},64,${INPUT_LEN},${OUTPUT_LEN},300"
  } >"$PREDICTOR_CSV"
fi

echo "start=$(date -Is)" | tee "${OUT}/run_meta.txt"
"$PY" "$BENCH" \
  --model "$MODEL" \
  --predictor-csv "$PREDICTOR_CSV" \
  --slo-p99-itl-ms "$SLO" \
  --request-rate "$RATE" \
  --num-prompts "$NUM_PROMPTS" \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --max-model-len 2048 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85 \
  --k-window 4 \
  --arms stock policy hybrid \
  --port-base 9900 \
  --out "$OUT" 2>&1 | tee "${OUT}/run.log"
echo "done=$(date -Is)" >>"${OUT}/run_meta.txt"
