#!/usr/bin/env bash
# C6: SLO-aware admission scheduler goodput vs stock, across multiple SLOs and
# offered request rates (open-loop Poisson).
#
# Loops (SLO x rate) calling bench_admission_policy_e2e.py with arms "stock policy",
# then summarize_goodput.py aggregates SLO-compliant throughput into a grid.
#
# Requires a predictor CSV (feature_on_k_sweep.py output). In SMOKE mode a tiny
# synthetic CSV is generated if PREDICTOR_CSV is unset, so the wrapper is
# self-contained for correctness validation.
#
#   SMOKE=1 bash run_c6_goodput.sh /tmp/c6_smoke
#   PREDICTOR_CSV=/data/.../feature_on_k_sweep.csv SLOS="100 200 300" RATES="4 8 16" \
#     bash run_c6_goodput.sh /data/.../C6_goodput
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-${ROOT}/.venv/bin/python}"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench/bench_admission_policy_e2e.py"
SUMM="${ROOT}/benchmarks/kv_pipelining_bench/paper_suite/summarize_goodput.py"
OUT="${1:-/tmp/c6_goodput}"
PORT_BASE="${PORT_BASE:-8700}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/ben/huggingface}"
export PYTHONUNBUFFERED=1
# vLLM is an editable/source install importable from the repo root; the launched
# server subprocess inherits this so `python -m vllm...` resolves.
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

SMOKE="${SMOKE:-0}"
if [[ "$SMOKE" == "1" ]]; then
  MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
  GPU_UTIL="${GPU_UTIL:-0.30}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
  K_WINDOW="${K_WINDOW:-4}"
  SLOS="${SLOS:-200}"
  RATES="${RATES:-4}"
  NUM_PROMPTS="${NUM_PROMPTS:-12}"
  INPUT_LEN="${INPUT_LEN:-256}"
  OUTPUT_LEN="${OUTPUT_LEN:-16}"
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
else
  MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
  GPU_UTIL="${GPU_UTIL:-0.85}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
  K_WINDOW="${K_WINDOW:-4}"
  SLOS="${SLOS:-100 150 200 300}"
  RATES="${RATES:-2 4 8 16 32}"
  NUM_PROMPTS="${NUM_PROMPTS:-200}"
  INPUT_LEN="${INPUT_LEN:-1024}"
  OUTPUT_LEN="${OUTPUT_LEN:-256}"
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
fi

# Predictor CSV: required by the policy arm. Synthesize a tiny one for smoke.
PREDICTOR_CSV="${PREDICTOR_CSV:-}"
if [[ -z "$PREDICTOR_CSV" ]]; then
  if [[ "$SMOKE" == "1" ]]; then
    PREDICTOR_CSV="${OUT}/synthetic_predictor.csv"
    {
      echo "model,K,input_len,output_len,p99_itl_ms"
      echo "${MODEL},1,${INPUT_LEN},${OUTPUT_LEN},20"
      echo "${MODEL},4,${INPUT_LEN},${OUTPUT_LEN},60"
      echo "${MODEL},8,${INPUT_LEN},${OUTPUT_LEN},120"
      echo "${MODEL},16,${INPUT_LEN},${OUTPUT_LEN},300"
    } >"$PREDICTOR_CSV"
    echo "[C6] SMOKE: generated synthetic predictor CSV at ${PREDICTOR_CSV}"
  else
    echo "ERROR: PREDICTOR_CSV is required for non-smoke runs (feature_on_k_sweep.py output)." >&2
    exit 1
  fi
fi

{
  echo "suite=C6_goodput smoke=${SMOKE} model=${MODEL}"
  echo "slos=${SLOS} rates=${RATES} predictor_csv=${PREDICTOR_CSV}"
  echo "start=$(date -Is)"
} | tee "${OUT}/run_meta.txt"

p="$PORT_BASE"
for slo in $SLOS; do
  for rate in $RATES; do
    cell="${OUT}/slo_${slo}/rate_${rate}"
    mkdir -p "$cell"
    echo ""
    echo "######### SLO=${slo}ms rate=${rate}req/s #########"
    "$PY" "$BENCH" \
      --model "$MODEL" \
      --predictor-csv "$PREDICTOR_CSV" \
      --slo-p99-itl-ms "$slo" \
      --request-rate "$rate" \
      --num-prompts "$NUM_PROMPTS" \
      --input-len "$INPUT_LEN" \
      --output-len "$OUTPUT_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --k-window "$K_WINDOW" \
      --arms stock policy \
      --port-base "$p" \
      --out "$cell" 2>&1 | tee "${cell}/run.log" || {
        echo "fail slo=${slo} rate=${rate} $(date -Is)" >>"${OUT}/failures.log"
      }
    p=$((p + 4))
    sleep 5
  done
done

echo "done=$(date -Is)" >>"${OUT}/run_meta.txt"
echo ""
echo "======================================================================"
echo "C6 GOODPUT SUMMARY"
echo "======================================================================"
"$PY" "$SUMM" "$OUT"
