#!/usr/bin/env bash
# C1 + C2: full-MHA capacity wall vs layer-wise KV pipelining throughput lift.
#
# Wraps eval_pd_decode.py (baseline = all KV in HBM; pipelined = K-window + CPU
# offload, dummy KV). Baseline hits an OOM/concurrency wall in a memory-bound
# regime; pipelined sustains more sequences -> higher throughput.
#
# Smoke (correctness) vs full (result-collection) via SMOKE=1.
#   SMOKE=1 bash run_c1_c2_capacity.sh /tmp/c1c2_smoke
#   MODEL=... CONC="64 128 256" bash run_c1_c2_capacity.sh /data/.../C1_C2
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-${ROOT}/.venv/bin/python}"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench/eval_pd_decode.py"
OUT="${1:-/tmp/c1c2_capacity}"
PORT="${PORT:-8600}"
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
  CONC="${CONC:-2 4}"
  NUM_PROMPTS="${NUM_PROMPTS:-12}"
  RIN="${RIN:-256}"
  ROUT="${ROUT:-16}"
  K_WINDOW="${K_WINDOW:-4}"
else
  MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
  GPU_UTIL="${GPU_UTIL:-0.90}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-33000}"
  CONC="${CONC:-64 128 256 384 512}"
  NUM_PROMPTS="${NUM_PROMPTS:-200}"
  RIN="${RIN:-8192}"
  ROUT="${ROUT:-128}"
  K_WINDOW="${K_WINDOW:-4}"
fi

{
  echo "suite=C1_C2_capacity"
  echo "smoke=${SMOKE}"
  echo "model=${MODEL}"
  echo "gpu_util=${GPU_UTIL} max_model_len=${MAX_MODEL_LEN} k_window=${K_WINDOW}"
  echo "concurrency=${CONC} num_prompts=${NUM_PROMPTS} rin=${RIN} rout=${ROUT}"
  echo "start=$(date -Is)"
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null || true
} | tee "${OUT}/run_meta.txt"

"$PY" "$BENCH" \
  --model "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --k-window "$K_WINDOW" \
  --modes baseline pipelined \
  --concurrency $CONC \
  --num-prompts "$NUM_PROMPTS" \
  --random-input-len "$RIN" \
  --random-output-len "$ROUT" \
  --enforce-eager \
  --no-inflate-kv-pool \
  --result-dir "$OUT" 2>&1 | tee "${OUT}/run.log"

echo "done=$(date -Is)" >>"${OUT}/run_meta.txt"
echo "[C1_C2] results in ${OUT}"
