#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR=/workspace/vllm/logs/kv_pipelining
mkdir -p "${RESULT_DIR}"
exec >"${RESULT_DIR}/pipelined_run.log" 2>&1

export PYTHONPATH=/workspace/vllm

MODE=pipelined
MODEL=facebook/opt-350m
PORT=8000
INPUT_LEN=1024
OUTPUT_LEN=256
MAX_MODEL_LEN=2048
GPU_UTIL=0.50
START_CONCURRENCY=32
END_CONCURRENCY=128
STEP=32
K_WINDOW=32

python3 -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --disable-log-requests \
  --max-num-seqs "${END_CONCURRENCY}" \
  --enable-kv-pipelining \
  --kv-k-window "${K_WINDOW}" \
  --free-original-kv-cache \
  --kv-transfer-config '{"kv_connector":"LayerwisePipelinedConnector","kv_role":"kv_both"}' \
  >"${RESULT_DIR}/server_${MODE}.log" 2>&1 &
SERVER_PID=$!
trap "kill ${SERVER_PID} || true; wait ${SERVER_PID} || true" EXIT

READY_TIMEOUT=300
READY_INTERVAL=2
DEADLINE=$((SECONDS + READY_TIMEOUT))
echo "Waiting for server readiness on ${PORT}..."
while (( SECONDS < DEADLINE )); do
  if python3 -c 'import os,urllib.request; port=os.environ.get("PORT","8000"); url=f"http://127.0.0.1:{port}/v1/models"; urllib.request.urlopen(url, timeout=2).read()' >/dev/null 2>&1; then
    echo "Server ready."
    break
  fi
  sleep "${READY_INTERVAL}"
done

for ((c = START_CONCURRENCY; c <= END_CONCURRENCY; c += STEP)); do
  echo "=== ${MODE} concurrency=${c} ==="
  RESULT_FILE="${RESULT_DIR}/${MODE}_conc${c}.json"
  if ! python3 -m vllm.entrypoints.cli.main bench serve \
    --backend openai \
    --base-url "http://127.0.0.1:${PORT}" \
    --model "${MODEL}" \
    --dataset-name random \
    --input-len "${INPUT_LEN}" \
    --output-len "${OUTPUT_LEN}" \
    --num-prompts "${c}" \
    --request-rate inf \
    --max-concurrency "${c}" \
    --disable-tqdm \
    --save-result \
    --result-filename "${RESULT_FILE}"; then
    echo "OOM or failure at concurrency=${c}. Stopping sweep."
    break
  fi
done
