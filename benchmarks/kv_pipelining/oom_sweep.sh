#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-baseline}"

MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
PORT="${PORT:-8000}"
INPUT_LEN="${INPUT_LEN:-4096}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
START_CONCURRENCY="${START_CONCURRENCY:-8}"
END_CONCURRENCY="${END_CONCURRENCY:-128}"
STEP="${STEP:-8}"
K_WINDOW="${K_WINDOW:-32}"
READY_TIMEOUT="${READY_TIMEOUT:-180}"
READY_INTERVAL="${READY_INTERVAL:-2}"

RESULT_DIR="${RESULT_DIR:-/tmp/vllm_kv_pipelining_results}"
mkdir -p "${RESULT_DIR}"

function start_server() {
  local mode="$1"
  local extra_args=()
  if [[ "${mode}" == "pipelined" ]]; then
    extra_args+=(
      --enable-kv-pipelining
      --kv-k-window "${K_WINDOW}"
      --free-original-kv-cache
      --kv-transfer-config "{\"kv_connector\":\"LayerwisePipelinedConnector\"}"
      --no-enable-prefix-caching
    )
  fi

  python3 -m vllm.entrypoints.cli.main serve "${MODEL}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --disable-log-requests \
    --max-num-seqs "${END_CONCURRENCY}" \
    "${extra_args[@]}" \
    >"${RESULT_DIR}/server_${mode}.log" 2>&1 &
  SERVER_PID=$!
}

function stop_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" || true
    wait "${SERVER_PID}" || true
  fi
}

trap stop_server EXIT

start_server "${MODE}"

function wait_for_server() {
  local deadline=$((SECONDS + READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if python3 - <<PY >/dev/null 2>&1; then
import urllib.request
import urllib.error
url = "http://127.0.0.1:${PORT}/v1/models"
try:
    with urllib.request.urlopen(url, timeout=2) as resp:
        resp.read()
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep "${READY_INTERVAL}"
  done
  echo "Server did not become ready within ${READY_TIMEOUT}s"
  return 1
}

wait_for_server

for ((c = START_CONCURRENCY; c <= END_CONCURRENCY; c += STEP)); do
  echo "=== ${MODE} concurrency=${c} ==="
  RESULT_FILE="${RESULT_DIR}/${MODE}_conc${c}.json"
  if ! python3 -m vllm.entrypoints.cli.main bench serve \
    --backend openai \
    --base-url "http://127.0.0.1:${PORT}" \
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
