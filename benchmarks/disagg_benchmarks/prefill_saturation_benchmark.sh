#!/usr/bin/env bash
set -euo pipefail

# Prefill-focused saturation sweep.
# We approximate prefill-only load by setting output length to 1 token.

MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
PROMPT_LENS_CSV="${PROMPT_LENS_CSV:-512,1024,2048,4096}"
CONCURRENCY_CSV="${CONCURRENCY_CSV:-1,2,4,8,16,32,64}"
NUM_PROMPTS_PER_RUN="${NUM_PROMPTS_PER_RUN:-200}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-300}"
READY_POLL_SEC="${READY_POLL_SEC:-2}"
TEMPERATURE="${TEMPERATURE:-0.0}"
RESULT_DIR="${RESULT_DIR:-$(pwd)/benchmarks/disagg_benchmarks/results/prefill_saturation_$(date +%Y%m%d_%H%M%S)}"
SERVER_LOG="${SERVER_LOG:-${RESULT_DIR}/server.log}"

# Optional extra args for vllm serve, e.g.
# SERVE_EXTRA_ARGS="--tensor-parallel-size 2 --enforce-eager"
SERVE_EXTRA_ARGS="${SERVE_EXTRA_ARGS:-}"

mkdir -p "${RESULT_DIR}"

SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

wait_for_server() {
  local deadline=$((SECONDS + READY_TIMEOUT_SEC))
  while (( SECONDS < deadline )); do
    if curl -s "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${READY_POLL_SEC}"
  done
  echo "Server did not become ready within ${READY_TIMEOUT_SEC}s" >&2
  return 1
}

echo "[1/4] Starting vLLM server for prefill saturation sweep..."
python3 -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --disable-log-requests \
  ${SERVE_EXTRA_ARGS} \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "[2/4] Waiting for server readiness on ${HOST}:${PORT}..."
wait_for_server

echo "[3/4] Running sweeps..."
IFS=',' read -r -a PROMPT_LENS <<< "${PROMPT_LENS_CSV}"
IFS=',' read -r -a CONCURRENCY_LEVELS <<< "${CONCURRENCY_CSV}"

for prompt_len in "${PROMPT_LENS[@]}"; do
  for concurrency in "${CONCURRENCY_LEVELS[@]}"; do
    run_id="prefill_input${prompt_len}_conc${concurrency}"
    result_file="${run_id}.json"
    echo "  -> ${run_id}"

    # output-len=1 minimizes decode work and keeps the run prefill-dominant.
    if ! python3 -m vllm.entrypoints.cli.main bench serve \
      --backend openai \
      --model "${MODEL}" \
      --base-url "http://${HOST}:${PORT}" \
      --dataset-name random \
      --input-len "${prompt_len}" \
      --output-len 1 \
      --temperature "${TEMPERATURE}" \
      --num-prompts "${NUM_PROMPTS_PER_RUN}" \
      --request-rate inf \
      --max-concurrency "${concurrency}" \
      --disable-tqdm \
      --percentile-metrics ttft \
      --metric-percentiles 95,99 \
      --save-result \
      --result-dir "${RESULT_DIR}" \
      --result-filename "${result_file}"; then
      echo "    Failed at ${run_id}; continuing to next point." >&2
    fi
  done
done

echo "[4/4] Building summary reports..."
python3 - "${RESULT_DIR}" <<'PY'
import csv
import json
import pathlib
import re
import sys

result_dir = pathlib.Path(sys.argv[1])
rows = []
pattern = re.compile(r"prefill_input(?P<input_len>\d+)_conc(?P<conc>\d+)\.json$")

for path in sorted(result_dir.glob("prefill_input*_conc*.json")):
    m = pattern.match(path.name)
    if not m:
        continue
    with path.open() as f:
        data = json.load(f)

    duration = float(data.get("duration") or 0.0)
    total_input_tokens = float(data.get("total_input_tokens") or 0.0)
    input_tok_s = (total_input_tokens / duration) if duration > 0 else 0.0

    rows.append(
        {
            "input_len": int(m.group("input_len")),
            "concurrency": int(m.group("conc")),
            "mean_ttft_ms": float(data.get("mean_ttft_ms") or 0.0),
            "p95_ttft_ms": float(data.get("p95_ttft_ms") or 0.0),
            "p99_ttft_ms": float(data.get("p99_ttft_ms") or 0.0),
            "request_throughput_rps": float(data.get("request_throughput") or 0.0),
            "input_tok_s": input_tok_s,
            "total_token_throughput_tok_s": float(
                data.get("total_token_throughput") or 0.0
            ),
            "failed_requests": int(data.get("failed") or 0),
        }
    )

rows.sort(key=lambda x: (x["input_len"], x["concurrency"]))

csv_path = result_dir / "summary.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "input_len",
            "concurrency",
            "mean_ttft_ms",
            "p95_ttft_ms",
            "p99_ttft_ms",
            "request_throughput_rps",
            "input_tok_s",
            "total_token_throughput_tok_s",
            "failed_requests",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

md_path = result_dir / "summary.md"
with md_path.open("w") as f:
    f.write("# Prefill Saturation Summary\n\n")
    f.write(
        "| input_len | concurrency | mean_ttft_ms | p95_ttft_ms | p99_ttft_ms | "
        "request_throughput_rps | input_tok_s | total_token_throughput_tok_s | "
        "failed_requests |\n"
    )
    f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(
            f"| {r['input_len']} | {r['concurrency']} | "
            f"{r['mean_ttft_ms']:.2f} | {r['p95_ttft_ms']:.2f} | "
            f"{r['p99_ttft_ms']:.2f} | {r['request_throughput_rps']:.2f} | "
            f"{r['input_tok_s']:.2f} | {r['total_token_throughput_tok_s']:.2f} | "
            f"{r['failed_requests']} |\n"
        )

print(f"Wrote: {csv_path}")
print(f"Wrote: {md_path}")
PY

echo "Done. Results are in: ${RESULT_DIR}"
echo "Server log: ${SERVER_LOG}"
