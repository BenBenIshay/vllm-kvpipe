#!/usr/bin/env bash
# P1: nsys profile decode on a running server (avoids nsys hang on teardown).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench"
OUT="${OUT:-/data/ben/kv_phase_a_internal/open_loop_eval_v1/profile_k1_server_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8235}"
K_WINDOW="${K_WINDOW:-4}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
NSYS_DURATION="${NSYS_DURATION:-120}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/ben/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUT"
SERVER_LOG="$OUT/server.log"
NSYS_BASE="$OUT/nsys_decode"

wait_health() {
  local deadline=$((SECONDS + 300))
  while (( SECONDS < deadline )); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

KV=$(printf '{"kv_connector":"PrefillStagingPipelinedConnector","kv_role":"kv_both"}')
SRV_CMD=(
  "$PY" -m vllm.entrypoints.openai.api_server
  --model "$MODEL" --port "$PORT" --max-model-len 4096
  --gpu-memory-utilization 0.85 --max-num-seqs 8
  --disable-log-requests --dtype auto --enforce-eager
  --enable-kv-pipelining --kv-k-window "$K_WINDOW"
  --free-original-kv-cache
  --kv-transfer-config "$KV" --no-enable-prefix-caching
)

run_workload() {
  "$PY" - "$PORT" "$MODEL" "$INPUT_LEN" "$OUTPUT_LEN" <<'PY'
import json, sys, urllib.request
port, model, inp, out_len = sys.argv[1:5]
prompt = "x " * max(1, int(inp))
body = json.dumps({
    "model": model, "prompt": prompt, "max_tokens": int(out_len),
    "stream": True, "temperature": 0,
}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=body, headers={"Content-Type": "application/json"}, method="POST",
)
with urllib.request.urlopen(req, timeout=600) as resp:
    for _ in resp:
        pass
print("stream_done", flush=True)
PY
}

echo "=== P1 server profile: $MODEL k_window=$K_WINDOW ==="
echo "OUT=$OUT"

# Start server (outside nsys capture delay window we use delay+duration on wrapper)
"${SRV_CMD[@]}" >"$SERVER_LOG" 2>&1 &
SRV_PID=$!
cleanup() { kill -TERM "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT

if ! wait_health; then
  echo "Server failed; tail $SERVER_LOG" >&2
  tail -30 "$SERVER_LOG" >&2
  exit 1
fi
echo "[OK] server up pid=$SRV_PID"

# Profile only the client workload while server already holds CUDA context.
# Worker activity appears in the trace via CUDA activity from the server process
# when using --trace=cuda with system-wide capture is NOT default.
# Instead: wrap server launch with delay so prefill+decode are captured.
kill -TERM "$SRV_PID" 2>/dev/null || true
wait "$SRV_PID" 2>/dev/null || true
trap - EXIT
sleep 5

nsys profile -o "$NSYS_BASE" --trace=cuda,nvtx --force-overwrite=true \
  --delay=45 --duration="$NSYS_DURATION" \
  "${SRV_CMD[@]}" >"$SERVER_LOG" 2>&1 &
SRV_PID=$!
trap 'kill -TERM $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null || true' EXIT

if ! wait_health; then
  echo "Server failed after nsys wrap" >&2
  tail -40 "$SERVER_LOG" >&2
  exit 1
fi
echo "[OK] server ready; waiting 5s then streaming request inside capture window ..."
sleep 5
run_workload
echo "[OK] workload done; waiting for nsys duration ..."
wait "$SRV_PID" 2>/dev/null || true
trap - EXIT

REP="${NSYS_BASE}.nsys-rep"
if [[ -f "$REP" ]]; then
  nsys stats --force-export true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
    "$REP" 2>&1 | tee "$OUT/nsys_stats.txt" || true
fi
"$PY" "$BENCH/kv_transfer_budget.py" \
  --model "$MODEL" --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" \
  --k-window "$K_WINDOW" --concurrency 1 | tee "$OUT/transfer_budget.txt"

echo "Done: $REP"
