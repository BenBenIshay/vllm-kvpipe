#!/usr/bin/env bash
# Validate R3 (fake transfer) with prefix caching ON at the matrix c32/long point.
# Success: server starts, 32/32 requests OK, steady ITL materially below the
# old prefix-OFF R3 (~28 ms at c32 long) and near R0 (~17 ms).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench/pipelined_decode_fast_bench.py"
OUT="${1:-/tmp/r3_prefix_cache_validate}"
PORT="${2:-9200}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/beegfs/benishay/huggingface}"
export VLLM_HF_TOKEN_FILE="${VLLM_HF_TOKEN_FILE:-${ROOT}/hf_token}"
export PYTHONUNBUFFERED=1
export VLLM_KVPIPE_BATCHED_D2H=1 VLLM_KVPIPE_STEP_PLAN=1
export VLLM_KVPIPE_SKIP_DELTA_STAGE=0 VLLM_KVPIPE_FAKE_TRANSFER=1
export VLLM_KVPIPE_PROFILE=0 VLLM_KVPIPE_PROFILE_JSON=0

mkdir -p "$OUT"

echo "=== R3 prefix-caching validation (c32, S=8192) ==="
"$PY" "$BENCH" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --modes pipelined \
  --k-window 4 \
  --concurrency 32 \
  --input-len 8192 --output-len 128 \
  --max-model-len 9216 \
  --enforce-eager \
  --prefix-caching-on \
  --no-auto-gpu-memory --gpu-memory-utilization 0.90 \
  --no-pipelined-gpu-mem-cap \
  --skip-isolated-latency \
  --itl-steady-skip 1 \
  --startup-timeout 900 \
  --request-timeout 900 \
  --port "$PORT" \
  --result-dir "$OUT" 2>&1 | tee "${OUT}/driver.log"

LOG="${OUT}/server_pipelined.log"
echo ""
echo "=== Server config check ==="
grep -E "enable_prefix_caching|prefix.cach" "$LOG" | head -5 || true
grep -E "Prefix cache hit rate|prefix_cache" "$LOG" | tail -5 || true

echo ""
echo "=== Result summary ==="
"$PY" - <<'PY' "$OUT/results_pipelined.json"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
row = d["throughput"][0]
itl = row.get("concurrent_mean_itl_steady_s_mean") or row.get("concurrent_mean_itl_s_mean")
ok = row.get("successes", 0)
conc = row.get("concurrency", 0)
print(f"  successes={ok}/{conc}")
if itl is not None:
    print(f"  steady ITL={itl*1000:.2f} ms")
    print(f"  tok/s (B/ITL)={conc/(itl):.0f}")
else:
    print("  ITL: missing")
    sys.exit(1)
if ok != conc:
    sys.exit(1)
# Old R3 prefix-OFF reference at this point: ~28 ms
if itl * 1000 > 24.0:
    print("  WARN: ITL still high — prefix caching may not be active (expect ~17-20 ms)")
    sys.exit(2)
print("  PASS: ITL in expected prefix-cached range")
PY
