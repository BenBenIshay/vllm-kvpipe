#!/usr/bin/env bash
# P1: Profile K=1 decode overhead with Nsight Systems (Qwen-0.5B default).
# Prefer run_profile_k1_server.sh (delay+duration capture); this wrapper calls it.
exec "$(dirname "$0")/run_profile_k1_server.sh" "$@"
exit 0
# --- legacy probe-only path below (nsys may hang on teardown) ---
#
# Produces: $OUT/nsys_k1_overhead.nsys-rep (open in Nsight Systems GUI)
#
# Prerequisites: nsys in PATH, GPU, HF cache.
#
# Usage:
#   bash benchmarks/kv_pipelining_bench/run_profile_k1_overhead.sh
#   MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct OUT=/tmp/llama_prof bash ...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench"
OUT="${OUT:-/data/ben/kv_phase_a_internal/open_loop_eval_v1/profile_k1_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
K_WINDOW="${K_WINDOW:-4}"
PORT="${PORT:-8230}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/ben/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUT"

if ! command -v nsys >/dev/null 2>&1; then
  echo "ERROR: nsys not found. Install Nsight Systems or load module." >&2
  exit 1
fi

echo "=== P1 profile: model=$MODEL K_window=$K_WINDOW conc=1 ==="
echo "=== OUT=$OUT ==="
echo ""
echo "Transfer budget (theory):"
"$PY" "$BENCH/kv_transfer_budget.py" \
  --model "$MODEL" --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" \
  --k-window "$K_WINDOW" --concurrency 1 | tee "$OUT/transfer_budget.txt"
echo ""

NSYS_BASE="$OUT/nsys_k1_overhead"
PROBE_LOG="$OUT/probe_server.log"

echo "=== nsys profile: one real_prefill streaming request ==="
nsys profile -o "$NSYS_BASE" \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  "$PY" "$BENCH/probe_one_decode.py" \
    --model "$MODEL" \
    --port "$PORT" \
    --k-window "$K_WINDOW" \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --log "$PROBE_LOG" \
    --startup-timeout 300 \
  2>&1 | tee "$OUT/nsys_stdout.txt"

echo ""
cat >"$OUT/HOWTO_READ_PROFILE.txt" <<'EOF'
Open in Nsight Systems:
  nsys-ui <path>/nsys_k1_overhead.nsys-rep

What to check (one decode forward after prefill):
  1. CUDA HW row: attention/memcpy kernels vs gaps (sync).
  2. cudaMemcpy* / Memcpy DtoH/HtoD — sum duration per decode step.
  3. cudaStreamWaitEvent — time on the default/compute stream before attn.
  4. Compare wall time of one forward vs kv_transfer_budget PCIe lower bound.

If Memcpy + WaitEvent dominate vs attention, focus on:
  - wait_for_saves / delta_queue.join once per forward
  - wait_for_layer per layer
  - block-mode host slot_mapping work

Probe log (connector):
  parse_pipeline_debug_log.py probe_server.log
EOF

REP="${NSYS_BASE}.nsys-rep"
if [[ -f "$REP" ]]; then
  echo "=== nsys stats (text summary) ==="
  nsys stats --force-export true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
    "$REP" 2>&1 | tee "$OUT/nsys_stats.txt" || true
fi

echo ""
echo "Done. Open: ${REP}"
echo "Notes: $OUT/HOWTO_READ_PROFILE.txt"
