#!/usr/bin/env bash
# Paper suite orchestrator. Runs the claim-indexed benchmarks and writes a
# PAPER_RESULTS.md index.
#
#   SMOKE=1 bash run_all.sh /tmp/paper_smoke      # correctness validation (0.5B)
#   bash run_all.sh /data/ben/paper_runs/$(date +%Y%m%d)   # full collection (later)
#
# SMOKE mode (Qwen2.5-0.5B only): runs the two NEW wrappers (C1/C2 capacity,
# C6 goodput). Full mode additionally runs C3/C4 (prove_*.sh) and C5
# (run_injection_sweep_v5.py). C5 is heavy/long; tune via C5_EXTRA.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "${HERE}/.." && pwd)"
ROOT="$(cd "${BENCH_DIR}/.." && pwd)"
PY="${PY:-${ROOT}/.venv/bin/python}"
OUT="${1:-/tmp/paper_suite}"
SMOKE="${SMOKE:-0}"
mkdir -p "$OUT"

# Reap ONLY engine cores orphaned to init (ppid==1) — these come from a crashed
# server frontend whose _kill_server returned early, leaving the engine core
# holding GPU. ppid==1 filter keeps this safe on a shared GPU (a healthy job
# still has its live api_server parent, ppid!=1).
reap_orphans() {
  local pid ppid
  for pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null || true); do
    ppid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"
    if [[ "$ppid" == "1" ]]; then
      echo "[reap] orphaned EngineCore pid=${pid} (ppid=1); killing"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  sleep 2
}

echo "==== paper suite (smoke=${SMOKE}) -> ${OUT} ===="
reap_orphans

# ---- C1 + C2: capacity wall + throughput ----
SMOKE="$SMOKE" bash "${HERE}/run_c1_c2_capacity.sh" "${OUT}/C1_C2_capacity" \
  2>&1 | tee "${OUT}/c1_c2.log"

reap_orphans

# ---- C6: SLO-aware goodput ----
if [[ "$SMOKE" == "1" ]]; then
  SMOKE=1 bash "${HERE}/run_c6_goodput.sh" "${OUT}/C6_goodput" \
    2>&1 | tee "${OUT}/c6.log"
else
  # Full path needs a predictor CSV (ITL(K) table). Build it via the C6
  # calibration sweep if the caller did not supply PREDICTOR_CSV.
  C6_MODEL="${C6_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
  if [[ -z "${PREDICTOR_CSV:-}" ]]; then
    echo "[C6] no PREDICTOR_CSV given — building one via feature_on_k_sweep ..."
    "$PY" "${BENCH_DIR}/feature_on_k_sweep.py" \
      --models "$C6_MODEL" \
      --k-values ${C6_K_VALUES:-16 32 64 128 192 256 384} \
      --input-len 1024 --output-len 1024 --k-window 4 --repeats 2 \
      --out "${OUT}/C6_calib" 2>&1 | tee "${OUT}/c6_calib.log"
    PREDICTOR_CSV="${OUT}/C6_calib/feature_on_k_sweep.csv"
    reap_orphans
  fi
  PREDICTOR_CSV="$PREDICTOR_CSV" MODEL="$C6_MODEL" \
    bash "${HERE}/run_c6_goodput.sh" "${OUT}/C6_goodput" \
    2>&1 | tee "${OUT}/c6.log"
fi

reap_orphans

if [[ "$SMOKE" != "1" ]]; then
  # ---- C3: conc=1 orchestration tax (reuse) ----
  TRIALS="${TRIALS:-5}" GPU_UTIL="${GPU_UTIL:-0.60}" \
    bash "${BENCH_DIR}/prove_conc1_thesis.sh" "${OUT}/C3_orchestration" 9300 \
    2>&1 | tee "${OUT}/c3.log"
  # ---- C4: amortization sweep (reuse) ----
  TRIALS="${TRIALS:-3}" CONC="1 2 4 8 16 32 64" GPU_UTIL="${GPU_UTIL:-0.60}" \
    bash "${BENCH_DIR}/prove_claim2_conc_sweep.sh" "${OUT}/C4_amortization" 9400 \
    2>&1 | tee "${OUT}/c4.log"
  reap_orphans
  # ---- C5: data-movement / HBM-competition injection sweep ----
  # Long: 12 scenarios x shapes per model. Override grid via C5_EXTRA
  # (e.g. C5_EXTRA="--smoke" or "--workloads-json my.json"). Self-manages env.
  "$PY" "${BENCH_DIR}/run_injection_sweep_v5.py" \
    --out "${OUT}/C5_injection" ${C5_EXTRA:-} \
    2>&1 | tee "${OUT}/c5.log"
  reap_orphans
fi

# ---- Index ----
{
  echo "# Paper suite results index"
  echo ""
  echo "Generated: $(date -Is)  (smoke=${SMOKE})"
  echo ""
  echo "| Claim | Dir | Key output |"
  echo "|-------|-----|------------|"
  echo "| C1+C2 capacity | C1_C2_capacity/ | run.log, results_*.json |"
  echo "| C6 goodput | C6_goodput/ | GOODPUT_RESULTS.md, goodput_summary.csv |"
  if [[ "$SMOKE" != "1" ]]; then
    echo "| C3 orchestration | C3_orchestration/ | THESIS_RESULTS.md |"
    echo "| C4 amortization | C4_amortization/ | THESIS_CLAIM2_RESULTS.md |"
    echo "| C5 data movement | C5_injection/ | injection_sweep_v5.jsonl, .csv |"
    echo "| C6 calibration | C6_calib/ | feature_on_k_sweep.csv |"
  fi
} >"${OUT}/PAPER_RESULTS.md"

echo ""
echo "[paper-suite] wrote ${OUT}/PAPER_RESULTS.md"
