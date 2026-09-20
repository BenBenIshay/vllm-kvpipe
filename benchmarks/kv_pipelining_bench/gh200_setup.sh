#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the environment on a freshly rented GH200, then hand over to the gate.
#
# WHY THIS EXISTS. `gh200_bringup.sh` verifies a machine; it presumed someone
# had already built the thing it verifies. That left the single longest and
# riskiest step -- an aarch64 + sm_90a build of unmeasured duration -- as prose
# in a runbook, which is exactly what that script's own header argues against:
# on a metered machine a checklist item that depends on someone remembering is
# a checklist item that gets skipped. So: everything after `git clone` is here.
#
# The clone itself cannot be, since this file arrives with it.
#
# THREE THINGS IT DOES THAT A PERSON WOULD NOT.
#   * Runs the 62 GB checkpoint download CONCURRENTLY with the build. They
#     contend for nothing -- one is network, the other CPU -- and done in
#     sequence they are the difference between one rental hour and two.
#   * TIMES every stage and writes them out. The plan's "8 hours, expect 1-3"
#     is a stop-loss, not an estimate: nobody has ever measured this build on
#     aarch64. After one run we will know, and the next rental can be sized.
#   * Is resumable. Re-running skips what already succeeded, so a failure at
#     minute 50 does not cost the first 50.
#
# Usage:
#   bash benchmarks/kv_pipelining_bench/gh200_setup.sh
#   MODEL=... HF_HUB_CACHE=/data/hf bash .../gh200_setup.sh
#   bash .../gh200_setup.sh --no-verify     # skip the bring-up gate at the end
#
# Exit 0 = built and the gate passed. Non-zero = stop and read the last stage.

set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT="$(pwd)"
P=benchmarks/kv_pipelining_bench

MODEL=${MODEL:-Qwen/Qwen2.5-32B-Instruct}
PYBIN=${PYBIN:-python3.12}
VENV=${VENV:-$ROOT/.venv}
ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0a}
TIMEBOX_MIN=${TIMEBOX_MIN:-480}          # the plan's 8 h stop-loss
VERIFY=1
[ "${1:-}" = "--no-verify" ] && VERIFY=0
LOG=${LOG:-$ROOT/gh200_setup_$(date +%Y%m%d_%H%M%S).log}
STAMP=$ROOT/.gh200_setup_stage

say()  { printf "\n\033[1m== %s\033[0m\n" "$*" | tee -a "$LOG"; }
note() { printf "   %s\n" "$*" | tee -a "$LOG"; }
done_stage() { grep -qx "$1" "$STAMP" 2>/dev/null; }
mark()       { echo "$1" >> "$STAMP"; }

START=$(date +%s)
elapsed() { echo $(( ($(date +%s) - START) / 60 )); }
check_timebox() {
  if [ "$(elapsed)" -ge "$TIMEBOX_MIN" ]; then
    say "TIME-BOX HIT: $(elapsed) min >= $TIMEBOX_MIN"
    note "The plan's rule is to stop here, release the instance and fix"
    note "offline. A failed bring-up is a cheap lesson; a week of debugging"
    note "on the meter is not."
    exit 3
  fi
}

echo "=====================================================" | tee -a "$LOG"
echo " GH200 SETUP   model=$MODEL   arch=$ARCH_LIST" | tee -a "$LOG"
echo " $(date -Is)  host=$(hostname)  log=$LOG" | tee -a "$LOG"
echo "=====================================================" | tee -a "$LOG"

# ------------------------------------------------------------ 0. environment
say "0. environment"
note "arch:   $(uname -m)   (expect aarch64 -- a Grace host)"
# Do not assume the interpreter is called python3.12. A missing binary here
# used to surface as "command not found" mid-stage and then a venv built by
# whatever `python3` happened to be -- the environment pin says 3.12.13, and a
# silently different minor version is the kind of difference that invalidates a
# comparison rather than failing it.
if ! command -v "$PYBIN" >/dev/null 2>&1; then
  for cand in python3.12 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 \
       && "$cand" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)' 2>/dev/null; then
      PYBIN=$cand; break
    fi
  done
fi
if ! command -v "$PYBIN" >/dev/null 2>&1; then
  note "FATAL: no python 3.12 found (tried python3.12, python3, python)."
  note "ENVIRONMENT_PIN.md records 3.12.13; install it before continuing."
  exit 1
fi
note "python: $($PYBIN --version 2>&1)  [$(command -v "$PYBIN")]"
note "nvcc:   $(command -v nvcc >/dev/null && nvcc --version | tail -1 || echo 'ABSENT -- 4.7 and FlashInfer drop out, not a blocker')"
if [ -z "${HF_HUB_CACHE:-}" ]; then
  export HF_HUB_CACHE="$ROOT/hf_cache"
  note "HF_HUB_CACHE unset -> $HF_HUB_CACHE"
fi
mkdir -p "$HF_HUB_CACHE"
avail_gb=$(df -BG --output=avail "$HF_HUB_CACHE" | tail -1 | tr -dc '0-9')
note "disk at HF_HUB_CACHE: ${avail_gb} GB available (need >=250)"
[ "${avail_gb:-0}" -lt 250 ] && note "WARNING: under 250 GB; the checkpoint alone is 62 GB"
note "pinned-page limit (ulimit -l): $(ulimit -l)  -- a low value crashes the"
note "  host pool later in a way that looks like an engine fault"

# ------------------------------------- 1. checkpoint download, IN BACKGROUND
DL_PID=""
if done_stage "checkpoint"; then
  say "1. checkpoint -- already staged, skipping"
else
  say "1. checkpoint download (background, runs alongside the build)"
  (
    "$PYBIN" - "$MODEL" <<'EOF' >>"$LOG" 2>&1
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], max_workers=8)
print("checkpoint staged")
EOF
  ) &
  DL_PID=$!
  note "started as pid $DL_PID -- the build does not wait for it"
fi

# ---------------------------------------------------------------- 2. venv
if done_stage "venv"; then
  say "2. venv -- already present, skipping"
else
  say "2. venv"
  t=$(date +%s)
  "$PYBIN" -m venv "$VENV" >>"$LOG" 2>&1 \
    && "$VENV/bin/pip" install -q -U pip >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then say "FAILED: venv creation. See $LOG"; exit 1; fi
  note "took $(( ($(date +%s) - t) / 60 )) min"
  mark venv
fi
PY="$VENV/bin/python"

# --------------------------------------------------------------- 3. build
if done_stage "build"; then
  say "3. vLLM build -- already installed, skipping"
else
  say "3. vLLM build (aarch64 + sm_$ARCH_LIST) -- THE LONG POLE, never measured"
  note "time-box $TIMEBOX_MIN min; elapsed $(elapsed) min"
  t=$(date +%s)
  TORCH_CUDA_ARCH_LIST="$ARCH_LIST" "$VENV/bin/pip" install -e . >>"$LOG" 2>&1
  rc=$?
  mins=$(( ($(date +%s) - t) / 60 ))
  if [ $rc -ne 0 ]; then
    say "FAILED after ${mins} min. Last lines:"
    tail -25 "$LOG"
    exit 1
  fi
  note "BUILD TOOK ${mins} MIN -- this is the number the next rental is sized by"
  mark build
  check_timebox
fi

# ----------------------------------------------------- 4. benchmark-only deps
if done_stage "deps"; then
  say "4. benchmark deps -- already installed, skipping"
else
  say "4. benchmark-only deps (pandas, xlsxwriter)"
  note "needed by the multi-turn driver behind 4.2c and 4.5; not imported by vllm/"
  "$VENV/bin/pip" install -q pandas xlsxwriter >>"$LOG" 2>&1 \
    && mark deps || { say "FAILED: see $LOG"; exit 1; }
fi

# ------------------------------------------------- 5. wait for the checkpoint
if [ -n "$DL_PID" ]; then
  say "5. waiting for the checkpoint download"
  note "elapsed $(elapsed) min; if the build was slower than the network this"
  note "returns immediately, which is the point of running them together"
  wait "$DL_PID"
  if [ $? -ne 0 ]; then say "FAILED: checkpoint download. See $LOG"; exit 1; fi
  mark checkpoint
  note "staged in $HF_HUB_CACHE"
fi

# --------------------------------------------------------------- 6. handover
say "6. import check"
$PY - "$ROOT/vllm" <<'EOF' | tee -a "$LOG"
import os
import sys

import vllm
got = os.path.dirname(os.path.abspath(vllm.__file__))
want = os.path.abspath(sys.argv[1])
print(f"   import vllm -> {got}")
sys.exit(0 if got == want else 1)
EOF
[ $? -ne 0 ] && { say "FAILED: vllm imports from the wrong tree"; exit 1; }

say "SETUP COMPLETE in $(elapsed) min"
if [ "$VERIFY" -eq 1 ]; then
  say "handing over to the bring-up gate"
  MODEL="$MODEL" HF_HUB_CACHE="$HF_HUB_CACHE" PY="$PY" bash "$P/gh200_bringup.sh"
  rc=$?
  if [ $rc -ne 0 ]; then
    say "BRING-UP FAILED ($rc hard gate(s)) -- do NOT take numbers from this machine"
    exit "$rc"
  fi
  say "GO. Next: phase 0, the rider cost curve --"
  note "$PY $P/rider_cost_curve.py --model $MODEL --ctx 8192 \\"
  note "    --batch <capacity printed above> --riders 0,2,4,6,8,12 \\"
  note "    --connector --repeats 3 --outdir ./runs/phase0"
fi
exit 0
