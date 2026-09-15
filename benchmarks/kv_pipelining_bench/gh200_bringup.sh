#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Bring-up for a rented GH200. ONE script, run it first, read the verdict.
#
# WHY THIS EXISTS. The plan used to describe bring-up in prose and assign two
# of its checks to a person ("confirm nvcc", "confirm the checkpoint is
# staged"). On a metered machine a checklist item that depends on someone
# remembering is a checklist item that gets skipped, and both of those
# particular ones are silent when they fail: no nvcc means FlashInfer and
# DirectKV quietly drop out of the plan, and a missing checkpoint means the
# first experiment spends its first twenty minutes downloading. Everything a
# human was supposed to confirm is now a check with an exit code.
#
# WHAT IT DOES NOT DO. It does not tune anything and it does not measure the
# feature. It answers one question: is this machine in a state where the
# numbers we take on it will mean something? Every gate below corresponds to a
# way we have already been wrong.
#
# Usage:
#   bash benchmarks/kv_pipelining_bench/gh200_bringup.sh                # full
#   bash benchmarks/kv_pipelining_bench/gh200_bringup.sh --quick        # skip suite
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct ... --quick                        # A100 rehearsal
#
# Exit 0 = GO. Non-zero = the number of hard gates that failed.

set -u
cd "$(dirname "$0")/../.." || exit 1
P=benchmarks/kv_pipelining_bench
PY=${PY:-.venv/bin/python}
MODEL=${MODEL:-Qwen/Qwen2.5-32B-Instruct}
CTX=${CTX:-8192}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

export HF_TOKEN="$(cat hf_token 2>/dev/null | tr -d '\n')"
export HF_HUB_CACHE=${HF_HUB_CACHE:-/mnt/beegfs/benishay/huggingface}

hard=0; soft=0
OUT=${OUT:-bringup_$(date +%Y%m%d_%H%M%S).json}

say()  { printf "\n\033[1m== %s\033[0m\n" "$*"; }
gate() { # gate <ok:0|1> <name> <detail>
  if [ "$1" -eq 0 ]; then printf "  [PASS] %s  %s\n" "$2" "${3:-}"
  else printf "  [FAIL] %s  %s\n" "$2" "${3:-}"; hard=$((hard+1)); fi
}
warn() { # warn <ok:0|1> <name> <detail>  -- records, does not block
  if [ "$1" -eq 0 ]; then printf "  [PASS] %s  %s\n" "$2" "${3:-}"
  else printf "  [WARN] %s  %s\n" "$2" "${3:-}"; soft=$((soft+1)); fi
}

echo "============================================================"
echo " GH200 BRING-UP   model=$MODEL ctx=$CTX util=$GPU_MEM_UTIL"
echo " $(date -Is)   host=$(hostname)"
echo "============================================================"

# ---------------------------------------------------------------- 1. toolchain
say "1. toolchain"
$PY -c "import sys; assert sys.version_info[:2]==(3,12)" 2>/dev/null
gate $? "python is 3.12" "$($PY -V 2>&1)"

$PY - <<'EOF' 2>/dev/null
import torch, sys
print(f"      torch {torch.__version__}  cuda {torch.version.cuda}")
sys.exit(0 if torch.cuda.is_available() else 1)
EOF
gate $? "torch sees a CUDA device"

# nvcc: this was a person's job and is exactly the kind that gets skipped.
# It is NOT a hard gate -- vLLM ships precompiled kernels and the experiments
# that matter run without it. It gates 4.7 (DirectKV) and FlashInfer only, so
# a missing nvcc must be LOUD but must not stop the trip.
if command -v nvcc >/dev/null 2>&1; then
  warn 0 "nvcc present" "$(nvcc --version | tail -1)"
else
  warn 1 "nvcc MISSING" "4.7 (DirectKV) and FlashInfer are OUT. Ask for a CUDA *devel* image, not *runtime*."
fi

# ptxas is NOT the same question and we got this wrong once: it ships inside
# the nvidia pip wheels and is simply not on PATH, so "no ptxas" is usually
# false. Report it, never gate on it.
PTXAS=$($PY -c "import glob,sys;g=glob.glob(sys.prefix+'/lib/python*/site-packages/nvidia/**/ptxas',recursive=True);print(g[0] if g else '')" 2>/dev/null)
warn $([ -n "$PTXAS" ] && echo 0 || echo 1) "ptxas (informational)" "${PTXAS:-not found in wheels}"

# ---------------------------------------------------------------- 2. device
say "2. device and interconnect (MEASURED)"
$PY - "$GPU_MEM_UTIL" <<'EOF'
import sys, torch
util = float(sys.argv[1])
p = torch.cuda.get_device_properties(0)
hbm = p.total_memory / 1e9
print(f"      {p.name}  {hbm:.0f} GB HBM  SM {p.major}.{p.minor}")
free = torch.cuda.mem_get_info()[0] / 1e9
print(f"      free now: {free:.0f} GB")
# An allocation we do not own means someone else's job is on this GPU. Every
# absolute we take here would be contended -- which already cost us a 9.4x
# error in the handover measurement before it was caught.
sys.exit(0 if free > 0.92 * hbm else 3)
EOF
rc=$?
gate $([ $rc -eq 0 ] && echo 0 || echo 1) "GPU is idle and exclusive" \
     "$([ $rc -ne 0 ] && echo 'someone else is resident -- absolutes taken now are contended')"

# ---------------------------------------------------------------- 3. preflight
say "3. arithmetic preflight (link, KV geometry, capacity, mirrors)"
$PY $P/gh200_preflight.py --model "$MODEL" --ctx "$CTX" \
    --gpu-mem-util "$GPU_MEM_UTIL" --out preflight.json 2>&1 | sed 's/^/  /'
gate ${PIPESTATUS[0]} "gh200_preflight"

# The capacity this machine ACTUALLY has at the util we ACTUALLY run. The
# paper's tables quote util-1.0 figures, which describe the DEVICE; our
# harnesses set gpu_memory_utilization 0.90, which for Qwen-32B costs 9.6 GB of
# KV space -- capacity 13.0 -> 8.6 at 8K. An --offered calibrated on the wrong
# column makes a correct run look broken.
$PY - preflight.json <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)          # preflight already failed and said so
cap = d.get("capacity_requests") or d.get("b_cap")
if cap:
    print(f"      >>> measured capacity at this util: {cap:.1f} requests")
    print(f"      >>> set --offered WELL ABOVE {cap:.0f} in 4.0 and 4.1, or the")
    print(f"          load never binds and every arm returns the same number.")
EOF

# ---------------------------------------------------------------- 3b. tree
say "3b. the engine we are about to measure is the one in THIS tree"
# On a RESTORED instance this is the check that matters. The editable install's
# finder is a MetaPathFinder with an absolute path baked in, consulted BEFORE
# sys.path, so PYTHONPATH cannot correct it: a venv restored from a snapshot
# can happily import vllm from a path that no longer holds our code, run
# clean, and measure the wrong thing. `pip list` does not catch this -- it
# reports the dist-info path, which is not what imports use. Ask the importer.
EXPECT="$(pwd)/vllm"
$PY - "$EXPECT" <<'EOF'
import sys, os, vllm
got = os.path.dirname(os.path.abspath(vllm.__file__))
want = os.path.abspath(sys.argv[1])
print(f"      import vllm -> {got}")
if got != want:
    print(f"      EXPECTED    -> {want}")
sys.exit(0 if got == want else 1)
EOF
gate $? "vllm imports from this tree"      "a restored venv pointing elsewhere runs the WRONG code and looks healthy"

# ...and that the engine in this tree still HAS the feature. Checking the path
# is not enough. On 2026-09-12 a commit recorded vllm/ as `main` --
# baseline_integrity.sh swaps it in place to measure the disabled-path cost,
# and the commit captured the swap: 2,478 deletions across 16 files. The
# fork-only modules (vllm/v1/hybrid_*.py) survived, because `git checkout main
# -- vllm/` cannot delete paths main does not know about, so the tree looked
# entirely normal and `git status` was clean for three days. The engine that
# would have flown to a rented GH200 was stock vLLM with our benchmarks
# attached to it -- and every arm would have run, and agreed, and meant
# nothing. Ask the parser, since that is what the experiments depend on.
$PY - <<'EOF'
import sys
from vllm.engine.arg_utils import EngineArgs, FlexibleArgumentParser
p = EngineArgs.add_cli_args(FlexibleArgumentParser())
flags = {"--enable-hybrid-kv-tiers", "--hybrid-b-hbm-max",
         "--enable-transfer-aware-admission"}
known = {s for a in p._actions for s in a.option_strings}
missing = sorted(flags - known)
print(f"      tiering flags parsed: {len(flags) - len(missing)}/{len(flags)}")
if missing:
    print(f"      MISSING: {' '.join(missing)}")
sys.exit(1 if missing else 0)
EOF
gate $? "the engine has the tiering feature" "a vllm/ reverted to main parses none of these and measures nothing"

# ---------------------------------------------------------------- 4. model
say "4. model is present (no downloading on the meter)"
$PY - "$MODEL" <<'EOF'
import sys
from huggingface_hub import snapshot_download
try:
    p = snapshot_download(sys.argv[1], local_files_only=True)
    print(f"      cached: {p}")
except Exception as e:
    print(f"      NOT cached locally: {type(e).__name__}")
    sys.exit(1)
EOF
gate $? "$MODEL is staged" "a 64 GB download inside experiment 4.0 is 20 min of rental"

# ---------------------------------------------------------------- 5. suite
if [ "$QUICK" -eq 1 ]; then
  say "5. correctness suite  -- SKIPPED (--quick)"
  warn 1 "suite not run" "do NOT take numbers from this machine until it passes"
else
  say "5. correctness suite (the port is what is on trial here)"
  bash $P/run_suite.sh 2>&1 | sed 's/^/  /'
  gate ${PIPESTATUS[0]} "run_suite 9/9 on this hardware" \
       "a failure here means we would be measuring a broken port"
fi

# ---------------------------------------------------------------- verdict
echo
echo "============================================================"
if [ "$hard" -eq 0 ]; then
  echo " GO  -- $soft warning(s); numbers taken here are trustworthy"
else
  echo " NO-GO -- $hard hard gate(s) failed, $soft warning(s)"
  echo " Fix before spending rental hours. A failed bring-up is cheap;"
  echo " a week of debugging on the meter is not."
fi
echo "============================================================"
exit "$hard"
