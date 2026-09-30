#!/bin/bash
# M0 -- pin the substrate and settle the connector A/B.
#
# ONE allocation for the whole sequence. Per-step srun from a login-node parent
# orphans children that keep running and keep writing to the output paths; that
# drained two nodes on 2026-09-06.
#
#SBATCH --job-name=ovh-m0
#SBATCH --partition=supernova
#SBATCH --nodelist=nova07
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m0_baseline.%j.out

set -u

TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/00_baseline
mkdir -p "$OUT"
cd "$TREE" || exit 1

export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

echo "================ M0 baseline: $(hostname) $(date -Is) ================"
echo "load: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "ulimit -l: $(ulimit -l)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

# ---------------------------------------------------------------- gate 0: tree
# `pip list` lies about which tree is running. Ask the importer. The editable
# finder hardcodes an absolute path and is consulted before sys.path, so a
# shared venv would silently execute the MAIN tree's vllm -- the failure that
# produced two wrong diagnoses on 2026-09-04.
echo "--- gate 0: tree identity"
VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
echo "vllm resolves to: $VLLM_PATH"
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "  OK: this worktree" ;;
  *) echo "  FATAL: wrong tree -- refusing to measure"; exit 10 ;;
esac

# The tiering flags must still exist in the argparse. A `git checkout main --
# vllm/` corrupted the tree for three days in September while git status stayed
# clean, because the fork-only modules survived and only the wiring was gone.
echo "--- gate 0b: tiering flags present"
"$PY" -c "
import sys
from vllm.engine.arg_utils import EngineArgs
import argparse
p = argparse.ArgumentParser()
EngineArgs.add_cli_args(p)
opts = {a for ac in p._actions for a in ac.option_strings}
need = ['--enable-hybrid-kv-tiers', '--hybrid-b-hbm-max',
        '--enable-transfer-aware-admission']
miss = [n for n in need if n not in opts]
print('  missing:', miss if miss else 'none')
sys.exit(1 if miss else 0)
" || { echo "  FATAL: tiering flags gone"; exit 11; }

# ------------------------------------------------------- gate 0c: nothing else
LIVE=$(ps -u "$USER" -o args= | grep -cE "[.]venv/bin/python.*(kv_pipelining_bench|vllm[.]entrypoints)|^VLLM::EngineCore")
echo "--- gate 0c: live engine processes = $LIVE (expect 1, this shell's grep parent aside)"

# ----------------------------------------------------------- gate 1: the suite
# Tests 4-9 need the GPU, so it has to run inside the allocation. Do NOT export
# VLLM_ENABLE_V1_MULTIPROCESSING here: migration_correctness/burst_recovery need
# it 0, hybrid_rungd_real FAILS if it leaks in. run_suite.sh sets it per test.
echo "--- gate 1: correctness suite"
bash benchmarks/kv_pipelining_bench/run_suite.sh 2>&1 | tail -25
SUITE=$?
echo "suite exit=$SUITE (0 = 9/9)"

# ------------------------------------------- A: decode-node rig (ground rule 1)
# Reproduces curve_a100_final.json: expect 53.5 / 71.6 / 97.5 / 146.9 ms.
# --force-riders is DEFAULT ON and sets VLLM_KVPIPE_B1_MEASURED=0, i.e. the
# admission controller is disabled for the measurement. That is deliberate here
# (we are measuring the mechanism, not the controller) and is recorded in the
# JSON, but it must be labelled as such in anything reported.
echo "--- A: decode-node rig, connector ON"
"$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
  --model Qwen/Qwen2.5-32B-Instruct --ctx 2048 --batch 16 \
  --riders 0,2,4,6 --connector --repeats 3 \
  --short-len 8 --long-len 136 --gpu-mem-util 0.90 --link-gbps 25.1 \
  --outdir "$OUT" --out A_decode_node.json 2>&1 | tail -45

# --------------------------------------------------- B: co-located rig, A/B only
# The 4.2b config as it was actually run (curve_32b_v2.json): no connector. This
# is the arm that produced the paper's F = 12.3 ms anchor and the 0.86x
# operating point. Riders structurally cap at 2 here, so r=4/6 are expected to
# fail the purity gate -- that failure IS the result.
echo "--- B: co-located rig, connector OFF (A/B only)"
"$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
  --model Qwen/Qwen2.5-32B-Instruct --ctx 2048 --batch 16 \
  --riders 0,2,4,6 --allow-colocated --repeats 3 \
  --short-len 8 --long-len 136 --gpu-mem-util 0.90 --link-gbps 25.1 \
  --outdir "$OUT" --out B_colocated.json 2>&1 | tail -45

# ------------------------------------------------------------------- teardown
echo "--- teardown"
"$PY" -c "
import sys; sys.path.insert(0,'benchmarks/kv_pipelining_bench')
try:
    from bench_teardown import terminate_engine
    terminate_engine()
    print('  teardown ok')
except Exception as e:
    print('  teardown helper unavailable:', e)
" 2>&1 | tail -5
sleep 5
ps -u "$USER" -o args= | grep -E "[.]venv/bin/python.*kv_pipelining_bench|^VLLM::EngineCore" | head -5

echo "================ M0 done $(date -Is) ================"
