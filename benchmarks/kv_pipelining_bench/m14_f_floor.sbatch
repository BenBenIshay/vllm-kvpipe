#!/bin/bash
# M14 -- decompose F and find the floor it can reach.
#
#   sbatch --export=ALL,SLICE=A m14_f_floor.sbatch
#
# WHY F. M13 settled that c is exposed work only (0.61 ms/rider at L=80) and
# clears the gate by 5.5x, while F is 7.68 ms -- 31.7% of the step, three times
# the entire per-rider cost at the knee, and paid the moment ONE rider is
# admitted. Halving F takes the C2C gain from 1.081x to 1.217x; removing c
# altogether would buy at most +10 points. So F is the only gap worth closing.
#
# WHAT F IS MADE OF. F = L * (per-layer host work the feature adds). Two
# mechanisms do that work on every layer of every step:
#
#   D2H  the mirror write -- the new token's KV copied device->host so the host
#        mirror stays current. T2 made it async into pinned staging with a
#        deferred scatter.
#   H2D  the streaming -- the rider window copied host->device into the rolling
#        slot, issued per layer.
#
# plus whatever bookkeeping exists around them that neither mechanism needs.
#
# THE ABLATION. Each knob removes one mechanism while leaving the rest of the
# path -- including every hook, event and stream ordering -- in place:
#
#   SKIP_D2H=1              the mirror write is not performed
#   XFER_MODE=nobus         the views are built, no H2D copy is issued
#   both                    neither mechanism runs
#
# so, at fixed L:
#
#   F(base) - F(skip_d2h)   = what the mirror write costs
#   F(base) - F(nobus)      = what the streaming costs
#   F(both)                 = THE IRREDUCIBLE REMAINDER -- bookkeeping that
#                             neither mechanism requires, and therefore the
#                             part that could in principle be engineered away
#
# All three ablations are NUMERICALLY WRONG by construction (stale or absent
# KV). They are timing-only, they announce themselves, and nothing here is a
# correctness claim -- the same standing rule as the layer-subset knob.
#
# AND THE DEPTH AXIS. F is per-layer, so the floor has to be expressed per
# layer, not as one number at one depth. Each arm runs at L=16 and L=64 so the
# per-layer cost of each component is measured rather than assumed linear.
#
# EVERY ARM STAYS BELOW THE KNEE: ctx 128, long_len 48, riders 0..8, which M13
# measured at load <= 0.28 -- and the ablations only lower the load further.
# F is the intercept, so it is the quantity these fits estimate best.
#
# No delay command anywhere: this cluster's submit filter rejects any batch
# script containing one.
#
#SBATCH --job-name=opt-m14
#SBATCH --partition=supernova
#SBATCH --nodelist=nova06
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m14_%j.out

set -u
SLICE=${SLICE:-A}

TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/16_f_floor
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct
RIDERS=0,1,2,3,4,5,6,8

echo "========= M14 slice $SLICE : $(hostname) $(date -Is) ========="
echo "head: $(git rev-parse --short HEAD)   uncommitted: $(git status --porcelain | wc -l)"
VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "tree OK: $VLLM_PATH" ;;
  *) echo "FATAL: wrong tree ($VLLM_PATH)"; exit 10 ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run () {
    local tag=$1 lay=$2; shift 2
    local extra=""
    [ "$lay" != "64" ] && extra="--rungd-layers $lay"
    echo
    echo "=== $tag  L=$lay $* $(date -Is)"
    VLLM_KVPIPE_PHASE_TIMERS=1 \
    VLLM_KVPIPE_PHASE_TIMERS_OUT="$OUT/${tag}_phases.jsonl" \
    "$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
        --model "$Q32" --ctx 128 --batch 16 --riders "$RIDERS" \
        --connector --repeats 5 --short-len 8 --long-len 48 \
        --gpu-mem-util 0.90 --link-gbps 25.1 --k-window 4 \
        --outdir "$OUT" --out "${tag}.json" $extra "$@" 2>&1 | tail -16
    echo "--- $tag done $(date -Is)"
}

case "$SLICE" in
A)
    echo
    echo "######## L=64: the four corners of the ablation"
    run FF_base_L64   64
    run FF_nod2h_L64  64 --skip-d2h
    run FF_nobus_L64  64 --xfer-mode nobus
    run FF_both_L64   64 --skip-d2h --xfer-mode nobus

    echo
    echo "######## L=16: the same four, so each component gets a PER-LAYER cost"
    run FF_base_L16   16
    run FF_nod2h_L16  16 --skip-d2h
    run FF_nobus_L16  16 --xfer-mode nobus
    run FF_both_L16   16 --skip-d2h --xfer-mode nobus

    echo
    echo "######## and the pre-T2 blocking write, to price what T2 already won"
    run FF_syncd2h_L64 64 --no-d2h-async
    ;;
*)  echo "unknown SLICE=$SLICE"; exit 2 ;;
esac

echo
echo "========= M14 slice $SLICE complete $(date -Is) ========="
ls -la "$OUT"
