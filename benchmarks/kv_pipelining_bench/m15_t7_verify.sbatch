#!/bin/bash
# M15 -- verify T7 (unified mirror + preallocated rider buffers) and measure F.
#
#   sbatch --export=ALL,SLICE=A m15_t7_verify.sbatch
#
# WHAT T7 IS. M14 priced F's components per layer: the mirror write at 0.0610
# (68% of F at L=64, and the only component linear in depth), a remainder of
# 0.022-0.037 that survives with both mechanisms off, and the streaming at ~0.
# T7 takes the two pieces that need no architectural change:
#
#   T7a  the L per-layer mirrors become ONE pinned allocation, so T2's deferred
#        scatter goes from 2*L index_copy_ calls to ONE per tensor with a
#        vectorised index.                     target: -0.018 ms/layer
#   T7b  the four per-layer allocations in the rider override (index_selects
#        for key/value/query, empty for the output) become grown-only buffers
#        cached per layer.                     target: -0.015 ms/layer
#
# Together ~0.033 ms/layer, which at L=80 is ~2.6 ms of a ~7.8 ms F.
#
# WHAT THIS MUST ESTABLISH, in order, and the order matters:
#
#   1. CORRECTNESS FIRST. Both changes can corrupt KV with nothing raised. T7a's
#      index arithmetic could write a rider's token into another LAYER's mirror;
#      T7b reuses buffers that feed a non_blocking D2H. The suite's
#      token-identical test is the gate and it runs before any timing is read.
#      Off-GPU this is covered by unified_mirror_equiv_test.py (8 cases,
#      bitwise) but that cannot see stream ordering.
#
#   2. THE MECHANISM FIRED. wt_flush_vectorised must be non-zero and
#      wt_flush_per_layer zero: if the uniformity guard never passes in the real
#      engine, T7a is a no-op however well it tests -- which is exactly how T1
#      could have failed and how T5 did disappoint.
#
#   3. WHAT IT BOUGHT, bracketed off/on/off at two depths, below the knee, so F
#      is comparable with M14 arm-for-arm. VLLM_KVPIPE_RUNGD_UNIFIED_MIRROR=0
#      restores the per-layer scatter for the control.
#
# Note T7b has no kill switch -- it is a pure allocation change with no
# behavioural flag -- so the off arms isolate T7a only, and T7b shows up as the
# difference between these base arms and M14's.
#
# Every arm: ctx 128, long_len 48, riders 0..8, which M14 measured at load <=
# 0.28. F is the intercept, so these fits estimate it best.
#
# No delay command anywhere: this cluster's submit filter rejects any batch
# script containing one.
#
#SBATCH --job-name=opt-m15
#SBATCH --partition=supernova
#SBATCH --nodelist=nova06
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m15_%j.out

set -u
SLICE=${SLICE:-A}

TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/17_t7
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct
RIDERS=0,1,2,3,4,5,6,8

echo "========= M15 slice $SLICE : $(hostname) $(date -Is) ========="
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
    env "$@" "$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
        --model "$Q32" --ctx 128 --batch 16 --riders "$RIDERS" \
        --connector --repeats 5 --short-len 8 --long-len 48 \
        --gpu-mem-util 0.90 --link-gbps 25.1 --k-window 4 \
        --outdir "$OUT" --out "${tag}.json" $extra 2>&1 | tail -16
    echo "--- $tag done $(date -Is)"
    echo "    vectorised flushes: $(grep -c 'wt_flush_vectorised' "$OUT/${tag}_phases.jsonl" 2>/dev/null)"
    echo "    per-layer  flushes: $(grep -c 'wt_flush_per_layer' "$OUT/${tag}_phases.jsonl" 2>/dev/null)"
}

case "$SLICE" in
A)
    echo
    echo "######## 1. CORRECTNESS -- both changes can corrupt KV silently ####"
    bash benchmarks/kv_pipelining_bench/run_suite.sh 2>&1 | tail -26
    SUITE=${PIPESTATUS[0]}
    echo "run_suite exit: $SUITE"
    if [ "$SUITE" -ne 0 ]; then
        echo "FATAL: suite failed with T7 on. NOT measuring -- fix or revert."
        exit 20
    fi

    echo
    echo "######## 2+3. DID IT FIRE, AND WHAT DID IT BUY (bracketed) ########"
    run T7_on_a   64
    run T7_off    64 VLLM_KVPIPE_RUNGD_UNIFIED_MIRROR=0
    run T7_on_b   64

    echo
    echo "######## 4. the same at L=16, for the per-layer cost ########"
    run T7_on_L16  16
    run T7_off_L16 16 VLLM_KVPIPE_RUNGD_UNIFIED_MIRROR=0

    echo
    echo "######## 5. and the mirror-write ablation again, to see F's split move"
    run T7_nod2h_L64 64 VLLM_KVPIPE_RUNGD_SKIP_D2H=1
    ;;
*)  echo "unknown SLICE=$SLICE"; exit 2 ;;
esac

echo
echo "========= M15 slice $SLICE complete $(date -Is) ========="
ls -la "$OUT"
