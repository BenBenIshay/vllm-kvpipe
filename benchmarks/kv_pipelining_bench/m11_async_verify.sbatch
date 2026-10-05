#!/bin/bash
# M11 -- is the step HOST-bound or DEVICE-bound? That decides the last change.
#
#   sbatch --export=ALL,SLICE=A m11_async_verify.sbatch
#
# A CORRECTION FIRST, because it changes what this job is for. I set out to
# test async double-buffering as an unexplored win, on the strength of the
# pool's comment that it is opt-in and was "measured worthless". It is NOT
# unexplored: `--rungd-async` defaults to TRUE in rider_cost_curve.py and
# kv_k_window defaults to 4, so self.k >= 2 and the side stream has been
# ENGAGED IN EVERY MEASUREMENT THIS CAMPAIGN MADE. c = 1.259 ms per rider is
# already a with-async number. There is no flag to flip.
#
# What that leaves is a sharper and more useful question. The per-rider cost is
# 2*L*r H2D copies, and the launch-pattern microbenchmark
# (launch_pattern_bench.py, A100) prices the HOST side of that pattern at 0.149
# ms per layer at 8 riders -- 9.5 ms per step at L=64, matching the in-engine
# issue_runs timer (10.757 ms) to 12%, so it measures the real thing. If the
# copies already overlap compute on the side stream, then what survives into
# step time is HOST work: the cost of issuing the launches, not of moving the
# bytes. The fitted coefficient is consistent with that -- 0.00838 ms per rider
# per layer against 0.0186 raw, i.e. 45% of the host issue cost reaching step
# time, which is what partial host-boundness looks like.
#
# THIS MATTERS BECAUSE IT DECIDES THE LAST CHANGE. The remaining candidate is
# extent adjacency: pack extents tightly and copy whole extents so all r of
# them are ONE contiguous range. The microbenchmark prices it at -89.5% host
# time and -9% device time, with the slack it moves costing gen/S extra bytes
# (3% at ctx 16384, 25% at ctx 2048). That is a win if and only if the step is
# host-bound. If it is device-bound, cutting host launches buys nothing and the
# extra slack bytes make it worse -- which is the T3 failure mode exactly, and
# T3 is the reason this is being measured before it is built.
#
# WHAT THIS MUST ESTABLISH, in order:
#
#   1. CORRECTNESS with the side stream forced on, since a missing wait_event
#      produces silently stale KV. The suite's token-identical test is the gate
#      and runs FIRST.
#
#   2. HOST-BOUND OR DEVICE-BOUND, read off async ON vs OFF. Turning the side
#      stream off serialises the copies onto the main stream, so the step must
#      get SLOWER by the device copy time if it is device-bound. If OFF and ON
#      are close, the copies were never on the critical path and the residual
#      is host work -- which is what adjacency removes.
#
#   3. THE MECHANISM, from stream_layer minus issue_runs: that difference is
#      the event bookkeeping, so the pair says whether the host is overlapping
#      or merely waiting somewhere else.
#
#   4. The same at ctx 2048, where the link is a larger share of c, because the
#      host/device balance is exactly what shifts with context.
#
# No delay command anywhere: this cluster's submit filter rejects any batch
# script containing one.
#
#SBATCH --job-name=opt-m11
#SBATCH --partition=supernova
#SBATCH --nodelist=nova06
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m11_async_%j.out

set -u
SLICE=${SLICE:-A}

TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/11_async
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct
REPEATS=5

echo "========= M11 slice $SLICE : $(hostname) $(date -Is) ========="
echo "branch: $(git rev-parse --abbrev-ref HEAD)  head: $(git rev-parse --short HEAD)"
VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "tree OK: $VLLM_PATH" ;;
  *) echo "FATAL: wrong tree ($VLLM_PATH)"; exit 10 ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run () {
    local tag=$1 ctx=$2 batch=$3 riders=$4; shift 4
    echo
    echo "=== $tag  ctx=$ctx batch=$batch riders=$riders $* $(date -Is)"
    "$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
        --model "$Q32" --ctx "$ctx" --batch "$batch" --riders "$riders" \
        --connector --repeats "$REPEATS" --short-len 8 --long-len 136 \
        --gpu-mem-util 0.90 --link-gbps 25.1 --k-window 4 \
        --outdir "$OUT" --out "${tag}.json" "$@" 2>&1 | tail -55
    echo "--- $tag done $(date -Is)"
}

# Class A phase timers, so the mechanism can be read as well as the effect
runpt () {
    local tag=$1; shift
    VLLM_KVPIPE_PHASE_TIMERS=1 \
    VLLM_KVPIPE_PHASE_TIMERS_OUT="$OUT/${tag}_phases.jsonl" \
    run "$tag" "$@"
}

case "$SLICE" in

A)
    echo
    echo "######## 1. CORRECTNESS -- a missing wait_event is SILENT ########"
    echo "# suite with the async path forced on (this is also the default)"
    VLLM_KVPIPE_RUNGD_ASYNC=1 bash benchmarks/kv_pipelining_bench/run_suite.sh 2>&1 | tail -30
    SUITE=${PIPESTATUS[0]}
    echo "run_suite exit (async on): $SUITE"
    if [ "$SUITE" -ne 0 ]; then
        echo "FATAL: suite fails with ASYNC=1. NOT measuring."
        exit 20
    fi

    echo
    echo "######## 2. HOST-BOUND OR DEVICE-BOUND, bracketed ########"
    echo "# ON is the DEFAULT (--rungd-async is default True), so OFF is the"
    echo "# arm that needs the explicit flag. Bracketed on/off/on because F"
    echo "# drifts ~0.7 ms per hour."
    run ASY_on_a  512 16 0,1,2,4,6,8
    run ASY_off   512 16 0,1,2,4,6,8 --no-rungd-async
    run ASY_on_b  512 16 0,1,2,4,6,8

    echo
    echo "######## 3. THE MECHANISM -- stream_layer vs issue_runs ########"
    echo "# read the LOW-RIDER rows only: a non_blocking copy still blocks the"
    echo "# host once the copy stream saturates, which is what the knee is."
    runpt ASY_pt_on  512 16 0,1,2,4
    runpt ASY_pt_off 512 16 0,1,2,4 --no-rungd-async

    echo
    echo "######## 4. and at a longer context, where the link dominates c ########"
    run ASY_c2048_on  2048 16 0,1,2
    run ASY_c2048_off 2048 16 0,1,2 --no-rungd-async
    ;;

*)  echo "unknown SLICE=$SLICE"; exit 2 ;;
esac

echo
echo "========= M11 slice $SLICE complete $(date -Is) ========="
ls -la "$OUT"
