#!/bin/bash
# M9 -- verify T2 (the deferred async mirror write) and measure what it bought.
#
#   sbatch --export=ALL,SLICE=A m9_t2_verify.sbatch
#
# T2 defers the host-side scatter of write_token's mirror update to the start of
# the next step, behind one event wait, so 2L BLOCKING copies per step become 2L
# async ones. Phase 0 priced the synchronisation it removes at 8.47 ms of F,
# against 0.25 ms of drift -- a 33.5x ratio -- which is why it was built.
#
# CORRECTNESS FIRST, and for T2 it carries more weight than it did for T1. T1
# was a pure-CPU rewrite whose output is asserted bitwise against the old
# algorithm. T2 changes WHEN a device-to-host copy becomes visible to the host,
# so the thing that can go wrong is ordering, and the failure mode is silent:
# _issue_runs re-streams the mirror host-to-device on the hot path next step, so
# a missed flush feeds attention stale KV and the tokens are quietly wrong
# rather than the run crashing. Only the token-identical end-to-end test catches
# that, which is why run_suite must be 9/9 before any timing arm runs.
#
# The CPU-side half is already covered off-GPU: build_step_equiv_test.py has
# four T2 cases (idempotent no-op, deferred scatter landing in the right rows
# and nowhere else, no re-apply on a second flush, and build_step flushing
# BEFORE it proceeds) and is 18/18.
#
# THE MEASUREMENT is a bracketed before/after on one configuration, because F
# drifts ~0.7 ms/hour and the effect is ~8 ms: async_a / blocking / async_b, so
# the drift is bounded INSIDE the comparison. Both arms are numerically valid --
# unlike the probe and skip arms, both of these update the mirror -- so the
# step times are real and the purity gate applies normally.
#
# No delay command anywhere: this cluster's submit filter rejects any batch
# script containing one.
#
#SBATCH --job-name=opt-m9
#SBATCH --partition=supernova
#SBATCH --nodelist=nova07
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=05:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m9_t2_%j.out

set -u
SLICE=${SLICE:-A}

TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/09_t2
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct
REPEATS=5

echo "========= M9 slice $SLICE : $(hostname) $(date -Is) ========="
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
        --gpu-mem-util 0.90 --link-gbps 25.1 \
        --outdir "$OUT" --out "${tag}.json" "$@" 2>&1 | tail -55
    echo "--- $tag done $(date -Is)"
}

case "$SLICE" in

A)
    echo
    echo "######## 1. CORRECTNESS -- T2's failure mode is SILENT ########"
    echo "# Only the token-identical test catches a missed flush: stale KV"
    echo "# produces wrong tokens, not a crash."
    bash benchmarks/kv_pipelining_bench/run_suite.sh 2>&1 | tail -30
    SUITE=${PIPESTATUS[0]}
    echo "run_suite exit: $SUITE"
    if [ "$SUITE" -ne 0 ]; then
        echo "FATAL: suite failed with T2 on. NOT measuring -- revert first."
        exit 20
    fi

    echo
    echo "# and again with T2 OFF, to show the suite is not simply insensitive"
    echo "# to this path (a 9/9 that would pass either way proves nothing)."
    VLLM_KVPIPE_RUNGD_D2H_ASYNC=0 \
        bash benchmarks/kv_pipelining_bench/run_suite.sh 2>&1 | tail -6
    echo "run_suite (T2 off) exit: ${PIPESTATUS[0]}"

    echo
    echo "######## 2. WHAT T2 BOUGHT, bracketed ########"
    run T2_async_a    512 16 0,1,2,4,6,8
    run T2_blocking   512 16 0,1,2,4,6,8 --no-d2h-async
    run T2_async_b    512 16 0,1,2,4,6,8

    echo
    echo "######## 3. and at the longest context we can run ########"
    run T2_c2048_async    2048 16 0,1,2
    run T2_c2048_blocking 2048 16 0,1,2 --no-d2h-async
    ;;

B)  # NEGATIVE CONTROL. Slice A showed the suite passes with T2 on and with
    # it off -- which proves T2 breaks nothing, but NOT that the suite could
    # catch a T2 bug. A test that passes either way is insensitive, and
    # "9/9 with T2 on" would then be no evidence at all.
    #
    # So suppress the deferred flush, which is exactly the silent failure mode
    # (the mirror goes stale and _issue_runs streams wrong KV next step), and
    # require the token-identical test to FAIL. If it passes, the correctness
    # argument for T2 rests on nothing and must be rebuilt.
    echo
    echo "######## NEGATIVE CONTROL: flush suppressed, test MUST fail ########"
    VLLM_KVPIPE_RUNGD_NOFLUSH=1 VLLM_KVPIPE_RUNGD_NOFLUSH_UNSAFE=1 \
        "$PY" benchmarks/kv_pipelining_bench/hybrid_rungd_real.py 2>&1 | tail -25
    RC=${PIPESTATUS[0]}
    echo "token-identical test exit with flush suppressed: $RC"
    if [ "$RC" -eq 0 ]; then
        echo
        echo "BAD: the token-identical test PASSED with the flush suppressed."
        echo "It cannot see a missed flush, so it is not evidence for T2."
        exit 30
    fi
    echo
    echo "GOOD: suppressing the flush breaks the token-identical test, so that"
    echo "test is sensitive to exactly T2's failure mode and its PASS in slice"
    echo "A is meaningful."
    ;;

*)  echo "unknown SLICE=$SLICE"; exit 2 ;;
esac

echo
echo "========= M9 slice $SLICE complete $(date -Is) ========="
ls -la "$OUT"
