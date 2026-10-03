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

B)  # NEGATIVE CONTROL, second attempt. The first one was wrong in two ways
    # and both are worth recording, because each would have produced a
    # confident but empty claim.
    #
    #   1. It checked the EXIT CODE. hybrid_rungd_real.py's main() has no
    #      sys.exit, so it always exits 0 -- run_suite.sh decides by grepping
    #      stdout for "REAL CORRECTNESS PASS". So the check was meaningless and
    #      read an inconclusive run as a pass.
    #   2. The suppression itself leaked. Returning early without clearing left
    #      _wt_dirty set and grew _wt_pending by one entry per layer per step,
    #      so the pool server failed its completions from the LEAK rather than
    #      from stale KV. The control was testing the wrong failure.
    #
    # Now: the suppression DROPS the pending rows, which is exactly a missed
    # flush and nothing else; and the verdict is read from the output string,
    # distinguishing PASS / FAIL / INCONCLUSIVE rather than collapsing them.
    #
    # A POSITIVE CONTROL runs first. Without it, an INCONCLUSIVE negative arm
    # cannot be told apart from a broken environment.
    verdict () {   # $1 = log file -> PASS | FAIL | INCONCLUSIVE | NONE
        if   grep -q "REAL CORRECTNESS PASS" "$1"; then echo PASS
        elif grep -q "REAL CORRECTNESS FAIL" "$1"; then echo FAIL
        elif grep -q "INCONCLUSIVE"          "$1"; then echo INCONCLUSIVE
        else echo NONE; fi
    }

    echo
    echo "######## POSITIVE CONTROL: flush intact, test must PASS ########"
    "$PY" benchmarks/kv_pipelining_bench/hybrid_rungd_real.py \
        > "$OUT/negctl_positive.log" 2>&1
    POS=$(verdict "$OUT/negctl_positive.log")
    echo "verdict with the flush intact: $POS"
    tail -6 "$OUT/negctl_positive.log"
    if [ "$POS" != "PASS" ]; then
        echo
        echo "FATAL: the token-identical test does not PASS even with the flush"
        echo "intact, so nothing can be concluded from the negative arm. This is"
        echo "an environment or harness problem, not a T2 result."
        exit 31
    fi

    echo
    echo "######## NEGATIVE CONTROL: flush suppressed, test must FAIL ########"
    VLLM_KVPIPE_RUNGD_NOFLUSH=1 VLLM_KVPIPE_RUNGD_NOFLUSH_UNSAFE=1 \
        "$PY" benchmarks/kv_pipelining_bench/hybrid_rungd_real.py \
        > "$OUT/negctl_negative.log" 2>&1
    NEG=$(verdict "$OUT/negctl_negative.log")
    echo "verdict with the flush suppressed: $NEG"
    tail -12 "$OUT/negctl_negative.log"

    echo
    case "$NEG" in
      FAIL)
        echo "GOOD: suppressing the flush makes the token-identical test FAIL,"
        echo "so that test is sensitive to exactly T2's failure mode and its"
        echo "PASS in slice A is meaningful evidence."
        ;;
      PASS)
        echo "BAD: the test PASSED with the flush suppressed. It cannot see a"
        echo "missed flush, so slice A's 9/9 is not evidence for T2 and the"
        echo "correctness argument must be rebuilt."
        exit 30
        ;;
      *)
        echo "INCONCLUSIVE ($NEG): the negative arm did not produce a token"
        echo "comparison, so it neither supports nor refutes T2. The positive"
        echo "arm passed, so the environment is fine -- look at"
        echo "$OUT/negctl_negative.log for why this arm did not run."
        exit 32
        ;;
    esac
    ;;

*)  echo "unknown SLICE=$SLICE"; exit 2 ;;
esac

echo
echo "========= M9 slice $SLICE complete $(date -Is) ========="
ls -la "$OUT"
