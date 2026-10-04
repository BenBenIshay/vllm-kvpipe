#!/bin/bash
# M10 -- verify T3 (the batched per-layer gather) and attribute what is left.
#
#   sbatch --export=ALL,SLICE=A m10_t3_verify.sbatch
#
# T3 attacks the LARGEST remaining item in c. Post T1+T2 the per-rider cost at
# the target is 1.259 ms and 53% of it is the 2*L*r memcpy LAUNCHES -- 0.670 ms
# per rider, which no interconnect touches. T3 replaces the per-run loop with
# ONE strided copy per tensor per layer when the runs are evenly spaced and
# equal length, so launches fall from 2*L*r to 2*L.
#
# Measured from M8: n_runs equals the rider count exactly and h2d_launches is
# 128*r per step at L=64, so the runs do NOT merge today. They do not merge
# because an extent is sized for max_model_len while a request uses only
# ceil(S/bs) of it, leaving a gap before the next extent's base -- but the bases
# ARE evenly spaced, by blocks_per_req, which is what makes one strided copy
# possible.
#
# WHAT THIS JOB MUST ESTABLISH, in order:
#
#   1. CORRECTNESS. T3 moves the same bytes by different index arithmetic. Wrong
#      arithmetic streams the WRONG KV with nothing raised, so the suite's
#      token-identical test is the gate. The CPU half is already covered off-GPU
#      by batch_gather_equiv_test.py (10 cases against the per-run loop, which
#      it carries as the reference) and build_step_equiv_test.py (18).
#
#   2. THE LAUNCH COUNT ACTUALLY DROPS IN THE ENGINE. h2d_launches should read 2
#      per layer-step instead of 2*r, and h2d_batched_steps should be non-zero.
#      If the regularity check never fires in the real engine, T3 is a no-op
#      however well it tests -- which is exactly how T1 could have failed, so it
#      is checked rather than assumed.
#
#   3. WHAT IT BOUGHT, bracketed on / off / on, because the expected effect is
#      ~0.67 ms per rider and F drifts ~0.7 ms per hour.
#
#   4. WHERE THE RESIDUAL F GOES. T2 left 0.048 ms per layer of D2H, 65% of the
#      remaining F, and I do not know how it splits between the async copy
#      launches and the deferred scatter. wt_flush_scatter now times the scatter
#      and write_token_d2h_async the copies, so this attributes it before
#      anything else is built against it.
#
# No delay command anywhere: this cluster's submit filter rejects any batch
# script containing one.
#
#SBATCH --job-name=opt-m10
#SBATCH --partition=supernova
#SBATCH --nodelist=nova07
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m10_t3_%j.out

set -u
SLICE=${SLICE:-A}

TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/10_t3
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct
REPEATS=5

echo "========= M10 slice $SLICE : $(hostname) $(date -Is) ========="
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

# with Class A phase timers, so the launch count and the F split are recorded
runpt () {
    local tag=$1; shift
    VLLM_KVPIPE_PHASE_TIMERS=1 \
    VLLM_KVPIPE_PHASE_TIMERS_OUT="$OUT/${tag}_phases.jsonl" \
    run "$tag" "$@"
    echo "--- launch + batching rollup for $tag:"
    grep -oE '"h2d_launches":\{"calls":[0-9]+' "$OUT/${tag}_phases.jsonl" \
        2>/dev/null | tail -2
    grep -c 'h2d_batched_steps' "$OUT/${tag}_phases.jsonl" 2>/dev/null \
        | xargs -I{} echo "  steps that used the batched gather: {}"
}

case "$SLICE" in

A)
    echo
    echo "######## 1. CORRECTNESS -- wrong index arithmetic is SILENT ########"
    bash benchmarks/kv_pipelining_bench/run_suite.sh 2>&1 | tail -30
    SUITE=${PIPESTATUS[0]}
    echo "run_suite exit: $SUITE"
    if [ "$SUITE" -ne 0 ]; then
        echo "FATAL: suite failed with T3 on. NOT measuring -- revert first."
        exit 20
    fi

    echo
    echo "######## 2. DOES THE LAUNCH COUNT DROP IN THE ENGINE? ########"
    echo "# expect h2d_launches 2 per layer-step with T3, 2*r without."
    runpt T3_on_c512  512 16 0,1,2,4,8
    runpt T3_off_c512 512 16 0,1,2,4,8 --no-batch-gather

    echo
    echo "######## 3. WHAT T3 BOUGHT, bracketed ########"
    run T3_on_a   512 16 0,1,2,4,6,8
    run T3_off    512 16 0,1,2,4,6,8 --no-batch-gather
    run T3_on_b   512 16 0,1,2,4,6,8

    echo
    echo "######## 4. and at the longest context we can run ########"
    run T3_c2048_on  2048 16 0,1,2
    run T3_c2048_off 2048 16 0,1,2 --no-batch-gather
    ;;

*)  echo "unknown SLICE=$SLICE"; exit 2 ;;
esac

echo
echo "========= M10 slice $SLICE complete $(date -Is) ========="
ls -la "$OUT"
