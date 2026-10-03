#!/bin/bash
# M8 -- verify T1 (the block-table rewrite) and price T2 (Phase 0).
#
#   sbatch --export=ALL,SLICE=A m8_opt_verify.sbatch
#
# SLICE A is ordered so that a failure stops the useful work early:
#
#   1. CORRECTNESS FIRST. run_suite.sh must be 9/9. T1 changes a hot path that
#      feeds attention, so nothing else in this job means anything until the
#      token-identical and bitwise gates pass. The CPU-only equivalence test
#      (build_step_equiv_test.py, 14 cases against the pre-change algorithm
#      carried verbatim) has already passed, but it exercises build_step in
#      isolation -- the suite is what exercises it inside the engine.
#
#      Env trap, learned the hard way: migration_correctness and burst_recovery
#      need VLLM_ENABLE_V1_MULTIPROCESSING=0, and hybrid_rungd_real FAILS if it
#      leaks in. run_suite.sh sets it per-test; never export it here.
#
#   2. T1's EFFECT, at the context where it is largest we can reach. The
#      per-block term was measured at 0.00765 ms/block, so at ctx 2048 / bs 16
#      with 8 riders it was ~1024 tensor __setitem__ calls per step. The Class A
#      timer `build_step_bt_python` brackets a pure-CPU block, so it is
#      trustworthy here -- unlike the phase timers that misled this campaign
#      twice, which were bracketing BLOCKING CUDA calls.
#
#      Also run at bs 64. T1 removes the dependence on the BLOCK COUNT, so a 4x
#      block-size change should now move build_step_bt_python hardly at all,
#      where before it moved it ~4x. That is the sharpest available test that
#      the term's scaling is actually gone rather than merely smaller.
#
#   3. PHASE 0, the T2 decision. Four arms at one config, with `real` brackets
#      straddling the test arms so drift is bounded INSIDE the comparison
#      rather than by appeal to a separate measurement -- F drifted 11.67 ->
#      12.39 over an hour of GPU work in chronological order, so arms 20
#      minutes apart are not comparable.
#
#         real_a   - probe  = the SYNCHRONISATION cost = the realisable prize
#         probe    - skip   = the async copy + scatter a correct fix still pays
#         real_a   - real_b = the drift bracket, which must be small against it
#
#      STANDING RECOMMENDATION from the proposal: if the prize is <= 2 ms,
#      do NOT implement T2. Spend the effort on T1, which is 3.9 ms per rider
#      at the target and carries no ordering risk.
#
# ctx 2048 is used for the T1 arms because the per-block term scales with
# context and 2048 is the longest we can run; its knee is ~2.4 riders, so the
# rider points are deliberately low and the step times are NOT a measurement of
# the overhead law here -- only build_step_bt_python is being read.
#
# No delay command anywhere: this cluster's submit filter rejects any batch
# script containing one.
#
#SBATCH --job-name=opt-m8
#SBATCH --partition=supernova
#SBATCH --nodelist=nova07
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m8_opt_%j.out

set -u
SLICE=${SLICE:-A}

TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/08_opt
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct
REPEATS=5

echo "========= M8 slice $SLICE : $(hostname) $(date -Is) ========="
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
    echo "######## 1. CORRECTNESS -- nothing below counts without this ########"
    bash benchmarks/kv_pipelining_bench/run_suite.sh 2>&1 | tail -30
    SUITE=${PIPESTATUS[0]}
    echo "run_suite exit: $SUITE"
    if [ "$SUITE" -ne 0 ]; then
        echo "FATAL: suite failed, T1 is not correct. Stopping before any"
        echo "timing arm, because a timing number from an incorrect path is"
        echo "worse than no number."
        exit 20
    fi

    echo
    echo "######## 2. T1 -- did the per-block term lose its scaling? ########"
    run T1_bs16 2048 16 0,1,2 --block-size 16
    run T1_bs64 2048 16 0,1,2 --block-size 64

    echo
    echo "######## 3. PHASE 0 -- price the synchronisation ########"
    run P0_real_a 512 16 0,1,2,4,6,8
    run P0_probe  512 16 0,1,2,4,6,8 --d2h-probe
    run P0_skip   512 16 0,1,2,4,6,8 --skip-d2h
    run P0_real_b 512 16 0,1,2,4,6,8
    ;;

*)  echo "unknown SLICE=$SLICE"; exit 2 ;;
esac

echo
echo "========= M8 slice $SLICE complete $(date -Is) ========="
ls -la "$OUT"
