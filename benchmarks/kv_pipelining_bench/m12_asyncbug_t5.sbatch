#!/bin/bash
# M12 -- the async-prefetch correctness bug, and T5 measured on the correct path
#
#   sbatch --export=ALL,SLICE=A m12_asyncbug_t5.sbatch
#   sbatch --export=ALL,SLICE=B m12_asyncbug_t5.sbatch
#
# THE BUG, found by M11 and reproduced by hand on nova07 2026-10-05:
#
#   hybrid_rungd_real.py                              -> PASS (3/3 identical)
#   hybrid_rungd_real.py  VLLM_KVPIPE_RUNGD_ASYNC=1   -> FAIL (2/3 diverged)
#
# The side-stream prefetch produces WRONG TOKENS. It went unnoticed because
# run_suite.sh never sets the variable -- so the suite's 9/9 has always been
# validating the NON-async path -- while rider_cost_curve.py declares
# `--rungd-async` with default=True and cache_config.kv_k_window defaults to 4,
# so self.k >= 2 and the side stream engages. Every archived artifact in
# paper_runs/20260930/overhead/ records async=True, k_window=4: M0 through M10,
# which is every measured coefficient this campaign produced, c and F included.
#
# A missing wait makes a step FASTER, so the published coefficients are
# plausibly optimistic. The size of that bias is the first thing this job
# measures, and it decides whether the campaign's conclusions stand.
#
# SUSPECTED MECHANISM, to be confirmed by slice A rather than assumed. T2 defers
# the mirror write: write_token copies the new token D2H into pinned staging and
# flush_token_writes scatters it into the mirror on the CPU at the top of the
# next build_step. flush_token_writes' docstring argues the ordering is safe
# because "nothing prefetches across a step boundary" -- but that is about where
# a prefetch is ISSUED, and what crosses the boundary is its COMPLETION.
# stream_layer consumes the last prefetch with
# current_stream().wait_event(ev), which orders the DEVICE and returns to the
# host immediately, so the host runs on into the next step and CPU-writes mirror
# rows that an in-flight copy-stream DMA is still reading.
#
# If that is the mechanism, the race needs T2: before T2 the per-layer mirror
# write was a BLOCKING D2H, which synchronised the host every layer and made the
# scatter safe by accident. So T2 did not introduce a bug of its own -- it
# removed the implicit host sync that was hiding this one. Slice A tests exactly
# that by running the async path with T2 switched off.
#
# WHY THERE IS NO ONE-LINE FIX, which is why this job measures instead of
# patching: any correct ordering makes the host wait on the main stream once per
# step, which is precisely the sync T2 removed, so it would hand back T2's gain.
# The cheap resolution is likely to DEFAULT ASYNC OFF -- it is untested, it is
# wrong, and launch_pattern_bench.py says the overlap only matters near the
# knee, where the copy stream saturates anyway. Slice A's delta decides.
#
# No delay command anywhere: this cluster's submit filter rejects any batch
# script containing one.
#
#SBATCH --job-name=opt-m12
#SBATCH --partition=supernova
#SBATCH --nodelist=nova06
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m12_%j.out

set -u
SLICE=${SLICE:-A}

TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/12_asyncbug
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct
REPEATS=5

echo "========= M12 slice $SLICE : $(hostname) $(date -Is) ========="
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

runpt () {
    local tag=$1; shift
    VLLM_KVPIPE_PHASE_TIMERS=1 \
    VLLM_KVPIPE_PHASE_TIMERS_OUT="$OUT/${tag}_phases.jsonl" \
    run "$tag" "$@"
}

case "$SLICE" in

A)
    echo
    echo "######## 1. ISOLATE THE RACE: does it need T2? ########"
    echo "# Each of these is the token-identical end-to-end test. The pattern"
    echo "# across the four says where the race lives."
    echo
    echo "--- (a) async OFF, T2 ON   = the suite's configuration: expect PASS"
    "$PY" benchmarks/kv_pipelining_bench/hybrid_rungd_real.py 2>&1 | tail -3
    echo
    echo "--- (b) async ON,  T2 ON   = every measurement: expect FAIL"
    VLLM_KVPIPE_RUNGD_ASYNC=1 \
        "$PY" benchmarks/kv_pipelining_bench/hybrid_rungd_real.py 2>&1 | tail -3
    echo
    echo "--- (c) async ON,  T2 OFF  = THE DIAGNOSTIC."
    echo "#     PASS here means the race is the T2 deferred CPU scatter racing"
    echo "#     an in-flight prefetch DMA, i.e. T2 removed the blocking D2H"
    echo "#     that was hiding a pre-existing hazard."
    echo "#     FAIL here means the prefetch is unsafe on its own and T2 is"
    echo "#     not implicated at all."
    VLLM_KVPIPE_RUNGD_ASYNC=1 VLLM_KVPIPE_RUNGD_D2H_ASYNC=0 \
        "$PY" benchmarks/kv_pipelining_bench/hybrid_rungd_real.py 2>&1 | tail -3
    echo
    echo "--- (d) async OFF, T2 OFF  = control, must PASS"
    VLLM_KVPIPE_RUNGD_D2H_ASYNC=0 \
        "$PY" benchmarks/kv_pipelining_bench/hybrid_rungd_real.py 2>&1 | tail -3

    echo
    echo "######## 2. WHAT THE RACY PATH WAS WORTH, bracketed ########"
    echo "# If OFF and ON are within noise then async bought nothing, it was"
    echo "# wrong, and the published coefficients stand with a provenance note."
    echo "# If OFF is materially SLOWER then c and F are worse than published"
    echo "# and every projection has to be recomputed."
    run ASY2_off_a 512 16 0,1,2,4,6,8 --no-rungd-async
    run ASY2_on    512 16 0,1,2,4,6,8
    run ASY2_off_b 512 16 0,1,2,4,6,8 --no-rungd-async

    echo
    echo "######## 3. and at ctx 2048 ########"
    run ASY2_c2048_off 2048 16 0,1,2 --no-rungd-async
    run ASY2_c2048_on  2048 16 0,1,2
    ;;

B)
    # T5 = whole-extent runs. Measured ONLY with async off, because the async
    # path is incorrect and a timing taken on it is not a measurement of
    # anything. ctx 512 + long_len 136 = 648 tokens = 41 blocks at bs 16, so
    # --rungd-bpr 41 is the tight extent; with the runner's default (141, sized
    # for max_model_len) the pool's slack guard refuses the whole-extent path
    # and the run silently measures the per-run one, so the bpr is not optional.
    echo
    echo "######## 1. CORRECTNESS -- wrong index arithmetic is SILENT ########"
    echo "# off-GPU this is covered by whole_extent_equiv_test.py (16 checks,"
    echo "# resolving every block through the block table); this is the engine."
    VLLM_KVPIPE_RUNGD_WHOLE_EXTENT=1 VLLM_KVPIPE_RUNGD_BPR=41 \
        bash benchmarks/kv_pipelining_bench/run_suite.sh 2>&1 | tail -30
    SUITE=${PIPESTATUS[0]}
    echo "run_suite exit (T5 on): $SUITE"
    if [ "$SUITE" -ne 0 ]; then
        echo "FATAL: suite fails with T5 on. NOT measuring."
        exit 20
    fi

    echo
    echo "######## 2. DID THE RUNS ACTUALLY MERGE IN THE ENGINE? ########"
    echo "# n_runs must read 1 and h2d_launches 2 per layer-step. If the slack"
    echo "# guard refused, n_runs stays at the rider count and T5 is a no-op"
    echo "# however well it tests -- which is exactly how T1 could have failed."
    runpt T5_on_c512  512 16 0,1,2,4,8 --no-rungd-async --whole-extent --rungd-bpr 41
    runpt T5_off_c512 512 16 0,1,2,4,8 --no-rungd-async

    echo
    echo "######## 3. WHAT T5 BOUGHT, bracketed ########"
    run T5_off_a 512 16 0,1,2,4,6,8 --no-rungd-async
    run T5_on    512 16 0,1,2,4,6,8 --no-rungd-async --whole-extent --rungd-bpr 41
    run T5_off_b 512 16 0,1,2,4,6,8 --no-rungd-async

    echo
    echo "######## 4. ctx 2048, where slack is a SMALLER fraction ########"
    echo "# 2048 + 136 = 2184 tokens = 137 blocks at bs 16."
    run T5_c2048_off 2048 16 0,1,2 --no-rungd-async
    run T5_c2048_on  2048 16 0,1,2 --no-rungd-async --whole-extent --rungd-bpr 137
    ;;

*)  echo "unknown SLICE=$SLICE"; exit 2 ;;
esac

echo
echo "========= M12 slice $SLICE complete $(date -Is) ========="
ls -la "$OUT"
