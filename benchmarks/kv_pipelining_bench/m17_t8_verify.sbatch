#!/bin/bash
# M16 -- T7a with the guard fixed: does the vectorised scatter fire, and buy?
#
#   sbatch --export=ALL,SLICE=A m16_t7a_fixed.sbatch
#
# WHY A SECOND ATTEMPT. M15 found T7a INERT: 2352 flushes, every one on the
# per-layer path, zero vectorised, in all four arms. The guard required the
# pending layer indices to equal list(range(nl)), but the pending list is built
# in EXECUTION order while `_wt_k` is indexed by the layer index the runner
# assigns from kv_cache_groups -- so any difference between those orders makes
# `lis` a permutation and the equality fails, even though every layer is present
# (the row counts were 64 x n_active exactly). The fix sorts by layer index,
# which costs nothing for 64 entries and makes the guard indifferent to
# execution order.
#
# So M15 measured T7a's cost (none) rather than its effect. This measures the
# effect.
#
# CORRECTNESS IS NOT OPTIONAL HERE, and this is the important difference from
# M15. In M15 the vectorised path never executed, so the suite passing said
# nothing about it. With the guard fixed it becomes LIVE code in the engine for
# the first time: a wrong global row would write a rider's token into another
# LAYER's mirror, with every shape valid and nothing raised. The off-GPU test
# (unified_mirror_equiv_test.py, 8 cases) checks it bitwise but cannot see
# stream ordering or the real index values. So the suite runs first and gates
# everything.
#
# AND THE MECHANISM CHECK STAYS. wt_flush_vectorised must be non-zero with
# wt_flush_per_layer at zero; the new wt_fallback_* counters say which condition
# failed if it still does not fire.
#
# WHAT TO EXPECT. The scatter is 0.018 ms/layer of measured cost (M14), so the
# ceiling here is ~1.2 ms at L=64 and ~1.4 at L=80. Against a within-job drift
# that ran 0.04 in M13 and 0.43 in M15 on identical configuration, the bracket
# is repeated THREE times rather than twice -- a 1.2 ms effect needs a drift
# figure that is itself trustworthy, and borrowing M13's 0.04 would be exactly
# the mistake of assuming a drift instead of measuring it.
#
# No delay command anywhere: this cluster's submit filter rejects any batch
# script containing one.
#
#SBATCH --job-name=opt-m17
#SBATCH --partition=supernova
#SBATCH --nodelist=nova07
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m17_%j.out

set -u
SLICE=${SLICE:-A}

TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/19_t8
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct
RIDERS=0,1,2,3,4,5,6,8

echo "========= M16 slice $SLICE : $(hostname) $(date -Is) ========="
echo "head: $(git rev-parse --short HEAD)   uncommitted: $(git status --porcelain | wc -l)"
VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "tree OK: $VLLM_PATH" ;;
  *) echo "FATAL: wrong tree ($VLLM_PATH)"; exit 10 ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run () {
    local tag=$1; shift
    echo
    echo "=== $tag  $* $(date -Is)"
    VLLM_KVPIPE_PHASE_TIMERS=1 \
    VLLM_KVPIPE_PHASE_TIMERS_OUT="$OUT/${tag}_phases.jsonl" \
    env "$@" "$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
        --model "$Q32" --ctx 128 --batch 16 --riders "$RIDERS" \
        --connector --repeats 5 --short-len 8 --long-len 48 \
        --gpu-mem-util 0.90 --link-gbps 25.1 --k-window 4 \
        --outdir "$OUT" --out "${tag}.json" 2>&1 | tail -14
    echo "--- $tag done $(date -Is)"
    echo "    vectorised: $(grep -c 'wt_flush_vectorised' "$OUT/${tag}_phases.jsonl" 2>/dev/null)"
    echo "    per-layer : $(grep -c 'wt_flush_per_layer' "$OUT/${tag}_phases.jsonl" 2>/dev/null)"
    for r in not_unified not_uniform single_layer layer_gap; do
      c=$(grep -c "wt_fallback_$r" "$OUT/${tag}_phases.jsonl" 2>/dev/null)
      [ "${c:-0}" != "0" ] && echo "    fallback reason $r: $c"
    done
}

case "$SLICE" in
A)
    echo
    echo "######## 1. CORRECTNESS -- the vectorised path is LIVE for the first"
    echo "########    time, so this gates everything that follows."
    bash benchmarks/kv_pipelining_bench/run_suite.sh 2>&1 | tail -16
    SUITE=${PIPESTATUS[0]}
    echo "run_suite exit: $SUITE"
    if [ "$SUITE" -ne 0 ]; then
        echo "FATAL: suite failed. NOT measuring -- fix or revert T7a."
        exit 20
    fi

    echo
    echo "######## 2. DID IT FIRE, AND WHAT DID IT BUY -- three ON arms, because"
    echo "########    the drift itself has to be measured, not borrowed."
    run T8_on_a
    run T8_off  VLLM_KVPIPE_RUNGD_UNIFIED_MIRROR=0
    run T8_on_b
    run T8_off2 VLLM_KVPIPE_RUNGD_UNIFIED_MIRROR=0
    run T8_on_c
    ;;
*)  echo "unknown SLICE=$SLICE"; exit 2 ;;
esac

echo
echo "========= M16 slice $SLICE complete $(date -Is) ========="
ls -la "$OUT"
