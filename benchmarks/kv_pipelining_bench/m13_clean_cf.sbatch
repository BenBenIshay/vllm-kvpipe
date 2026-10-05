#!/bin/bash
# M13 -- fit c and F on THE CURRENT SOLUTION, wholly below the knee.
#
#   sbatch --export=ALL,SLICE=A m13_clean_cf.sbatch
#
# WHY. Every c and F in the report was fitted across a rider sweep whose top
# sits AT the knee, so the knee's cost got smeared backwards into c. The re-fit
# that corrected it used ARCHIVED pre-T1 arms and then INFERRED the current
# value by subtracting the per-block term. That inference is the confusing part
# and it is unnecessary: measure the current code directly.
#
# "Current solution" = exactly the defaults, nothing exotic:
#   T1 block-table rebuild        IN  (not switchable)
#   T2 deferred mirror write      IN  (VLLM_KVPIPE_RUNGD_D2H_ASYNC default 1)
#   async side-stream prefetch    ON  (default; load-bearing -- c is 0.9 with
#                                      it and 7.2 without)
#   T3 batched gather             OFF (default; 6x regression)
#   T5 whole-extent runs          OFF (default; measured null)
#   T6 mirror ordering            OFF (default; costs the whole overlap)
# Nothing is set in this script that changes the feature's behaviour.
#
# STAYING BELOW THE KNEE IS THE WHOLE POINT, and it has to account for
# GENERATION: a rider's sequence grows from `ctx` to `ctx + long_len`, so the
# bytes it streams grow with it. Sizing the sweep on the PROMPT length alone is
# how a run that looks below the knee ends up at it. With long_len 48 and a
# 49.3 ms step on Qwen-32B/A100:
#
#     ctx   final tokens   xfer/rider   knee   riders at load <= 1/3
#     128            176      1.84 ms   26.8                      8
#     256            304      3.18 ms   15.5                      5
#     512            648      6.77 ms    7.3                      2
#
# so ctx 128 gives 8 usable rider points and ctx 256 gives 5 -- enough for a
# conditioned fit, which truncating a ctx-512 sweep to r<=2 was not.
#
# ARMS
#   1  ctx 128, riders 0..8   -- the primary fit, 8 points, max load 0.30
#   2  ctx 256, riders 0..5   -- does c depend on CONTEXT? It should not: below
#                               the knee the bytes are hidden, so a c that
#                               scales with S would mean they are not.
#   3  repeat of arm 1        -- the drift bracket. Without it no effect can be
#                               called resolvable (three claims were retracted
#                               today for want of one).
#   4  ctx 128, L=16          -- the L slope on current code, so projecting to
#   5  ctx 128, L=32             an 80-layer model does not rest on the archived
#                               pre-T1 arms. Uses the layer-subset knob, which
#                               is numerically wrong by construction and used
#                               for TIMING ONLY, double-gated, excluded from the
#                               suite -- the mitigation the campaign plan set.
#                               At L=16 the bytes are a quarter, so these arms
#                               are even further below the knee.
#
# No delay command anywhere: this cluster's submit filter rejects any batch
# script containing one.
#
#SBATCH --job-name=opt-m13
#SBATCH --partition=supernova
#SBATCH --nodelist=nova06
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m13_%j.out

set -u
SLICE=${SLICE:-A}

TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/15_clean_cf
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct
REPEATS=5
GEN=48

echo "========= M13 slice $SLICE : $(hostname) $(date -Is) ========="
echo "head: $(git rev-parse --short HEAD)   uncommitted: $(git status --porcelain | wc -l)"
VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "tree OK: $VLLM_PATH" ;;
  *) echo "FATAL: wrong tree ($VLLM_PATH)"; exit 10 ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run () {
    local tag=$1 ctx=$2 riders=$3; shift 3
    echo
    echo "=== $tag  ctx=$ctx riders=$riders gen=$GEN $* $(date -Is)"
    VLLM_KVPIPE_PHASE_TIMERS=1 \
    VLLM_KVPIPE_PHASE_TIMERS_OUT="$OUT/${tag}_phases.jsonl" \
    "$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
        --model "$Q32" --ctx "$ctx" --batch 16 --riders "$riders" \
        --connector --repeats "$REPEATS" --short-len 8 --long-len "$GEN" \
        --gpu-mem-util 0.90 --link-gbps 25.1 --k-window 4 \
        --outdir "$OUT" --out "${tag}.json" "$@" 2>&1 | tail -24
    echo "--- $tag done $(date -Is)"
}

case "$SLICE" in
A)
    echo
    echo "######## 1. PRIMARY FIT: ctx 128, 8 rider points, all below the knee"
    run CF_c128_a 128 0,1,2,3,4,5,6,8

    echo
    echo "######## 2. CONTEXT CHECK: does c scale with S? It should NOT."
    run CF_c256   256 0,1,2,3,4,5

    echo
    echo "######## 3. DRIFT BRACKET: repeat of arm 1"
    run CF_c128_b 128 0,1,2,3,4,5,6,8

    echo
    echo "######## 4-5. LAYER SLOPE on current code (timing-only knob)"
    run CF_L16    128 0,1,2,3,4,5,6,8 --rungd-layers 16
    run CF_L32    128 0,1,2,3,4,5,6,8 --rungd-layers 32
    ;;
*)  echo "unknown SLICE=$SLICE"; exit 2 ;;
esac

echo
echo "========= M13 slice $SLICE complete $(date -Is) ========="
ls -la "$OUT"
