#!/bin/bash
# M18 -- re-split F now that T8 and T7a have removed 42% of it.
#
#   sbatch --export=ALL,SLICE=A m18_f_split_post_t8.sbatch
#
# WHY RE-MEASURE. M14 split F at L=64 into mirror write 3.95 (68%) and a
# remainder of 1.85 that survives with both mechanisms off. That split was taken
# BEFORE T8, so the 3.95 included the waste T8 removed: a disabled feature
# (writeback_layer, the swap emulation) forcing a flush on every layer, each
# with one pending entry and each synchronising the D2H event. With F now 3.28,
# the mirror write cannot still be 3.95, and the ranking of what to do next
# depends on the new split rather than the old one.
#
# Inferring it would be easy and wrong: 3.28 - 1.85 = 1.43 assumes the remainder
# is untouched, which is plausible (T8 and T7a both act on the mirror write) but
# is exactly the kind of assumption this campaign has been punished for. Two
# arms measure it instead.
#
#   FS_base      as shipped                     -> F (expect ~3.3)
#   FS_nod2h     SKIP_D2H=1, mirror write off   -> F(base) - F(nod2h) = mirror
#   FS_both      SKIP_D2H=1 + nobus             -> the remainder
#
# Same configuration as M13/M14/M17 so the numbers are comparable arm-for-arm:
# ctx 128, long_len 48, riders 0..8, every point below the knee at load <= 0.28.
#
# The ablations are numerically invalid by construction, timing only, and
# announce themselves -- the standing rule for the measurement knobs.
#
# No delay command anywhere: this cluster's submit filter rejects any batch
# script containing one.
#
#SBATCH --job-name=opt-m18
#SBATCH --partition=supernova
#SBATCH --nodelist=nova07
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m18_%j.out

set -u
TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/20_f_split_post_t8
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct
RIDERS=0,1,2,3,4,5,6,8

echo "========= M18 : $(hostname) $(date -Is) ========="
echo "head: $(git rev-parse --short HEAD)   uncommitted: $(git status --porcelain | wc -l)"
VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "tree OK: $VLLM_PATH" ;;
  *) echo "FATAL: wrong tree ($VLLM_PATH)"; exit 10 ;;
esac

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
        --outdir "$OUT" --out "${tag}.json" 2>&1 | tail -12
    echo "--- $tag done $(date -Is)"
    echo "    vectorised: $(grep -c 'wt_flush_vectorised' "$OUT/${tag}_phases.jsonl" 2>/dev/null)"
    echo "    flushes/step (scatter calls): $(grep -oE '"wt_flush_scatter":\{"calls":[0-9]+' "$OUT/${tag}_phases.jsonl" 2>/dev/null | tail -1)"
}

run FS_base
run FS_nod2h VLLM_KVPIPE_RUNGD_SKIP_D2H=1
run FS_both  VLLM_KVPIPE_RUNGD_SKIP_D2H=1 VLLM_KVPIPE_RUNGD_XFER_MODE=nobus
run FS_base_b

echo
echo "========= M18 complete $(date -Is) ========="
ls -la "$OUT"
