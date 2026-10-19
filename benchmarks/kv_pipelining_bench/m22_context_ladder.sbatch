#!/bin/bash
# M22 -- F and c across context up to the paper's target, with the addressing
# ablation on the same binary, plus the k-window question. Ben's review of
# 2026-10-19:
#   (3) "a graph that shows F is static and c depends on the context size ...
#       the per-step table still depends on the rider blocks but affects
#       significantly less -- prove it with a benchmark"
#   (1) "isn't k = 2 the best one for us? did we benchmark this?"
#
#   sbatch benchmarks/kv_pipelining_bench/m22_context_ladder.sbatch
#
# WHY THESE CHOICES.
#   * BYTES REMOVED (--xfer-mode minbytes), as in M21: every launch, gather and
#     table entry is issued but ~no bytes move, so every point is below the knee
#     at every context and c and F can be fitted at 14724 tokens as at 512.
#   * ONE RESIDENT, fixed on every rung. Qwen-32B fp16 leaves ~6.5 GB of KV; one
#     request at 14724 tokens is 3.9 GB. M21 used 4 residents, which capped it at
#     4096. The resident cohort does not move F (report: 4x cohort -> 0.2%).
#   * RIDERS 0..5 at every context, so every fit uses the same points. Six riders
#     at 14724 keep the k=4 rolling slots near 1.2 GB.
#   * ARMS ours vs PERBLOCK_BT (the pre-T1 per-block table build, same binary),
#     interleaved across contexts so machine drift cannot read as context.
#   * The k sweep uses REAL bytes at ctx 128, 16 residents, where M13 kept every
#     point below the knee; k = 2, 4, 8 interleaved, two repeats.
#   * Phase-timer arms (Class A host timer on build_step_bt_python) are separate
#     because the timers perturb step time.
#
#SBATCH --job-name=opt-m22
#SBATCH --partition=supernova
#SBATCH --nodelist=nova07
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=14:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20261019/m22/m22_%j.out

set -u
TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20261019/m22
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct

echo "========= M22 : $(hostname) $(date -Is) ========="
echo "head: $(git rev-parse --short HEAD)   uncommitted: $(git status --porcelain -- vllm benchmarks | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "tree OK: $VLLM_PATH" ;;
  *) echo "FATAL: wrong tree ($VLLM_PATH)"; exit 10 ;;
esac

# run <tag> <ctx> <batch> <riders> [flags...]
run () {
    local tag=$1 ctx=$2 batch=$3 riders=$4; shift 4
    echo
    echo "=== $tag  ctx=$ctx batch=$batch riders=$riders $* $(date -Is)"
    "$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
        --model "$Q32" --ctx "$ctx" --batch "$batch" --riders "$riders" \
        --connector --repeats 3 --short-len 8 --long-len 48 \
        --gpu-mem-util 0.90 --link-gbps 25.1 \
        "$@" \
        --outdir "$OUT" --out "${tag}.json" 2>&1 | tail -6
    echo "--- $tag done $(date -Is)"
    echo "    valid: $(grep -o '"numerically_valid": [a-z]*' "$OUT/${tag}.json" 2>/dev/null | head -1)"
}

LADDER_R=0,1,2,3,4,5

# 1. THE LADDER: ours vs per-block table, bytes removed, 1 resident.
for rep in a b; do
  for ctx in 512 2048 4096 8192 14724; do
    run "LD_ours_c${ctx}_$rep" "$ctx" 1 "$LADDER_R" --k-window 4 --xfer-mode minbytes
    VLLM_KVPIPE_RUNGD_PERBLOCK_BT=1 \
    run "LD_pblk_c${ctx}_$rep" "$ctx" 1 "$LADDER_R" --k-window 4 --xfer-mode minbytes
  done
done

# 2. THE k WINDOW: real bytes, below the knee.
for rep in a b; do
  for k in 2 4 8; do
    run "KW_k${k}_$rep" 128 16 0,1,2,3,4,5,6 --k-window "$k"
  done
done

# 3. HOST TIMER on the table build, ours vs per-block, three contexts.
for ctx in 512 4096 14724; do
  VLLM_KVPIPE_PHASE_TIMERS=1 VLLM_KVPIPE_PHASE_TIMERS_OUT="$OUT/PT_ours_c${ctx}_phases.jsonl" \
  run "PT_ours_c${ctx}" "$ctx" 1 "$LADDER_R" --k-window 4 --xfer-mode minbytes
  VLLM_KVPIPE_RUNGD_PERBLOCK_BT=1 \
  VLLM_KVPIPE_PHASE_TIMERS=1 VLLM_KVPIPE_PHASE_TIMERS_OUT="$OUT/PT_pblk_c${ctx}_phases.jsonl" \
  run "PT_pblk_c${ctx}" "$ctx" 1 "$LADDER_R" --k-window 4 --xfer-mode minbytes
done

echo
echo "========= M22 complete $(date -Is) ========="
ls -la "$OUT"
