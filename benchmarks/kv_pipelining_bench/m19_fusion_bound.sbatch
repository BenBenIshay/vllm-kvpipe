#!/bin/bash
# M19 -- price the attention fusion BEFORE building it.
#
#   sbatch benchmarks/kv_pipelining_bench/m19_fusion_bound.sbatch
#
# `F` is now the binding constraint (c clears the T0/B budget by 5.5x) and ~56%
# of F at L=64 is the per-layer SECOND attention pass over the rider cohort. The
# only way to remove it is Solution 1 step 3: fuse it into the resident call.
# That is days of work in the attention backend -- a per-layer block-table
# rebuild on the hot path, probably its own KV-cache group -- and the +10 points
# of projected gain assumes fusing REMOVES that cost rather than relocating it.
# Nothing measured says so, and four changes last campaign had a mechanism that
# fired and a benefit that did not follow (T3, T5, T6, T7b).
#
# So: bound the prize first.
#
#   FB_noattend  skips gather + kernel + scatter. UPPER BOUND -- a fused call
#                cannot beat not doing the work at all.
#   FB_nokernel  skips only the flash call, keeping gather and scatter.
#
#   base - noattend            = the whole second pass
#   noattend - nokernel        = the data movement a fused call removes
#   nokernel's own saving      = kernel + launch, of which only launch is
#                                recoverable (the arithmetic has to happen
#                                somewhere even after fusion)
#
# THREE base arms, not one. F's within-job drift has been 0.04 / 0.43 / 0.04 /
# 0.51 ms across four jobs on identical configuration -- bimodal and not
# borrowable between jobs -- so the effect must be compared against drift
# measured HERE. Arms interleave base/ablation/base so drift cannot masquerade as
# the result.
#
# DECISION GATE: build the fusion only if (base - noattend) >= 1.2 ms at L=64,
# i.e. >= 60% of the measured 1.94 ms remainder, AND that saving exceeds this
# job's own base spread by at least 2x. Below that, Solution 1 step 3 is not
# worth the work and the +10 point projection gets retracted rather than shipped.
#
# No delay command anywhere: this cluster's submit filter rejects any batch
# script containing one.
#
#SBATCH --job-name=opt-m19
#SBATCH --partition=supernova
#SBATCH --nodelist=nova07
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=05:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m19_%j.out

set -u
TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/21_fusion_bound
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct
RIDERS=0,1,2,3,4,5,6,8

echo "========= M19 : $(hostname) $(date -Is) ========="
echo "head: $(git rev-parse --short HEAD)   uncommitted: $(git status --porcelain | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The venv is an EDITABLE install, so this guard is the only thing standing
# between us and measuring a different tree than the one we think we are.
VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "tree OK: $VLLM_PATH" ;;
  *) echo "FATAL: wrong tree ($VLLM_PATH)"; exit 10 ;;
esac

# ---------------------------------------------------------------------------
# Correctness gate. The knob is NOT exported -- the suite must see the shipped
# default, and hybrid_rungd_real fails if VLLM_ENABLE_V1_MULTIPROCESSING leaks
# into its server subprocess, so per-test env stays inside run_suite.sh.
# ---------------------------------------------------------------------------
bash benchmarks/kv_pipelining_bench/run_suite.sh 2>&1 | tail -18
SUITE=${PIPESTATUS[0]}
echo "run_suite exit: $SUITE"
if [ "$SUITE" -ne 0 ]; then
    echo "FATAL: suite failed. NOT measuring -- fix the knob or revert it."
    exit 20
fi

# ---------------------------------------------------------------------------
# Arms. Config IDENTICAL to M13/M14/M15/M17/M18 so F and c are comparable to
# the whole campaign, not just to each other.
# ---------------------------------------------------------------------------
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
    # Prove from the artifact which arm this actually was, rather than trusting
    # the filename: the ablation counters only fire when the knob took effect.
    echo "    attend_mode in artifact: $(grep -o '"attend_mode": "[a-z]*"' "$OUT/${tag}.json" 2>/dev/null | tail -1)"
    echo "    numerically_valid:       $(grep -o '"numerically_valid": [a-z]*' "$OUT/${tag}.json" 2>/dev/null | tail -1)"
    echo "    attend ablated:          $(grep -c 'rider_attend_ablated' "$OUT/${tag}_phases.jsonl" 2>/dev/null)"
    echo "    kernel ablated:          $(grep -c 'rider_kernel_ablated' "$OUT/${tag}_phases.jsonl" 2>/dev/null)"
}

run FB_base_a
run FB_noattend  --attend-mode noattend
run FB_base_b
run FB_nokernel  --attend-mode nokernel
run FB_base_c

# ---------------------------------------------------------------------------
# E3 -- locate the knee empirically, which is what the context sawtooth rests
# on. LAST on purpose: if the 5h budget runs short, only this is lost.
#
# NOT comparable to the arms above: different ctx and batch. Qwen-32B fp16 is
# 65.5 GB, so at 0.90 util only ~6.5 GB is left for KV and batch 16 does not fit
# at these contexts -- hence --batch 8. Predicted knees on A100/PCIe (T0 ~45 ms,
# 0.0104 ms per context token): ctx 1080 -> ~4 riders, ctx 1440 -> ~3. The test
# is whether the measured departure from the affine law lands where
# `T0 / link_ms` says, since the integer floor of that is what sets the rider
# count the operating-point recommendation turns on.
# ---------------------------------------------------------------------------
knee_run () {
    local tag=$1 ctx=$2
    echo
    echo "=== $tag  ctx=$ctx $(date -Is)"
    "$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
        --model "$Q32" --ctx "$ctx" --batch 8 --riders "$RIDERS" \
        --connector --repeats 3 --short-len 8 --long-len 48 \
        --gpu-mem-util 0.90 --link-gbps 25.1 --k-window 4 \
        --outdir "$OUT" --out "${tag}.json" 2>&1 | tail -14
    echo "--- $tag done $(date -Is)"
    echo "    fit: $(grep -o '"knee_riders": [0-9.]*' "$OUT/${tag}.json" 2>/dev/null | tail -1)"
}

knee_run E3_ctx1080 1080
knee_run E3_ctx1440 1440

echo
echo "========= M19 complete $(date -Is) ========="
ls -la "$OUT"
