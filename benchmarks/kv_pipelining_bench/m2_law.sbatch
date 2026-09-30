#!/bin/bash
# M2 -- the factorial that identifies the overhead law.
#
# Submit one SLICE at a time:  sbatch --export=ALL,SLICE=A m2_law.sbatch
# Slices exist because the cluster has one free A100 node, so this is serial
# anyway, and because a slice that fails should not take the others down.
#
#   A  rider x context     the core axis. Separates A (switch-on) from B
#                          (per-rider) and, across S, B from C (the transfer).
#   B  block size          bs {16,32} at FIXED ctx and rider count. The ONLY
#                          way to tell a per-BLOCK cost from a per-BYTE one:
#                          halving bs doubles the block count while bytes stay
#                          put. Per-block is host work a faster link never
#                          touches; per-byte shrinks with bandwidth.
#   C  layer depth         rungd_layers {8,16,32,48,64} inside ONE model, so the
#                          per-layer slope is measured at fixed head geometry.
#                          The model charges F strictly proportional to depth on
#                          the evidence of a single trace. NUMERICALLY INVALID
#                          arm -- timing only.
#   D  resident cohort     batch {8,16,32}. Do residents still pay for the
#                          riders' existence now that flash_attn.py:1034
#                          bypasses the two-pass split under the pool override?
#   E  other models        Llama-8B (32L) and Qwen-0.5B (24L): external validity
#                          for the layer slope, confounded with head geometry,
#                          which is why slice C exists as well.
#   F  k_window            the slots that "batching across layers" would use.
#
# Every arm runs --connector (ground rule 1) and behind the per-step
# decode-purity gate; rows that fail it carry usable=false and the fitter drops
# them. Rider counts are chosen BELOW the knee, r* = T0*X/(b_tok*S): validated
# at 2.50 against the paper's measured 2.4 at ctx 2048, giving ~8.8 at ctx 512
# and ~17.2 at ctx 256 for Qwen-32B.
#
# No delay command anywhere: this cluster's submit filter rejects any batch
# script containing one.
#
#SBATCH --job-name=ovh-m2
#SBATCH --partition=supernova
#SBATCH --nodelist=nova07
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m2_law_%j.out

set -u
SLICE=${SLICE:-A}

TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/02_law
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct
L8=meta-llama/Llama-3.1-8B-Instruct
Q05=Qwen/Qwen2.5-0.5B-Instruct
REPEATS=5

echo "========= M2 slice $SLICE : $(hostname) $(date -Is) ========="
VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "tree OK" ;;
  *) echo "FATAL: wrong tree ($VLLM_PATH)"; exit 10 ;;
esac

# run <tag> <model> <ctx> <batch> <riders> [extra flags...]
run () {
    local tag=$1 model=$2 ctx=$3 batch=$4 riders=$5; shift 5
    echo
    echo "=== $tag  model=$(basename "$model") ctx=$ctx batch=$batch riders=$riders $*"
    "$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
        --model "$model" --ctx "$ctx" --batch "$batch" --riders "$riders" \
        --connector --repeats "$REPEATS" --short-len 8 --long-len 136 \
        --gpu-mem-util 0.90 --link-gbps 25.1 \
        --outdir "$OUT" --out "${tag}.json" "$@" 2>&1 | tail -42
}

case "$SLICE" in

A)  # rider x context, Qwen-32B. Rider lists stop below each context's knee.
    run A_c256  "$Q32"  256 16 0,1,2,4,6,8,10,12,14,16
    run A_c512  "$Q32"  512 16 0,1,2,3,4,5,6,7,8
    run A_c1024 "$Q32" 1024 16 0,1,2,3,4
    run A_c2048 "$Q32" 2048 16 0,1,2
    ;;

B)  # per-block vs per-byte. Same ctx and riders, block size doubled.
    run B_bs16_c512 "$Q32" 512 16 0,1,2,4,6,8 --block-size 16
    run B_bs32_c512 "$Q32" 512 16 0,1,2,4,6,8 --block-size 32
    run B_bs16_c256 "$Q32" 256 16 0,2,4,8,12  --block-size 16
    run B_bs32_c256 "$Q32" 256 16 0,2,4,8,12  --block-size 32
    ;;

C)  # layer slope inside one model. NUMERICALLY INVALID arms, timing only.
    for NL in 8 16 32 48; do
        run "C_L${NL}_c512" "$Q32" 512 16 0,1,2,4,6,8 --rungd-layers "$NL"
    done
    # L=64 is slice A's A_c512 (the unrestricted run) -- do not repeat it.
    ;;

D)  # does the resident cohort pay for the riders' existence?
    run D_b8_c512  "$Q32" 512  8 0,1,2,4,6,8
    run D_b32_c512 "$Q32" 512 32 0,1,2,4,6,8
    # b=16 is slice A's A_c512.
    ;;

E)  # external validity across real models (head geometry differs too).
    run E_l8_c512  "$L8"  512 16 0,1,2,3,4
    run E_l8_c256  "$L8"  256 16 0,1,2,4,6,8
    run E_q05_c512 "$Q05" 512 16 0,1,2,4,6,8,12,16
    ;;

F)  # rolling slots: the mechanism "batching across layers" would use.
    run F_k2_c512 "$Q32" 512 16 0,1,2,4,6,8 --k-window 2
    run F_k8_c512 "$Q32" 512 16 0,1,2,4,6,8 --k-window 8
    # k=4 is the engine default, i.e. slice A's A_c512.
    ;;

*)  echo "FATAL: unknown SLICE=$SLICE (expected A..F)"; exit 2 ;;
esac

# ---- refit over everything gathered so far ---------------------------------
echo
echo "=== fit over all real arms collected to date"
"$PY" paper/model/overhead_fit.py "$OUT"/*.json \
    --link-gbps 25.1 --out "$OUT/fit_latest.json" 2>&1 | tail -50

echo
# STRAY-PROCESS CHECK, not a teardown call: bench_teardown.terminate_engine
# takes a subprocess handle and is for the SERVER-launching harnesses, while
# rider_cost_curve runs the engine in-process and dies with the script. What
# matters is that nothing outlived the job -- orphaned EngineCore processes
# drained two nodes on 2026-09-06 -- and that is checked by polling processes,
# not nvidia-smi, which reads 0 between engines and is not a lock.
echo "--- stray processes (expect none)"
ps -u "$USER" -o args= | grep -E "[.]venv/bin/python.*kv_pipelining_bench|^VLLM::EngineCore" | head -5
echo "========= M2 slice $SLICE done $(date -Is) ========="
