#!/bin/bash
# M7 -- pin the rider-sharing exponent alpha at high concurrency.
#
#   sbatch --export=ALL,SLICE=A m7_amortization.sbatch
#
# WHY. The campaign reported dT = F + c*r with a constant c. That was too
# strong: fitting dT = F + c*r^alpha rejects the affine law at ctx 512
# (F(1,5)=10.28, alpha=0.69, bootstrap P(alpha<1)=98%), the affine residuals
# carry the concavity signature, and it survives dropping r=1. So riders DO
# share some of c. But affine and alpha=0.69 agree to 1.3% at r=8 and are 21.6%
# apart at r=32, so alpha is unconstrained exactly where it matters, and ctx 512
# on PCIe cannot reach there -- its knee is 8.
#
# HOW. The knee is r* = T0*X/(b_tok*S), so it goes as 1/S. Shortening the
# context widens the rider axis on the A100 we already hold. No new hardware:
#
#   ctx    bytes/rider   link ms/rider   knee (T0 ~ 47-49 ms)   measured so far
#   2048     537 MB          21.4              2.2             (2 points)
#    512     134 MB           6.17             8.0             alpha 0.69
#    256      67 MB           3.50            14.0             alpha 0.75
#    192      50 MB           2.01            23               --
#    128      34 MB           1.34            35               --
#
# ctx 128 and 192 are PROBES, not operating points. They exist to give alpha a
# wide enough rider span to separate from affine; nobody serves 128-token
# contexts from a host tier. Labelled as such in the report.
#
# TWO PREDICTIONS, REGISTERED BEFORE THE RUN so this is a test and not a fish:
#
#   P1  If alpha comes from LAUNCH PIPELINING -- 2*L*R back-to-back memcpys
#       queueing better in the copy engine than one alone -- then alpha should
#       FALL with shorter context, because the amortizable share of c (launch +
#       D2H) rises as context shrinks: ~60% at ctx 512, ~80% at ctx 128. So
#       predict alpha(128) < alpha(256) < alpha(512).
#
#   P2  The already-measured alpha(256)=0.75 > alpha(512)=0.69 is the WRONG
#       order for P1. Either P1 is wrong, or that gap is noise -- the two CIs,
#       [0.40,1.16] and [0.40,0.86], overlap heavily. A1/A2 (the same ctx twice,
#       adjacent in time) measures the run-to-run spread of alpha ITSELF, which
#       is what decides between those two readings. Without A2 this whole slice
#       could not tell a real context trend from job-to-job scatter, and F has
#       already been seen to drift 11.67 -> 12.39 over an hour of GPU work.
#
# long-len is 256, not the campaign's 136: at r=32 the admission ramp is roughly
# 2(r-1) ~ 62 steps, and the DIRECT estimator needs a steady window well clear
# of it. The differenced estimator is expected to be contaminated at high r
# (short-len 8 cannot clear a 62-step ramp) and the fits use step_ms_direct.
# Pool check: 48 seqs * (128+256+64) = 21504 tokens against a ~35472-token pool,
# 61% occupancy -- unlike the published ctx-2048 config, which sat at 98.5% and
# tipped into preemption when a sequence lengthened.
#
# repeats 7, not 5: alpha is a shape parameter and needs tighter per-point bars
# than a slope does.
#
# No delay command anywhere: this cluster's submit filter rejects any batch
# script containing one.
#
#SBATCH --job-name=ovh-m7
#SBATCH --partition=supernova
#SBATCH --nodelist=nova07
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m7_amort_%j.out

set -u
SLICE=${SLICE:-A}

TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/07_amortization
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct
REPEATS=7

echo "========= M7 slice $SLICE : $(hostname) $(date -Is) ========="
VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "tree OK: $VLLM_PATH" ;;
  *) echo "FATAL: wrong tree ($VLLM_PATH)"; exit 10 ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# run <tag> <ctx> <batch> <riders>
run () {
    local tag=$1 ctx=$2 batch=$3 riders=$4; shift 4
    echo
    echo "=== $tag  ctx=$ctx batch=$batch riders=$riders $(date -Is)"
    "$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
        --model "$Q32" --ctx "$ctx" --batch "$batch" --riders "$riders" \
        --connector --repeats "$REPEATS" --short-len 8 --long-len 256 \
        --gpu-mem-util 0.90 --link-gbps 25.1 \
        --outdir "$OUT" --out "${tag}.json" "$@" 2>&1 | tail -70
    echo "--- $tag done $(date -Is)"
}

case "$SLICE" in

A)  # ctx 128 TWICE, adjacent in time. A1 vs A2 is the run-to-run spread of
    # alpha itself, which P2 needs. Then 192 and the denser 256.
    run A1_c128 128 16 0,1,2,3,4,6,8,12,16,20,24,28,32
    run A2_c128 128 16 0,1,2,3,4,6,8,12,16,20,24,28,32
    run B_c192  192 16 0,1,2,3,4,6,8,12,16,20
    run C_c256  256 16 0,1,2,3,4,5,6,8,10,12,13
    ;;

B)  # Falsification arm, only if slice A finds alpha < 1 at ctx 128.
    #
    # If the sharing is LAUNCH pipelining then it must survive removing the
    # bytes: the minbytes arm issues the same 2*L*R launches moving one block
    # row each, so launches are intact and the bus is not. alpha should be
    # UNCHANGED. If instead alpha is a hiding/overlap effect that scales with
    # bytes, minbytes should drive alpha to 1.
    #
    # This is the arm that says WHICH component is doing the sharing, and so
    # whether batching the launches would capture it.
    run D_c128_minbytes 128 16 0,1,2,4,8,16,24,32 --xfer-mode minbytes
    run E_c128_nobus    128 16 0,1,2,4,8,16,24,32 --xfer-mode nobus
    run F_c128_skipd2h  128 16 0,1,2,4,8,16,24,32 --skip-d2h
    ;;

*)  echo "unknown SLICE=$SLICE"; exit 2 ;;
esac

echo
echo "========= M7 slice $SLICE complete $(date -Is) ========="
ls -la "$OUT"
