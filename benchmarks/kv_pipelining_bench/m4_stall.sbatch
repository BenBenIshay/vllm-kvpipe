#!/bin/bash
# M4 -- where does the law stop holding, and what is the >60x stall?
#
# THE TARGET. byte_volume_stall.py records a >60x gap between predicted and
# observed step time at 20 riders x ctx 1024 on A100: 5.37 GB/step, predicted
# ~216 ms, did not finish in 900 s at ~0% GPU utilisation. Nothing in any model
# captures it, and C2C's whole promise is admitting many more riders -- so a
# superlinear blowup at 20 would erase the projected gain.
#
# WHAT THIS CAMPAIGN HAS ALREADY ESTABLISHED about it. The fitted E_superlinear
# term over 105 points came out NEGATIVE and significant (-0.00027, CI excluding
# zero), i.e. the law is very slightly CONCAVE in r below the knee, about -5% at
# r=8. So the stall is not the tail of the law we measured; it is a distinct
# mechanism that switches on somewhere above the knee. This job finds where.
#
# DESIGN. Walk r from below the knee to well past it at two contexts, and record
# r_max -- the rider count beyond which the affine law stops predicting. The
# knees here: ctx 512 -> 8.0, ctx 1024 -> 4.4. So the ctx-1024 sweep crosses its
# knee at r=4 and reaches 4.5x past it at r=20, which is the documented stall
# point exactly.
#
# EVERY RUN IS WRAPPED IN `timeout`. The documented failure mode is a hang, not a
# crash: 900 s at ~0% GPU. Without a bound one arm would eat the allocation and
# the rest would never run. A killed arm is itself the measurement -- it tells us
# r_max is at or below that point -- so the timeout is recorded, not treated as
# an error.
#
# No delay command anywhere: this cluster's submit filter rejects any batch
# script containing one.
#
#SBATCH --job-name=ovh-m4
#SBATCH --partition=supernova
#SBATCH --nodelist=nova07
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m4_stall_%j.out

set -u
TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/04_stall
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface
Q32=Qwen/Qwen2.5-32B-Instruct

echo "========= M4 stall: $(hostname) $(date -Is) ========="
VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "tree OK" ;;
  *) echo "FATAL: wrong tree ($VLLM_PATH)"; exit 10 ;;
esac

# run <tag> <ctx> <riders> <timeout_s>
run () {
    local tag=$1 ctx=$2 riders=$3 tmo=$4
    echo
    echo "=== $tag  ctx=$ctx riders=$riders  timeout=${tmo}s"
    timeout --signal=KILL "$tmo" \
      "$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
        --model "$Q32" --ctx "$ctx" --batch 16 --riders "$riders" \
        --connector --repeats 3 --short-len 24 --long-len 136 \
        --gpu-mem-util 0.90 --link-gbps 25.1 \
        --outdir "$OUT" --out "${tag}.json" > "$OUT/${tag}.log" 2>&1
    local rc=$?
    if [ "$rc" -eq 137 ]; then
        echo "  KILLED at ${tmo}s -- this IS the measurement: r_max is at or"
        echo "  below the highest rider count this arm reached. Last lines:"
        grep -E "r= *[0-9]+ rep|PURITY" "$OUT/${tag}.log" | tail -6
    else
        echo "  exit=$rc"
        grep -E "^ *[0-9]+ +[0-9]+\.[0-9]+ +[-0-9]|KNEE|overhead \(median\)" \
            "$OUT/${tag}.log" | tail -14
    fi
}

# ---- ctx 512, knee 8.0: cross it and keep going -----------------------------
run s512_lo  512 0,2,4,6,8          900
run s512_hi  512 0,10,12,16         1800
run s512_xhi 512 0,20,24            2400

# ---- ctx 1024, knee 4.4: the documented stall point is r=20 -----------------
run s1024_lo  1024 0,2,4            900
run s1024_mid 1024 0,6,8,12         1800
run s1024_hi  1024 0,16,20          2400

echo
echo "--- stray processes (expect none)"
ps -u "$USER" -o args= | grep -E "[.]venv/bin/python.*kv_pipelining_bench|^VLLM::EngineCore" | head -5
echo "========= M4 done $(date -Is) ========="
