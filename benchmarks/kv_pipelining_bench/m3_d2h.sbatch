#!/bin/bash
# M3 -- is F mostly one blocking copy?
#
# WHY THIS BECAME THE MAIN EVENT. Slice E measured F per layer across three
# models and found it is NOT model-independent: 0.419 ms/layer on Qwen-0.5B,
# 0.246-0.263 on Llama-8B, 0.183 on Qwen-32B, falling as GPU work per layer
# rises. That is an overlap signature -- a roughly fixed amount of host work per
# layer, more of which hides when the layer has more compute to hide behind.
# Taking the 0.5B as the no-hiding limit puts the host work at ~0.42 ms/layer,
# which is ~27 ms per step at 64 layers against a 49 ms step.
#
# And the phase timers put write_token's two PAGEABLE (hence blocking) D2H copies
# at 0.335 ms/layer -- about 80% of that 0.42. If that single sync is most of F,
# then the dominant cost in the whole system is one pageable copy per layer, and
# removing it attacks F directly instead of nibbling at c.
#
# The phase timers cannot settle it: host wall time around a blocking call
# measures queue depth, not cost, and they reproduced the repo's documented 21 ms
# artifact to the millisecond. So this prices it the way the bus copies were
# priced -- by DIFFERENCING a run with the host-mirror write against one without.
#
# Arms, all adjacent in ONE job because F carries a 14% BETWEEN-job offset
# (measured: one job read 1.2 ms high on every arm it contained):
#
#   real              baseline
#   skip_d2h          no host-mirror write -> prices the blocking D2H
#   skip_d2h+nobus    neither D2H nor H2D  -> what is left when no bus traffic
#                                             crosses in either direction
#
# Interleaved real arms bracket the others so drift is visible rather than
# assumed. Three models, because the exposed fraction differs by model and that
# is the whole point.
#
# No delay command anywhere: this cluster's submit filter rejects any batch
# script containing one.
#
#SBATCH --job-name=ovh-m3
#SBATCH --partition=supernova
#SBATCH --nodelist=nova07
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m3_d2h_%j.out

set -u
TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/03_d2h
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

Q32=Qwen/Qwen2.5-32B-Instruct
L8=meta-llama/Llama-3.1-8B-Instruct
Q05=Qwen/Qwen2.5-0.5B-Instruct

echo "========= M3 D2H: $(hostname) $(date -Is) ========="
VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "tree OK" ;;
  *) echo "FATAL: wrong tree ($VLLM_PATH)"; exit 10 ;;
esac

run () {   # run <tag> <model> <ctx> <riders> [extra...]
    local tag=$1 model=$2 ctx=$3 riders=$4; shift 4
    echo
    echo "=== $tag  $(basename "$model") ctx=$ctx riders=$riders $*"
    "$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
        --model "$model" --ctx "$ctx" --batch 16 --riders "$riders" \
        --connector --repeats 5 --short-len 24 --long-len 136 \
        --gpu-mem-util 0.90 --link-gbps 25.1 \
        --outdir "$OUT" --out "${tag}.json" "$@" 2>&1 | tail -30
}

# ---- Qwen-32B: the model the paper's numbers come from --------------------
run q32_real_a  "$Q32" 512 0,1,2,4,6,8
run q32_skipd2h "$Q32" 512 0,1,2,4,6,8 --skip-d2h
run q32_nobus_both "$Q32" 512 0,1,2,4,6,8 --skip-d2h --xfer-mode nobus
run q32_real_b  "$Q32" 512 0,1,2,4,6,8          # bracket, for drift

# ---- Qwen-0.5B: the no-hiding limit, where F/layer was highest ------------
run q05_real_a  "$Q05" 512 0,1,2,4,6,8,12,16
run q05_skipd2h "$Q05" 512 0,1,2,4,6,8,12,16 --skip-d2h
run q05_real_b  "$Q05" 512 0,1,2,4,6,8,12,16    # bracket

# ---- Llama-8B: the middle of the hiding range -----------------------------
run l8_real_a   "$L8"  512 0,1,2,3,4
run l8_skipd2h  "$L8"  512 0,1,2,3,4 --skip-d2h

echo
echo "--- stray processes (expect none)"
ps -u "$USER" -o args= | grep -E "[.]venv/bin/python.*kv_pipelining_bench|^VLLM::EngineCore" | head -5
echo "========= M3 D2H done $(date -Is) ========="
