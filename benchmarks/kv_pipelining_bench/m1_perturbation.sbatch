#!/bin/bash
# M1 -- decide, in writing, whether the phase timers may be used for absolute
# values, and take the direct byte/launch decomposition from the xfer arms.
#
# Ground rule 2: timers must be proven harmless or their effect carried
# explicitly. The reference arm for "timers off" is NOT a separate run of older
# code -- M0 already ran this same instrumented tree with timers off and had to
# reproduce the published, uninstrumented curve within 5%. That gate IS the
# flags-off regression test, so this job only has to measure what turning the
# timers ON does.
#
# Run below the knee. r* = T0*X/(k*S), so ctx 512 puts the knee near 8.7 riders
# for Qwen-32B and ctx 256 near 17 -- which is what makes a >=10-point rider
# axis reachable on PCIe at all. Above the knee the step IS the transfer and
# there is nothing to attribute.
#
#SBATCH --job-name=ovh-m1
#SBATCH --partition=supernova
#SBATCH --nodelist=nova07
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/beegfs/benishay/paper_runs/20260930/overhead/m1_perturbation.%j.out

set -u

TREE=/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char
PY="$TREE/.venv/bin/python"
OUT=/mnt/beegfs/benishay/paper_runs/20260930/overhead/01_perturbation
mkdir -p "$OUT"
cd "$TREE" || exit 1
export HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface

MODEL=Qwen/Qwen2.5-32B-Instruct
BATCH=16
REPEATS=5          # 5, not 3: this job's output is a CI, not a point estimate

echo "================ M1 perturbation: $(hostname) $(date -Is) ==========="
VLLM_PATH=$("$PY" -c "import vllm; print(vllm.__file__)" 2>&1)
case "$VLLM_PATH" in
  "$TREE"/vllm/*) echo "tree OK: $VLLM_PATH" ;;
  *) echo "FATAL: wrong tree ($VLLM_PATH)"; exit 10 ;;
esac

run () {  # run <tag> <ctx> <riders> <timers> <xfer>
    local tag=$1 ctx=$2 riders=$3 timers=$4 xfer=$5
    echo
    echo "=== $tag  ctx=$ctx riders=$riders timers=$timers xfer=$xfer"
    if [ "$timers" = "0" ]; then unset VLLM_KVPIPE_PHASE_TIMERS
    else
        export VLLM_KVPIPE_PHASE_TIMERS="$timers"
        export VLLM_KVPIPE_PHASE_TIMERS_OUT="$OUT/${tag}_phases.jsonl"
    fi
    "$PY" benchmarks/kv_pipelining_bench/rider_cost_curve.py \
        --model "$MODEL" --ctx "$ctx" --batch "$BATCH" \
        --riders "$riders" --connector --repeats "$REPEATS" \
        --short-len 8 --long-len 136 --gpu-mem-util 0.90 --link-gbps 25.1 \
        --xfer-mode "$xfer" \
        --outdir "$OUT" --out "${tag}.json" 2>&1 | tail -40
    unset VLLM_KVPIPE_PHASE_TIMERS VLLM_KVPIPE_PHASE_TIMERS_OUT
}

# ---- 1. paired perturbation study, at two contexts and a real rider axis -----
# timers-off vs level 1 (host only, class A) vs level 2 (adds device, class B).
# If off-vs-1 is within noise, level 1 may carry absolute values. If not, the
# law is fitted from untimed runs and these supply SHARES only.
RIDERS_512=0,1,2,3,4,6,8
RIDERS_256=0,1,2,4,6,8,12,16

run t0_c512  512 "$RIDERS_512" 0 real
run t1_c512  512 "$RIDERS_512" 1 real
run t2_c512  512 "$RIDERS_512" 2 real

run t0_c256  256 "$RIDERS_256" 0 real
run t1_c256  256 "$RIDERS_256" 1 real

# ---- 2. the direct decomposition, timers OFF so it is uncontaminated --------
# real - minbytes = the byte cost
# minbytes - nobus = the launch cost   <-- link-invariant; the C2C crux
# nobus            = everything else the host path costs
# minbytes is an infinitely fast link and therefore a CEILING for any
# interconnect: no link removes a launch. These arms are numerically invalid by
# construction and are recorded as such in their JSON.
run xfer_min_c512  512 "$RIDERS_512" 0 minbytes
run xfer_nob_c512  512 "$RIDERS_512" 0 nobus
run xfer_min_c256  256 "$RIDERS_256" 0 minbytes
run xfer_nob_c256  256 "$RIDERS_256" 0 nobus

# ---- 3. the engine-free cross-check on concern C3 ---------------------------
# Does a memcpy's launch cost depend on its size? If it does, the minbytes arm
# is a bound rather than a measurement, and this is the truth. Also bounds the
# CUDA-graph-removable fraction, which nothing in this repo has ever measured.
echo
echo "=== launch-pattern microbenchmark (engine-free)"
"$PY" benchmarks/kv_pipelining_bench/launch_pattern_bench.py \
    --layers 64 --riders 1,2,4,8,16 \
    --bytes-per-copy 8192,65536,262144,1048576,4194304 \
    --iters 30 --out "$OUT/launch_pattern.json" 2>&1 | tail -45

# ---- 4. fit whatever is fittable so far ------------------------------------
echo
echo "=== fit (real arms only; co-located and invalid arms are filtered out)"
"$PY" paper/model/overhead_fit.py "$OUT"/t0_*.json \
    --link-gbps 25.1 --out "$OUT/fit_t0.json" 2>&1 | tail -40

echo
echo "--- teardown"
"$PY" -c "
import sys; sys.path.insert(0,'benchmarks/kv_pipelining_bench')
try:
    from bench_teardown import terminate_engine; terminate_engine(); print('ok')
except Exception as e: print('teardown helper unavailable:', e)
" 2>&1 | tail -3
sleep 5
ps -u "$USER" -o args= | grep -E "[.]venv/bin/python.*kv_pipelining_bench|^VLLM::EngineCore" | head -5
echo "================ M1 done $(date -Is) ================"
