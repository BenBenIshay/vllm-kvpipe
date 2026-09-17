# GH200 runbook — what to do, in order, from the moment the box exists

Written 2026-09-17. Phase 0 is the rider cost curve; everything else is sized
from its output. Follow this top to bottom.

---

## 0. BEFORE you rent — push the work (it is not pushed)

`git status -sb` on the cluster reads `## hybrid-kv-tiers` with **no upstream**.
The branch has never left this machine, so there is nothing to clone on the
rented box. This needs your GitHub credentials, so only you can do it:

```bash
cd /mnt/beegfs/benishay/vllm-kvpipe
git push -u origin hybrid-kv-tiers
```

Also worth pushing the worktree branch if you want its history: `kvpipe-66`.

Nothing else here is portable-blocking: the cluster paths in the scripts
(`/mnt/beegfs/...`) are all defaults that an env var or flag overrides.

**Also take with you:** the HF token (`hf_token`, not in git) and the value of
`HF_HUB_CACHE` you want on the box.

---

## 1. Minute one on the box

```bash
nvcc --version              # expect present. Absent => 4.7/FlashInfer drop out; NOT a blocker
nvidia-smi                  # confirm GH200 96GB, and that it is exclusively yours
free -g                     # >=256 GB; the KV mirrors are PINNED
ulimit -l                   # pinned-page limit. "unlimited" is what you want
```

A low `ulimit -l` fails later as an unexplained crash, so check it before
spending a model load.

## 2. Get the code and build (the long pole)

```bash
git clone -b hybrid-kv-tiers https://github.com/BenBenIshay/vllm-kvpipe
cd vllm-kvpipe
python3.12 -m venv .venv && . .venv/bin/activate
pip install -U pip
TORCH_CUDA_ARCH_LIST=9.0a pip install -e .      # aarch64 + sm_90a, UNMEASURED duration
pip install pandas xlsxwriter                    # benchmark-only, needed by 4.2c/4.5
export HF_HUB_CACHE=/path/with/250GB
export HF_TOKEN=...                              # or drop the hf_token file in place
```

The venv from the cluster is **x86_64 and cannot be copied** — only the
checkpoint is portable, and HF's CDN will almost certainly beat copying it from
here. Start the download in the background while the build runs:

```bash
huggingface-cli download Qwen/Qwen2.5-32B-Instruct &
```

Time-box the build at the plan's 8 hours. If it overruns, release and fix
offline — a failed bring-up is cheap, a day of debugging on the meter is not.

## 3. Bring-up gate

```bash
bash benchmarks/kv_pipelining_bench/gh200_bringup.sh
```

One script, eight stages, exits with the number of failed hard gates.
**Non-zero means do not take numbers from this machine.** It checks: toolchain,
device exclusivity, **measured** C2C bandwidth, KV geometry and capacity at the
util actually in force, that `vllm` imports from this tree **and still parses
the tiering flags**, that the checkpoint is staged, and then runs the 9/9
correctness suite.

Write down two numbers it prints — **measured link GB/s** and **capacity
(requests at your context)**. Phase 0 needs the second one.

Useful overrides: `MODEL=`, `CTX=`, `GPU_MEM_UTIL=`, `PY=`, `--quick` (skips
the suite; do not use for the real gate).

## 4. PHASE 0 — the rider cost curve (~30 min, this is the trip's question)

```bash
python benchmarks/kv_pipelining_bench/rider_cost_curve.py \
  --model Qwen/Qwen2.5-32B-Instruct --ctx 8192 \
  --batch <capacity from step 3> \
  --riders 0,2,4,6,8,12 --connector --repeats 3 \
  --outdir ./runs/phase0
```

The link is measured automatically — do **not** pass `--link-gbps 25.1`, that
is the A100's PCIe figure and it misprices everything downstream.

It prints the model directly:

```
step ~= max(T_compute, r * S * b_tok / X) + overhead
  KNEE r = T/per-rider        <- how many riders fit inside a step
  overhead (median)           <- the part a faster link does NOT remove
  best operating point / VERDICT
```

**What to look for.** A100 measured a knee at 2.4 riders and ~13 ms of overhead
on a 53 ms step, which is why it loses. GH200 at 8K should put the knee near
**5-6 riders**. The question the trip exists to answer is whether the overhead
stays near ~10 ms there; if it does, the feature pays, and the verdict line
says so.

If the note says *"the knee lies BEYOND the sweep"*, re-run with `--riders`
extended past it before quoting any gain.

## 5. Only then, the rest

Phase 0's knee and overhead size everything after it:

| next | why |
|---|---|
| 4.0 gate (`gate_goodput.py --mode gate`) | goodput A vs C at one cell. Run `--mode verify-goodput` once first. |
| 4.2c workload gate (`affinity_spread_gate.py`) | gates 4.3; already passes on A100 |
| 4.3 policy ablation | only meaningful if phase 0 shows a usable rider range |
| 4.5 Pie-comparable run | ShareGPT + Poisson via `vllm bench serve` |

Set `--max-num-seqs` to ~3x capacity deliberately and record it: it moved T0 by
**12x** on the A100 and the whole SLO grid hangs off T0.

## 6. Housekeeping that has bitten us

* **Snapshot before releasing.** `.venv` + checkpoint ≈ 71 GB turns a second
  rental's bring-up from 8 h into ~1 h. Verify the restore by checking
  `python -c "import vllm; print(vllm.__file__)"` resolves into the restored
  tree — the editable finder has an absolute path baked in and will happily
  import the wrong one.
* **Tear engines down properly.** Everything here routes through
  `bench_teardown.terminate_engine`; orphaned EngineCore processes drained two
  of our own nodes on 2026-09-06.
* **Analyse off the meter.** Runs write JSON; copy it out and plot at home.
* Keep the A100 allocation alive for reproducing failures on 0.5B.
