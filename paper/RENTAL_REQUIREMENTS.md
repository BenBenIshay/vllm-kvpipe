# GH200 rental — what to ask for, and what to reject

Written 2026-09-12. Every line is derived from the plan or measured, not
guessed. Take this to a provider; it is meant to be read as a spec.

---

## 1. Hard requirements — a machine missing any of these cannot run the experiment

| | requirement | why, and what breaks without it |
|---|---|---|
| GPU | **1× NVIDIA GH200 with 96 GB HBM3** | See §2 — 144 GB is actively worse for us, not better. |
| Interconnect | **NVLink-C2C between the Grace CPU and the Hopper GPU** | The whole result is that a coherent link makes host DRAM a usable KV tier. A PCIe-attached H100 is a different experiment. |
| Exclusivity | **whole GPU, not MIG, not shared** | A contended node already produced a **9.4× error** in one of our measurements before anyone noticed. Every number here is a ratio; a neighbour's job invalidates it silently. |
| Host RAM | **≥ 256 GB** | The host-tier KV mirrors are **pinned**. Measured for Qwen-32B fp16: 44 GB at 20 requests × 8K, 65 GB at 30 × 8K, **87 GB at 20 × 16K**. Plus the engine, the OS and page-cache headroom. GH200 boards typically ship 480 GB LPDDR5X, which is comfortable — but confirm rather than assume, because a pinned allocation that fails is not obvious from a log. |
| Disk | **≥ 250 GB usable** | Checkpoint 62 GB + venv 8.6 GB + build tree + logs, with room for a second model. |
| Arch | **aarch64** (Grace is ARM) | Stated so nobody supplies an x86 host with a discrete H100 and calls it GH200. |
| Billing | **on-demand, hourly, releasable early** | §4.0 is a gate: if it fails we stop at ~16 h instead of 82. A committed block removes that option and is worth taking only for a large discount. |

## 2. Reject GH200-144

Counter-intuitive and worth stating explicitly to a salesperson who will offer
it as the better part: **more HBM makes our effect smaller**, because the gain
tracks `HBM/C` where `C` is the memory left after weights. For Qwen-32B fp16 it
does not merely shrink — it **switches off**: at `C ≈ 73 GB` the workload stops
being capacity-bound at all and the applicability gate stops admitting. The
96 GB part is the configuration the paper is about.

## 3. Strongly preferred — costs us specific experiments if missing

| | requirement | what is lost |
|---|---|---|
| Image | **CUDA `devel`, not `runtime`** (must contain `nvcc`) | `nvcc` is needed to compile DirectKV (SM90a CUTLASS/CuTe) and for FlashInfer's JIT. Without it §4.7 and the FlashInfer arm drop out. **Not a blocker** — vLLM ships precompiled kernels and everything load-bearing runs without it — but it is free to ask for and expensive to retrofit. Verify in minute one: `nvcc --version`. |
| Storage | **persistent volume, or fast same-region object storage** | Decides the rental shape (§4). Without it, every rental repeats bring-up. |

**`ptxas` is not the same question.** It ships inside the nvidia pip wheels and
is simply not on `PATH`; "no ptxas" was a false alarm here once. The gap that
matters is specifically `nvcc`.

## 4. Rental shape

```
bring-up (<=8 h cap, expect 1-3)  ->  4.0b baseline integrity  ->  4.0 gate
                                                 |
                             FAIL --> release. ~16 h, ~$65 spent, not 82 h.
                             PASS --> SNAPSHOT the built environment,
                                      then run 4.1 onward continuously.
```

**Why not simply rent one long block.** Cost is not the deciding factor: at
~$4/GPU-h a second bring-up is $12–32 against a ~$330 trip. The reason to keep
4.1 onward *continuous* is **measurement continuity** — if 4.1 runs on
different silicon from 4.0, the baselines shift underneath every ratio the
paper reports. The reason not to pre-commit the whole block is that §4.0 is a
real gate we might fail.

**The snapshot is what makes this cheap.** ~71 GB (venv 8.6 + checkpoint 62)
turns a later bring-up from *rebuild + verify* (up to 8 h) into *restore +
verify* (~1 h). The restore must be verified, not assumed: our editable install
uses a `MetaPathFinder` with an absolute path baked in, consulted before
`sys.path`, so a restored venv can import the *wrong tree* and run clean.
`gh200_bringup.sh` gates on the importer's answer.

## 5. Budget

74–82 GPU-h total, ~$250–450 at typical on-demand rates, in two blocks:

| block | hours | contents |
|---|---|---|
| **1 — decide** | ~16 | bring-up, 4.0b, 4.0 gate |
| **2 — measure** | ~60 | 4.1, 4.2, 4.2b, 4.3, 4.3b, 4.5, (4.7 if `nvcc`) |

Block 2 is only entered if block 1's gate passes.

## 6. Minute-one checklist on the rented box

```bash
nvcc --version                                    # expect present; warn if not
bash benchmarks/kv_pipelining_bench/gh200_bringup.sh
```

`gh200_bringup.sh` is the whole of bring-up: toolchain, device exclusivity,
measured interconnect bandwidth, KV geometry, capacity at the util actually in
force, the import-path check, checkpoint staging, and the 9/9 correctness
suite. It exits with the number of failed hard gates. **Non-zero means do not
take numbers from this machine.**

## 7. Questions for the provider

1. Is the GPU **exclusive**, or shared/MIG-partitioned?
2. **Persistent volumes** or same-region object storage — available, and at
   what throughput? (We restore ~71 GB.)
3. Is the image **CUDA devel** (with `nvcc`) or runtime-only? Can we choose?
4. **Host RAM** on the instance, and is there any limit on **pinned** pages?
   (cgroup or `ulimit -l` — a low pinned limit fails in a way that looks like a
   crash, and it cost us a day on our own cluster.)
5. On-demand hourly with **early release**, or minimum committed term? What is
   the discount for committing?
6. GH200 **96 GB** availability specifically — not 144.
