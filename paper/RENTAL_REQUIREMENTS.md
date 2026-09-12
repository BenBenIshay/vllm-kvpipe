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


---

## 8. Provider survey (2026-09-12)

Researched against §1. **Price is not the binding constraint — availability
and billing model are.** At the rates below, 74–82 GPU-h is **$150–210**, well
under the $250–450 the plan budgeted, so optimising for the cheapest hour is
optimising the wrong variable.

| provider | $/GPU-h | HBM | billing | stock | verdict |
|---|---|---|---|---|---|
| **Lambda Labs** | **2.29** | 96 GB | on-demand | out | **best fit if it comes back in stock** — persistent shared network filesystem, preinstalled CUDA stack, documented GH200 support |
| **Runcrate** | 2.52 | 96 GB | on-demand | **in** | only verified in-stock on-demand; unknown to us otherwise — needs the §7 questions |
| Sesterce | 2.52–4.65 | 96 **or 144** | on-demand | out | make sure any quote is the **96**, not 144 (§2) |
| CoreWeave | 6.50 | 96 GB | on-demand | — | ~2.8× Lambda for no benefit we need |
| Vultr | 1.99 | 96 GB | **SPOT** | out | **disqualified — see below** |

**Vultr's headline $1.99 is spot, and spot is disqualifying for this
experiment.** A preemption mid-run does not merely cost the hours: it ends the
continuous block that §4 exists to protect, and the measurements taken before
and after it are on different silicon. Our own data already shows what that
does — a contended node produced a 9.4× error in one measurement. Spot is fine
for the 4.0 gate if it is cheap enough to redo; it is not fine for 4.1 onward.

**Reported host memory across these configs is 432–480 GB**, comfortably above
the ≥256 GB in §1 (we need up to 87 GB pinned at 20×16K). Confirm on the
instance rather than from a spec page, and check the pinned-page limit (§7 Q4).

### Recommendation

1. **Lambda Labs** if available — it is the only one whose persistent
   filesystem is documented, and §4's two-block shape depends on exactly that.
   Watch for restock.
2. **Runcrate** as the in-stock fallback, gated on the §7 questions —
   particularly persistent storage and whether the GPU is exclusive.
3. Treat **availability as the scheduling risk**: multiple providers report
   sold out through Q3 2026. That, not price, is what could push the trip past
   the MLSys deadline and into the OSDI window.

*Caveat: prices and stock move weekly, and no spec page substitutes for
running `gh200_bringup.sh` on the actual instance. Re-check before booking.*


---

## 9. Runcrate — what their docs answer, and what they do not (2026-09-12)

Researched before contacting them, so the enquiry asks only what is genuinely
unknown. Draft message: `paper/RUNCRATE_ENQUIRY.md`.

**The finding that comes first: their own site does not list GH200.** Their
pricing page carries 18 SKUs (H200 down to A16) with no GH200, and their docs'
`/dedicated/available-gpus` lists **H100, H200, B200, B300**. The $2.52/hr
"in stock" figure comes from third-party trackers. Runcrate is a *marketplace
aggregating* capacity across providers and regions, so a tracker entry may
reflect a partner listing, stale data, or a region we cannot get. **Confirm the
product exists before evaluating anything else about them.**

Answered from their docs:

| §7 question | answer |
|---|---|
| Q2 persistent volumes | **Yes** — `/storage/overview`, `creating`, `mounting`, `file-explorer`. Survival across termination and restore throughput still unknown. |
| Q5 billing | **Per-second, no minimum commitment**, "stop the instance, stop the meter" — better than the on-demand hourly we asked for. |
| contact | Enquiry form, 24 h response; **Slack Connect** offered for direct engineering contact. |

Still unknown, and they matter more here than for a first-party cloud:

- **Bare-metal or VM.** Their docs mention bare-metal only for *reserved
  clusters of 16–128+ nodes*. A single on-demand GH200 may therefore be a VM,
  which raises both exclusivity and virtualization-overhead questions for a
  measurement paper.
- **Exclusivity**, unaddressed anywhere in the docs.
- **Image / `nvcc`** — no documentation page on images or CUDA at all.
- **Host RAM and the pinned-page limit** (§7 Q4), which is the one most likely
  to fail silently.
- **Same-host continuity across two sessions**, which for an aggregator is a
  harder guarantee than for a single operator.

**Assessment.** Per-second billing and persistent volumes suit our two-block
shape well. The marketplace model is the risk: it is the opposite of the
measurement continuity §4 depends on. Worth the enquiry — it costs nothing —
but Lambda restocking would still be the better outcome.
