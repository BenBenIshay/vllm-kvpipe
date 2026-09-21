# EuroHPC **Development Access** application — JUPITER Booster (FZJ)

Drafted 2026-09-21 against the official call documents. Supersedes the earlier
Benchmark-call draft: this one covers the **whole evaluation the paper needs**,
which Benchmark Access (3 months, explicitly a stepping stone) cannot carry.

**Fields marked `[FILL]` need Ben's details before submission.**

| | |
|---|---|
| Call | **Development Access** — continuously open, cut-off 1st of each month |
| Turnaround | decision ≤2 weeks after cut-off, access 2–3 weeks after |
| Allocation | **1 year**, renewable up to twice |
| Award basis | first-come-first-served against the monthly quota |
| Scope per the call | *"code and algorithm development and optimisation, as well as development of AI application methods"* — which is what this is |
| Eligibility | EU Member State **or a country associated to Horizon Europe / Horizon 2020**. Israel qualifies. |
| PI contract | must be valid **≥3 months after the allocation ends** — with a 1-year allocation, ~15 months from start `[FILL: confirm]` |

---

## 1. Administrative

| field | value |
|---|---|
| Project title | **Interconnect-bounded admission for host-memory KV cache in LLM decode nodes** |
| Principal Investigator | `[FILL: full name]` |
| Institution | Bar-Ilan University, Israel `[FILL: department]` |
| Contract validity | `[FILL]` |
| Co-investigators | `[FILL, if any]` |
| Field | Computer Science — machine-learning systems |
| AI application? | **Yes** (LLM inference serving) |
| System | **JUPITER Booster, JSC (DE)** |
| Request | **110 node hours**, 12-month allocation |

## 2. Abstract

Serving a large language model is limited by how much key-value (KV) cache fits
in GPU memory: once the model weights are resident, what remains caps the number
of concurrent requests, and that cap — not arithmetic throughput — sets the
serving rate. Host memory is an obvious place to put the overflow, and several
recent systems do so. They differ in *mechanism*: some read host memory directly
during attention, some swap cache lines back and forth. All of them, however,
apply the same *policy* — every request is treated identically, either all
resident or all offloaded.

We argue the policy is where the performance is. A request served from host
memory must move its entire KV cache across the CPU–GPU link once per decoded
token, so the number of such requests that can be sustained is fixed by the
interconnect, not by memory capacity. We have built an admission controller that
derives that number from the measured link bandwidth and the measured decode
step, and characterised it on PCIe-attached A100s, where the crossover lies just
out of reach. NVLink-C2C on Grace-Hopper is the first interconnect predicted to
sit above it.

We request access to JUPITER's GH200 Booster to measure the crossover, compare
our design against the two competing mechanisms, and establish where in the
hardware landscape host-memory KV tiering is worth doing.

## 3. Technical case

### 3.1 The quantity that decides everything

At the capacity-limited operating point a decode step reads essentially all of
HBM once. Weights take most of it, leaving a KV budget `C` that fixes the batch.
Offloading a request frees `S·b_tok` of HBM and costs `S·b_tok` of link traffic
**per step** — so the question is whether that traffic hides behind compute.

### 3.2 What we have already measured (A100-80, PCIe, exclusive node)

Using a decode-node emulation — KV delivered as from a remote prefill node, so
no prefill contaminates the measurement — with riders added on top of a full
HBM, Qwen2.5-32B at 2048 context, 3 repeats, spread under 1%:

| host-tier requests | decode step | Δ per request | link traffic |
|---|---|---|---|
| 0 | 53.5 ms | — | 0.000 GB/step |
| 2 | 71.6 ms | 9.1 ms | 1.115 GB/step |
| 4 | 97.6 ms | 11.0 ms | 2.230 GB/step |
| 6 | 146.9 ms | 15.6 ms | 3.379 GB/step |

The cost is **not linear**. It fits a kink:

```
step  ≈  max(T_compute,  r · S · b_tok / X)  +  overhead
```

Below the knee `r = T_compute·X/(S·b_tok)` the transfer hides; above it the step
*is* the transfer. Measured here: knee **2.41** requests, residual overhead
**8.6–18 ms**, so the best attainable gain is **0.86–0.99×** — just below
break-even, exactly as a 25 GB/s link predicts.

### 3.3 Why this needs Grace-Hopper specifically

The knee scales with link bandwidth and should reach **5–6** at 8K context on
NVLink-C2C. The *overhead* is software and does **not** scale with the link —
and it is the term that decides the outcome. It cannot be extrapolated from
PCIe hardware. JUPITER's Booster carries **GH200 with 96 GB HBM3**, which is the
configuration our model requires: the 144 GB variant enlarges `C` and *weakens*
the effect, so this is not a case where bigger hardware would do.

### 3.4 Competing mechanisms

Three answers exist to "how do you serve attention over host-resident KV", and
they trade differently. We will measure all three on identical hardware:

| | KV path | link traffic | HBM cost | transfer hidden |
|---|---|---|---|---|
| **DirectKV** (OSDI'26) | DDR → SM, zero-copy | 1× | none | no — the read happens inside the operation that needs it |
| **Pie** (swap) | device copy is *owned*, written back | **2×** | full | partial |
| **ours** | prefetch into a rolling buffer | 1× | one k-window | yes |

No prior work compares them on one substrate, and the trade-offs are not
orthogonal: a bounce buffer costs HBM but buys the only opportunity to overlap.

## 4. Work plan

| WP | experiment | GPU-h |
|---|---|---|
| WP0 | port and validate: aarch64 + SM90a build, 9-test correctness suite, bring-up gate | (see §5) |
| WP1 | **request cost curve** — Δstep vs host-tier count, the knee, the overhead. Sizes every later WP | 8 |
| WP2 | crossover figure — our PCIe points plus throttled C2C points, spanning ~5→350 GB/s | 6 |
| WP3 | re-parameterisation under FlashAttention-3 / Hopper kernels | 8 |
| WP4 | **ours vs unmodified vLLM** at matched latency target, 3 context lengths | 12 |
| WP5 | **interconnect-bounded count vs a uniform offload fraction** — the direct test of the contribution | 8 |
| WP6 | per-request placement and mid-flight migration — confirmation | 4 |
| WP7 | **vs Pie's swap semantics** (reconstructed; already validated at exactly 2.00× link traffic on our hardware) | 8 |
| WP8 | **vs DirectKV** — their artifact if available, else our SM90a backend | 12 |
| WP9 | composition with fp8 KV quantisation — exactness preserved, capacity multiplied | 4 |
| WP10 | negative control — Llama-3.1-8B, where the model predicts no benefit | 4 |
| | **total measurement** | **74** |

**Models.** Headline **Llama-3.1-70B fp8** — `RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8`,
72.7 GB of weights across 15 shards (compressed-tensors) — on a **single**
GH200, which avoids tensor parallelism and the all-reduce traffic it would add
to the very step time we are measuring. At the 0.90 memory utilisation we
actually run, that leaves `C ≈ 13.7 GB` for KV and a resident capacity of ~5
requests at 8K context, i.e. `HBM/C ≈ 7` — the strongest point in our gain law
and one prior work has not evaluated. **Qwen2.5-32B fp16** provides a second point on
the weights-to-KV ratio, and **Llama-3.1-8B** the negative control. All three
are open-weight and already validated in our test suite at smaller scale.

## 5. Resource request

| field | value |
|---|---|
| **Total node hours** | **110** |
| Job size min / avg / max | 1 node |
| Wall clock, typical / max | 2 h / 6 h |
| Simultaneous jobs | 1–4 |
| Memory | 480 GB per node — host-tier mirrors are **pinned**; measured peak ~64 GB per instance |
| Storage — scratch | 1 TB |
| Storage — work | 500 GB (checkpoints ~155 GB: 70B fp8 72.7 + 32B 65.5 + 8B 16, plus logs and result JSON) |
| Storage — home | 20 GB (default) |
| Data in / out | ~160 GB checkpoints in; <10 GB results out |

**How 74 GPU-hours becomes 110 node-hours.** A Booster node carries four GH200
Superchips and the minimum allocation is one full node. Each experiment uses one
GPU, so we run **2–4 independent configurations per node** — four where the
pinned host mirrors are small, two at 70B where they are not (4 × 70B would
exceed a comfortable share of the node's 480 GB). That puts the measurement at
roughly 30 node-hours. Adding **WP0** (build, correctness suite and bring-up,
hard-capped at 8 node-hours) and contingency for the risks in §6 gives 110.
System availability is assumed at ~80% per the technical guidelines.

Every script is written and has been dry-run on A100 hardware, so the allocation
is spent measuring rather than developing.

## 6. Risks and mitigations

| risk | mitigation |
|---|---|
| **fp8 70B does not take the fp8 kernel path on SM90** and dequantises to 140 GB, exceeding one GH200 | Verified at bring-up before any measurement. Fallback: Qwen2.5-32B fp16 as headline (fully validated in our suite), or 70B fp16 with tensor parallelism across 2 GH200 — noting that TP introduces all-reduce traffic as a confound |
| **DirectKV artifact may not be public** (WP8) | Our own SM90a backend exists as a skeleton but is untested; if neither path works, WP8 is reported as a qualitative comparison and its hours revert to WP4/WP7 |
| **aarch64 + SM90a build time is unmeasured** | Scripted, resumable and time-boxed at 8 hours; overrun means stopping and fixing offline |
| Pinned-page limits on the host | Checked in the first minutes; the failure mode is silent otherwise |

## 7. Obligations

* Acknowledgement: *"We acknowledge the EuroHPC Joint Undertaking for awarding
  us access to Jupiter at FZJ, Germany."*
* Final Report to `access@eurohpc-ju.europa.eu` within 3 months of the
  allocation ending.
* Open research; no military use. No personal data — all workloads are synthetic
  token sequences or public conversational datasets.
* Code is a fork of vLLM (Apache-2.0) and will be released with the publication.

---

## Before submitting

1. `[FILL]` administrative fields; confirm the **contract-validity rule**
   (≥3 months past a 1-year allocation).
2. Confirm whether Bar-Ilan requires institutional endorsement.
3. **Check whether DirectKV released an artifact** — this decides whether WP8 is
   12 hours or a paragraph.
4. Submit **just after a cut-off**, not just before: awards are
   first-come-first-served against each month's quota, so you want a full
   month's runway.
