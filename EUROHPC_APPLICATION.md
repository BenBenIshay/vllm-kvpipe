# EuroHPC **Development Access** application — JUPITER Booster, JSC

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
| AI application? | **Yes** (LLM inference serving) — the call waives the scalability-challenges requirement for AI applications, which §3 invokes |
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

The assessment criteria for this call are *pertinence and feasibility of the
request for the targeted system*, a *plan for code development*, and *quantified
human resources and expertise*. §3 addresses pertinence, §4 the plan, §5 and §6
feasibility and people. The call waives the scalability-challenges requirement
for AI applications, and this is one: the request is deliberately **single-GPU**,
because tensor parallelism would introduce all-reduce traffic into the very step
time under measurement. There is therefore no scaling story to tell, and that is
a property of the experiment rather than a gap in it.

### 3.1 The use case, and the quantity that decides it

A large-language-model decode server is limited by memory, not arithmetic. Each
generated token requires reading every model weight and every resident KV cache
byte, at almost no arithmetic per byte, so the GPU's arithmetic units idle while
the memory system saturates — and what caps the number of concurrent requests is
the KV cache exhausting device memory.

Writing `HBM` for device memory, `W` for the weights and `C = HBM − W` for what
is left, the resident request count is `C / (S · b_tok)` for context length `S`
and KV bytes per token `b_tok`. Relocating a request's KV to host memory frees
`S · b_tok` of HBM and costs `S · b_tok` of link traffic **once per decoded
token**. Two consequences set up the whole request:

* **The sustainable number of host-memory requests is fixed by the interconnect,
  not by how much host memory exists.** It is `B_off = T_step · X / (S · b_tok)`
  for achieved link bandwidth `X` — a quantity an engine can measure at runtime
  and admit against.
* **The value of a second tier scales with `HBM/C`** — how much of device memory
  the weights consumed — and not with model size or context length, which cancel.
  So the technique is worth most exactly where weights fill the device.

### 3.2 What the model predicts, and what A100 confirms

The step-time model `ITL(B) = W/BW_w + B·S·b_tok/BW_kv` is **validated on
A100-80 across 41 configurations on two models**: inter-token latency is linear
in `B·S` within each model (`R² = 0.975` and `0.92`), and the intercept — which
the equation predicts rather than fits — is recovered exactly for Qwen2.5-32B and
20% high for Llama-3.1-8B. The parameters it needs are measured on the same
hardware, including the finding that paged-KV read bandwidth is **not** constant
but degrades by 1.6× as the working set grows, worst at precisely the
capacity-bound operating point the technique targets.

Fed with GH200-96 parameters, the model then says the technique is **marginal and
positive** for weight-heavy configurations and **negative** otherwise:

| model, GH200-96 | `HBM/C` | capacity | throughput | affordable fixed cost `F*` |
|---|---|---|---|---|
| Llama-3.1-8B fp16 | 1.26 | ×1.22 | **×0.98** | 4.8 ms vs 6.2 paid |
| Qwen2.5-32B fp8 | 1.60 | ×1.28 | **×0.91** | 6.9 ms vs 12.3 paid |
| Qwen2.5-32B fp16 | 3.63 | ×1.50 | **×1.01** | 12.8 ms vs 12.3 paid |
| Llama-3.1-70B fp8 | 4.97 | ×1.67 | **×1.05** | 18.2 ms vs 15.4 paid |

Here `C = HBM − W − 4 GB`, a flat device reserve for activations and
fragmentation; §4 quotes `HBM/C ≈ 7` for the 70B because it uses the 0.90 memory
utilisation the runs will actually be configured with, which is the operational
figure rather than the model's.

Two configurations clear break-even, by **0.5 ms and 2.8 ms** of a single measured
quantity: `F`, the fixed per-step cost of using the host path at all. That is the
number this request is about.

**On PCIe the model predicts a loss, and A100 measures one.** At its own budget
the controller admits two host-memory requests on top of sixteen resident
(`B_off = 2.5` at a measured `X = 25.1 GB/s`) and refuses more at any offered
load. That yields a batch `1.13×` against a step `1.31×` longer — **0.86×**, a
loss — and the same arithmetic puts break-even at `X* = 60 GB/s`, against PCIe's
25.1. The negative is therefore predicted rather than surprising, and it fixes
one end of the bandwidth axis. These are development-run numbers, reproducible
from scripted harnesses; WP1 re-measures the curve properly on the target system.

### 3.3 The solution, at a high level

We manage HBM and host DRAM as **one KV store** and assign each request a tier.

* A **device-tier** request behaves exactly as in unmodified vLLM.
* A **host-tier** request holds **no device blocks at all**. Its KV lives in
  pinned host memory and each layer's bytes are streamed across the link just
  before that layer's attention consumes them, into a small rolling buffer reused
  across all layers — four slots for a 64-layer model, so such a request occupies
  about 6% of the device memory a resident one needs. Layer `ℓ+1`'s transfer is
  issued while layer `ℓ` computes, which is the window the whole design depends on.
* **Both tiers are rows of the same batch in a single forward pass**, so the
  weights are read once however the batch is split. Attention is the only
  operation that distinguishes the tiers, and only in where it reads keys and
  values from.
* The host mirror has its **own block-id space**, held until a request finishes,
  so device block ids may be freed and reissued underneath it. One compaction per
  step maps the active cohort onto the rolling buffer.
* An **admission controller** sizes the host cohort at runtime from the measured
  link bandwidth and the measured rider-free step time — not from a configured
  offload fraction — and a placement policy chooses *which* requests, ranked by
  remaining generation rather than by context length.

Implemented in vLLM v1 in roughly 3,900 lines across 22 files. Every path is off
by default and selected at runtime, so a build with the feature disabled
exercises the stock engine — verified to match unmodified vLLM within 0.08%
across six interleaved runs. Exactness is checked in-process: a host-tier
request's attention output is bitwise identical to the same request served
resident, over 576 mixed decode layer-steps.

### 3.4 Why Grace-Hopper, and why the 96 GB part

`F` is **launch-bound, not bandwidth-bound**: on our hardware it is ~640 extra
kernel launches per step, about ten per layer across 64 layers. A faster
interconnect therefore does not shrink it, while the per-request transfer cost
shrinks in proportion. The two terms extrapolate in opposite directions, and
which one dominates on a coherent link decides whether the technique pays. `F`
depends on the host CPU and on issuing across 64 layers, so an x86 host behind
PCIe cannot predict what a Grace host will do.

**The measurement is also only possible there.** Separating a fixed per-step term
from a per-request term needs several cohort sizes *below* the knee
`r = T_step·X/(S·b_tok)`. On PCIe that knee is 2.4, so there is exactly one
usable point — which is why our own data cannot do it. At 2048 context on
NVLink-C2C it sits near 26, leaving room for ten or more.

JUPITER's Booster carries **GH200 with 96 GB HBM3**, which is the configuration
the model requires. The 144 GB variant enlarges `C` and *weakens* the effect, so
this is not a case where larger hardware would substitute.

### 3.5 Competing mechanisms

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
| WP1 | **request cost curve** — Δstep vs host-tier count, reported against riders *admitted*, not offered. Run at 2048 context, where the C2C knee sits near 26 requests and so leaves room for ten or more points below it — on PCIe the knee is 2.4 and there is room for one. That is what separates the fixed per-step term from the per-request term (§3.4). Sizes every later WP | 8 |
| WP2 | crossover figure — our PCIe points plus throttled C2C points, spanning ~5→350 GB/s | 6 |
| WP3 | re-parameterisation under FlashAttention-3 / Hopper kernels | 8 |
| WP4 | **ours vs unmodified vLLM** at matched latency target, 3 context lengths | 12 |
| WP5 | **interconnect-bounded count vs a uniform offload fraction** — the direct test of the contribution | 8 |
| WP6 | per-request placement and mid-flight migration — confirmation | 4 |
| WP7 | **vs Pie's swap semantics** (reconstructed; already validated at exactly 2.00× link traffic on our hardware) | 8 |
| WP8a | **DirectKV artifact triage** — build it, run its unit tests, measure what fraction of decode steps reach its fast kernel. Gates WP8b | 2 |
| WP8b | **vs DirectKV**, ratio-matched against each engine's own baseline (they build on SGLang, we on vLLM) | 10 |
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
hard-capped at 8 node-hours) and contingency for the risks in §7 gives 110.
System availability is assumed at ~80% per the technical guidelines.

Every script is written and has been dry-run on A100 hardware, so the allocation
is spent measuring rather than developing.

## 6. Human resources and expertise

> The call requires that *"the available human resources and expertise (available
> and/or required) to execute this planning must be quantified"*. This is a
> scored part of the technical assessment, so it needs numbers, not adjectives.

| role | person | commitment over the 12 months | responsible for |
|---|---|---|---|
| PI, lead developer | `[FILL: name]` | `[FILL: e.g. 0.5 FTE, ~6 person-months]` | WP0–WP7, WP9–WP10 |
| `[FILL: co-investigator or advisor role]` | `[FILL: name]` | `[FILL: e.g. 0.1 FTE, advisory]` | `[FILL: which WPs]` |
| `[FILL: student / collaborator, if any]` | `[FILL]` | `[FILL]` | `[FILL: e.g. WP8 DirectKV triage]` |

**Expertise already available.** `[FILL — see the note below]`

**Expertise required from the hosting entity.** `[FILL: state a need or write
"none required" — see the note below]`

**Evidence of readiness.** The system is already implemented and working: ~3,900
lines across 22 files in a vLLM v1 fork, a 9-test correctness suite, and every
measurement script in §4 written and dry-run on A100 hardware. The allocation is
spent measuring, not developing. WP0 is time-boxed at 8 node-hours precisely so
that a build problem cannot consume the allocation.

## 7. Risks and mitigations

| risk | mitigation |
|---|---|
| **fp8 70B does not take the fp8 kernel path on SM90** and dequantises to 140 GB, exceeding one GH200 | Verified at bring-up before any measurement. Fallback: Qwen2.5-32B fp16 as headline (fully validated in our suite), or 70B fp16 with tensor parallelism across 2 GH200 — noting that TP introduces all-reduce traffic as a confound |
| **DirectKV comparison is cross-engine** — their artifact is public (`github.com/shutianluo/DirectKV`) and Hopper-only, so JUPITER is the first machine on which it can run at all, but it is built on SGLang while ours is vLLM | Compare each system against **its own engine's baseline** and contrast the ratios, rather than putting two engines' absolute numbers side by side. Their own paper reports SGLang as the fastest engine when memory suffices, so the engine difference is real and must be controlled, not ignored |
| **Their artifact's fast kernel is gated on 64-aligned sequence length**; during decode that condition recurs once every 64 steps, with a per-request fallback path otherwise. If that is what the published artifact does, a naive head-to-head would measure an artifact property rather than their design | WP8a measures the fast-path hit rate first and gates WP8b on it. If the artifact is dominated by its fallback, we report that as a measured property of the code, make no claim about their published numbers, and revert the hours to WP4/WP7 |
| **aarch64 + SM90a build time is unmeasured** | Scripted, resumable and time-boxed at 8 hours; overrun means stopping and fixing offline |
| Pinned-page limits on the host | Checked in the first minutes; the failure mode is silent otherwise |

## 8. Obligations

* Acknowledgement, in the call's prescribed wording and using its own example
  for this machine: *"We acknowledge EuroHPC Joint Undertaking for awarding us
  access to Jupiter at JSC, Germany."*
* Final Report to `access@eurohpc-ju.europa.eu` within 3 months of the
  allocation ending.
* Open research; no military use. No personal data — all workloads are synthetic
  token sequences or public conversational datasets.
* Code is a fork of vLLM (Apache-2.0) and will be released with the publication.

---

## Before submitting

1. `[FILL]` administrative fields; confirm the **contract-validity rule** — the
   call requires the project leader's employment contract to be valid *"for at
   least 3 months after the end of the allocation period"*, so ~15 months from
   start for a 1-year allocation.
2. **Fill §6 in full.** Quantified human resources are a scored part of the
   technical assessment, not a formality — the guidance note in that section says
   what each field wants.
3. Confirm whether Bar-Ilan requires institutional endorsement.
4. **Re-check the DirectKV artifact is still reachable** at
   `github.com/shutianluo/DirectKV` — §7 and the 12 hours in WP8a/WP8b assume it
   is. If it has gone, WP8 becomes a paragraph and those hours move to WP4/WP7.
5. Submit **just after a cut-off**, not just before: awards are
   first-come-first-served against each month's quota, so you want a full
   month's runway.
