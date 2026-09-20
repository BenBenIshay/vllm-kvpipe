# EuroHPC Benchmark Access — application draft (JUPITER Booster, FZJ)

Drafted 2026-09-20 against the official call documents. **Fields marked
`[FILL]` need Ben's details before submission.**

Submit at https://access.eurohpc-ju.europa.eu — Benchmark Access call.
Cut-off is the **1st of every month**; review within 2 weeks; access 2–3 weeks
after cut-off. First-come-first-served if the resource is oversubscribed.

**Eligibility check (done):** researchers located in a country associated to
Horizon Europe / Horizon 2020 are eligible, and Israel qualifies. The project
leader must hold an employment contract valid for **at least 3 months after the
end of the allocation period** — with a 3-month allocation, that means a
contract running ~6 months from the start date. `[FILL: confirm]`

---

## 1. Administrative

| field | value |
|---|---|
| Project title | **Interconnect-bounded admission for host-memory KV cache in LLM decode nodes** |
| Principal Investigator | `[FILL: full name]` |
| Institution | Bar-Ilan University, Israel `[FILL: department]` |
| Position / contract validity | `[FILL: must extend ≥3 months past allocation end]` |
| Co-investigators | `[FILL, if any]` |
| Research field | Computer Science / Machine Learning systems |
| Is this an AI application? | **Yes** (LLM inference serving) |
| System requested | **JUPITER Booster, JSC (DE)** |
| Allocation period | 3 months |
| Follow-up intention | Regular Access proposal (see §6) |

## 2. Abstract (~200 words)

Large-language-model inference on a decode node is limited by how much KV cache
fits in GPU HBM: once the model weights are resident, the remaining memory caps
the number of concurrent requests, and that cap — not compute — determines
throughput. We have implemented, in a fork of vLLM, a mechanism that places some
requests' KV cache in host memory and streams it to the GPU layer-by-layer
during decoding, so that the batch can exceed what HBM alone allows. Whether
this helps is decided entirely by the CPU–GPU interconnect: a request served
from host memory must move its whole KV cache across the link within one decode
step, or the step lengthens and the extra concurrency is wasted.

We have characterised the mechanism on PCIe-attached A100s and derived the
crossover condition, which PCIe (25 GB/s measured) sits below. NVLink-C2C on
Grace-Hopper is roughly fourteen times faster and is the first interconnect
predicted to sit above it. We request a small benchmark allocation on JUPITER's
GH200 Booster to measure the per-request cost curve and locate that crossover,
as preparation for a full evaluation under a Regular Access proposal.

## 3. Scientific / technical case

**The problem.** At the capacity-limited operating point, a decode step reads
essentially all of HBM once. Model weights consume most of it (Qwen2.5-32B in
fp16 is 65.5 GB), leaving a small KV budget `C` that fixes the maximum batch.
Adding host-tier requests ("riders") raises concurrency but costs interconnect
bandwidth every step.

**What we have measured (A100-80, PCIe, Qwen2.5-32B, exclusive node).** With a
decode-node emulation — KV delivered as from a remote prefill node, so no
prefill contaminates the measurement — and riders added on top of a full HBM:

| riders | decode step | ΔT/rider | link traffic |
|---|---|---|---|
| 0 | 53.5 ms | — | 0.000 GB/step |
| 2 | 71.6 ms | 9.1 ms | 1.115 GB/step |
| 4 | 97.6 ms | 11.0 ms | 2.230 GB/step |
| 6 | 146.9 ms | 15.6 ms | 3.379 GB/step |

Reproducible to under 1% across repeats. The cost is not linear in the rider
count; it fits a kink,

```
step ≈ max(T_compute, r · S · b_tok / X) + overhead
```

where `S` is sequence length, `b_tok` the KV bytes per token, `X` the link
bandwidth. Below the knee `r = T_compute·X/(S·b_tok)` the transfer hides behind
compute; above it the step *is* the transfer. On this hardware the knee is 2.4
riders and the residual overhead is 8.6–18 ms, so the best attainable gain is
0.86–0.99× — just below break-even, which is what theory predicts for a 25 GB/s
link.

**What only Grace-Hopper can answer.** The knee scales with `X` and should move
to ~5–6 riders at 8K context on NVLink-C2C, which against a capacity of ~6
resident requests would nearly double the batch. The *overhead*, however, is
software and does **not** scale with the link — and it is the term that decides
the outcome. It cannot be extrapolated from PCIe hardware, which is precisely
why this measurement needs a GH200 node.

**Deliverable.** The measured cost curve and crossover point on NVLink-C2C, as
the scalability evidence underpinning a Regular Access proposal (§6).

## 4. Code and method

| | |
|---|---|
| Code | vLLM (Apache-2.0), research fork with a host-tier KV subsystem |
| Languages | Python, CUDA; PyTorch 2.9.1 / CUDA 12.8 |
| Parallelism | **Single node, single GPU per experiment.** No MPI; concurrency is over inference *requests*, not ranks |
| Build | `pip install -e .` with `TORCH_CUDA_ARCH_LIST=9.0a`; scripted and time-boxed (`gh200_setup.sh`) |
| Correctness gate | 9-test suite, including bitwise and token-identical checks against unmodified vLLM; must pass before any measurement |
| Models | Qwen2.5-32B-Instruct (fp16, 65.5 GB), Qwen2.5-0.5B for smoke tests |
| Data | None. Synthetic token sequences; no external datasets, no personal data |

**Note on job shape, stated plainly.** Each experiment occupies one GPU, while
the Booster's minimum allocation is one full node of four GH200 Superchips. We
will run **four independent configurations concurrently per node** (different
rider counts and contexts) so the node is fully used. The object of study is the
Grace–Hopper CPU–GPU link itself, which is why an exascale machine is being
asked for a job that does not scale across nodes — the experiment is a
memory-system characterisation, not a scaling run.

## 5. Resource request

| field | value |
|---|---|
| **Total node hours** | **40** |
| Job size (minimum) | 1 node |
| Job size (average) | 1 node |
| Job size (maximum) | 1 node |
| Wall clock (typical job) | 1–2 h |
| Wall clock (maximum job) | 4 h |
| Simultaneous jobs | 1–2 |
| Memory | 480 GB/node (host-tier KV mirrors are **pinned**; peak measured ~64 GB) |
| Storage — scratch | 1 TB |
| Storage — work | 500 GB (model weights 62 GB + logs/JSON) |
| Storage — home | 20 GB (default) |
| Data transfer in | ~70 GB (model checkpoint, from Hugging Face) |
| Data transfer out | <5 GB (result JSON and logs) |

**Justification of the 40 node hours**

| work package | node-h |
|---|---|
| Environment build, correctness suite, bring-up gate (hard time-box) | 8 |
| WP1 — rider cost curve: rider count × 3 contexts, 3 repeats each | 8 |
| WP2 — crossover confirmation: measured link bandwidth, knee, overhead | 4 |
| WP3 — throughput gate: host-tier vs unmodified vLLM at matched latency | 12 |
| Contingency (~20%; system availability is ~80% per the guidelines) | 8 |
| **total** | **40** |

Every script is already written and has been dry-run on A100 hardware, so the
allocation is spent measuring rather than developing. The work plan is
front-loaded: WP1 alone answers the central question, and WP2–3 are conditional
on its outcome.

## 6. Link to a future Regular Access proposal

This benchmark establishes the two constants a larger evaluation must be sized
from — the crossover point and the per-step overhead on NVLink-C2C. Conditional
on a positive result we intend to submit a **Regular Access** proposal covering
the full study: a policy ablation over admission strategies, a comparison
against published host-offloading systems on their own workloads
(ShareGPT/Alpaca with Poisson arrivals), and multiple model scales and
precisions. That evaluation is estimated at several hundred node hours and
cannot be sized responsibly without the measurements requested here.

## 7. Acknowledgement and reporting (obligations accepted)

* Publications will state: *"We acknowledge the EuroHPC Joint Undertaking for
  awarding us access to Jupiter at FZJ, Germany."*
* A Final Report will be submitted to `access@eurohpc-ju.europa.eu` within
  3 months of the allocation ending.
* Results are open research; no military use.

---

## Before you submit

1. `[FILL]` the administrative fields, and check the **contract-validity rule**
   (≥3 months beyond allocation end).
2. Confirm with Bar-Ilan's research office whether an institutional endorsement
   is needed.
3. Confirm JUPITER is listed as selectable in the current Benchmark call form —
   the call text lists it, but availability per cut-off can vary.
4. Consider naming a **EuroHPC-country collaborator** as co-investigator if you
   have one; it is not required, but it removes any ambiguity about eligibility.
5. Submit shortly *after* a cut-off date rather than before — allocation is
   first-come-first-served within each monthly round.
