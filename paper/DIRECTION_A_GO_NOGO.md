# Direction A — go/no-go evidence (A100, 2026-08-11)

Decision: does the **mixture** (some requests resident in HBM, some streamed
from host) have enough value to justify the 12–14 week systems paper?

All gates below were agreed in advance. Data in `paper/model/`.

---

## Gate 1 — the two-source attention tax  ✅ PASS

**Question (Ben's):** serving KV from two groups in one step needs specialised
attention logic. Does that cost more than the mixture gains?

**Method:** `two_source_tax.py`. **All KV in HBM in both arms** — no host
memory, no transfer, no pool. The only difference is one
`flash_attn_varlen_func` call over B rows vs two calls over B1+B2. CUDA graphs
on both arms (vLLM runs decode under FULL graphs, so launch overhead is not
part of the real tax). This cost is **interconnect-independent**, so it
transfers to GH200 unchanged.

**Result** (Llama-70B shape: 80L, 64q/8kv, S=8192; target config is all-HBM
serving B_cap=8 in one call vs the mixture serving 8+5=13 in two):

| | attention/layer | per request |
|---|---|---|
| all-HBM, B=8, single call | 0.196 ms | 24.4 µs |
| mixture, B=13, split 7+6 | 0.333 ms | **25.6 µs (+4.7%)** |

- pure structural tax at **equal** batch (13 single vs 13 split): **+12.6%**
- cost per request vs the **baseline it replaces**: **+4.7%**
- attention is ~36% of a 70B step ⇒ **~4.5% of ITL**

Decision rule was "under ~10% ⇒ concept sound." **Passes on the decision
metric.** The mixture runs at a *higher* batch, which amortises better and
largely absorbs the split cost.

**Unexpected: the tax can be negative.** At B=32 the single call hits a bad
FlashDecoding regime (S=8192: B=24→0.522 ms, B=32→0.870 ms, i.e. +67% time for
+33% batch), and splitting into 24+8 is **18% FASTER** than not splitting. Same
at the Qwen-32B shape (−13.5%, −10.7%). So the two-call structure sometimes
*dodges* a kernel pathology. This also reproduces, in ~200 lines with no engine,
the B=32 kink we found in-engine during P4 — confirming it is a FlashAttention
property, not a vLLM artifact.

## Gate 2 — achieved transfer bandwidth (X_eff)  ✅ PASS + a key finding

**Question:** `riders = ITL·X_eff/u`, so X_eff multiplies the whole value of A.
The roofline assumes 25.1 GB/s on A100/PCIe4. What do real access patterns give?

**Method:** `transfer_bw_probe.py`, streaming into a rolling one-layer window
(the k-window design), sweeping copy granularity.

| granularity | chunk | GB/s | vs assumption |
|---|---|---|---|
| **per-layer bulk** | 167.8 MB | **24.95** | **99%** |
| per-block | 327.7 KB | 18.47 | 74% |
| per-rider-block | 65.5 KB | 8.21 | 33% |
| **per-row (naive rung-d today)** | 4.1 KB | **0.52** | **2%** |

**The roofline's X_eff assumption is sound — but only at per-layer
granularity.** A 48× spread between the design we want and the code we have.

**This diagnoses our known slowness.** rung-d measured 2.25× slower than
baseline on A100 and we attributed it vaguely to "per-row Python loops." It is
now quantified: per-row copies get 0.52 GB/s against 24.95 GB/s for batched
per-layer copies. That is an **implementation defect with a known cheap fix**,
not a property of the approach.

## Gate 3 — does the transfer hide under compute?  ✅ PASS

Copy on a side stream concurrent with GEMM work: compute alone 27.79 ms,
compute+copy 28.05 ms, copy alone 6.72 ms ⇒ **96% of the transfer is hidden.**
The roofline's overlap assumption holds.

## Gate 4 — end-to-end estimate  ⚠️ COMPOSITION, NOT A MEASUREMENT

**Read this gate carefully — it mixes measured and modelled inputs.**

- **Measured:** the attention times at B=8 and B=13 (Gate 1).
- **Modelled, NOT measured:** that the mixture serves 13 where all-HBM serves 8.
  `B_cap=8` comes from HBM capacity (22 GB / 2.68 GB per req); `riders=5` from
  the transfer budget (31.9 ms x 450 GB/s / 2.68 GB). **The batch increase is
  still an assumption.** No experiment here has shown the batch actually rises.

Combining Gate 1's measured attention times with the weight-read term, scaled
to GH200 by the HBM bandwidth ratio (itself a crude approximation):

| Llama-70B fp8, GH200-96, S=8192 | attention | + weights | step | requests | rate |
|---|---|---|---|---|---|
| all-HBM (8 req) | 7.6 ms | 23.3 ms | 30.9 ms | 8 | 259 |
| mixture (13 req) | 12.9 ms | 23.3 ms | 36.2 ms | 13 | 359 |

**×1.39** — against the pure model's ×1.59. The gap is the riders' own attention
time, which the model's zero-copy idealisation omitted. **1.39× is the honest
number to plan against.**

## Gate 5 — kernel optimality  ✅ PASS (with one gap)

Every number above was taken on vLLM's auto-selected `FLASH_ATTN` (= FA2 on
A100; `num_splits` is **FA3-only**, so FA2's splitting cannot be tuned here).

**In-engine comparison, Llama-8B, S=4096** (`p4_8b_TRITON_ATTN.json`):

| B | FA2 ITL | Triton ITL | Triton vs FA2 | FA2 eff_kv | Triton eff_kv |
|---|---|---|---|---|---|
| 16 | 17.22 | 18.59 | +8.0% | 0.72 | 0.59 |
| 64 | 42.80 | 37.37 | −12.7% | 0.56 | 0.68 |
| 80 | 56.10 | 44.38 | −20.9% | 0.49 | 0.67 |
| 96 | 69.01 | 46.85 | **−32.1%** | 0.46 | **0.74** |

**There is a crossover near B≈50: FA2 wins below it, Triton wins above.**
Confirmed in isolation at matched shape (70B shape, B=8, S=8192; warmup swept
20/200/600 to rule out Triton autotune effects): FA2 **0.2015 ms** vs Triton
**0.4585 ms** — FA2 2.3× better. Against the 0.1387 ms bandwidth floor, FA2
achieves **eff_kv ≈ 0.69** at the target config.

**Consequence for Gates 1 and 4: our target config (weight-heavy model,
B_cap≈8, mixture≈13) sits in the small-batch regime where FA2 is the better
kernel.** The tax and end-to-end numbers were therefore measured on the right
kernel and stand.

**Consequence for direction B:** the eff_kv degradation we reported
(0.77 → 0.46) is **FA2-at-large-batch**, not a property of paged attention.
Triton holds ~0.67–0.74 there. The claim must be narrowed from "paged attention
leaves 40% of HBM bandwidth on the floor, invalidating this literature's
rooflines" to "**vLLM's backend auto-selection is up to 32% suboptimal at large
batch and does not switch at the crossover**" — still a real, deployment-relevant
finding, but a smaller one. **Ben called this risk in advance.**

**Gap:** FlashInfer could not be evaluated — its JIT needs `nvcc`, which this
cluster lacks (pip ships only `ptxas`; `ninja` then fails with exit 127). A
third backend could shift the crossover. Must be checked on GH200, where the
FA3 path also becomes available.

---

## Why A100 cannot show the magnitude (expected, not a failure)

At 24.95 GB/s, the 5-rider cohort's 13.42 GB/step needs **538 ms** of transfer
against a ~50 ms step ⇒ **0.47 riders**. On GH200 at 450 GB/s the same budget
buys **5.4 riders**. This is exactly what the gain law predicted, and it is why
the *magnitude* of A needs GH200 while the *structural* gates above do not.

## Standing on the verdict

Gates 1–5 pass. The mixture's structural premise survives contact with hardware:
the two-source tax is small (+4.7% on the decision metric), the transfer
bandwidth assumption is 99% accurate at the right granularity, transfers hide
at 96%, and the composed estimate is ×1.39 on the right kernel for this regime.

**What the gates establish:** nothing structural blocks the mixture.
**What they do NOT establish:** that the batch actually increases 8→13. That is
still the capacity model. Converting it into a measurement requires running the
real mixture in-engine and counting **requests served at a fixed SLO** —
all-HBM vs all-host vs mixture. On A100 that experiment is uninformative
(0.47 riders); it is exactly what the GH200 week should buy.

**Recommendation pending Gate 5:** proceed to a short GH200 rental for the
endpoint comparison (all-HBM vs all-host vs mixture) — a week and a few hundred
dollars to buy the deciding measurement — rather than committing 12–14 weeks up
front.
