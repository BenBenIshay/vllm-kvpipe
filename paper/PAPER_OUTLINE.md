# Paper outline — request-level KV tiering for capacity-bound decode

Status: **reframed 2026-08-10** after a full read of the 2025–2026 literature
(seven papers read end-to-end; PDFs + extracted text in the session scratchpad).
Supersedes the "free-rider decoding / riding the weight read" framing, which was
built on an inaccurate premise and is pre-empted on mechanism. See
§"What changed and why" at the end.

Companion state: memory `project-hybrid-kv-tiers`, `state-hybrid-kv-execution`.

---

## 1. Title (pick/refine)

1. "Who Rides Along? Request-Level KV Tiering for Capacity-Bound LLM Decode"
2. "Tiered KV Residency: SLO-Targeted Request Placement on Coherent CPU–GPU Interconnects"
3. "The Policy Layer for Host-Memory KV: When Offloading Pays, and Which Requests to Offload"

## 2. Thesis

Decode batch size is capped by **HBM capacity**, not by compute — the GPU runs at
single-digit utilization because KV cache, not arithmetic, exhausts the device.
Coherent CPU–GPU interconnects (NVLink-C2C) make host memory a usable KV tier,
and the *mechanism* for reading host-resident KV efficiently is now solved
(DirectKV, OSDI'26). What is not solved is the **policy**: which requests should
live in which tier, how many host-tier requests a step can absorb without
breaking the resident cohort's ITL, and when a request should migrate between
tiers. We contribute that policy layer, plus a closed-form model of **when host
KV offload pays at all** — a question the field currently answers by trial.

**Non-goal:** we do not claim a new offload mechanism. We build on one.

## 3. The analytical spine (contribution 1)

Let `X` = interconnect bandwidth, `BW` = achieved HBM bandwidth, `HBM` = device
memory, `C` = HBM left for KV after weights, `S` = context, `k` = KV bytes/token.

Four closed-form results. **(a) and (c) validated on hardware 2026-08-10; (b)
falsified; (d) survives only in a weakened form.** See "measured parameters".

**(a) Context length cancels.** Capacity-limited batch `= C/(S·k)` and hiddable
host-tier requests `= ITL·X/(S·k)`. Their ratio is independent of `S`. Long
context does *not* make offloading more attractive; it shrinks both cohorts
equally.

**(b) ~~The rider budget is a hardware constant.~~ FALSIFIED — measured
2026-08-10.** The claim was `ITL₀ ≈ HBM/BW`, model-independent. Measurement says
otherwise, because weights and paged KV do **not** stream at the same efficiency
(see "measured parameters" below). Full-pool ITL on A100-80: **79.3 ms for
Llama-8B vs 55.7 ms for Qwen-32B** — a 1.4× spread where equality was predicted.
The gate in (d) survives only in the weaker, model-*dependent* form.

What remains true: the rider budget is still `ITL·X`, and most of it is still
bought with **weight-read time** — on GH200, 70B spends ~22 ms of a ~32 ms step
reading weights, during which nothing else wants the link.

**(c) The gain law.** capacity multiplier `≈ 1 + (X/BW)·(HBM/C)`. The bandwidth
ratio is a 0.15 handicap on GH200; `HBM/C` is the amplifier. **Value tracks how
much of HBM the weights ate**, not model size or context per se.

Capacity multiplier on GH200-96 @8K, recomputed with the **measured** parameters
(eff_w=0.75, eff_kv from the measured working-set curve) — zero-copy, i.e. riders
cost no HBM pass:

| Model on GH200-96 | weights | C (KV HBM) | full-pool ITL | riders/B_cap | **capacity** |
|---|---|---|---|---|---|
| Llama-8B fp16 | 16 GB | 76 GB | 50.4 ms | 0.30 | **×1.22** |
| Qwen-32B fp16 | 64 GB | 28 GB | 33.0 ms | 0.53 | **×1.46** |
| **Llama-70B fp8 / fp16 TP=2** | 70 GB/GPU | 22 GB | 31.9 ms | 0.65 | **×1.59** |

Moved <4% from the pre-measurement projections — the gain law depends on the
*ratio* HBM/C, so absolute-bandwidth errors largely cancel between the ITL
numerator and the capacity denominator.

**(d) The applicability gate** (weakened by measurement). Capacity binds only
when the **ITL SLO exceeds the full-pool ITL for your model**. The clean
model-independent form ("time to read all of HBM") is falsified; the gate is now
`SLO > W/(BW·eff_w) + C/(BW·eff_kv)`, which varies by model. Below it you are
latency-limited — the scheduler is already refusing requests HBM could hold, and
offloading is strictly harmful. Still a *throughput-oriented serving* feature
(agentic, batch, long-form, reasoning), not an interactive-chat feature.

### Measured parameters (A100-80 PCIe, 2026-08-10/11, `paper/model/p4_*.json`)

Two models (Llama-3.1-8B, Qwen2.5-32B); 41 single-shot configurations plus a
13-configuration n=3 repeat sweep (`p4_llama8b_a100_rep3.json`). Differential
timing (ITL = [T(N2)-T(N1)]/(N2-N1), same prompts both runs, so prefill/warmup
cancel exactly); distinct random contexts, prefix caching OFF.

- **Structure VALIDATED:** ITL is linear in **total KV tokens (B·S)** —
  `ITL = 9.37 ms + 0.138 ms/1000 tok`, R²=0.975, residuals unbiased across
  S=2048/4096/8192 (means +0.09/−0.09/+0.00 ms). B=16/S=4096 and B=8/S=8192 both
  measure 17.17 ms. **Design rule (a) — context cancels — is now empirical.**
- **Weight read: eff_w = 0.75 of peak** (~1450 GB/s). Stable across both models,
  all contexts, and robust to the eff_kv assumed at B=1 (0.739–0.773 over that
  whole range). Weight term for Llama-8B on A100-80 = 11.07 ms.
- **Paged KV read: NOT a constant** — degrades ~1.6× with working set
  (n=3 repeats, error bars propagated):

  | KV working set | 4.3 GB | 8.6 GB | 17–34 GB | 43 GB | 51.5 GB |
  |---|---|---|---|---|---|
  | **eff_kv** | 0.77±.04 | 0.72±.02 | 0.54–0.56±.02 | 0.49±.01 | **0.46–0.48±.03** |

  This is the real correction to the model, and it matters beyond us: every
  fixed-bandwidth roofline in this literature (incl. the IEEE CAL letter)
  mis-predicts the capacity-bound operating point.
- **Caveat on rule (a) at kernel granularity.** The "only total KV bytes matter"
  claim holds at 3 of 4 same-byte test points (8.6 GB: 0.722 vs 0.726; 34.4 GB:
  0.560 vs 0.540; 51.5 GB: 0.460 vs 0.475) but **breaks at 17.2 GB** (B=32/S=4096
  → 0.561 vs B=16/S=8192 → 0.747). Cause: a *reproducible* anomaly at
  B=32/S=4096 — marginal cost per request along S=4096 runs 0.34, 0.41, 0.61,
  **0.24**, 0.76, 0.83, 0.81 ms, and the same kink appeared in an independent
  earlier run. Attention-kernel scheduling artifact (likely FlashDecoding
  split-KV heuristics), not measurement noise (±0.61 ms vs a ~3 ms deviation).
  The aggregate law holds; the per-configuration surface has kernel-regime steps.
- **Conclusions robust:** across three rounds of parameter revision the GH200
  projections moved <4% (70B ×1.56→×1.59, 32B ×1.43→×1.45, 8B ×1.14→×1.22).
  Sensitivity: sweeping eff_kv over the plausible 0.56–0.68 band for 70B's 22 GB
  working set moves it only ×1.58→×1.61. The gain law depends on the *ratio*
  HBM/C, so absolute-bandwidth errors largely cancel between the ITL numerator
  and the capacity denominator.
- **Open:** eff_kv(working set) is measured on A100/FA-2-era kernels. If FA-3 on
  H100/GH200 closes that gap, our projected gain **falls** (eff_kv sits in the
  capacity denominator). **Must be re-measured on GH200** — now the top open
  risk. Error bars: n=3 repeats give ±0.02 ms at B=1 rising to ±3.6 ms at B=96
  (2–5% relative); the eff_kv degradation is far outside them.

**Independent validation.** DirectKV reports that on Llama-3.1-8B it is *slightly
slower* than the HBM-resident baseline (SGLang) and wins only on OPT-13B/30B
where SGLang OOMs, with a 1.2× headline. Our model predicts exactly this: 8B
leaves C=76 GB (nothing to gain); OPT-30B's 60 GB of weights leaves C≈32 GB.
**Our analysis explains a published OSDI result its own authors do not explain.**
That is the strongest single asset in the paper.

Compute is not the binding term in the regimes we
claim. Decode-attention arithmetic intensity `= q_heads/kv_heads` (the GQA group
size) — 8 for 70B, 4 for 8B — versus an H100 balance point of ~116 FLOP/byte, so
attention sits 14–29× below the compute roof. For 70B at batch 8, total compute
is 7% of memory time. (Llama-8B at 4K *does* cross into compute-bound at batch
142 — another reason 8B is the wrong evaluation model.)

Scripts and data: `paper/model/` — `killer_scenario.py`, `killer_v2.py`,
`compute_check.py` (projections); `p4_llama8b_a100.json`, `p4_qwen32b_a100.json`
(measurements). Harness: `benchmarks/kv_pipelining_bench/p4_itl_model_validation.py`
+ `p4_fit.py`.

## 4. Positioning — what is taken, and what is left

| Prior work | Venue | Owns | Why it does not pre-empt us |
|---|---|---|---|
| **DirectKV** | OSDI'26 | Zero-copy GPU attention over CPU-resident KV on GH200/GB200; CPU-memory-aware tiling, warp pipelining, kernel fusion on FA-3 | §8.1 verbatim: *"DirectKV does not change scheduling, batching, or KV-management policies… leaving the scheduler… unchanged."* Two contributions, both kernel-level. **It is our substrate.** |
| **DAK** | arXiv'26 | Per-*operation* offload ratios via greedy analysis, TMA direct-to-SMEM; offloads weights too | Decides *which tensors*, given a **global offload ratio as input**. Not which requests, no SLO, no admission. |
| **OmniServe** | arXiv'26 | Mixed GPU/host batch with per-layer piggyback admission under SLO | Different regime: **PCIe** A100s + CPU-only hosts over RoCE, attention computed **on the CPU**, tiering by **externally-given service class** (LS vs BE). Ours: coherent C2C, GPU-side attention, tiering inferred from request properties. |
| **Dynamic KV Placement** | IEEE CAL'25 | Formal placement model + oracle upper bound on aggregate bandwidth | **Simulation only**, synthetically shrunk HBM (8 GB for KV), **token**-granularity by importance, requires **perfect foresight**, gains contingent on 60–80% attention sparsity. Closes by explicitly calling for runtime policies. Ours: request-granularity, lossless, online, implemented. |
| **ShadowKV** | ICML'25 | Low-rank K on GPU + V offloaded + sparse reconstruction; 6× footprint, 3.04× | **Lossy.** Different mechanism. Our head-to-head question: what does exactness cost at matched capacity? |
| **NEO** | MLSys'25 | CPU attention *compute* offload to grow batch | Same problem statement, opposite mechanism; gains decay as GPUs strengthen (T4 7.5× → H100 14%). |
| **Bottlenecks** | arXiv'26 | Trace-driven study finding **"VRAM capacity, not token budget"** binds | **Supports our premise.** Opening citation; names scheduler improvements as future work. |

**The gap, stated precisely — the endpoint argument.** DirectKV §6/§7.1: *"During
prefill, the manager stores the initial K,V entries from each layer **in CPU
memory**"*, and Alg. 3 appends newly projected KV to CPU memory too. **DirectKV
puts 100% of KV on the host, for every request, always.** That is why it needs no
scheduler — placement is static and total. Stock vLLM/SGLang is the opposite
endpoint at 0%. Nobody runs the mixture, and the optimum is interior:

| Llama-70B fp8, 8K, GH200-96 | KV bandwidth used | batch |
|---|---|---|
| all-HBM (vLLM/SGLang) | 92 GB/step HBM, 0 link | 8 |
| all-host (DirectKV) | 0 HBM KV, ~14 GB/step link | 5 |
| **mixture (ours)** | 22 GB HBM **+** 14 GB link | **13** |

**1.46× over all-HBM, 2.3× over all-host** — and unreachable from either endpoint
without deciding *which* requests go where and *how many* the link carries per
step. Note DirectKV leaves the 22 GB of HBM not holding weights **entirely
idle**; that is why it is slower than SGLang whenever SGLang fits.

Caveat on the composability reading of §8.1: the vLLM/SGLang mechanisms it names
(continuous batching, prefix reuse, eviction, hierarchical KV caching) decide
**reuse** and **preemption** — the residency of blocks whose request is *not
currently decoding*. None decides the tier of an **actively decoding** request.
That state did not exist before the zero-copy mechanism created it.

## 5. Contributions

1. **A closed-form model of when host-KV offload pays** (§3), validated against
   both our measurements and published systems' results. Deployment rule:
   `gain ≈ 1 + (X/BW)(HBM/C)`, applicable only above the `ITL_SLO > HBM/BW` gate.
2. **Request-level tier assignment**: an online policy that assigns each request
   to the HBM or host tier from its own observable properties — context length,
   estimated remaining generation, inter-turn idle time — with no external
   service-class label and no accuracy loss.
3. **Tier migration under SLO pressure**: promotion/demotion of whole requests
   with hysteresis and a migration cost model, closing the gate-new-only
   limitation of our current controller.
4. **SLO-targeted rider admission**: seq-length-aware roofline (B1) plus a
   closed-loop measured-ITL guardrail (B2) that holds the resident cohort's ITL
   while the host cohort varies.
5. **Implementation in vLLM v1** and an evaluation isolating *policy* gain from
   *mechanism* gain on a common zero-copy substrate.

## 6. What is already built (assets)

| Piece | State | Reuse |
|---|---|---|
| Host-block-id KV pool (`hybrid_rungd_pool.py`) | bitwise-validated, seed/grow/stream/write-redirect/compact-read | Becomes the host-tier allocator behind the zero-copy path |
| CPU-decode attention override (`flash_attn.py::_pool_cpu_override`) | token-identical to vanilla | Replaced by zero-copy kernel; kept as correctness oracle |
| Scheduler block-growth cap (`scheduler.py`) | reclaims host-tier HBM | Keep |
| Roofline (`hybrid_kv_roofline.py`) | hide-budget model, interconnect-parameterized | Extend with §3 design rules |
| B1+B2 admission controller (`transfer_aware_admission.py`) | unit test PASS + in-server staggered validation | **Core of contribution 4** |
| Correctness ladder | bitwise pool test; `=2` in-server self-verify (576 layer-steps, max\|diff\|=0); token-identical real-KV generation | **Core of the lossless claim** |
| Opt #5 (skip redundant standard pass over CPU rows) | 16.4s → 12.8s wall (~22%) | Keep; obsoleted if zero-copy lands |
| Serve-more harness | 2.67× concurrent at fixed 160-block pool | **Demote from headline to mechanism validation** (synthetic pool, 0.5B model) |

## 7. Development plan for the missing pieces

Each item states the gap, the build, and **how it differs from existing work** —
the differentiation is the reviewable claim, so it is specified up front.

### P1 — Zero-copy substrate  *(mechanism; no novelty claimed)*
- **Gap:** our path stages host→HBM k-window→attention, costing a redundant HBM
  pass. Our own model says this is the difference between 1.43× and 1.57×.
- **Build:** integrate DirectKV's open-source kernel
  (`github.com/shutianluo/DirectKV`) behind the existing pool allocator, or, if
  integration fights vLLM v1's attention path, a mapped-pinned-pointer zero-copy
  read inside our FlashAttention override.
- **Differs:** it does not — we cite it as substrate. The paper's honesty depends
  on saying so plainly, and on running our policy *and* the baselines on it.
- **Effort:** 2–3 weeks. **Risk:** medium (FA-3/CUTLASS vs our vLLM fork).
- **Fallback:** keep the staged path and report policy gains at the lower ceiling,
  noting the substrate delta.

### P2 — Request-level tier assignment  *(contribution 2)*
- **Gap:** today "which requests are host-tier" is effectively arrival order; the
  controller decides only *how many*.
- **Build:** a scorer over observable per-request state — current context length,
  estimated remaining decode length, time since last turn (multi-turn/agentic
  sessions), and the request's marginal HBM cost — feeding tier assignment at
  admission. Start with a transparent linear/threshold rule so the paper can
  explain it; compare against a learned variant only if the simple one leaves
  measurable headroom.
- **Differs from:** OmniServe (uses an **externally supplied** LS/BE class label —
  ours infers tier from request properties, so it works in a single-class
  deployment); DAK (per-**operation** ratio from a **given** global ratio); the
  CAL letter (per-**token** by importance, needs an **oracle**, lossy-sparsity
  dependent); DirectKV (no policy at all).
- **Effort:** 3–4 weeks including the ablation. **Risk:** low-medium — the risk is
  that the simple rule captures nearly all the gain, which is a fine result but
  weakens the "policy is load-bearing" story. Mitigation: P5's uniform-offload
  baseline is what makes the claim, not the sophistication of the rule.

### P3 — Tier migration (promotion/demotion)  *(contribution 3)*
- **Gap:** documented limitation today — gate-new-only. A burst admits everyone
  before the loop engages, and a long-running cohort can drift over SLO with no
  recourse but to finish.
- **Build:** demote resident→host under ITL pressure and promote host→resident
  when headroom appears; hysteresis on both edges; a migration cost model
  (bytes moved vs steps of benefit) to prevent thrash; correctness-preserving
  handoff reusing the pool's seed/compact-read primitives.
- **Differs from:** the CAL letter's reactive/LRU migration is **token**-level and
  their own results show migration overhead dominates below ~60% sparsity —
  **our testable claim is that request-granularity migration avoids that thrash**
  because whole requests move once, at controlled moments, rather than pages
  chasing shifting token importance. That is a direct, citable disagreement with
  a published result: good paper material.
- **Effort:** 4–5 weeks (the hardest piece). **Risk:** high — correctness during
  migration is where bugs live. Gate it behind the existing bitwise/token-
  identical harness before any performance number is taken.

### P4 — Model validation on real hardware  *(contribution 1)*
- **Gap:** §3 is a roofline projection; nothing is measured against it.
- **Build:** (i) **A100 now** — sweep measured decode ITL across batch × context ×
  model and fit against `(W + B·S·k)/BW`, publishing the achieved-bandwidth
  factor and the residual; (ii) **GH200 later** — repeat, plus a direct test of
  the `ITL_SLO > HBM/BW` gate by sweeping the SLO across the predicted knee.
- **Differs from:** the CAL letter validates in a **simulator** against an oracle;
  we validate closed-form rules against **measured** ITL on two interconnect
  generations, and against third-party published results (DirectKV's 8B-vs-OPT-30B
  inversion).
- **Effort:** 1 week (A100) + 1 week (GH200). **Risk:** low.
- **STATUS: A100 done 2026-08-10 — GATE PASSED with corrections.** Structure
  validated (R²=0.975, context/batch interchangeable); eff_w=0.78 confirmed on
  two models; eff_kv found to be working-set dependent (0.71→0.45), which
  falsified design rule (b); headline projections moved <4%. Remaining: repeats
  for error bars, and the GH200 re-measurement of eff_kv under FA-3.

### P5 — Evaluation re-platform and the load-bearing ablation  *(contribution 5)*
- **Gap:** everything measured so far is Qwen2.5-0.5B / Llama-8B on A100 with a
  synthetic 160-block pool — provably the corner where the feature cannot pay.
- **Build:**
  - Models: **Qwen-32B and Llama-70B** (fp8 on one GH200-96, or fp16 TP=2).
    **Drop Llama-8B** — DirectKV already published that it does not benefit.
  - Realistic pool (`gpu_memory_utilization≈0.9`), not a synthetic cap.
  - Workloads: ShareGPT + LongBench (matching DirectKV/OmniServe for
    comparability), Poisson arrivals, plus a bursty trace to exercise P3.
  - Metrics: **SLO attainment and requests served at fixed SLO** first, raw
    throughput second — the honest framing given a 1.4–1.6× ceiling.
  - **The spine experiment:** same zero-copy substrate, three policies —
    (a) no offload, (b) **uniform offload** (DirectKV as published, no policy),
    (c) ours. The paper lives or dies on (c) − (b).
- **Differs from:** nobody has isolated policy gain from mechanism gain in this
  space; every prior paper changes both at once.
- **Effort:** 3 weeks + GH200 access. **Risk:** medium — if (c) − (b) is small,
  we fall back to the characterization paper (§9).

### P6 — Baselines
- HBM-resident vLLM **and** SGLang (DirectKV's baseline, for comparability).
- **Uniform zero-copy offload** — the critical one (P5).
- vLLM CPU swap (the pre-C2C status quo).
- **KV fp8** — answers "just quantize the KV." Our model says quantization is
  **neutral** to our relative gain (it scales `B_cap` and rider count together),
  so the two **compose**; measure it rather than asserting it.
- ShadowKV if time permits — for the lossless-vs-lossy point, not as a
  throughput competitor we expect to beat.
- **Effort:** 2 weeks. **Risk:** low-medium (ShadowKV integration is the tail).

### P7 — The losslessness argument
- **Gap:** we have bitwise/token-identical evidence treated as hygiene.
- **Build:** promote it to a contribution — capacity expansion at **bit-exact**
  output, with an accuracy comparison against KV quantization and ShadowKV at
  **matched effective capacity** on long-context benchmarks.
- **Differs from:** every high-gain system in this space (ShadowKV, Quest,
  H2O, quantization) trades accuracy for capacity; none reports the exchange rate.
- **Effort:** 1–2 weeks. **Risk:** low. High value for rebuttals.

### Sequencing and gates

```
P4-A100 (1w)  ──► GATE: model within ~10% of measured?  if no, stop and rethink
   └─► P1 (2-3w) ──┬─► P2 (3-4w) ──┐
                   └─► P6 (2w)     ├─► P5 spine ablation (3w) ──► GATE: (c)−(b) material?
                                   │                                 if no → §9 fallback
                   P7 (1-2w) ──────┘
   P3 (4-5w, parallel once P1 lands; hardest, correctness-gated)
   P4-GH200 (1w, needs access)
```

Critical path ≈ **12–14 weeks** to a full submission, assuming GH200 access lands
before P5. P4-A100 and P7 need no new hardware and should start immediately.

## 8. Section plan

1. Introduction — capacity, not compute, caps the batch (cite Bottlenecks).
2. Background & Motivation — the four design rules (§3); why mechanism is solved
   and policy is not; the DirectKV 8B-vs-OPT-30B inversion as evidence.
3. Design — tier assignment, migration, SLO-targeted admission (B1+B2).
4. Implementation — vLLM v1, env-gated; host pool; zero-copy substrate.
5. Evaluation — the spine ablation, SLO attainment, model validation, sweeps.
6. Related Work — the §4 table.
7. Discussion & Limitations — the narrow applicability window is a *finding*, not
   an apology; host-memory pressure; multi-tier/disk; what our gain law implies
   as HBM grows.

## 9. Fallback paper (if the P5 gate fails)

"When does host-KV offload actually pay?" — the §3 model, validated on A100 and
GH200, explaining published systems' results and mapping the applicability
window. ~80% already in hand, needs no GH200 heroics, honest. Target
ISPASS/IISWC/CAL. Lower ceiling, much lower risk.

## 10. Known risks

1. **Magnitude.** 1.43× lossless ceiling (1.57× zero-copy); DirectKV got 1.2×.
   This is a modest-gain area. Mitigation: SLO-attainment framing, and being
   first to state *when* it pays.
2. **Policy on someone else's mechanism** needs the mechanism to be demonstrably
   insufficient alone — that is exactly the P5 spine ablation.
3. **OmniServe** needs an explicit differentiation paragraph (regime, attention
   placement, label source), not a one-line dismissal.
4. **Anti-scaling.** The gain shrinks as HBM grows (70B: 1.43× on GH200-96 →
   1.12× on GH200-144) — structurally the same decay we cite against NEO. Address
   it head-on: weights are growing faster than HBM (MoE, 405B+), so `HBM/C`
   trends favorably; say it deliberately before a reviewer does.
5. **GH200 access** remains the schedule gate for P5.

## 11. Open decisions (need Ben)

1. **Direction confirmed?** Policy paper (this doc) vs the §9 characterization
   fallback. Affects ~10 weeks of work.
2. **GH200 access** — rent now, or run P4-A100 + P7 first and rent at the P1 gate?
   Note 70B needs fp8 on one GH200-96 or fp16 TP=2 across a pair.
3. **Author list + affiliations.**
4. **Venue + deadline.** MLSys vs EuroSys/SoCC changes the framing weight between
   analysis and system. Need the actual CFP dates.
5. **Where we write** — Overleaf with the venue style, or scaffold LaTeX in `paper/`.
6. **ShadowKV baseline** — build it (P6 tail) or cite and argue?

---

## What changed and why (2026-08-10)

The previous outline claimed decode is **weight-bandwidth-bound** and that extra
requests "ride free" on spare bandwidth. Ben challenged this; he was right.

- At the capacity-limited operating point the step is ~79% KV read, ~21% weight
  read — "weight-bound" describes the *under-filled* regime, which is precisely
  where the feature is not needed. The binding constraint is **KV capacity
  capping the batch**, exactly as Ben stated, and "Bottlenecks" (arXiv 2601.19910)
  confirms it with real traces.
- The free ride is real but it is **time**, not bandwidth: the interconnect
  streams during the long stretch when the GPU is reading weights. That is why
  the gain scales with `HBM/C` and why 70B works and 8B does not.
- The old contribution set (mixed forward, host pool, interconnect-aware
  admission) is pre-empted on mechanism by DirectKV/DAK and partly on policy by
  OmniServe. The surviving, defensible gap is the **policy layer**, which three
  papers explicitly leave open.
- Consequently: **Llama-8B and A100 leave the evaluation**, the 2.67×
  synthetic-pool result is demoted to mechanism validation, and GH200 moves from
  "headline gate" to "the platform where the applicability window is open."
