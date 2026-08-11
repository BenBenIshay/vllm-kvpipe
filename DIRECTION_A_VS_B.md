# Directions A and B — full plans (2026-08-11)

Decision document. Written before the backend-calibration run so the plans are
not retrofitted to whatever that run produces.

**Key scheduling fact up front: A and B are not exclusive.** B is A's Section 2.
Everything B requires, A needs anyway as motivation and parameterization. So the
cheap path is: do B's work now (~4–6 wk, no new hardware beyond a short GH200
rental), then decide on A at a defined gate. Choosing A first does *not* save the
B work; choosing B first costs A nothing.

---

# DIRECTION A — Request-level KV tiering (systems paper)

## A.0 Thesis

Decode batch is capped by HBM capacity. Coherent CPU–GPU interconnects make host
memory a usable KV tier, and the *mechanism* is solved (DirectKV, OSDI'26).
The **policy** is not. Both deployed points are endpoints — stock vLLM/SGLang
keeps 100% of KV in HBM; DirectKV puts 100% on the host for every request,
always (§6 + Alg. 3), which is why it needs no scheduler. The optimum is
interior, and reaching it requires deciding *which* requests live where and *how
many* the interconnect can carry per step.

Llama-70B fp8, 8K, GH200-96 (projected):

| configuration | KV bandwidth used | batch |
|---|---|---|
| all-HBM (vLLM/SGLang) | 92 GB/step HBM, 0 link | 8 |
| all-host (DirectKV) | 0 HBM KV, ~14 GB/step link | 5 |
| **mixture (ours)** | 22 GB HBM **+** 14 GB link | **13** |

## A.1 Contributions

1. **The interior-optimum result** — the mixture beats both endpoints (1.46× over
   all-HBM, 2.3× over all-host), and neither endpoint system can reach it.
2. **Request-level tier assignment** from observable request properties (context
   length, estimated remaining decode, inter-turn idle time), with **no external
   service-class label** and no accuracy loss.
3. **SLO-targeted rider admission** — seq-length-aware roofline (B1) + closed-loop
   measured-ITL guardrail (B2). *Already built and validated.*
4. **Tier migration** — promotion/demotion of whole requests with hysteresis and a
   migration cost model. *Not built; the hardest piece.*
5. **An evaluation that isolates policy gain from mechanism gain** — nobody in
   this space has done it; every prior paper changes both at once.

## A.2 What must be built

| # | Piece | Effort | Risk | Status |
|---|---|---|---|---|
| P1 | Zero-copy substrate — integrate DirectKV kernel (or mapped-pinned zero-copy in our FA override) behind the existing host pool | 2–3 wk | med | not started |
| P2 | Request-level tier assignment scorer + admission integration | 3–4 wk | low-med | not started |
| P3 | Migration (promote/demote, hysteresis, cost model, correctness-preserving handoff) | 4–5 wk | **high** | not started |
| P5 | Re-platform evaluation to 30B–70B on GH200; realistic pool; real traces | 3 wk | med | not started |
| P6 | Baselines (see A.4) | 2 wk | low-med | not started |
| P7 | Losslessness promoted to a contribution + accuracy comparison | 1–2 wk | low | evidence exists |

Reused as-is: host block-id KV pool (bitwise validated), scheduler block-growth
cap, roofline, B1+B2 controller, correctness ladder, opt #5.

## A.3 Experiments

- **Spine ablation (make-or-break):** one substrate, three policies — (a) no
  offload, (b) uniform offload (DirectKV as published), (c) ours. The paper is
  `(c) − (b)`.
- **SLO attainment & requests served at fixed SLO** — the primary metric, not raw
  throughput, given a 1.5–1.6× ceiling.
- **Migration benefit** under a shifting request mix (bursty arrivals, mixed
  context lengths, multi-turn idle gaps) — isolates contribution 4.
- **Sweeps:** model size (8B / 32B / 70B, spanning the predicted dud→win range),
  context, arrival rate, burstiness.
- **Interconnect sensitivity:** PCIe vs C2C, validating the gain law's `X` term.
- **Correctness:** token-identical output maintained across migration events.

## A.4 Baselines

1. HBM-resident vLLM **and** SGLang (SGLang for comparability with DirectKV).
2. **Uniform zero-copy offload** — the critical one.
3. vLLM CPU swap (pre-C2C status quo).
4. **KV fp8** — answers "just quantize." Our model says it is *neutral* to our
   relative gain (scales `B_cap` and rider count together), so the two compose;
   measure rather than assert.
5. ShadowKV if time permits — for the lossless-vs-lossy exchange rate, not as a
   throughput competitor we expect to beat.

## A.5 Hardware

GH200-96 — one for 70B fp8, or two for 70B fp16 TP=2. This is the schedule gate.
A100 cannot host this evaluation (the gain law says ×1.0–1.17 there).

## A.6 Timeline

```
P4-A100 (done) ─► P1 (2-3w) ─┬─► P2 (3-4w) ─┐
                             └─► P6 (2w)    ├─► P5 spine (3w) ─► GATE
                 P7 (1-2w) ────────────────┘
                 P3 (4-5w, parallel after P1; correctness-gated)
                 P4-GH200 (1w, needs access)
```
Critical path **12–14 weeks** assuming GH200 access lands before P5.

## A.7 Kill criteria

- Spine ablation `(c) − (b)` below ~15% at matched SLO → **abandon A, fall back to B.**
- DirectKV integration fails and our staged path stays >20% slower than published
  zero-copy → the mechanism gap swamps the policy gain; fall back to B.
- A concurrent publication lands request-level tiering on C2C → fall back to B.

## A.8 Venue

MLSys (fit: systems+ML, roofline-friendly) or EuroSys/SoCC (fit: scheduling
policy). Need actual CFP dates.

## A.9 Risks

1. **Magnitude.** ×1.59 ceiling lossless; DirectKV published 1.2×. Modest-gain area.
2. **Policy on someone else's mechanism** — needs the mechanism to be
   demonstrably insufficient alone. That is precisely the spine ablation.
3. **OmniServe adjacency** — needs an explicit differentiation paragraph
   (regime, attention placement, label source), not a one-line dismissal.
4. **Anti-scaling** — gain shrinks as HBM grows (70B: ×1.59 on GH200-96 → ×1.12
   on GH200-144). Structurally the same decay we cite against NEO. Address head-on:
   weights grow faster than HBM (MoE, 405B+), so `HBM/C` trends our way.
5. **GH200 access** is the schedule gate.

---

# DIRECTION B — When does host-KV offload pay? (characterization paper)

## B.0 Thesis

The field has accumulated many host-KV offload mechanisms (FlexGen, NEO, Pie,
ShadowKV, InfiniGen, DAK, DirectKV) and **no validated model of when any of them
helps.** Published results disagree in ways nobody explains — NEO reports 7.5× on
T4 and 14% on H100; DirectKV is *slower* than HBM-resident on Llama-8B but wins
on OPT-30B. We supply the model that explains both, parameterize it with
measurements, and map the applicability window.

## B.1 Contributions

1. **Design rules**, closed-form: context length cancels; capacity multiplier
   `≈ 1 + (X/BW)·(HBM/C)`, i.e. value tracks *how much of HBM the weights ate*;
   an SLO gate at the full-pool ITL.
2. **Measured parameterization** — the finding that carries the paper:
   `eff_w` is a stable constant (0.75 on A100) while **`eff_kv` is not** —
   it degrades ~1.6× with KV working set (0.77 → 0.46). **Every fixed-bandwidth
   roofline in this literature, including the IEEE CAL letter that claims the
   "first formal treatment," mis-predicts precisely the capacity-bound operating
   point those models exist to describe.**
3. **Explanatory power** — the model retrodicts DirectKV's 8B/OPT-30B inversion,
   NEO's decay across GPU generations, and our own A100 ~1.1×.
4. **The applicability map** — which (model, hardware, SLO) cells benefit and by
   how much, as deployment guidance.
5. **The interior-optimum observation** as an analytical result (both deployed
   systems sit at endpoints) — stated and modelled, not built. This is the hook
   that makes A the natural follow-up.

## B.2 What must be built

| # | Piece | Effort | Status |
|---|---|---|---|
| B-i | Backend-controlled eff_kv characterization (FLASH_ATTN / FLASHINFER / TRITON_ATTN) + synthetic streaming ceiling | **2–3 h** | **next** |
| B-ii | Extend sweep: models (0.5B/8B/32B), contexts, block sizes, KV dtype (fp16/fp8) | 1 wk | harness exists |
| B-iii | GH200 measurements — re-measure eff_kv under FA-3; validate the `X` term | 1 wk | needs short rental |
| B-iv | Retrodiction section: reproduce DirectKV's config in the model, compare to their published numbers | 3 days | analysis only |
| B-v | Optional: one *measured* mixture datapoint using the existing rung-d path | 1 wk | code exists |

## B.3 Experiments

- eff_w / eff_kv across backends, models, contexts, block sizes, KV dtypes.
- Synthetic streaming read at matched footprint → the memory-system ceiling.
- Full-pool ITL across models → the falsification of the model-independent gate
  (already have: 79.3 ms for 8B vs 55.7 ms for 32B on A100).
- SLO-gate sweep: vary the SLO across the predicted knee, show where capacity
  starts binding.
- A100 vs GH200 → the interconnect term.

## B.4 Timeline

**4–6 weeks**, of which only B-iii needs hardware we don't have (and only briefly).

## B.5 Venue

ISPASS / IISWC (measurement studies are the point there), IEEE CAL (short, and
directly answers the letter we would be correcting), or an MLSys workshop.
Lower prestige than A; substantially lower risk.

## B.6 Kill criteria

Essentially none — the measurements are the deliverable. The only real failure
mode is that all backends land at the same eff_kv *and* the synthetic ceiling
also lands there, which would demote contribution 2 from "kernels leave 40% on
the floor" to "this access pattern costs 40%" — still a publishable finding,
just a less striking one.

## B.7 Risks

1. Reads as "just a measurement study" without a crisp novel finding. The eff_kv
   result is that finding; if the backend sweep dissolves it, B weakens.
2. Someone publishes the same characterization first. Mitigated by B's short
   timeline.
3. Our A100-derived curve may not transfer to Hopper — which is why B-iii is
   mandatory, not optional.

---

# Recommendation

**Run B-i now (2–3 h), then do B, holding A as the follow-up gated on B-iii.**

Rationale:
- B's work is required by A regardless — it is A's Section 2 and its
  parameterization. Doing it first costs A nothing.
- B is publishable on its own with what we have plus ~4–6 weeks.
- The one thing that could invalidate A's premise (eff_kv behaving differently
  on Hopper/FA-3) is measured inside B, at B-iii, *before* we commit 12–14 weeks
  and a GH200 rental to A.
- A's make-or-break experiment (the spine ablation) is unrunnable without GH200
  anyway, so A cannot start in earnest before B-iii regardless.

**Decision point: after B-iii.** If GH200 confirms the capacity-bound regime and
eff_kv stays low enough that the rider budget holds, commit to A. If FA-3 closes
the KV-bandwidth gap and the window narrows, publish B and stop.
