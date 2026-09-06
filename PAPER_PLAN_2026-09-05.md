# Paper construction plan — target MLSys 2027

Written 2026-09-05. Supersedes the *scheduling* half of `PAPER_OUTLINE.md`
(2026-08-11), which predates everything from 2026-08-13 onward. The outline's
§3 (analytical spine) and §4 (positioning) survive intact and are still the
best text we have; its §6 (assets) and §7 (development plan) are stale and are
replaced below.

Companion docs: `PAPER_OUTLINE.md` (spine + positioning),
`GH200_EXPERIMENT_PLAN.md` (the rental), `BLOCKER_PREFILL_RESIDENCY.md` (the
full measurement narrative Aug-12 → Sep-04).

---

## 1. Venue: MLSys 2027, with OSDI '27 as the roll-forward

| Venue | Abstract | Paper | Notification | Verdict |
|---|---|---|---|---|
| ASPLOS '27 Sep cycle | — | Sep 9, 2026 | Dec 21 | **Out** — 4 days |
| EuroSys '27 Fall | Sep 17, 2026 | Sep 24, 2026 | Jan 29, 2027 | **Out** — the headline experiment is unmeasured and the GPU is unrented |
| **MLSys 2027** | — | **Oct 30, 2026** (opens Oct 10) | Feb 28, 2027 | **TARGET** |
| OSDI '27 | Dec 1, 2026 | Dec 8, 2026 | — | **Backstop**, +5.5 weeks |
| ASPLOS '28 Apr cycle | — | ~Apr 2027 | — | far backstop |

**Why MLSys and not EuroSys Fall.** EuroSys Fall is 19 days out. Everything that
makes this a systems paper rather than a measurement report — mixture beats both
endpoints on a coherent interconnect — is currently *modelled*. Submitting an
A100-only version means submitting a paper whose own evaluation section says
`-2.3%`. That is a desk-reject shaped like a paper.

**Why MLSys is the right home anyway.** NEO (our closest problem statement) is
MLSys'25. The audience calibrates correctly on a 1.4–1.6× ceiling in a
capacity-bound regime, which a EuroSys reviewer may read as thin. 55 days is
enough for a rental plus writing, and the review load falls on the analytical
spine, which is our strongest asset.

**Why OSDI '27 is a genuinely good backstop, not a consolation.** It is
DirectKV's own venue; a paper that says "the mechanism you published leaves the
scheduler on the table, here is the policy layer and here is what it buys" reads
naturally there. It buys 5.5 more weeks for the same draft. **Decision rule: if
the GH200 window has not started by Oct 3, roll to OSDI Dec 8** — do not
compress the evaluation into two weeks.

---

## 2. What actually changed since the outline (and must be folded in)

Six things, all load-bearing. Every one of them changes text that already exists.

### 2.1 The system model is now a disaggregated-prefill decode node

This is the largest change and the outline does not mention it. The Aug-13
conclusion — *"on PCIe, in a combined prefill+decode engine, the mixture cannot
win"* — was correct, and the fix was not an optimisation but a change of
setting. Riders are admitted with **zero main-pool blocks**, their prompt is
declared already computed, and their KV is delivered straight into the host
mirrors (`preload_req`, `_rungd_admit_preloaded`, plus the head-of-line fix in
vLLM's waiting loop). That is only legitimate on a **decode node in a P/D
deployment**, where prefill happens elsewhere and KV arrives in host DDR.

Consequences for the paper:
- §2 must state the system model explicitly and early. It is a *scoping
  assumption*, not a detail.
- It is a **strength**: P/D disaggregation is the production default
  (DistServe / Splitwise / Mooncake, and vLLM's own connector API). We are
  targeting the architecture the field already runs.
- It is also the **single largest attack surface** — see §6.1. The handover is
  built (`VLLM_KVPIPE_RUNGD_PREFILL_STASH=1`, `stash_from_host`/
  `preload_from_stash`, bitwise-proven by `prefill_stash_roundtrip.py`), so we
  can answer it with a measurement rather than an argument.

### 2.2 The capacity result got much stronger — and it is A100-measurable

`hybrid_preload_capacity.py`, Qwen-32B, ctx 2048, `b_hbm_max` = the *measured*
all-HBM ceiling:

| offered | all-HBM peak running | mixture peak running |
|---|---|---|
| 48 | 8 | **48** |
| 96 | 7 | **96** |

At the same KV pool. Both are **lower bounds** — the mixture hit the offered
load, not its own ceiling. This does not need GH200 and it falsifies the whole
`b_hbm_max` sweep that preceded it (riders and residents used to compete for the
same headroom; the sum crept toward capacity from below and never passed it).

**Caveat that must travel with the number, always:** riders are filled
synthetically, so generated tokens are meaningless. **This is a capacity claim,
not a goodput claim.** Never quote it in the same sentence as a tok/s number.

### 2.3 The cost side is now decomposed, and that is the paper's spine

`rider_hide_efficiency.py` on current HEAD (Qwen-32B, ctx 2048, 12 offered,
8 residents):

    ITL = 49.06 ms + 12.29 ms fixed + 4.45 ms per rider     (79% of the transfer hides)

| riders | ITL (ms) | tok/s |
|---|---|---|
| 0 | 49.06 | 57.3 |
| 1 | 65.98 | 54.8 |
| 2 | 69.92 | 55.8 |
| 3 | 74.89 | **57.1** |

Empirical break-even **3–4 riders**; B1's roofline admits **2** at ctx 2048
(1.107 GB budget ÷ 0.537 GB per rider). **On PCIe we are one rider short of
parity** — and the residual 12.29 ms is worth almost exactly one rider.

This is worth more to the paper than a win would be, because it is a *model with
a measured crossing point*. It converts "GH200 will be better" from a hope into
an arithmetic prediction the GH200 run either confirms or falsifies.

### 2.4 The fixed cost was located, halved twice, and then explained

37.02 → 24.78 → 20.05 → 16.10 → **12.29 ms**, four sites, one species:
loop-invariant work executed inside the per-layer loop. Headline moved
−11.3% → −5.8% → −2.8% → **−2.3%**, control reproducing to 0.1 tok/s over five
runs. A fresh operator trace then showed `cudaStreamSynchronize` gone from the
table entirely and **no dominant remaining term** — the residual is ~640 extra
kernel launches per step (10 ops × 64 layers), i.e. per-layer launch overhead.

**This is a floor of the current design, not a defect**, and saying so is the
right call for GH200: the residual is machine-independent, so it will not
corrupt the C2C measurement. Getting past it needs batching *across* layers
(the `k_window` slots already exist for it) — future work, honestly labelled.

Methodologically this is a *contribution to the implementation section*: two
rounds of aggregate phase timers produced two wrong diagnoses; only an
operator-level diff (0-rider vs 1-rider) found it. Worth a paragraph — it is a
real lesson about profiling async GPU pipelines.

### 2.5 P1 (integrate DirectKV) is dead on our hardware, and that is fine

DirectKV's kernel is SM90a CUTLASS/CuTe source and this cluster has no `nvcc`.
It becomes possible only on the rented GH200, where it is §4.5 — *optional, and
framed as characterisation, never a takedown*. The paper does not depend on it.
Our `StagedHostBackend` is the substrate; `KVTierBackend` already makes it
swappable, which is exactly the abstraction a reviewer will ask for.

### 2.6 Retractions that must not resurface

- **`steady_concurrency` is a broken metric.** It is a mean over sampled steps,
  so a *slower* arm holds requests longer and scores *higher*. It produced the
  retracted "+12% concurrency" claim. It must not appear in any figure, table,
  or sentence. `endpoint_comparison.py`'s headline metric still has this flaw —
  fix it before the GH200 run.
- **Token identity under near-ties.** bf16 attention reductions are batch-order
  dependent; vanilla vLLM gives different text for the same prompt at
  concurrency 1 vs 6. Sound gates are (a) **serial** — batch of one on both
  sides, and (b) concurrent comparison restricted to prompts screened as
  batching-stable. State the method in the paper; a reviewer who knows this will
  otherwise assume we do not.
- **B1's `b_tok` was 2× wrong** and the hide budget was a hand-set constant.
  Both fixed (derived from the model; measured from rider-free steps). Any
  number from before 2026-09-04 that depended on admission counts is suspect —
  regenerate, do not reuse.

---

## 3. The claim ladder — what we can assert, and how strongly

| # | Claim | Evidence today | Strength | Needs GH200? |
|---|---|---|---|---|
| C1 | Capacity, not compute, caps decode batch | Bottlenecks (arXiv 2601.19910) + our pool occupancy | cited + measured | no |
| C2 | `gain ≈ 1 + (X/BW)(HBM/C)`; context cancels; applicability gate | P4, 54 configs, 2 models, R²=0.975 | **validated on A100** | re-measure eff_kv |
| C3 | `eff_kv` degrades 0.77 → 0.46 with working set | P4 n=3 repeats, error bars | measured; **corrects the field's rooflines** | FA-3 check is the top downside risk |
| C4 | The model explains DirectKV's own 8B-vs-OPT-30B inversion | their published numbers + our law | strong, free | no |
| C5 | Riders are additive: 48/96 concurrent vs all-HBM 7/8 at fixed pool | `hybrid_preload_capacity.py` | measured, **synthetic KV, capacity only** | no |
| C6 | 79% of the transfer hides; cost = 12.29 fixed + 4.45/rider | `rider_hide_efficiency.py`, 5 reproduced controls | measured, real KV | no |
| C7 | Break-even is 3–4 riders on PCIe; we admit 2 | same | measured | no |
| C8 | The residual fixed cost is per-layer launch overhead, not a defect | operator trace, current HEAD | measured | no |
| C9 | Lossless: bitwise / token-identical | `rungd_host_pool_test`, `hybrid_rungd_real`, `prefill_stash_roundtrip` (max\|diff\|=0) | strong | re-run on Hopper |
| C10 | **Mixture > both endpoints at matched SLO** | modelled: 7 / 5 / 12 for 70B fp8 @8K | **UNMEASURED — this is the rental** | **yes** |
| C11 | Policy is load-bearing: ours − uniform-offload is material | not run | **UNMEASURED** | **yes** |
| C12 | The adaptive probe flips harmful → beneficial on C2C | measured *harmful* on PCIe (correctly reverted) | prediction | **yes** |

**C1–C9 are 65% of the paper and are writable today.** C10–C12 gate only the
headline of §5. That split is what makes a 55-day schedule feasible.

---

## 4. Backward schedule to Oct 30

### W0 — Sep 5–12: write what GH200 cannot change; book the box

- [ ] **`PAPER_OUTLINE.md` → v2.** Fold in §2 above; retire the stale asset
      table and P1; restate the system model as P/D decode node.
- [ ] **LaTeX scaffold** in `paper/tex/` with the MLSys 2027 style, section
      skeleton, and a `data/` → figure build (`make figs`).
- [ ] **§3 analytical spine — written to completion.** It is done work: the gain
      law, context-cancels, the applicability gate, the eff_kv curve, and the
      P4 validation. This is the section that survives every branch below.
- [ ] **§7 related work — written to completion.** Seven papers read; the
      positioning table in `PAPER_OUTLINE.md` §4 is already near-final prose.
- [ ] **Freeze the A100 dataset.** One clean re-run on current HEAD of
      `rider_hide_efficiency.py`, `hybrid_preload_capacity.py`,
      `pd_decode_goodput.py`, `p4_itl_model_validation.py` → `paper/data/*.json`
      with a manifest recording commit SHA, node, and flags. **Figures must be
      generated from that frozen set, never scraped from logs.** Several numbers
      in the repo predate the b_tok and budget fixes.
- [ ] **BEN: book GH200.** 1× GH200-96, **CUDA `devel` image — verify `nvcc` in
      minute one**, ~5–7 days, disk for a 72.7 GB fp8 checkpoint plus the pinned
      host pool. ~$250–450 at 60–76 GPU-h.
- [ ] **BEN: pre-stage the checkpoint** to object storage. Downloading 72.7 GB on
      a metered box is wasted money.

### W1 — Sep 12–19: design + implementation sections; GH200 pre-flight

- [ ] **§4 Design** — tier assignment, B1 seq-aware roofline, B2 closed-loop
      guardrail, the adaptive probe, migration with hysteresis.
- [ ] **§5 Implementation** — host-block-id pool, block-major split K/V mirrors,
      zero-block preload admission + the head-of-line fix, reclaim-on-worker-ack,
      the two-pass attention, and the loop-invariant hoists (§2.4). Include the
      profiling-methodology paragraph.
- [ ] **Figures F1–F5** (A100-only, see §5) generated from the frozen dataset.
- [ ] **Close the GH200 pre-flight checklist** — several boxes in
      `GH200_EXPERIMENT_PLAN.md` §2 are still unticked: dry-run every experiment
      script on A100 with Qwen-0.5B, pin the environment, fix
      `endpoint_comparison.py`'s `steady_concurrency` headline metric, and
      decide the SLO grid in advance. **Rental time is for measuring, not
      debugging.**

### W2–W3 — Sep 19 – Oct 3: the GH200 window

Execute `GH200_EXPERIMENT_PLAN.md` in its stated priority order. Bring-up is
time-boxed at **8 GPU-h, hard**; overrun means release the instance and fix
offline.

1. **4.1 endpoint comparison** (24 GPU-h) — the decision. all-HBM / all-host /
   mixture, SLO grid {30, 40, 50, 75, 100} ms, contexts {4K, 8K, 16K},
   token-identical verification in every arm.
2. **4.2 re-parameterise under FA-3** (8 GPU-h) — eff_w and eff_kv vs working
   set. Runs early because it is the top *downside* risk.
3. **4.3 policy ablation** (16 GPU-h) — uniform-offload vs ours on one
   substrate. This is C11, and it is what makes it a *policy* paper.
4. **4.4 negative control, Llama-8B** (4 GPU-h) — a model that predicts its own
   failure cases is far more credible than one that only predicts wins.
5. **4.5 DirectKV head-to-head** (16 GPU-h) — only if time remains;
   characterisation framing only.

**GATE (end of window):** mixture > max(all-HBM, all-host) by **≥15%** at
matched SLO?

### W4 — Oct 3–10: §5 Evaluation, written against real numbers

Branch on the gate:

- **PASS → the policy paper.** "Request-level KV tiering for capacity-bound
  decode": the model, the policy, and the measured 3-arm result.
- **FAIL → the crossover paper**, which is *stronger than the Aug-11 fallback
  was*. We would then have the cost law measured on **two interconnect
  generations** with a predicted crossing point and a hardware test of it —
  "when does host-KV offload actually pay?", answered rather than projected.
  C5–C8 carry it. Target stays MLSys; ISPASS/IISWC becomes the safety net.

Either branch uses the same §1–§4 and §6–§8. **Only §5 forks**, which is why the
writing order above is deliberate.

### W5 — Oct 10–17: intro, background, discussion; complete draft v1
### W6 — Oct 17–24: internal review + red-team pass (§6); fix; re-run what it kills
### W7 — Oct 24–30: polish, abstract, page limit, artifact appendix, submit

**Slack:** none in W2–W3. If the rental has not *started* by **Oct 3**, roll to
OSDI '27 (Dec 8) rather than compressing the evaluation.

---

## 5. Figure plan

A100-only figures (buildable in W1, survive every branch):

- **F1 — Capacity binds.** GPU utilisation vs KV occupancy at the operating
  point; the batch is capped by bytes, not FLOPs. (Motivation, §1.)
- **F2 — The gain law.** `1 + (X/BW)(HBM/C)` surface with 8B / 32B / 70B placed
  on it, and DirectKV's published inversion annotated. (§3, C2+C4.)
- **F3 — eff_kv vs KV working set**, 0.77 → 0.46 with n=3 error bars. The
  correction to the field's rooflines. (§3, C3.)
- **F4 — Rider cost decomposition.** ITL vs rider count: 49.06 intercept,
  12.29 fixed step, 4.45/rider slope, 79% hidden, with the tok/s recovery
  overlaid and the empirical 3–4 rider break-even marked. (§5, C6+C7.)
- **F5 — Riders are additive.** Peak running vs offered, mixture vs all-HBM
  (48/96 vs 7/8) at fixed pool. Caption carries the synthetic-KV caveat.
  (§5, C5.)

GH200 figures:

- **F6 — Endpoint comparison.** Requests served at fixed SLO, 3 arms × SLO grid.
  The 30 ms point should show *no* benefit — a prediction, and worth as much as
  the positive result. (C10.)
- **F7 — Policy ablation.** Ours − uniform-offload on one substrate. (C11.)
- **F8 — THE CROSSOVER.** Per-rider marginal cost vs interconnect bandwidth:
  PCIe measured (4.45 ms), C2C measured, and the model line through both, with
  the break-even rider count as a second axis. **This is the money figure** — it
  turns two point measurements into a law with a validated crossing, and it is
  what a reviewer will remember. Design F4 now so F8 is a one-point extension.

---

## 6. Reviewer attack surface — answer each with a measurement, not an argument

Red-team the draft against this list in W6. Each item names its answer.

1. **"Your riders skip prefill — you are measuring a decode node with free
   input."** → State the P/D system model up front (§2.1). The handover is
   *built* and bitwise-proven (`prefill_stash_roundtrip.py`, max|diff|=0 across
   block boundaries, unaligned lengths, and after the producer's extent is
   recycled). **Measure and report the handover cost** rather than assuming it
   away. This is the one that decides the paper's reception.
2. **"48 vs 8 uses synthetic KV."** → Two claims, two experiments, never mixed
   in one sentence: capacity (synthetic is fine — it is an allocator property)
   and goodput (real KV, `pd_decode_goodput.py`).
3. **"You are a policy on someone else's mechanism."** → The spine ablation
   (§4.3 / C11), on one substrate. If it is not run, this criticism is correct.
4. **"The gain shrinks as HBM grows"** (70B: ×1.71 on GH200-96 → ×1.32 on
   GH200-144). → Answer head-on before a reviewer does: weights are growing
   faster than HBM (MoE, 405B+), so `HBM/C` trends favourably. Structurally the
   same decay we cite against NEO, and we should say so ourselves.
5. **"eff_kv is FA-2 era and sits in your capacity denominator."** → GH200 §4.2.
   If FA-3 closes the gap our projected gain *falls*; this is disclosed as the
   top downside risk, not discovered by a reviewer.
6. **"How can you claim losslessness when bf16 reductions are batch-order
   dependent?"** → State the two-part gate: serial comparison plus a
   batching-stability screen. Our divergences are a strict subset of vLLM's own
   prefix cache's — that measurement exists and belongs in the paper.
7. **"Your headline metric rewards being slow."** → `steady_concurrency` is
   retracted and must not appear anywhere.
8. **"12 ms of unexplained overhead."** → Named: per-layer launch overhead,
   ~640 launches/step, machine-independent, with cross-layer batching as
   labelled future work.
9. **OmniServe differentiation** needs a full paragraph (regime, attention
   placement, label source), not a table row.

---

## 7. Title candidates

1. **"Who Rides Along? Request-Level KV Tiering for Capacity-Bound Decode"** —
   from the outline; still the best if C10 passes.
2. **"One Rider Short: When Host-Memory KV Buys Decode Capacity"** — fits the
   crossover branch, and is honest about PCIe.
3. **"The Policy Layer for Host-Memory KV: Which Requests, How Many, and When
   It Pays"**

---

## 8. Decisions

**SETTLED 2026-09-05 (Ben):**

1. **GH200 window: book for ~Sep 19**, ending ~Oct 3. W0–W1 writing proceeds
   while the box is being provisioned. The Oct-3 roll-forward rule to OSDI '27
   (Dec 8) stands if the window slips.
2. **Venue: MLSys 2027, Oct 30.**
3. **Substrate: Overleaf, MLSys 2027 style.** Working practice, so that the
   traceability we need is not lost to a web editor:
   - `paper/tex/` in git is the **source of truth for prose**; Overleaf syncs
     from it (Overleaf imports a zip, or links a git remote).
   - `paper/data/*.json` + `paper/figs/*.py` stay in git and are **never edited
     in Overleaf**. Figures are built locally and the PDFs uploaded.
   - Every numeric claim in the text carries a `% src:` comment naming the JSON
     file and commit SHA it came from. Given how many numbers here have been
     measured, retracted and re-measured, this is not bureaucracy.

**STILL OPEN:**

4. **Author list and affiliations.**
5. **ShadowKV baseline** — build it, or cite and argue? Recommend
   cite-and-argue given 55 days; it is the lossless-vs-lossy point, not a
   throughput competitor we expect to beat.
