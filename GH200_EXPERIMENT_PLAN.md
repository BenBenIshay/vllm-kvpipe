# GH200 experiment plan

Purpose of the rental, in one line: **measure the one claim we cannot measure on
A100 — that the mixture serves more requests than either endpoint.** Everything
else is secondary and must not be allowed to consume the hours that buys.

Governing principle: **rental time is for measuring, not debugging.** Every
script below is written and dry-run on the A100 *before* we rent. Anything that
can fail at bring-up gets a hard time-box and a defined fallback.

---

## 0. What is actually unresolved

| Claim | Status |
|---|---|
| **Mixture > both endpoints (13 / 7 / 19 for Qwen-32B fp16 @8K)** | **MODELLED, never measured. This is the rental.** |
| Adaptive probe verdict flips harmful → beneficial on C2C | measured *harmful* on PCIe; expected to flip |
| eff_kv under FA3 on Hopper | measured 0.46–0.77 on A100/FA2 only. **Sits in the capacity denominator — if FA3 closes the gap, our projected gain FALLS** |
| FA2/Triton crossover at B≈50 | A100 only; FA3 + buildable FlashInfer change the picture |
| DirectKV head-to-head | impossible on A100 (SM90a); becomes possible here |

## 1. Hardware and model choice

**REVISED 2026-09-06. Primary is now Qwen2.5-32B, not Llama-70B.** The previous
recommendation (70B fp8 as the headline) is superseded; 70B is demoted to a
confirmatory arm. Reasoning below, because the reversal matters.

**Recommend: 1× GH200-96, Qwen2.5-32B, evaluated at fp16 and fp8.**

| arm | W | C | HBM/C | B_cap@8K | B_cap@4K | predicted |
|---|---|---|---|---|---|---|
| **Qwen-32B fp16** (primary) | 64 GB | 28.0 GB | 3.4 | **13.0** | 26.1 | ×1.50 |
| **Qwen-32B fp8** (second point) | 32 GB | 60.0 GB | 1.6 | 27.9 | 55.9 | ×1.31 |
| Llama-8B fp16 (negative control) | 16 GB | 76.0 GB | 1.3 | 70.8 | 141.6 | ×1.26 |
| ~~Llama-70B fp8~~ (DROPPED 2026-09-10) | 72.7 GB | 19.3 GB | 5.0 | **7.2** | 14.4 | ×1.71 |

**Why 70B lost the headline — statistical resolution.** At C = 19.3 GB, 70B
gives `B_cap = 7.2` at 8K. The decisive experiment claims the mixture serves
more requests, against a **15% kill threshold**. With a baseline of 7, one
request is 14%: the entire go/no-go decision would turn on resolving a single
request. Qwen-32B fp16 gives 13 at 8K and 26 at 4K, where one request is 4%.
The rental exists to make one measurement; it should be made where it can be
resolved.

**Why the fp8/fp16 pair is a better experiment than a model-size sweep.**
Comparing 8B/32B/70B moves the weight footprint *and* layer count, KV-head
count and architecture — four variables for a law about one. The two Qwen-32B
arms share architecture, tokenizer and KV geometry and **differ only in weight
precision**, so they move `HBM/C` and nothing else. They also test directly the
claim that weight quantization *hurts* a memory system by giving back its
headroom, which the paper currently only asserts.

**Practical:** every harness and correctness gate we have is validated on
Qwen2.5-32B, and it is already in the HF cache. 70B means re-validating the
whole ladder on the meter, against this document's own rule that rental time is
for measuring, not debugging.

Rejected or demoted, with reasons:
- **Llama-70B fp8** — DROPPED entirely on 2026-09-10 (§4.6), not merely
  demoted. Its unique
  value is real: it is the configuration the gain law most favours and the one
  DirectKV never evaluated (they use 8B and OPT-13B/30B). Worth having, not
  worth risking the trip on. Requires: 72.7 GB pre-staged, ~19 min per load,
  and fp8 kernels working on SM90 (unverified).
- **70B fp16 TP=2** — doubles cost and injects ~160 all-reduces/step as a
  confound in exactly the metric we are measuring. Fallback only.
- **Llama-8B** — ×1.26, and DirectKV published that it does not benefit.
  Useless as a headline; genuinely valuable as the *negative control* (§4.4).
- **GH200-144** — larger HBM makes C bigger and the effect *smaller*. For
  Qwen-32B it does not merely shrink, it **switches off** (×1.00 at C = 73 GB,
  below the applicability gate). Actively worse for us, and a clean
  demonstration of the gate if anyone asks.

**Known counter-argument, recorded rather than buried:** a reviewer may want a
70B-class headline because that is what people deploy, and may read 32B as a
convenience choice. The answer is that the fp8/fp16 pair demonstrates the law
across `HBM/C` more rigorously than a single heavy model does — but if §4.6
lands, quote it.

## 2. Pre-flight — do ALL of this before renting

Every item here is free and removes rental risk.

- [ ] **NEW (2026-09-06): source a Qwen2.5-32B fp8 checkpoint** for the second
      HBM/C point. It must share the fp16 model's architecture exactly — same
      64 layers, 8 kv-heads, head_dim 128 — or the pair stops isolating weight
      precision, which is the entire reason for running it. Verify the config
      before renting, the same way the 70B checkpoint was verified.
- [ ] **NEW: confirm Qwen2.5-32B fp16 is staged** (it is in the A100 HF cache;
      it still has to reach the rented box).

The next two items concerned **§4.6, the 70B arm, which is now DROPPED**, so
they are moot and kept only as a record. They were no
longer on the critical path; do not let them consume pre-flight time.

- [x] **fp8 70B checkpoint checked (2026-08-12) — largely de-risked.**
      `RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8`: 80 layers, 8 kv-heads,
      64 q-heads, hidden 8192 — exactly the shape our projections assumed.
      `quant_method=compressed-tensors`, and **our fork recognises it**
      (`CompressedTensorsConfig`). Checked without downloading weights.
      *Still unverifiable here:* whether the fp8 **kernels** run on SM90 — needs
      Hopper. Keep the TP=2 fp16 fallback ready.
      *Note* `format=naive-quantized` (not `float-quantized`) — confirm at
      bring-up that it takes the fp8 path rather than a dequant path.
- [x] **Weight size measured: 72.7 GB across 15 shards**, not the ~70 GB
      assumed. Consequences, worth knowing before we interpret anything:

      | | plan assumed | actual |
      |---|---|---|
      | C (KV space on GH200-96) | 22 GB | **19.3 GB** |
      | B_cap @8K | 8 | **7.2** |
      | HBM/C | 4.4 | **4.96** |

      A smaller C makes the effect **larger** (gain tracks `HBM/C`), so the
      prediction moves in our favour: **7 / 5 / 12** for 70B, and the model now
      reproduces it (§4.6).
- [x] **Pre-staging the 70B checkpoint: NOT NEEDED — §4.6 is dropped.**
      Decided 2026-09-10: skip Llama-70B entirely. Qwen-32B is the primary
      model and every harness and correctness gate is already validated on
      it, so nothing on the meter depends on a 72.7 GB download.
- [x] **Dry-run on A100 with Qwen-0.5B — done 2026-09-05.** `transfer_bw_probe`
      25.16 GB/s vs the 25.1 assumed. `endpoint_comparison` runs three arms x
      two SLOs end to end; its first dry run came back VACUOUS (all-HBM
      absorbed the whole offered load) which a 0.5B KV pool makes
      unavoidable, so it gained `--num-gpu-blocks-override` and the
      non-vacuous verdict path was then validated: all-HBM 450 / all-host 302
      / mixture 438 tok/s at SLO 40. Its headline metric was ALSO replaced --
      it gated on `steady_concurrency`, which rewards a slower arm and is how
      the retracted +12% was produced. `two_source_tax` and `policy_ablation`
      arg-checked.
- [x] **All remaining scripts dry-run end-to-end (2026-09-05).**
      `policy_ablation` writes its JSON; `run_suite.sh` is **9/9 PASS**,
      including `rungd_real`, `restore_roundtrip` and `migrate_roundtrip` —
      the promotion and pool paths the loop-invariant hoists touched, which
      had not been re-run since.
      **Caveat on `p4_itl_model_validation`: it RUNS, but a 0.5B model at
      ctx 512 cannot validate the ITL model.** KV is only 0.6–4.8% of
      bytes/step there and the step is 2.2 ms, so it is launch-bound rather
      than bandwidth-bound and the `implied_eff ≈ 0.22` it reports is an
      artifact, not a measurement. The model is only testable in a regime
      where KV dominates traffic — i.e. the 70B at 8K on GH200. Do not quote
      the dry-run figure.
- [x] ~~Write and dry-run every experiment script on A100~~ with Qwen-0.5B, so
      only the constants change. Scripts needed:
      `endpoint_comparison.py`, `policy_ablation.py`, plus the existing
      `p4_itl_model_validation.py`, `two_source_tax.py`, `transfer_bw_probe.py`,
      `run_suite.sh`.
- [x] **Environment pinned — `paper/ENVIRONMENT_PIN.md`.** python 3.12.13,
      torch 2.9.1+cu128, CUDA 12.8. Notes two traps: `pip list` reports the
      MAIN tree's path even from the worktree (stale dist-info; the finder is
      what imports use), and `ptxas` does exist inside the nvidia wheels --
      the A100 gap is specifically `nvcc`.
- [ ] ~~Pin the environment~~: exact torch/vLLM/CUDA versions, and a build
      script that runs unattended.
- [ ] **Confirm `nvcc` exists on the target image.** `nvcc` is the CUDA
      compiler; vLLM itself ships precompiled kernels and does not need it, but
      **FlashInfer JIT-compiles its attention kernels at runtime** and
      **DirectKV is CUDA source (SM90a CUTLASS/CuTe) that must be compiled**.
      Our A100 cluster has only `ptxas` (a later stage, shipped via pip) and no
      `nvcc`, which is exactly why those two were unmeasurable there.
      **Ask for a CUDA `devel` image, not a `runtime` image**
      (`nvidia/cuda:12.x-devel-*` has it; `*-runtime-*` does not).
      Verify in minute one: `nvcc --version`. Without it we lose 4.2's
      FlashInfer arm and all of 4.5.
- [x] **SLO grid decided — `paper/SLO_GRID_DECISION.md`.** NOT the proposed
      absolute {30,40,50,75,100} ms: an SLO below an arm's own rider-free step
      is unsatisfiable by every arm and measures nothing at full price, and
      the GH200 weights-bound floor alone is ~24 ms. Instead measure the
      all-HBM median ITL in hour 1 (`T0`) and run `T0 x {1.0,1.2,1.5,2.0,3.0}`.

## 2b. RESOLVED — the `all_host` stall does not reproduce (2026-09-11)

**Was a blocker; measured and closed.** `byte_volume_stall.py` walked BOTH arms
up the byte-volume axis on an idle exclusive A100:

| config | GB/step | mixture | all_host |
|---|---|---|---|
| 4 x 1024 | 1.07 | ok 4s | ok 4s |
| 8 x 1024 | 2.15 | ok 7s | ok 7s |
| 12 x 1024 | 3.22 | ok 9s | ok 9s |
| 8 x 2048 | 4.29 | ok 12s | ok 12s |
| **20 x 1024** | **5.37** | **ok 15s** | **ok 14s** |

**20 x 1024 is the exact configuration that did not finish in 900 s on
2026-08-12. It now completes in 14 s** -- a >64x improvement -- and neither arm
has a knee: the ratio to the link's own bound is flat-to-decreasing (5.4x ->
4.3x) as volume rises, which is fixed overhead amortising rather than a defect
emerging.

**Why it is fixed:** the symptom was ~0% GPU utilisation, which is a CPU-bound
launch storm, and the four loop-invariant hoists of 2026-09-04/05 removed
~21 ms/step of exactly that (`int(_crr.max())` per layer, per-layer scatter
index rebuilds, `int(seqused_k.max())` per layer, per-layer `seqused` clone).
No fix was written for 2b itself: there was nothing left to fix, and a
speculative change could not have been verified.

**Two caveats that survive this result.**
1. The highest volume tested is **5.37 GB/step**. The rental's mixture arm at
   8K runs at **~15 GB/step**, ~2.8x higher. Passing at 5.4 does not prove 15 is
   safe -- 4.0 should watch for it on the machine.
2. The "ratio to link" column is a LOWER bound that ignores prefill and compute
   entirely, so 4.3x is not a measure of overhead. It is used here only to show
   the ratio does not grow with volume.

**Consequence for 4.1:** `all_host` is no longer a broken arm, so the objection
that it cannot be learned from is answered on the merits. The separate
objection -- that it is still OUR host tier rather than a published system --
stands, and is what 4.5 exists for.

## 3. Phase 0 — bring-up  (time-box: 8 GPU-hours, HARD)

1. Record ground truth: HBM size, achieved HBM bandwidth, **measured** C2C H2D
   bandwidth via `transfer_bw_probe.py` (do NOT trust the 450 GB/s spec figure —
   on A100 our measured 24.95 vs 25.1 nominal was close, but assume nothing).
2. Build the fork; run `run_suite.sh`.
3. **GATE: the 9/9 suite must pass on Hopper.** If it does not, we are measuring
   a broken port, and every downstream number is meaningless.

**If Phase 0 overruns 8 hours: stop, release the instance, fix offline.** A
failed bring-up is a cheap lesson; a week of debugging on the meter is not.

## 4. The experiments, in priority order

### 4.0 DOES THE FEATURE WORK AT ALL — saturation test  (4 GPU-h, RUN FIRST)

**Added 2026-09-10 at Ben's direction, and it goes before everything.** 4.1 is a
24 GPU-h grid whose every cell is conditional on one thing being true: that
admitting riders raises throughput on this link. Measure that first, cheaply,
with a harness already validated on A100 (`pd_decode_goodput.py`).

1. Saturate the GPU with all-HBM at ctx >= 2K: raise offered load until tok/s
   stops rising. Record tok/s, ITL and GPU utilisation. That is the ceiling
   this machine can do WITHOUT us.
2. Turn the feature on, admit riders on top, and re-measure.

**tok/s must go UP.** On A100 it did not (-2.3%) because only 2 riders fit; here
the budget allows ~7 at 8K and ~28 at 2K. If tok/s does not rise, the transfer
is not hiding and 4.1 cannot produce a win no matter how its grid is set —
stop and diagnose rather than spending the 24 h to learn the same thing slower.

Gate: **tok/s(mixture) > tok/s(all-HBM) at matched or better ITL.** Everything
after this assumes it passed.

### 4.1 THE DECISIVE ONE — mixture vs stock vLLM  (16 GPU-h)

**`all_host` DROPPED as an arm (Ben, 2026-09-11).** Nobody deploys it, its
capacity is arithmetic rather than a discovery (budget / bytes-per-request), and
the system that *is* the all-host endpoint — DirectKV — is a substrate we
compose with, not a rival (its §8.1 says so). Measuring our own all-host arm and
beating it proves nothing about anyone else's system. Two arms:

| arm | configuration | model predicts (Qwen-32B fp16 @8K) |
|---|---|---|
| A | all-HBM (tiers off, **stock vLLM**) | ~13 requests |
| C | **mixture** (our memory system) | **~19** |

The baseline is stock vLLM, which is also Pie's baseline — so the numbers are
comparable to their published 1.9x if the workload matches (see
`COMPETITORS_AND_BENCHMARKS.md`).

**The rider arithmetic behind arm C, so a wrong run is obvious in minutes:**

| quantity | value |
|---|---|
| KV bytes/token, Qwen-32B (64L x 8kv x 128 x 2 x fp16) | 0.26 MB |
| a host-tier request at 8K ctx, per step | **2.15 GB** |
| hide budget = ITL_0 x X x 0.9 = 34.1 ms x 450 GB/s x 0.9 | **13.8 GB** |
| host-tier requests the link can carry | **~6.9** |
| arm C = B_cap + host-tier = 13.0 + 6.9 | **~19** |

**Use ~7 host-tier requests as the live sanity check.** If B1 admits ~2 the link
constant is still 25.1 somewhere — run `gh200_preflight.py` and pass the
MEASURED `--hybrid-pcie-gbps`. If it admits ~30, `b_tok` is wrong for a
64-layer model. Either way stop and fix before spending hours.

- Primary metric: **SLO-attaining goodput**. Secondary: tokens/s, p99 ITL,
  achieved C2C bandwidth (so "the transfer hides" is measured, not inferred).
- SLO grid anchored to the MEASURED baseline, not absolutes — see
  `SLO_GRID_DECISION.md`. A target below the arm's own step time is
  unsatisfiable by every arm and measures nothing at full price.
- Contexts {4K, 8K, 16K}; the ratio is predicted S-independent and P4 confirmed
  that on A100, so confirming on Hopper is nearly free.
- **Token-identical verification in both arms.**

**Kill criteria (restated for two arms):**
- C <= A -> the mixture does not beat stock vLLM. Direction is dead.
- C > A by < 15% -> characterization paper, with this as a section.
- C > A by >= 15% -> this is the paper.

**Set `b_hbm_max` = the MEASURED all-HBM ceiling from arm A, riders on top.**
The old rule to set it *below* capacity is superseded and now backwards: it
predates zero-block preload admission, after which riders never own a block and
therefore add rather than displace. **Diagnostic that survives: if the tier
histogram shows no `cpu` entries, no rider was created and the arms are
identical by construction.** Check it before believing any comparison.

### 4.2 Re-parameterize the model under FA3  (8 GPU-h)

`p4_itl_model_validation.py` on Hopper: eff_w, and **eff_kv vs KV working set**.

This is the top *downside* risk, so it runs early: eff_kv sits in the capacity
denominator, so if FA3 reads paged KV much more efficiently than FA2 did, the
step shortens, the rider budget shrinks, and our projected gain **falls**.
Also re-run the FA2/Triton/**FlashInfer** crossover — on A100 FA2 won below
B≈50 and Triton above; FA3 and a buildable FlashInfer may move that entirely.

### 4.2b The rider cost curve — the crossover figure  (6 GPU-h)

**Added 2026-09-05. It was missing, and figure F8 cannot be drawn without it.**

`rider_hide_efficiency.py`, sweeping rider count on an otherwise identical
engine, and fitting

    ITL(r) = ITL_0 + fixed + r x (bytes_per_rider / B_effective)

This is the one experiment that turns two machines into a *law*: PCIe gives one
point on the bandwidth axis, C2C gives the second, and the model line through
both — with break-even rider count as a second axis — is F8, the figure a
reviewer will actually remember. 4.1 alone yields a point measurement on one
machine.

**Sweep at least `--caps 1,2,3,4,6,8,12,16,24,32`, not the A100 default of
1,2,3.** Two reasons, both learned the hard way:

1. **Three points do not pin the fixed/slope split.** Two runs of the same three
   points on A100 fitted 12.29 ms fixed + 4.45 ms/rider and 9.55 + 4.66 — the
   cap=2 point moved 69.92 → 65.93 between them. The *split* is exactly what
   sets C2C break-even, because the fixed term is the part a faster link does
   **not** shrink. A 30% swing in it is a 30% swing in the answer this whole
   trip is buying.
2. **The A100 fit was made at 1–3 riders while C2C needs 20–50.** Two slopes in
   our own data would only bite at scale: `build_step` (0.45/0.83/1.23 ms at
   1/2/3) and `_issue_runs` (two copies per contiguous run per layer, which
   explodes if rider extents fragment). A ctx-256 sweep to 16 riders on A100
   found both linear, but that is a proxy regime, not this one.

Predicted here: the per-rider term collapses ~18x (25.1 → 450 GB/s), so a rider
at 8K costs ~0.3 ms of link plus whatever is structural, and break-even should
land at the **first or second rider** rather than PCIe's 3–4. **If break-even is
still beyond ~8 riders, the host path has a fixed per-step cost that survives a
fast link** — which is a finding, and one worth more to the paper than a clean
win, because it bounds the technique for everyone.

Also record achieved C2C bandwidth per step, so "the transfer hides" is
measured rather than inferred from ITL.

### 4.3 Policy ablation — is the policy load-bearing?  (16 GPU-h)

On one substrate:
- **uniform offload vs our tiered admission** — this is `(c) − (b)` from the
  original plan, the thing that justifies a *policy* paper.
- `VLLM_KVPIPE_TIER_POLICY=arrival` vs `scored`.
- **Adaptive probe verdict.** On PCIe it measured *harmful* and correctly
  reverted. Here it should measure *beneficial* and keep the demotion. If it
  still says harmful, our host path has a fixed per-step overhead problem that
  survives a fast link — important either way.
- Burst recovery end-to-end, which on A100 could only converge by doing nothing.

### 4.4 Negative control — Llama-8B  (MOVED TO A100, 0 GPU-h here)

**Moved off the rental 2026-09-10.** The model predicts x1.22 -- i.e. no
meaningful benefit -- and DirectKV published that 8B does not benefit either, so
this arm is expected to be null. Ben's objection is right that a predicted null
does not justify metered hours.

It is still worth running, because reviewers do ask whether a model only fits
where it wins, and a model that predicts its own failure cases is worth more
than one that only predicts wins. So run it on the A100, where it is free. The
prediction is weaker there (less bandwidth) but the direction -- no benefit --
is the whole point.

### 4.4b Qwen-32B fp8 — DROPPED (2026-09-10)

Ben's objection, and it is fair: the claim this was to prove -- that quantizing
weights gives back HBM/C headroom and therefore shrinks our benefit -- is
straightforward once the gain law is granted, not counterintuitive, and does not
need 6 GPU-h.

**What we give up, so the paper does not overclaim.** This was the only second
point on the **HBM/C axis**. Without it the gain law is validated across
BANDWIDTH (A100 vs C2C, via 4.2b) and its HBM/C term is **asserted, not
measured**. Section 3 must say so rather than presenting the law as validated in
both dimensions. If a reviewer attacks the HBM/C dependence, this is the
experiment we chose not to run, and the honest answer is that we traded it for
the bandwidth axis.

### 4.5 Head-to-head with Pie — the only real competitor  (12 GPU-h)

**Reframed 2026-09-11.** `COMPETITORS_AND_BENCHMARKS.md` works through the nine
cited systems and concludes the competitor set is **Pie, and only Pie**:
DirectKV is a substrate by its own account, OmniServe computes attention on the
CPU in a different regime, FlexGen is offline, DAK takes the offload ratio as an
input, and NEO/ShadowKV/CALKV shrink KV rather than relocate it.

Pie is the nearest neighbour: same hardware, same goal, **different axis of
partition** — it splits KV by *layer* uniformly across all requests, we split by
*request*. It reports **1.9x over vLLM** on ShareGPT/Alpaca with Poisson
arrivals.

Run **our system on Pie's workload**: ShareGPT and Alpaca, Poisson arrivals,
throughput plus queuing/compute latency reported separately as they do. If Pie
builds on the box, run it directly; if not, compare against their published
figure over a shared vLLM baseline and state plainly that the vLLM versions
differ.

**Prerequisite, and it is not free:** our harnesses use synthetic fixed-length
prompts and closed-loop arrivals. Pie's numbers cannot be compared to ours until
we run their workload shape. `vllm bench serve` already ships ShareGPT, Poisson
`--request-rate` and SLO-aware `--goodput` in our own tree — use it for serving
claims and keep the bespoke harnesses for mechanism claims.

### 4.6 DirectKV — composition, not competition  (8 GPU-h, if time remains)

Not a rival: DirectKV removes the staging buffer beneath us. The question is
whether our policy *composes* with it — riders served through a zero-copy kernel
instead of `StagedHostBackend`. A win here is additive to both papers. Requires
`nvcc` (SM90a CUTLASS/CuTe source).

### 4.6 Llama-70B fp8 — DROPPED (2026-09-10)

Not run. Decided with Ben: skip the 70B entirely and spend the window on the
models the claim actually rests on.

The reasoning was already in this plan before the decision. `B_cap = 7.2` at 8K
means a single request is 14% against a 15% kill threshold, so this arm could
never have carried the decision -- and §1 had already demoted it for exactly
that. What remained was 10 GPU-h for a confirmatory point plus a 72.7 GB
download on a metered box, to re-validate a harness ladder that is verified on
Qwen-32B and nowhere else.

What we give up, stated rather than glossed: HBM/C = 5.0 is the configuration
the law most favours (predicted x1.71), and DirectKV never evaluated a
70B-class model, so this was non-overlapping evidence. **§4.4b replaces it
properly** -- Qwen-32B at fp8 vs fp16 moves HBM/C while holding architecture,
tokenizer and KV geometry fixed, which is a cleaner test of the same law than
changing model families would have been.

Consequence for the budget: 10 GPU-h freed, and the pre-flight no longer needs
the 72.7 GB pre-stage.

### Model-load accounting (matters at 72.7 GB)

Loading the fp8 70B checkpoint is a first-class cost: on this cluster's shared
FS we measured Llama-8B (15 GB) at 237 s -- ~63 MB/s -- which extrapolates to
**~19 min per load**. On local NVMe expect 2-5 min, but plan for the worst.

- `endpoint_comparison.py` sweeps SLOs **inside one engine per arm** (the SLO is
  only a scheduler setting), so 5 SLOs x 3 arms costs **3 loads, not 15**. That
  is the difference between ~1 h and ~5 h of a 24 h phase.
- `policy_ablation.py` cannot do this: its arms differ in ENGINE config
  (`hybrid_b_hbm_max`, tier policy), so **5 loads** are unavoidable (~1.6 h).
- Subprocess isolation is kept **between arms**, where it is genuinely needed --
  vLLM does not release the CUDA context in-process, and at 70B the next
  engine's memory profiling then fails.

## 5. Budget

| phase | GPU-h |
|---|---|
| 0 bring-up | 8 |
| 4.1 endpoint comparison (Qwen-32B fp16) | 24 |
| 4.2 re-parameterisation | 8 |
| **4.2b rider cost curve (F8)** | **6** |
| 4.3 policy ablation | 16 |
| 4.4 negative control (Llama-8B) | 4 |
| **4.4b second HBM/C point (Qwen-32B fp8)** | **6** |
| 4.5 DirectKV (optional) | 16 |
| **4.6 confirmatory 70B (optional, LAST)** | **10** |
| **total** | **72–98** |

Model loads are cheaper than the old plan: Qwen-32B is ~64 GB rather than
72.7 GB, and the fp8 arm ~32 GB, so §4.4b costs little beyond its own runs.
The 10 h for §4.6 includes the 72.7 GB stage-in and two ~19 min loads.

At typical GH200 on-demand rates that is roughly **$250–450**, plus the 70 GB
checkpoint transfer. Call it **a 5–7 day reservation** to absorb model-load time
(each 70B load is 10–20 min and dominates short runs — batch configs per process
rather than restarting).

## 6. Ordering rationale

4.1 first, not 4.2, even though 4.2 de-risks the model. Reason: 4.1 is the
**decision**. If the mixture does not beat both endpoints, the parameterisation
does not matter and we stop. Running the decisive experiment while the instance
is fresh also means a bring-up overrun eats the optional phases, not the
essential one.

**4.2b is the exception to "optional phases go last" (added 2026-09-05).** It
costs 6 hours and it is the only experiment that survives a failed 4.1: if the
mixture does not clear the 15% bar, the fallback is the crossover paper, and
the crossover paper *is* 4.2b plus the A100 curve we already have frozen. Run
it immediately after 4.1, before 4.3, and do not let a bring-up overrun eat it.
Everything after 4.2b is genuinely optional.

## 7. What we are NOT doing

- Not tuning kernels. Not our fight (see `DIRECTION_A_VS_B.md`).
- Not building a NEO/ShadowKV baseline. Different mechanisms; related work.
- Not chasing the eff_kv inefficiency — it turned out to be a backend-selection
  artifact, not headroom.

## 8. Decisions

**AGREED 2026-08-12:**
2. **1× GH200-96.** Model choice REVISED 2026-09-06: **Qwen2.5-32B fp16 is the
   primary** (see §1) with fp8 as the second HBM/C point; Llama-70B fp8 is
   demoted to the optional confirmatory arm §4.6. The original decision was 70B
   fp8 as headline, superseded on statistical-resolution grounds — B_cap = 7
   against a 15% kill threshold means one request decides the trip.
3. **15% is the kill threshold** for 4.1. Mixture must beat both endpoints by
   >=15% at matched SLO, or we fall back to the characterization paper.

**STILL OPEN:**
1. **Provider and timing.** Requirement fixed by the above: a CUDA **devel**
   image (must have `nvcc` — see §2), 1× GH200-96, ~5-7 days, and enough disk
   for a ~70 GB fp8 checkpoint plus the host-tier pinned pool.
