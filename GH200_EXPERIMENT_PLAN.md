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
| **Mixture > both endpoints (8 / 5 / 13 for 70B@8K)** | **MODELLED, never measured. This is the rental.** |
| Adaptive probe verdict flips harmful → beneficial on C2C | measured *harmful* on PCIe; expected to flip |
| eff_kv under FA3 on Hopper | measured 0.46–0.77 on A100/FA2 only. **Sits in the capacity denominator — if FA3 closes the gap, our projected gain FALLS** |
| FA2/Triton crossover at B≈50 | A100 only; FA3 + buildable FlashInfer change the picture |
| DirectKV head-to-head | impossible on A100 (SM90a); becomes possible here |

## 1. Hardware and model choice

**Recommend: 1× GH200-96, Llama-3.1-70B FP8.**

Weights ~70 GB leave C ≈ 22 GB for KV — the exact configuration our gain law
targets (`HBM/C ≈ 4.4`, predicted ×1.59). One GPU keeps the measurement clean.

Rejected alternatives, with reasons:
- **70B fp16 TP=2** — doubles cost and injects ~160 all-reduces/step as a
  confound in exactly the metric we are measuring. Keep as fallback only if the
  fp8 checkpoint misbehaves.
- **Llama-8B** — our own model says ×1.22, and DirectKV published that it does
  not benefit. Useless as the headline; keep only as the *negative control*
  (§4.4), which is genuinely valuable.
- **GH200-144** — larger HBM makes C bigger and the effect *smaller* (70B drops
  to ×1.12). Actively worse for us.

## 2. Pre-flight — do ALL of this before renting

Every item here is free and removes rental risk.

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
      prediction moves slightly in our favour — but the arithmetic in §4.1
      should be restated as **7 / 5 / 12**, not 8 / 5 / 13.
- [ ] **Pre-stage the checkpoint** (~70 GB). Downloading it on a metered rented
      box is wasted money; stage to object storage or plan the download into
      hour 0 explicitly.
- [ ] **Write and dry-run every experiment script on A100** with Qwen-0.5B, so
      only the constants change. Scripts needed:
      `endpoint_comparison.py`, `policy_ablation.py`, plus the existing
      `p4_itl_model_validation.py`, `two_source_tax.py`, `transfer_bw_probe.py`,
      `run_suite.sh`.
- [ ] **Pin the environment**: exact torch/vLLM/CUDA versions, and a build
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
- [ ] **Decide the SLO grid** in advance (§4.1) so we are not tuning live.

## 2b. KNOWN OPEN ISSUE — the `all_host` arm stalls at scale (2026-08-12)

Found during pre-flight, **not yet resolved**. Must be fixed or scoped before
4.1 is trusted, because `all_host` is one of the three decisive arms.

**Symptom.** Qwen-32B, 20 requests x 1024 ctx on A100: `all_hbm` completes in
**45 s**; `all_host` does not finish in **900 s** at ~0% GPU utilisation. At
Qwen-0.5B the same arm completes in 1.6 s, so it is scale-dependent, not a
logic error in the arm's configuration.

**First hypothesis, tested and WRONG.** With `hybrid_b_hbm_max=0` every request
must be host-tier, and B1's transfer budget admitted only ~5 of 20 (268 MB each
against a 1.5 GB budget) with no fallback tier — so the rest would queue
forever. Disabling B1 and B2 for this arm (correct modelling anyway: DirectKV
does not gate admission) **did not fix it**.

**Remaining suspects, in order:**
1. **Seeding cost.** `seed_req` does a D2H `index_select(...).to("cpu")` per
   layer per request. For 20 requests x 64 layers that is ~5.4 GB of *unpinned*
   D2H, one time — minutes, not seconds.
2. **Per-step transfer volume.** The rolling window streams the whole active set
   every layer: ~84 MB/layer x 64 layers = **5.4 GB per decode step** over
   ~24 GB/s PCIe = ~224 ms/step *at best*. That is the A100 physics, and it is
   precisely why this arm is only meaningful on C2C (18x faster).
3. Pinned-host allocation of the 5.8 GB mirror at engine build.

**RESOLVED (partly): it is NOT a deadlock.** The smaller probe — 6 requests,
512 ctx — completes in **3.2 s**. So the arm works; it scales badly. But the
scaling is *superlinear and still unexplained*: 20x1024 is ~13x the work of
6x512, which should be ~40 s, yet it exceeded 900 s. Suspect #1 (one-time
seeding: 1280 separate unpinned D2H copies, each an implicit sync) and #2
(per-step volume) do not obviously produce a >20x superlinear factor. **This
needs one focused profiling pass before 4.1 is trusted.**

Note the arm is *expected* to be slow on PCIe — 5.4 GB/step over ~24 GB/s is
~224 ms/step at best. On C2C that same step is ~12 ms, so the regime that makes
this arm painful here is exactly the one the rental removes.

**Mitigation regardless:** `--arm-timeout-s` (default 3600) now fails a stalled
arm fast instead of silently burning metered hours. That watchdog worked.

**If unresolved at bring-up:** run 4.1 as `all_hbm` vs `mixture` only, and
report `all_host` from the model plus DirectKV's published behaviour rather than
from a broken measurement. The `mixture` vs `all_hbm` comparison is the one that
carries the paper; `all_host` mainly establishes that DirectKV leaves HBM idle,
which their own design makes true by construction.

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

### 4.1 THE DECISIVE ONE — endpoint comparison  (24 GPU-h)

The whole point of the trip. Same model, same HBM, same workload, **same
substrate** (our `StagedHostBackend`) — so this is a *policy* comparison, not a
mechanism comparison.

| arm | configuration | model predicts (70B@8K) |
|---|---|---|
| A | all-HBM (tiers off, stock vLLM) | ~7 requests |
| B | all-host (every request CPU-tier) | ~5 |
| C | **mixture** (our policy) | **~12** |

(Updated from 8/5/13 after measuring the real fp8 checkpoint at 72.7 GB, which
leaves C = 19.3 GB rather than the assumed 22 GB.)

- Primary metric: **requests served at a fixed ITL SLO**. Secondary: tokens/s,
  p99 ITL, achieved C2C bandwidth (to confirm the transfer actually hides).
- SLO grid fixed in advance: **{30, 40, 50, 75, 100} ms**. Our gate says
  capacity only binds above the full-pool ITL (~32 ms predicted here), so the
  30 ms point should show *no* benefit — that is a prediction, and worth as much
  as the positive result.
- Contexts {4K, 8K, 16K} — our rule says the *ratio* is S-independent, which
  P4 validated on A100. Confirming it on Hopper is nearly free here.
- **Token-identical verification in every arm**, not just arm C.

**Kill criteria:**
- C ≤ max(A, B) → **direction A is dead.** Write the characterization paper.
- C > both but by < 15% → fall back to characterization, with this as a section.
- C > both by ≥ 15% → proceed; this is the paper.

### 4.2 Re-parameterize the model under FA3  (8 GPU-h)

`p4_itl_model_validation.py` on Hopper: eff_w, and **eff_kv vs KV working set**.

This is the top *downside* risk, so it runs early: eff_kv sits in the capacity
denominator, so if FA3 reads paged KV much more efficiently than FA2 did, the
step shortens, the rider budget shrinks, and our projected gain **falls**.
Also re-run the FA2/Triton/**FlashInfer** crossover — on A100 FA2 won below
B≈50 and Triton above; FA3 and a buildable FlashInfer may move that entirely.

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

### 4.4 Negative control — Llama-8B  (4 GPU-h)

Our model says ×1.22 and DirectKV published that 8B does not benefit. Running it
costs little and **a model that predicts its own failure cases is far more
credible than one that only predicts wins.**

### 4.5 If time remains — DirectKV head-to-head  (16 GPU-h)

Only now possible (SM90a). Two things worth knowing:
- staged (ours) vs zero-copy (theirs) at matched offload ratio — is there a
  crossover by offload ratio, as we hypothesised?
- how much of their decode actually takes the fast path, given the
  `S_past % 64 == 0` gate. **Frame any result as characterisation, never as a
  takedown** — the artifact has constraints (no radix cache, head_dim 128, bf16)
  that make a naive head-to-head unfair and unpublishable.

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
| 4.1 endpoint comparison | 24 |
| 4.2 re-parameterisation | 8 |
| 4.3 policy ablation | 16 |
| 4.4 negative control | 4 |
| 4.5 DirectKV (optional) | 16 |
| **total** | **60–76** |

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

## 7. What we are NOT doing

- Not tuning kernels. Not our fight (see `DIRECTION_A_VS_B.md`).
- Not building a NEO/ShadowKV baseline. Different mechanisms; related work.
- Not chasing the eff_kv inefficiency — it turned out to be a backend-selection
  artifact, not headroom.

## 8. Decisions

**AGREED 2026-08-12:**
2. **1× GH200-96 with Llama-70B FP8.** TP=2 fp16 is the fallback only if the
   fp8 checkpoint misbehaves.
3. **15% is the kill threshold** for 4.1. Mixture must beat both endpoints by
   >=15% at matched SLO, or we fall back to the characterization paper.

**STILL OPEN:**
1. **Provider and timing.** Requirement fixed by the above: a CUDA **devel**
   image (must have `nvcc` — see §2), 1× GH200-96, ~5-7 days, and enough disk
   for a ~70 GB fp8 checkpoint plus the host-tier pinned pool.
