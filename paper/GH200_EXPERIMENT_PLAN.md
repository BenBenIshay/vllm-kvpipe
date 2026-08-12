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

- [ ] **Verify the fp8 70B checkpoint loads in our fork.** Cannot be fully
      validated on A100 (SM80 has no fp8 compute; vLLM may fall back to Marlin),
      so treat this as the top bring-up risk and have the TP=2 fallback ready.
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
- [ ] **Confirm nvcc exists on the target image** — our A100 cluster has none,
      which is why FlashInfer and DirectKV were unmeasurable. Without it we lose
      two of five phases.
- [ ] **Decide the SLO grid** in advance (§4.1) so we are not tuning live.

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
| A | all-HBM (tiers off, stock vLLM) | 8 requests |
| B | all-host (every request CPU-tier) | 5 |
| C | **mixture** (our policy) | **13** |

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

## 8. Open decisions (need Ben)

1. **Provider and timing** — and whether an image with `nvcc` is available
   (without it we lose 4.2's FlashInfer arm and all of 4.5).
2. **1× GH200-96 fp8 vs 2× TP=2 fp16** — recommendation is 1×fp8; TP=2 is the
   fallback if the fp8 checkpoint misbehaves.
3. **Is 15% the right kill threshold** for 4.1, or would you accept less given
   the modest ceiling (measured-grounded estimate ×1.39, model ×1.59)?
