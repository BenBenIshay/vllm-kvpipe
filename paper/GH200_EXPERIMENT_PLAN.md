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
| Llama-70B fp8 (confirmatory, last) | 72.7 GB | 19.3 GB | 5.0 | **7.2** | 14.4 | ×1.71 |

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
- **Llama-70B fp8 as headline** — see above. Kept as §4.6, run LAST. Its unique
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

The next two items concern **§4.6, the confirmatory 70B arm only**. They are no
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
- [ ] **Pre-stage the 70B checkpoint** (72.7 GB) — **only if §4.6 is being
      attempted.** Downloading it on a metered box is wasted money. Since 70B is
      no longer the headline, it is legitimate to skip this and drop §4.6.
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

**PROFILED 2026-08-12 (`profile_host_tier.py`).** Per-phase breakdown at
Qwen-32B / 512 ctx, scaling 4 -> 16 requests:

| phase | calls | time | verdict |
|---|---|---|---|
| `seed_layer` (one-time resident->host) | 4.0x | 4.1x | linear, correct |
| `build_step` (per-step compaction) | 1.0x | 3.5x | linear in requests, tiny (12 ms) |
| **`stream_layer`** (per-layer H2D) | 1.0x | **5.1x** | dominant: **48% of wall at n=16** |
| `write_token` | 1.0x | 1.7x | sublinear |

Wall scaled 3.6x for 4x requests. **Nothing pathological at this scale** — the
transfer dominates, which is exactly the PCIe physics (~5.4 GB/step at ~24 GB/s
= ~224 ms/step). So the stall is NOT in the host tier's transfer path.

**TWO HYPOTHESES TESTED AND BOTH WRONG.** (i) B1 admitting only ~5 of 20 with
no fallback tier — disabling B1/B2 did not fix it. (ii) The pool's rolling slots
and pinned mirror sitting outside vLLM's `gpu_memory_utilization` budget —
running at util 0.92 **and** 0.82 both still stalled, so it is not headroom.

**Still unexplained.** The break is somewhere between 16 requests x 512 ctx
(works, 7.5 s) and 20 x 1024 (stalls > 360 s) — only ~5x the work. Leading
remaining suspect: the *resident* KV pool cannot hold 20 x 1024 = 20,480 tokens
of PREFILL, because the host tier frees decode GROWTH but prefill blocks stay
resident; the scheduler then thrashes. That is a real design limit worth
understanding, not just a benchmark artifact.

**Decision: stop chasing it here.** This arm is only meaningful on C2C, is slow
on PCIe by construction, and is not the load-bearing comparison. Item 1
(does the harness detect a difference?) is being closed with `all_hbm` vs
`mixture`, which both work.

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

| arm | configuration | model predicts (Qwen-32B fp16 @8K) |
|---|---|---|
| A | all-HBM (tiers off, stock vLLM) | ~13 requests |
| B | all-host (every request host-tier) | ~7 |
| C | **mixture** (our memory system) | **~19** |

(Updated from 8/5/13 after measuring the real fp8 checkpoint at 72.7 GB, which
leaves C = 19.3 GB rather than the assumed 22 GB.)

**The rider arithmetic behind arm C, written out so a wrong run is obvious
within minutes** (checked 2026-09-05 against the corrected 450 GB/s
per-direction link — see `BLOCKER_PREFILL_RESIDENCY.md`'s correction entry):

| quantity | value |
|---|---|
| KV bytes/token, Qwen-32B (64 layers x 8 kv-heads x 128 x 2 x fp16) | 0.26 MB |
| a host-tier request at 8K ctx, per step | **2.15 GB** |
| hide budget = ITL_0 x X x 0.9 = 34.1 ms x 450 GB/s x 0.9 | **13.8 GB** |
| host-tier requests the link can carry | **~6.9** |
| arm C = B_cap + host-tier = 13.0 + 6.9 | **~19** |

The 13 / 7 / 19 table reproduces from `paper/model/killer_v2.py` as committed
(`analyze(S=8192, GH200, slo=50, stage=0)`), which is the gate a table has to
pass before it goes in the paper.

**Use ~7 host-tier requests as the live sanity check.** If B1 admits ~2, the
link constant is still 25.1 somewhere -- run `gh200_preflight.py` and pass the
MEASURED `--hybrid-pcie-gbps`. If it admits ~30, `b_tok` is wrong for a
64-layer model.
Either way, stop and fix before spending hours on arms that cannot mean
anything.

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

**Set `b_hbm_max` = the MEASURED all-HBM ceiling from arm A, with riders on
top.** `BLOCKER_PREFILL_RESIDENCY.md`'s setup-trap #2 says to set it *below*
capacity; that rule is **superseded and now backwards** — it predates zero-block
preload admission, after which riders never own a block and therefore add
rather than displace. Capping the resident cohort below what HBM holds now
handicaps arm C. The surviving half of that trap is the diagnostic: **if the
mixture's tier histogram shows no `cpu` entries, no rider was created and the
arms are identical by construction.** Check the histogram before believing any
comparison.

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

### 4.2c PRE-FLIGHT BLOCKER — the heterogeneous workload  (0 GPU-h, build on A100)

**Found 2026-09-11 while auditing coverage. Without this, §4.3 produces a null
result that means nothing, and we would burn 16 GPU-hours concluding our own
policy is decoration.**

`host_affinity` scores requests on **temporal** signals — idle time since the
last token, and estimated remaining generation — because context length
provably cannot discriminate between tiers (the cancellation of design rule
(a)). Every harness we have generates **identical** requests: same prompt
length, `max_tokens=out_len` with `min_tokens=out_len` and `ignore_eos`, all
arriving in one burst. So:

- `idle_score` = 0 for every request (nothing ever idles), and
- `rem_score` is identical for every request at the same point in the run.

⇒ `host_affinity` returns **the same value for everyone**, so the `scored` arm
is byte-for-byte the `arrival` arm. **The ablation cannot show a difference
that exists, because the workload contains nothing to choose between.**

**What to build (A100, no rental time):** a generator with
1. **mixed remaining generation** — e.g. `max_tokens` drawn from a wide
   distribution rather than fixed, so "this request will hold memory for a long
   time" is true of some requests and not others;
2. **multi-turn idle gaps** — requests that stop generating, wait, and resume,
   which is the case the policy is actually designed for (agentic and
   conversational sessions) and the only one where holding memory and consuming
   bandwidth come apart;
3. **staggered arrivals** (Poisson), since a burst admits everyone before any
   gate engages — a trap that has already produced two inconclusive runs.

**Gate before §4.3 runs:** on the new workload, `host_affinity` must have a
spread (report its min/median/max across the running set). If the spread is
near zero the workload is still inert and §4.3 must not be run.

This is also what the paper needs independently: §6.1 now states that a
homogeneous workload makes scored placement degenerate to arrival *by
construction*, so a reviewer will look for the heterogeneous one.

### 4.3 Policy ablation — is the policy load-bearing?  (16 GPU-h)

On one substrate:
- **uniform offload vs our tiered admission** — the `(c) − (b)` comparison that
  justifies a *policy* paper at all.
- `VLLM_KVPIPE_TIER_POLICY=arrival` vs `scored`.
- **Adaptive probe verdict.** On PCIe it measured *harmful* and correctly
  reverted. Here it should measure *beneficial* and keep the demotion. If it
  still says harmful, our host path has a fixed per-step overhead that survives
  a fast link — important either way.
- Burst recovery end-to-end, which on A100 could only converge by doing nothing.

**MATCH THE BYTES, NOT THE REQUEST COUNT (added 2026-09-11).** The `uniform_*`
arms cap `b_hbm_max` at a fraction of the offered load, which fixes how many
requests are host-tier but *not* how much they move — the policies select
requests of different contexts, so part of any delta would really be a
difference in offload ratio. `policy_ablation.py` now counts bytes on both
directions per arm and prints GB/step; the verdict flags any pair whose traffic
differs by more than 1.25× as **UNMATCHED** rather than reporting a throughput
delta that means something else. **A result where the arms are unmatched is not
reportable** — re-run with `b_hbm_max` adjusted until the traffic lines up.

**Report tok/s, never `steady_concurrency`** — it is a mean over sampled steps,
so a slower arm holds requests longer and scores higher. It produced a claim we
retracted, and it was the headline of this harness's own verdict until
2026-09-11.

### 4.3b The return leg — swap semantics vs streaming  (4 GPU-h)

**Added 2026-09-11.** The paper claims that an offload design whose unit is
*owned* by one side must write that unit back once per generated token, while
one whose device-side copy is a *cache* need not — a factor of two on the
resource that bounds the whole technique (§7, and Pie's swap controller swaps
out "the last layer that finished computation"). That claim currently rests on
reading someone else's protocol. This measures it on ours.

`policy_ablation.py --arms scored,scored_swapback`. The two arms are identical
except that `scored_swapback` sets `VLLM_KVPIPE_RUNGD_WRITEBACK=1`, which writes
each layer's whole rolling slot back to the host mirror — the exact inverse of
the refresh, same runs, same granularity, opposite direction. Policy held fixed,
so the delta is the return leg and nothing else.

What to report: tok/s for both, total link bytes for both, and the ratio. The
prediction is that the swap arm moves ~2× the bytes and loses throughput
accordingly *once the link is the binding constraint* — on C2C, at high rider
counts. **If the two arms perform the same, the return leg is not on the
critical path at our operating point and the §7 claim must be softened to a
statement about traffic rather than performance.** That is a real possible
outcome and the honest version is cheap to state.

**Apparatus validated on A100 (2026-09-11), both levels:**

*Unit* (`writeback_unit.py`): disabled is a true no-op — it does not propagate
even a deliberately corrupted slot; enabled moves exactly the extent the
refresh read (6144 = 6144 bytes); the round trip is lossless.

*Integration* (`policy_ablation.py --arms scored,scored_swapback`,
Qwen2.5-0.5B, ctx 512, 8 offered — `paper_runs/20260911_ablation_smoke.json`):

| arm | tok/s | link in | link back | total |
|---|---|---|---|---|
| streaming | 323.2 | 0.602 GB | 0.000 GB | 0.602 GB |
| swap-back | 300.0 | 0.602 GB | 0.602 GB | **1.205 GB (2.00×)** |

Both arms read identical volume, and the return leg is **exactly 2.00×** total
traffic — the factor-of-two the paper claims for swap semantics, now measured
on our own hardware rather than inferred from someone else's protocol. Token
output is identical across the two arms, so the emulation is lossless end to
end and not only in the unit test.

**Do not quote the −7.2% tok/s as a result.** A 0.5B model over a 0.6 s window
is a smoke test, not a measurement: the step is short enough that the extra
copy is a large fraction of it, and the sample is 25 steps. The *traffic* ratio
is structural and does transfer; the *throughput* cost is what §4.3b measures
at a real operating point.

### 4.4 Negative control — Llama-8B  (4 GPU-h)

Our model says ×1.22 and DirectKV published that 8B does not benefit. Running it
costs little and **a model that predicts its own failure cases is far more
credible than one that only predicts wins.**

### 4.4b Second HBM/C point — Qwen-32B fp8  (6 GPU-h)

**Added 2026-09-06.** Same model, same tokenizer, same KV geometry, half the
weight footprint: C goes 28.0 -> 60.0 GB and HBM/C 3.4 -> 1.6. The gain law
predicts the benefit falls from x1.50 to x1.31.

This is the cleanest validation of the law we can run, because it moves the
weight footprint and **nothing else**. A model-size sweep confounds it with
layer count, KV-head count and architecture. It also converts the paper's
current assertion that weight quantization *hurts* a memory system -- by giving
back the HBM/C headroom -- into a measurement.

Run the same arms as 4.1 at one context (8K) and one SLO; the quantity of
interest is the gain RATIO between the two precisions, not the absolute
numbers, so it does not need the full grid.

### 4.5 If time remains — DirectKV head-to-head  (16 GPU-h)

Only now possible (SM90a). Two things worth knowing:
- staged (ours) vs zero-copy (theirs) at matched offload ratio — is there a
  crossover by offload ratio, as we hypothesised?
- how much of their decode actually takes the fast path, given the
  `S_past % 64 == 0` gate. **Frame any result as characterisation, never as a
  takedown** — the artifact has constraints (no radix cache, head_dim 128, bf16)
  that make a naive head-to-head unfair and unpublishable.

### 4.6 Confirmatory — Llama-70B fp8  (10 GPU-h, OPTIONAL, run LAST)

**Demoted from headline 2026-09-06** (see §1). Run only after 4.1, 4.2, 4.2b
and 4.3 have landed, and only if the fp8 path works on SM90 at bring-up.

Its value is genuine and specific: HBM/C = 5.0 is the configuration our law
most favours (predicted x1.71), and DirectKV never evaluated a 70B-class model,
so it is non-overlapping evidence. But `B_cap = 7.2` at 8K means one request is
14% against a 15% kill threshold, so this arm **cannot carry the decision** --
it can only corroborate one already made at 32B. Report it at 4K as well
(B_cap = 14.4), where the resolution is tolerable.

**Abort conditions, decided in advance so they are not argued about live:** fp8
kernels not taking the fp8 path (`format=naive-quantized` -- confirm it is not
dequantizing); load time over 30 min; or fewer than 6 hours left in the
reservation. TP=2 fp16 is NOT an acceptable substitute here — the all-reduces
confound the metric.

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

## 4.7 Coverage audit — is this enough benchmarking?

Added 2026-09-11, asked as a reviewer would: for each claim the paper makes,
which experiment establishes it, and what is still missing.

| # | Claim in the paper | Experiment | Status |
|---|---|---|---|
| C1 | Capacity, not compute, binds the batch | cited + pool occupancy | done |
| C2 | Capacity law and applicability gate | §4.2 + A100 P4 | A100 done; Hopper pending |
| C3 | `eff_kv` degrades with working set | §4.2 | **downside risk** — re-measure early |
| C4 | The law explains DirectKV's 8B/OPT-30B inversion | their numbers + our law | done, free |
| C5 | Host-tier requests are additive | §4.1 + A100 preload | A100 done (synthetic KV) |
| C6 | Transfer hides; cost = fixed + per-request | §4.2b | A100 done; C2C pending |
| C7 | Break-even count, and the crossover | §4.2b | **needs ≥10 counts, not 3** |
| C8 | Residual fixed cost is launch overhead | operator trace | done |
| C9 | Placement is exact | correctness ladder | done; re-run on Hopper |
| C10 | **Mixture > both endpoints** | **§4.1** | **the rental** |
| C11 | **Policy is load-bearing** | **§4.3** | **blocked on §4.2c workload** |
| C12 | Probe verdict flips on C2C | §4.3 | pending |
| C13 | Weight precision moves the gain as predicted | §4.4b | pending |
| C14 | Swap semantics costs a return leg | §4.3b | unit-tested; pending |

**Gaps we are accepting, and why.** Stating them here so they are choices
rather than oversights, and so §8 can list them as limitations:

- **No real request traces** (ShareGPT, LongBench). Our workloads are
  synthetic. The capacity and cost results do not depend on token content, but
  *SLO attainment under a realistic arrival process* does, and we report it
  only under Poisson arrivals. A reviewer may reasonably ask for a trace; the
  honest answer is that §4.2c's heterogeneous generator is shaped to match the
  *structure* a trace would provide (mixed generation lengths, idle gaps)
  without claiming to be one.
- **KV quantization composition is asserted, not measured.** §3 claims KV
  quantization is neutral to the relative gain because it scales both cohorts
  together. That is a one-line argument and probably right, but it is currently
  untested. Cheap to add if fp8 KV works out of the box; drop the claim if not.
- **No sensitivity sweep on `k`** (rolling slots). `k` sets the hiding window
  and the device footprint, and we fixed it at 4 by construction. A sweep would
  strengthen §4, and costs little — add if hours remain.
- **Host memory pressure is not characterised.** The pinned mirrors are real
  memory sized to `max_num_seqs × ceil(max_model_len / block_size)`; we report
  the number but do not explore the regime where the host is the constraint.
- **No multi-GPU.** Single-device throughout. TP would inject all-reduces into
  exactly the metric we measure, and we say so rather than sweeping it.

**What would make the evaluation insufficient**, as opposed to incomplete:
§4.1 without §4.3 (a result with no attribution), or §4.3 on a homogeneous
workload (an attribution that cannot detect its own subject). Those two are
the load-bearing pair.

## 5. Budget

| phase | GPU-h |
|---|---|
| 0 bring-up | 8 |
| 4.1 endpoint comparison (Qwen-32B fp16) | 24 |
| 4.2 re-parameterisation | 8 |
| **4.2b rider cost curve (F8)** | **6** |
| 4.3 policy ablation (matched bytes) | 16 |
| **4.3b return leg: swap vs streaming** | **4** |
| 4.2c heterogeneous workload (built on A100) | 0 |
| 4.4 negative control (Llama-8B) | 4 |
| **4.4b second HBM/C point (Qwen-32B fp8)** | **6** |
| 4.5 DirectKV (optional) | 16 |
| **4.6 confirmatory 70B (optional, LAST)** | **10** |
| **total** | **76–102** |

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
