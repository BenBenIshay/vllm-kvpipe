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

- [ ] **SNAPSHOT AND RESTORE THE BUILT ENVIRONMENT — and test the restore.**
      *Added 2026-09-12.* Bring-up is a **per-rental** cost, and the plan's
      8-hour figure is a stop-loss rather than an estimate (expect 1–3 h if the
      build behaves; the aarch64/SM90 build time is unmeasured). Making the
      environment restorable converts that into a **one-time** cost and removes
      the incentive to hold an idle instance purely to avoid rebuilding.

      What has to survive, measured on the A100:

      | | size |
      |---|---|
      | `.venv` (incl. 2.8 GB compiled `.so`) | **8.6 GB** |
      | Qwen2.5-32B fp16 checkpoint | **62 GB** |
      | **total** | **~71 GB** |

      Target: a second rental's bring-up becomes *restore + verify* (~1 h)
      instead of *rebuild + verify* (up to 8).

      **The restore must be verified, not assumed.** A restore that silently
      produces a subtly different environment is worse than rebuilding, because
      every number taken afterwards is wrong and nothing says so. This codebase
      has already been bitten by exactly that class of problem: the editable
      install's finder is a `MetaPathFinder` with an absolute path baked in,
      consulted **before** `sys.path`, so `PYTHONPATH` cannot correct it — a
      restored venv pointing at the wrong tree would run the wrong code while
      looking healthy. `pip list` does not catch it either; it reports the
      *dist-info* path, which is not what imports use.

      Acceptance gate, in this order:
      1. restore onto a clean instance;
      2. `python -c "import vllm; print(vllm.__file__)"` resolves **into the
         restored tree**, not to any other path;
      3. `gh200_bringup.sh` exits 0 (which re-runs the 9/9 suite).

      Only then are numbers from a restored instance comparable to numbers from
      the original one.

- [ ] **Confirm the provider's storage model**, because it decides the rental
      shape. Persistent volume or fast same-region object storage ⇒ the split
      below is clearly right. Ephemeral-only ⇒ prefer one continuous block and
      accept paying for analysis time.

- [ ] **Confirm Qwen2.5-32B fp16 reaches the box** (it is in the A100 HF cache;
      `gh200_bringup.sh` gates on it, so this is a staging task, not a check).

**Rental shape this implies.** Rent **on-demand and releasable**; do not
pre-commit to a reserved block unless the discount is large.

```
bring-up (<=8 h, expect 1-3) -> 4.0b baseline integrity -> 4.0 gate
                                       |
                   FAIL --> release. ~16 h spent, not 82.
                   PASS --> SNAPSHOT, then run 4.1 onward continuously.
```

Cost is not the argument here: at ~$4/GPU-h a second bring-up is $12–32
against a ~$330 trip. The argument is **measurement continuity** — if 4.1 runs
on different silicon from 4.0, the baselines shift underneath every ratio the
paper reports, and a contended node has already cost us a 9.4x error in one
measurement before anyone noticed.

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

### 4.0 FEASIBILITY GATE — one cell of 4.1, on 4.1's metric  (4 GPU-h, RUN FIRST)

**Rewritten 2026-09-12.** The earlier version gated on raw tok/s. That is the
wrong quantity and it disagreed with the experiment it is supposed to protect:
an arm can raise tokens/second while breaking the latency target it was run
under, which is not a win, it is a different question. 4.0 now measures
**exactly what 4.1 measures, at one point of 4.1's grid.**

**The metric is SLO-attaining goodput** --- requests/second that met a
per-output-token latency target --- via `vllm bench serve --goodput tpot:<ms>`,
which ships in our own fork and is what Pie's vLLM baseline is measured with.
Our bespoke `pd_decode_goodput.py` is named for goodput but does not compute
it; it reports tok/s and p99 ITL separately, which is how "tok/s went up" could
have been read as success while the SLO was being missed.

**The single cell.** 4.1 sweeps {3 contexts} x {5 SLOs} x {2 arms}. 4.0 runs
one context, one SLO, two arms:

| | 4.0 | 4.1 |
|---|---|---|
| arms | all-HBM, mixture | same |
| harness | `vllm bench serve` | same |
| primary metric | **SLO-attaining goodput** | same |
| context | 8K only | 4K, 8K, 16K |
| SLO | **T0 x 1.5 only** | T0 x {1.0, 1.2, 1.5, 2.0, 3.0} |
| arrivals | Poisson, one rate above capacity | same, rate swept |
| cost | 4 GPU-h | 16 GPU-h |

`T0` is the all-HBM arm's measured median ITL on this machine, per
`SLO_GRID_DECISION.md` --- never a round number, because a target below an
arm's own step time is unsatisfiable by every arm and measures nothing.

**Why T0 x 1.5 and not another point.** At T0 x 1.0 the mixture is expected to
lose: riders add per-step latency, so the tightest target all-HBM can meet is
the one they are least able to meet. At T0 x 3.0 everything passes and nothing
discriminates. T0 x 1.5 is where the claim actually lives --- enough headroom
for the transfer to hide, tight enough that missing it costs goodput.

**Procedure.**
1. Arm A, all-HBM, tiers off. Sweep Poisson `--request-rate` upward until
   goodput stops rising. Record the peak and the median ITL at it; that median
   is `T0`.
2. Fix the SLO at `tpot:1.5*T0`. Re-run arm A at the rate that maximised its
   goodput.
3. Arm C, mixture, `b_hbm_max` = arm A's measured ceiling, riders on top. Same
   rate, same SLO.

**GATE, and it is a decision rule rather than a direction:**

> **goodput(C) >= 1.10 x goodput(A)**, on **3 repeats**, with **p99 ITL(C) no
> worse than 1.5 x T0** (i.e. the SLO it was run under is actually being met).

10% rather than 4.1's 15%: this is one cell, and the grid's best cell should
beat its typical cell. An effect that cannot clear 10% at the most favourable
realistic point will not clear 15% averaged over a grid that includes
unfavourable ones. Three repeats because our controls reproduce to ~0.1 tok/s,
so a single reading that lands at +3% is noise we can already resolve.

**If the gate fails, stop.** Do not run 4.1. Diagnose with 4.2b (the rider cost
curve), which separates a fixed per-step cost from a per-byte cost that is not
overlapping --- those extrapolate differently and only one of them is fixable
by a faster link.

**What this gate is and is not.** It is a NECESSARY condition, not a sufficient
one: passing means 4.1 is worth 16 hours, not that 4.1 will pass at every point
of its grid. That asymmetry is the point --- it is cheap to fail and honest
about what it proves.

**Sanity checks before believing either arm** (each has produced a plausible
wrong answer before):
- the tier histogram for arm C must contain `cpu` entries. No entries means no
  rider was created and the two arms are the same configuration.
- arm A's peak must be below the offered rate, or the load never bound and
  neither arm was under pressure.
- B1 should admit ~7 riders at 8K. ~2 means the link constant is still 25.1
  somewhere; ~30 means `b_tok` is wrong for a 64-layer model.

### 4.0b IS ARM A A REAL BASELINE?  (0 GPU-h — run on the A100, before renting)

**Added 2026-09-12.** Every number 4.0 and 4.1 report is a ratio against arm A,
"our fork with the tiers off". Our fork adds ~3,900 lines across 22 files in
`vllm/`. If any of it costs time on the *disabled* path — a per-step check, an
attribute lookup in the runner's hot loop, a branch in attention — then arm A
is slower than the stock engine we claim to compare against, and part of the
reported win is self-inflicted. **A handicapped baseline produces perfectly
plausible numbers**, which is why this needs a measurement rather than a
reading of the code.

Not hypothetical here: four loop-invariant computations were found inside the
per-layer loop in 2026-09, worth ~21 ms/step, and only an operator trace found
them. The same species of mistake on the disabled path would be invisible.

`baseline_integrity.sh`. `main` (f417a0e25) is this fork immediately before the
tiering work; our changes are pure Python, so the two versions swap in place
with no rebuild. Interleaved repeats, not blocked, so a thermal or contention
drift cannot be read as a code difference. **Gate: |delta| <= 3%.** If arm A is
slower, either find the overhead or make arm A the *stock tree* and say so in
the paper.

**Why the choice of baseline is load-bearing for comparability.** The two
systems we position against do not use the same one:

| | HBM-resident baseline | their headline |
|---|---|---|
| **Pie** | **vLLM** (Pie is built on vLLM) | 1.9x over vLLM |
| **DirectKV** | **SGLang** | 1.2x, and *slower* than SGLang on 8B |
| **ours** | **vLLM** (tiers off) | — |

Matching Pie is the right call: Pie is our only real competitor
(`COMPETITORS_AND_BENCHMARKS.md`), their number is the one a reviewer will hold
us against, and their baseline has the same shape as ours — their own ablation
baseline is Pie at *expansion = 1*, which is exactly our "tiers off". But it
means our ratio is **not** directly comparable to DirectKV's, because SGLang is
generally the faster HBM-resident engine and a ratio against it is a harder
number to earn. State that in the paper rather than leaving the two 1.2x/1.9x
figures to be read as commensurable.

**If hours allow, add SGLang as a second reference point** (not an arm): run it
HBM-resident on the same workload and report where stock vLLM sits relative to
it. That converts "our baseline is weaker than theirs" from an objection into a
measured offset. Cheap, and it is the first thing a reviewer who knows both
papers will ask.

### 4.1 THE DECISIVE ONE — mixture vs stock vLLM  (16 GPU-h)

**`all_host` DROPPED as an arm (Ben, 2026-09-11).** Nobody deploys it, its
capacity is arithmetic rather than a discovery (budget / bytes-per-request), and
the system that *is* the all-host endpoint — DirectKV — is a substrate we
compose with, not a rival (its §8.1 says so). Measuring our own all-host arm and
beating it proves nothing about anyone else's system. Two arms:

**Same harness and same primary metric as 4.0** (`vllm bench serve
--goodput tpot:<ms>`), so passing 4.0 actually predicts something about 4.1.
4.1 adds the grid: contexts {4K, 8K, 16K} x SLO T0 x {1.0, 1.2, 1.5, 2.0, 3.0}.

| arm | configuration | @8K, util 1.0 | **@8K, util 0.90 (what we run)** |
|---|---|---|---|
| A | all-HBM (tiers off, **stock vLLM**) | 13.0 | **8.6** |
| C | **mixture** (our memory system) | 19.5 | **14.2** |
| | riders | 6.4 | 5.6 |

**Run against the 0.90 column.** The paper's tables quote the util-1.0 figures,
which describe the *device*; `gpu_memory_utilization` 0.90 is what our
harnesses set, and it costs 9.6 GB of KV space --- a third of C --- so capacity
drops 13.0 to 8.6. Setting `--offered` for 13 when the machine does 8.6 does
not break the run, but every sanity check calibrated on 13 will read as a
failure.

*(An earlier draft of this table said 5 and 11. That was the 70B's C = 13.7 GB
applied to the 32B by mistake; 64 GB of weights at util 0.90 leaves 18.4 GB,
not 13.7. Corrected from `killer_v2.py` rather than by hand.)*

The rider count barely moves (6.4 to 5.6) because the link budget depends on
the step time, not on how much of HBM we were allowed to allocate. That is why
the *ratio* improves as C shrinks and the *absolute* capacity falls.

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

**Use ~7 host-tier requests as the live sanity check** (unchanged: the link
budget does not depend on `gpu_mem_util`, only the resident capacity does). If B1 admits ~2 the link
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

**Use §4.5's workload rather than building a bespoke one.** §4.5 already
requires ShareGPT/Alpaca with Poisson arrivals, via `vllm bench serve`, which
ships in our own tree. Real conversational traces carry the variation this gate
needs — generation lengths differ per request, and multi-turn sessions have
genuine gaps — so the two requirements collapse into one piece of work. Build
the arrival/idle plumbing once, verify the `host_affinity` spread on it, and
both §4.3 and §4.5 become runnable. A synthetic generator is the fallback if
the trace path fights the harness, not the first choice.

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

### 4.7 DirectKV — composition, not competition  (8 GPU-h, if time remains)

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

Reconciled 2026-09-12 after merging the two working branches; the dropped arms
are recorded in §4 with what each costs us, not silently removed.

| phase | GPU-h |
|---|---|
| 0 bring-up | 8 |
| **4.0 feasibility gate — one cell of 4.1, on goodput (RUN FIRST)** | **4** |
| 4.0b baseline integrity (arm A vs stock) | 0 (A100) |
| 4.1 the decisive one: mixture vs stock vLLM | 16 |
| 4.2 re-parameterise under FA-3 | 8 |
| 4.2b rider cost curve (the crossover figure) | 6 |
| 4.2c heterogeneous workload | 0 (built on A100) |
| 4.3 policy ablation (matched cap **and** matched bytes) | 16 |
| 4.3b return leg: swap vs streaming | 4 |
| 4.4 negative control, Llama-8B | 0 (moved to A100) |
| 4.4b Qwen-32B fp8 | — dropped |
| **4.5 head-to-head with Pie on Pie's workload** | **12** |
| 4.7 DirectKV composition (if time remains) | 8 |
| 4.6 Llama-70B fp8 | — dropped |
| **total** | **74–82** |

Down from 76–102: dropping 70B removes 10 h and its 72.7 GB pre-stage, dropping
the fp8 pair removes 6 h, and moving the negative control to the A100 removes 4.
Added since: the saturation test (4 h), the return leg (4 h), and the Pie
head-to-head (12 h), which is the one comparison a reviewer will look for.

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
1. **Provider and timing.** The full spec is now `paper/RENTAL_REQUIREMENTS.md`
   — written to be handed to a provider, with every figure derived or measured
   rather than estimated. Summary: 1× GH200-**96** (reject 144, it switches the
   effect off for Qwen-32B), exclusive not MIG, **≥256 GB host RAM** because the
   KV mirrors are pinned and reach 87 GB at 20×16K, ≥250 GB disk, aarch64,
   **on-demand and releasable** so §4.0 can end the trip at ~16 h. CUDA `devel`
   (`nvcc`) is *preferred* rather than required — it costs §4.7 and FlashInfer,
   not the paper. Six questions for the provider are listed there, including the
   one most likely to bite: any limit on **pinned** pages.
