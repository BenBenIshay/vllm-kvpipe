# Overhead characterisation: how `c` scales, and whether a faster link buys throughput

> ## Bottom line, 2026-10-05
>
> 1. **A faster interconnect does buy throughput, but only after two host-side
>    fixes, and the link was never what was stopping it.** T1 took `c` from 5.173
>    to 1.259 ms per rider and T2 took `F` from 13.44 to 5.90, turning the target
>    from *never pays at any bandwidth* into **paying from 2.8 riders at 1.06×**.
> 2. **`c` is almost entirely transfer, and the side-stream overlap already hides
>    92.6% of it.** `c_link` = 0.0744 is the fraction that *fails* to hide, not a
>    small cost share. The overlap is **load-bearing**: `c` = 0.92 with it against
>    **7.20** without, which fails the gate by more than 2×. That leaves much less
>    launch-side headroom than the earlier sections imply.
> 3. **`c` was fitted across the knee and is ~2.5× smaller than reported — see
>    §0b.** Re-fitting on arms wholly below the knee gives
>    **`c`(L) = 0.116 + 0.00382·L, context-independent** — **0.42 ms/rider** at
>    the target, not 1.259. Below the knee the rider transfer hides completely
>    (measured: at 1–2 riders the bytes cost *nothing*), so `c` contains only the
>    exposed extra work, which is what it was always supposed to mean.
> 6. **So the gate is not what binds, and the interconnect buys more.** `c` = 0.42
>    against a budget of 3.37 passes with 8× margin at *every* bandwidth. What
>    binds is the **knee** — how many riders can be admitted before the transfer
>    stops hiding — and the knee scales linearly with link bandwidth. Gain at C2C
>    is **1.17×** (was 1.06×) and at a doubled link **1.54×**.
> 7. **Which is why T3, T4 and T5 all failed.** They attacked launch count. With
>    `c` = 0.42 and a budget of 3.37, `c` was never the constraint, so no
>    reduction of it could have changed the verdict.
> 4. **The gain is bought with latency**, and at the target the post-admission ITL
>    (34.9 ms) exceeds the design's own `HBM/BW` precondition (~32 ms).
> 5. **The correctness scare was the test, not the feature.** `hybrid_rungd_real`
>    was not an equality test: *vanilla disagreed with itself*. Serial prompt issue
>    makes both sides reproducible and the pool then matches vanilla exactly. But
>    **"suite 9/9" was never the correctness evidence it was read as** — in the
>    T1/T2 commits or in `B-correctness.tex`. The bitwise tests always were.
>
> §0–§8 describe the system as first characterised. §9 is what was then built,
> §9b–§9d the retarget, the optimisation programme and the floor. The defect note
> below §0 records the correctness investigation, including three conclusions I
> reached and retracted; it is kept in that order because the retractions are the
> part worth learning from.

Final report, campaign M0–M6, branch `overhead-characterisation`.
Substrate: one A100-80 PCIe, Qwen-32B / Llama-8B / Qwen-0.5B, P/D-emulated with
the feature on the decode node, decode purity asserted per step on every
retained point.

---

## 0. The question, and the answer

The question this campaign existed to answer:

> **Does a better interconnect, which lets us admit more riders, end up giving
> better throughput?**

The answer is **yes, but only after two host-side fixes, and at the time this
was measured the interconnect was not what was stopping it.**

> **Both fixes have since been built and measured — see §9.** `c` at the target
> is now **1.259 ms** rather than 5.173, and `F` is **5.90** rather than 13.44,
> so the target has gone from *never pays at any bandwidth* to **paying from 2.8
> riders at 1.06×**. §1–§9 describe the system as characterised, which is the
> baseline those changes are measured against; read §10 for the current state.

> ## ⚠ PROVENANCE DEFECT, found 2026-10-05: read before using any number here
>
> **Every measurement in this campaign was taken with `VLLM_KVPIPE_RUNGD_ASYNC`
> enabled, and that path produces incorrect output.** Same binary, one variable:
>
> ```
>   hybrid_rungd_real.py                     ->  PASS, 3/3 token-identical
>   hybrid_rungd_real.py  RUNGD_ASYNC=1      ->  FAIL, 2/3 diverged
> ```
>
> It hid because the campaign's two arms exercised different paths and neither
> could see it. `run_suite.sh` never sets the variable, so **the suite's 9/9 has
> always been validating the non-async path**, while `rider_cost_curve.py`
> declares `--rungd-async` with `default=True` and `kv_k_window` defaults to 4,
> so `self.k >= 2` and the side stream engages. Every archived artifact under
> `paper_runs/20260930/overhead/` records `async=True, k_window=4` — M0 through
> M10, which is `c`, `F`, and the T1 and T2 before/afters.
>
> **The failure is intermittent**, which is itself the signature of a race rather
> than a logic error: two failures on nova07, a pass on nova06 with the same
> binary and flags. So a single passing run is not evidence of correctness, and
> any check of a fix needs a negative control on the same binary.
>
> ### What it does to the numbers, measured
>
> `F` is untouched and `c` is **7.3× larger** with the async path off. Identical
> least-squares fits of `ΔT = F + c·r` at ctx 512, Qwen-32B, batch 16, T1+T2 in:
>
> | run | async | `F` | `c` |
> |---|---|---|---|
> | `T3_off` | True | 4.98 | 0.883 |
> | `T2_async_a` | True | 4.95 | 0.931 |
> | `T2_async_b` | True | 4.29 | 1.004 |
> | **`ASYNC_off`** | **False** | **4.98** | **7.196** |
>
> That is exactly the right shape: `F` is fixed host cost and overlap cannot
> touch it, while `c` is per-rider transfer and the side stream hides ~88% of it
> behind compute.
>
> **So the async path is not an optional optimisation — the positive result
> depends on it.** With it off, `c` = 7.196 against a budget `T0/B` of 3.08 ms,
> and the gate fails by more than 2×. The feature pays *only* on the overlapped
> path, and that is the path that produces wrong output. An earlier draft of this
> note suggested defaulting async off; that would have destroyed the result and
> is retracted.
>
> ### Mechanism, and the fix
>
> `flush_token_writes` argued its ordering was safe because "nothing prefetches
> across a step boundary". That is where a prefetch is *issued*; what crosses the
> boundary is its *completion*. `stream_layer` consumes the last prefetch with
> `current_stream().wait_event(ev)`, which orders the **device** and returns to
> the host at once, so the host proceeds into the next `build_step` and
> CPU-scatters into mirror rows an in-flight copy-stream DMA is still reading.
>
> The race **needs T2**: before T2 the per-layer mirror write was a blocking D2H
> that synchronised the host on every layer and made the scatter safe by
> accident. T2 did not introduce the hazard — it removed the sync that was
> hiding it.
>
> ### T6 closes the window, and it costs the whole benefit
>
> **T6** orders the deferred scatter after the mirror's *readers* — the H2D
> copies — rather than after the main stream, on the reasoning that at the top of
> a step the newest such DMA was issued a layer before the end of the previous
> step and has long since retired. **Measured, that reasoning is wrong**:
>
> | run | async | T6 | `F` | `c` |
> |---|---|---|---|---|
> | `T6_on` | on | **on** | 4.60 | 7.024 |
> | `T6_fix_a` | on | **on** | 4.15 | 7.037 |
> | **`T6_nofix`** | on | **off** | 4.87 | **0.927** |
> | `ASYNC_off` | off | — | 4.98 | 7.196 |
> | archived (3 runs) | on | off | 4.3–5.0 | 0.883–1.004 |
>
> The first three rows are **one binary, one node, one session**, differing only
> in whether the ordering is applied: **7.6× on `c`**, with `F` flat at 4.15–4.87
> (inside the known ~0.7 ms/hour drift), so the entire effect lands on `c`. The
> control reproduces the archived band at 0.927, which is what makes the bracket
> internally consistent rather than a cross-session comparison that drift could
> explain.
>
> With T6 the overlapped path performs like the *un*-overlapped one. The reason
> is visible in `_launch_prefetch`: it begins with
> `copy_stream.wait_stream(current_stream())`, so every queued copy is gated
> behind main-stream progress and the copy stream therefore *always* has
> outstanding work. That is simultaneously why the hazard counter reads 100% and
> why `copy_stream.synchronize()` is close to a full device sync. The hazard is
> intrinsic to the host running a step ahead of the device, and **any host-side
> ordering pays for the whole gap.**
>
> So T6 stays behind a flag and is not defaulted on.
>
> ### The three numbers decompose exactly, which confirms the mechanism
>
> The harness independently reports the per-rider transfer at ctx 512 as **6.17
> ms** (0.155 GB at 25.1 GB/s). Against that:
>
> | configuration | `c` | reading |
> |---|---|---|
> | async off | 7.196 | transfer 6.17 + host ~1.0 |
> | **async on** | **0.883–1.004** | **host only — the transfer is hidden** |
> | async on + T6 | 7.024–7.037 | transfer + host; the overlap is nullified |
>
> T6's cost per rider is 7.03 − 0.9 = **6.1 ms**, which is the per-rider
> transfer to within 1%. So `copy_stream.synchronize()` makes the host wait for a
> full step's outstanding transfer — precisely the work the overlap was
> deferring — and the three configurations are not three mysteries but one
> quantity appearing or not appearing in the step.
>
> **And this reframes the law in this document.** The fitted `c_link`
> coefficient is 7.44% of nominal link time, i.e. **92.6% of the transfer is
> hidden**. Every coefficient here was fitted on the overlapped path, so §3's
> decomposition, §5's link sweep and §8's verdict all describe *the overlapped
> path* — the one whose correctness is in question. That is why the defect
> blocks the headline claim rather than qualifying it.
>
> One failure mode checked and ruled out: if the overlap were merely *deferring*
> transfer into an unbounded backlog, the step time would understate the
> steady-state cost and `c` = 0.9 would be an artefact. It is not. At 8 riders
> the overlapped step is ~61 ms while the transfer needs 49.4 ms of link time,
> so the copy stream can keep up and the overlap is real.
>
> ### RESOLVED: vanilla disagrees with itself, so the gate was never valid
>
> ```
> [DIFF] vanilla vs vanilla     2 of 3 prompts differ
>       first diff at char 60 of 131 (46% through)
>       run A: ...' statesman, invaded Gaul (modern-day France) in 50'
>       run B: ...' statesman, invaded Gaul in 509 BC. The Romans had'
> [SAME] pool vs pool           all 3 prompts identical
> VERDICT: THE GATE IS NOT AN EQUALITY TEST.
> ```
>
> **Two runs of plain vLLM — no pool, no feature — produce different text.** And
> in the same attempt the pool emitted `"invaded Gaul in 509 BC"`, which is
> precisely what vanilla run B produced: **the pool's output lies inside the set
> of outputs vanilla itself generates.** On this attempt the pool was the *more*
> stable of the two — 3 of 3 self-identical against vanilla's 1 of 3.
>
> Both divergences sit 46–53% through the output, between two fluent
> continuations. That is the flipped-near-tie signature, not corruption.
>
Across three attempts, all with the concurrent issue:
>
> | attempt | vanilla vs vanilla | pool vs pool | pool vs vanilla |
> |---|---|---|---|
> | 1 | **DIFF (2 of 3)** | SAME | **DIFF (2 of 3)** |
> | 2 | SAME | SAME | SAME |
> | 3 | SAME | SAME | SAME |
>
> **The correlation is the clincher.** The only attempt in which the pool
> differed from vanilla is the only attempt in which vanilla differed from
> *itself*. The observed pool-vs-vanilla divergence is therefore fully accounted
> for by vanilla's own non-determinism, with no residual left to attribute to the
> feature.
>
> Two further readings. The self-disagreement is **intermittent** (1 of 3), which
> is what made single green runs worthless throughout this investigation and why
> the negative control passing twice cleared nothing — but one demonstration of
> vanilla ≠ vanilla is enough to show the gate cannot certify equality, and later
> agreements do not un-prove it. And `pool vs pool` was identical **3 of 3**
> against vanilla's 2 of 3, so on this evidence **the pool is the more
> reproducible of the two** — the opposite of the concern that opened the
> investigation.
>
> So every "correctness failure" in this investigation was a comparison of two
> draws from a non-deterministic process. One mechanism accounts for all of it:
> the intermittency, the failure on the async-OFF arm, the negative control
> passing twice, the late divergence position, the identical printed prefixes,
> and why a historical run of "9/9" mostly held.
>
> **What this establishes, and what it does not.** It removes the evidence that
> the feature is incorrect. It does **not** show the feature is correct — that
> still rests on the bitwise tests, which are unaffected and were always the
> stronger half (`build_step_equiv` 18 cases, `whole_extent_equiv` 16,
> `restore_roundtrip`, `host_pool`, and T2's negative control failing 3 of 3 on
> demand).
>
> ### The fix, and how strongly it is supported
>
> Serial issue makes both sides reproducible **and the pool then matches vanilla
> exactly** (`rungd_serial_gate.py`: vanilla 1 vs 2 identical, pool 1 vs 2
> identical, pool vs vanilla identical). The gate is fixed in place.
>
> Re-validated on a clean tree: **6 of 6 passes.** That is consistent with the
> fix but is *not* proof of determinism — at the old gate's observed flake rate
> a run of 6 would come up clean about half the time anyway, so six greens bound
> the rate only loosely from above. The weight is carried by the **mechanism**:
> serial issue pins batch composition, and the self-consistency test showed the
> baseline's non-determinism directly rather than by inference.
>
> **The suspected source** is the harness issuing its three prompts concurrently
> through a `ThreadPoolExecutor`, so batch composition varies run to run and with
> it the reduction order inside attention and the MLP. Greedy decoding then turns
> a last-bit logit difference into a different token. That is a hypothesis about
> the *mechanism*; the *finding* — that the gate is not an equality test — does
> not depend on it, because vanilla's self-disagreement is measured directly.
>
> ### Superseded: the divergence is not async-specific
>
> **The rate study's first arm: `base` (async OFF) = 7 pass, 1 fail.** That is
> the suite's own configuration — the path this document has been treating as
> correct — failing the token-identical test.
>
> One failure in eight gives a point estimate of 12.5%, but the binomial
> confidence interval on 1/8 runs from roughly 0.3% to 53%, so the honest claim
> is **"it fails occasionally"**, not a rate. That is already enough: a gate that
> fails occasionally on a configuration assumed correct cannot certify anything,
> and a historical string of "9/9" is exactly what an occasional flake looks
> like.
>
> So the framing above is wrong where it blames the prefetch. It rested on
> async-ON failing twice and async-OFF passing a handful of times; with more
> samples async-OFF fails too. Three consequences, and they point in a different
> direction from everything above:
>
> 1. **The async path is not implicated as the cause.** Whatever diverges,
>    diverges without it.
> 2. **The suite's 9/9 has been passing partly by luck.** `hybrid_rungd_real` is
>    flaky, so "the suite is green" was never the correctness evidence it was
>    taken for — in this document or in the T1 and T2 commits that cited it.
> 3. **The leading explanation is now numerical, not a race.** The pool sums
>    attention over two KV sources — resident and streamed — in a different order
>    from vanilla, so small floating-point differences are guaranteed by
>    construction. Under greedy decoding a near-tie argmax then flips, and strict
>    token identity fails by chance. That single mechanism accounts for the
>    intermittency, the late divergence position, the identical printed prefixes,
>    and the negative control passing twice — none of which the race hypothesis
>    explained.
>
> If that is right, the headline is not "the feature is incorrect" but **"the
> correctness test cannot distinguish a real fault from floating-point
> tie-breaking, and has not been able to for the whole campaign."** Which is a
> different and more tractable problem: the test needs a tolerance on logits or a
> comparison of KV tensors rather than of emitted text.
>
> ### What still stands, which is more than it first appears
>
> The suite has a **trustworthy half and an affected half**, and they are already
> labelled as such in `run_suite.sh`:
>
> | group | tests | affected? |
> |---|---|---|
> | unit | `controller_unit` | no — pure logic |
> | **mechanism (bitwise)** | `restore_roundtrip`, `host_pool` | **no — compares tensors** |
> | end-to-end (token-identical) | `rungd_real`, `migrate_demote`, `migrate_roundtrip` | **yes — compares emitted text** |
> | closed loop | `burst_recovery` | no — tests recovery, not equality |
>
> Only the token-identical group is in question, and the changes this campaign
> made are not certified by that group alone:
>
> * **T1** rests on `build_step_equiv_test.py` — 18 cases that carry the
>   pre-change algorithm verbatim as the reference and compare **bitwise**,
>   off-GPU. That was always the stronger evidence, and it is untouched.
> * **T2** rests on a bracketed before/after plus a **negative control that
>   failed 3 of 3** when the flush was suppressed. A control that fails on
>   demand is evidence the test is sensitive, which is exactly what a flaky gate
>   lacks.
> * **T5** rests on `whole_extent_equiv_test.py` — 16 checks resolving every
>   block through the block table.
> * **T3's** reversal rests on timing, not identity, and is unaffected.
>
> So the correct statement is narrower than "the campaign's correctness evidence
> is void": **the end-to-end text-identity claim is void, and the component-level
> bitwise claims are intact.** The feature's mechanism is still supported; what is
> no longer supported is the inference from "suite 9/9" to "the engine produces
> identical output", which is the claim the paper's correctness appendix leans on.
>
> What remains true regardless: the coefficients were all fitted on the
> overlapped path, and the overlap is load-bearing (`c` 0.927 against 7.03 when
> ordered, 7.196 with async off).
>
> ### The earlier causal chain, kept for the record, is NOT established
>
> Two things are confirmed: the hazard window exists on every step, and the
> async path diverges intermittently. What is *not* confirmed is that the first
> causes the second, and on closer reading it probably does not.
>
> The row the scatter overwrites holds the **current** token. At step N−1 the
> streamed window only needs tokens up to t−1; token t's KV comes from the
> resident GPU cache, written by the attention kernel. So whatever the in-flight
> prefetch reads in that row is masked out by `seq_len` and never consumed —
> which makes the race, as far as this analysis goes, **benign**.
>
> That leaves the divergence unexplained, and it is the thing that matters. It
> also means the hazard counter, for all its 3456/3456, measured a real window
> around a value nobody reads. Every hypothesis so far — including this one — has
> been tested against single runs, which is worthless for an intermittent fault,
> so the next step is a failure **rate** per configuration before any further
> attribution.
>
> **Current standing of the result:** the feature's positive numbers were all
> taken on a path that produces wrong output, the only known way to make that
> path correct costs the entire benefit, and the root cause is open. That is a
> blocker on the headline claim, not a caveat to it.

### Notation

| | |
|---|---|
| `r` | rider count — requests whose KV lives in host memory and is re-streamed each step |
| `S` | context length in tokens |
| `X` | CPU–GPU link bandwidth |
| `L` | layer count; `b_tok` bytes of KV per token; `bs` KV block size |
| **`T0`** | **the rider-free step time** — the decode step with `B` residents and *zero* riders. Everything is differenced against it: `ΔT(r) = T_step(r) − T0`. Measured 49.29 ms (Qwen-32B, A100, ctx 512); 24.2 ms modelled for the 70B on GH200, where it is weight-read bound. |
| **`B`** | **how many HBM-resident requests fit** — a *capacity*, not a chosen batch size: `(HBM − weights − overhead)/(b_tok·S)`. 7.2 at the target. |
| **`T0/B`** | **the reciprocal of baseline throughput** — the average time the system takes to deliver *one* token counting all `B` residents together, `1/(B/T0)`. Not a per-request latency: `T0` itself is the ITL every resident experiences. It is the budget because a rider buys one extra token per step, and a token is "worth" `T0/B` ms at the baseline rate. See §1. |
| `F`, `c` | the fixed and per-rider parts of `ΔT`: `ΔT = F + c·r` |
| `r*` | the knee, `T0·X/(b_tok·S)` — above it the step *is* the transfer |

Concretely, at the paper's own target — Llama-70B fp8, ctx 8192, GH200-class
HBM, residents set by capacity (7.2, not a chosen 16):

| | `c` ms/rider | vs `T0/B` = 3.37 | best gain |
|---|---|---|---|
| as built, NVLink-C2C 419 GB/s | 5.17 | **fails** | 0.65× |
| as built, **infinite** link | 4.70 | **fails** | 0.71× |
| per-block loop removed + D2H fixed, PCIe4 | 8.63 | fails | 0.88× |
| …PCIe5 64 GB/s | 3.79 | fails | 0.92× |
| …PCIe6 128 GB/s | 2.23 | passes, knee too low | 0.97× |
| …**NVLink-C2C 419 GB/s** | **1.15** | **passes** | **1.21×** |
| …hypothetical 900 GB/s | 0.89 | passes | 1.54× |
| …infinite link | 0.67 | passes | 3.46× |

Three things to read off that table, in order of importance:

1. **As built, no interconnect wins.** An *infinite* link still leaves
   `c` = 4.70 against a budget of 3.37. The feature's verdict today does not
   depend on the link at all, because the link is only 9.2% of `c` at this
   operating point. This is the single most important result of the campaign.
2. **Once the two host artifacts are removed, bandwidth converts into
   throughput monotonically**, 0.88× → 3.46×. The mechanism the feature is
   premised on is real.
3. **C2C is the first bandwidth point that works at all.** PCIe6 already
   satisfies the necessary condition and *still* loses. So what the link buys is
   not primarily the bandwidth term — it is the knee (§5).
4. **Riders do not measurably share `c`** — the per-rider cost is constant
   within the resolution of these sweeps (§2). But **only the 9.2% that is bytes
   on the wire is per-rider by *physics*;** the launches and the mirror write are
   batchable and the block-table rebuild is vectorisable (§3.2b). So sharing is
   an available engineering lever rather than an observed effect, and what it
   would be worth is capped by the knee — the same conclusion as (3), from the
   other side.

Everything below is the derivation and the evidence.

---

## 0b. CORRECTION, 2026-10-05: `c` was fitted across the knee, and is ~2.5× smaller than reported

**Headline: `c` at the target is 0.42 ms/rider, not 1.259, and it is
context-independent. The gate `c < T0/B` is not what binds — the knee is.**

### The error

Every `c` in this report was fitted as one straight line through a whole rider
sweep. At the short contexts the campaign used, the **top of that sweep sits at
the knee.** Below the knee a rider's transfer hides behind compute and costs
nothing; at the knee it stops hiding and lands in the step. A straight line
through both regimes smears the knee cost backwards and reports it as a constant
per-rider charge.

Differencing the `xfer_mode` arms at ctx 512 shows it directly — no fitting:

| riders | cost of the bytes | transfer they need | un-hidden |
|---|---|---|---|
| 1 | **−0.26** | 5.3 | ~0% |
| 2 | **−0.25** | 10.7 | ~0% |
| 4 | 1.44 | 21.4 | 6.7% |
| 8 | 3.50 | 42.8 | 8.2% |

At 1–2 riders the bytes cost **nothing**, exactly as the design intends. The
cost appears only on approach to the knee (9.2 riders here).

### The re-fit

Truncating a sweep to its lowest points does not work — with `F` ≈ 11 ms and
only r = 1,2 surviving, the fit is ill-conditioned and returns a *higher* `c`.
The arms to use are those **wholly below the knee**, which need no truncation,
plus `minbytes` (bytes removed, so no knee exists) for the full-depth point:

| arm | max load | `c` |
|---|---|---|
| `C_L8_c512` (8 layers stream) | 0.14 | 0.358 |
| `C_L16_c512` | 0.26 | 0.455 |
| `xfer_min_c512` (L=64, bytes removed) | — | 0.597 |

```
  clean fit:  c(L) = 0.358 + 0.00382 * L        (pre-T1)
  at L=64 it predicts 0.602; minbytes measures 0.597  -> 0.8% agreement
```

Two independent routes agree, which is the evidence that the cut is in the right
place. The law's `b_per_rider_layer` = 0.00838 is **overstated 2.2×**.

The 0.358 intercept is itself explained: these arms are **pre-T1**, so they still
carry the per-block term (0.245 at ctx 512) plus d2h (0.109) = 0.354. With T1 in,
that intercept collapses to 0.116:

```
  POST-T1:  c(L) = 0.116 + 0.00382 * L      and it does NOT depend on context
            L=64  ->  0.361        L=80  ->  0.422
```

Context-independence is the strongest part of the claim: `minbytes` gives `c` =
0.520 at S=256 and 0.597 at S=512, and after removing the S-scaling per-block
term both reduce to ≈0.35–0.40. **A per-rider cost that does not scale with the
bytes cannot be the transfer.**

### What it does to the projection

| link GB/s | `c` | budget `T0/B` | knee | gain (old) | gain (clean) |
|---|---|---|---|---|---|
| PCIe4 25 | 0.422 | 3.37 | 0.23 | — | 0.83× |
| PCIe6 128 | 0.422 | 3.37 | 1.16 | 0.85× | 0.92× |
| **C2C 419** | **0.422** | 3.37 | 3.78 | 1.06× | **1.166×** |
| **2× C2C 900** | 0.422 | 3.37 | 8.12 | ~1.21× | **1.538×** |

### And the framing of this whole report changes

`c` = 0.42 against a budget of 3.37 is **not close to binding at any
bandwidth.** So §1's claim that "the condition that decides everything" is
`c < T0/B` is wrong as a description of the real system: that gate passes with
8× margin. **What binds is the knee** — how many riders can be admitted before
the transfer stops hiding — which is what §5 already said and which the fitted
coefficient contradicted only because it had the knee folded into it.

> **What still has to be confirmed.** The post-T1 intercept is an *inference*
> (subtracting the measured per-block term), not a direct measurement, and the
> clean fit rests on three points, two of which use the numerically-invalid
> layer knob for timing. **The confirming run is cheap:** ctx 512 with riders
> 0–3 and T1+T2 in, which is wholly below the knee (load ≤ 0.33), ~30 minutes.
> Until then treat `c` = 0.42 as provisional in the same way the old 1.259 was —
> but note the old value is now known to be contaminated, and this one is not
> yet known to be.

### Superseded by the above

An earlier version of this section claimed `c` was "62% link-bound" and that the
interconnect therefore buys *more* than projected. That rested on the same
across-the-knee fit and is withdrawn. The conclusion that the interconnect buys
more does survive, but for a different and better reason: `c` is small and
constant, so the link's whole contribution is the **knee**, and the knee scales
linearly with bandwidth.

---

## 0c. Superseded: `c` is 62% link-bound (withdrawn)

This is the most consequential finding of the session, it reverses the
pessimistic reading of §9d, and **it required no new GPU time** — the decisive
arm was measured in M1 and its implication was never drawn.

### The measurement

`VLLM_KVPIPE_RUNGD_XFER_MODE` has three settings, all keeping every gather,
hook, event and stream ordering identical, and all run with the same timer
state so the *differences* are valid:

| arm | what it does | `c` (ctx 512, L=64) |
|---|---|---|
| `real` | the real path | 1.166 |
| `minbytes` | one block row per copy — **same launches**, almost no bytes | 0.597 |
| `nobus` | views built, **no copy issued at all** | 0.509 |

Differencing them attributes `c` directly, with no fitting:

```
  bytes (link)      = real - minbytes = 0.569   49% of c
  launch issuing    = minbytes - nobus = 0.088    8% of c
  survives nobus                       = 0.509   44% of c
```

### What that does to the law's decomposition

| part | measured | the law said | error |
|---|---|---|---|
| link / bytes | 0.569 | 0.398 | **understated 1.43×** |
| launch issuing | 0.088 | 0.536 | **OVERSTATED 6.1×** |
| per-block | — | 0.245 | (removed by T1) |
| d2h per rider | — | 0.109 | — |
| residual | 0.155 | *no such term* | — |

The reconstruction closes: with T1's per-block term removed, these parts sum to
**0.924** against today's measured **0.913** — **98.8% agreement**.

So today's `c` at ctx 512 is:

| part | share |
|---|---|
| **link / bytes** | **62%** |
| d2h per rider | 12% |
| residual, unidentified | 17% |
| launch issuing | 10% |
| block table | 0.3% |

### Why the fit got it wrong, and why that was predictable

`b_tok = L · 2 · n_kv · d_h · b`, so **the bytes a rider streams are proportional
to `L`.** A transfer-bound per-rider cost therefore has exactly the shape
`r · L` — indistinguishable from a per-rider-per-layer *launch* cost. The two
candidate terms are **collinear in `L`**, the fit split one physical cost across
both, and it put most of it in the wrong one.

### Three consequences

1. **T3, T4 and T5 were all doomed from the start.** Each attacked the launch
   count, which is **10% of `c`**. T5's predicted effect was ~0.08 ms against a
   drift of 0.054 — **its null result was already implied by this M1 arm**, and I
   spent a day of GPU time rediscovering it. §9d's "the launch term is 51% of
   `c`" came from the law, not from a measurement, and the measurement
   contradicting it was already in the archive.
2. **The "mystery term" is 17%, not 51%.** Still unidentified, but a much smaller
   prize, and identifying it is no longer on the critical path.
3. **A faster interconnect buys *more* than this report projected**, because the
   link share is double what the law used:

| link GB/s | `c` as modelled | `c` corrected | gain |
|---|---|---|---|
| PCIe4 25 | 8.74 | 11.80 | 0.76× |
| PCIe5 64 | 3.90 | 4.88 | 0.80× |
| PCIe6 128 | 2.34 | 2.65 | 0.85× |
| **C2C 419** | 1.259 | **1.098** | **1.079×** |
| **2× C2C 900** | 1.004 | **0.733** | **1.430×** |

Being more link-bound cuts both ways — the corrected `c` is *worse* below PCIe6
and *better* at and above C2C — which sharpens rather than softens the original
answer: **the interconnect is the thing that decides this, and C2C is the first
point where it decides in our favour.**

> **What is not revised.** `F` is unaffected: it is fixed cost, and the three
> arms differ only in `c`. The gate `c < T0/B` and the knee are unchanged in
> form. And the coefficients were all taken with the §0 data race open, so the
> absolute magnitudes remain provisional — but this correction rests on
> *differences between arms that share that defect*, which is exactly the class
> of statement the defect does not touch.

---

## 1. The condition that decides everything

Admitting `r` riders on top of `B` residents multiplies served requests by
`(1 + r/B)` and multiplies step time by `(1 + (F + c·r)/T0)`. So

```
    gain = (1 + r/B) / (1 + (F + c*r)/T0)  >  1
      <=>  r * (T0/B - c)  >  F
```

which splits into two conditions of **different kind**:

```
  (1)  c  <  T0/B            NECESSARY, and independent of r.
  (2)  r  >  F/(T0/B - c)    and that r must be reachable, i.e. <= the knee.
```

Condition (1) is the one that has been missing from the discussion, and `T0/B`
needs stating carefully because `T0` *is* the ITL — every one of the `B`
residents emits one token per step, so each experiences `T0` as its inter-token
latency. `T0/B` is therefore **not** a per-request latency. It is

```
    T0/B  =  1 / (B/T0)  =  1 / baseline throughput
```

the average time the system takes to deliver **one token**, counting all `B`
residents together. At the target: `B/T0` = 7.2/24.2 = 298 tokens/s, so
`T0/B` = 3.37 ms per token delivered.

That is why it is the budget a rider must beat. A rider **buys** one extra token
per step, and at the baseline rate a token is worth `T0/B` ms; it **costs** `c`
ms added to *every* step. So the surplus per rider is `T0/B − c`, and it has to
cover `F`:

```
    r·(T0/B − c) > F      i.e.    ΔT/r < T0/B
```

At the target: 3.37 − 1.259 = 2.11 ms of surplus per rider against `F` = 5.90,
so `r ≥ 2.8`. Since `ΔT/r = F/r + c` and `F/r` vanishes as riders are added,
`c < T0/B` is **necessary**: if a rider costs more step time than a token is
worth, no rider count helps — the gain curve never crosses 1 and `F` is
irrelevant.

> **And the gain is bought with latency.** Throughput rises because more requests
> share a longer step, so the ITL rises too. At the target, admitting riders to
> the knee takes the ITL from 24.2 to **34.9 ms (1.44×)** for **1.06×**
> throughput. The design's own precondition is an ITL budget above `HBM/BW`
> (~32 ms here) — and 34.9 is *above* that, so at this operating point the
> feature consumes the headroom that justified it. A deployment needs an ITL
> budget well clear of the rider-free step time, not merely clear of the
> memory-read floor. This is why `c`, not `F`, is the core question: `F` sets *where*
break-even is, but `c` sets *whether there is one*.

> **Condition (1) is exact only if `ΔT` is affine in `r`, and §2 establishes
> that it is** — within the resolution of these sweeps. An earlier draft
> reported a sublinear exponent, which would have demoted condition (1) from a
> gate to a screen, because with α < 1 the left side of the gain condition grows
> linearly while the right grows sublinearly and a large enough cohort always
> wins. **That exponent did not survive scrutiny** (§2): it was set by the fit
> window rather than by the system. Condition (1) is therefore a gate, and
> "never pays at any rider count" is a statement about the measured law rather
> than an artefact of a particular fit.

`T0/B` is not a free parameter. `B` is a capacity, `B = (HBM − weights −
overhead)/(b_tok·S)`, so

```
    T0/B = T0 * b_tok * S / C        -- RISES with context
```

because at longer context fewer residents fit and each is worth more. On
GH200/70B/ctx 8192: `C` = 96 − 72.7 − 4 = 19.3 GB, one request's KV is 2.68 GB,
so `B` = 7.2 and `T0/B` = 24.2/7.2 = **3.37 ms**. Using a hardcoded 16 here was
a real error in an earlier draft and it halved the budget.

---

## 2. Does `c` grow as riders are added? No, and it does not shrink either

Two sub-questions, and earlier drafts of this report got the second one wrong
twice, in opposite directions.

**It does not grow.** `r_max` does not exist in range. The documented >60× stall
does not reproduce: at the stall's own operating point — ctx 1024, r = 20, 5.825
GB/step, *more* bytes than the 5.37 GB/step that once failed to finish in 900 s —
the model predicts 254.02 ms and the measurement is **254.02 ms**. Clean to
r = 24 at ctx 512 as well. The stall belonged to the old gather path (22 ms/layer
host staging, 6.4 GB/s scattered `index_select`); the extent-allocation and
block-major layout fix removed it. This **removes a risk** the plan carried:
C2C's larger rider count does not walk into a blow-up regime.

**And it is constant, within what this measurement can resolve.** Getting here
took three attempts and the second one is the instructive failure, so both are
recorded.

*Attempt 1* claimed `c` is constant on the strength of a fitted `r²` coefficient
of −0.00027. That is the wrong statistic: an `r²` term is not how one tests for
a power law.

*Attempt 2* fitted `ΔT = F + c·r^α` and got **α = 0.69 at ctx 512**, reporting
the affine law as *rejected*. Every check it was given passed:

| check | result |
|---|---|
| F-test for α ≠ 1 | F(1,5) = 10.28, affine rejected |
| bootstrap | P(α<1) = 98%, CI [0.40, 0.86] |
| residual sign pattern | negative at both ends, positive in the middle |
| leave-one-out | α stays in 0.52–0.74, never near 1 |
| dropping the r=1 point | moves `c` by only −3.8% |

**The conclusion was still wrong, and this is worth dwelling on.** The fit is
nearly *unidentified*: at the first rider point `F` is about 95% of `ΔT`, so `F`
and `c·r^α` trade off against each other with almost no leverage on the
exponent. Every check above conditions on the fit window and the points inside
it, so none of them can see that the estimate moves wholesale when the window
changes. It does:

| arm | r≤4 | r≤6 | r≤8 | r≤12 | r≤16 | swing |
|---|---|---|---|---|---|---|
| ctx 512 | 0.900 | 0.855 | 0.690 | 0.690 | 0.690 | 0.21 |
| ctx 256 | — | 0.890 | 0.960 | 0.750 | 0.750 | 0.21 |
| **ctx 128 (A1)** | 0.400 | 0.400 | 0.590 | 0.940 | **1.085** | **0.69** |
| **ctx 128 (A2)** | 0.400 | 0.615 | 0.980 | 0.795 | **0.980** | **0.58** |

The bottom two rows are **two runs of the identical configuration**, adjacent in
time. α runs 0.40 → 1.085 on one of them as the window widens, and the two
disagree by **0.39 at the same window** (r≤8). The 0.400 cells are the grid
floor, meaning the fit wanted a smaller exponent still — those windows are
degenerate. For reference, run-to-run spread of the quantities that *are*
identified: `c` repeats to **1.5%** and `F` to 10%, consistent with the known
±14% job offset.

**So α is not a property of the system, and no value for it should be quoted.**
The affine law `ΔT = F + c·r` is the right description — not because sharing is
proven absent, but because any deviation is below the resolution of these
sweeps. This was obtained by widening the rider axis to r=16 at short context,
which was the experiment designed to *pin* α and instead showed it unpinnable.

The lesson generalises past this campaign: **a goodness-of-fit statistic
computed inside one fit window cannot detect a model that is unidentified across
windows.** The window sweep is now the primary diagnostic in
`paper/model/amortization.py`, ahead of the F-test and the bootstrap, and its
selftest asserts that α is unstable so the finding cannot be quietly
reinstated.

---

## 3. How `c` scales

### 3.1 Measured: `∝ S`, plus a floor

Per-context affine fits, below the knee only, direct (non-differenced)
estimator, A100/Qwen-32B/L=64/bs=16
(`paper_runs/20260930/overhead/02_law/A_c{256,512,1024,2048}.json`):

| prompt | `S_eff` | pts | `F` ms | `c` ms/rider | `c/prompt` | knee | max resid | hidden |
|---|---|---|---|---|---|---|---|---|
| 256 | 335 | 7 | 12.07 | 0.726 | 0.00283 | 14.0 | 0.45 | 79% |
| 512 | 591 | 8 | 11.85 | 1.073 | 0.00210 | 8.0 | 0.42 | 83% |
| 1024 | 1103 | 4 | 10.89 | 2.114 | 0.00207 | 4.4 | 0.57 | 82% |
| 2048 | 2127 | 2 | — | — | — | 2.2 | — | — |

> **Which context does `c` scale with — the prompt, or what is actually
> streamed?** The two differ: a rider streams its *current* sequence length,
> which grows as it generates, and the measured streamed length `S_eff` sits a
> constant 79 tokens above the prompt at a fixed generation length.
>
> **Settled by experiment, against my expectation.** M7's C arm has the same
> nominal prompt as the ctx-256 row above but a generation length of 256 rather
> than 136, so its `S_eff` is 18% larger at an identical prompt. `S_eff` was
> predicted exactly — 395 against 395 measured — and **`c` did not follow it**:
> 0.703 against 0.726, a 3% *fall* where proportionality demands an 18% rise,
> against 1.5% run-to-run repeatability. Pooled R² cannot discriminate (0.981
> for the prompt against 0.987 for `S_eff`) because the two are collinear across
> the sweep; that one same-prompt pair is the only comparison that separates
> them, and it favours the prompt. **The fits below therefore use the nominal
> prompt**, and the 7% this costs on the slope is carried inside the band in
> §3.1b rather than claimed either way.

ctx 2048 has only two below-knee points and is excluded from fitting — which is
*precisely why* the campaign ran short-context. The knee is
`r* = T0·X/(b_tok·S)`, so shortening the context raises it (2.2 → 14.0) and that
is what made ≥4 rider points reachable at all on PCIe.

`c` grows with context, and **79–83% of the nominal transfer time hides inside
compute** at every context — a stable, encouraging number.

**`c/prompt` falls monotonically by a factor of 2.3** across the sweep, 0.0048
at prompt 128 down to 0.0021 at 1024, which is the signature of a
context-independent **floor** inside `c`. The pooled intercept is **0.34–0.45
ms/rider** depending on the inclusion filter — the same order as the 0.645 that
the independent component fit derives from the layer axis (§4), the two agreeing
to 1.9×.

### 3.1b Every arm, and the slope the projection extrapolates

M7 added two shorter contexts and a **repeat pair**, taking this fit from three
points over three jobs to six and extending the rider axis to 16:

| arm | prompt | `S_eff` | pts | r_max | `F` | `c` | `c/prompt` |
|---|---|---|---|---|---|---|---|
| M2 | 256 | 335 | 7 | 12 | 12.07 | 0.726 | 0.00283 |
| M2 | 512 | 591 | 8 | 8 | 11.85 | 1.073 | 0.00210 |
| M2 | 1024 | 1103 | 4 | 4 | 10.89 | 2.114 | 0.00207 |
| M7 A1 | 128 | 267 | 8 | **16** | 11.66 | 0.616 | 0.00481 |
| M7 A2 | 128 | 267 | 8 | **16** | 12.85 | 0.607 | 0.00474 |
| M7 B | 192 | 331 | 7 | 12 | 11.56 | 0.699 | 0.00364 |
| M7 C | 256 | 395 | 8 | 10 | 12.12 | 0.703 | 0.00275 |

The A1/A2 pair is the honest error bar on a single arm: **`c` repeats to 1.5%
and `F` to 10%**, the latter inside the known ±14% job offset. So `c` and `F`
are well identified even though α is not (§2).

**But pooling them exposes an extrapolation risk the three-point fit hid:**

| filter | n | prompt span | fit | R² | max resid | `c` at 8192 |
|---|---|---|---|---|---|---|
| ≥4 below-knee pts | 7 | 128–1024 | `0.337 + 1.673e-3·S` | 0.981 | 0.121 | **14.04** |
| ≥5 (better conditioned) | 6 | 128–512 | `0.446 + 1.186e-3·S` | 0.971 | **0.047** | **10.16** |

The slopes differ by **41%**, which at prompt 8192 is a **38%** difference in
the extrapolated `c`. The tighter fit spans only 4× and **the target is 16×
beyond its upper end**, so which slope is used matters more than the quality of
either. And the arm the strict filter drops is the **longest context, the one
closest to the target** — so the strict fit is the optimistic reading, and it is
also the least well determined arm, at four below-knee points.

Both slopes give the same verdicts (§4), so nothing moves. It is reported as a
band with each fit's span beside it, because a slope quoted without its fit
range is how a 14× extrapolation gets made silently.

> **What this retracts.** An earlier draft divided by the nominal prompt, got
> `c/S` falling 37% from the shortest context, and read that as the signature of
> a context-independent floor inside `c`, quoting an intercept of 0.205. On the
> measured context the spread is 19% and non-monotonic, and the intercept is
> 0.059. **These three points therefore carry almost no evidence about the size
> of the floor.** The floor is real — the layer axis measures it directly at
> fixed head geometry — but it must be estimated from that axis and not by
> extrapolating a 3.3× context sweep back to `S = 0`. The selftest now asserts
> the corrected claim, and it was that assertion failing that caught the error.

### 3.2 The four components, and only one of them is link-bound

The factorial (block-size, layer-subset, three transfer arms, resident cohort)
separates `c` into four terms. This is the decomposition that matters, because
**a faster link divides exactly one of them**:

| component | ctx 512 | scales with | link? | code site |
|---|---|---|---|---|
| per-rider per-layer launch | 0.536 (L=64) | **∝ L**, not S | invariant | `hybrid_rungd_pool.py` `_issue_runs` — 2·L·R memcpys/step |
| per-block host work | 0.245 | **∝ S/bs** | **invariant** | `build_step` — one `__setitem__` per rider block per step |
| the transfer itself | 0.398–0.709 | ∝ S, **÷X** | **BOUND** | the bus |
| D2H write-back | 0.109 | constant | invariant | `write_token` host mirror |

```
  c(S, X, L) = 0.00838 * L          per-rider per-layer    [invariant]
             + 0.00765 * S/bs       per-block              [invariant]
             + 0.0744 * b_tok*S/X   the transfer           [LINK-BOUND]
             + 0.109                D2H per rider          [invariant]
```

Three independent methods put the **link-bound share of `c` at ctx 512** at
37% (the 105-point component fit), 47% (the `minbytes` arm — same 2·L·R launches,
one block row each, i.e. an infinitely fast link) and 57% (the `nobus` arm).
Honest interval: **37–57%**. Already at ctx 512 on PCIe, *most of `c` is not the
link.*

### 3.2b Which components are per-rider by *physics*, and which only as written

§2 established that riders share some cost empirically. This is the structural
counterpart, and it is the question that decides whether the sharing can be
engineered rather than merely observed. **Only one component is a floor:**

| component | per-rider because | status | if batched |
|---|---|---|---|
| link / bytes on the wire | each rider's KV is **distinct data** that must cross the bus | **FLOOR** | nothing to batch; stays ∝ r |
| per-rider per-layer launch | 2·L·R *separate* `cudaMemcpyAsync`, one pair per rider per layer | **batchable** | one copy per layer → 2·L launches; **moves out of `c` into `F`** |
| D2H write-back | r *separate* pageable `.to("cpu")` calls, one token each | **batchable** | one copy for all riders; **moves into `F`** |
| per-block host work | one Python `__setitem__` per rider block per step | count irreducible | vectorising collapses cost *per item*, not the count; stays ∝ r |

So at the target only the **9.2%** that is bytes on the wire is physically
per-rider. The other 90.8% is per-rider because of how the path is written —
though note that no sharing is actually *observed* (§2): these are costs that
*could* be batched, not costs that currently amortise on their own.

**Batching is not the same as making it cheaper.** Moving a cost from `c` into
`F` converts a charge paid r times into one paid once, which *helps in
proportion to r* — and r is capped by the knee. At the target the knee is 3.8,
so there are under four riders to amortise over. That is why batching is worth
only a few percent *there*, and a great deal at a high knee (§5).

### 3.3 Where that lands at the real operating point

GH200/70B/L=80/ctx 8192, with the measured 419 GB/s H2D (not the 450 spec):

| component | ms/rider | share |
|---|---|---|
| per-block host work | **3.917** | **75.7%** |
| per-rider per-layer | 0.670 | 13.0% |
| **the transfer** | **0.477** | **9.2%** |
| D2H | 0.109 | 2.1% |
| **total `c`** | **5.173** | vs `T0/B` = 3.37 |

Scaling to the target *inverts* the composition. On PCIe at ctx 512 the link is
roughly half of `c`; at the target it is a **ninth**, because `per_block` scales
with `S` while the link term scales with `S/X` and `X` went up 17×. The feature's
own success case is what makes the interconnect stop mattering.

### 3.4 Both sides rise with `S`. The slope decides.

`c ≈ c₀ + k·S` and `T0/B = (T0·b_tok/C)·S`. Comparing slopes:

| | `c` | `T0/B` | crossover |
|---|---|---|---|
| A100 / Qwen-32B, as built | 0.645 + 1.255e-3·S | 1.131e-3·S | **`c` slope is larger → fails at every context** |
| A100, per-block removed | 0.645 + 7.770e-4·S | 1.131e-3·S | pays above ctx 1825 |
| GH200 / 70B, as built | 0.779 + 5.363e-4·S | 4.114e-4·S | **`c` slope is larger → fails at every context** |
| GH200, per-block removed | 0.779 + 5.818e-5·S | 4.114e-4·S | pays above ctx 2206 |

**As built, `c` grows faster with context than the budget does**, on both
platforms. There is no context at which it pays, and going to longer context —
the regime the feature is designed for — makes it monotonically worse. Removing
the per-block loop flips the slope comparison, and then longer context *helps*:
the feature pays above ctx ≈2200 on GH200.

---

## 4. Sensitivity: does the verdict survive how `c` is apportioned?

The two routes to `c`'s split disagree, and the disagreement is worth stating
because the invariant share is exactly what decides condition (1):

| route | `c₀` | `k` (per token) | x used | `c` on that arm |
|---|---|---|---|---|
| seven arms pooled, direct estimator (§3.1b) | 0.337 | 1.673e-3 | 1.194 |
| 105-point within-job component fit + M3 | 0.645 | 1.255e-3 | 1.288 |
| **measured** | — | — | **1.073** |

Both are now in the same x-variable, the nominal prompt, and evaluated at 512.
**They agree on the floor to within 1.9×** (0.337 against 0.645), which they did
not under the streamed-length reading — that agreement is part of why the
prompt is the right variable. Both still **overshoot** the arm, by 11% and 20%,
so neither is a tight description of a single configuration, which is the
argument for using them as the sensitivity band below rather than quoting
either as the coefficient.

Route 1 puts far less in the floor and ~42% more in the slope, and after the
`S_eff` correction its floor is small enough that it should not be read as a
floor estimate at all (§3.1). It remains useful for the *slope*, which is what
the projection extrapolates, and that is what the sensitivity below tests —
noting that route 1's own slope carries the ±26% band of §3.1b. Both land within
7% of the ctx-512 measurement; **they diverge only on extrapolation**, which is
exactly where the projection lives. Running the verdict under both:

| route / configuration | `c` | `T0/B` | verdict |
|---|---|---|---|
| route 1, as built | 8.24 | 3.37 | never pays |
| route 2, as built | 5.20 | 3.37 | never pays |
| route 1, block+D2H fixed | 0.79 | 3.37 | PAYS from r = 0.8 |
| route 2, block+D2H fixed | 1.17 | 3.37 | PAYS from r = 0.9 |

**Both verdicts are robust**, and the uncertainty pushes the two cases apart
rather than together: the larger measured S-slope makes "as built" worse and
(because per-block absorbs the extra slope and is then removed) makes the fixed
case better. The conclusion does not rest on which fit is right.

---

## 5. What a faster link actually buys: the knee, not the bandwidth term

Holding the model, the context and the host fixes constant and changing **only**
`X` (`overhead_model.py --> core_question()`):

```
 per-block loop removed + D2H fixed, Llama-70B fp8, ctx 8192
   link                c   c_link   share    knee    r_be    gain
   PCIe4 x16       8.627    7.957   92.2%     0.2       -    0.88x  no win
   PCIe5 x16       3.791    3.121   82.3%     0.6       -    0.92x  no win
   PCIe6 x16       2.231    1.560   69.9%     1.2     1.8    0.97x  no win
   NVLink-C2C      1.147    0.477   41.6%     3.8     0.9    1.21x
   2x C2C          0.892    0.222   24.9%     8.1     0.8    1.54x
   INFINITE        0.672    0.002    0.3%   902.8     0.8    3.46x
```

Note **PCIe6**: `c` = 2.23 is already below `T0/B` = 3.37, so condition (1)
passes — and it still loses, because break-even needs 1.8 riders while its knee
admits only 1.2. It fails on **reachability**, not on cost.

This is the precise mechanism by which the interconnect helps, and it is not the
obvious one:

* The link divides *one* term of `c`, which at the target is 9% of it.
* The link multiplies the **knee**, `r* ∝ X`, one-for-one. C2C's 17× over PCIe4
  is a 17× wider rider window.

So **the interconnect's contribution to throughput is dominated by raising the
knee** — by making a profitable rider count *admissible* — not by making each
rider cheaper. Which also means the knee is the quantity to design against, and
it is the one that C2C improves most cleanly.

### 5b. The knee also caps what batching could ever be worth

§3.2b identifies three components of `c` that are batchable in principle, which
would move them out of `c` and into `F` — a charge paid `r` times becoming one
paid once. That helps **in proportion to `r`**, so the knee bounds it just as it
bounds everything else.

At the target the knee is 3.8 riders, so there are under four riders to amortise
over and batching the launches and the mirror write is worth little *there*. Its
leverage scales with the knee, which goes as `X/S`: at **2× C2C and ctx 2048 the
knee is 30 riders**, where moving a per-rider charge into the fixed cost divides
it by thirty.

This is the same conclusion §5 reached from the other direction. **Batching is a
real lever, but the knee gates how much it returns, and the link sets the
knee.** It is also why the optimisation proposal targets the per-block rebuild
first: that one is not a batching play at all but the removal of an `∝ S` term,
and it does not depend on the rider count to pay off.

---

## 6. `F`, and why it is the secondary constraint

`F` is the cost that appears the moment any rider exists. It is large, it is
mostly an artifact, and it sets break-even — but it cannot rescue or kill the
feature on its own, because condition (1) does not contain it.

* `F` = **11.8 ms** = 0.184 ms/layer on Qwen-32B, context-independent across a
  4× range (12.07 / 11.85 / 10.89, ±5%).
* **Proportional to depth within a model** — R² 0.997 over five layer counts,
  intercept −0.41 — but the *rate* is model-specific, so `F = 12.3·L/64` is the
  right shape and the wrong constant off-model.
* **85% of it is a blocking pageable D2H.** 9.10 ms of the 11.8, at 71 µs per
  copy for a few-KB transfer — **446× the transfer time**, so it is a
  synchronisation cost, not a bandwidth cost (`write_token`,
  `hybrid_rungd_pool.py`, 64×/step).
* The mechanism is **KV geometry, not compute**: Llama-8B and Qwen-32B show
  identical D2H per layer (0.138 vs 0.142 ms) despite 1.8× different compute per
  layer. My earlier "queue drain" explanation was wrong and this falsified it.
* Resident cohort does not affect `F` — a 4× cohort change moves it 0.2% within
  one job. An apparent 15.3% "batch effect" was a **between-job offset**.
* CUDA graphs buy nothing at realistic copy sizes (bounded offline by
  `launch_pattern_bench.py`, eager vs captured). More `k_window` slots make it
  worse, not better (+5.1%).

The remaining 15% is async host work that hides in proportion to compute per
layer, and that is the one quantity an A100 cannot tell us (§7).

---

## 7. The honest envelope

Four uncertainties, stated rather than absorbed.

0. **The hiding fraction, promoted to first place** — see item 3, which is now
   the leading open question again. The exponent α that briefly held this slot
   has been **resolved negatively**: it is unidentified rather than unresolved
   (§2), so there is nothing left to measure there and the sweep that would have
   settled it has been run.
1. **Between-job offset ±14%** on `F` (within-job is ±5%; the noise floor is
   ±0.47% per step). An earlier pooled fit laundered this offset into the
   coefficients — held-out max error 11.0% ≈ the 1.78 ms job spread. Fixed with
   per-job additive offsets (RMSE 0.601 → 0.481).
2. **Link-bound share of `c`: 37–57%** at ctx 512, three methods.
3. **The hiding fraction of the async host work is unmeasurable here, and it
   bands `F` over 18×** (2.1 to 37.4 ms on GH200/70B). Three measured points:

   | | compute/layer | exposed async/layer |
   |---|---|---|
   | Qwen-0.5B | 0.369 | 0.326 |
   | Llama-8B | 0.426 | 0.143 |
   | Qwen-32B | 0.772 | 0.026 |

   Deliberately **not** curve-fitted — three points. On GH200 the 70B's compute
   per layer is 0.303 ms, *below every point measured*, because 4 TB/s HBM reads
   the weights faster. Hiding requires the host to run ahead of the GPU, so a
   faster GPU hides **less**. Two unmeasurables pull opposite ways: an H100-class
   GPU has less slack per layer (hurts), a Grace CPU may do host work faster
   (helps).

   This band is why §0's "fixed" row is stated at the 32B's hiding. At the 8B's
   hiding, break-even moves to 5.1 riders against a knee of 3.8 and the fixed
   configuration **also** loses. So condition (2) is live and the band is not
   academic.

**The one experiment that would close it:** run the three-arm D2H measurement on
*any* GPU faster than an A100. It needs no C2C link and no 70B checkpoint — the
wanted quantity is the exposed async host work per layer at compute/layer ≈ 0.3
ms, which an H100 with Qwen-32B reaches directly. Under an hour of GPU time, and
it collapses an 18× band to a point. This is the highest-value remaining
measurement in the whole programme.

---

## 8. Verdict

**On `c`'s behaviour — the core question:**

`c` is **constant in `r`** within the resolution of these sweeps — no
superlinearity, no `r_max` in range, and no measurable rider-sharing either; an
earlier sublinear exponent proved to be a fit-window artefact (§2). It is
**proportional to the streamed context**, `c = 0.059 + 1.841e-3·S_eff`.

`c` has four components and the link divides only one. At ctx 512 on PCIe that
one is roughly half of `c`; at ctx 8192 on C2C it is 9%, because the link term
scales as `S/X` while per-block host work scales as `S`. Structurally, **only
that 9% is per-rider by physics** — the launches and the D2H are batchable, and
the per-block loop is vectorisable. **`c`'s scaling is dominated by a host
property, not by the interconnect.**

**On whether a better interconnect gives better throughput:**

* **As built: no, and not at any bandwidth.** `c` = 5.17 against `T0/B` = 3.37;
  an infinite link leaves 4.70. Robust to both apportionment routes (5.2 / 8.24).
* **With the per-block loop removed and the blocking D2H fixed: yes, and the
  gain is monotone in bandwidth** — 0.88× (PCIe4) → 1.21× (C2C) → 3.46×
  (infinite). C2C is the first point that wins at all.
* **The link's dominant contribution is the knee**, `r* ∝ X`, not the per-rider
  bandwidth term. PCIe6 satisfies `c < T0/B` and still loses on reachability.
* **Conditional on the hiding fraction.** At the 32B's hiding the fixed
  configuration pays from r = 0.9; at the 8B's it does not pay at all. One
  short experiment on an H100 closes this.

**Recommendation.** The two blocking items are both host-side and both already
localised to a line range: `build_step`'s per-block Python loop and
`write_token`'s blocking pageable D2H. Together they are 88% of `c` and 85% of
`F` at the target. Fix those two before any further interconnect argument — as
the code stands, a C2C machine would be measured against a ceiling of 0.71×, and
the result would be read as a verdict on the feature when it is a verdict on two
host loops.

---

## 9. What was then built, and what it measured

The campaign was measurement only. Its two named artifacts were then
implemented, and this section reports what they actually bought — measured on
this code, not projected.

### How T2 was decided, since two measurements disagreed

`SKIP_D2H` differencing put the blocking mirror write at 9.10 ms (§6), but an
earlier commit had made exactly that copy asynchronous, measured **1.6 ms**, and
reverted it on cost. Rather than pick, a **probe arm** priced the
synchronisation on its own: it keeps both copies and only makes them
non-blocking into pinned staging, skipping the scatter, so launch count and byte
volume are held fixed.

| arm | `F` |
|---|---|
| `real_a` | 11.31 |
| **`probe`** (async, no scatter) | **2.84** |
| `skip` (no write at all) | 1.86 |
| `real_b` | 11.06 |

```
real − probe   = 8.47 ms   the synchronisation cost = the realisable prize
probe − skip   = 0.98 ms   the async copy a correct fix still pays
real_a − real_b = 0.25 ms  drift, a 33.5× ratio against the effect
```

So the prize is **8.47 ms, not 1.6**, and the two measurements are of different
code: the async prefetch chain now puts far more behind the same drain, and the
per-row host loop that the reverting commit named as its own cost driver has
since been vectorised away. `c` moved only 14% across the arms, as a per-*layer*
cost must. The standing rule was to abandon the work below 2 ms; it cleared that
by 4×.

### T1 — the block-table rebuild, now O(r)

The rebuild did one Python-level tensor element assignment per rider **block**
per step. The allocator already guarantees each request one contiguous extent
(and `_issue_runs` already relied on it), so a request's *compact* ids are
contiguous too and its whole row is one slice assignment from a preallocated
range. Measured, on the Class A host timer:

| | pre-T1 | post-T1 | |
|---|---|---|---|
| ctx 512 | 0.275 ms/rider | **0.00313** | 87× |
| **ctx 8192 (target)** | **3.917 ms/rider** | **0.00313** | **1234×** |

`bt_ms = 0.0216 + 0.00317·riders`, max residual 0.00055 ms — fitted in *riders*,
with no block term. **The term lost its dependence on context, which was the
claim**, and that is shown three ways rather than fitted once:

* **riders**: linear to r = 12, no block term
* **block size**: a 4× block-count cut (bs 16→64) moves the timer **2–6%** at
  every rider count, where before it would have moved ~4×; the two arms' slopes
  agree to 3% while differing 4× in block count throughout
* **context**: 132 blocks at ctx 2048 against 36 at ctx 512 — a 3.7× change —
  and the timer is **identical** (0.0240 vs 0.0243 ms)

### T2 — the mirror write, deferred

`write_token` now copies into pinned staging with `non_blocking=True` and the
host-side scatter is deferred to the next step behind one event wait, so 2·L
*blocking* copies per step become 2·L async ones. Bracketed before/after, both
arms numerically valid:

| | `F` blocking | `F` async | saved |
|---|---|---|---|
| ctx 512 | 11.48 | **4.74** | **6.74 ms** |
| ctx 2048 | 9.70 | 4.69 | 5.01 ms |

Bracket drift **0.19 ms** against a 6.74 ms effect, a 35× ratio; `c` spread 16%,
consistent with a per-layer cost; every row passed the purity gate.

**It realises 80% of Phase 0's 8.47 ms ceiling, and the shortfall is
principled**: the probe arm skipped the scatter entirely and T2 still performs
it, just later. Per layer, 0.1655 → 0.0737 ms, so after subtracting the 0.026
of exposed async host work the blocking-D2H part goes 0.142 → 0.048 — **T2
removes 66% of it, not all of it.** The model carries that as its own
coefficient rather than reusing the idealised "D2H removed" figure, and a
selftest asserts the post-T2 `F` stays *above* that floor so the two cannot be
conflated.

### The delivered projection

Llama-70B fp8, ctx 8192, GH200 at the measured 419 GB/s; 7.2 residents from
capacity, `T0/B` = 3.37 ms, knee 3.8 riders:

| state | `c` | `F` | break-even | gain | verdict |
|---|---|---|---|---|---|
| as built | 5.173 | 13.44 | — | 0.65× | never pays: `c` ≥ `T0/B` |
| **T1 only** | **1.259** | 13.44 | 6.4 | 0.87× | break-even past the knee |
| **T1 + T2** | **1.259** | **5.90** | **2.8** | **1.06×** | **pays from r = 2.8** |

Three things this says that the pre-implementation projection did not.

1. **T1 alone is not enough.** It fixes condition (1) — `c` goes under the
   budget — but break-even then lands at 6.4 riders against a knee of 3.8. The
   pair is required; neither alone turns the target positive.
2. **The gain is 1.06×, not the 1.21× projected in §5.** That projection assumed
   the blocking D2H went away entirely. Measured, 34% of it remains, `F` = 5.90
   still dominates, and at a knee of only 3.8 riders there is no room to amortise
   it. The optimistic row of §5 was optimistic by exactly the amount T2 does not
   remove.
3. **It is now positive rather than impossible**, which is the qualitative
   change: the as-built case failed condition (1), which no interconnect could
   fix at any bandwidth (§5). That is no longer the situation.

### 9b. Retargeting the model, and the trap in doing so

Searching every (model, context) pair under the law puts **Qwen-72B fp8 at
ctx 16384** first, at **1.22×**, and `overhead_model.py` has been retargeted to
it. It beats Llama-70B fp8 on nothing but 76.0 GB of weights against 72.7:
heavier weights raise `T0` and shrink the capacity left for KV, so fewer
residents fit (3.0 against 7.2) and each rider is worth more (`T0/B` 8.50 against
3.37 ms). `c` is 1.736 rather than 1.259 because the context is doubled, and the
budget still wins.

**But the ratio and the absolute rate rank in opposite directions, and that
changes what should be *deployed* as distinct from what should be *built*.**

| model | ctx | `B` | knee | gain | ITL | base tok/s | with riders |
|---|---|---|---|---|---|---|---|
| Llama-70B fp8 | 4096 | 14.4 | 7.6 | 0.98× | 37.9 | 593 | 580 |
| Llama-70B fp8 | 8192 | 7.2 | 3.8 | 1.06× | 34.9 | 297 | **314** |
| Llama-70B fp8 | 16384 | 3.6 | 1.9 | 1.11× | 33.4 | 148 | 164 |
| Llama-70B fp8 | 32768 | 1.8 | 0.9 | 1.13× | 32.7 | 74 | 84 |
| Qwen-72B fp8 | 4096 | 11.9 | 7.9 | 1.07× | 39.3 | 471 | 505 |
| Qwen-72B fp8 | 8192 | 6.0 | 4.0 | 1.16× | 36.2 | 235 | 274 |
| **Qwen-72B fp8** | **16384** | **3.0** | **2.0** | **1.22×** | **34.7** | **118** | **143** |
| Qwen-72B fp8 | 32768 | 1.5 | 1.0 | 1.24× | 33.9 | 59 | 73 |

The best-ratio point delivers **143 tok/s**; the 70B at ctx 8192 delivers
**314**, 2.2× more, at a gain of only 1.06×. The gain rises monotonically as the
resident count falls, because a smaller `B` is exactly a *weaker baseline* — and
maximising a ratio over the choice of baseline is not a throughput result.

The two readings are not in conflict. The ratio is the correct test of **the
mechanism**: it compares the same system with and without the feature, and the
feature's value is largest precisely where the baseline is weakest — heavy
weights, long context, few residents — which is the regime this project exists to
serve. It must simply not be reported as a throughput record. `absolute_table()`
prints both, and a selftest asserts the inversion so the caveat cannot silently
go stale.

### 9c. fp8 KV: neither helps nor hurts — it rescales the context

This was logged as an open question that might flip the verdict. It cannot, and
the reason is structural: **the law sees the per-token KV footprint and the
context only through their product `b_tok·S`.** Halving the bytes per token and
doubling the context are the same point:

| KV | ctx | `B` | `T0/B` | `c` | knee | gain |
|---|---|---|---|---|---|---|
| fp16 | 8192 | 6.0 | 4.25 | 1.259 | 4.0 | 1.164× |
| fp16 | 16384 | 3.0 | 8.50 | 1.736 | 2.0 | **1.216×** |
| fp16 | 32768 | 1.5 | 17.00 | 2.689 | 1.0 | 1.244× |
| fp8 | 8192 | 11.9 | 2.13 | 1.021 | 7.9 | 1.072× |
| fp8 | 16384 | 6.0 | 4.25 | 1.259 | 4.0 | 1.164× |
| fp8 | 32768 | 3.0 | 8.50 | 1.736 | 2.0 | **1.216×** |

fp8 at 32768 is fp16 at 16384 to every decimal the model carries — gain
1.215762 against 1.215762. So KV quantisation is neutral *at matched `b_tok·S`*,
and what it actually buys is **the same gain at twice the context**, which is a
real operational win the law is blind to.

> **Correction.** An earlier note in this campaign recorded that fp8 KV loses
> (break-even 8.9 against a knee of 7.6). That compared fp16 and fp8 at a *fixed*
> context, where fp8 doubles the resident count and so halves `T0/B` while `c`
> falls by much less. Held at fixed context the note is arithmetically right and
> the comparison is the wrong one: nobody runs fp8 KV in order to serve the same
> context with more residents, they run it to serve a longer one. The verdict is
> unchanged either way, which is why this was never a blocker.

### 9d. The optimisation programme, and where the floor actually is

Ben's instruction was to keep reducing `F` and `c` until they are as small as
they can be. This records what was tried, what each attempt cost to evaluate,
and what is left — including two attempts that were **killed before being
built**, which is the point of the method rather than a failure of it.

**The instrument.** `benchmarks/kv_pipelining_bench/launch_pattern_bench.py`
replays the pool's exact copy patterns engine-free, in minutes. It exists because
T3 was implemented, proved correct, and turned out to be a 6× regression — the
mechanism it relied on (fewer launches) was not the binding cost, and the
warning was in a comment in the file being edited. The benchmark carries **T3
itself as a control**, and the control reproduces the regression off-engine (4.6
GB/s against 22.3 contiguous, device time +382%), so the harness is validated
against a known answer before it is trusted on an unknown one. Its per-run arm
also predicts the in-engine `issue_runs` timer to 12%.

| attempt | what it does | verdict | cost to find out |
|---|---|---|---|
| T1 | block-table rebuild O(r) not O(r·S/bs) | **built**, `c` 5.17→1.26 | — |
| T2 | defer the mirror write | **built**, `F` 13.4→5.90 | — |
| T3 | one strided copy per layer | **reverted**, 6× worse | a day, in the engine |
| T4 | batch the per-layer D2H | **killed unbuilt** | one GPU-minute |
| T5 | whole-extent runs | **measured in-engine: null on both `c` and `F`**, flag-gated off | one GPU-minute to price, ~1 h to measure |
| T6 | order the mirror write | correct, but costs the **whole overlap**; flag-gated off | — |

**T4 is dead, and it was the largest remaining item in `F`.** `write_token`
spends 2.477 ms/step issuing D2H launches and that figure is *flat* in riders
(+2% from 1 to 8), so it is pure fixed cost. The candidate replaced 2L D2H
launches with 2L device-to-device copies into device staging plus one batched
D2H. Measured: **18% slower** (33% for the single-D2H variant). You cannot
remove 2L launches by adding 2L more. The benchmark's "2 D2H only" row shows the
prize was real (−98.8% host) — it is simply not reachable through a staging
copy, because every layer produces its KV at a different time in a different
tensor and aggregating it costs a touch per layer either way.

> ### T5 MEASURED IN THE ENGINE: it fires, and it buys nothing
>
> Everything in this subsection below this box was a projection from the
> off-engine pricing. The in-engine measurement contradicts it, and the
> contradiction is worth more than the projection was.
>
> Same job, back-to-back, ctx 512, Qwen-32B on A100:
>
> | arm | `n_runs` @ r=8 | launches | `w_blocks` | `F` | `c` |
> |---|---|---|---|---|---|
> | T5 off | 8 | 1024 | 264 | 5.27 | **0.899** |
> | T5 on | **1** | **128** | 328 | 5.75 | **0.948** |
>
> The mechanism fired exactly as designed — verified in the phase JSONL *before*
> any timing was read, which is the gate T1 nearly failed and T3 did fail. The
> launch count is now independent of the rider count.
>
> **And nothing moved.** Full bracket, three off arms and two on:
>
> | | off arms | on arms | effect | drift | verdict |
> |---|---|---|---|---|---|
> | `c` | 0.899, 0.946, 0.892 | 0.948, 0.928 | +0.025 | 0.054 | **not resolvable** |
> | `F` | 5.27, 4.78, 5.35 | 5.75, 5.49 | +0.484 | 0.568 | **not resolvable** |
>
> T5 moves **neither coefficient** resolvably, while the model predicted a
> **0.374 ms** cut in `c` — seven times the drift. Two earlier readings of mine
> are retracted by this table: the "+5% increase in `c`" (from one pair) and the
> "marginally real `F` increase" (from two arms). Both were drift-sized.
>
> **So `b_per_rider_layer` — 0.536 ms/rider, 51% of `c` at this config — is not
> the launch count.** Removing seven eighths of the launches did not move it.
>
> Leading interpretation, untested: it is the extra **attention pass over the
> streamed window**. That scales with the window (∝ r), runs once per layer
> (hence the `r·L` shape the fit found), is GPU compute rather than host work,
> and T5 makes it slightly *worse* by enlarging the window with slack — 264 → 328
> blocks, +24%, the right sign and order for the +5% observed. Confirming it needs
> a sweep of window size at **fixed** rider count, which this campaign has not
> run.
>
> **Two consequences.** The "launches fixed in the allocator → 1.19×" rung in the
> ladder below **does not exist**. And if the interpretation holds, the dominant
> term in `c` is irreducible by any host-side change *and* by any interconnect,
> putting the ceiling nearer **1.1–1.2×** than the 1.32× floor computed from
> treating it as removable. T5 stays default-off: correct, mechanism working, not
> worth enabling.

**T5 was projected to help at short context and hurt at long — the flip below is
the model's reasoning, now superseded by the box above.**
`_plan_extents` has always merged adjacent runs, and the merge has never fired:
consecutive extents *are* adjacent, but a decode request grows a block every
`block_size` tokens, so an extent sized for its final length always has unused
slack at its tail and the next extent's base sits past it. No packing removes
that. Taking the whole extent closes the gap — one contiguous run, 2 launches
per layer instead of 2r — at the price of carrying the slack.

| ctx | slack | `c` now | `c` T5 | knee | knee T5 | gain | gain T5 |
|---|---|---|---|---|---|---|---|
| 2048 | 25.0% | 0.902 | 0.331 | 15.8 | 12.7 | 0.926× | **1.077×** |
| 4096 | 12.5% | 1.021 | 0.451 | 7.9 | 7.0 | 1.072× | 1.152× |
| 8192 | 6.2% | 1.259 | 0.689 | 4.0 | 3.7 | 1.164× | 1.197× |
| 16384 | 3.1% | 1.736 | 1.166 | 2.0 | 1.9 | 1.216× | 1.223× |
| 32768 | 1.6% | 2.689 | 2.119 | 1.0 | 1.0 | 1.244× | 1.237× |

T5 buys launches with bytes. `c` is **launch-dominated at short context and
link-dominated at long** — at ctx 16384 the link is already 55% of `c` and the
launch term only 39% — so the same trade is good at one end and bad at the
other. It also *moves* cost from `c` into `F` rather than deleting it, which is a
**structural** improvement even where the arithmetic is a wash: the gate is
`c < T0/B`, and `F` only sets where break-even lands.

So what T5 is worth is **absolute throughput, not headline ratio**: with it,
ctx 8192 reaches 1.197× at ~274 tok/s, where today matching that ratio means
going to ctx 32768 and 73 tok/s. That is the same point §9b makes — the best
ratio is not the best deployment — arriving from the other direction.

> One correction worth recording. My first reading took the microbenchmark's 24%
> slack penalty at ctx 512 as a general figure, which made T5 look like a loss
> everywhere. **The slack fraction is `gen/S`** — set by the generation length,
> not the context — so the same 8 blocks is 24% at ctx 512 and 3% at ctx 16384.
> The number was right and the inference from it was wrong.

**Where the floor is — and the honest answer is that the dominant term is
unidentified.** An earlier version of this paragraph computed a floor of
`c` ≈ 1.07 and `F` ≈ 3–4, worth ~1.32×, by assuming T5 would remove ~90% of the
"launch" term. **T5 removed 7/8 of the launches and the term did not move**, so
that floor is withdrawn.

What the decomposition looks like with the identity of each part marked by what
is actually known about it (ctx 512, A100, where the engine measurements were
taken):

| part | size | share | what is established |
|---|---|---|---|
| `b_per_rider_layer` | 0.536 | **51%** | **identity unknown.** Not launches (T5: 8× fewer, no change). Not proportional to blocks per rider (T5 raised them 24%; a proportional term would have moved +0.13, which is 2.4× the 0.054 drift and would have shown). Per-rider, per-layer, and O(1) in window size. |
| link, un-hidden | 0.398 | 38% | `bytes/bandwidth` × the 7.4% that fails to hide. Irreducible by definition — it is what the feature exists to spend. |
| d2h per rider | 0.109 | 10% | — |
| block table | 0.003 | 0.2% | at floor (T1 took it from 3.917) |

And for `F`: the D2H half is **at floor** — T4 measured the only available
aggregation as 18% *worse*. The exposed-host-work half shrinks only with slower
compute per layer, i.e. it is a property of the GPU, not of this code.

So the defensible statement is **not** "`F` and `c` cannot be optimised
further". It is:

* **the host-side launch-count avenue is exhausted** — three attempts (T3, T4,
  T5), every mechanism verified firing, none helped;
* **the link term is irreducible** and already 92.6% hidden;
* **the largest single term, 51% of `c`, has no established identity**, and an
  unidentified term cannot be called either reducible or irreducible.

> **A hypothesis I offered and then had to weaken.** I suggested the mystery term
> is the extra attention pass over the streamed window. T5's own data argues
> against it: T5 raised the window 264 → 328 blocks, +24% rows *and* bytes per
> rider, and a window-proportional term would have risen +0.13 — detectable
> against 0.054 drift. It did not. So the term is not window-proportional
> either, and I do not have a mechanism for it. Candidates that survive the
> constraints are per-rider metadata or `cu_seqlens` setup in the two-pass
> attention, or a fixed per-rider kernel cost.

**That is what the next experiment has to settle**, and it is an identification
experiment rather than an optimisation attempt — see
`paper/OPT_PROPOSAL_IDENTIFY_C.md`. It matters beyond this report: the project's
Solution 1 sequencing lists "fuse to a single unified attention call" as step 3
with the prediction that the delta would be *second-order*. If the mystery term
turns out to be per-rider attention-call overhead, that prediction is wrong in
exactly the way the launch attribution was wrong, and fusing becomes the
high-value change. If it is not, fusing stays second-order and the ceiling
stands near **1.1–1.2×**.

**And the honest caveat on all of it:** every coefficient above was fitted from
runs with the data race of §0 open on every step. The relative statements — which
term dominates, which axis a term scales with, the sign of T5's effect — are
robust, because the defect is common to every arm of every sweep. The absolute
magnitudes are provisional until re-measured on the ordered path.

### Verification

| check | result |
|---|---|
| `run_suite.sh` with T2 on | **9/9**, including the token-identical end-to-end test |
| `run_suite.sh` with T2 off | 9/9 — the baseline |
| **negative control** | flush intact → **PASS**; flush suppressed → **FAIL, 3/3 tokens diverged** |
| `build_step_equiv_test.py` | **18/18**: 14 T1 cases bitwise against the pre-change algorithm (carried verbatim so it cannot drift), plus 4 for T2 |
| fast path actually taken | `build_fast_path: 1` on every recorded step inside the real engine |
| purity gate | every retained row; one arm discarded in full, see below |
| `overhead_model.py` | 12/12, including two assertions that T2 must cut `F` substantially but stay above the idealised floor |

**The suite passing with T2 both on and off would not by itself show the suite
could *catch* a T2 bug**, and T2's failure mode is silent: a missed flush leaves
the mirror stale, `_issue_runs` streams wrong KV at the next step, and tokens
come out quietly wrong rather than the run crashing. So the evidence includes a
**negative control** — suppress the deferred flush, and require the
token-identical test to fail.

It does. With the flush intact the test passes; with it suppressed, **all three
prompts diverge**, and the divergence looks like what stale KV should produce:

```
vanilla -> ' Julius Caesar, a Roman general and statesman,'
pool    -> ' Julius Caesar Augustus, cano\nPlease implement'
```

A positive arm runs first and must pass, because otherwise an inconclusive
negative arm cannot be told apart from a broken environment. **That is what
makes the 9/9 evidence rather than merely true**: the test is demonstrably
sensitive to exactly the failure mode T2 introduces.

Two defects in the first version of this control are worth recording, because
each would have produced a confident but empty claim. It checked the *exit
code*, but the test's `main()` has no `sys.exit` and always exits 0 — the suite
decides by grepping stdout — so it read a run that never compared tokens as a
pass. And the suppression itself leaked: returning early without clearing left
the pending list growing by one entry per layer per step, so the server failed
its completions from the leak rather than from stale KV, and the test reported
*inconclusive*. The control now drops the pending rows, which is a missed flush
and nothing else, and reads the verdict string rather than the exit status.

**One arm was discarded entirely.** At ctx 2048 with `block_size` 64 the gate
rejected all three rows for prefill leakage: block rounding inflates each
request's allocation from 2256 to 2304 tokens, 16 residents then need 36,864
against a pool between the two, and it tips into eviction and recompute. Its own
baseline shows it (61.42 ms against 53.76 at bs 16). So the published ctx-2048
configuration has no headroom for a coarser block size either, not only none for
longer sequences. Nothing here rests on that arm; it was the fourth and
redundant confirmation of T1.

### What was deliberately not built

The proposal's unified-mirror layout — one pinned allocation per K/V with
per-layer views, so the deferred scatter is a single `index_copy_` instead of
2·L. It is worth roughly 0.4 ms of an ~8 ms win and is the only structural
change in the proposal, touching every `host_k` call site. Skipped as a poor
risk trade; available as a follow-up if that 0.4 ms ever matters.

---

## 10. Provenance

* Model + projection: `paper/model/overhead_model.py` (`--selftest`, 8 checks,
  PASS). `core_question()` prints §0 and §5; `projection_table()` prints the
  interval.
* **The amortization analysis of §2 and §3.2b**:
  `paper/model/amortization.py` (`--selftest`, 6 checks, PASS) — fits the
  exponent with a bootstrap CI, runs the r=1 and residual-pattern diagnostics,
  and carries the physics-vs-implementation table.
* **Sections 3.1 and 4 regenerate from the archived JSONs**:
  `paper/model/c_scaling.py` (`--selftest`, 7 checks, PASS). It re-fits `c` per
  context behind the knee filter and re-runs the verdict under both
  apportionment routes, so the two tables that decide the answer are not typed
  numbers.
* Fitter with model selection, bootstrap CIs, held-out error, and the knee /
  model / job-offset filters: `paper/model/overhead_fit.py`.
* Decode-purity gate: `benchmarks/kv_pipelining_bench/decode_purity.py`
  (selftest 11/11, including a `_record` wiring test).
* Raw data: `paper_runs/20260930/overhead/{00_baseline,00b_ramp,01_perturbation,
  02_law,03_d2h,04_stall}/`, copied verbatim to `paper/data/overhead/` with
  `MANIFEST.json` (58 run records, each with its config, source commit, purity
  verdict and a sha256 prefix). These are harness artifacts, not
  transcriptions — the two previously-transcribed files were replaced.

### Ground-rule compliance, read off the manifest

| rule | check | result |
|---|---|---|
| P/D-emulated, feature on the D node | artifacts with `connector: false` | **1 of 58** — `00_baseline/B_colocated.json`, the deliberate M0 A/B control |
| provenance | artifacts recorded from a modified tree | **0 of 58** |
| numerics flagged | `numerically_valid: false` | 12 — all layer-subset arms, timing-only by construction, double-gated behind `..._LAYERS_UNSAFE=1` and never cited for a quality claim |
| timers must not perturb | Class A cost | `perf_counter_ns` 439 ns; the whole `_record` body 28.5 µs = **0.053%** of a 53.5 ms step |
| timers must not perturb | Class B | **excluded from every absolute coefficient**; the law is fitted from untimed runs only |
* Chronological log with every correction and dead end:
  `paper/OVERHEAD_PROGRESS.md`.
* Paper prose, `killer_v2.F_REF_MS`, the tables and the `check_tables.py` pins
  are **untouched** by this campaign, per the deliverable decision.

### Corrections this campaign made to its own earlier claims

Recorded because several were confident and wrong, and the measurement caught
them:

0. **"Riders share part of `c`, α̂ = 0.69" — RETRACTED.** This was itself a
   correction to an earlier overstatement, and it went too far the other way.
   The exponent is **unidentified**: on one arm at one context it runs 0.40 to
   1.085 purely as the fit window widens, and two runs of the identical
   configuration disagree by 0.39 at the same window (§2). An F-test, a
   bootstrap, a residual-sign check and leave-one-out all endorsed α = 0.69,
   and all four condition on the fit window, so none could see the problem.
   **The affine law stands.** What survives from that round, and is the useful
   part, is the *structural* finding: only 9.2% of `c` at the target is
   per-rider by physics, and the rest is batchable or vectorisable (§3.2b).
0b. **The streamed-context "correction" — itself retracted, and the floor claim
   restored.** I switched the context fit from the nominal prompt to the
   measured streamed length, which made `c/S` look flat and the floor look
   negligible. Then I tested it: M7's C arm has the same prompt with an 18%
   larger streamed length, `S_eff` was predicted exactly, and **`c` did not
   follow it** (0.703 against 0.726, a 3% fall where proportionality demands an
   18% rise). Pooled R² cannot discriminate the two variables — they are
   collinear — so that one pair decides it, and it says the prompt. Reverting
   restores the floor at **0.34–0.45**, which now agrees with the component
   fit's 0.645 to 1.9× where the streamed reading disagreed by 10×. Net effect
   on the projection: 7% on the slope, inside the band already carried, and no
   verdict moves.
1. "Not affine in `r`" (the ramp hypothesis) — **refuted** as to the ramp; the
   rise is the knee crossing. Note this is a separate question from item 0:
   below the knee `c` does not *rise*, but it is mildly sublinear.
2. "The 12.3 ms anchor is stale" — **closed negatively**; `F+c` = 15.56 measured
   vs 16.74 predicted, 7% agreement. That column is `F+c`, not `F`.
3. "Per-byte, not per-block" — **too absolute**; both terms are significant, and
   per-block is the dominant one at the target.
4. "Batch effect 15.3%" — it was a **between-job offset**; within-job it is 0.2%.
5. "`F` is ±3%" — that was within-job; between-job is **±14%**.
6. D2H "queue drain" mechanism — **falsified** by 8B/32B having identical
   D2H/layer at 1.8× different compute/layer. It tracks KV geometry.
7. Phase timers as a cost-attribution tool — **abandoned**; they accounted for
   214–304% of `ΔT` and read 21.4 ms/step for a few-KB copy, reproducing the
   repo's own documented artifact. Replaced by `SKIP_D2H` differencing.
8. An automated fit selected a superlinear term over the affine law — the kink
   was absorbed as curvature while held-out error *passed* at 4.2%. Fixed by the
   knee filter; a pooled fit also produced a **negative** link coefficient,
   i.e. "more bytes makes the step faster", which is how a misspecified model
   announces itself.
