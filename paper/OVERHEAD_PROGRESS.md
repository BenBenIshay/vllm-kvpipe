# Overhead characterisation — live progress log

Campaign: decompose `F` (the per-step cost of using the host path) into its
static / per-rider / per-byte / per-block parts, then project it onto NVLink-C2C.
Plan: `~/.claude/plans/fluttering-dreaming-kite.md`. Measurement only — no
feature optimisation.

**Read the "Open concerns" section first.** It is where anything that might need
Ben's steer is parked, newest first. Everything else is append-only history.

Branch `overhead-characterisation`, worktree `.claude/worktrees/overhead-char`,
forked from `05b32de56` (`hybrid-kv-tiers`).

**Where to read this.** The work happens in an isolated worktree, so the copy
in git is at
`.claude/worktrees/overhead-char/paper/OVERHEAD_PROGRESS.md` — not at
`paper/OVERHEAD_PROGRESS.md` in the main checkout. A mirror is kept at
`/mnt/beegfs/benishay/paper_runs/20260930/overhead/OVERHEAD_PROGRESS.md`,
next to the campaign's run outputs, and updated whenever this file is.

---

## Scope: what counts as the feature's cost

**Seeding a rider does NOT.** Ben, 2026-09-30: *"this is an artifact of the
emulation itself, if we are focusing on the decode node we assume the P nodes put
the data in the DDR, it's not part of the cost of our feature/riders."*

So the one-off cost of *creating* a rider — `seed_req`, which copies the
request's whole prefill into the host pool as `2*L` copies — is out of scope. In
a real P/D deployment the prefill node has already put that KV in host memory;
the decode node never pays to put it there. It is a cost of emulating prefill
locally.

Already excluded by construction, which is worth stating so it stays deliberate
rather than incidental: seeding happens once per request, identically in the
short and the long run, so the differencing estimator cancels it; the phase
timers do not instrument `seed_req`; and the `minbytes`/`nobus` arms never reach
it, since it does not go through `_issue_runs`.

What remains in scope is the **per-step** cost of *having* riders — the `~11*L`
switch-on ops, the `2*L*R` memcpys, and the transfer itself. That is `F` and `c`,
and it is what this campaign decomposes.

Not in scope either, and not a cost of anything: the **admission ramp**. The
scheduler admits the cohort over several steps rather than all at once, so the
batch climbs to its target (`~2(r-1)` extra steps, up to `r-1` requests waiting)
and drains at the end. Only the steady phase between them is "a decode step with
r riders". The estimator is supposed to subtract ramp and drain by differencing
two generation lengths; C4 is about the case where it fails to.

---

## Open concerns

### C1 — ANSWERED 2026-09-30. Measured: the rigs differ 5-8%, with inconsistent sign.

Arm A (`--connector`) vs arm B (co-located), same config, same session:

| r | arm A dT | arm B dT | B vs A | A GB/step | B GB/step |
|---|---|---|---|---|---|
| 2 | 17.91 | 16.95 | **-5.4%** | 1.115 | 1.115 |
| 4 | 43.90 | 47.20 | **+7.5%** | 2.230 | 2.290 |
| 6 | 92.93 | 98.35 | **+5.8%** | 3.379 | 3.464 |

Arm B is heavily contaminated, as expected — 32,749 to 45,031 prefill tokens over
5 to 10 steps, `max_waiting` 12 to 18, and even its r=0 baseline impure. Yet the
differencing estimator cancels most of that: what survives is 5-8%, **and the
sign is not consistent**, so it is bias rather than a correctable offset.

Verdict: material for a paper whose headline margin is "a few milliseconds", but
it does not by itself flip a conclusion. The bigger threat is C4.

**A correction to my own earlier claim.** I wrote that co-located "structurally
cannot exceed 2 riders", citing the harness's `--connector` help text ("riders
cap at 2 however many are offered") and the archived `curve_32b_v2.json`. This
fresh co-located run created 4 and 6 riders without difficulty — `cpu: 4` and
`cpu: 6` with GB/step rising 1.115 -> 2.290 -> 3.464. So the cap is
configuration-dependent, not structural, and whatever pinned the archived run at
2 was something else. The substantive point stands unchanged: that archived
dataset's r>=4 rows are still invalid, because its GB/step *fell* from 1.115 to
0.543 while this run shows it need not.

### C1-orig — the original concern, for the record

**The paper's `F = 12.3 ms` anchor comes from a co-located run, and rests on
about one usable rider point.**

It is derived from `curve_32b_v2.json`, which ran **without** `--connector`, i.e.
co-located, not the P/D-emulated decode-node setup ground rule 1 requires. The
same file is the source of the `0.86x` operating point quoted in the intro, §6
and the EuroHPC application.

Worse, its rider axis is not real above r=2: at r=4, 6 and 8 the engine admitted
only **2** riders, and `stream_gb_per_step` *falls* from 1.115 (r=2) to 0.543
(r=4) — fewer bytes moved at a higher nominal rider count. So r=4/6/8 measure
degradation at a fixed 2 riders, exactly as that file's own provenance warning
says. Usable rider points: r=1 and r=2.

I am proceeding as planned (M0 re-measures both arms), but flagging now because
if the connector A/B shows a material gap then **the `0.86x` figure and
Figure 1's cost curve are both affected**, and that is a paper-level decision,
not mine.

### C4 — **REFUTED 2026-09-30. My hypothesis was wrong; the paper is right.**

I proposed that the "not affine" rise was admission-ramp residue. Tested it by
holding `long_len` at 136 and raising `short_len` to 24 so the short run clears
the ramp. The gate passed 4/4, and `ramp_end` came out **identical in the short
and long runs at every rider count** (0/0, 2/2, 6/6, 10/10), so the ramp
genuinely cancels — 127 to 136 steady steps in the long run, the short run always
past it, `max_waiting_steady = 0` throughout.

The rise survives almost untouched:

| dT/rider | r=2 | r=4 | r=6 |
|---|---|---|---|
| s8 (published shape) | 8.96 | 10.97 | 15.49 |
| s24 (ramp cleared) | 8.67 | 10.97 | 15.11 |

Ramp residue is **<=3%** (dT 17.91 -> 17.35, 43.90 -> 43.86, 92.93 -> 90.66). It
does not explain a 70% rise.

**And the real explanation is the paper's own kink model.** Per-rider link time is
`u/X = 21.4 ms` at ctx 2048 and the knee is `r* = 2.50`:

| r | step | dT | dT/rider | marginal cost of the NEXT rider | kink residual |
|---|---|---|---|---|---|
| 2 | 70.92 | 17.35 | 8.67 | **8.67** | 17.35 |
| 4 | 97.44 | 43.86 | 10.97 | **13.26** | 8.25 |
| 6 | 144.23 | 90.66 | 15.11 | **23.40** | 10.44 |

The marginal rider costs 8.67 ms below the knee, where its transfer still hides
inside compute, and **23.40 ms above it — essentially its full 21.4 ms of link
time, now exposed**. `dT/rider` rises because the marginal rider stops being
hidden, not because of a superlinear overhead.

So §6.5's conclusion stands: `dT = F + cr` is the wrong shape, an affine fit
returns a meaningless negative fixed cost, and the kink is the right model. The
ramp mechanism I described is real and correctly characterised — `ramp_end` is
exactly the `2(r-1)` I predicted — it is simply too small to matter.

**What this confirms about the campaign's design:** the kink residual is
non-monotone (17.35 / 8.25 / 10.44) because above the knee it is a small
difference of large numbers. That is precisely why the overhead has to be
measured **below** the knee at short context, where `dT` is dominated by overhead
rather than by exposed transfer. The below-knee regime is not a convenience, it
is the only place the overhead terms are identifiable.

### C4-orig — the hypothesis as first stated, for the record

Arm A reproduced the published decode-node curve to within **0.35%**
(53.45 / 71.36 / 97.35 / 146.38 ms against 53.512 / 71.632 / 97.546 / 146.886),
so the substrate is pinned and what follows applies to the published data too.

But the purity gate failed every rider row on one condition: **requests were
queued**, `max_waiting = r - 1`.

| r | running min/mode/max | tiers | max_waiting | steps | prefill |
|---|---|---|---|---|---|
| 0 | 16/16/16 | hbm 16 | 0 | 136 | 0 |
| 2 | 18/18/18 | hbm 16, cpu 2 | 1 | 138 | 0 |
| 4 | 3/20/20 | hbm 16, cpu 4 | 3 | 142 | 0 |
| 6 | 3/22/22 | hbm 16, cpu 6 | 5 | 146 | 0 |

The tier composition is *correct* — riders are created as requested, and the
modal batch is exactly `B_res + r`. The problem is the ramp: admission staggers,
so the step count grows 136 -> 146, about `2(r-1)` extra steps.

**Why that breaks the estimator.** Step time is
`(wall_long - wall_short) / (136 - 8)`, which assumes prefill, ramp and teardown
are identical in both runs and cancel. At r=6 the ramp needs ~10 steps while the
**short run is only 8 tokens long** — so the short run never reaches steady
state, its ramp is truncated relative to the long run's, and the residue
`R_long - R_short` lands in the numerator. The step is therefore
**overestimated, by an amount that grows with r**.

That is precisely the shape §6.5 reports as the paper's "not affine" evidence:
per-rider cost rising 9.06 -> 11.01 -> 15.56 ms at r = 2, 4, 6, from which it
concludes `dT = F + cr` "fits a negative fixed cost and the fixed-versus-slope
split is not a meaningful decomposition". If the rise is ramp residue, that
conclusion is an artifact — and so is the `F = -23.4 ms` the affine fit returns.

**It is directly testable and cheap:** raise `--short-len` until the short run
clears the ramp (48 instead of 8) and see whether the rise survives. Queued at
the top of the next job. I am NOT assuming the artifact explanation is right —
the rise may be real, and the test is what decides it.

**Consequence if it is an artifact.** The "not affine" claim, Figure 1's cost
curve, the negative-`F` argument, and the choice to model a kink rather than an
affine law would all need revisiting. That is a paper decision, not mine.

### C5 — CLOSED 2026-09-30, negatively. The anchor is not stale; it is confirmed.

I flagged the risk that `fixed_cost_locator`'s 37.0 ms was stale in our favour,
because it attributed the cost to residents losing their fused attention pass and
`flash_attn.py:1034` now bypasses the two-pass split under the pool override. If
so, every projection built on it would be wrong. Re-measured:

| residents | 2026-09-04 base / +1 / F+c | today base / +1 / F+c |
|---|---|---|
| 2 | 46.91 / 86.07 / **39.16** | 46.58 / 62.47 / **15.89** |
| 4 | 48.03 / 86.87 / **38.84** | 47.71 / 63.04 / **15.33** |
| 8 | 49.91 / 86.93 / **37.02** | 49.87 / 65.34 / **15.47** |

**The cost did fall 2.46x** (38.34 -> 15.56 mean), so 37.0 is history, not the
current state — which is how the paper uses it, as the optimisation cycle's
starting point. Fine.

**The trap: that column is `F + c`, not `F`.** It is
`ITL(+1 rider) - ITL(base)` at ONE rider, so it carries the fixed term plus one
rider's marginal cost. Comparing 15.56 against the model's `F = 12.3` directly
would be an error, and an easy one to make.

**Read correctly it validates the model.** `F = 12.29 + c = 4.45` predicts
`F + c = 16.74`; measured **15.56**, ratio **0.93**. Seven percent agreement from
an independent script and a different estimator entirely (server ITL medians
versus wall-clock differencing). The anchor holds.

Supporting: still **flat across residents** (3.6% spread), so it remains a
per-step/per-layer effect and not a batch-wide gather — the original structural
conclusion survives the magnitude change. Baselines reproduced to **0.7%**, a
third independent confirmation of the substrate.

### C6 — OPEN, for the record. The published configuration runs at 98.5% of the KV pool.

My first attempt at the C4 test was wrong, and the way it failed is informative.

I held the token difference at 128 by raising both lengths (short 48 / long 176,
then 96 / 224). The KV pool on this machine is
**`GPU KV cache size: 35,472 tokens`**. Arm A's 16 residents at ctx 2048 with
long_len 136 need `16 * 2184 = 34,944` — it fits with **1.5%** to spare. At
long_len 176 they need `16 * 2224 = 35,584`, which does not fit; at 224,
`36,352`.

So the engine evicted and RECOMPUTED, and the gate caught it immediately:

* `prefill leaked into the window: 2208 tokens over 1 steps` — one request
  re-prefilling its 2048-token prompt plus ~160 generated tokens,
* 15 extra steps at r=0, where arm A had exactly 136 for 136 tokens,
* the r=0 baseline moved **53.45 -> 64.21 ms**,
* reps at r=2 ranging **45 to 90 ms**, one of them below the baseline,
* `stream_gb_per_step` *falling* (1.115 -> 0.990).

None of that is about riders. My error, caught by the gate rather than by
reading the numbers and inventing a story for them — which is what the gate is
for.

**The finding worth keeping:** the published measurement configuration sits at
98.5% pool occupancy, so anything that lengthens a sequence tips it into
preemption. That is worth knowing before anyone reruns or extends those
measurements, and it is an argument for reporting pool headroom alongside any
step-time number.

Corrected test holds `long_len` fixed at 136 and varies only `short_len`
(8 / 24 / 48, token differences 128 / 112 / 88). Pool state then stays identical
across arms and the experiment has one variable. Job 2179686.

### C3 — OPEN, low risk, will be cross-checked in M3.

**Does the `minbytes` arm really keep the launch cost intact?** It issues the
same `2*L*R` `cudaMemcpyAsync` calls but each moves one block row instead of
`n`, so the per-byte term goes to ~0 while every launch is still made. The
assumption is that a memcpy's launch overhead is independent of its size, which
is true to first order but not exactly: very small transfers can be scheduled
differently by the DMA engine.

This is why M3's standalone launch-pattern microbenchmark is not optional — it
measures per-launch cost against transfer size directly, engine-free, so the
`minbytes` reading can be checked against it rather than trusted. If the two
disagree, the microbenchmark wins and `minbytes` is reported as a bound rather
than a measurement.

### C2 — OPEN. Methodological; already handled by the plan's gate.

**At least one archived curve silently measured resident-batch growth as "rider
cost".** See F1(e): `curve_marginal_eta.json` has r=4 and r=8 with **zero riders
admitted** (`stream_gb_per_step = 0.0`, `tiers = {hbm: 16}`) yet step time rises
53.5 -> 101.5 -> 103.9 ms.

The mechanism is almost certainly that `max_num_seqs = batch + max(riders)`, so a
request the controller refuses as a rider is still served **as a resident** — the
batch grows, the step lengthens, and the harness attributes it to riders.

This is precisely the contamination Ben warned about. The decode-purity gate's
`n_hbm == B_res` assertion catches it, which is why that assertion is not
optional. No action needed beyond keeping the gate.

---

## History

### 2026-09-30 — FIRST BELOW-KNEE SWEEP. The law is affine, and it decomposes.

M1 arm `t0_c512` (ctx 512, batch 16, timers off, xfer real, 5 repeats), the first
measurement taken in the regime this campaign was designed around. Knee is
`r* = 49.29 / 5.35 = 9.2`, so r = 1..8 are all below it.

| r | dT | dT/rider | marginal | link_ms |
|---|---|---|---|---|
| 1 | 12.39 | 12.39 | 12.39 | 6.17 |
| 2 | 14.06 | 7.03 | 1.67 | 12.34 |
| 3 | 15.29 | 5.10 | 1.23 | 18.52 |
| 4 | 16.46 | 4.12 | 1.17 | 24.69 |
| 6 | 18.35 | 3.06 | 0.95 | 37.03 |
| 8 | 20.57 | 2.57 | 1.11 | 49.38 |

**`dT/rider` FALLS monotonically** — 12.39 down to 2.57 — which *is* the affine
signature, since `F/r` shrinks. Above the knee it rose. So the convexity at ctx
2048 was the knee crossing and nothing else, confirming C4 from the other
direction, and there is no superlinear term hiding here.

Least squares over the six points:

    dT = 11.67 + 1.129 * r      max residual 0.41 ms (3.3% of dT at r=1)

**Three structural claims tested, all passing:**

| quantity | ctx 512, measured | ctx 2048, paper | what the model requires |
|---|---|---|---|
| `F` | **11.67 ms** | 12.29 ms | context-INdependent if launch-bound — **agrees to 5%** |
| `c` | **1.13 ms/rider** | 4.45 ms/rider | proportional to S — **ratio 3.94 for a 4x context change** |
| hidden fraction | **82%** | 79% claimed | — |

This is the first direct test of the model's central structural claim — that `F`
is launch-bound and therefore does not follow the link or the context, while `c`
is a transfer term and does. It passes. It also resolves the earlier 9.93-vs-12.29
worry: at ctx 512 the intercept is 11.67, not 9.93.

Caveat on scope: one context, one model, timers off, six points. The factorial is
still required to separate per-block from per-byte and to test the per-layer
claim. But the shape is established and the two coefficients the C2C projection
needs are now measured rather than assumed.

### F4 — `F` is context-independent to 0.1%, and `c` splits into two terms

Second below-knee sweep, ctx 256 (knee `r* = 48.79/3.50 = 13.9`, so r=16 is the
only point above it and is excluded from the fit).

| ctx | `F` | `c` | `c/S` | hidden | max resid |
|---|---|---|---|---|---|
| 256 | **11.68** | 0.722 | 0.00282 | 79% | 0.53 ms |
| 512 | **11.67** | 1.129 | 0.00221 | 82% | 0.41 ms |
| 2048 (paper) | 12.29 | 4.45 | 0.00217 | 79% claimed | — |

Two independent contexts give `F` = 11.68 and 11.67 — **agreement to 0.1%**, and
5.3% spread across the full 8x range including the paper's own value. A
launch-bound cost must not follow the context, and now it demonstrably does not.

**`c` is not purely per-byte.** `c/S` is 0.00282 at ctx 256 against 0.00221 and
0.00217 — the short context is 27% high, the signature of a context-independent
per-rider piece inside `c`. Solving `c = B*L + k*S` on the two measured contexts:

    B*L ~= 0.32 ms/rider        k ~= 0.00159 ms per rider per token

At 64 layers, 0.32 ms spread over 128 memcpys is ~2.5 us per launch, which is
plausible — and it is exactly the `B*L*r` term the plan hypothesised. Emerging
shape:

    dT ~= 11.67          fixed, switches on with rider 1     [link-invariant]
        + 0.32 * r       per-rider launch, the 2*L*R memcpys [link-invariant]
        + 0.00159 * r*S  the transfer that does not hide     [link-bound]

**Provisional, and the reason matters:** two contexts give two equations for two
unknowns, so that split is exactly determined with no residual degrees of freedom
to test it. M2's ctx 1024 and 2048 arms are what falsify or confirm it.

**The two estimators cross-validate.** Differenced vs direct agrees within 0.2%
at r = 0..12 — except at r=16, the only point above the knee, where differenced
reads 74.21 against direct 72.18, a **-2.7%** gap. The ramp at r=16 is ~30 steps,
far beyond what an 8-token short run can clear, so differencing inflates while
the direct estimator is immune. That is the C4 mechanism appearing as a
disagreement between two independent estimators, and it confirms the direct one
is the more robust.

### F12 — `F` IS proportional to layer depth, and the layer axis confirms the per-block term

Slice C, the layer-subset knob: rider KV streamed for only the first N layers, so
depth is a free variable inside ONE model at fixed head geometry. ctx 512,
batch 16.

| layers | `F` | `c` | `F`/layer |
|---|---|---|---|
| 8 | 1.12 | 0.358 | 0.140 |
| 16 | 2.61 | 0.456 | 0.163 |
| 32 | 5.54 | 0.659 | 0.173 |
| 48 | 8.08 | 0.908 | 0.168 |
| 64 | 11.71 | 1.097 | 0.183 |

    F = -0.412 + 0.18519 * L      R2 0.9967, max residual 0.40 ms

**Proportional.** The intercept is -0.41 ms where proportionality requires ~0,
and the per-layer slope **0.18519** sits against the pooled 45-point fit's
0.18736 and the paper's operator trace at 0.19 — **three independent routes
within 2.5%**. This is the first direct test of a claim the paper rests on a
single trace at one depth, and it passes.

Extrapolating to the 70B row's 80 layers: **14.40 ms** fitted against 14.63 by
strict proportionality, 1.6% apart. So the structure is sound; the paper's 15.4
(from a 12.3 base at 64 layers) runs about 5% high.

**An unplanned cross-validation of the per-block term.** `c` scales with depth
but *not* proportionally — `c/L` falls 0.0447 -> 0.0171 — and fitting
`c = p + q*L` gives **0.252 ms/rider layer-independent** plus `0.0132*L`.

Of the three per-rider terms, only **per-block is layer-independent**:
`blocks = S/bs` contains no `L`, while bytes scale with `b_tok` which is
proportional to `L`, and the per-rider-per-layer term is proportional by
construction. So the layer-subset experiment — which never varies block size —
independently measures the per-block term at **0.252 ms/rider** against the
block-size experiment's **0.324**, comfortably inside its CI of [0.18, 0.49].

Two experimental axes sharing no machinery now agree the per-block term is real
and about **0.25-0.32 ms/rider**.

### F11 — FINAL per-block vs per-byte, and why the split matters more at long context

**Both terms are real and significant**; my earlier "per-byte, not per-block" was
too absolute. What the block-size experiment established was narrower: per-block
is not the *dominant* cause of the `r*S`-proportional cost. At ctx 512, bs 16:

| component | ms/rider | share | link behaviour |
|---|---|---|---|
| per-byte | 0.514 | 46% | **shrinks ~17x on C2C** |
| per-block | 0.324 | 29% | invariant |
| per-rider-per-layer | 0.283 | 25% | invariant |
| **total `c`** | **1.121** | | **54% link-invariant** |

Three cross-checks: the fitted per-block coefficient *predicts* the block-size
experiments (ctx 256 predicted -11.0% against **-12.4% measured**; ctx 512
predicted -14.4% against -3.7%, inside the 12% noise but low); the fit says 54%
link-invariant where the fit-free `minbytes` arm says **53%**; and the layer axis
independently recovers the per-block term (F12). Per-block's CI spans 2.7x, so
honestly **0.18-0.49 ms/rider, 19-38% of `c`**.

**And the split matters far more at realistic context than at the one measured.**
Per-block scales with `S` and is link-invariant, while the byte term gets divided
by the link:

| | per-rider | per-block | per-byte | `c` | invariant |
|---|---|---|---|---|---|
| ctx 512, PCIe | 0.283 | 0.324 | 0.446 | 1.053 | 58% |
| ctx 512, C2C | 0.283 | 0.324 | 0.026 | 0.633 | 96% |
| ctx 2048, PCIe | 0.283 | 1.295 | 1.783 | 3.361 | 47% |
| **ctx 2048, C2C** | 0.283 | **1.295** | 0.105 | **1.683** | **94%** |

**At ctx 2048 on C2C the per-block term is 77% of the per-rider cost.** A faster
link evaporates the byte term and leaves per-block standing as the dominant
per-rider cost at exactly the context that matters. So this is not a detail — it
is most of what remains after C2C. And it is the term the microbenchmark
attributed to host bookkeeping (Python loops at ~8 us per iteration), i.e.
**removable engineering, not a floor**.

Break-even with these coefficients, 16 residents:

    ctx 512   PCIe  r > 5.9   knee   9.2   reachable
    ctx 512   C2C   r > 4.9   knee 157     reachable
    ctx 2048  PCIe  c = 3.36 >= T0/B = 3.36  ->  NEVER pays
    ctx 2048  C2C   r > 7.2   knee  42.7   reachable

That is the paper's thesis with measured coefficients: at realistic context PCIe
cannot pay at any rider count, and C2C can.

### F10 — **DEGENERACY BROKEN: the `r*S` term is per-BYTE, not per-block**

ctx 512, batch 16, the two arms adjacent in time within one job:

| | `F` | `c` | max resid |
|---|---|---|---|
| bs 16 | 12.39 | 1.032 | 0.76 |
| bs 32 | 11.60 | 0.994 | 0.47 |

Bytes moved unchanged by design (1.3% mean difference); block count halved.

    pre-registered, per-BLOCK :  -37.7%
    pre-registered, per-BYTE  :    0.0%
    MEASURED                  :   -3.7%
    noise floor               :    12%

**3.7 pp from the per-byte prediction, 34.0 pp from per-block.** Halving the
block count barely moved `c`. The `dT` differences are also flat in r (+1.02,
+0.41, +1.25, +1.08, +0.98 ms), so what little moved sits in `F` — and the
-0.79 ms shift there is inside the 0.72 ms drift range measured across an hour,
between arms 12 minutes apart. Having measured drift first (F9) is what permits
that statement rather than a guess.

**This is the better outcome for the feature.** A per-byte term is link-bound and
shrinks on C2C; a per-block term would have been host work no interconnect
touches. The fitter should now select `C_link` over `D_per_block`.

**But it opens a quantitative tension.** Two measurements of the byte-bound share
of `c` at ctx 512 disagree:

* the fit's `r*S` coefficient contributes **0.851 ms/rider**
* the `minbytes` arm removes only **0.532 ms/rider**

leaving ~0.32 ms/rider proportional to `r*S` yet not recovered by collapsing the
bytes. The plausible reading is that in `real` mode part of the transfer overlaps
compute, so removing bytes does not recover their full serial time — but that is
not assertable on two contexts. **Slice A's ctx 1024 and 2048 arms tighten it**,
since the separation of the r-proportional from the `r*S`-proportional term
currently rests on exactly two points.

Status of the law: shape settled (affine below the knee, per-byte), `F` = 11.77
+/- 3%, and the open number is the link-bound share of `c` — **47% by the arms,
up to 75% by the fit**.

### F9 — PRE-REGISTERED, written before the bs=32 arm reported

Also: **a correction to a claim I made minutes earlier.** I said slice B was
underpowered, using 0.18 — the S-dependent part of the NOBUS arm — where the
relevant quantity is the **0.851** the degenerate fit attributes to `r*S`, which
at ctx 512 / bs 16 is **75% of `c` = 1.129**.

Changing block size at fixed context changes the block COUNT and leaves the bytes
alone, so the hypotheses separate sharply:

| bs | blocks/rider | `c` if per-BLOCK | `c` if per-BYTE |
|---|---|---|---|
| 16 | 32 | 1.129 | 1.129 |
| 32 | 16 | **0.704 (-37.7%)** | 1.129 (0%) |
| 64 | 8 | **0.491 (-56.5%)** | 1.129 (0%) |

**-37.7% against a 12% replicate noise floor.** The running 2x lever is already
decisive; I was wrong to doubt it.

As detection thresholds on `p`, the per-block share of `c`: the 2x lever resolves
`p > 24%`, the 4x resolves `p > 16%`. Neither can rule out a *small* per-block
term; both separate the hypotheses cleanly if the `r*S` part is mostly blocks,
which is the case that matters for the projection.

**The replicate noise floor itself**, from four runs of the identical ctx-512
configuration across two code versions and two jobs:

| run | `F` | `c` |
|---|---|---|
| t0_c512 | 11.67 | 1.129 |
| t1_c512 | 11.91 | 1.166 |
| t2_c512 | 12.25 | 1.074 |
| B_bs16_c512 | 12.39 | 1.032 |

`F` mean 12.05, range 6.0%; `c` mean 1.100, range 12.2%. So a single 5-6 point
fit carries **`F` +/-3%, `c` +/-6%**.

**And `F` drifts monotonically upward in chronological order** — 11.67, 11.91,
12.25, 12.39 across an hour of GPU work. Thermal or clock, most likely. The
consequence is methodological: **arms separated by twenty minutes are not safely
comparable**, so A/B pairs must run adjacent, which is why B2 interleaves them.

### F8 — M1 COMPLETE. The law, and a failed instrument of my own.

**The law, from the three-arm decomposition (2 contexts x 3 arms):**

    dT = F + c*r,  F = 11.77 ms  (0.184 ms/layer; paper's trace: 0.19)

`F` across six fits: 11.67, 11.68, 11.73, 11.77, 11.97, and one 12.95 artifact.
Excluding it: **mean 11.77, range 0.30 ms (2.5%)**.

Per-rider `c`, split three ways (ms/rider):

| ctx | bytes | launches | rest | total |
|---|---|---|---|---|
| 256 | 0.200 | 0.103 | 0.419 | 0.722 |
| 512 | 0.532 | 0.088 | 0.509 | 1.129 |

The launch term self-validates across contexts: 0.103/128 = **0.81 us** and
0.088/128 = **0.69 us** per memcpy, and the launch *count* is 2*L per rider
regardless of S, so these must agree — they do, to 15%.

**53% of `c` is link-invariant, and `F` is entirely so.** Substituting a 17x
link: `c` falls 1.129 -> ~0.63 at ctx 512, and `F` does not move.

**The `rest` term is a removable artifact, not an intrinsic cost.** The
engine-free microbenchmark puts a Python loop iteration at ~8 us, and
`_issue_runs` runs 64 of them per rider per step: 64 x 8 us = **0.512 ms/rider**
against **0.509 measured**. It is `narrow()`/`copy_` called from Python 64 times
per rider, not memcpy cost.

**The CUDA-graph question is closed, negatively.** Graph-removable fraction is
38.4% at 8 KB copies and **-6% to -0.4% at every larger size**. The engine's real
copies are ~1.2 MB, where a graph buys nothing — with `non_blocking=True` the
Python issue already overlaps the DMA at large sizes. The eager-only substrate
hides no opportunity.

**Automated fit, with the knee filter, 11 points:**

    A_switch_per_layer  0.18183 ms/layer  [0.1752, 0.1899]  -> F = 11.64
    B_per_rider_layer   0.00473           [0.0031, 0.0065]
    D_per_block         0.02660           [0.0216, 0.0354]
    held out 2.2% mean, 4.1% max, R2 0.988

Three independent routes to `F`: 11.77 (arm fits), 11.64 (regression), 0.19
ms/layer (the paper's operator trace). But **three models tie exactly at BIC
-21.96** — with block size fixed at 16, `blocks = r*S/16` and
`bytes = b_tok*r*S` are exactly proportional, so per-block and per-byte are
mathematically indistinguishable here. M2 slice B breaks it; more contexts
cannot.

### F7 — my phase timers cannot price a blocking call, exactly as the repo warned

M1's last gate asked for phase shares accounting for >=90% of `dT`. They account
for **214-304%** — not a pass, but evidence the instrument measures something
other than additive cost.

| r | dT | top-level phases | naive sum | top/dT |
|---|---|---|---|---|
| 1 | 12.62 | 38.41 | 103.73 | **304%** |
| 4 | 17.06 | 41.82 | 116.67 | 245% |
| 8 | 20.98 | 44.91 | 130.05 | 214% |

The phases are **nested** (`pool_cpu_override` wraps `prepare_layer` ->
`stream_layer` -> `issue_runs`, `write_token` -> `write_token_d2h_blocking`, and
the gather/attend/scatter), so `host_total_ms` double-counts. And the culprit
behind the remaining overshoot is unmistakable:

    write_token_d2h_blocking: 21.4 ms/step, 0.335 ms/layer,
    for two pageable D2H copies of ONE token's KV -- a few KB.
    At ~4 KB that is 12 MB/s, which is not a bandwidth.

It is the host **blocked on the GPU**. And the repo documents this precise trap:
the old profiler *"charged `write_token` the model's own compute, producing a
'21 ms' blocking D2H that was worth ~4 ms at a realistic queue depth."* I quoted
that warning in my own docstring and then built an instrument that reproduces it,
**to the same 21 ms**.

1. The gate was the wrong **shape**, not merely unmet — host wall time around a
   blocking sync cannot be additive, and I should have seen that when writing it.
2. The phase timers are unusable for cost attribution, which is what M3's plan
   relied on. Method replaced: `VLLM_KVPIPE_RUNGD_SKIP_D2H` prices the D2H by the
   same differencing that priced the bus copies.
3. **The xfer arms are unaffected** — they difference step times between
   configurations, so blocking behaviour is in both arms and cancels. Every
   number above stands.

### F6 — about half the exposed per-rider cost survives an infinitely fast link

The `minbytes` arm at ctx 512 — same `2*L*R` launches, one block row per copy, so
the bytes collapse (0.034 GB/step at r=8 against 1.239 real, 97% removed).

| arm | `F` | `c` |
|---|---|---|
| real | 11.67 | **1.129** |
| minbytes (infinite link) | 12.95 | **0.597** |

`c` falls 1.129 -> 0.597, so **0.532 ms/rider is byte-bound and 0.597 ms/rider
(53%) is link-invariant**. And only **8.6%** of the nominal per-rider transfer
time (0.532 of 6.17 ms) is on the critical path as bytes at all.

**If the 53% holds, the C2C implication is direct:** a 17x faster link removes
only the byte half of `c` and leaves `F = 11.67 ms` entirely untouched.

Three caveats, two of them about numbers that misbehave:

1. **`F` rose 1.28 ms, which it cannot.** `F` is the switch-on cost, not bytes;
   removing bytes cannot add to it. The cause is visible: minbytes reads *higher*
   than real at r=1 and r=2 (+0.49, +0.51) and lower from r=3, so the two lines
   cross near r~2.5 — and the low-r points are the ones that set an intercept.
   **Treated as a fit artifact, not a finding.** I am not claiming minbytes
   changes `F`.
2. **Two methods disagree at ~2 sigma on the link-invariant per-rider term.**
   The context-scaling solve (F4) gave `B*L ~= 0.32 ms/rider`; minbytes gives
   0.597. Propagating the +/-8% coefficient noise, the two-context solve is
   `0.32 +/- 0.12` — it is exactly determined, so errors amplify — against
   `0.597 +/- 0.05`. Same ballpark, not agreeing within errors. M2's extra
   contexts resolve it, and this is exactly why the plan wanted the arms *and*
   the regression rather than either alone.
3. At r=1-2 the real-minus-minbytes difference is **below the +/-0.5% noise
   floor** and comes out the wrong sign, which is why these are fits rather than
   point differences.

### F5 — the timers exceed the 2% gate at ctx 256, and the estimator gap is systematic

**Timer cost, ctx 256:** mean **+1.66%**, max **+2.22%** — over the gate, where ctx
512 was +0.69% / +1.10%. In absolute terms 1.0-1.4 ms per step for r>=1 and
**0.07 ms at r=0**, which is a clean internal check: the cost appears only when
the instrumented rider path actually runs. So shares-only is no longer marginal,
it is mandatory.

**Estimator gap (differenced minus direct), knee at r\* = 13.9:**

| r | 0 | 1 | 2 | 4 | 6 | 8 | 12 | **16** |
|---|---|---|---|---|---|---|---|---|
| arm t0 | -0.02 | -0.03 | -0.05 | -0.14 | +0.02 | -0.17 | -0.14 | **-2.74%** |
| arm t1 | -0.04 | +0.02 | 0.00 | -0.09 | -0.21 | -0.21 | -0.50 | **-2.46%** |

Below the knee the two independent estimators agree to **<=0.5%**; at the single
point above it, differencing inflates ~**2.6%**, reproduced in both arms. That
isolates the C4 ramp effect cleanly and makes the below-knee regime an empirical
requirement rather than an argument.

### F3b — a methodological error of mine: editing the harness mid-job

I edited the harness while job 2179687 was running. Each arm of a sweep is a
fresh process, so later arms picked the changes up: `t0_c512` ran without the
per-rider warmup and without a working direct estimator; `t0_c256` ran with both.
The job is not internally consistent, and nothing in its output said so — which
arm had which code had to be inferred from whether the direct column was
populated.

Fixed forward: every artifact now records `git_commit` and `git_dirty`, with
dirty recorded rather than hidden, since a dirty tree means the SHA does not
identify the code that ran. Rule for M2 onward: freeze the harness before
submitting; if something needs changing, let the job finish or cancel it.

Silver lining, but luck rather than method: `F` came out 11.67 and 11.68 across
two *different* code versions, which is a stronger robustness signal than either
arm alone.

### F3 — the noise floor, measured by accident. **Corrects an earlier claim of mine.**

`VLLM_KVPIPE_PHASE_TIMERS=2` sets a `dev` flag that nothing reads, so level 2
executes exactly the level-1 path — the docstring described CUDA event pairs
never written. My oversight, and it is left unimplemented on purpose: host issue
time is the right primary metric for a launch-bound cost, and the
byte-versus-launch split is already measured directly by the three XFER_MODE
arms with no stream operation at all.

The oversight paid for itself. Because level 2 *is* level 1, M1's `t1` and `t2`
arms were an unintended **pure replicate** — two independent runs at identical
configuration — which is the run-to-run noise floor nothing else here had
measured:

| comparison | per-step deltas | mean | sign |
|---|---|---|---|
| t0 -> t1 (timers on) | +0.24 +0.57 +0.60 +0.59 +1.10 +0.96 +0.76 | **+0.69%** | **7/7 positive** |
| t1 -> t2 (identical code) | +0.14 +0.27 +0.47 +0.45 +0.11 -0.29 -0.47 | +0.10% | 5/7, mixed |

So the timer effect is real and systematic, distinguishable by its **sign
consistency** rather than its magnitude.

**And it corrects what I reported earlier.** I said the timers shift the fitted
coefficients by +2.0% on `F` and +3.2% on `c`, and that this was outside noise:

    t0: F = 11.67  c = 1.129
    t1: F = 11.91  c = 1.166
    t2: F = 12.25  c = 1.074      <- identical configuration to t1

Two *identical* runs differ by **2.9% on F and 8% on c** — as much as the effect
I attributed to instrumentation. That shift is inside coefficient noise, and my
inference from it was wrong.

The shares-only decision stands, because the per-step effect is genuinely
systematic, but for a better-supported reason: **a six-point fit amplifies
sub-1% step noise into ~3% on `F` and ~8% on `c`**. That is a direct instruction
for M2 — more rider points, and the bootstrap CIs are load-bearing rather than
decoration.

### F2 — a controller defect, visible in the same data

The `eta` column reads **-1.01 at r=1**, -0.14 at r=2, and only turns positive
from r=3. A negative hiding efficiency is not a physical quantity; it is `F`
leaking into a per-byte one, because the controller computes
`eta = 1 - dT/link_ms` and so charges the fixed cost to the bytes. The honest
value is `1 - c/link_per_rider = 1 - 1.13/6.17 = 0.82`.

This confirms empirically what the code review only suggested about
`overlap_efficiency()` in `transfer_aware_admission.py`. Because
`VLLM_KVPIPE_ADMIT_MARGINAL` gates on that eta, **the marginal-admission
criterion is broken at exactly the low rider counts where the decision matters**,
and would refuse riders it should admit. Recorded as a finding; not fixed, since
this campaign characterises rather than optimises.

### 2026-09-30 — M0 + M0b COMPLETE. Gate satisfied.

| gate | result |
|---|---|
| correctness suite | **9/9** with instrumentation present |
| published curve reproduced | **0.35%** (53.45 / 71.36 / 97.35 / 146.38) |
| decode purity | green **4/4** once `short_len >= 24`; the failure at 8 produced C4 |
| connector A/B | answered: **5-8%, inconsistent sign** (C1) |
| fixed anchor re-measured | **15.56 ms** `F+c`, validating the model to 7% (C5) |

Three claims of mine were tested against measurement in the process. One was
confirmed (C5, the anchor holds), one was **refuted** (C4, the "not affine" rise
is real and is the kink), and one was a configuration error of my own that the
gate caught (C6, KV-pool exhaustion). That ratio is the point of running the
gates rather than reasoning about the numbers.

**Estimator hardened.** Ben rejected the argument that differencing cancels rider
seeding — correctly: riders are seeded when ADMITTED, during the ramp, so a short
run ending before the last admission pays less seeding than the long run. Rather
than patch the argument, the monitor now also measures the steady step
**directly**, as the median interval between consecutive `schedule()` calls over
the steady window. Ramp, drain, seeding and prefill are then excluded by
construction, with nothing to cancel. Both estimators are reported side by side;
agreement is the evidence and a divergence is itself a finding.

**Instrumentation cost, measured not asserted** (Ben's follow-up): 439 ns for the
clock read, 28.5 us for the whole per-step record at a 22-request batch — 0.053%
of a 53.5 ms step, forty times under the 2% gate. The end-to-end bound is
stronger: 0.35% against uninstrumented published numbers, i.e. inside noise.

### 2026-09-30 — M0 started

**Setup.** Isolated worktree created from `hybrid-kv-tiers` HEAD, *not* from
`origin/main` — the tooling default would have branched from `origin/main` and
silently dropped the entire fork (this branch has no upstream and has never been
pushed). Own `.venv` copied (8.6 GB, a real copy; beegfs has no hardlinks) so the
editable-install `MetaPathFinder` can be repointed at this tree. That finder
hardcodes `MAPPING = {'vllm': '/mnt/beegfs/benishay/vllm-kvpipe/vllm'}` and is
consulted **before** `sys.path`, so `PYTHONPATH` cannot override it; sharing the
main venv would mean editing code here and measuring the main tree's code, the
failure that produced two wrong diagnoses on 2026-09-04.

Cluster state at start: no jobs queued, `nova06`/`nova07` idle (`supernova`,
1x A100-80 PCIe, ~1 TB host RAM, 64 CPU). 1.4 TB free on beegfs. Environment:
Python 3.12.13, torch 2.9.1+cu128, CUDA 12.8.

**F1 — audit of the eight archived rider curves in
`paper_runs/20260916/rider_curve/`, done before spending any GPU time.**

Connector status recovered from the run logs
(`grep -c PipelinedDecodeBenchConnector`):

| file | connector (P/D) | tier accounting | verdict |
|---|---|---|---|
| `curve_a100_final.json` | **ON** | `cpu` = 2/4/6, tracks the request | **the only clean rider axis** |
| `curve_32b_v2.json` | OFF | `cpu` pinned at 2 for r=4/6/8 | source of `F=12.3` and `0.86x`; only r=1,2 usable |
| `curve_32b.json` | OFF | broken (`{hbm: 1}`) | garbage: r=1 step 21.7 ms vs a 54.2 baseline |
| `curve_final_a100.json` | ON | `cpu` = 2/4/6 | rejected twin; r=2 **below** baseline (47.99 vs 53.41) |
| `curve_32b_stage12.json` | ON | `cpu` pinned at 1-2 | r=2 **below** baseline (47.71 vs 51.56) |
| `curve_marginal.json` | ON | `cpu` pinned at 1-2 | rider axis not real |
| `curve_marginal_eta.json` | ON | **zero riders** at r=4,8 | see concern C2 |
| `curve_32b_decode_node.json` | ON | `cpu` pinned at 1-2 | rider axis not real above r=2 |

Consequences, stated plainly:

* **(a) Exactly one archived dataset has a genuine rider axis** —
  `curve_a100_final.json`, connector on. Its affine fit is
  `F = -23.4 ms, c = 18.8 ms`. That negative fixed cost is what §6.5 reports as
  evidence the affine split "is not a meaningful decomposition".
* **(b)** The `F = 12.3 ms` the model actually uses comes from a *different*,
  co-located run whose rider axis is not real above r=2. So the paper's two
  overhead claims are not merely inconsistent in value — they come from datasets
  differing in the P/D setup **and** in whether the rider count is real.
* **(c)** Below-baseline r=2 samples appear in two files. The harness already
  discards samples below the rider-free baseline, so these are medians that still
  landed low — the differencing estimator is noisy at this operating point. That
  is independent motivation for measuring below the knee, where `dT` is large
  relative to its own error.
* **(d)** Verified against the transcriptions the paper cites:
  `curve_a100_final` = 53.512 / 71.632 / 97.546 / 146.886 ms with per-rider
  9.06 / 11.01 / 15.56 matches `paper/data/rider_cost_phase0.json` exactly;
  `curve_32b_v2` = 54.10 / 70.79 / 113.33 / 114.79 matches
  `rider_cost_4p2b.json`. Replacing the transcriptions with the real artifacts is
  therefore safe, and is still planned.

**Decisions taken without asking**

1. Branched the worktree from `hybrid-kv-tiers` HEAD rather than the tooling
   default `origin/main`; the default was simply wrong here.
2. Paid the 8.6 GB venv copy instead of a `sys.meta_path` hack to redirect the
   `vllm` import. The hack works but fails *silently*, and silent failure here
   means measuring the wrong code.
3. Audited the archived curves before booking GPU time. That turned up F1, which
   changes what M0 has to prove.

**Next**

* Finish the venv, repoint the finder, verify `import vllm` resolves into this
  tree.
* Allocate `nova07`; `run_suite.sh` -> 9/9; confirm `ulimit -l` is `unlimited`.
* Build the decode-purity gate checker, then run the connector A/B at the 4.2b
  config with the gate live (concern C1).

### 2026-09-30 — decode-purity gate landed (`8fd0fbc34`)

Asserts the four conditions per step by wrapping `Scheduler.schedule()`, which
gives per-step ground truth rather than the tier sampler's 50 ms sampling gap.
Class A: one dict pass and two `len()` calls per step, no CUDA, nothing that can
reorder the stream — cheap enough to leave on for every run, which is the point,
since a gate you have to switch on will be off for the run that needed it.

A rep failing the gate is dropped before the estimator sees it, not adjusted: it
is not a noisy measurement of the right thing, it is a measurement of a
different thing. Rows carry `usable`, which is what the fitter filters on.

Selftest is 6/6 and covers all four conditions plus the three real failure
shapes from F1 — and confirms that legitimate admission ramp at the window edges
still passes, which matters because the estimator's short run is mostly ramp.
The short run therefore gets only the contamination checks (prefill, queueing);
the long run, which carries the decode steps the estimate is made of, gets all
four.

Also: `--connector` is now required unless `--allow-colocated` is passed, and
the output JSON records rig, connector, force_riders, dtype, repeats, block
size, k-window and the usable row count — so no future audit has to grep logs
the way this one did.

### 2026-09-30 — M1 instrumentation landed (`9e3b8f3d3`)

**Design change I made without asking, and it is an improvement on the plan.**
The plan said "extend `FAKE_TRANSFER` to rung-d", i.e. one arm that skips the
bus copies. That arm conflates two costs which extrapolate in opposite
directions, so `VLLM_KVPIPE_RUNGD_XFER_MODE` takes three values instead:

| arm | what crosses the bus | what it is |
|---|---|---|
| `real` | everything | the design |
| `minbytes` | same `2*L*R` launches, one block row each | **an infinitely fast link** |
| `nobus` | nothing; loop and views still run | the unreachable floor |

`real - minbytes` is the byte cost, `minbytes - nobus` is the launch cost, and
`nobus - baseline` is everything else the host path costs. That is the entire
decomposition the C2C projection turns on, **measured directly rather than
fitted** — and `minbytes` is a hard ceiling for any interconnect, because no
link removes a launch. See C3 for the one assumption it makes.

The pool now also counts `stream_runs` (H2D launches issued) and
`stream_bytes_nominal` (what `real` would have moved). Launch count is the
link-invariant quantity, so it is counted rather than inferred from rider count.

**Phase timers** (`VLLM_KVPIPE_PHASE_TIMERS=1`) cover the previously
uninstrumented phases: `build_step`, `stream_layer`, `issue_runs`,
`write_token`, `prepare_layer`, the rider gather/attend/scatter, and the
runner's per-step split setup. Level 1 is host issue time only — no CUDA call,
no event, no sync — which for a launch-bound cost is the right primary metric,
since the quantity being sized *is* host time spent issuing work. Level 2 adds
device time and is Class B.

Two phases are timed separately because they are **named artifacts, not
intrinsic costs**, and a projection carrying them understates the design:
`write_token`'s two pageable (blocking) D2H copies per layer — 64
host/device serialisation points per step for trivial bytes — and `build_step`'s
Python per-block block-table build.

**Sweep axes added:** `--block-size` (separates per-block from per-byte at fixed
bytes — nothing in the current model distinguishes them), `--k-window`,
`--xfer-mode`, `--rungd-layers`. The layer knob is double-gated behind
`_UNSAFE=1`, resolved once and cached, and both measurement imports in
`flash_attn.py` are hoisted to module level — an `os.environ` read per layer
would be 64 dict lookups per step on a path whose whole subject is small
per-layer costs.

Every numerically-invalid arm prints a banner and sets `numerically_valid:
false` in the output JSON, so none can be quoted for correctness or tokens/s.

### 2026-09-30 — a worktree trap caught before it cost an allocation

A fresh `git worktree` cannot run this engine, and it fails in a way that looks
like a code bug rather than a setup one. An editable install keeps its compiled
extensions in the **source tree**, and `git worktree add` only materialises
tracked files — build outputs are gitignored. So the new tree had **zero** `.so`
files against the main tree's five, and `import vllm` would have failed outright.

Checked before submitting, not after. `paper_runs` would have shown a job that
died in the first minute with an import error.

Fixed the way `WORKTREE_SETUP.md` describes for `vllm-kvpipeline-66`: symlink the
build outputs, which are products identical for both trees (`_vllm_fa3_C` alone
is 1.1 GB, so copying them is not sensible). But rather than replicate that
tree's list, I computed the set actually missing here — **56 files, against the
22 symlinks `-66` carries**, because several `third_party/triton_kernels`
subdirectories postdate that setup. Copying the older list would have left
`matmul_ogs_details`, `numerics_details`, `tensor_details`, `topk_details` and
the `vllm_flash_attn` triton ops unresolved.

`git status` stays clean afterwards, confirming all 56 are gitignored.

**Consequence to remember:** rebuilding the C extensions in the main tree now
changes this tree too. Usually desirable; worth knowing before debugging a
mystery.

**Status:** all instrumented files compile; gate selftest 6/6; layer-knob gating
verified standalone (off / ignored-without-unsafe / active / no-op when the
limit exceeds depth). **Nothing has run on a GPU yet** — the worktree's venv
copy (8.6 GB, needed so the editable finder resolves `vllm` to this tree rather
than the main one) has been running ~70 minutes, currently inside `torch`. No
measurement can start until it finishes and the tree-identity gate passes.
