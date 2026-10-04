# Overhead characterisation: how `c` scales, and whether a faster link buys throughput

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

### Notation

| | |
|---|---|
| `r` | rider count — requests whose KV lives in host memory and is re-streamed each step |
| `S` | context length in tokens |
| `X` | CPU–GPU link bandwidth |
| `L` | layer count; `b_tok` bytes of KV per token; `bs` KV block size |
| **`T0`** | **the rider-free step time** — the decode step with `B` residents and *zero* riders. Everything is differenced against it: `ΔT(r) = T_step(r) − T0`. Measured 49.29 ms (Qwen-32B, A100, ctx 512); 24.2 ms modelled for the 70B on GH200, where it is weight-read bound. |
| **`B`** | **how many HBM-resident requests fit** — a *capacity*, not a chosen batch size: `(HBM − weights − overhead)/(b_tok·S)`. 7.2 at the target. |
| **`T0/B`** | **the per-rider step-time budget**: the largest average step-time increase per rider that still improves throughput. See §1 for where it comes from. Crucially it is in **ms per rider**, the same units as `c`, which is what makes the two directly comparable. |
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

Condition (1) is the one that has been missing from the discussion. **`T0/B` is
the per-rider step-time budget**: rearranging the line above, gain > 1 is
exactly `ΔT/r < T0/B`, so a rider may lengthen the step by up to `T0/B` and
still pay for itself. (It is *not* "what a resident contributes to throughput",
which would be `1/T0` requests per unit time — a different quantity in different
units. The useful property of `T0/B` is that it is in ms per rider, so it is
directly comparable with `c`.) Since `ΔT/r = F/r + c` and `F/r` vanishes as
riders are added, `c < T0/B` is necessary: if a rider costs more step time than
that, **no rider count helps** — the gain curve never crosses 1, and `F` is
irrelevant. This is why `c`, not `F`, is the core question: `F` sets *where*
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
