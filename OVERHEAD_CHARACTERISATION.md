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

The answer is **yes, but only after two host-side fixes, and as the code stands
today the interconnect is not what is stopping it.**

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
4. **Riders do share some of `c`** — the per-rider cost is mildly sublinear,
   α̂ = 0.69 — and only the 9.2% that is bytes on the wire is per-rider by
   physics; the rest is batchable or vectorisable (§2, §3.2b). Sharing moves the
   target numbers by about three gain points (0.65→0.66× as built, 1.21→1.24×
   fixed) and changes neither verdict, **because the knee caps how many riders
   there are to share over.** That is the same conclusion as (3), reached from
   the other side.

Everything below is the derivation and the evidence. The `c` table above is
computed under the affine law; §2 and §5b give the sublinear reading and show it
does not move the verdicts.

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

Condition (1) is the one that has been missing from the discussion. `T0/B` is
what one resident contributes to throughput. If a rider costs more step time
than that, **no rider count helps** — the gain curve never crosses 1, and `F`
is irrelevant. This is why `c`, not `F`, is the core question: `F` sets *where*
break-even is, but `c` sets *whether there is one*.

> **Caveat, and it matters — condition (1) is exact only if `ΔT` is affine in
> `r`.** §2 shows it is mildly concave (α̂ = 0.69 at ctx 512). Under
> `ΔT = F + c·r^α` with α < 1 the left side of the gain condition grows
> *linearly* in r while the right side grows *sublinearly*, so a large enough r
> always wins and **"never pays at any rider count" is false in principle.**
>
> What makes it true in practice is the **knee**. The sublinearity exists only
> below it; above it M4 measured the step as the full nominal link time — linear
> in r with a large slope, predicting 254.02 ms against 254.02 measured. So the
> saving switches off exactly where it would start to compound.
>
> The consequence is that the real gate is *"does some r ≤ knee give gain > 1"*,
> and since `r* ∝ X`, **the interconnect is more central under this reading, not
> less.** Condition (1) remains the right first screen because at the target the
> knee is only 3.8 riders, where α has almost no room to act (§5), but it is a
> screen rather than a theorem.

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

## 2. Does `c` grow as riders are added? No — and it partly *shrinks*

Two sub-questions, and an earlier draft of this report got the second one wrong.

**It does not grow.** `r_max` does not exist in range. The documented >60× stall
does not reproduce: at the stall's own operating point — ctx 1024, r = 20, 5.825
GB/step, *more* bytes than the 5.37 GB/step that once failed to finish in 900 s —
the model predicts 254.02 ms and the measurement is **254.02 ms**. Clean to
r = 24 at ctx 512 as well. The stall belonged to the old gather path (22 ms/layer
host staging, 6.4 GB/s scattered `index_select`); the extent-allocation and
block-major layout fix removed it. This **removes a risk** the plan carried:
C2C's larger rider count does not walk into a blow-up regime.

**But it is not a constant either — riders do share some cost.** An earlier
draft asserted "`c` is constant in `r`" on the strength of a fitted `r²`
coefficient of −0.00027. That was the wrong statistic, and the assertion was too
strong. Fitting `ΔT = F + c·r^α` instead (`paper/model/amortization.py`):

| ctx | n | affine | power law | F-test | P(α<1) |
|---|---|---|---|---|---|
| 256 | 7 | `12.07 + 0.726·r`, RSS 0.824 | α = 0.75, RSS 0.534 | F(1,4) = 2.18, not rejected | 86% |
| **512** | **8** | `11.85 + 1.073·r`, RSS 0.530 | **α = 0.69**, RSS 0.173 | **F(1,5) = 10.28, affine REJECTED** | **98%** |
| 1024 | 4 | `10.89 + 2.114·r`, RSS 0.650 | α = 0.40 (grid floor) | F(1,1) = 7.55, underpowered | 100% |

At ctx 512 — the best-powered context — the affine law is rejected and
α̂ = 0.69. Two checks that this is a real effect and not one bad point:

* The affine **residuals carry the concavity signature**: `−0.29 −0.12 +0.12
  +0.17 +0.42 +0.23 −0.32 −0.23` — negative at *both* ends, positive in the
  middle. A single inflated point gives one large residual and no pattern.
* **Dropping r = 1 moves `c` by only −3.8%**, so the bend is not an endpoint
  artifact at the one place where `F` dominates.

**However, α is not pinned where it would matter.** The two models agree to
**1.3% at r = 8** and diverge only outside the measured range:

| r | affine | power law | gap | |
|---|---|---|---|---|
| 8 | 20.43 | 20.18 | −1.3% | measured |
| 16 | 29.02 | 26.31 | −9.3% | extrapolation, past the knee |
| 32 | 46.19 | 36.21 | −21.6% | extrapolation |
| 64 | 80.52 | 52.18 | −35.2% | extrapolation |

ctx 512 on PCIe has a knee of 8, so **the regime where the distinction matters
is unreachable on this substrate.** Both fits describe the data; they disagree
about a regime nobody measured. See §7 — this is now the campaign's leading open
question.

---

## 3. How `c` scales

### 3.1 Measured: `∝ S`, plus a floor

Per-context affine fits, below the knee only, direct (non-differenced)
estimator, A100/Qwen-32B/L=64/bs=16
(`paper_runs/20260930/overhead/02_law/A_c{256,512,1024,2048}.json`):

| ctx | `S_eff` | pts | `F` ms | `c` ms/rider | `c/S_eff` | knee | max resid | hidden |
|---|---|---|---|---|---|---|---|---|
| 256 | 335 | 7 | 12.07 | 0.726 | 0.002166 | 14.0 | 0.45 | 79% |
| 512 | 591 | 8 | 11.85 | 1.073 | 0.001816 | 8.0 | 0.42 | 83% |
| 1024 | 1103 | 4 | 10.89 | 2.114 | 0.001917 | 4.4 | 0.57 | 82% |
| 2048 | 2127 | 2 | — | — | — | 2.2 | — | — |

> **Correction: `ctx` is the prompt length, but `c` scales with what is actually
> streamed, which is the *current* sequence length and therefore grows as the
> request generates.** Taking it from the measured bytes at r=1 gives `S_eff`,
> a constant **79 tokens above the prompt** in all four sweeps — the average
> generated length over the steady window. An earlier draft divided by the
> nominal prompt and drew a conclusion from it that does not survive; see below.
> The slope is unaffected, since a constant x-offset cannot change it.

ctx 2048 has only two below-knee points and is excluded from fitting — which is
*precisely why* the campaign ran short-context. The knee is
`r* = T0·X/(b_tok·S)`, so shortening the context raises it (2.2 → 14.0) and that
is what made ≥4 rider points reachable at all on PCIe.

`c` grows with context, and **79–83% of the nominal transfer time hides inside
compute** at every context — a stable, encouraging number.

**`c/S_eff` is very nearly constant**, spanning only 19% across a 3.3× context
range and not monotonically. Fitting `c = c₀ + k·S_eff` over the three contexts
(`paper/model/c_scaling.py` regenerates this whole subsection):

```
    c = 0.059 + 1.841e-3 * S_eff    ms/rider     (A100, Qwen-32B)
```

So on the corrected axis `c` is **almost purely proportional to streamed
bytes**, with an intercept of 0.06 ms/rider — about 8% of `c` at the shortest
context.

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
which is also the mechanism behind the measured α < 1, since back-to-back
memcpys pipeline in the copy engine while a single one cannot.

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

| route | `c₀` (invariant) | `k` (per token) | ctx 512 |
|---|---|---|---|
| three contexts, 3 jobs, direct estimator | 0.059 | 1.841e-3 | 1.002 |
| 105-point within-job component fit + M3 | 0.645 | 1.255e-3 | 1.288 |
| measured | — | — | 1.073 |

Route 1 puts 91% less in the floor and 47% more in the slope, and after the
`S_eff` correction its floor is small enough that it should not be read as a
floor estimate at all (§3.1). It remains useful for the *slope*, which is what
the projection extrapolates, and that is what the sensitivity below tests. Both land within
7% of the ctx-512 measurement; **they diverge only on extrapolation**, which is
exactly where the projection lives. Running the verdict under both:

| route / configuration | `c` | `T0/B` | verdict |
|---|---|---|---|
| route 1, as built | 9.26 | 3.37 | never pays |
| route 2, as built | 5.20 | 3.37 | never pays |
| route 1, block+D2H fixed | 0.44 | 3.37 | PAYS from r = 0.7 |
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

### 5b. The knee also caps what rider-sharing can ever be worth

§2's α < 1 and §3.2b's batchable components both pay off *in proportion to r*,
so the knee bounds them too. Running the target under both laws:

| scenario | ΔT at r = knee | gain | |
|---|---|---|---|
| as built, affine | 33.01 | 0.65× | loses |
| as built, α = 0.69 on the batchable share | 32.01 | **0.66×** | loses |
| block removed + D2H fixed, affine | 6.42 | 1.21× | pays |
| …+ α = 0.69 on the batchable share | 5.56 | **1.24×** | pays |

**Sharing does not rescue the as-built case and does not decide the fixed one.**
At the target it acts over 3.8 riders and moves the gain by about three points.
It does not change either verdict.

But the leverage scales with the knee, and the knee goes as `X/S`. At **2× C2C
and ctx 2048 the knee is 30 riders**, and there `r^0.69` against `r` is a **65%
saving** on the per-rider share. So rider-sharing is nearly worthless at the
narrow operating point this campaign could reach, and potentially decisive at a
wide one — which is the same conclusion as §5 arrived at from the other
direction: **the knee is the quantity that governs, and the link sets it.**

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

0. **The rider-sharing exponent α is unresolved past r = 8, and this is now the
   leading open question** — ahead of item 3, because its leverage is larger.
   Affine and α = 0.69 agree to 1.3% at r = 8 and are **35% apart at r = 64**,
   and the gap is pure extrapolation. At the target's knee of 3.8 riders it is
   worth about three gain points (§5b) and changes nothing; at a knee of 30 it
   is a 65% saving on the per-rider share and would change a great deal.

   **It is also the cheapest thing left to measure, and needs no new
   hardware.** The knee goes as `1/S`, so shortening the context widens the
   rider axis on the A100 we already hold: ctx 256 already gives a knee of 14,
   and ctx 128 would give ≈28 — enough span to fit α over a 4× rider range
   instead of the current 8 points under a knee of 8. One sweep, a few GPU
   hours, same harness and the same purity gate. Do this before the H100 run.
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

`c` does **not grow** with `r` (no superlinearity, no `r_max` in range) and is
**affine in `S`** with a floor, `c = c₀ + k·S`. It is **mildly sublinear in
`r`** — riders do share some cost, α̂ = 0.69 at ctx 512 with the affine law
rejected — but α is unpinned past r = 8 and worth only ~3 gain points at the
target's knee of 3.8.

`c` has four components and the link divides only one. At ctx 512 on PCIe that
one is roughly half of `c`; at ctx 8192 on C2C it is 9%, because the link term
scales as `S/X` while per-block host work scales as `S`. Structurally, **only
that 9% is per-rider by physics** — the launches and the D2H are batchable, and
the per-block loop is vectorisable. **`c`'s scaling is dominated by a host
property, not by the interconnect.**

**On whether a better interconnect gives better throughput:**

* **As built: no, and not at any bandwidth.** `c` = 5.17 against `T0/B` = 3.37;
  an infinite link leaves 4.70. Robust to both apportionment routes (5.2 / 9.26).
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

## 9. Provenance

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

0. **"`c` is constant in `r`" — overstated, and corrected in §2.** The evidence
   cited was a fitted `r²` coefficient of −0.00027, which is the wrong statistic
   for the question. A power-law fit rejects the affine law at ctx 512
   (F(1,5) = 10.28, α̂ = 0.69, bootstrap P(α<1) = 98%), the residual sign
   pattern carries the concavity signature, and it survives dropping r = 1.
   **Riders do share cost.** The report also treated `c`'s per-rider-ness as
   intrinsic when only the 9.2% that is bytes on the wire actually is (§3.2b).
   Neither verdict changes — sharing moves the target gain 0.65→0.66× and
   1.21→1.24× — but the *reason* the as-built case fails shifts from "`c`
   exceeds the budget, full stop" to "`c` exceeds the budget and the knee is too
   narrow for sharing to close the gap", which puts more weight on the link, not
   less.
0b. **"`c/S` falls, so `c` has a context-independent floor" — retracted.** It
   came from dividing by the nominal prompt length when what a rider streams is
   its *current* sequence length, a constant 79 tokens longer here. On the
   corrected axis the spread is 19% and non-monotonic and the intercept is 0.06,
   so the context sweep says almost nothing about the floor. The floor itself
   stands on the layer axis, which measures it directly. Caught by its own
   selftest assertion.
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
