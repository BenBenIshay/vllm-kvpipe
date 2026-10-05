# Proposal: identify the 51% of `c` that nothing we have tried can move

**Status: FOR REVIEW. No code written.**
Author: Claude, 2026-10-05. Supersedes the optimisation half of
`OPT_PROPOSAL_PERBLOCK_D2H.md` (that one is EXECUTED: it became T1 and T2).

---

## 0. What is being proposed, in five lines

1. **One experiment, four arms, ~4 GPU-hours**, whose only job is to find out
   what the 51% of `c` actually *is*. No optimisation, no new feature code.
2. **Then, and only then, the optimisation follows from the answer.** §6 fixes in
   advance what each possible answer implies.
3. **The direction I expect to open:** if the term is per-rider overhead in the
   *second attention call* (candidate C1), then the real optimisation is the
   project's own **Solution 1 step 3 — fuse the two attention calls into one** —
   which the design memo currently predicts would be *second-order*.
4. **Nothing here revisits T6.** T6 was a correctness attempt, not an
   optimisation; it appears below once, as history.
5. **What I want from you** is a yes/no on running this, plus the §8 question
   about the layer-subset knob.

---

## 1. Why this is an identification experiment and not an optimisation

The instruction was to keep shrinking `F` and `c`. Three optimisation attempts
later the honest position is that **the largest single term in `c` has no
established identity**, and every attempt to shrink it has failed *because the
law mislabels it*.

**The three optimisation attempts against this term:**

| attempt | mechanism | fired? | effect on `c` |
|---|---|---|---|
| T3 | one strided copy per layer; launches 2Lr → 2L | yes | **6× worse** (broke contiguity) |
| T4 | batch the per-layer D2H | priced before building | **18% worse** |
| T5 | whole extents; launches 2Lr → 2L, contiguity kept | yes, verified in-engine | **null** (+0.025 vs 0.054 drift) |

All three assumed the per-rider term is **host launch work**. It is not — see
the T5 numbers below.

> **T6 is not one of these, and is not part of this proposal.** It was an attempt
> to *fix a data race*, not to make anything faster: it orders the deferred mirror
> write against the H2D copies that read the mirror. It is listed here only
> because it shares the same failure mode — a correct causal story resting on a
> wrong guess about cost, in its case that the host wait would be cheap, when it
> costs **+6.1 ms/rider**, the entire overlap. It is flag-gated **off** and should
> stay off: the divergence it was built for turned out to be the *test's*
> non-determinism, not the feature's, so there is nothing for it to fix.
> Full account in `OVERHEAD_CHARACTERISATION.md` §0.

Three designs, three correct causal stories, three wrong guesses about which
cost dominates:

```
  T5, same job, back-to-back, ctx 512, Qwen-32B, A100
    off arms   c = 0.899, 0.946, 0.892      n_runs 8,  1024 launches
    on  arms   c = 0.948, 0.928             n_runs 1,   128 launches
    effect +0.025   drift 0.054   ->  NOT RESOLVABLE
    model predicted a 0.374 ms cut: 7x the drift, and absent
```

So before building anything else, the term has to be identified. **Building
against an unidentified dominant cost is what produced all three results above.**

## 2. What is already known about the term, which is more than it looks

`b_per_rider_layer` = 0.00838 ms/rider/layer, so **0.536 ms/rider at L=64 —
51% of `c` = 0.913** at the measurement point. Three properties are established,
each by a measurement rather than an argument:

1. **Not the launch count.** T5 cut launches 8× (1024 → 128 per step, verified in
   the phase JSONL before any timing was read) with no resolvable change.
2. **Not proportional to blocks per rider.** The same T5 arms raised the window
   from 264 to 328 blocks — 33 → 41 blocks per rider, +24%. A term proportional
   to that would have risen **+0.13 ms**, which is 2.4× the 0.054 drift and would
   have shown. It did not.
3. **It scales with `r` and with `L`.** That is how the fit found it (105 points,
   within-job, CI [0.0055, 0.0119]), and the layer axis independently gave
   0.252 ms/rider at ctx 512 against this term's 0.245 — agreeing to 3%.

Together: **per-rider, per-layer, O(1) in window size, and not host launches.**

A hypothesis I floated and must withdraw: the extra attention pass over the
streamed window. Property 2 rules out the window-proportional form of it. What
survives is only an **O(1)-per-rider-per-layer** form — e.g. metadata or
`cu_seqlens` construction for the rider rows, or a fixed per-rider cost inside
the second attention call.

## 3. The four candidates, and what separates them

| # | candidate | predicts | separated by |
|---|---|---|---|
| C1 | per-rider setup in the **second attention call** (metadata, `cu_seqlens`, varlen bookkeeping) | ∝ r·L, O(1) in window | vanishes when the rider cohort is empty but the window is not; scales with *number of rider rows*, not their length |
| C2 | per-rider **event/stream bookkeeping** in `stream_layer` (`_events` dict, `wait_event` per layer) | ∝ r·L, O(1) in window | survives with transfers disabled (`FAKE_TRANSFER`), dies when the k-window path is bypassed |
| C3 | the **un-hidden transfer**, misattributed — i.e. the overlap's residue is per-rider rather than per-byte | ∝ r, weak in window | changes with **link bandwidth** at fixed r; C1/C2 do not |
| C4 | per-rider work in the **model runner's** python path (`compact_of` per rider per step, slot mapping) | ∝ r, *not* ∝ L | the **layer-count** axis: C4 is flat in L, C1/C2 are linear |

Note C4 is already partly excluded by property 3 (the term is linear in `L`),
but the layer sweep below pins it rather than inferring it.

## 4. The experiment: four arms, all discriminating, ~4 GPU-hours

Every arm holds the **rider count fixed** and varies one other axis. That is the
discipline the original campaign lacked — its `S` sweep moved bytes, blocks and
window together, which is precisely how a bytes-shaped cost got fitted into a
launch-shaped term.

**A. Layer sweep at fixed `r` and fixed window.** `VLLM_KVPIPE_RUNGD_LAYERS` ∈
{8, 16, 32, 48, 64} (the measurement-only knob, double-gated behind `_UNSAFE`,
numerically wrong by construction and used for timing only). Slope in `L` at
fixed `r` separates **C4** (flat) from **C1/C2** (linear).
*Expected cost: 5 points × 2 rider levels ≈ 1 h.*

**B. Window sweep at fixed `r`.** Vary `k_window` ∈ {2, 4, 8} and, separately,
the blocks per rider via `--rungd-bpr` with `--whole-extent`, so the streamed
window changes while the cohort does not. Property 2 predicts a flat response;
if instead it rises, the term *is* window work and C1's window-proportional form
returns.
*≈ 1 h.*

**C. Link-bandwidth arm.** `VLLM_KVPIPE_RUNGD_XFER_MODE=minbytes` (one block row
per copy: same launches, same ordering, almost no bytes) and `FAKE_TRANSFER`
(bytes removed, every gather/hook/event kept). **C3** collapses under both;
**C1/C2** survive both. This is the cleanest single discriminator in the set and
should run first.
*≈ 30 min.*

**D. Cohort-shape arm.** Fixed total window, varying rider count — e.g. 2 riders
× 164 blocks against 8 riders × 41 blocks. **C1** tracks the rider count; a
window-proportional term is flat. This is the arm that most directly tests
"per-rider" against "per-row".
*≈ 1 h.*

**Instrumentation.** Class A only (host counters, `Event.query()`), plus the
existing `build_step_bt_python`, `issue_runs`, `stream_layer`,
`write_token_d2h_async` and `wt_flush_scatter` timers. `stream_layer` minus
`issue_runs` is already the event-bookkeeping bracket, which is **C2** measured
directly — worth reading before any new timer is added.

## 5. Gates

1. **Decode purity** asserted per step on every retained point; failures
   discarded, not adjusted (the campaign's standing rule).
2. **Tree frozen** at a committed revision, HEAD printed into every log, and no
   edits to `vllm/` while a job runs. Four jobs were lost to this today.
3. **Report every effect against its off-arm drift**, and say *not resolvable*
   when `|effect| <= drift`. Three of today's claims had to be retracted for
   quoting drift-sized numbers.
4. **A negative result is the expected outcome for at least two candidates** and
   is reportable as such.

## 6. What each outcome implies — decided in advance, so the result cannot be
rationalised afterwards

* **C1 confirmed** → the term is per-rider attention-call overhead. Then
  **Solution 1 step 3 — "fuse to a single unified attention call" — is the
  high-value change**, not the second-order one the design memo predicts, and it
  is worth building. This is the outcome that reopens the optimisation programme.
* **C2 confirmed** → removable with ordinary engineering (hoist the per-layer
  event bookkeeping out of the per-rider path). Cheap, and worth doing.
* **C3 confirmed** → the term is transfer after all, the law's four-way
  decomposition is wrong in a way that *helps*: a faster link shrinks it, and the
  C2C projection is **understated** rather than overstated.
* **C4 confirmed** → host-side python per rider; contradicts property 3, so this
  outcome would mean the layer-axis fit was wrong and the law needs refitting.
* **Nothing confirmed** → report that 51% of `c` is unattributed, which is a
  weaker but publishable statement, and stop optimising blind.

## 7. What this proposal deliberately does not do

* **No optimisation is proposed.** Three have been tried against this term and
  all three failed. The next change should follow the identification, not
  precede it.
* **It does not reopen T5.** T5 is correct, its mechanism works exactly as
  designed, and it is default-off because it buys nothing measurable. If C1 or
  C3 is confirmed, T5 may be worth re-testing at long context where the slack
  fraction is small (`gen/S` = 3% at ctx 16384 against 25% at ctx 512) — but only
  then, and only with a drift bracket.
* **It does not re-measure the campaign's coefficients** on the fixed
  correctness gate. Worth doing, but separate: the gate fix changes what the
  *suite* proves, not what the *timings* were.

## 8. Open question for Ben

The layer-subset knob (arm A) is numerically wrong by construction — it streams
rider KV for the first N layers only. It is timing-only, double-gated, and
excluded from the suite, and it is the cleanest way to get a slope in `L`. **Is
that acceptable here, or would you rather arm A be dropped and C4 left excluded
on the strength of the existing layer-axis fit alone?** Dropping it costs ~1 h
of the 4 and leaves one candidate resting on a fit rather than a direct
measurement.
