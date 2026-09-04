# BLOCKER: the host tier never frees prefill blocks

> **STATUS 2026-08-13: the blocker is FIXED and validated. The mixture still
> does not beat all-HBM on PCIe, for two reasons that are now MEASURED rather
> than argued — the link hides only ~2 riders, and riders are purchasable only
> by surrendering resident slots (transient prefill residency). A separate,
> larger defect was found and fixed on the way: the host tier was running 19x
> below link speed because of our own CPU memcpy, a cost C2C would NOT have
> removed.**
>
> An intermediate result in this file's history claimed +12% concurrency for
> the mixture. **That is retracted** — see "A/B of the block-major fix" at the
> bottom. Commit 56589dab9's title carries the retracted claim.
>
> The analysis below is the original diagnosis and is kept as written.

Found 2026-08-12 while validating that the endpoint-comparison harness can
detect a difference. **This invalidates the current implementation's ability to
demonstrate the paper's central claim, and it must be fixed before the GH200
rental.** It also explains two mysteries at once.

## The defect

`scheduler.py` caps a CPU-tier request's block *growth*
(`_rungd_cap_run -> new_blocks = empty_kv_cache_blocks`). It never releases the
blocks the request already holds. Our own comment says so:

> "cap prevents block *growth* for CPU-tier requests, it does not reclaim"

So a host-tier request keeps its **entire prefill** in HBM forever. The only HBM
it frees is what it *would have grown* during decode.

## Why that guts the model

The gain law assumes a host-tier request costs `f = k_window/L` (~10-25%) of a
resident request's HBM. In reality it costs the full prefill plus the k-window.

Llama-70B fp8, 8K context, C = 19.3 GB:

| generation G | request KV | stays resident | freed |
|---|---|---|---|
| 256 | 2.77 GB | 2.68 GB | 0.08 GB (**3%**) |
| 1024 | 3.02 GB | 2.68 GB | 0.34 GB (**11%**) |
| 4096 | 4.03 GB | 2.68 GB | 1.34 GB (**33%**) |

At G=1024:

| configuration | B_cap |
|---|---|
| all-HBM | 6.4 |
| tiering **as implemented** (prefill stays) | 7.2 — **+12%** |
| tiering **as modelled** (`f`=0.1) | 63.9 |

**Worse than a small gain: a "rider" whose prefill stays resident frees nothing
at all, while still costing `S*k` of interconnect every step.** It is strictly
worse than leaving the request in HBM. The mixture cannot beat all-HBM under
this implementation, at any interconnect speed.

## It is confirmed by measurement, not just by reading code

A100, Qwen-32B, 24 requests offered at 512 ctx / 12 generated, capacity binding
(9.6 of 24 sustained, so the vacuity guard did not fire):

| arm | steady concurrency | tok/s |
|---|---|---|
| all-HBM | 9.6 | 61.4 |
| mixture | **9.6 (identical)** | **25.5 (2.4x slower)** |

Same concurrency, much slower — exactly what the analysis predicts when the
offload frees nothing. At 512 prefill / 12 generated there is essentially
nothing to reclaim.

## It also explains the `all_host` stall

With `hybrid_b_hbm_max=0`, all 20 requests are host-tier — but every one still
needs its **full prefill resident**, i.e. 20 x 1024 = 20,480 tokens. That does
not fit the resident pool, so the scheduler thrashes. One root cause explains
both open mysteries, which is a good sign the diagnosis is right.

## Why the earlier "2.67x serve-more" result did not catch this

That harness used **short prompts with long generation**, where decode-grown
blocks dominate (126 -> 48 blocks). That is the one workload shape where
capping growth reclaims most of a request's HBM. It is a favourable special
case, and the "CPU-tier requests consume ~0 main-pool blocks" framing was
overstated: they consume their whole prefill.

## What has to happen

**Release a host-tier request's prefill blocks once they are seeded to host.**
This is clearly possible — DirectKV keeps 100% of KV on the host — our
implementation simply never does it. Required:

1. after `seed_req` succeeds, return the request's resident blocks to the KV
   cache manager;
2. ensure attention for that request reads **only** from the rolling window
   (the pool override already does this — the standard pass over CPU rows was
   already masked out by opt #5);
3. make promotion re-allocate, which the two-phase commit already does;
4. re-run the full correctness ladder — this changes what memory a request owns,
   so token-identical is the gate.

**Until this lands, the GH200 endpoint comparison would measure ~no concurrency
gain and we would wrongly conclude direction A is dead.** That is the rental
this finding saves.

## Honest status of the paper's claims

- "2.67x concurrent at the same HBM" — true only for short-prompt/long-generation
  workloads. Must be restated with that condition.
- The 7/5/12 prediction assumes near-full offload. **Not yet delivered by our
  implementation.**
- The gain law itself is unaffected: it describes what a *correct* offload
  achieves. The implementation does not yet meet it.

---

# Resolution (2026-08-13)

## The fix

A host-tier request now releases **all** its main-pool blocks once the worker
has copied its history to host. Four parts, and the first alone is not enough:

1. `KVCacheManager.reclaim_tier_blocks` — nulls and frees the request's whole
   block list. Same primitive sliding-window attention already uses
   (`remove_skipped_blocks`), applied to the entire sequence instead of a
   window's tail.
2. **The worker acknowledges the seed.** Only the worker knows the D2H copy
   happened; the scheduler reclaims strictly on that ack
   (`ModelRunnerOutput.kv_tier_seeded_req_ids`). Guessing would destroy a
   request's history whenever pool setup failed.
3. **Promotion had to be repaired.** `get_num_blocks_to_allocate` is
   length-based, so an in-place null counts as a present block:
   `allocate_slots` would never refill it, and — worse — the phase-2 promotion
   gate ("blocks cover the whole sequence") would have *passed on an all-null
   table* and served garbage. Added `refill_tier_blocks` (all-or-nothing),
   `num_real_blocks` (the gate now counts non-null), and a `rewrite_block_ids`
   channel, since refilled ids replace positions the runner already holds and
   `new_block_ids` only appends.
4. Two corruption paths closed while the blocks are gone: a preempted request
   drops its now-stale pool entry, and a pool-managed request scheduled with a
   non-decode query shape has its resident write suppressed.

Gated `VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM=1`, default off.

## What is now established

| claim | evidence |
|---|---|
| The prefill is actually released | 6 host-tier requests, ~326-token prompts, 192-block pool: **133 blocks returned, free 79 → 191**, all six still decoding |
| It converts to capacity | `rungd_reclaim_unit.py`: pool exactly full at two requests → reclaim one → **a third is admitted** |
| It is correct | token-identical to vanilla, serially at long prefill and under concurrency with the freed blocks recycled |
| Promotion still works | refill is all-or-nothing; the null-table gate is closed |

## One false alarm, worth recording

The first long-prefill run showed 1 of 6 prompts diverging from vanilla, with a
vanilla-vs-vanilla control clean at the same concurrency. It was **not** a
defect:

- with the reclaim OFF the divergence is byte-identical, so the reclaim does
  not cause it;
- a **single** request on the failing prompt is token-identical to vanilla,
  with the reclaim firing;
- **stock vLLM, no tiers at all, gives different text for that prompt at
  concurrency 6 than at concurrency 1**, while being byte-reproducible at each.

The prompt sits on a near-tie; bf16 attention reductions depend on batch
composition. Half the prompts in this set behave that way. A control that holds
concurrency fixed cannot screen for it, and any arm that changes the batching —
which the host tier necessarily does — will trip it. The gate now runs vanilla
concurrent **and** serial, excludes prompts that move, and compares serially as
well so every prompt is gateable.

**The `=2` self-verify cannot gate the host tier at long sequences.** Its
reference is a flash pass over the *resident* blocks, but the block cap means a
host-tier request gets no new resident blocks during decode, so the reference
goes stale as soon as the sequence outgrows its prefill. The signature is
early layer-steps clean and later ones not (217 of 1152). `NOWRITE=0` is
necessary for that tool and is not sufficient. Use token-level gates instead.

## End-to-end, measured (A100 / PCIe, Qwen-32B, ctx 2048, 48 offered)

The document's own shape (ctx 512, 24 offered) turned out **not to bind** on
this node — all three arms sustained 22.5 of 24 at gpu-mem-util 0.92 *and* 0.83,
byte-identically, because a 512-token request is only ~128 MiB of KV and all 24
fit whatever the pool size. Binding had to come from the load: at ctx 2048 a
request is ~512 MiB, so 48 offered cannot fit. There `all_hbm` sustains **6.0**.

| SLO | arm | steady conc | tok/s | tiers |
|---|---|---|---|---|
| 100 ms | all-HBM | 6.0 | 36.8 | — |
| 100 ms | mixture | **4.8 (-19.9%)** | 6.2 | hbm 973 / cpu 611 |
| open | all-HBM | 6.0 | 36.6 | — |
| open | mixture | **6.7 (+12.0%)** | 4.1 | hbm 784 / cpu 800 |

Both rows are honest and they answer different questions.

**With the SLO gate open, the mixture serves 12% more concurrent requests than
all-HBM.** Pre-fix, the same comparison gave *identical* concurrency (9.6 vs
9.6). So the reclaim does convert freed blocks into served requests — the defect
this document is about is genuinely gone. But +12% is below the agreed 15% bar,
and it costs 9x in throughput (4.1 vs 36.6 tok/s).

**Under a real 100 ms SLO the mixture LOSES 19.9%.** That is the roofline
working as designed, not a bug: a 2048-token rider is ~512 MiB per step ≈ 20 ms
of PCIe time, so B1 refuses riders it cannot hide and admits *fewer* requests
than all-HBM would. On this interconnect the capacity exists but is not worth
spending.

This is the A100 result the project already expected — the win needs C2C — with
the difference that the blocker no longer masks it. Two caveats on the +12%:
`all_host` did not finish within 1800 s at this shape (~24 GB of KV streamed per
step at ~25 GB/s), so the three-way kill criterion is not evaluated here; and
the gain is bounded by transient prefill residency, since the reclaim happens
*after* prefill and an arriving host-tier request holds full blocks until its
first decode step.

## Three setup traps, each of which would have wasted GH200 hours

1. **The workload must bind.** ctx 512 / 24 offered never binds; all arms return
   ~22.5/24 and the mixture cannot differ. The harness's vacuity guard fires at
   >=95% of offered and 93.75% slipped under it.
2. **`b_hbm_max` must be BELOW the measured all-HBM capacity.** Admission tags
   the first `b_hbm_max` requests "hbm" and only the overflow "cpu", so with
   `b_hbm_max`=8 against a capacity of 6 the mixture reported tier samples
   `{'hbm': 1584}` — not one host-tier request — and was identical to all-HBM.
   `b_hbm_max`=8 is the GH200 plan's default. Derive it from a measured
   `all_hbm` run; do not fix it in advance.
3. **A timed-out arm must not discard the arms that finished.** `all_host`
   hitting its cap raised and threw away a completed `all_hbm` and `mixture`.
   Fixed: unfinished arms are recorded and reported, the rest still print, and a
   missing endpoint is never silently treated as zero (which would flatter the
   mixture).

---

## 2026-08-13 (late): why the mixture cannot win in a combined prefill+decode engine

After the transfer path was fixed (block-major mirrors, ~138x on `stream_layer`),
the endpoint comparison was re-run at `b_hbm_max` = the MEASURED all-HBM
capacity (6). Result: `{'hbm': 1584}` -- **zero host-tier requests**, both arms
6.0 concurrent, +0.0%.

Both settings of `b_hbm_max` fail, for opposite reasons:

| `b_hbm_max` | outcome |
|---|---|
| 3 (< capacity) | riders exist, but the resident cohort is capped below what all-HBM gets -- close to zero-sum; measured 6.7 vs 6.0 |
| 6, 8 (>= capacity) | NO riders at all; mixture is all-HBM wearing a label |

The cause is structural and has nothing to do with the interconnect. **A
host-tier request must materialise its full prefill in HBM before the worker can
seed it to host and the scheduler can reclaim.** Once HBM holds `capacity`
resident requests, the next request cannot prefill, so it is never admitted and
never becomes a rider. Riders can only be bought by giving up resident slots.

This is transient prefill residency, and it is not a second-order caveat: it is
the binding constraint on the whole design in this configuration.

**It dissolves in the P/D architecture the project has targeted throughout.** On
a decode node, KV arrives from the prefill node into host memory; a rider needs
NO HBM at any point, so concurrency becomes `B_hbm(full) + riders` -- the actual
claim. `decode_only_bench.py` already has the right shape (stage KV to CPU, free
the GPU tensors, THEN measure decode) but has no `mixture` mode, so the policy
arm cannot be run there yet. Adding it is now the load-bearing next step, not a
refinement.

NOTE for whoever runs the endpoint comparison next: `b_hbm_max` has no good
value in a combined engine. Do not spend GH200 hours looking for one.

### A/B of the block-major fix, and a retraction

Same config as the +12% run (ctx 2048, 48 offered, b_hbm_max 3), differing only
by the block-major transfer path and ASYNC=1:

| SLO | arm | conc | tok/s | before |
|---|---|---|---|---|
| 100 ms | all-HBM | 6.0 | 36.4 | 6.0 / 36.8 |
| 100 ms | mixture | 5.7 | **18.9** | 4.8 / 6.2 |
| open | all-HBM | 6.0 | 36.0 | 6.0 / 36.2 |
| open | mixture | 5.3 | **27.0** | 6.7 / 4.1 |

Throughput 3.0x (100 ms) and 6.6x (open) -- the CPU gather removal, end to end.

**RETRACTION: the earlier "+12% concurrency" does not hold.** At open SLO the
mixture is now 5.3 vs 6.0, i.e. -11.8%. `steady_concurrency` is a mean over
sampled steps, so a SLOWER arm holds requests longer and scores higher on it;
part of that +12% was the mixture being slow. Do not use steady_concurrency to
compare arms with very different service rates.

The corrected numbers agree with the roofline, which is the reassuring part.
PCIe hides ~2 riders at ctx 2048, so the mixture gets `3 resident + ~2 riders
= 5` against all-HBM's full pool of 6. **Buying 2 rider slots costs 3 resident
slots: a net loss.** The only configuration that could win is b_hbm_max at FULL
capacity with riders on top (6 + 2 = 8) -- which is precisely what transient
prefill residency forbids.

Both limits are now measured rather than argued:
1. LINK -- ~2 riders hide on PCIe; ~75 on C2C (32 MiB/layer at 900 GB/s).
2. STRUCTURE -- riders are only purchasable by surrendering resident slots,
   until KV arrives already in host memory.

The implementation tax that was masking both is gone.

---

# 2026-08-30: the mixture holds more than HBM alone

The 2026-08-13 conclusion -- "in a combined prefill+decode engine the mixture
cannot win" -- was right about the cause and is now fixed at the cause.

**The b_hbm_max sweep, completed.** There is no good value, and the reason is
that riders and residents drew on the same headroom:

| b_hbm_max | riders | total (all-HBM = 6.0) |
|---|---|---|
| 3 | ~2 | 5.3 |
| 4 | ~1.5 | 5.7 |
| 5 | ~0.3 | 5.8 |
| 6 | 0 | 6.0 |
| 8 | 0 | 6.0 |

Riders shrank exactly as b_hbm_max grew, so the sum crept toward capacity from
below and never passed it. (This also falsifies the capacity-1 hypothesis that
was recorded as untested.)

**The fix: riders that never own a block.** `preload_req` delivers a rider's KV
straight into the host mirrors, and the scheduler admits such a request with no
main-pool blocks at all -- not even for prefill, whose work is declared already
done except the final token. Three pieces were needed, and the third was only
found by testing:

1. zero-block admission (`_rungd_admit_preloaded`);
2. no prefill for a rider (declare the prompt computed);
3. **head-of-line**: vLLM's waiting loop stops at the first request it cannot
   allocate, which also stopped riders behind it that needed nothing. A
   block-starved head is now deferred so the scan continues. Without this, a
   rider that costs nothing still could not get in.

**Measured** (Qwen-32B, ctx 2048, b_hbm_max = the measured all-HBM ceiling):

| offered | all-HBM peak | mixture peak | mixture wall vs all-HBM |
|---|---|---|---|
| 48 | 8 | **48** | 89.5s vs 47.6s |
| 96 | 7 | **96** | 119.4s vs 79.3s |

The mixture matches whatever is offered while all-HBM sits at 7-8. **HBM is no
longer the binding constraint**; host memory and the interconnect are. Both
figures are LOWER BOUNDS -- the mixture hit the offered load, not its ceiling.

The throughput cost is the link, as measured: 90 riders x 512 MiB is ~49 GB of
transfer per step, ~1.96 s at 25 GB/s. The same 49 GB is ~54 ms on C2C against a
~45 ms compute step, so ~90 riders at ctx 2048 sits right at the C2C hiding
limit -- consistent with the independent ~75-rider estimate from the roofline.

**What this does NOT establish.** Riders are filled synthetically, because the
prefill-node transport does not exist yet: generated TOKENS are meaningless, as
with the bench connectors' fake fills. This is a CAPACITY result. Token
correctness of the pool path is covered bitwise by `pool_preload_test.py`,
`rungd_host_pool_test.py` and `hybrid_rungd_real.py`, and the reclaim path by
`hybrid_reclaim_correctness.py`. Riders also skip prefill (P/D decode-node
semantics), so the wall is not comparable to a baseline that prefills all.

**GH200.** The load-bearing unknown -- does the transfer hide on C2C -- needs no
real KV, so it can be measured with synthetic delivery. The real prefill->pool
transport is follow-on work for an end-to-end correctness claim, not a
prerequisite for the hiding measurement.

# 2026-09-04: the handover is real, and the transfer does not hide on PCIe

Two things were built before booking a GH200: a real prefill->pool handover so
riders stop being synthetic, and an admission controller that decides the rider
count from the interconnect and one forward iteration. Both are done. The
resulting end-to-end measurement is NEGATIVE on A100, and the reason is more
useful than the result.

## Three defects in the code the claim depends on

1. **`b_tok` was 2x wrong.** Every harness passed `--hybrid-b-tok 131072`,
   which is a 32-layer model; Qwen2.5-32B has 64 layers and therefore 262144
   bytes of KV per token. The three roofline call sites also fell back to a
   hardcoded `num_layers=32`. B1 was charging each rider HALF its interconnect
   cost, so it would admit about twice as many as can hide. It never surfaced
   because the gate was disabled (`--hybrid-itl-ms 100000`) in every run that
   mattered. Now derived from the model; a disagreeing flag logs at ERROR.
2. **The hide budget was a hand-set constant.** It is measured now, from steps
   carrying no rider -- the live EMA would be circular, since a step slowed by
   rider transfer would enlarge the budget that admitted those riders.
3. **`_ema_window` was only populated when an SLO happened to be configured**,
   so the demotion probe's baseline was empty in every run without one.

## Riders now carry real prefill KV

A prompt-keyed stash: a request prefills normally, is seeded into the host tier
by the ordinary reclaim path, and its REAL KV is kept under its prompt key,
outliving it. A later request for the same prompt is admitted with zero HBM
blocks and never prefills. The scheduler admits block-free only after the
WORKER confirms the KV exists.

**The token test cannot be decisive, and says so.** A rider's last prompt token
is recomputed by the decode kernel -- vLLM's own prefix cache does this, and
every P/D decode node must. Qwen2.5-0.5B, greedy, serial, vs a fresh full
prefill:

| | identical |
|---|---|
| round 1 (prefill+seed+reclaim) | 6/6 |
| **vanilla + vLLM's own prefix cache** | **2/6** |
| round 2 (rider on real stashed KV) | 4/6 |

vLLM's own cache perturbs MORE prompts than our handover, and ours are a strict
subset of its. So `prefill_stash_identity.py` returns INCONCLUSIVE rather than
claiming a pass. `prefill_stash_roundtrip.py` is the proof: bitwise exact
(max|diff| = 0) across block boundaries, at unaligned lengths, after the
producer's host extent is recycled and overwritten by another request, and read
back through the rolling slots the attention kernel actually uses.

## The result: correct admission, wrong model

`pd_decode_goodput.py`, Qwen-32B, ctx 2048, 12 offered, one engine differing
only in `b_hbm_max`. Phase 1 prefills every prompt and is not measured; phase 2
decodes and neither arm prefills. 12/12 completed, no errors, engine alive,
2 riders confirmed on real stashed KV.

| arm | peak | median ITL | tok/s |
|---|---|---|---|
| no_riders | 8 | 50.89 ms | 57.4 |
| riders | **10** | **88.91 ms** | 50.9 (**-11.3%**) |

Concurrency rose exactly as designed -- B1 admitted precisely the 2 riders its
roofline permits. Throughput fell, because ITL rose 75%.

**Two riders at ctx 2048 is 1.074 GB per step, which is 42.8 ms at 25.1 GB/s if
nothing overlaps. The step got 38.0 ms slower.** So ~89% of the transfer did
not hide. The controller is doing its job; the model it optimises against
assumes an overlap that is not happening.

This is the most load-bearing thing measured so far, because "the transfer
hides behind compute" is the premise the whole design rests on.

## What it does NOT establish

It does not sink C2C. The same 1.074 GB is 1.2 ms at 900 GB/s rather than 42.8.
Even at ZERO overlap, an 8% ITL budget (~4 ms on a 50 ms step) buys ~3.3 GB per
step, about 6 riders at ctx 2048 -- 8 residents + 6 riders is +75% concurrency
for +8% ITL. That is a lower bound assuming the overlap is never fixed; C2C's
36x bandwidth advantage swamps the overlap failure.

A single point also cannot separate a FIXED cost of using the host path at all
from a PER-BYTE cost that fails to overlap, and those extrapolate to very
different C2C numbers. So the rider count was swept, and the fit REVERSES the
reading above.

## The transfer does hide. A fixed cost is what kills it.

`rider_hide_efficiency.py`, same config, all four points clean (12/12
completed, no errors, engine alive, delivered riders exactly matching the cap):

| riders | peak | median ITL | tok/s |
|---|---|---|---|
| 0 | 7 | 49.91 | 57.3 |
| 1 | 9 | 86.12 | 50.0 |
| 2 | 10 | 87.93 | 51.0 |
| 3 | 11 | 96.95 | 51.8 |

Least squares on the rider points: **fixed +29.59 ms, marginal 5.42 ms per
rider**. A rider at ctx 2048 is 0.537 GB per step = 21.39 ms unoverlapped at
25.1 GB/s, so the marginal cost is a quarter of that: **~75% of the transfer
hides**. The apparent bandwidth is 99 GB/s, 3.95x the link -- which is not an
error but the signature of overlap, since part of the transfer is paid out of
time the GPU was computing anyway.

**The earlier "~89% of the transfer did not hide" was wrong.** It was inferred
from the single r=2 point, which cannot distinguish a steep slope from a large
intercept. The sweep inverted it rather than refining it. (The harness then
printed "0% hidden" from an inverted formula, agreeing with the wrong answer by
coincidence; fixed, and it now reports the apparent-bandwidth multiple.)

Note also that throughput RISES with rider count (50.0 -> 51.0 -> 51.8) while
staying under the 57.3 baseline. Each marginal rider pays for itself; it is the
entry fee that puts the arm underwater.

## What this means for GH200

Break-even -- the rider count at which concurrency/ITL beats the rider-free
arm -- is **~17 riders on PCIe**, which B1 will never allow on bytes. Not
reachable.

On C2C the marginal term is dominated by the visible quarter of the transfer,
which shrinks by 36x. Keeping the 29.59 ms fixed cost exactly as it is,
break-even lands at **~4 riders**, with everything beyond it profit. Caveat
stated: this treats the whole 5.42 ms slope as transfer, and any part of it
that is really rider-side ATTENTION compute does not shrink with bandwidth --
so ~4 is optimistic and the honest range is roughly 4-8.

So C2C is justified even without touching the fixed cost. But removing that
29.59 ms moves break-even to the FIRST rider, so it is worth a profiling pass
before booking: `profile_host_tier.py` times the pool's own entry points, and
if the pool accounts for only a few ms, the cost is in the attention path --
most likely `VLLM_KVPIPE_HYBRID_TWOPASS` re-routing the WHOLE batch rather than
just the riders, which would explain a cost of this size when nothing
rider-shaped can.

## Hunting the fixed cost: three hypotheses dead, one profiler artifact

**It is not the batch-wide gather/scatter.** `_maybe_hybrid_two_pass` re-routes
BOTH tiers through index_select/scatter once a split is published, so the
resident cohort stops being one fused flash call. That predicts a fixed cost
growing with resident count. `fixed_cost_locator.py` holds riders at ONE and
varies residents:

| residents | ITL base | ITL +1 rider | fixed cost |
|---|---|---|---|
| 2 | 46.91 | 86.07 | 39.16 |
| 4 | 48.03 | 86.87 | 38.84 |
| 8 | 49.91 | 86.93 | 37.02 |

Flat across 4x the residents, slightly DECREASING. Hypothesis dead, and the
structural fix it implied would have bought nothing.

**It is not `build_step`** -- 0.45 to 1.23 ms per step (`profile_host_tier.py`).

**It is not `write_token`'s blocking device->host copy, and that reading was a
profiler artifact.** The pool profiler attributes ~21 ms of CPU per rider per
step to `write_token`, which appends ONE token (a few KB over 64 layers). The
blocking `keys.to("cpu")`, twice per layer, looked like an obvious pipeline
drain. `write_token_d2h_microbench.py` sizes it against a REALISTIC queue,
because a blocking copy on an idle GPU is nearly free:

| GPU work queued | blocking | async | gain |
|---|---|---|---|
| ~3 ms | 4.78 | 1.50 | 3.2x |
| ~14 ms | 18.71 | 14.64 | 1.3x |
| ~50 ms (engine-realistic) | 62.39 | 58.30 | 1.1x |

At engine load the blocking form costs ~4 ms, not 21. Both variants sit near
60 ms because they are waiting on GPU work that has to happen anyway -- so the
profiler's CPU column charges `write_token` for the model's own compute. This
is the same class of mistake as the earlier profiler bug that put
`torch.cuda.synchronize()` inside its timing wrapper and made async
double-buffering look worthless; it has now misled this investigation twice,
in opposite directions.

The async patch was written and NOT applied: ~4 ms of a 37 ms problem, paid
for with async-ordering risk in three paths that read the mirrors from the CPU
(`stash_from_host`, `restore_layer`, the `host` property). Bad trade before the
real cost is known.

**What is left.** ~37 ms, switched on by rider #1, flat in both resident and
rider count: about 0.58 ms per layer over 64 layers. Insensitive to size but
sensitive to existence is the signature of LAUNCH-BOUND overhead, and the
two-pass path does two `index_select`s, a `torch.empty`, a `torch.arange`, two
flash calls and a scatter EVERY layer. The resident sweep ruled out the gather
VOLUME scaling with batch size; it does not rule out the per-call cost of doing
it 128 times a step.

That is elimination, not evidence. The next step is a real trace (torch
profiler / nsys) of one decode step at 0 vs 1 rider -- not another round of
inference from aggregate timers.
