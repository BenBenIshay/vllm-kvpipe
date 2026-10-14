# Co-serving a host-DRAM KV tier: the design, and a cost floor for the mechanism class

**Status:** working document, to be agreed before chapters 3 and 4 are rebuilt from
it. Not a paper section. Measured numbers are marked `[MEASURED]`; everything else
argues from structure.

**Revision note.** An earlier draft of this document proposed a five-component floor
whose centre was the claim that a host-tier request's *attention addressing must be
rebuilt every layer*. **That claim is false, and our own code refutes it** (§5.4).
Two of the other four were also not floors. The corrected version is below; §5.9
records what was withdrawn and why, because the error is instructive and because the
same mistake is currently in the paper.

---

## What this document is for

The paper currently reads as "we built a heterogeneous KV cache and measured it".
That is not a contribution: heterogeneous KV management is not novel on its own,
four other 2026 systems do a version of it, and our only end-to-end measurement on
hardware we own is a loss.

The claim that survives all three facts is a **cost floor for the mechanism class**
-- what *any* system co-serving two KV tiers in one forward pass must pay per decode
step -- organised by a two-parameter description of the design space:

> **`f`** = the fraction of a request's KV that is non-resident. This is what you
> save.
> **`q`** = the transfer quantum you pipeline at. This is what you pay.
>
> **`q` sets both your device footprint and the amount of compute available to hide
> the transfer behind, and it sets them in opposite directions.** That is the trade
> no co-serving design escapes.

Ours is `f = 1, q = one layer`. DirectKV is `f = 1, q = one tile`. Pie is
`f = m/L, q = one layer`. The floor is expressible in `(f, q)` plus hardware
parameters, and it *predicts* which link each design needs -- which is a design-space
result that does not depend on our speedup being real.

**Two scoping constraints this document obeys.**

1. Chapters 3 and 4 describe the *solution*, not the results. A measured number
   appears only where it is an *input* to a design argument.
2. The contribution must stand on A100 measurements plus the projection model. A
   coherent-link run later confirms a prediction; it does not supply the result.

---

## Part 1 -- The mechanism, stated once

### 1.1 What a host-tier request is

A decoding request is assigned to one of two **tiers**.

- A **resident** holds its whole KV cache in device memory, in vLLM's ordinary
  paged pool, exactly as unmodified vLLM would.
- A **rider** (host tier) holds its KV cache in pinned host DRAM. While it is
  generating it holds **zero blocks of the device KV pool**. The only device memory
  its KV occupies is a share of a small rotating region of `k` layer-slots, which
  all layers and all riders share.

The device-pool blocks a rider's prefill occupied are not merely capped; they are
*released* back to vLLM's allocator once the worker acknowledges that the KV has
reached the host mirror. This is the whole feature, and an earlier design that capped
block *growth* without releasing is the subject of §6.1.

### 1.2 What "co-serving" means operationally

Three properties, all load-bearing:

1. **Both tiers are rows of the same forward pass.** Not alternating steps, not
   separate engines, not a paused state. With `B` residents and `r` riders, `B + r`
   rows pass through every layer.
2. **The model weights are read from HBM once for the whole batch.** This is the
   entire economic argument. Decode is weight-bandwidth-bound: a step costs roughly
   `(weights + KV read) / X_HBM`. Adding a row adds its KV read and nothing else, so
   the weight read -- the dominant term -- is amortised over `B + r` rows rather
   than `B`.
3. **A rider's generation is not suspended.** It produces a token on every step it
   is scheduled.

Property 2 is why time-multiplexing cannot substitute for co-serving. A system that
pauses a request while its KV is on the host, resumes it later, and reports more
requests *served over time* has amortised nothing: each request still has the
weights read for it during its own active phase. Co-serving converts host DRAM into
**concurrency within a step**; time-multiplexing converts it into **admission over
time**. Only the first improves the per-step weight amortisation that decode is
bound by.

> **Sourcing warning.** The claim that SuperInfer time-multiplexes -- that a
> rotated-out request is not generating -- is currently the sentence separating our
> contribution from the closest-granularity competitor, and `paper/tex/refs.bib`
> states plainly that **SuperInfer's body has not been read, only its abstract.**
> There is no local copy. Either the paper gets read before this claim is final, or
> the claim is softened to exactly what the abstract supports. This is the highest
> risk sentence in the paper.

### 1.3 The assumption that makes riders additive

**Disaggregated prefill.** A rider's KV arrives from a prefill instance and is
written straight into the host mirror without being materialised in device memory.

This is not a convenience. vLLM charges device allocation at *peak*, so a request
that prefilled into the device pool and then released would still have raised the
peak, and the batch is bounded by peak residency. In a combined engine, once device
memory holds `capacity` residents the next request cannot prefill, so it never
becomes a rider: **riders could only be bought by giving up resident slots, and the
mixture could not beat all-resident at any link speed.**

`[MEASURED]` `paper/BLOCKER_PREFILL_RESIDENCY.md`. The note there -- that the
resident cap has no good value in a combined engine, and not to spend rented GH200
hours looking for one -- is why the decode-node premise is an assumption in the paper
rather than a hidden implementation detail.

The enabling code path is `preload_layer` / `preload_req`
(`vllm/v1/hybrid_rungd_pool.py`), which fills host-mirror rows from an external
source and never touches HBM, as distinct from `seed_layer` / `seed_req`, which
gather from resident device blocks and exist for migration and single-engine
experiments.

---

## Part 2 -- Building blocks

### 2.1 The host pool and its extent allocator
`vllm/v1/hybrid_rungd_pool.py` -- `RungdHostPool`

Owns a **private host block-id space** `[0, h_max)`, the pinned host KV mirror
indexed by it, and the per-request mapping. Each request gets one **contiguous
extent**, so its blocks are always an ascending run.

*Why:* with scattered ids, assembling a request's KV needs a gather, and a host-side
gather is the one cost a faster interconnect cannot touch (§6.2).

### 2.2 The rolling device region
same file -- `rolling_k[i]`, `rolling_v[i]`, `i in [0, k)`

`k` layer-slots on the device, **shared by all layers**: layer `l` reads slot
`l mod k`.

*Why:* this is the device-memory saving, and `k` quanta of residency is what buys
the overlap. §5.3 is about how little of it is actually needed.

### 2.3 The two address spaces and the per-step compaction
same file -- `build_step`, `_plan_extents`, `compact_of`

Each step, the active riders' scattered host block ids are mapped onto a dense
**compacted** range `[0, W)` addressing the rolling slots, producing (i) the runs
that drive the transfers and (ii) an `[r, max_blk]` block table for attention.

**This happens once per step, not once per layer** -- the point §5.4 turns on.

### 2.4 The deferred token-write path
same file -- `write_token`, `flush_token_writes`

Each layer, the new K and V are written into the rolling slot on the device (so the
rider can attend its own key this step) and copied asynchronously into a single
pinned staging tensor -- **two copies per layer covering all riders at once**, not
per rider. The host-side scatter into the authoritative mirror is **deferred to the
top of the next step** behind one event wait, and on the fast path is one
`index_copy_` per tensor for the whole step.

*Why:* the mirror must be correct before it is re-read, but it is not re-read until
the next step's transfers, so the synchronisation moves off the critical path.

### 2.5 The two-source attention path
`vllm/v1/attention/backends/flash_attn.py` -- `_pool_cpu_override`;
`vllm/v1/kv_tier_backend.py` -- `KVTierBackend`, `StagedHostBackend`

Per layer: make the rider KV readable, write the new token, run a second attention
call over the rider rows against the rolling slots, and scatter the result into the
model's output tensor.

The `KVTierBackend` protocol separates "which requests live where" (policy) from
"how non-resident KV is stored and read" (mechanism), and admits a zero-copy
host-reading kernel in the same slot -- the sense in which a competing mechanism is a
*substitutable component* rather than a rival system.

### 2.6 The admission controller
`vllm/v1/core/sched/transfer_aware_admission.py`

Decides tier at admission and, under the scored policy, migrates. The binding gate
is a **byte roofline**: admit a rider only if the step's rider bytes fit a measured
hide budget.

Two points worth keeping:

- The hide budget is an EMA of step time measured **only on steps with zero
  riders**. Otherwise a step already slowed by rider transfer enlarges the budget
  that admitted those riders -- a loop that inflates itself.
- Tier affinity is **temporal, not contextual**: idle time and remaining generation,
  not sequence length. Offloading frees bytes proportional to context and costs link
  bytes proportional to context every step, so *context cancels* and cannot
  discriminate between tiers at all. This is the same cancellation chapter 3 turns
  on, surfacing in the policy layer.

---

## Part 3 -- The two address spaces, and why they cannot be one

The cleanest forced-by-the-substrate argument in the system; belongs in chapter 4
close to verbatim.

The point of the feature is to **release** a rider's device blocks. Under vLLM's
allocator, releasing a block nulls the owner's block-table entry and returns the id
to a free list from which it is promptly reissued. So a mirror keyed by *device*
block id would, after a release:

1. **lose the read mapping** -- the owner's entry is now null; and
2. **alias** -- the reissued id now denotes another request's data.

Both follow from the release, which is the thing we are trying to do. Hence a private
host address space, freed only when the request finishes.

`[MEASURED]`, in the sense that matters: an earlier version mirrored by resident
block id and **corrupted KV under block reuse**.

The predecessor design (rung-c, `_build_rolling_compaction` in
`gpu_model_runner.py`) keys its compaction by resident block id *because it frees
nothing*. That is why it validated the two-source attention arithmetic and could
never deliver the capacity win: the property that makes the keying safe is the
property that makes the feature pointless.

---

## Part 4 -- The per-step flow

`B` residents, `r` riders, `L` layers. CP = critical path; OV = overlapped.

### 4.1 Once per step, before layer 0

| # | what | CP? |
|---|---|---|
| S1 | scheduler: observe the previous step's time (feeds the hide budget only if that step had zero riders); plan migrations; admit | CP, trivial |
| S2 | scheduler: a rider past prefill gets an empty block allocation -- **the capacity win** | CP, trivial |
| S3 | scheduler: release main-pool blocks for riders acknowledged last step | CP, trivial |
| S4 | runner: classify rows by tier; build rider row and query index lists | CP, `O(B+r)` |
| S5 | pool: **flush** the previous step's deferred mirror writes -- one event wait, two vectorised host scatters | CP, normally already satisfied |
| S6 | pool: plan extents into runs and compact bases; build the `[r, max_blk]` block table and the `seqused` vector | CP, `O(r log r)` |
| S7 | runner: publish the step's index tensors to the device; mark rider rows `-1` in the slot mapping so vLLM's cache write skips them | CP, a few small transfers |

### 4.2 Per layer, L times

| # | what | CP? |
|---|---|---|
| L1 | QKV projection, RoPE, vLLM's cache write (rider rows write nowhere) | CP, device |
| L2 | **standard attention over all `B + r` rows**, rider rows masked to one key | CP, device |
| L3 | **transfer**: issue this layer's rider KV from the pinned mirror into rolling slot `l mod k` -- one contiguous copy per run per tensor | **OV**; the host issue is CP |
| L4 | gather this step's new K and V for the rider rows | CP, device |
| L5 | write them into the rolling slot (device); copy them asynchronously toward host staging -- two copies for all riders | device write CP; copy **OV**, awaited at next step's S5 |
| L6 | gather rider queries; **second attention call** over `r` rows against the rolling slots, reusing the step's one block table | CP, device |
| L7 | scatter the result into the model's output tensor | CP, device |

Plus, per layer, **one stream wait and one event record** to order the prefetch
against compute. §5.6 argues this is the real content of the fixed cost.

### 4.3 What the flow makes visible

- **The transfer is overlapped and the host issue is not.** L3's bytes hide under L2
  and L6; S6's bookkeeping, L4-L7's per-row movement and the per-layer ordering ops
  hide under nothing. That asymmetry is why a faster interconnect buys **the knee**
  rather than a cheaper rider.
- **Nothing prefetches across the step boundary.** Layer 0's transfer is issued
  synchronously, so at least one quantum per rider is *always* exposed. §5.1.
- **The resident cohort is not isolated from the riders.** On the shipped path L2
  runs over all `B + r` rows with rider rows masked to one key, and L6 overwrites
  those rows. So "two tiers, two disjoint passes" describes the rung-c predecessor
  and is **wrong for the shipped path**, which is one full pass plus a rider override
  (`gpu_model_runner.py:1139` -- the resident index set is empty). Chapter 4 must say
  this; it is exactly what a reviewer who reads the code will find.

---

## Part 5 -- The cost floor

### 5.0 What a floor claim has to do to be worth making

A floor is a **minimum over implementations**. Three consequences, each of which
killed a component of the first draft:

- A number measured in our engine is not a floor unless the *reason* for it is
  structural. "Our Python loop cost 5.17 ms" is a maximum over one implementation.
- A component is only (a) if **no** co-serving design escapes it. If one shipped
  system escapes it, it is not a floor -- it is an axis.
- The claim must be falsifiable: **a system reporting a per-step cost below an (a)
  row has either changed the mechanism or mismeasured.**

Tags: **(a)** forced by the mechanism; **(b)** forced by a substrate with a stated
property P, written so a reader can check whether their substrate has P; **(c)**
ours, and therefore *not* part of the floor.

### 5.1 The bytes -- (a), an identity, plus a pipeline-fill term

Each rider's layer-`l` KV must be readable by a GPU kernel before layer `l`'s
attention. Per step:

```
bytes = b_tok * sum_i S_i        b_tok = 2 * L * n_kv * d_head * sizeof(dtype)
```

Hideable iff the *rate* condition holds, `bytes / X <= T_compute`. Rearranged at
fixed context, this is the **knee**:

```
r* = T0 * X / (S * b_tok)
```

Two refinements the first draft got wrong:

1. **The rate condition is not sufficient.** The per-layer deadline chain has a
   pipeline-fill term: nothing prefetches across the step boundary, so the first
   quantum is always exposed. For `q` = one layer that is
   `b_tok * S / (L * X)` per rider, i.e. `1/L` of the per-rider byte time. So the
   paper's "the per-rider cost contains no bandwidth term at all" is true *at
   measurement resolution* -- 0.0075 ms of a 0.61 ms coefficient at the target --
   **not structurally.** State it as `c >= b_tok*S/(L*X)` and say it is 1/L of the
   byte time. This matters because it is the one place the knee and the per-rider
   cost are not independent.
2. **`X` is the achieved bandwidth of the chosen primitive, which is a host
   property.** `[MEASURED]` 2.3-4.3 GB/s from a strided pinned source against
   22-25 contiguous, on the same link. Quoting a link's nameplate figure as `X`
   smuggles in an assumption about the transfer primitive.

**Riders are integers.** The usable count is `floor(r*)`, so gain is a *sawtooth* in
context. Reading an operating point off the smoothed curve is an error this project
has made and corrected (`paper/model/operating_point.py`).

### 5.2 Two extra HBM passes -- (a) for any staged design, zero for direct access

**This is the component the first draft missed entirely**, and it is first-order in a
bandwidth-bound step.

A staged design moves each rider KV byte across HBM **twice**: the copy engine writes
it into the window, and the attention kernel reads it. A resident pays one read. So

```
dT >= (1 + w) * r * S * b_tok / X_hbm        for a staged design
dT >= r * S * b_tok / X_hbm                  for direct access
```

This is already charged in our own capacity model (`killer_v2.py`, `stage=2`) and
already named in Appendix F -- it simply never made it into the floor list. It is
also **the one component a zero-copy kernel genuinely escapes**, which makes it the
sharpest structural statement separating the two mechanisms, and far better grounded
than the addressing claim it replaces.

### 5.3 Device residency -- (a), and the bound is `2q`, not two layers

To overlap transfer with compute you must be filling one region while attention reads
another. So residency is at least **two transfer quanta** -- and the quantum is a free
design parameter:

```
device KV residency >= 2q
saving = 1 - (k * q_bytes) / (total KV bytes)
```

The first draft said "at least two *layers*", bounding the saving by `(L-2)/L`.
**That is wrong, and three things refute it, one of them ours:**

- With streaming-softmax / LSE merging you can attend over chunk `j` while chunk
  `j+1` lands. Residency is then two *chunks*. DirectKV's decode kernel does exactly
  this at tile granularity in shared memory.
- **Our own Appendix F already says so**: a zero-copy kernel "overlaps its fetch with
  the arithmetic of the attention tile it is already computing -- a window of one
  tile."
- Our own related-work table lists DirectKV's device KV as "none" while running. We
  cannot keep both that row and a `(L-2)/L` bound.

#### 5.3.1 The real trade, and the actual "one mechanism seen twice"

`q` does two things at once, in opposite directions:

> **A larger quantum costs more device memory (`2q`) and buys more compute to hide
> the transfer behind (one quantum's worth of arithmetic). A smaller quantum costs
> almost no device memory and buys almost nothing to hide behind.**

At decode arithmetic intensity -- roughly 4-8 FLOP per byte -- a tile's own
arithmetic is nearly nothing. So:

- `q = one tile` (DirectKV): footprint ~0, but essentially no compute to hide
  behind, so **the transfer must be fast rather than hidden** -- which is why that
  design needs a coherent link and reports itself as a capacity mechanism on PCIe.
- `q = one layer` (ours, Pie): footprint `2/L` per request, and a whole layer of
  compute to hide behind -- so **a slower link can still be hidden**, at the price of
  device memory.

That is a genuine trade in hardware parameters, it is falsifiable, and **it predicts
which link each design needs.** It replaces the withdrawn framing and is strictly
stronger: the withdrawn version survived contact with none of the three systems,
including ours.

#### 5.3.2 Where we must not overclaim

Two things, both checkable, both currently wrong in the paper:

1. **The 0.92 vs 7.20 ms figure is not a `k` ablation.** The arm is `async = false`
   with `k_window = 4` (`paper/data/overhead/12_asyncbug/ASYNC_off.json`), and the
   slot rotation happens regardless of the async flag -- only the side-stream
   prefetch is gated. `04-design.tex:143` attributes to "the `k>1` lookahead" a
   measurement that varied **whether the transfer is issued ahead of its
   consumption**. Relabel it; the pillars table already words it correctly.
2. **`k = 4` is not supported by the one sweep we have.** `[MEASURED]` at ctx 512,
   step time at `r=1` is 61.24 / 62.10 / 61.91 ms for `k` = 2 / 4 / 8, and at `r=8`
   69.16 / 69.71 / 69.81 -- with **`k=2` marginally best**, all inside the documented
   drift. Meanwhile `killer_v2.py` charges every rider `k/L = 4/64` of device memory
   for it. We are paying twice the minimum residency for slack no measurement
   supports. Either justify `k=4` as jitter tolerance -- noting that the knee contains
   no `k` at all, so `k` buys burst absorption, not hiding capacity -- or sweep it
   properly.

Also: the rolling region is sized `w_max = h_max` (`gpu_model_runner.py:1387`), i.e.
by *policy maximum* rather than by the live cohort. The saving `(L-k)/L` holds only
**conditional on the window tracking the cohort**; stated unconditionally it is
wrong, and at production settings that allocation does not fit.

### 5.4 Addressing two KV sources -- **withdrawn as a floor**

The first draft's centrepiece was: a resident's blocks are static across all `L`
layers, but a rider's roll, so a fused call needs the rider block table rebuilt per
layer -- hence "the saving and the addressing cost are one mechanism seen twice".

**Our own code refutes this.** `build_step` builds the rider block table **once per
step** -- one slice assignment per request, `hybrid_rungd_pool.py:1042` -- it is
uploaded once
(`gpu_model_runner.py:1437`), and every layer attends with *that same tensor* while
only the slot **contents** rotate (`flash_attn.py:705`). `04-design.tex:113` already
says so in words: "every layer reuses one block table while the slot contents rotate
beneath it." **The rolling window costs zero per-layer addressing work.** The rider's
addressing is already address-stable within a step.

A whole-plane rotation is a **base-pointer change**, not an address rebuild. Pie is
the independent existence proof: it rotates at layer granularity with a two-buffer
window and pays no addressing work for it.

So the second kernel call exists for a different and much smaller reason: the window
is a **separate tensor**, and the kernel takes one `key_cache` pointer. What survives
is:

> Attention must be told, per row, which address space to read. That is `O(B)`
> metadata per step, and **in a fixed-kernel ecosystem it materialises as one extra
> dispatch per layer** -- `Omega(L)` launches per step. **(b)**, not (a).

And it is **not distinctive**: single-tier DirectKV already issues several attention
calls per layer, one per sequence-length bucket. "One extra call per layer" is not a
cost of co-serving.

**Consequence for our backlog, in the honest direction.** `H-ridercost.tex:128` and
`OVERHEAD_CHARACTERISATION.md:816` say fusing needs "the rider cohort's table rebuilt
per layer on the hot path". That **overstates the obstacle**. A fused single-table
formulation needs `k` tables built once per step -- `O(k*r)` rows, device-side,
microseconds -- plus a pointer selection per layer. With a custom kernel it needs one
scalar block-id offset per layer and a single table. Fusion is cheaper to reach than
we have been telling ourselves; what it *buys* is still only the launch half of
~1.2 ms, per the direct ablation (§6.5).

Also newly available: the per-rider gather/scatter half (`[MEASURED]` 0.134
ms/rider, from the `index_select`s) is avoidable **without** fusing, by ordering
rider rows contiguously in the batch so the gather becomes a `narrow()` view. vLLM
already reorders batches per step. That is a candidate identity for part of the
unexplained ~51% of the per-rider coefficient, and it is per-rider, per-layer, and
`O(1)` in window size -- exactly the constraint set the whole-extent null left
standing.

### 5.5 The new token's write-back -- bytes (a), launches **not** `2L`

Two obligations, and the first draft conflated them *and* got the count wrong.

- **The device-side write is (a) and per layer.** The rider attends its own
  just-computed key, so that key must be in the readable region before the second
  call.
- **The link crossing is (a) in bytes: `r * b_tok` per step.** The host copy is
  authoritative and must be correct before that block is re-streamed.

**What the count is not.** Our code does **two async copies per layer covering all
riders at once** -- `2L` per *step*, not per rider, which the first draft stated
wrongly about our own system. And `2L` is not forced: DirectKV's fused kernel writes
the new K,V from registers straight to the host copy -- zero copy launches, zero D2H
sync. On a non-coherent link the floor is **one** crossing per step, if the layer's
KV-write is pointed at a per-step contiguous device staging region. Our T4 negative
result -- 18% slower -- kills a *different* candidate: `2L` extra device-to-device
copies *into* staging after the fact. Its conclusion holds only if you aggregate
afterwards rather than **writing into the aggregate**, which is free because the
layer already writes its KV somewhere.

So: `0.834 ms` at `L = 64` is **the measured launch cost of one particular pattern on
one A100 host**, and the discussion's "bounded below by 0.83 ms of unavoidable
copies" should not ship.

**Can the crossing be zero?** Only by letting the device copy be authoritative until
eviction -- at which point you have reinvented swap-on-evict and pay `O(S)` on the
return leg in the slower direction. `[MEASURED]` the swap emulation moves exactly
**2.00x** the link traffic at identical offload volume. So a non-zero return leg is
(a), and `O(1)` *bytes* is a property of cache rather than ownership semantics
(§6.3).

### 5.6 Per-layer ordering, and per-step metadata -- (a) for a copy-engine pipeline

Any design that pipelines through a copy engine needs, per layer: one stream wait,
one event record, one event wait -- plus per-step syncs guarding the mirror.
`Omega(L)` ordering operations per step, **zero** for intra-kernel pipelining.

This, and not launch count, is what the fixed cost is made of -- which is consistent
with the whole-extent null: cutting launches eightfold moved neither coefficient. It
is also why the fixed cost is a host-versus-device-speed property rather than a
device property, and our own overhead appendix already bands it over a factor of 18
on that basis.

Per-step metadata is `Omega(r)`: something must say where each rider's keys are. The
honest floor is `O(r)`, and between steps only the sequence length changes.

### 5.7 Two components that are **not** floors

Both were in the first draft. Both belong in the pillars/ablations, where they are
strong, and nowhere near a minimum-over-implementations claim.

- **`O(blocks)` metadata is a maximum, not a minimum.** The 5.17 ms figure rests on
  0.00765 ms **per block**, which is the cost of a Python `Tensor.__setitem__` *in our
  engine*. At any plausible C++/CUDA constant -- tens of nanoseconds per block -- an
  `O(blocks)` build is a few microseconds per rider at the target context and never
  binds. The *design rule* (`c < T0/B`, and compare slopes not points) is sound and
  genuinely useful; the number is ours.
- **Admission is not a floor.** It is per-step `O(#waiting)` policy work shared with
  the baseline scheduler, and our own measurement credits it rather than charging it.

### 5.8 The floor, assembled

For `r` riders, `L` layers, one decode step, a design with non-residency `f` and
transfer quantum `q`:

| term | floor | cat |
|---|---|---|
| link bytes | `b_tok * sum_i S_i` per step, hideable iff `<= T_compute * X`, with at least one quantum per rider always exposed | (a) |
| HBM passes | `(1+w)` per rider KV byte for a **staged** design; `1` for direct access | (a) for staged; the escape is the mechanism change |
| device residency | `>= 2q`; saving `= 1 - k*q_bytes/total`, conditional on the window tracking the cohort | (a) |
| ordering ops | `Omega(L)` per step for a copy-engine pipeline; `0` intra-kernel | (a) for staged |
| per-step metadata | `Omega(r)`; only `seqused` changes between steps | (a) |
| tier dispatch | `O(B)` metadata per step; one extra dispatch per layer in a fixed-kernel ecosystem | (b) |
| new-token crossing | `r * b_tok` bytes per step; `O(1)` launches, `0` on a coherent link | (a) bytes, (c) our `2L` |

**And what we pay against it:** three blocking device-to-host setup drains for data
already in host memory; `2rL` transfer launches instead of `2L`; `2L` link crossings
instead of `O(1)`; four per-layer gather/scatter kernels a batch reorder would
remove; `k = 4` where `k = 2` measures the same; and a rolling region sized by policy
maximum rather than by the live cohort.

**Why stating the gap is a strength.** Our A100 result is a loss. If the floor
equalled what we pay, that loss would be the mechanism's verdict. It does not, and
the distance is itemised -- so the loss is a verdict on *this implementation on this
link*. That is the difference between a negative result and a negative result that
tells you where to look.

### 5.9 What the first draft of this floor got wrong, and why it matters

Recorded because the paper currently contains two of these errors, and because the
shape of the mistake is the same one the whole overhead campaign exists to correct.

| claimed | actual | how it failed |
|---|---|---|
| a rider's addressing must be rebuilt every layer | built **once per step**; rotation is a base-pointer change | asserted a cost from the *design intent* of the rolling window without checking what the code does. Pie and our own `04-design.tex:113` both contradict it |
| device saving `<= (L-2)/L` | `>= 2q`, and `q` is free; a tile-granular design has ~0 footprint | took our own quantum for the mechanism's quantum. Our own Appendix F says the right thing |
| `2L` D2H copies per rider, floor `0.834` ms | `2L` per **step**; floor is `O(1)` crossings, `0` on a coherent link | mistook our launch pattern for a requirement, and misread our own code's per-rider structure |
| `O(blocks)` metadata is fatal, so `O(r)` is a floor | `O(blocks)` is fatal *at Python constants*; a compiled build never binds | quoted a maximum over one implementation as a minimum over all |
| two extra HBM passes | **missing entirely** | the one first-order term, already in our own capacity model |

The common thread: **four of the five reasoned from our implementation's costs to the
mechanism's requirements.** That is the same error as fitting across the knee --
generalising from a regime to a law -- and it is worth one sentence in chapter 3,
because it is why the floor is stated with explicit (a)/(b)/(c) tags rather than as a
list of numbers.

### 5.10 What the floor says about when the mechanism pays

In model and hardware parameters only, no measured coefficient:

- A rider must cost less than a resident is worth: `c < T0 / B`. Below that, no rider
  count helps, at any bandwidth.
- The usable rider count is `floor(r*)`, `r* = T0 * X / (S * b_tok)` -- and it
  contains no `k`, so the window buys burst absorption, not hiding capacity.
- Gain is governed by how much of device memory the **weights** took, not by context:
  context appears in both the memory freed and the bytes moved, and cancels.

The third point surprises readers most, and it is the same cancellation that forces
the admission policy to score on *time* rather than length (§2.6). Chapter 3 should
make it once analytically, then show it twice -- once in the gain law, once in the
policy.

---

## Part 6 -- How the design arrived here

### 6.0 One paragraph of pre-history

The project began as **layer-wise KV pipelining**: keep `k+1` of `L` layer tensors on
the device to cut a single request's KV footprint. It regressed inter-token latency
roughly 20x against a 7 ms baseline, sync-bound rather than bytes-bound -- so the
regression itself was reframed as the object of study, an "orchestration tax" a DPU
would remove. That thesis died when the gap turned out to be in the **baseline**: the
comparison ran all-identical prompts with prefix caching on for the baseline and off
for the pipelined arm, so the "tax" was the baseline's cascade-attention speedup. At
matched prefix caching it is 2-5%. Two measurements had already undercut the thesis
independently -- CUDA graphs save only 0.5 ms at batch 32, and the host control plane
has slack at long context.

That paragraph earns its place because it is the first instance of the pattern that
recurs through Part 6 and through §5.9: **almost every wrong number in this project
came from comparing across a regime boundary.**

### 6.1 Release, not cap

The first reclaim design capped a rider's block *growth* and never released what it
held, so a rider kept its entire prefill resident forever. `[MEASURED]` it freed 3% /
11% / 33% of footprint at generation 256 / 1024 / 4096 while paying full interconnect
every step.

The verdict is stronger than "a small gain": **a rider whose prefill stays resident
frees nothing while still paying the link, so it is strictly worse than leaving the
request resident -- at any bandwidth.** The served total crept toward the all-device
capacity from below without ever passing it, at any resident cap.

Why it hid: the harness used short prompts with long generation, the one shape where
capping growth reclaims most of a request's memory. That favourable special case is
also the source of a retracted "2.67x serve-more" claim.

This is what makes the disaggregated-prefill assumption (§1.3) **load-bearing rather
than convenient**.

### 6.2 Block-major, split K/V -- not the interleaved layout

The mirror was originally one `[2, heads, blocks, ...]` tensor per layer. A run of
blocks is then two chunks strided by a whole key/value plane, and **a strided
host-to-device copy cannot be a plain pinned DMA**.

`[MEASURED]` that gather cost **22 ms per layer -- 98% of the streaming phase and 72%
of the engine's wall clock** -- while the transfer itself ran at full link speed.
Relaid out block-major with key and value separate, the per-layer cost fell to 1.38 ms
and achieved bandwidth rose from 4.87 to 24.38 GB/s.

**Why this belongs in the paper and not just the changelog:** the cost was *invisible
to the metric one would naturally watch*. It is a host memcpy, and a faster
interconnect does not touch it. On a coherent link the same code would spend hundreds
of milliseconds per step gathering in order to hide a transfer of a few -- and the
natural conclusion would have been that **the interconnect does not help**. A paper
whose thesis is "here is when a faster link pays" must show it knows how to be wrong
about that in the most tempting direction.

This is also the measurement behind §5.1's second refinement: `X` is a property of
the primitive, not of the link.

### 6.3 Cache semantics, not swap

The host copy stays authoritative and the device copy is a **cache**, so a slot can
be overwritten with nothing to save and only the new token travels back.

If the offloaded unit is instead **owned** by one side -- as a layer's KV is in a swap
design, where the device buffer is reclaimed for the next layer -- a unit that has
just had this token's K,V appended is dirty and must be persisted before reuse.
**Every offloaded byte then crosses the link twice per generated token**, on the
slower leg.

Structural, not an optimisation, so the paper prices the alternative rather than
asserting it: the swap emulation is built, default-off, and makes our per-token
traffic match a swap design's at identical offload volume. `[MEASURED]` exactly
**2.00x** the link traffic, output verified identical.

The irony worth one sentence: this disabled path was later found flushing the deferred
mirror writes on every layer, and fixing that was the largest single optimisation of
the campaign (§6.7).

### 6.4 Extent allocation -- measured worthless, then load-bearing

Contiguous per-request extents were built to kill the host-side gather. `[MEASURED]`
**no measurable benefit**: vLLM's free list is LIFO, so a steady-state cohort already
keeps a clustered block set -- one run per request with and without extents, including
after 40 rounds of churn. The entire 10.1x of that round came from the layout change
of §6.2.

They were kept as **insurance**: a guarantee by construction rather than a reliance on
allocator behaviour. The insurance paid twice over -- the one-run guarantee is what
makes the `O(blocks)` to `O(r)` rewrite possible, and what lets whole-extent issuing
collapse `2rL` launches to `2L`.

Lesson for chapter 4: a structural guarantee is worth building when it is *currently
free*, because the thing it enables may not exist yet.

### 6.5 Two-pass attention, not fused -- and what that choice actually was

Chosen for two stated reasons: a fused kernel reading two KV regions needs a dual
block table and an in-kernel region selector, a kernel we do not own; and it would
foreclose dropping in a competing zero-copy kernel later.

The value of fusing was estimated three times:

| estimate | value | fate |
|---|---|---|
| at design time | second-order | contradicted |
| by subtracting one ablation from another | +10 points | retracted |
| by **direct ablation** | +2.0 certain, +8.5 ceiling | current |

The direct ablation found the second pass costs **1.21 ms**, not the 1.94 inferred by
differencing -- high by 60% -- and that the cost **splits across both coefficients**:
the gather and scatter size with the rider count, so they live in the per-rider term,
while the fixed term holds the kernel and its launch, and a fused call still performs
the attention arithmetic.

**What §5.4 changes about this.** The *obstacle* to fusing is smaller than we have
been saying -- `k` tables once per step, not a hot-path rebuild -- while the *prize* is
unchanged and still mostly unreachable. And the gather/scatter half is reachable
**without** fusing, by reordering rider rows contiguously. That is the candidate worth
pricing next, not fusion.

> **Open item for Ben.** `paper/model/overhead_model.py:112` and
> `paper/model/fusion_bound.py:37` both still carry 1.94 for this component. That was
> a *residual* -- the fixed cost minus the mirror write minus the streaming -- then
> **attributed** to the second attention call. The direct ablation says the call is
> 1.21. So the residual is not straightforwardly wrong; rather **about 0.7 ms of it is
> something other than the attention call and is currently unattributed** -- and
> §5.6's per-layer ordering operations are the obvious candidate. Changing the
> constants without re-deriving that 0.7 ms would replace a mislabelled number with a
> missing one. Flagged, not changed.

### 6.6 The `k`-window: what it buys, and what it does not

Carefully, because the paper currently overstates both halves.

- **Issuing the transfer ahead of its consumption is load-bearing.** `[MEASURED]`
  per-rider cost 0.92 ms with the side-stream prefetch, 7.20 ms without -- the latter
  failing the `T0/B = 3.37 ms` budget by more than twofold. **This is a prefetch
  ablation, not a `k` ablation**: both arms ran `k_window = 4`. §5.3.2.
- **More slots buy nothing measurable, and nor does `k = 4` over `k = 2`.**
  `[MEASURED]` 61.24 / 62.10 / 61.91 ms at `r = 1` for `k` = 2 / 4 / 8, `k = 2`
  marginally best, all within drift.

So the slots are a **necessary condition, not the mechanism** -- the code still issues
one layer at a time -- and the specific value 4 is unjustified by anything we have
measured while being charged in full in the capacity model. Either sweep it properly
or justify it as jitter tolerance and say so.

### 6.7 The optimisation ledger, and its meta-result

Nine attempted, four landed. Figures belong in chapter 6; the **shape** belongs here:

- **Landed:** `O(blocks)` to `O(r)` addressing; deferring the mirror write off the
  critical path; unifying the mirror so the deferred scatter vectorises; and stopping
  a *disabled* feature from flushing those writes every layer -- 64 flushes per step,
  63 redundant waits, defeating the deferral since the day it landed. That last was
  the largest single win and **it was a defect, not an optimisation.**
- **Did not land:** a strided batched gather (correct, mechanism verified, **6x
  slower**); staging device-side to batch the crossings (**18% slower** off-engine);
  whole-extent issuing (launches 8x down, **neither coefficient moved**);
  preallocated buffers (nothing resolvable); and ordering the deferred scatter after
  its readers (correct, and it costs **the entire overlap**).

**The meta-result, the most transferable thing the project produced:**

> **Four of the nine had a mechanism that demonstrably fired and a benefit that did
> not follow.** Whole-extent issuing cut the launch count eightfold and moved nothing,
> which refuted the launch-bound account of the fixed cost -- and points instead at the
> per-layer ordering operations of §5.6. The strided gather achieved exactly the
> bandwidth improvement predicted for it in isolation and was six times slower in the
> engine. **A confirmed mechanism is no evidence at all about a cost.**

So every remaining candidate is priced off-engine or by ablation before it is built,
and the microbenchmark that does the pricing **carries a known-bad control** -- the 6x
regression -- and must reproduce it before the harness is trusted on an unknown.

### 6.8 What the design does not rest on

Retracted, and not relied on in chapters 3 or 4:

- an "overlap efficiency" parameter in any of its values -- **a quantity that does not
  exist**, since below the knee the overlap is complete. (Note: the retracted "hides
  about 90%" figure is still in the related-work section in two places.)
- a per-rider cost that is any fraction of link time -- removing the bytes while
  keeping the launches moves it by 0.010 ms. §5.1 refines this to `>= 1/L` of the byte
  time rather than exactly zero;
- a sublinear rider-sharing exponent -- **unidentified, not sublinear**: it ran 0.40 to
  1.085 on one arm purely as the fit window widened, and four statistical checks
  endorsed it because all four conditioned on that window;
- a launch-bound account of the fixed cost;
- "the law is not affine" and the inflated per-rider values that came with it --
  fitted across the knee, which **raises the apparent slope and depresses the apparent
  intercept together, from one cause**;
- a context-dependent per-rider cost. Retracted, **then the retraction was itself
  retracted**: the test equalised point counts across contexts and thereby admitted
  points from *above* the knee at the shorter ones -- fixing one confound by
  introducing another. Settled by removing the bytes, which removes the knee and makes
  the host component measurable at any context: +54% +- 25% over 32x of context, not
  resolvable, where the withdrawn law needed +1657%;
- the bidirectional link figure for a unidirectional rider stream;
- a batch-size increase, which remains **modelled, not measured**;
- and now, per §5.9, the four withdrawn floor components.

Four of these are the same error -- **fitting or generalising across a regime
boundary.** Chapter 3 should name the boundary once, define the exposure ratio that
locates it, and state the gate that keeps measurements on one side of it. One
paragraph, and it is a methodological contribution.

---

## Appendix -- Defects found while writing this, in priority order

**1. `07-related.tex:24` makes a provably false claim about an OSDI paper.** It says
of DirectKV "they pay no device memory and get no overlap". DirectKV's abstract claims
"warp-level pipelining to overlap KV fetching, computation, and write-back", and it
has a section measuring it. Our own `F-coherence.tex:19` states the correct version --
a window of one tile, with almost no arithmetic to hide behind. **Fix the body to the
appendix's wording.** The same sentence also carries the retracted "hide about 90% of
the transfer", which appears again in `C-related.tex:238`. This is the single most
damaging item in the paper.

**2. The SuperInfer claim is sourced to an abstract.** `refs.bib` says so explicitly
and tells us to read the body first; there is no local copy. The claim it supports is
the one separating us from the closest competitor. Read it or soften it.

**3. `07-related.tex` maps Pie onto the `k` axis incorrectly.** Pie relocates `m` of
`n` layers and swaps them continuously with a **two-buffer** device window, refining
its own expansion factor for "4 buffers in the entire effective memory space and 2
buffers on the physical GPU". So Pie is `f = m/L, q = one layer, window 2` -- the same
quantum as ours, smaller `f`, and bidirectional traffic because it is a swap design.
"Pie fixes `k = L-m`" conflates permanent residency with the staging window and will
be caught. The existing text is accurate as far as it goes; do not "improve" it onto a
`k` axis.

**4. `04-design.tex:143` mislabels a prefetch ablation as a `k>1` ablation.** §5.3.2.

**5. `paper/model/overhead_model.py:112` and `fusion_bound.py:37`** -- the 1.94
attribution. Needs a decision, not an edit; see §6.5.

**6. `08-discussion.tex:53`** -- "bounded below by 0.83 ms of unavoidable copies".
Not a bound; see §5.5.

**7. `paper/REVIEW_PROCESS_20261008.md`** -- never amended after the
retraction-of-the-retraction, so it reads as though the headline is withdrawn to a
range including no benefit. The settled position hours later is x1.2-1.4 for the two
weight-heavy configurations.

**8. `paper/tex/sections/E-implementation.tex`** -- still presents the launch-bound
account of the fixed cost as fact, contradicting `05-implementation.tex` and
`06-evaluation.tex`.

**9. `benchmarks/kv_pipelining_bench/ORCHESTRATION_TAX_PLAN.md`** -- header still
reads "Status: active" with the dead +62%/+105% claim; the nullification is 300 lines
down.

**10. `paper/tex/sections/09-overhead.tex`** -- correctly headed stale and excluded
from the build. Listed so it is not "fixed" by accident.
