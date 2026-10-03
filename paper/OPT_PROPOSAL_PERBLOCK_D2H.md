# Optimisation proposal: the per-block table rebuild and the blocking D2H

**Status: for review. No code written.**
Prepared 2026-10-03 against `overhead-characterisation` @ `908d2a773`.

These are the two costs the overhead campaign identified as the binding
constraints on the feature. Together they are **88% of `c` and 85% of `F`** at
the paper's target (Llama-70B fp8, ctx 8192, GH200/C2C). Both are host-side,
both are link-invariant, and both are localised to a single code site.

This document proposes what to change, argues the correctness, states what I
expect to gain and — for the second target — explains why a previous attempt at
the same thing was reverted and what has changed since. It ends with four
decisions I need from you.

---

## 0. Scope

**In scope.** Two sites in `vllm/v1/hybrid_rungd_pool.py`:

| | site | measured cost | share |
|---|---|---|---|
| **T1** | `build_step` block-table rebuild, `:525-531` | 0.00765 ms per rider **block** | **75.7% of `c`** at the target |
| **T2** | `write_token` pageable D2H, `:738` | 0.142 ms per **layer** | **85% of `F`** |

**Explicitly out of scope**, and I will not touch them:

* **Seeding / prefill handover** (`seed_layer`, `preload_layer`). By your own
  scoping rule the P node puts the data in DDR, so this is emulation cost, not
  the feature's. `preload_layer` already has a contiguous fast path.
* **`writeback`** (`:631`) — an off-by-default comparison mode, already async.
* The controller, the admission policy, the attention kernels.
* **The paper**: prose, `killer_v2.F_REF_MS`, the tables and the
  `check_tables.py` pins stay untouched, as agreed.

Note this is **feature optimisation**, which the characterisation campaign
deliberately excluded. It is new work, and the chapter will present these as
*proposed* with a projected effect, not as measured, unless you greenlight the
build and I then measure it.

---

## T1 — The per-block block-table rebuild

### The mechanism

```python
# hybrid_rungd_pool.py:525-531
_tbt = PT.now()
bt = torch.zeros(len(active_req_ids), max_blk, dtype=torch.int32)
for row, r in enumerate(active_req_ids):
    for j, hb in enumerate(self.req_blocks.get(r, [])):
        bt[row, j] = self._compact[hb]
PT.add("build_step_bt_python", _tbt)
```

One **Python-level `__setitem__` on a torch tensor per rider block, every
step.** A single-element tensor assignment carries the full indexing machinery —
measured at roughly 8 µs in the engine-free microbenchmark, against ~60 ns for
the equivalent list operation.

Blocks per rider are `S/bs`, so this is `r · S/bs` assignments per step: at
ctx 2048 / bs 16 with 8 riders, **1024 calls per step**. The coefficient
0.00765 ms/block was measured by varying block size at fixed bytes and rider
count — bs 16→64 is a 4× block-count change that leaves the byte volume
untouched, which is the only way to separate a per-block cost from a per-byte
one. At ctx 8192 it is **3.917 ms per rider**, and a faster interconnect does
nothing to it.

Three further loops in the same function are also `O(W)` in Python, where
`W = ` total active blocks:

* `host_set.update(...)` per request, then `sorted(host_set)` (`:495-497`)
* the `self._compact` dict comprehension (`:498`)
* the maximal-contiguous-runs loop (`:503-513`)

So the whole of `build_step` scales with `r · S/bs`.

### The change: make it O(r) by using the extent allocator

The allocator already guarantees what we need. With `blocks_per_req > 0` —
which is how the runner builds the pool — `ensure_req` hands each request
`base .. base+n-1`:

```python
# :215-228
# Contiguous extent per request: block ids are base..base+n-1, so a
# request's active blocks always form ONE ascending run.
base = self._req_extent[req_id] * self._bpr
while len(blks) < num_logical_blocks:
    blks.append(base + len(blks))
```

`_issue_runs` already relies on this. The block-table build does not. Given
per-request contiguity:

1. `_active_host` is the sorted concatenation of `r` contiguous ranges, so it is
   described by `r` `(start, length)` pairs — derivable in `O(r log r)` by
   sorting the extents, with no per-block iteration.
2. A request's compact ids are therefore **also contiguous**:
   `_compact[hb] = cbase + (hb - base)`. The whole row is one `arange`.
3. `bt[row, :n] = torch.arange(cbase, cbase + n)` — **one tensor op per
   request** instead of one per block.
4. `self._runs` *is* the extent list, already ascending — `O(r)`, replacing the
   scan.
5. `_compact` as a dict is then needed only by `compact_of()`; it is replaced by
   a per-request `(base, cbase)` pair and an `O(1)` arithmetic lookup.

**The `∝ S` scaling disappears** rather than shrinking by a constant. The
per-block term becomes `O(r)`, which is why this is the higher-value of the two
targets despite `F` being the more dramatic story.

**The fallback matters.** When `blocks_per_req == 0` the free-list path gives
scattered blocks and none of the above holds. I will keep a general path,
guarded by an explicit contiguity check rather than by assuming the mode, and
vectorise it anyway: build a padded nested list with comprehensions and call
`torch.tensor(...)` once. That is still `O(blocks)` but at list-operation cost,
~75× cheaper per item. The runner does not select this mode; correctness there
still has to hold.

### Correctness

`bt` must be **bitwise identical** to today's output. The ragged case is the one
to watch: requests of differing length pad to `max_blk`, and the padding must
stay zero-filled exactly where a request owns fewer blocks.

Gate: compute both paths and assert equality, exercised in the suite's existing
bitwise mechanism test `rungd_host_pool_test.py`, over (a) contiguous extents,
(b) a deliberately fragmented free-list allocation, (c) ragged cohort lengths,
(d) a single request, (e) the empty cohort.

### Expected effect

| | now | after | basis |
|---|---|---|---|
| `build_step_bt_python`, ctx 2048, r=8 | measurable via the existing Class A timer | → `O(r)` | direct |
| per-block term of `c`, ctx 8192 | 3.917 ms/rider | ~0.01 | removal of the S-dependence |
| **`c` at the target** | **5.173** | **≈1.27** | per-block was 75.7% |

`c` = 1.27 against a budget of `T0/B` = 3.37 crosses the necessary condition,
which as built it fails. On its own, this is the change that makes the feature
arguable at all.

**Verification is direct and cheap**: `build_step_bt_python` is a host-side
`perf_counter` bracket around a pure-CPU block. It is Class A and trustworthy —
unlike the phase timers that misled this campaign twice, which were bracketing
*blocking CUDA calls*. No CUDA involvement, no attribution hazard.

---

## T2 — The blocking pageable D2H

### The mechanism

```python
# hybrid_rungd_pool.py:737-741
if not self._skip_d2h:
    k_cpu, v_cpu = keys.to("cpu"), values.to("cpu")   # PAGEABLE => BLOCKING
    hk.view(...).index_copy_(0, host_pos, k_cpu)
    hv.view(...).index_copy_(0, host_pos, v_cpu)
```

A `.to("cpu")` onto a **pageable** destination is synchronous: it blocks the
calling thread and drains the stream. `write_token` runs once per layer, so this
is **64 host/device serialisation points per decode step**.

The bytes are one token's KV — a few kilobytes. At 71 µs per copy that is
**446× the time the transfer itself needs**, so this is a synchronisation cost,
not a bandwidth one, and no interconnect removes it. Because it is per-layer and
not per-rider, it lands in `F`, and it is **85% of `F`**.

### The honest uncertainty: is the prize 9.1 ms or 1.6 ms?

Two measurements disagree, and I would rather put this in front of you than
pick one.

* **`SKIP_D2H` differencing, M3, today's code:** `F` 10.76 → 1.66, so
  **9.10 ms**. But `SKIP_D2H` removes the copies *and* the CPU scatter, so it is
  an **upper bound** on any correct fix.
* **Commit `c609d93c7` made exactly this copy async and measured 1.6 ms**
  (fixed cost 24.78 → 23.14), then reverted it: *"correct, but 1.6 ms and it
  scales the wrong way."*

They are not in contradiction, because they measure different code:

1. The revert's own diagnosis is that **"a drain costs what is queued behind
   it"**, and `write_token`'s copies landed "where little is pending". Since
   then the async prefetch chain (`_launch_prefetch`, a side stream, per-layer
   events) puts the *next* layer's H2D in flight during the current layer. There
   is now substantially more queued behind the same drain, so the same sync
   costs more.
2. **The thing that killed the earlier attempt no longer exists.** The revert
   names its main cost driver as *"a per-row Python loop whose cost GROWS with
   rider count — microbenchmarked 1.48 → 3.28 ms from 1 to 3 riders"*, and
   concludes *"a win that shrinks where we need to scale is not worth the race
   surface."* `write_token` has since been vectorised to one `index_copy_` per
   target with no per-row loop (`:701-703`, `:727-731`). That objection was
   correct then and does not apply now.

### Phase 0 — price it before building anything

Rather than argue 1.6 against 9.1, measure it. A measurement-only arm,
`VLLM_KVPIPE_RUNGD_D2H_PROBE=1`, replaces the two `.to("cpu")` calls with
`non_blocking=True` copies into a pre-allocated pinned buffer and **skips the
scatter**. That isolates exactly *blocking → async* with launch count, byte
volume and every other variable held fixed:

```
real  −  probe      =  the synchronisation cost  =  the realisable prize
probe −  skip_d2h   =  the async-copy + scatter cost a correct fix still pays
```

Numerically invalid, since the mirror goes stale — so timing-only,
double-gated behind an `_UNSAFE` flag exactly like the layer-subset knob,
excluded from the suite, never cited for a quality claim. About 20 lines, one
sbatch slice, under an hour on the A100 already held.

> **If the prize comes in at ≤2 ms I recommend not doing T2 at all** and
> spending the effort on T1, which is 3.9 ms *per rider* at the target and
> carries no ordering risk. That is the main reason to run Phase 0 first: it can
> save the entire T2 effort, and it costs an hour.

### The change, if Phase 0 justifies it

1. **One pinned staging buffer** `[2, L, n, nkv, hd]`, allocated in `build_step`
   and reallocated only when the cohort **grows** (pinned allocation costs
   ~ms; cohort changes are rare in steady state, and `build_step` already knows
   `n`). At the measured point this is 2·64·8·8·128·2 B = **2 MB**.
2. **Per layer:** `stage[0, li, :n].copy_(keys, non_blocking=True)` — async into
   pinned memory, no host wait, no stream drain.
3. **One event** recorded after the last layer's copy.
4. **Defer the CPU scatter to the next step's `build_step`**, behind a single
   `event.synchronize()`. By then the copy is a full step old, so the wait is
   already satisfied and costs nothing.
5. **Unify the host mirrors** into one pinned allocation per K/V with per-layer
   views, so every existing `self.host_k[ln]` call site is untouched but the
   deferred scatter becomes **one** `index_copy_` across all `L` layers instead
   of `2L`. Destination index `li·(h_max·bs) + host_block·bs + off` is one
   vectorised `[L, n]` expression per step.

Net: sync points per step **64 → 1** (already satisfied); blocking copies
**2L → 0**; CPU scatter calls **2L → 2**.

### The ordering hazard — including one the earlier attempt did not cover

Between the async copy and the deferred scatter, the host mirror does **not**
yet hold the new token. Every reader of the mirror must be ordered against that.
Enumerated by grep over `host_k[`/`host_v[`, not from memory:

| line | site | kind | needs ordering |
|---|---|---|---|
| `:561` | `_issue_runs` | **device** read (H2D source) | **YES — this is the hot path and the correctness-critical one** |
| `:286` | `restore_layer` | CPU read | yes |
| `:424` | `stash_from_host` | CPU read | yes |
| `:483` | host-mirror property | CPU read | yes |
| `:263` | `seed_layer` | CPU write | yes — a seed must not race a pending token write |
| `:346` | `preload_layer` | CPU write | yes, same reason |
| `:631` | `writeback` | device write | off by default; flush for safety |
| `:711` | `write_token` | the writer itself | n/a |

**`:561` is the one that matters most, and the revert's three guards were all
CPU paths.** `_issue_runs` re-streams the mirror H2D at the *next* step; if the
previous step's token has not been scattered in, the stream carries stale KV and
the output is silently wrong. The ordering that makes this safe is that
**`build_step` runs once per step before any `_issue_runs` for that step**, so
doing the flush at the top of `build_step` covers it by construction. I checked
the one thing that could break that: `_launch_prefetch` is guarded by
`layer_idx + 1 < len(self.layers)` (`:682`), so nothing prefetches across a step
boundary.

Also worth recording: `write_token`'s *other* half — the rolling-slot write
`kslot.index_copy_(0, slot_pos, keys)` — is device-side and stays exactly as is.
In-step attention reads the slot, not the mirror, so **within-step correctness
is untouched by this change.** Only the history the next step re-streams is
deferred.

**One idempotent flush, not three call-site guards.** A `_pending_d2h` dirty
flag plus one `_flush_pending()` (sync the event, run the scatter, clear the
flag), called at the top of each site above. In decode steady state those extra
sites are never hit, so the cost is zero; when hit, it is one sync plus a
scatter that was going to happen anyway. One place to audit, rather than three
guards the next person must remember — which is how the earlier patch
accumulated its "race surface".

### Expected effect

Conditional on Phase 0, and I will not pretend to a point estimate:

| Phase 0 says the prize is | `F` at the target | break-even r (knee 3.8) |
|---|---|---|
| 9.1 ms (the `SKIP_D2H` bound) | 13.4 → ≈4.5 | pays from ≈1.0 |
| 1.6 ms (the 2026 measurement) | 13.4 → ≈11.9 | marginal |

---

## Risks

| risk | handling |
|---|---|
| T2's prize is 1.6 ms, not 9.1 | **Phase 0 prices it first**; abandon T2 if ≤2 ms |
| a reader sees a stale mirror | one idempotent dirty-flag flush; reader list derived by grep and **asserted in a test**, so a new reader added later fails loudly |
| `_issue_runs` races the deferred scatter | flush at the top of `build_step`, which provably precedes every `_issue_runs` of that step; prefetch cannot cross a step boundary (`:682`) |
| pinned staging thrash if the cohort oscillates | reallocate on **grow only**, cap at `max_num_seqs`, and expose a counter so thrash is visible rather than silent |
| pinned-memory pressure (mirrors are already pinned) | stage is 2 MB at the measured point; report it alongside the mirrors' footprint |
| the unified-mirror layout breaks a call site | per-layer **views** keep `host_k[ln]` byte-identical; the suite's bitwise roundtrip test already covers the mirrors |
| T1's extent-contiguity assumption is false | fast path behind an **explicit contiguity check**, general path retained and tested on a deliberately fragmented allocation |
| T1 changes `bt` numerically | both paths computed and asserted equal under a flag, five cases incl. ragged and empty |
| the optimisation contaminates the campaign's measurements | land on a **separate branch**; the campaign's archived JSONs and the report stay as the pre-change baseline (decision 4) |

---

## Verification plan

1. `run_suite.sh` → **9/9** before and after. Env trap, learned the hard way:
   `migration_correctness` and `burst_recovery` need
   `VLLM_ENABLE_V1_MULTIPROCESSING=0`, and `hybrid_rungd_real` **fails** if it
   leaks in — never `export` it around the suite.
2. **Bitwise**: `rungd_host_pool_test.py` extended with the `bt`-equality cases
   (T1) and mirror-equality after a deferred flush (T2).
3. **Token-identical end-to-end**: `hybrid_rungd_real.py`.
4. **Class A timer** `build_step_bt_python` before/after at ctx 2048 (T1).
5. **`SKIP_D2H` differencing re-run** after T2: the gap between `real` and
   `skip_d2h` should collapse to the residual, which is the direct check that
   the sync was actually removed and not merely moved.
6. **Refit the overhead law** on a post-change sweep: `c`'s per-block
   coefficient should go to ≈0 and `F` should drop. Both tables regenerate from
   `paper/model/c_scaling.py` and `paper/model/amortization.py`, so the
   before/after comparison is mechanical rather than hand-copied.
7. **Decode-purity gate** on every measured point, as throughout the campaign.

---

## Decisions I need from you

1. **Phase 0 first, or straight to implementation?** I recommend Phase 0 — one
   hour, and a result of ≤2 ms would save the whole T2 effort and its ordering
   risk.
2. **T1 unconditionally?** It is 75.7% of `c` at the target, removes an `∝ S`
   term outright, and has no ordering risk — only a bitwise-equality obligation.
   I would do it regardless of what Phase 0 says about T2.
3. **The unified-mirror layout (step 5)** is the only structural change in the
   proposal. Skipping it costs `2L` CPU `index_copy_` calls (≈0.4 ms) instead of
   2, for a materially smaller diff. Skip it?
4. **Where should this land?** This worktree's branch carries ~50 commits of
   measurement, and mixing an optimisation into it would make the campaign's
   baseline harder to cite. I suggest a fresh branch off `hybrid-kv-tiers`, with
   the campaign branch preserved as the pre-change baseline.

---

## What this does to the projection, if both land

At the target — Llama-70B fp8, ctx 8192, GH200 with the measured 419 GB/s, 7.2
residents from capacity, `T0/B` = 3.37 ms, knee 3.8:

| | `c` | `F` | gain at r = knee |
|---|---|---|---|
| as built | 5.17 | 13.4 | **0.65×** |
| T1 only | 1.27 | 13.4 | ≈0.95× |
| T1 + T2 (9.1 ms prize) | 1.27 | ≈4.5 | **≈1.21×** |
| T1 + T2, hiding as the 8B | 1.27 | ≈11.9 | break-even beyond the knee |

The last row is the standing caveat from the campaign and these changes do not
address it: the hiding fraction of the remaining async host work is unmeasurable
on an A100 and bands `F` over 18×. That is a separate experiment (one run on any
GPU faster than an A100), and it is **second** in priority behind the α sweep
now on `nova07`.

**The reason to do this work is not that it makes the feature win.** It is that
as the code stands, a C2C machine would be benchmarked against a ceiling of
0.71×, and the result would be read as a verdict on the feature when it is a
verdict on two host loops.
