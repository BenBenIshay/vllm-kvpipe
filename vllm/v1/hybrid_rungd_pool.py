# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-block-id KV pool for the hybrid CPU tier (rung-d serve-more).

The rung-c/write-redirect mechanisms proved the CPU cohort can be served from a
k_window-shared rolling region + host mirror. To actually FREE the cohort's
main-pool blocks (so more requests fit), the host mirror must not be indexed by
*resident* block id: after a free the resident block table is nulled and the id
is reused by another request, which both loses the read mapping and aliases
another request's host data.

This pool gives the CPU cohort its OWN host block-id space, decoupled from the
resident pool:
  * a host block id is assigned per (request, logical block) and freed only when
    the request finishes -> no cross-request aliasing;
  * the host mirror (one pinned CPU tensor per layer) is indexed by host block
    id; a small k_window rolling region (GPU) is streamed per layer;
  * growth is handled by assigning a fresh host block for each new logical block
    (its tokens arrive via write_token; no resident block is ever needed).

Per decode step: build_step() compacts the active cohort's host blocks into
[0:W) and returns the rolling block table; then for each layer stream_layer()
refreshes the rolling slot from the host mirror and write_token() folds in the
new token. Attention reads the rolling slot with the compacted block table.
"""
from __future__ import annotations

import torch

from vllm.v1.hybrid_phase_timers import PT


def _runs_from_sorted(srt: list[int]) -> list[tuple[int, int, int]]:
    """Maximal contiguous runs of a SORTED block list, as
    (host_start, compact_start, length). Correct for any allocation pattern;
    contiguity only makes the runs longer."""
    runs: list[tuple[int, int, int]] = []
    if not srt:
        return runs
    start = prev = srt[0]
    cstart = 0
    for i, hb in enumerate(srt[1:], 1):
        if hb != prev + 1:
            runs.append((start, cstart, prev - start + 1))
            start, cstart = hb, i
        prev = hb
    runs.append((start, cstart, prev - start + 1))
    return runs


class RungdHostPool:
    def __init__(
        self,
        layer_names: list[str],
        kv_shape: tuple,      # a resident layer tensor's shape [2, nblk, bs, nkv, hd]
        block_size: int,
        k_window: int,
        h_max_blocks: int,
        w_max_blocks: int,
        device: torch.device | str,
        dtype: torch.dtype,
        blocks_per_req: int = 0,
    ) -> None:
        import os as _os          # bound first: env is read throughout __init__
        self.layers = list(layer_names)
        self.layer_index = {ln: i for i, ln in enumerate(self.layers)}
        self.block_size = int(block_size)
        self.k = max(1, int(k_window))
        self.h_max = int(h_max_blocks)
        self.w_max = int(w_max_blocks)
        self.device = device
        # BLOCK-MAJOR, SPLIT K/V. The mirror used to be one [2, H, bs, nkv, hd]
        # tensor per layer, which forced a CPU staging copy on every layer: a
        # run of blocks is `narrow(1, ...)`, i.e. two chunks strided by a whole
        # K/V plane, and a strided H2D cannot be a plain pinned DMA. Measured on
        # A100 at Qwen-32B/ctx-2048 that gather was 22 ms per layer -- 98% of
        # `stream_layer` and 72% of the engine's wall -- while the H2D itself
        # ran at full link speed.
        #
        # With the block dimension first, and K and V in separate mirrors, a run
        # is ONE contiguous byte range on both sides. `stream_layer` becomes a
        # handful of `cudaMemcpyAsync`s straight from pinned host memory into
        # the right offsets of the device slot: the DMA engine performs the
        # compaction, and no CPU staging buffer exists at all. Microbenchmarked
        # (cold, 32 mirrors): 6.89 ms -> 1.38 ms per layer, 4.87 -> 24.38 GB/s.
        #
        # K and V must be SEPARATE tensors, not one [W, 2, ...]: attention needs
        # key_cache/value_cache as contiguous [blocks, bs, nkv, hd], and dim-0
        # indexing a block-major tensor yields a strided view flash cannot take.
        _, _, bs, nkv, hd = kv_shape
        hshape = (self.h_max, bs, nkv, hd)
        # T7a. ONE pinned allocation per tensor, with per-layer VIEWS, rather
        # than L separate allocations.
        #
        # Nothing about the per-layer interface changes: `_host_k_all[i]` is a
        # contiguous [h_max, bs, nkv, hd] tensor, pinned (a view of pinned
        # memory is pinned), so every existing path -- _issue_runs, seed_layer,
        # restore_layer, preload_layer, writeback_layer, stash_from_host -- sees
        # exactly what it saw before. Checked that none of them depends on the
        # mirrors being distinct allocations: they all go through `host_k[ln]`
        # indexing, and the one place that combines them (`host`) stacks views
        # just as happily.
        #
        # WHY IT IS WORTH DOING. The deferred scatter of T2 is 2*L
        # `index_copy_` calls per step -- one per layer per tensor -- measured
        # at 0.018 ms/layer, 29% of the mirror write's 0.061. With the layers in
        # ONE tensor the whole scatter becomes 2 calls with a vectorised index,
        # because a rider's row within its extent does not depend on the layer:
        # the global row is `host_pos + layer * h_max * bs`. See
        # flush_token_writes.
        self._unified_mirror = (
            _os.environ.get("VLLM_KVPIPE_RUNGD_UNIFIED_MIRROR", "1") == "1"
        )
        if self._unified_mirror:
            self._host_k_all = torch.empty(
                (len(self.layers),) + hshape, dtype=dtype, device="cpu",
                pin_memory=True)
            self._host_v_all = torch.empty(
                (len(self.layers),) + hshape, dtype=dtype, device="cpu",
                pin_memory=True)
            self.host_k: dict[str, torch.Tensor] = {
                ln: self._host_k_all[i] for i, ln in enumerate(self.layers)
            }
            self.host_v: dict[str, torch.Tensor] = {
                ln: self._host_v_all[i] for i, ln in enumerate(self.layers)
            }
        else:
            self._host_k_all = None
            self._host_v_all = None
            self.host_k = {
                ln: torch.empty(hshape, dtype=dtype, device="cpu",
                                pin_memory=True)
                for ln in self.layers
            }
            self.host_v = {
                ln: torch.empty(hshape, dtype=dtype, device="cpu",
                                pin_memory=True)
                for ln in self.layers
            }
        # Row offset of each layer inside the unified mirror, precomputed so the
        # fast scatter allocates nothing on the hot path.
        self._wt_layer_off = (
            torch.arange(len(self.layers), dtype=torch.int64) * (self.h_max * bs)
            if self._unified_mirror else None
        )
        # Do all pending writes in a step share one host_pos and n? They do in
        # steady state -- a rider's row within its extent is the same for every
        # layer -- and that is what lets the scatter be one vectorised call.
        # Tracked rather than assumed, with the per-layer loop as the fallback.
        self._wt_uniform = True
        # k_window GPU rolling slots, sized to the compact region.
        rshape = (self.w_max, bs, nkv, hd)
        self.rolling_k: list[torch.Tensor] = [
            torch.empty(rshape, dtype=dtype, device=device) for _ in range(self.k)
        ]
        self.rolling_v: list[torch.Tensor] = [
            torch.empty(rshape, dtype=dtype, device=device) for _ in range(self.k)
        ]
        # Async double-buffering (opt #2): a side copy stream so layer L+1's
        # host->rolling H2D overlaps layer L's attention. Requires k_window >= 2
        # so consecutive layers use distinct slots. It was previously OPT-IN and
        # measured worthless -- correctly, but for a reason that has now been
        # removed: it hid the 1.3 ms H2D while the 22 ms CPU gather was the
        # cost. With the gather gone there is nothing left but the transfer, so
        # this is the regime it was written for. Still opt-in
        # (VLLM_KVPIPE_RUNGD_ASYNC=1) until re-measured on the new layout.
        # No staging buffers: the copies now come straight from the pinned
        # mirrors, so there is nothing to double-buffer but the stream itself.
        # Swap-semantics emulation (see writeback_layer). Off by default: this
        # is measurement apparatus, not part of the design.
        self._writeback = (
            _os.environ.get("VLLM_KVPIPE_RUNGD_WRITEBACK", "0") == "1"
        )
        self.writeback_bytes = 0
        # Bytes pulled host->device by the per-layer refresh. Counted so a
        # harness can compare arms on TRAFFIC rather than on request counts.
        self.stream_bytes = 0
        # What `real` WOULD have moved, so an arm that moves less can still be
        # charged the link time the real path would have paid.
        self.stream_bytes_nominal = 0
        # H2D memcpy launches issued (2 per run per layer). The launch COUNT is
        # what a faster link does not change, so it has to be counted, not
        # inferred from the bytes.
        self.stream_runs = 0
        # MEASUREMENT APPARATUS. Separates the two costs the whole C2C
        # projection turns on, by changing only what crosses the bus:
        #
        #   real      the design.
        #   minbytes  same run structure, same 2*L*R launches, same views,
        #             same stream order -- but each copy moves ONE block row
        #             instead of n. This is an infinitely fast link: the
        #             per-byte term goes to ~0 while every launch is still
        #             issued. It is therefore the CEILING for any interconnect,
        #             C2C included, since no link removes a launch.
        #   nobus     the loop and the views run, no bus copy is issued at all.
        #             The floor -- unreachable by any link, and the difference
        #             from `minbytes` is precisely the cost of the launches.
        #
        # So: real - minbytes = the byte cost; minbytes - nobus = the launch
        # cost; nobus - baseline = everything else the host path costs.
        #
        # Numerically INVALID: the rolling slot does not receive the right KV,
        # so generated tokens are meaningless. Timing only, never a correctness
        # or quality claim.
        self._xfer_mode = _os.environ.get(
            "VLLM_KVPIPE_RUNGD_XFER_MODE", "real").strip().lower()
        if self._xfer_mode not in ("real", "minbytes", "nobus"):
            raise ValueError(
                f"VLLM_KVPIPE_RUNGD_XFER_MODE={self._xfer_mode!r}: expected "
                "one of real, minbytes, nobus")
        if self._xfer_mode != "real":
            print(f"*** RungdHostPool: XFER_MODE={self._xfer_mode} -- bus "
                  f"copies altered, OUTPUT IS NUMERICALLY INVALID. "
                  f"Timing measurement only.", flush=True)
        # PRICE THE BLOCKING D2H BY DIFFERENCING, not by a stopwatch.
        #
        # write_token does two PAGEABLE (hence blocking) `.to("cpu")` copies per
        # layer -- 64 host/device serialisation points per step. Timing the host
        # around them does NOT price them: measured 21.4 ms per step, 0.335 ms
        # per layer, for two copies of one token's KV. At ~4 KB that would be
        # 12 MB/s, which is not a bandwidth -- it is the host blocked on the
        # GPU. The repo had already documented that exact trap ("charged
        # write_token the model's own compute, producing a 21 ms blocking D2H
        # worth ~4 ms at a realistic queue depth") and the phase timers
        # reproduced it anyway, to the same 21 ms.
        #
        # So this skips the host-mirror write entirely and lets the SAME
        # differencing that priced the bus copies price it too. The mirror then
        # goes stale, so subsequent refreshes read wrong KV: numerically invalid
        # by construction, timing only.
        self._skip_d2h = (
            _os.environ.get("VLLM_KVPIPE_RUNGD_SKIP_D2H", "0") == "1"
        )
        if self._skip_d2h:
            print("*** RungdHostPool: SKIP_D2H=1 -- host mirror not updated, "
                  "OUTPUT IS NUMERICALLY INVALID. Timing measurement only.",
                  flush=True)
        # PHASE 0 PROBE, timing only. SKIP_D2H above prices the mirror write
        # by removing it, which bounds the prize from ABOVE because it also
        # removes the CPU scatter that any correct fix must still do. This arm
        # instead keeps the copy and only makes it NON-BLOCKING -- same launch
        # count, same bytes, pinned destination -- and skips the scatter. So
        #
        #   real  - probe  = the SYNCHRONISATION cost, i.e. the realisable prize
        #   probe - skip   = the async copy plus scatter a fix still pays
        #
        # which is the number that decides whether the real fix is worth its
        # ordering risk. Two measurements disagree about it: SKIP_D2H
        # differencing says 9.10 ms on this code, while commit c609d93c7 made
        # this copy async, measured 1.6 ms and reverted on cost. They measure
        # different code -- the async prefetch chain now puts more behind the
        # same drain, and the per-row Python loop that commit named as its cost
        # driver has since been vectorised away.
        #
        # The mirror goes stale, so this is numerically invalid by construction
        # and double-gated like the layer-subset knob.
        self._d2h_probe = (
            _os.environ.get("VLLM_KVPIPE_RUNGD_D2H_PROBE", "0") == "1"
            and _os.environ.get("VLLM_KVPIPE_RUNGD_D2H_PROBE_UNSAFE", "0") == "1"
        )
        if _os.environ.get("VLLM_KVPIPE_RUNGD_D2H_PROBE", "0") == "1" \
                and not self._d2h_probe:
            raise RuntimeError(
                "VLLM_KVPIPE_RUNGD_D2H_PROBE=1 also needs "
                "VLLM_KVPIPE_RUNGD_D2H_PROBE_UNSAFE=1: it leaves the host "
                "mirror stale and the output numerically invalid."
            )
        if self._d2h_probe:
            print("*** RungdHostPool: D2H_PROBE=1 -- mirror write is async "
                  "into pinned staging and NOT scattered. OUTPUT IS "
                  "NUMERICALLY INVALID. Timing measurement only.", flush=True)
        self._probe_k: torch.Tensor | None = None
        self._probe_v: torch.Tensor | None = None
        # T2. Phase 0 priced the blocking mirror write's SYNCHRONISATION at
        # 8.47 ms of F (real 11.31 -> probe 2.84), against 0.25 ms of drift
        # between the bracketing real arms -- a 33.5x ratio, so it is the
        # effect and not drift. It sits at 90% of the 9.45 ms SKIP_D2H upper
        # bound, so a correct fix pays only ~1 ms back.
        #
        # That settles a disagreement: commit c609d93c7 made this copy async,
        # measured 1.6 ms and reverted on cost. On today's code it is 8.47.
        # Two things changed -- the async prefetch chain now puts far more
        # behind the same drain, and the per-row Python loop that commit named
        # as its own cost driver has since been vectorised away.
        #
        # So: copy asynchronously into pinned staging, and DEFER the host-side
        # scatter to the start of the next step, behind ONE event wait. 2L
        # blocking copies per step become 2L async ones plus a single wait that
        # is already satisfied by the time it is reached.
        #
        # Set VLLM_KVPIPE_RUNGD_D2H_ASYNC=0 to get the old blocking path back,
        # which is what the before/after measurement needs.
        # T3, DISABLED BY DEFAULT: IT IS A LARGE REGRESSION AS IMPLEMENTED.
        #
        # The idea was sound -- the per-layer gather issues 2*L*r contiguous
        # copies and the sources are evenly spaced, so one strided copy per
        # tensor would cut launches to 2*L. It is also CORRECT: the suite is
        # 9/9 with it on and the launch count does drop to 128 from 1024 at
        # eight riders, 861 of 861 steps taking the batched path.
        #
        # But it is 6x SLOWER, measured:
        #
        #   ctx 512    c 0.845 -> 35.771 ms/rider   (25.1 GB/s -> 4.3)
        #   ctx 2048   c 3.081 -> 231.700           (25.1 GB/s -> 2.3)
        #
        # torch's copy_ from a STRIDED pinned host view into a contiguous device
        # tensor does not become a 2D DMA; it materialises the source on the
        # host first. Those 2.3-4.3 GB/s rates are the band this repo already
        # documents for host-side gathers -- the old scattered path at 6.4 GB/s,
        # "the binding constraint on the whole path", and preload_layer's
        # block-by-block loop at "~2.9 GB/s flat across contexts, which is
        # Python overhead, not memory bandwidth". So T3 reintroduces exactly the
        # CPU staging copy the block-major layout was built to remove, and the
        # per-run loop is fast BECAUSE each of its 2*L*r copies is a contiguous
        # DMA at full link rate.
        #
        # The launch count was never the binding cost; contiguity is. Cutting
        # launches is only worth doing if every copy stays contiguous, which
        # means making the extents ADJACENT so the runs merge naturally rather
        # than striding across the gaps between them. That is an allocator
        # change and is not attempted here.
        #
        # Kept behind a flag rather than deleted because the correctness proof
        # and the equivalence test are valid, and a real cudaMemcpy2D binding
        # would make the idea work.
        self._batch_gather = (
            _os.environ.get("VLLM_KVPIPE_RUNGD_BATCH_GATHER", "0") == "1"
        )
        # T5. Build the per-step runs from whole EXTENTS rather than from the
        # blocks a request has actually used. See `_plan_extents` for why that
        # is what makes the runs merge, and what it costs.
        #
        # This is the change the T3 comment above points at ("making the extents
        # ADJACENT so the runs merge naturally"), arrived at the other way
        # round: the extents are already adjacent -- consecutive slots are
        # `base, base + bpr, base + 2*bpr` -- and what breaks the merge is the
        # GAP between a request's used blocks and the next extent's base. A
        # decode request grows one block per `block_size` tokens, so an extent
        # sized for its final length always has unused slack at its tail. Taking
        # the whole extent closes the gap by definition, and every copy stays
        # CONTIGUOUS, which is the one property T3 destroyed.
        #
        # Priced off-engine first, on the real shapes
        # (benchmarks/kv_pipelining_bench/launch_pattern_bench.py, A100):
        #
        #   per-run loop, 2r launches   0.149 ms host   0.776 ms dev   22.3 GB/s
        #   whole extent, 2 launches    0.016 ms host   0.706 ms dev   24.5 GB/s
        #
        # -89.5% host AND -9% device, because one big DMA beats r small ones.
        # The cost is the slack it moves, which is gen/S of the bytes -- 3% at
        # ctx 16384, 25% at ctx 2048 -- and the device time rises with it
        # (+1.7% at 4 slack blocks, +12.5% at 8, +34% at 16). So this is a win
        # while the step is HOST-bound and a loss once it is device-bound, which
        # is why it is a flag and why M11 measures which regime the step is in.
        #
        # REQUIRES A TIGHT blocks_per_req. The runner sizes extents for
        # max_model_len, so an extent is ~4x the blocks a request uses and whole
        # -extent copying would move 4x the bytes. VLLM_KVPIPE_RUNGD_BPR sets a
        # tighter value; with a loose one this flag makes things worse, so it
        # refuses to turn on unless the slack is bounded (see _plan_extents).
        self._whole_extent = (
            _os.environ.get("VLLM_KVPIPE_RUNGD_WHOLE_EXTENT", "0") == "1"
            and int(blocks_per_req) > 0
        )
        # Largest slack fraction worth paying. 0.5 is past the measured
        # break-even (host+dev turns positive near 10-12 slack blocks on 33
        # used, i.e. ~0.35) and is a backstop against a loose bpr, not a tuned
        # value: the flag is meant to be used with an extent sized for the
        # request, and this stops it silently costing 4x the bytes if it is not.
        self._whole_extent_max_slack = float(
            _os.environ.get("VLLM_KVPIPE_RUNGD_WHOLE_EXTENT_MAX_SLACK", "0.5")
        )
        self.whole_extent_steps = 0
        self.whole_extent_refused = 0
        self._d2h_async = (
            _os.environ.get("VLLM_KVPIPE_RUNGD_D2H_ASYNC", "1") == "1"
            and torch.cuda.is_available()
            and not self._d2h_probe
            and not self._skip_d2h
        )
        self._wt_k: torch.Tensor | None = None
        self._wt_v: torch.Tensor | None = None
        self._wt_pending: list[tuple[int, torch.Tensor, int]] = []
        self._wt_event = torch.cuda.Event() if self._d2h_async else None
        self._wt_dirty = False
        self.wt_flushes = 0
        self.wt_deferred_rows = 0
        # Negative control. Suppresses the deferred flush so the mirror goes
        # stale, which is exactly the T2 bug that would be silent in
        # production. Double-gated, and it announces itself.
        self._wt_noflush = (
            _os.environ.get("VLLM_KVPIPE_RUNGD_NOFLUSH", "0") == "1"
            and _os.environ.get("VLLM_KVPIPE_RUNGD_NOFLUSH_UNSAFE", "0") == "1"
        )
        if self._wt_noflush:
            print("*** RungdHostPool: NOFLUSH=1 -- deferred mirror writes are "
                  "SUPPRESSED. This is a deliberate negative control and the "
                  "output WILL be wrong. If a correctness test passes with "
                  "this set, that test cannot see a missed flush.", flush=True)
        self._async = (
            _os.environ.get("VLLM_KVPIPE_RUNGD_ASYNC", "0") == "1"
            and self.k >= 2 and torch.cuda.is_available()
        )
        self._copy_stream = torch.cuda.Stream() if self._async else None
        self._events: dict[int, torch.cuda.Event] = {}
        # T6: records where the mirror was last READ, so the deferred mirror
        # WRITE can be ordered after it. One event, re-recorded per layer --
        # recording overwrites, and since every copy here goes on the MAIN
        # stream in issue order, the newest event completing implies all the
        # older ones did. Prefetches go on `_copy_stream` instead and are
        # covered by synchronising that stream, because one event cannot stand
        # for two streams: a later copy-stream record would overwrite the
        # main-stream one and the main-stream DMA would go unwaited.
        #
        # DEFAULT OFF, and that is a deliberate and uncomfortable choice.
        #
        # Measured: with the ordering on, c = 7.024-7.037 ms per rider against
        # 0.883-1.004 without it -- it costs 6.1 ms per rider, which is the
        # per-rider transfer (6.17 ms at ctx 512) to within 1%. The reason is
        # `_launch_prefetch`'s opening `copy_stream.wait_stream(current_stream)`:
        # every queued copy is gated behind main-stream progress, so the copy
        # stream always has about a step's transfer outstanding and
        # synchronising it makes the host wait for all of it. The hazard is
        # intrinsic to the host running a step ahead of the device, so ANY
        # host-side ordering pays the whole gap.
        #
        # Turning it on therefore removes the feature's entire reason to exist:
        # the overlap is what makes c small enough to pass the gate at all.
        #
        # And it is NOT established that it fixes anything. The row the scatter
        # overwrites holds the CURRENT token, and at step N-1 the streamed
        # window only needs tokens up to t-1 -- token t's KV comes from the
        # resident GPU cache, written by the attention kernel -- so what the
        # in-flight prefetch reads there is masked out by seq_len and never
        # consumed. On that reading the race is benign and this ordering buys
        # nothing at full price.
        #
        # So: off by default, available with VLLM_KVPIPE_RUNGD_MIRROR_SYNC=1,
        # and the hazard counter below keeps measuring the window either way.
        # The intermittent token divergence on the async path remains OPEN and
        # is not explained by this.
        self._mirror_sync = (
            _os.environ.get("VLLM_KVPIPE_RUNGD_MIRROR_SYNC", "0") == "1"
            # kept as the old name so existing scripts keep working; it meant
            # the opposite, hence the inversion.
            and _os.environ.get(
                "VLLM_KVPIPE_RUNGD_NO_MIRROR_SYNC", "0") != "1"
        )
        self._no_mirror_sync = not self._mirror_sync
        if self._mirror_sync:
            print("*** RungdHostPool: MIRROR_SYNC=1 -- the deferred mirror "
                  "scatter is ordered against in-flight H2D reads. This is "
                  "CORRECT but costs ~6 ms per rider (the whole overlap); see "
                  "the comment in __init__.", flush=True)
        self._mirror_ev = (torch.cuda.Event()
                           if torch.cuda.is_available() else None)
        # How often the mirror's readers were STILL IN FLIGHT when the deferred
        # scatter was about to overwrite their rows. This is the race window
        # measured directly rather than inferred from output divergence, which
        # is intermittent and so cannot prove the window either way.
        self.mirror_hazard_steps = 0
        self.mirror_flush_steps = 0
        self.free_list: list[int] = list(range(self.h_max))
        # Item 1b: per-request CONTIGUOUS host extents (DirectKV's pool is
        # "contiguous per-request, no scatter/gather" for exactly this reason).
        # With scattered block ids the per-step refresh needs an index_select,
        # measured at 6.4 GB/s on A100 -- 3.9x the cost of the pinned H2D it
        # feeds, and the binding constraint on the whole path. Allocating each
        # request a contiguous extent makes its blocks one ascending run, so
        # the refresh becomes a handful of contiguous slice copies.
        self._bpr = int(blocks_per_req)
        self._extent_free: list[int] = (
            list(range(self.h_max // self._bpr)) if self._bpr > 0 else []
        )
        self._req_extent: dict[str, int] = {}
        # Maximal contiguous runs of the active set: (host_start, compact_start,
        # length). Correct for ANY allocation pattern -- contiguity only makes
        # the runs longer.
        self._runs: list[tuple[int, int, int]] = []
        self.req_blocks: dict[str, list[int]] = {}   # req_id -> [host_block_id]
        self.req_seeded: set[str] = set()            # prefill seeded
        # Prefill stash: KV that OUTLIVES the request that produced it, keyed by
        # prompt rather than request id. This is the P/D handover made local --
        # a prefill node's output, held until a decode-side request claims it.
        # Plain pageable CPU tensors: the stash is written and read outside the
        # measured decode window, so pinning it would waste pinned memory that
        # the mirrors need.
        self.prefill_stash: dict[str, dict[str, tuple[torch.Tensor,
                                                      torch.Tensor]]] = {}
        self.stash_tokens: dict[str, int] = {}
        # Resident tensors (layer_name -> [2, nblk, bs, nkv, hd]) for seeding.
        self.resident: dict[str, torch.Tensor] = {}
        # Per-step state (set by build_step).
        self._compact: dict[int, int] = {}           # host block id -> [0:W)
        # req_id -> (host_base, compact_base), the extent fast path's whole
        # per-request mapping. Empty when the general path ran.
        self._req_cbase: dict[str, tuple[int, int]] = {}
        # T3. (host_start, stride, length, count) when this step's runs are
        # evenly spaced and equal length, which makes the whole per-layer
        # gather ONE strided copy per tensor instead of one per run. None when
        # the pattern is irregular and the per-run loop has to be used.
        #
        # Why the pattern is usually regular: ensure_req pops consecutive
        # extent slots, so the bases are evenly spaced by blocks_per_req, and a
        # homogeneous cohort uses the same number of blocks in each. Why the
        # runs do not simply MERGE instead: an extent is sized for
        # max_model_len but a request only uses ceil(S/bs) of it, so there is a
        # gap between one request's used blocks and the next extent's base.
        # Measured: n_runs == rider count exactly, and 2*L*r launches per step.
        self._batch_plan: tuple[int, int, int, int] | None = None
        self.batched_steps = 0
        self.unbatched_steps = 0
        self._w_cur = 0
        # Preallocated so a block-table row is a VIEW, not an allocation. It
        # only ever needs to index the rolling region, which is w_max blocks.
        self._ar = torch.arange(self.w_max, dtype=torch.int32)

    # -- allocation ---------------------------------------------------------
    def ensure_req(self, req_id: str, num_logical_blocks: int) -> list[int]:
        blks = self.req_blocks.setdefault(req_id, [])
        if self._bpr > 0:
            # Contiguous extent per request: block ids are base..base+n-1, so a
            # request's active blocks always form ONE ascending run.
            if req_id not in self._req_extent:
                if not self._extent_free:
                    raise RuntimeError("RungdHostPool exhausted host extents")
                self._req_extent[req_id] = self._extent_free.pop()
            if num_logical_blocks > self._bpr:
                raise RuntimeError(
                    f"request needs {num_logical_blocks} blocks > extent "
                    f"{self._bpr}"
                )
            base = self._req_extent[req_id] * self._bpr
            while len(blks) < num_logical_blocks:
                blks.append(base + len(blks))
            return blks
        while len(blks) < num_logical_blocks:
            if not self.free_list:
                raise RuntimeError("RungdHostPool exhausted host blocks")
            blks.append(self.free_list.pop())
        return blks

    def free_req(self, req_id: str) -> None:
        # the extent is about to be recycled; pending rows must land first
        self.flush_token_writes()
        if self._bpr > 0:
            ext = self._req_extent.pop(req_id, None)
            if ext is not None:
                self._extent_free.append(ext)
            self.req_blocks.pop(req_id, None)
        else:
            for b in self.req_blocks.pop(req_id, []):
                self.free_list.append(b)
        self.req_seeded.discard(req_id)

    # -- seeding (resident prefill -> host, once per request) ---------------
    def seed_layer(
        self,
        req_id: str,
        layer_name: str,
        resident_kv: torch.Tensor,        # [2, nblk, bs, nkv, hd] resident tensor
        resident_block_ids: list[int],    # logical i -> resident block id
    ) -> None:
        # the extent is about to be recycled; pending rows must be applied first
        self.flush_token_writes()
        hblks = self.req_blocks[req_id]
        n = min(len(hblks), len(resident_block_ids))
        if n == 0:
            return
        res_idx = torch.tensor(resident_block_ids[:n], dtype=torch.long,
                               device=resident_kv.device)
        gathered = resident_kv.index_select(1, res_idx).to("cpu")  # [2, n, ...]
        dst = torch.tensor(hblks[:n], dtype=torch.long)
        self.host_k[layer_name].index_copy_(0, dst, gathered[0])
        self.host_v[layer_name].index_copy_(0, dst, gathered[1])

    # -- restore (host -> resident, for PROMOTION host-tier -> HBM-tier) -----
    def restore_layer(
        self,
        req_id: str,
        layer_name: str,
        resident_kv: torch.Tensor,        # [2, nblk, bs, nkv, hd] resident tensor
        resident_block_ids: list[int],    # logical i -> resident block id
    ) -> None:
        """Inverse of `seed_layer`: copy this request's host history back into
        resident blocks so it can be served from HBM again.

        Promotion is only correct if the resident blocks hold the FULL history
        afterwards, so the caller must have allocated `resident_block_ids` for
        the request's whole sequence before calling this.
        """
        # reads the mirror on the CPU, so pending token writes must have landed
        self.flush_token_writes()
        hblks = self.req_blocks.get(req_id, [])
        n = min(len(hblks), len(resident_block_ids))
        if n == 0:
            return
        src = torch.tensor(hblks[:n], dtype=torch.long)
        gk = self.host_k[layer_name].index_select(0, src)        # CPU [n, ...]
        gv = self.host_v[layer_name].index_select(0, src)
        res_idx = torch.tensor(resident_block_ids[:n], dtype=torch.long,
                               device=resident_kv.device)
        dev = resident_kv.device
        resident_kv[0].index_copy_(0, res_idx, gk.to(dev))
        resident_kv[1].index_copy_(0, res_idx, gv.to(dev))

    def restore_req(self, req_id: str, resident_block_ids: list[int]) -> None:
        """Restore all layers' KV from host into resident blocks."""
        if not self.resident:
            raise RuntimeError("pool has no resident tensors registered")
        for ln in self.layers:
            self.restore_layer(req_id, ln, self.resident[ln], resident_block_ids)
        # It is no longer host-authoritative; a later demotion must re-seed.
        self.req_seeded.discard(req_id)

    def mark_seeded(self, req_id: str) -> None:
        self.req_seeded.add(req_id)

    # -- external load (KV arrives from OUTSIDE, never touching HBM) ---------
    def preload_layer(
        self,
        req_id: str,
        layer_name: str,
        keys: torch.Tensor,      # [num_tokens, nkv, hd], token-major
        values: torch.Tensor,
        start_pos: int = 0,
    ) -> None:
        """Write one layer of a request's KV straight into the host mirrors.

        This is the door into the host tier that does NOT go through HBM.

        `seed_layer` reads a request's KV out of its *resident* blocks, which
        means a request can only become a rider by first being an ordinary
        fully-resident request: it must be allocated HBM, prefill into it, and
        only then be copied out and reclaimed. That transient residency is what
        forces `b_hbm_max` below capacity -- riders end up purchased with
        resident slots rather than added on top of them.

        Here the KV comes from the caller (a prefill node, a connector that
        staged it during prefill, or a test fixture), so a request can be live
        on the host tier having never owned a main-pool block.

        Token-major in, block-major out: tokens [start_pos, start_pos+n) land in
        the host blocks this request already owns via `ensure_req`.
        """
        # writes the mirror, so pending token writes must land first or be lost
        self.flush_token_writes()
        n = keys.shape[0]
        if n == 0:
            return
        hblks = self.req_blocks.get(req_id)
        if not hblks:
            raise RuntimeError(f"preload_layer: {req_id} has no host blocks; "
                               "call ensure_req first")
        bs = self.block_size
        need = (start_pos + n + bs - 1) // bs
        if need > len(hblks):
            raise RuntimeError(
                f"preload_layer: {req_id} needs {need} host blocks for "
                f"positions [{start_pos}, {start_pos + n}) but owns {len(hblks)}")
        hk, hv = self.host_k[layer_name], self.host_v[layer_name]
        k_cpu = keys.to("cpu", non_blocking=False)
        v_cpu = values.to("cpu", non_blocking=False)
        # Flatten (block, offset) so a contiguous token range is a contiguous
        # destination range whenever it sits inside one block.
        hk_flat = hk.view(hk.shape[0] * bs, *hk.shape[2:])
        hv_flat = hv.view(hv.shape[0] * bs, *hv.shape[2:])
        # FAST PATH: a request allocated a contiguous extent (`blocks_per_req`
        # > 0, which is how the runner builds the pool) has host blocks
        # base..base+k-1, so token p lands at base*bs + p and the WHOLE range
        # is one contiguous destination. The block-by-block loop below is then
        # 128 copies per layer at ctx 2048 -- 8192 for the request -- and
        # measured at ~2.9 GB/s flat across contexts, which is Python overhead,
        # not memory bandwidth.
        need = (start_pos + n + bs - 1) // bs
        if need > 1 and hblks[need - 1] - hblks[0] == need - 1:
            base = hblks[0] * bs + start_pos
            hk_flat[base:base + n].copy_(k_cpu)
            hv_flat[base:base + n].copy_(v_cpu)
            return
        written = 0
        while written < n:
            pos = start_pos + written
            blk, off = divmod(pos, bs)
            take = min(bs - off, n - written)
            dst = hblks[blk] * bs + off
            hk_flat[dst:dst + take].copy_(k_cpu[written:written + take])
            hv_flat[dst:dst + take].copy_(v_cpu[written:written + take])
            written += take

    def preload_req(
        self,
        req_id: str,
        num_tokens: int,
        kv_by_layer: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Admit a request to the host tier from external KV, with no HBM.

        `kv_by_layer` maps layer name -> (keys, values), each
        [num_tokens, nkv, hd]. Afterwards the request is marked seeded, so the
        runner will not try to seed it from resident blocks that it never had.
        """
        nb = max(1, (num_tokens + self.block_size - 1) // self.block_size)
        self.ensure_req(req_id, nb)
        for ln in self.layers:
            kv = kv_by_layer.get(ln)
            if kv is None:
                raise RuntimeError(f"preload_req: no KV supplied for layer {ln}")
            self.preload_layer(req_id, ln, kv[0], kv[1])
        self.req_seeded.add(req_id)

    # -- prefill stash (the P/D handover) -----------------------------------
    def stash_from_host(self, key: str, req_id: str, num_tokens: int) -> None:
        """Copy a seeded request's REAL KV out of the host mirrors, under `key`.

        This is what makes a rider's KV genuine instead of synthetic. The
        request that produced it prefilled normally, into HBM, and was seeded
        and reclaimed by the ordinary path -- so this is model output, not a
        fill value. Keeping it keyed by prompt lets it survive that request's
        completion and be claimed later by a decode-side request, which is
        exactly the handover a prefill node performs.

        Token-major out (the inverse of `preload_layer`), so the result can be
        fed straight back through `preload_req`.
        """
        # reads the mirror on the CPU, so pending token writes must have landed
        self.flush_token_writes()
        if key in self.prefill_stash:
            return
        hblks = self.req_blocks.get(req_id)
        if not hblks:
            raise RuntimeError(f"stash_from_host: {req_id} owns no host blocks")
        n = int(num_tokens)
        bs = self.block_size
        if (n + bs - 1) // bs > len(hblks):
            raise RuntimeError(
                f"stash_from_host: {req_id} claims {n} tokens but owns "
                f"{len(hblks)} host blocks")
        out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for ln in self.layers:
            hk, hv = self.host_k[ln], self.host_v[ln]
            hk_flat = hk.view(hk.shape[0] * bs, *hk.shape[2:])
            hv_flat = hv.view(hv.shape[0] * bs, *hv.shape[2:])
            k = torch.empty((n, *hk.shape[2:]), dtype=hk.dtype, device="cpu")
            v = torch.empty((n, *hv.shape[2:]), dtype=hv.dtype, device="cpu")
            # Same contiguity fast path as `preload_layer`, for the same
            # reason: this is the inverse read and was equally loop-bound.
            nb_need = (n + bs - 1) // bs
            if nb_need > 1 and hblks[nb_need - 1] - hblks[0] == nb_need - 1:
                src0 = hblks[0] * bs
                k.copy_(hk_flat[src0:src0 + n])
                v.copy_(hv_flat[src0:src0 + n])
            else:
                taken = 0
                while taken < n:
                    blk, off = divmod(taken, bs)
                    take = min(bs - off, n - taken)
                    src = hblks[blk] * bs + off
                    k[taken:taken + take].copy_(hk_flat[src:src + take])
                    v[taken:taken + take].copy_(hv_flat[src:src + take])
                    taken += take
            out[ln] = (k, v)
        self.prefill_stash[key] = out
        self.stash_tokens[key] = n

    def has_stash(self, key: str) -> bool:
        return key in self.prefill_stash

    def preload_from_stash(self, req_id: str, key: str) -> int:
        """Admit a request to the host tier from stashed prefill KV. Returns the
        number of tokens delivered."""
        kv = self.prefill_stash.get(key)
        if kv is None:
            raise RuntimeError(f"preload_from_stash: nothing stashed for {key}")
        n = self.stash_tokens[key]
        self.preload_req(req_id, n, kv)
        return n

    def drop_stash(self, key: str) -> None:
        self.prefill_stash.pop(key, None)
        self.stash_tokens.pop(key, None)

    def seed_req(self, req_id: str, resident_block_ids: list[int]) -> None:
        """Seed all layers' prefill KV for a request from resident, once."""
        if req_id in self.req_seeded or not self.resident:
            return
        for ln in self.layers:
            self.seed_layer(req_id, ln, self.resident[ln], resident_block_ids)
        self.req_seeded.add(req_id)

    # -- back-compat view ---------------------------------------------------
    @property
    def host(self) -> dict[str, torch.Tensor]:
        """The old single-tensor [2, H, ...] mirror, as a STACKED COPY.

        Kept only so external readers do not silently break. It materialises a
        copy, so it must never be used on the hot path -- the whole point of the
        split layout is that no such tensor is ever formed.
        """
        # materialises the mirror, so pending token writes must have landed
        self.flush_token_writes()
        return {ln: torch.stack((self.host_k[ln], self.host_v[ln]))
                for ln in self.layers}

    # -- per-step compaction ------------------------------------------------
    @property
    def _active_host(self) -> list[int]:
        """The sorted active host-block ids, materialised on demand.

        This used to be built eagerly every step, along with a `torch.tensor`
        copy of it that nothing ever read. Both were O(W) per step -- that is,
        per rider BLOCK -- which is exactly the cost this path was rebuilt to
        remove, so they are now derived from the runs only if someone asks.
        """
        out: list[int] = []
        for hs, _cs, ln in self._runs:
            out.extend(range(hs, hs + ln))
        return out

    @_active_host.setter
    def _active_host(self, value: list[int]) -> None:
        # Kept so the pre-existing __init__ assignment and any direct setter
        # still work; the runs are the real state.
        self._runs = _runs_from_sorted(list(value))

    def _plan_extents(self, active_req_ids: list[str]):
        """(runs, per-request compact bases, W) in O(r log r), or None.

        Returns None when the fast path does not apply, which is the signal to
        fall back to the general block-by-block construction.

        The fast path rests on a guarantee the allocator already makes and that
        `_issue_runs` already relies on: with `blocks_per_req > 0` a request is
        given ONE contiguous extent, `base .. base+n-1` (see `ensure_req`). Two
        consequences follow, and they are what make this O(r) instead of
        O(r*S/bs):

          * the sorted union of the cohort's blocks is the concatenation of r
            sorted ranges, so it never has to be enumerated block by block; and
          * a request's COMPACT ids are then also contiguous, so its whole
            block-table row is one arithmetic range rather than one dictionary
            lookup per block.

        Contiguity and disjointness are CHECKED, not assumed. Disjointness
        matters because the old code took a set union: if two requests shared a
        block, deduplication would make W smaller than the sum of the extent
        lengths, and the compact numbering computed here would drift from the
        numbering the dictionary produced.

        T5, `_whole_extent`: the run lengths become the EXTENT length rather
        than the used-block count. The merge below (`hs == prev_start + prev_len`
        extends the previous run) has always been here and has never fired,
        because consecutive extents are adjacent -- `base`, `base + bpr`,
        `base + 2*bpr` -- while the used blocks stop short of each extent's end.
        A decode request grows one block per `block_size` tokens, so that slack
        is permanent, and no packing of the extents removes it. Taking the whole
        extent closes the gap, the runs merge into ONE, and the per-layer H2D
        becomes 2 contiguous launches instead of 2r.

        The compact numbering is then at extent granularity, which is exactly
        why the used blocks must start AT the extent base: a request's compact
        ids are `extent_compact_base .. +n-1`, so the block-table row and
        `compact_of` both stay the arithmetic they already were, and the slack
        rows sit unused at the tail of each extent's compact range. Nothing
        reads them -- no block-table entry points there -- they are simply
        carried over the link, which is the cost of the change.
        """
        whole = self._whole_extent
        used_total = 0
        exts: list[tuple[int, int, int]] = []      # (host_start, length, row)
        for row, rid in enumerate(active_req_ids):
            blks = self.req_blocks.get(rid)
            if not blks:
                return None                        # a cohort member with no
                                                   # blocks: let the general
                                                   # path handle the shape
            n = len(blks)
            if blks[-1] - blks[0] != n - 1:
                return None                        # not one ascending run
            if whole:
                # T5: take the request's WHOLE extent, not just the blocks it
                # has filled. `ensure_req` hands out `base .. base+n-1` from
                # `extent * bpr`, so the used blocks start AT the extent base --
                # checked here rather than assumed, because if they did not, the
                # compact base computed below would be the extent's while
                # `_req_cbase` recorded the request's, and `compact_of` would
                # silently read the wrong rows.
                ext = self._req_extent.get(rid)
                if ext is None:
                    return None
                base = ext * self._bpr
                if blks[0] != base or n > self._bpr:
                    return None
                used_total += n
                exts.append((base, self._bpr, row))
            else:
                exts.append((blks[0], n, row))
        if not exts:
            return [], {}, 0
        if whole:
            # Refuse rather than silently pay for a loose extent. With the
            # runner's default bpr (sized for max_model_len) the slack is ~75%
            # and this would move 4x the bytes for the same launch saving, which
            # is a net loss -- the T3 failure shape. Falling back to the
            # per-block runs is correct and merely slower.
            span = sum(ln for _, ln, _ in exts)
            if span and 1.0 - used_total / span > self._whole_extent_max_slack:
                self.whole_extent_refused += 1
                whole = False
                exts = []
                for row, rid in enumerate(active_req_ids):
                    blks = self.req_blocks[rid]
                    exts.append((blks[0], len(blks), row))
            else:
                self.whole_extent_steps += 1
        exts.sort()
        runs: list[tuple[int, int, int]] = []
        bases: dict[int, int] = {}                 # row -> compact base
        w = 0
        for hs, ln, row in exts:
            if w and hs < runs[-1][0] + runs[-1][2]:
                return None                        # overlapping extents
            bases[row] = w
            if runs and hs == runs[-1][0] + runs[-1][2]:
                hs0, cs0, ln0 = runs[-1]           # adjacent: extend the run
                runs[-1] = (hs0, cs0, ln0 + ln)
            else:
                runs.append((hs, w, ln))
            w += ln
        return runs, bases, w

    def _plan_batched_gather(self):
        """(host_start, stride, length, count), or None.

        The per-layer gather copies each run from the host mirror into the
        rolling slot. The DESTINATIONS are contiguous and cumulative by
        construction, so if the SOURCES are evenly spaced and all runs are the
        same length, the whole gather is one strided copy per tensor rather
        than one per run -- 2L launches per step instead of 2*L*r.

        Every condition is checked rather than assumed, because getting this
        wrong would stream the wrong KV with no error:

          * all runs the same length, so one `length` describes them all;
          * compact starts exactly cumulative, i.e. run i lands at i*length,
            which is what lets the destination be one reshaped view;
          * host starts evenly spaced by a constant stride;
          * stride >= length, so the source windows do not overlap;
          * the last window inside the mirror.
        """
        runs = self._runs
        if len(runs) < 2:
            return None                    # nothing to batch
        n0 = runs[0][2]
        stride = runs[1][0] - runs[0][0]
        if stride < n0:
            return None
        for i, (hs, cs, n) in enumerate(runs):
            if n != n0 or cs != i * n0:
                return None
            if hs != runs[0][0] + i * stride:
                return None
        if runs[0][0] + (len(runs) - 1) * stride + n0 > self.h_max:
            return None
        return (runs[0][0], stride, n0, len(runs))

    def _strided_views(self, host: torch.Tensor, slot: torch.Tensor):
        """(dst, src) views for the batched gather, shaped [count, length, ...]."""
        hs0, stride, n0, cnt = self._batch_plan
        s0 = host.stride(0)
        src = host.as_strided(
            (cnt, n0) + tuple(host.shape[1:]),
            (stride * s0, s0) + tuple(host.stride()[1:]),
            hs0 * s0,
        )
        dst = slot.narrow(0, 0, cnt * n0).view(
            (cnt, n0) + tuple(slot.shape[1:]))
        return dst, src

    def build_step(
        self, active_req_ids: list[str], seq_lens: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compact the active cohort's host blocks into [0:W) and return
        (block_table [n_req, max_blk], seqused [n_req]) in active order."""
        _t = PT.now()
        # FIRST, before anything reads the mirror this step. See
        # flush_token_writes for why this placement is the correctness
        # argument and not merely tidy.
        self.flush_token_writes()
        self._events = {}
        n_req = len(active_req_ids)
        plan = self._plan_extents(active_req_ids)
        fast = plan is not None

        if fast:
            self._runs, row_cbase, self._w_cur = plan
            # compact_of() needs (host_base -> compact_base) per request; with
            # contiguous extents that is all it needs, so the per-BLOCK
            # dictionary is not built at all on this path.
            self._compact = {}
            self._req_cbase = {
                rid: (self.req_blocks[rid][0], row_cbase[row])
                for row, rid in enumerate(active_req_ids)
            }
        else:
            host_set: set[int] = set()
            for r in active_req_ids:
                host_set.update(self.req_blocks.get(r, []))
            srt = sorted(host_set)
            self._compact = {b: i for i, b in enumerate(srt)}
            self._req_cbase = {}
            self._w_cur = len(srt)
            self._runs = _runs_from_sorted(srt)

        if self._w_cur > self.w_max:
            raise RuntimeError(
                f"rolling region W={self._w_cur} exceeds w_max={self.w_max}"
            )
        self._batch_plan = self._plan_batched_gather()
        if self._batch_plan is not None:
            self.batched_steps += 1
        else:
            self.unbatched_steps += 1
        max_blk = max((len(self.req_blocks.get(r, [])) for r in active_req_ids),
                      default=1)

        # WAS A NAMED ARTIFACT: one Python-level tensor element assignment per
        # rider block, every step -- 1024 __setitem__ calls at R=8 / ctx 2048 /
        # block 16, measured at 0.00765 ms per block, and 3.9 ms per rider at
        # ctx 8192, which was 76% of the whole per-rider cost at the projection
        # target. The fast path below is one slice assignment per REQUEST, so
        # the term loses its dependence on context rather than shrinking by a
        # constant: O(r) instead of O(r*S/bs).
        _tbt = PT.now()
        bt = torch.zeros(n_req, max_blk, dtype=torch.int32)
        if fast:
            # `self._ar` is a preallocated arange, so the row is a VIEW and the
            # assignment allocates nothing.
            for row, rid in enumerate(active_req_ids):
                n = len(self.req_blocks[rid])
                cb = row_cbase[row]
                bt[row, :n] = self._ar[cb:cb + n]
        else:
            # General path: still O(blocks), but built as Python ints and
            # handed to torch ONCE rather than one tensor __setitem__ per
            # block, which is the expensive part.
            cm = self._compact
            rows = [[cm[hb] for hb in self.req_blocks.get(r, ())]
                    for r in active_req_ids]
            for row, vals in enumerate(rows):
                if vals:
                    bt[row, :len(vals)] = torch.tensor(vals, dtype=torch.int32)
        PT.add("build_step_bt_python", _tbt)
        seq = torch.tensor(seq_lens, dtype=torch.int32)
        PT.context(n_active=n_req, w_blocks=self._w_cur,
                   n_runs=len(self._runs), max_blk=int(max_blk),
                   layers=len(self.layers), block_size=self.block_size,
                   xfer_mode=self._xfer_mode, build_fast_path=int(fast))
        PT.add("build_step", _t)
        return bt, seq

    def compact_of(self, req_id: str, position: int) -> tuple[int, int]:
        """(compact rolling block id, offset) for a token position of a req.

        Called once per rider per step from the runner, so it is on the hot
        path. On the extent fast path it is arithmetic: the request's blocks
        are contiguous and so are its compact ids, so the compact id is the
        request's compact base plus the block offset within the request.
        """
        j = position // self.block_size
        cb = self._req_cbase.get(req_id)
        if cb is not None:
            return cb[1] + j, position % self.block_size
        hb = self.req_blocks[req_id][j]
        return self._compact[hb], position % self.block_size

    # -- per-layer streaming + write ---------------------------------------
    def _issue_runs(self, layer_idx: int) -> None:
        """Copy this layer's cohort KV from the pinned mirrors into the rolling
        slot, one contiguous `cudaMemcpyAsync` per run per tensor.

        This is the whole of the old `_gather_into` + H2D. There is no CPU
        staging step: because the mirrors are block-major, a run of host blocks
        [hs, hs+ln) is a single contiguous byte range, and so is its destination
        [cs, cs+ln) in the compact slot. The DMA engine does the compaction.

        With per-request extents the run count is just the number of active
        requests, so this is a handful of large copies. A badly fragmented
        active set degrades to more, smaller copies -- still correct, and still
        without touching the data on the CPU.
        """
        _t = PT.now()
        ln_name = self.layers[layer_idx]
        hk, hv = self.host_k[ln_name], self.host_v[ln_name]
        kslot = self.rolling_k[layer_idx % self.k]
        vslot = self.rolling_v[layer_idx % self.k]
        moved = 0
        nominal = 0
        launches = 0
        mode = self._xfer_mode
        if (mode == "real" and self._batch_plan is not None
                and self._batch_gather):
            # T3: one strided copy per tensor for the whole cohort.
            _hs0, _stride, n0, cnt = self._batch_plan
            dk, sk = self._strided_views(hk, kslot)
            dv, sv = self._strided_views(hv, vslot)
            dk.copy_(sk, non_blocking=True)
            dv.copy_(sv, non_blocking=True)
            full = 2 * cnt * n0 * kslot[0].numel() * kslot.element_size()
            self.stream_bytes += full
            self.stream_bytes_nominal += full
            self.stream_runs += 2
            PT.count("h2d_launches", 2)
            PT.count("h2d_batched_steps")
            PT.add("issue_runs", _t)
            return
        for hs, cs, n in self._runs:
            if n <= 0:
                continue
            full = 2 * kslot.narrow(0, cs, n).numel() * kslot.element_size()
            nominal += full
            if mode == "real":
                kslot.narrow(0, cs, n).copy_(hk.narrow(0, hs, n),
                                             non_blocking=True)
                vslot.narrow(0, cs, n).copy_(hv.narrow(0, hs, n),
                                             non_blocking=True)
                moved += full
                launches += 2
            elif mode == "minbytes":
                # One block row per copy: the launch is still issued on the same
                # stream in the same order, but there are almost no bytes behind
                # it. See _xfer_mode in __init__.
                kslot.narrow(0, cs, 1).copy_(hk.narrow(0, hs, 1),
                                             non_blocking=True)
                vslot.narrow(0, cs, 1).copy_(hv.narrow(0, hs, 1),
                                             non_blocking=True)
                moved += 2 * kslot.narrow(0, cs, 1).numel() * \
                    kslot.element_size()
                launches += 2
            # nobus: the narrow() views above were still built; no copy issued.
        self.stream_bytes += moved
        self.stream_bytes_nominal += nominal
        self.stream_runs += launches
        # The launch COUNT is the quantity a faster link leaves untouched, so
        # it is counted per step rather than derived from the rider count.
        PT.count("h2d_launches", launches)
        # T6. Mark where this layer's mirror READS were issued, so the deferred
        # mirror WRITE in `flush_token_writes` can be ordered after them -- see
        # the long comment there for the race and why it was invisible. Only
        # the MAIN stream is recorded here: inside `_launch_prefetch` the
        # current stream is the copy stream, and that whole stream is
        # synchronised instead, because one event cannot cover two streams.
        if (self._mirror_ev is not None and mode != "nobus"
                and (self._copy_stream is None
                     or torch.cuda.current_stream() != self._copy_stream)):
            self._mirror_ev.record()
        PT.add("issue_runs", _t)

    def writeback_layer(self, layer_idx: int) -> int:
        """SWAP-SEMANTICS EMULATION. Write this layer's whole rolling slot back
        to the host mirrors, and report the bytes moved.

        This is NOT part of the streaming design and is off by default. It
        exists to measure, on our own hardware, the cost of the return leg that
        a *swap*-based offload pays and a *streaming* one does not.

        The distinction is structural rather than an optimisation. If the
        offloaded unit is OWNED by one side -- as a layer's KV cache is in a
        swap design, where the device-side buffer is reclaimed for the next
        layer arriving -- then a unit that has just had this token's K,V
        appended is dirty and must be persisted before the buffer is reused. So
        every offloaded byte crosses the interconnect twice per generated
        token, and the return leg uses the slower direction of the link. If the
        device-side copy is a CACHE, as ours is (the host mirror stays
        authoritative and `write_token` writes the new token to both), a slot
        can be overwritten with nothing to save, and only the new token travels
        back.

        Enabling this makes our per-token traffic match a swap design's at the
        same offload volume, so the two policies can be compared with the
        mechanism difference removed -- or the mechanism difference measured
        with the policy held fixed.

        Returns bytes written (0 when disabled or the cohort is empty), so a
        harness can report traffic rather than infer it.
        """
        # T8. RETURN BEFORE FLUSHING when the emulation is off, which is the
        # default and therefore every production step.
        #
        # This method is called from the attention override on EVERY LAYER --
        # its call site guards on `getattr(pool, "writeback_layer", None) is not
        # None`, which tests that the method exists rather than that the
        # emulation is enabled, so it always fires. The flush used to run before
        # this early return, so a disabled feature was forcing a flush 64 times
        # per step.
        #
        # Measured: 64 flushes per step, every one with a single pending layer,
        # which is why T2's deferral never batched and why T7a's vectorised
        # scatter could never fire (it needs more than one pending layer). Worse,
        # each flush calls `_wt_event.synchronize()`, so this was 63 redundant
        # host-side waits on the D2H per step, partially serialising the host
        # with the copy on every layer.
        #
        # The flush is still required when the writeback actually runs, because
        # that overwrites mirror rows and pending token writes must land first.
        # It just belongs AFTER the decision to run.
        if not self._writeback or self._w_cur == 0:
            return 0
        # overwrites mirror rows, so pending token writes must land first
        self.flush_token_writes()
        ln_name = self.layers[layer_idx]
        hk, hv = self.host_k[ln_name], self.host_v[ln_name]
        kslot = self.rolling_k[layer_idx % self.k]
        vslot = self.rolling_v[layer_idx % self.k]
        moved = 0
        # Exactly the inverse of _issue_runs: same runs, same granularity, same
        # contiguity, opposite direction. Anything cheaper would understate the
        # cost we are trying to measure.
        for hs, cs, n in self._runs:
            hk.narrow(0, hs, n).copy_(kslot.narrow(0, cs, n), non_blocking=True)
            hv.narrow(0, hs, n).copy_(vslot.narrow(0, cs, n), non_blocking=True)
            moved += (kslot.narrow(0, cs, n).numel()
                      + vslot.narrow(0, cs, n).numel()) * kslot.element_size()
        self.writeback_bytes += moved
        return moved

    def _launch_prefetch(self, layer_idx: int) -> None:
        """Issue layer_idx's host->rolling copies on the side stream, so they
        overlap the current layer's attention."""
        if self._w_cur == 0 or not self._async:
            return
        # Don't overwrite the slot until its previous (main-stream) read is done.
        self._copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._copy_stream):
            self._issue_runs(layer_idx)
        ev = torch.cuda.Event()
        ev.record(self._copy_stream)
        self._events[layer_idx] = ev

    def stream_layer(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refresh this layer's rolling slot from the host mirrors.

        Returns (key_slot, value_slot), each a contiguous
        [w_max, block_size, nkv, head_dim] tensor that attention can consume
        directly.
        """
        _t = PT.now()
        kslot = self.rolling_k[layer_idx % self.k]
        vslot = self.rolling_v[layer_idx % self.k]
        if self._w_cur == 0:
            PT.add("stream_layer", _t)
            return kslot, vslot
        if not self._async:
            self._issue_runs(layer_idx)
        else:
            ev = self._events.pop(layer_idx, None)
            if ev is not None:
                torch.cuda.current_stream().wait_event(ev)  # prefetched copies
            else:
                self._issue_runs(layer_idx)                 # first layer / miss
            if layer_idx + 1 < len(self.layers):
                self._launch_prefetch(layer_idx + 1)
        # stream_layer minus issue_runs is the async event bookkeeping, so the
        # two together say whether RUNGD_ASYNC is buying or costing.
        PT.add("stream_layer", _t)
        return kslot, vslot

    def _wt_staging(self, like: torch.Tensor, n_layers: int):
        """Pinned staging for the deferred mirror write, [L, n, ...].

        Grown only, never shrunk: pinned allocation is expensive and the cohort
        size is stable in steady state. A regrow can only legitimately happen at
        the FIRST layer of a step, because the cohort does not change mid-step
        and `build_step` has flushed by then -- so a regrow while rows are still
        pending would silently drop them, and is refused.
        """
        n = like.shape[0]
        need = (self._wt_k is None or self._wt_k.shape[1] < n
                or self._wt_k.shape[0] != n_layers
                or self._wt_k.shape[2:] != like.shape[1:]
                or self._wt_k.dtype != like.dtype)
        if need:
            if self._wt_dirty:
                raise RuntimeError(
                    "rungd: mirror staging regrow with %d rows still pending; "
                    "the cohort changed mid-step, which would drop them"
                    % len(self._wt_pending))
            shape = (n_layers, max(n, 1)) + tuple(like.shape[1:])
            self._wt_k = torch.empty(shape, dtype=like.dtype, device="cpu",
                                     pin_memory=True)
            self._wt_v = torch.empty(shape, dtype=like.dtype, device="cpu",
                                     pin_memory=True)
            PT.count("wt_staging_realloc")
        return self._wt_k, self._wt_v

    def flush_token_writes(self) -> int:
        """Apply the deferred mirror writes. Idempotent; returns rows applied.

        ORDERING, which is the whole correctness argument. Between the async
        copy and this scatter the host mirror does NOT yet hold the new token,
        so every reader of the mirror must be ordered after it. The
        correctness-critical one is `_issue_runs`, which re-streams the mirror
        host-to-device on the HOT PATH at the next step: if it ran first it
        would carry stale KV and the output would be silently wrong.

        That is safe by construction rather than by a guard: `build_step` runs
        once per step before any `_issue_runs` of that step, and it calls this
        first. Nothing prefetches across a step boundary -- `stream_layer`'s
        prefetch is guarded by `layer_idx + 1 < len(self.layers)`.

        The CPU readers (`restore_layer`, `stash_from_host`, the host-tensor
        property) and the CPU writers (`seed_layer`, `preload_layer`) call this
        too. In decode steady state none of them is hit, so the cost is zero;
        when one is, it is a wait that has already elapsed plus a scatter that
        was going to happen anyway. ONE idempotent flush keyed on a dirty flag,
        rather than the three separate sync guards that the earlier attempt
        accumulated and was reverted partly for.
        """
        if not self._wt_dirty:
            return 0
        if self._wt_noflush:
            # DELIBERATELY BROKEN, for a negative control: DROP the pending
            # rows instead of applying them. The mirror then goes stale, which
            # is precisely T2's failure mode, and the token-identical test must
            # FAIL. If it passes, that test cannot see a missed flush and "the
            # suite is 9/9 with T2 on" is not evidence of T2's correctness --
            # a test that passes either way proves only that it is insensitive.
            #
            # Dropping rather than merely returning early matters. A first
            # version returned without clearing, which left _wt_dirty set and
            # let _wt_pending grow by one entry per layer per step forever. The
            # pool server then failed its completions, but from the LEAK rather
            # than from stale KV -- so the control was testing the wrong thing
            # and the test reported INCONCLUSIVE instead of FAIL.
            n_dropped = sum(x[2] for x in self._wt_pending)
            self._wt_pending.clear()
            self._wt_dirty = False
            self._wt_uniform = True
            PT.count("wt_flush_suppressed_rows", n_dropped)
            return 0
        if self._wt_event is not None:
            self._wt_event.synchronize()
        # T6, AND IT IS A CORRECTNESS FIX, not a tuning knob. The scatter below
        # CPU-writes the host mirror. The mirror's READERS are the H2D copies in
        # `_issue_runs`, issued with non_blocking=True from pinned memory, so
        # they are live DMAs that outlive the host call that launched them.
        #
        # The ordering argument in the docstring above says "nothing prefetches
        # across a step boundary", and that is true of where a prefetch is
        # ISSUED. What crosses the boundary is its COMPLETION. `stream_layer`
        # consumes the last prefetch with current_stream().wait_event(ev), which
        # orders the DEVICE and returns to the host immediately, so the host
        # runs on into the next step's build_step -- and straight into this
        # scatter -- while that DMA may still be reading the rows it is about to
        # overwrite.
        #
        # Measured 2026-10-05: with RUNGD_ASYNC=1 the token-identical test fails
        # INTERMITTENTLY (2 failures on nova07, a pass on nova06 -- which is the
        # signature of a race rather than a logic error). It stayed invisible
        # because run_suite.sh never sets RUNGD_ASYNC while rider_cost_curve.py
        # defaults it ON, so the suite tested one path and every measurement ran
        # the other.
        #
        # Before T2 the mirror write was a BLOCKING D2H, which synchronised the
        # host on every layer and made this safe by accident. T2 did not
        # introduce the hazard, it removed the sync that was hiding it.
        #
        # The wait is on the stream that READ the mirror, not on the main stream,
        # which is why it is nearly free: at the top of a step the newest such
        # DMA was issued a layer before the end of the PREVIOUS step and has long
        # since retired. Waiting on the main stream instead would hand back
        # exactly the sync T2 removed.
        #
        # BOTH stream families have to be covered, which a single event cannot
        # do: prefetches are issued on `_copy_stream` and the event-miss
        # fallback in `stream_layer` issues on the main stream, so one
        # re-recorded event would have the later record silently stand in for
        # the other stream's outstanding DMA. The copy stream is covered by
        # synchronising the stream itself (all its work, in issue order) and the
        # main stream by `_mirror_ev`.
        #
        # MEASURE THE HAZARD WINDOW, don't wait for its symptom. Output
        # divergence is a terrible detector for this: the race is intermittent,
        # so a passing run says nothing, and the negative control passed on its
        # first attempt. `query()` is non-blocking -- the Class A idiom this
        # repo already uses in `_drain_wait_events` -- so asking whether the
        # mirror's readers are STILL RUNNING at the moment the scatter is about
        # to overwrite their rows costs nothing and answers the question
        # directly. A non-zero count here is the race window existing, whether
        # or not any token came out wrong on this run.
        if self._mirror_ev is not None or self._copy_stream is not None:
            busy = False
            if self._copy_stream is not None:
                busy = busy or not self._copy_stream.query()
            if self._mirror_ev is not None:
                busy = busy or not self._mirror_ev.query()
            if busy:
                self.mirror_hazard_steps += 1
                PT.count("mirror_hazard_steps")
            self.mirror_flush_steps += 1
        if not self._no_mirror_sync:
            # Both or neither: the control has to restore the race in full, and
            # the event sync alone would silently close most of the window and
            # make the control look like the fix.
            if self._copy_stream is not None:
                self._copy_stream.synchronize()
            if self._mirror_ev is not None:
                self._mirror_ev.synchronize()
        _tf = PT.now()
        rows = 0
        # T7a FAST PATH: the whole scatter as ONE index_copy_ per tensor.
        #
        # Conditions, all checked:
        #   * the mirror is one allocation, so a single flat view spans layers;
        #   * every pending write shares host_pos and n (`_wt_uniform`), which
        #     holds because a rider's row inside its extent does not depend on
        #     the layer; and
        #   * the pending layers are 0..nl-1, so the source is a slice rather
        #     than a gather.
        # Any of them false and the per-layer loop below runs instead, which is
        # the pre-T7 behaviour exactly.
        # SORT BY LAYER INDEX FIRST. The pending list is appended in EXECUTION
        # order, while `_wt_k` is indexed by the layer index the runner assigns
        # from kv_cache_groups. The first version of this guard required
        # `lis == list(range(nl))`, which fails whenever those two orders differ
        # even by a swap -- and M15 measured exactly that: 2352 flushes, every
        # one on the slow path, with the row counts confirming all 64 layers
        # were present. Sorting 64 ints costs nothing and makes the guard
        # indifferent to execution order, which it should always have been.
        pend = sorted(self._wt_pending, key=lambda e: e[0])
        lis = [e[0] for e in pend]
        nl = len(lis)
        fast = (self._unified_mirror and self._wt_uniform and nl > 1
                and lis == list(range(nl)))
        if not fast:
            # Say WHY, so a fallback is diagnosable from the artifact instead of
            # needing a Stub repro. This is what finding the bug above cost.
            if not self._unified_mirror:
                PT.count("wt_fallback_not_unified")
            elif not self._wt_uniform:
                PT.count("wt_fallback_not_uniform")
            elif nl <= 1:
                PT.count("wt_fallback_single_layer")
            else:
                PT.count("wt_fallback_layer_gap")
        if fast:
            pos, n = pend[0][1], pend[0][2]
            tail = tuple(self._host_k_all.shape[3:])
            # global row = layer_offset + host_pos, vectorised over both
            idx = (pos.view(1, -1)
                   + self._wt_layer_off[:nl].view(-1, 1)).reshape(-1)
            self._host_k_all.view(-1, *tail).index_copy_(
                0, idx, self._wt_k[:nl, :n].reshape(-1, *tail))
            self._host_v_all.view(-1, *tail).index_copy_(
                0, idx, self._wt_v[:nl, :n].reshape(-1, *tail))
            rows = nl * n
            PT.count("wt_flush_vectorised")
        else:
            for li, host_pos, n in pend:
                ln = self.layers[li]
                hk, hv = self.host_k[ln], self.host_v[ln]
                bs = hk.shape[1]
                hk.view(hk.shape[0] * bs, *hk.shape[2:]).index_copy_(
                    0, host_pos, self._wt_k[li, :n])
                hv.view(hv.shape[0] * bs, *hv.shape[2:]).index_copy_(
                    0, host_pos, self._wt_v[li, :n])
                rows += n
            PT.count("wt_flush_per_layer")
        self._wt_pending.clear()
        self._wt_dirty = False
        self._wt_uniform = True
        self.wt_flushes += 1
        self.wt_deferred_rows += rows
        PT.count("wt_flush_rows", rows)
        # Attributing the post-T2 residual F needs this split: the async copy
        # launches are timed in write_token_d2h_async, and the deferred scatter
        # here. Without both, "65% of F is residual D2H" cannot be acted on.
        PT.add("wt_flush_scatter", _tf)
        return rows

    def _probe_staging(self, like: torch.Tensor):
        """Pinned staging for the probe arm, allocated once and grown only.

        Pinned allocation is expensive, so it must not happen per step. The
        cohort size changes rarely in steady state, and shrinking is never
        worth a reallocation.
        """
        n = like.shape[0]
        if (self._probe_k is None or self._probe_k.shape[0] < n
                or self._probe_k.shape[1:] != like.shape[1:]
                or self._probe_k.dtype != like.dtype):
            shape = (max(n, 1),) + tuple(like.shape[1:])
            self._probe_k = torch.empty(shape, dtype=like.dtype,
                                        device="cpu", pin_memory=True)
            self._probe_v = torch.empty(shape, dtype=like.dtype,
                                        device="cpu", pin_memory=True)
            PT.count("d2h_probe_realloc")
        return self._probe_k, self._probe_v

    def write_token(
        self,
        layer_idx: int,
        compact_ids: list[int],
        offsets: list[int],
        host_blocks: list[int],
        keys: torch.Tensor,     # [n, nkv, hd] on device
        values: torch.Tensor,
        slot_pos: torch.Tensor | None = None,
        host_pos: torch.Tensor | None = None,
    ) -> None:
        """Fold the new decode-token KV into the host mirror (host block) and the
        current rolling slot (compact block). Vectorized: flatten (block, offset)
        into one axis and scatter all rows with a single index_copy_ per target
        (no per-row Python loop)."""
        _t = PT.now()
        n = keys.shape[0]
        if n == 0:
            return
        kslot = self.rolling_k[layer_idx % self.k]
        vslot = self.rolling_v[layer_idx % self.k]
        ln = self.layers[layer_idx]
        hk, hv = self.host_k[ln], self.host_v[ln]
        bs = kslot.shape[1]
        dev = kslot.device
        # `slot_pos`/`host_pos` are the same for every layer of a step, so the
        # caller derives them once. Deriving them here meant three Python-list
        # conversions plus a PAGEABLE host->device transfer per layer -- 64x
        # over, on data fixed before the first layer ran. The fallback keeps
        # direct callers (tests, the promotion path) working.
        if slot_pos is None or host_pos is None:
            offs_l = torch.as_tensor(offsets, dtype=torch.long)
            # slot: [W, bs, nkv, hd] -> [W*bs, nkv, hd]; pos = compact*bs + off
            slot_pos = (torch.as_tensor(compact_ids, dtype=torch.long) * bs
                        + offs_l).to(dev)
            host_pos = torch.as_tensor(host_blocks, dtype=torch.long) * bs + offs_l
        kslot.view(kslot.shape[0] * bs, *kslot.shape[2:]).index_copy_(
            0, slot_pos, keys)
        vslot.view(vslot.shape[0] * bs, *vslot.shape[2:]).index_copy_(
            0, slot_pos, values)
        # host mirrors (pinned CPU): pos = host_block*bs + off
        # NAMED ARTIFACT: two PAGEABLE, hence blocking, D2H copies per layer --
        # 64 host/device serialisation points per step. The bytes are trivial
        # (one token's KV), so this is a synchronisation cost, not a bandwidth
        # one, and a faster link does not remove it. Timed on its own because
        # the earlier profiler charged this call the model's own compute and
        # produced a "21 ms" figure that was worth ~4 ms.
        if self._d2h_async:
            # Async into pinned staging; the host-side scatter is deferred to
            # the next step's build_step, behind one event wait.
            _td = PT.now()
            li = layer_idx
            sk, sv = self._wt_staging(keys, len(self.layers))
            sk[li, :n].copy_(keys, non_blocking=True)
            sv[li, :n].copy_(values, non_blocking=True)
            self._wt_event.record()
            # host_pos is derived once per step by the caller and handed to
            # every layer, so it may be a buffer that gets reused. The scatter
            # reads it a step later, so hold a COPY rather than aliasing it.
            # Does every layer in this step write the same rows? It does in
            # steady state -- a rider's row inside its extent does not depend
            # on the layer -- and that is the condition the vectorised scatter
            # in flush_token_writes needs. Tracked here rather than rechecked
            # there, so the check costs one comparison per layer instead of a
            # scan of the whole pending list.
            if self._wt_pending and self._wt_uniform:
                p0, n0 = self._wt_pending[0][1], self._wt_pending[0][2]
                if n != n0 or not torch.equal(host_pos, p0):
                    self._wt_uniform = False
            self._wt_pending.append((li, host_pos.clone(), n))
            self._wt_dirty = True
            PT.add("write_token_d2h_async", _td)
        elif self._d2h_probe:
            # Same two copies, same bytes, pinned destination, NON-BLOCKING,
            # and no scatter. Prices the synchronisation on its own.
            _td = PT.now()
            pk, pv = self._probe_staging(keys)
            pk[:n].copy_(keys, non_blocking=True)
            pv[:n].copy_(values, non_blocking=True)
            PT.add("write_token_d2h_probe", _td)
            PT.count("write_token_d2h_probe_calls")
        elif not self._skip_d2h:
            _td = PT.now()
            k_cpu, v_cpu = keys.to("cpu"), values.to("cpu")
            PT.add("write_token_d2h_blocking", _td)
            hk.view(hk.shape[0] * bs, *hk.shape[2:]).index_copy_(
                0, host_pos, k_cpu)
            hv.view(hv.shape[0] * bs, *hv.shape[2:]).index_copy_(
                0, host_pos, v_cpu)
        else:
            PT.count("write_token_d2h_skipped")
        PT.add("write_token", _t)
