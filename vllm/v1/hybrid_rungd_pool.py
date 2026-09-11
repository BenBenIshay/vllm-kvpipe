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
        self.host_k: dict[str, torch.Tensor] = {
            ln: torch.empty(hshape, dtype=dtype, device="cpu", pin_memory=True)
            for ln in self.layers
        }
        self.host_v: dict[str, torch.Tensor] = {
            ln: torch.empty(hshape, dtype=dtype, device="cpu", pin_memory=True)
            for ln in self.layers
        }
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
        import os as _os
        # Swap-semantics emulation (see writeback_layer). Off by default: this
        # is measurement apparatus, not part of the design.
        self._writeback = (
            _os.environ.get("VLLM_KVPIPE_RUNGD_WRITEBACK", "0") == "1"
        )
        self.writeback_bytes = 0
        # Bytes pulled host->device by the per-layer refresh. Counted so a
        # harness can compare arms on TRAFFIC rather than on request counts.
        self.stream_bytes = 0
        self._async = (
            _os.environ.get("VLLM_KVPIPE_RUNGD_ASYNC", "0") == "1"
            and self.k >= 2 and torch.cuda.is_available()
        )
        self._copy_stream = torch.cuda.Stream() if self._async else None
        self._events: dict[int, torch.cuda.Event] = {}
        self._active_host_cpu: torch.Tensor | None = None
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
        self._active_host: list[int] = []            # sorted host block ids
        self._compact: dict[int, int] = {}           # host block id -> [0:W)
        self._w_cur = 0

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
        return {ln: torch.stack((self.host_k[ln], self.host_v[ln]))
                for ln in self.layers}

    # -- per-step compaction ------------------------------------------------
    def build_step(
        self, active_req_ids: list[str], seq_lens: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compact the active cohort's host blocks into [0:W) and return
        (block_table [n_req, max_blk], seqused [n_req]) in active order."""
        host_set: set[int] = set()
        for r in active_req_ids:
            host_set.update(self.req_blocks.get(r, []))
        self._active_host = sorted(host_set)
        self._compact = {b: i for i, b in enumerate(self._active_host)}
        self._w_cur = len(self._active_host)
        # Fresh per-step: the compact host-block set + a clean prefetch chain.
        self._active_host_cpu = torch.tensor(self._active_host, dtype=torch.long)
        self._events = {}
        # Maximal contiguous runs -> contiguous slice copies replace the gather.
        runs: list[tuple[int, int, int]] = []
        if self._active_host:
            start = prev = self._active_host[0]
            cstart = 0
            for i, hb in enumerate(self._active_host[1:], 1):
                if hb != prev + 1:
                    runs.append((start, cstart, prev - start + 1))
                    start, cstart = hb, i
                prev = hb
            runs.append((start, cstart, prev - start + 1))
        self._runs = runs
        if self._w_cur > self.w_max:
            raise RuntimeError(
                f"rolling region W={self._w_cur} exceeds w_max={self.w_max}"
            )
        max_blk = max((len(self.req_blocks.get(r, [])) for r in active_req_ids),
                      default=1)
        bt = torch.zeros(len(active_req_ids), max_blk, dtype=torch.int32)
        for row, r in enumerate(active_req_ids):
            for j, hb in enumerate(self.req_blocks.get(r, [])):
                bt[row, j] = self._compact[hb]
        seq = torch.tensor(seq_lens, dtype=torch.int32)
        return bt, seq

    def compact_of(self, req_id: str, position: int) -> tuple[int, int]:
        """(compact rolling block id, offset) for a token position of a req."""
        hb = self.req_blocks[req_id][position // self.block_size]
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
        ln_name = self.layers[layer_idx]
        hk, hv = self.host_k[ln_name], self.host_v[ln_name]
        kslot = self.rolling_k[layer_idx % self.k]
        vslot = self.rolling_v[layer_idx % self.k]
        moved = 0
        for hs, cs, n in self._runs:
            kslot.narrow(0, cs, n).copy_(hk.narrow(0, hs, n), non_blocking=True)
            vslot.narrow(0, cs, n).copy_(hv.narrow(0, hs, n), non_blocking=True)
            moved += 2 * kslot.narrow(0, cs, n).numel() * kslot.element_size()
        self.stream_bytes += moved

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
        if not self._writeback or self._w_cur == 0:
            return 0
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
        kslot = self.rolling_k[layer_idx % self.k]
        vslot = self.rolling_v[layer_idx % self.k]
        if self._w_cur == 0:
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
        return kslot, vslot

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
        k_cpu, v_cpu = keys.to("cpu"), values.to("cpu")
        hk.view(hk.shape[0] * bs, *hk.shape[2:]).index_copy_(0, host_pos, k_cpu)
        hv.view(hv.shape[0] * bs, *hv.shape[2:]).index_copy_(0, host_pos, v_cpu)
