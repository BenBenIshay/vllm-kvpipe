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
        # Per-layer pinned host mirror indexed by host block id.
        _, _, bs, nkv, hd = kv_shape
        hshape = (2, self.h_max, bs, nkv, hd)
        self.host: dict[str, torch.Tensor] = {
            ln: torch.empty(hshape, dtype=dtype, device="cpu", pin_memory=True)
            for ln in self.layers
        }
        # k_window GPU rolling slots, sized to the compact region.
        rshape = (2, self.w_max, bs, nkv, hd)
        self.rolling: list[torch.Tensor] = [
            torch.empty(rshape, dtype=dtype, device=device) for _ in range(self.k)
        ]
        # Async double-buffering (opt #2): a side copy stream + one pinned
        # staging buffer per layer (per-layer avoids a stage-reuse race), so
        # layer L+1's host->rolling H2D overlaps layer L's attention compute.
        # Requires k_window >= 2 so consecutive layers use distinct slots.
        # Async double-buffering hides the host->rolling transfer under compute.
        # It pays off in the TRANSFER-bound regime (large cohort / many layers /
        # high-BW C2C); on a small model + small cohort over PCIe its per-layer
        # bookkeeping can exceed the tiny transfer it hides, so it is OPT-IN
        # (VLLM_KVPIPE_RUNGD_ASYNC=1). Default sync path is unchanged.
        import os as _os
        self._async = (
            _os.environ.get("VLLM_KVPIPE_RUNGD_ASYNC", "0") == "1"
            and self.k >= 2 and torch.cuda.is_available()
        )
        self._async_min_w = 0
        self._copy_stream = torch.cuda.Stream() if self._async else None
        self._stage: list[torch.Tensor] = [
            torch.empty(rshape, dtype=dtype, device="cpu", pin_memory=True)
            for _ in range(len(self.layers))
        ] if self._async else []
        # Item 1a: the SYNC path used to do `index_select(...).to(device)`, and
        # index_select's output is NOT pinned -> a pageable H2D. Measured on
        # A100: pageable 11.17 GB/s vs pinned 25.08 GB/s (2.25x). Gather
        # straight into this always-present pinned buffer instead, so the
        # default path gets a pinned copy too.
        self._sync_stage: torch.Tensor = torch.empty(
            rshape, dtype=dtype, device="cpu", pin_memory=True)
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
        self.host[layer_name].index_copy_(1, dst, gathered)

    def mark_seeded(self, req_id: str) -> None:
        self.req_seeded.add(req_id)

    def seed_req(self, req_id: str, resident_block_ids: list[int]) -> None:
        """Seed all layers' prefill KV for a request from resident, once."""
        if req_id in self.req_seeded or not self.resident:
            return
        for ln in self.layers:
            self.seed_layer(req_id, ln, self.resident[ln], resident_block_ids)
        self.req_seeded.add(req_id)

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
    def _sync_stream_into(self, layer_idx: int, slot: torch.Tensor) -> None:
        """Blocking host->rolling refresh (fallback / first layer of a step).

        Gathers directly into a PINNED staging buffer (`out=`), then does a
        pinned H2D. The previous form built an unpinned intermediate via
        `index_select(...).to(device)`, costing a pageable copy (2.25x slower
        on A100). Same stream as the consumer, so ordering is preserved.
        """
        w = self._w_cur
        stage = self._sync_stage.narrow(1, 0, w)
        self._gather_into(layer_idx, stage)
        slot.narrow(1, 0, w).copy_(stage, non_blocking=True)

    def _gather_into(self, layer_idx: int, stage: torch.Tensor) -> None:
        """Fill `stage` (pinned, compact order) from this layer's host mirror.

        Uses maximal contiguous runs, so with per-request extents this is a few
        big memcpys instead of a block-granular index_select. Falls back to
        index_select if the active set is badly fragmented (legacy scattered
        allocation), where run-by-run would be slower than one gather.
        """
        src = self.host[self.layers[layer_idx]]
        if self._runs and len(self._runs) * 8 <= self._w_cur:
            for hs, cs, ln in self._runs:
                stage.narrow(1, cs, ln).copy_(src.narrow(1, hs, ln))
        else:
            torch.index_select(src, 1, self._active_host_cpu, out=stage)

    def _launch_prefetch(self, layer_idx: int) -> None:
        """Issue layer_idx's host->rolling H2D on the side stream (overlaps the
        current layer's attention). Uses layer_idx's own pinned stage buffer."""
        if self._w_cur == 0 or not self._async:
            return
        slot = self.rolling[layer_idx % self.k]
        stage_v = self._stage[layer_idx].narrow(1, 0, self._w_cur)
        # Gather the cohort's host blocks into this layer's pinned stage (CPU).
        self._gather_into(layer_idx, stage_v)
        # Don't overwrite the slot until its previous (main-stream) read is done.
        self._copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._copy_stream):
            slot.narrow(1, 0, self._w_cur).copy_(stage_v, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(self._copy_stream)
        self._events[layer_idx] = ev

    def stream_layer(self, layer_idx: int) -> torch.Tensor:
        """Refresh this layer's rolling slot from the host mirror and prefetch the
        next layer. With double-buffering the H2D overlaps compute."""
        slot = self.rolling[layer_idx % self.k]
        if self._w_cur == 0:
            return slot
        if not self._async or self._w_cur < self._async_min_w:
            self._sync_stream_into(layer_idx, slot)         # tiny transfer / off
        else:
            ev = self._events.pop(layer_idx, None)
            if ev is not None:
                torch.cuda.current_stream().wait_event(ev)  # prefetched H2D done
            else:
                self._sync_stream_into(layer_idx, slot)     # first layer / miss
            if layer_idx + 1 < len(self.layers):
                self._launch_prefetch(layer_idx + 1)
        return slot

    def write_token(
        self,
        layer_idx: int,
        compact_ids: list[int],
        offsets: list[int],
        host_blocks: list[int],
        keys: torch.Tensor,     # [n, nkv, hd] on device
        values: torch.Tensor,
    ) -> None:
        """Fold the new decode-token KV into the host mirror (host block) and the
        current rolling slot (compact block). Vectorized: flatten (block, offset)
        into one axis and scatter all rows with a single index_copy_ per target
        (no per-row Python loop)."""
        n = keys.shape[0]
        if n == 0:
            return
        slot = self.rolling[layer_idx % self.k]
        ln = self.layers[layer_idx]
        host = self.host[ln]
        bs = slot.shape[2]
        dev = slot.device
        offs_l = torch.as_tensor(offsets, dtype=torch.long)
        # slot: [2, W, bs, nkv, hd] -> [2, W*bs, nkv, hd]; pos = compact*bs+off
        slot_pos = (torch.as_tensor(compact_ids, dtype=torch.long) * bs
                    + offs_l).to(dev)
        slot_flat = slot.view(2, slot.shape[1] * bs, *slot.shape[3:])
        slot_flat[0].index_copy_(0, slot_pos, keys)
        slot_flat[1].index_copy_(0, slot_pos, values)
        # host mirror (pinned CPU): pos = host_block*bs+off
        host_pos = torch.as_tensor(host_blocks, dtype=torch.long) * bs + offs_l
        host_flat = host.view(2, host.shape[1] * bs, *host.shape[3:])
        host_flat[0].index_copy_(0, host_pos, keys.to("cpu"))
        host_flat[1].index_copy_(0, host_pos, values.to("cpu"))
