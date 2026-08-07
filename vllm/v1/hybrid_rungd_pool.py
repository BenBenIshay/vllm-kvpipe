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
        self.free_list: list[int] = list(range(self.h_max))
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
        while len(blks) < num_logical_blocks:
            if not self.free_list:
                raise RuntimeError("RungdHostPool exhausted host blocks")
            blks.append(self.free_list.pop())
        return blks

    def free_req(self, req_id: str) -> None:
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
    def stream_layer(self, layer_idx: int) -> torch.Tensor:
        """Refresh the rolling slot for this layer from the host mirror."""
        slot = self.rolling[layer_idx % self.k]
        if self._w_cur == 0:
            return slot
        ln = self.layers[layer_idx]
        idx = torch.tensor(self._active_host, dtype=torch.long)
        src = self.host[ln].index_select(1, idx)          # [2, W, ...] host
        slot.narrow(1, 0, self._w_cur).copy_(src.to(slot.device))
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
        current rolling slot (compact block)."""
        slot = self.rolling[layer_idx % self.k]
        ln = self.layers[layer_idx]
        host = self.host[ln]
        kc = keys.to("cpu")
        vc = values.to("cpu")
        for r in range(keys.shape[0]):
            off = offsets[r]
            slot[0, compact_ids[r], off].copy_(keys[r])
            slot[1, compact_ids[r], off].copy_(values[r])
            host[0, host_blocks[r], off].copy_(kc[r])
            host[1, host_blocks[r], off].copy_(vc[r])
