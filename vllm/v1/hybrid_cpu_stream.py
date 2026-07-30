# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal, self-contained CPU-tier KV streaming for hybrid mixed batches.

Decoupled from the DirectKWindow rolling state machine (which is built for R1's
K-tensor layout and mis-maps in hybrid's full-N layout). This mirrors the
CPU-tier cohort's KV blocks to pinned host memory and refreshes them into their
*resident* GPU blocks per layer, so single-pass attention reads correct data.

Correctness-first: synchronous, whole-block index copies. No rolling, no
eviction, no event pipeline. Perf (async double-buffering) comes later, once
rung (c) passes. Whole KV blocks are copied along the block dimension, so the
K/V split and head layout are preserved verbatim (no per-element math).
"""
from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _block_dim(kv: torch.Tensor) -> int:
    # Exact match to cpu_kv_table._block_dim. Paged [blocks, 2, page, ...] is
    # "blocks-first" (check shape[1]==2 FIRST — a 2-block slice [2,2,...] would
    # otherwise be misread as K/V-first); the [2, blocks, ...] K/V-first layout
    # puts blocks on dim 1.
    if kv.ndim >= 3 and kv.shape[1] == 2:
        return 0
    return 1 if kv.ndim >= 1 and kv.shape[0] == 2 else 0


class HybridCpuStreamer:
    def __init__(
        self,
        kv_caches: dict[str, torch.Tensor],
        layer_names: list[str],
    ) -> None:
        self._kv = kv_caches  # layer_name -> resident GPU tensor (authoritative)
        self._layers = [ln for ln in layer_names if ln in kv_caches]
        # Pinned host mirror, one per layer, same shape as the resident tensor.
        self._cpu: dict[str, torch.Tensor] = {}
        for ln in self._layers:
            t = kv_caches[ln]
            self._cpu[ln] = torch.empty(
                t.shape, dtype=t.dtype, device="cpu", pin_memory=True
            )
        self._bd = _block_dim(kv_caches[self._layers[0]]) if self._layers else 0
        # Physical block ids currently owned by the CPU-tier cohort.
        self._cpu_blocks: set[int] = set()

    # -- staging (GPU resident -> host mirror) ------------------------------
    def stage_blocks(self, physical_block_ids: list[int]) -> None:
        """D2H copy the given blocks (all layers) resident GPU -> host mirror."""
        blocks = [b for b in physical_block_ids if b is not None and b >= 0]
        if not blocks:
            return
        idx_cpu = torch.tensor(sorted(set(blocks)), dtype=torch.long)
        g0 = self._kv[self._layers[0]]
        nblk = g0.shape[self._bd]
        if int(idx_cpu.max()) >= nblk or int(idx_cpu.min()) < 0:
            logger.error(
                "HybridCpuStreamer.stage_blocks: block id out of range "
                "(max=%d, min=%d, nblocks=%d, block_dim=%d, tensor_shape=%s)",
                int(idx_cpu.max()), int(idx_cpu.min()), nblk, self._bd,
                tuple(g0.shape),
            )
            return
        self._cpu_blocks.update(blocks)
        torch.cuda.synchronize()
        try:
            for ln in self._layers:
                g = self._kv[ln]
                idx_g = idx_cpu.to(g.device)
                gathered = g.index_select(self._bd, idx_g).to("cpu")
                self._cpu[ln].index_copy_(self._bd, idx_cpu, gathered)
            torch.cuda.synchronize()
        except Exception as e:
            logger.error("HybridCpuStreamer.stage_blocks failed: %s "
                         "(block_dim=%d, tensor_shape=%s, nblocks_idx=%d)",
                         e, self._bd, tuple(g0.shape), idx_cpu.numel())
            raise

    def stage_new_blocks(self, physical_block_ids: list[int]) -> None:
        """Stage only blocks not already mirrored (captures cohort growth)."""
        fresh = [
            b for b in physical_block_ids
            if b is not None and b >= 0 and b not in self._cpu_blocks
        ]
        if fresh:
            self.stage_blocks(fresh)

    # -- per-layer refresh (host mirror -> GPU resident) -------------------
    def load_layer(self, layer_name: str) -> None:
        """H2D copy this layer's CPU-cohort blocks host mirror -> resident GPU."""
        if layer_name not in self._cpu or not self._cpu_blocks:
            return
        idx_cpu = torch.tensor(sorted(self._cpu_blocks), dtype=torch.long)
        g = self._kv[layer_name]
        try:
            idx_g = idx_cpu.to(g.device)
            host_slice = self._cpu[layer_name].index_select(self._bd, idx_cpu)
            g.index_copy_(self._bd, idx_g, host_slice.to(g.device))
        except Exception as e:
            logger.error("HybridCpuStreamer.load_layer(%s) failed: %s "
                         "(block_dim=%d, g_shape=%s, host_shape=%s, nidx=%d)",
                         layer_name, e, self._bd, tuple(g.shape),
                         tuple(self._cpu[layer_name].shape), idx_cpu.numel())
            raise

    # -- save new decode-token KV for the cohort (resident -> host) --------
    def save_layer(self, layer_name: str) -> None:
        """D2H refresh this layer's cohort blocks (captures the new token KV)."""
        if layer_name not in self._cpu or not self._cpu_blocks:
            return
        idx_cpu = torch.tensor(sorted(self._cpu_blocks), dtype=torch.long)
        g = self._kv[layer_name]
        idx_g = idx_cpu.to(g.device)
        gathered = g.index_select(self._bd, idx_g).to("cpu")
        self._cpu[layer_name].index_copy_(self._bd, idx_cpu, gathered)

    def release_blocks(self, physical_block_ids: list[int]) -> None:
        for b in physical_block_ids:
            self._cpu_blocks.discard(b)

    def clear(self) -> None:
        self._cpu_blocks.clear()

    # ================= Step 2: compact rolling region ======================
    # The saving: instead of the CPU cohort holding its KV resident in all
    # `num_layers` per-layer tensors, hold only `k_window` COMPACT rolling
    # tensors (each big enough for the cohort's block footprint W) and stream
    # layer L's KV into slot `L % k_window` just before layer L's attention.
    # A block-table remap sends the CPU rows into the compacted [0:W) ids.

    def build_rolling_region(self, k_window: int, w_max_blocks: int) -> None:
        """Allocate k_window compact GPU rolling tensors (once).

        Each mirrors a resident layer tensor's shape but with `w_max_blocks`
        blocks along the block dimension instead of the full pool.
        """
        if not self._layers:
            return
        g = self._kv[self._layers[0]]
        shape = list(g.shape)
        shape[self._bd] = int(w_max_blocks)
        self._k_roll = max(1, int(k_window))
        self._w_max = int(w_max_blocks)
        self._rolling = [
            torch.empty(shape, dtype=g.dtype, device=g.device)
            for _ in range(self._k_roll)
        ]
        # Populated by set_cpu_cohort each step.
        self._cpu_sorted_host: torch.Tensor | None = None  # resident ids, long
        self._compact_map: dict[int, int] = {}
        self._w_cur = 0

    def set_cpu_cohort(self, cpu_block_ids: list[int]) -> None:
        """Fix this step's resident->compact block map (sorted, deduped)."""
        blocks = sorted({b for b in cpu_block_ids if b is not None and b >= 0})
        if len(blocks) > self._w_max:
            raise ValueError(
                f"CPU cohort footprint {len(blocks)} exceeds rolling region "
                f"w_max={self._w_max}"
            )
        self._compact_map = {b: i for i, b in enumerate(blocks)}
        self._cpu_sorted_host = torch.tensor(blocks, dtype=torch.long)
        self._w_cur = len(blocks)
        # Keep the block set coherent so save_layer/load_layer (host mirror
        # refresh) operate on exactly this cohort.
        self._cpu_blocks = set(blocks)

    def load_layer_rolling(self, layer_name: str, layer_idx: int) -> torch.Tensor | None:
        """Stream layer L's CPU-cohort KV host mirror -> rolling slot L%k.

        Returns the rolling slot tensor (so the caller can hand it to the
        attention impl), or None if inactive.
        """
        if (
            layer_name not in self._cpu
            or self._cpu_sorted_host is None
            or self._w_cur == 0
            or not getattr(self, "_rolling", None)
        ):
            return None
        slot = self._rolling[layer_idx % self._k_roll]
        # host mirror rows for this cohort, in sorted(=compact) order
        src = self._cpu[layer_name].index_select(self._bd, self._cpu_sorted_host)
        slot.narrow(self._bd, 0, self._w_cur).copy_(src.to(slot.device))
        return slot

    def rolling_slot_for(self, layer_idx: int) -> torch.Tensor | None:
        if not getattr(self, "_rolling", None):
            return None
        return self._rolling[layer_idx % self._k_roll]

    def remap_rows(self, rows_block_ids: list[list[int]]) -> list[list[int]]:
        """Map each row's resident block ids -> compacted rolling ids."""
        out: list[list[int]] = []
        for row in rows_block_ids:
            out.append([self._compact_map[b] for b in row if b in self._compact_map])
        return out
