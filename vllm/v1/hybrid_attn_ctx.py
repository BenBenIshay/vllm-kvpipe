# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step hybrid decode attention context.

Carries the tier split for the current forward from the model runner into
``FlashAttentionImpl.forward`` without threading a new argument through the
whole call stack. Safe as a module global because a KV connector forces eager
execution (no CUDA-graph capture), so each step sets/reads/clears this
synchronously on one worker thread.

Step 1 (same-pool two-pass validation) only needs the row-index split. Step 2
(the capacity win) will additionally carry the CPU cohort's rolling KV caches
and rolling block table so the CPU pass reads a different physical region.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class HybridDecodeSplit:
    # Row indices (into the decode query / block_table) for each tier.
    hbm_idx: torch.Tensor  # int64, HBM-resident rows
    cpu_idx: torch.Tensor  # int64, CPU-tier rows
    # Step 2 (optional): CPU cohort reads a separate rolling region.
    rolling_kv_caches: dict[str, torch.Tensor] | None = None  # layer_name -> tensor
    rolling_block_table: torch.Tensor | None = None  # [n_cpu, max_blk], cpu_idx order
    rolling_seqused_k: torch.Tensor | None = None    # [n_cpu], cpu_idx order
    # Step 2 rung-c (gather-from-resident): the CPU pass gathers its resident
    # blocks into a compacted layout and attends with rolling_block_table. This
    # validates the compacted dual-region attention math without needing the
    # host->rolling streaming or the block-manager saving (those are rung-d).
    rolling_from_resident: bool = False
    cpu_sorted_ids: torch.Tensor | None = None  # resident block ids, long, sorted
    # rung-d (serve-more): the CPU cohort's KV is streamed from a host mirror
    # into a k_window-shared rolling region (HybridCpuStreamer), so it costs
    # k_window layer-copies instead of num_layers. When set, the CPU pass reads
    # the streamed rolling slot for each layer instead of gathering resident.
    rungd: bool = False
    rungd_streamer: object | None = None  # HybridCpuStreamer
    rungd_layer_index: dict | None = None  # layer_name -> layer_idx
    # Current decode token position per CPU row (cpu_idx order), for the
    # write-redirect: new-token KV -> host mirror + rolling slot.
    cpu_cur_resident: list | None = None  # resident block id of the current token
    cpu_cur_compact: list | None = None   # its compacted rolling block id
    cpu_cur_offset: list | None = None    # offset within the block
    # rung-d FREE (serve-more): CPU cohort KV in a host-block-id pool, resident
    # blocks reclaimed. When set, the CPU pass streams/writes via the pool.
    rungd_pool: object | None = None      # RungdHostPool
    cpu_cur_hostblock: list | None = None  # host block id of the current token
    # Request-row indices of the CPU decode rows (seqused_k is per-request), used
    # to mask them out of the standard attention pass (opt #5) since the pool
    # override recomputes them anyway.
    cpu_req_rows: torch.Tensor | None = None
    # The largest value in `cpu_req_rows`, as a plain int. Carried rather than
    # recomputed because the attention path needs it EVERY layer, and taking
    # `int(cpu_req_rows.max())` there is a GPU reduction plus a .item() -- 64
    # pipeline drains per step from a guard whose whole purpose is to save
    # work. The rows are fixed for the step, so this is loop-invariant.
    cpu_req_rows_max: int | None = None
    # Flattened (block, offset) write positions for this step's new token, in
    # cpu_idx order: `slot_pos` indexes the rolling GPU slot, `host_pos` the
    # pinned host mirror. Derived ONCE here rather than in `write_token`, which
    # runs per layer and was rebuilding both from Python lists 64 times over --
    # three `as_tensor` conversions and a pageable host->device transfer each
    # time, on data that cannot change between layers.
    cpu_slot_pos: torch.Tensor | None = None   # device
    cpu_host_pos: torch.Tensor | None = None   # cpu
    # Longest pooled sequence this step, and the trivial cu_seqlens_q for the
    # CPU rows. Both are fixed once `build_step` has run, but the tier backend
    # was recomputing `int(seqused_k.max())` per layer -- a device reduction
    # plus a .item(), i.e. another 64 pipeline drains per step -- and building
    # the same arange 64 times. Taken on the CPU-side tensor before it moves to
    # device, the max costs nothing.
    rolling_max_seqlen_k: int | None = None
    cpu_cu_seqlens_q: torch.Tensor | None = None
    # `seqused_k` with the pooled rows masked to length 1, so the standard pass
    # skips KV reads the pool override is going to overwrite anyway. Cached
    # because it is identical for every layer of a step.
    seqused_masked: torch.Tensor | None = None
    meta: dict = field(default_factory=dict)


_CURRENT: HybridDecodeSplit | None = None


def set_hybrid_decode_split(split: HybridDecodeSplit | None) -> None:
    global _CURRENT
    _CURRENT = split


def get_hybrid_decode_split() -> HybridDecodeSplit | None:
    return _CURRENT


def clear_hybrid_decode_split() -> None:
    global _CURRENT
    _CURRENT = None
