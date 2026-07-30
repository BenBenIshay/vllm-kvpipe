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
