# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Roofline helpers for hybrid HBM + CPU-tier KV admission."""

from __future__ import annotations

import os


def effective_pcie_gbps(config_gbps: float = 0.0) -> float:
    """Effective pinned H2D GB/s: config > env > default."""
    if config_gbps > 0:
        return config_gbps
    env = os.environ.get("VLLM_KVPIPE_PCIE_GBPS", "").strip()
    if env:
        try:
            return max(0.0, float(env))
        except ValueError:
            pass
    return 25.1


def bytes_per_token(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    return num_layers * 2 * num_kv_heads * head_dim * dtype_bytes


def hide_budget_tok_slots(itl_ms: float, x_eff_gbps: float, b_tok: int) -> float:
    if itl_ms <= 0 or x_eff_gbps <= 0 or b_tok <= 0:
        return 0.0
    return (itl_ms / 1000.0) * x_eff_gbps * 1e9 / b_tok


def max_offload_batch(
    seq_tokens: int,
    itl_ms: float,
    x_eff_gbps: float,
    b_tok: int,
) -> int:
    if seq_tokens <= 0:
        return 0
    theta = hide_budget_tok_slots(itl_ms, x_eff_gbps, b_tok)
    return max(0, int(theta // seq_tokens))


def max_offload_from_env() -> int | None:
    """Optional cap on CPU-tier slots (VLLM_KVPIPE_MAX_OFFLOAD_REQS)."""
    raw = os.environ.get("VLLM_KVPIPE_MAX_OFFLOAD_REQS", "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None
