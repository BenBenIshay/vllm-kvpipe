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


def bytes_per_token_from_config(vllm_config) -> int | None:
    """Per-request KV bytes per token, derived from the running model.

    This exists because the hand-set `--hybrid-b-tok` was wrong. Every harness
    in `benchmarks/kv_pipelining_bench` passed 131072, which is a 32-layer
    model; Qwen2.5-32B has 64 layers, so the true figure is 262144. B1 was
    therefore charging each host-tier request HALF its real interconnect cost
    and would admit about twice as many riders as can hide inside a step --
    silently, and in the direction that breaks the ITL claim.

    Returns None when it cannot be derived, so callers keep whatever explicit
    value they were given rather than substituting a guess.

    Uses the PER-WORKER KV head count: the transfer budget is one worker's
    link, and under TP each worker moves only its own shard.
    """
    try:
        from vllm.utils.torch_utils import get_kv_cache_torch_dtype

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        num_layers = int(model_config.get_num_layers(parallel_config))
        num_kv_heads = int(model_config.get_num_kv_heads(parallel_config))
        head_dim = int(model_config.get_head_size())
        dtype = get_kv_cache_torch_dtype(
            vllm_config.cache_config.cache_dtype, model_config.dtype)
        dtype_bytes = int(dtype.itemsize)
    except Exception:                                          # noqa: BLE001
        return None
    if min(num_layers, num_kv_heads, head_dim, dtype_bytes) <= 0:
        return None
    return bytes_per_token(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
    )
