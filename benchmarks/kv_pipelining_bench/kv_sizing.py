"""Shared helpers for fake KV tensor sizes (fp16 K+V one layer, one block)."""
from __future__ import annotations

import math

_ALIGN = 256


def numel_from_chunk_mib_fp16(chunk_mib: int) -> int:
    return int(chunk_mib) * 1024 * 1024 // 2


def numel_from_kv_block_one_layer(
    block_tokens: int, n_kv_heads: int, head_dim: int
) -> int:
    return 2 * int(block_tokens) * int(n_kv_heads) * int(head_dim)


def align_numel(n: int) -> int:
    return int(math.ceil(n / _ALIGN) * _ALIGN)
