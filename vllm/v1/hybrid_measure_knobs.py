# SPDX-License-Identifier: Apache-2.0
"""Measurement-only knobs for the hybrid host-tier path.

These exist to make a quantity IDENTIFIABLE, not to make the system better.
Every one of them can produce numerically wrong output, so each is gated twice:
the knob itself, plus an explicit `..._UNSAFE=1` acknowledgement. Nothing here
is reachable from a default configuration, and the correctness suite must never
set any of it.

Keep this module free of heavy imports: it is read on the attention hot path.
"""

from __future__ import annotations

import os

_WARNED: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(msg, flush=True)


def _int_env(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def rungd_layer_limit(n_layers: int) -> int:
    """How many layers may use the host path for riders. `n_layers` = all.

    WHY THIS KNOB EXISTS. The model charges `F` strictly in proportion to layer
    depth -- `F_REF_MS * m.L / 64` -- on the evidence of a single operator trace
    at one model and one rider count. Testing that needs `L` varied, and the
    only checkpoints on disk (Qwen-0.5B 24L, Llama-8B 32L, Qwen-32B 64L) differ
    in head geometry, hidden size and baseline step time as well as in depth, so
    a non-proportional result could not be attributed to layers.

    This makes the layer count a free variable INSIDE one model at fixed
    geometry: with the limit at N, only layers [0, N) run the rider pass, so the
    per-layer cost is measured against a clean axis.

    NUMERICALLY INVALID by construction. Rider rows above the limit keep the
    output of the standard pass, which read their STALE resident KV, so
    generated tokens are wrong. Timing only -- never a correctness, quality or
    throughput-per-token claim.

    Returns `n_layers` when the knob is off, so callers can compare without a
    branch.

    Resolved once and cached: this is called per layer per step, and an
    `os.environ` read there would be 64 dict lookups per step on a path whose
    whole subject is small per-layer costs.
    """
    cached = _LIMIT_CACHE.get(n_layers)
    if cached is not None:
        return cached
    val = _resolve_layer_limit(n_layers)
    _LIMIT_CACHE[n_layers] = val
    return val


_LIMIT_CACHE: dict[int, int] = {}


def _resolve_layer_limit(n_layers: int) -> int:
    n = _int_env("VLLM_KVPIPE_RUNGD_LAYERS", 0)
    if n <= 0 or n >= n_layers:
        return n_layers
    if os.environ.get("VLLM_KVPIPE_RUNGD_LAYERS_UNSAFE", "0") != "1":
        _warn_once(
            "layers_nounsafe",
            f"*** VLLM_KVPIPE_RUNGD_LAYERS={n} IGNORED: it produces "
            f"numerically invalid output, so it also requires "
            f"VLLM_KVPIPE_RUNGD_LAYERS_UNSAFE=1. Running with all "
            f"{n_layers} layers.")
        return n_layers
    _warn_once(
        "layers_on",
        f"*** RUNGD layer limit = {n}/{n_layers}: only the first {n} layers "
        f"run the rider host path. OUTPUT IS NUMERICALLY INVALID. Timing "
        f"measurement only.")
    return n
