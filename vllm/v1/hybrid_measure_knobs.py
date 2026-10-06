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


ATTEND_MODES = ("full", "noattend", "nokernel")


def rider_attend_mode() -> str:
    """How much of the riders' SECOND attention call to run. "full" = all of it.

    WHY THIS KNOB EXISTS. `F` is now the binding constraint on the feature, and
    ~56% of it is the per-layer second attention pass over the rider cohort --
    the pass that exists only because the riders' KV sits in the rolling slot
    rather than the resident pool. Removing it means fusing the two attention
    calls into one (Solution 1 step 3): days of work in this backend, a per-layer
    block-table rebuild on the hot path, and probably its own KV-cache group.

    Nothing measured says that fusing *removes* that cost rather than relocating
    it. The attention arithmetic still has to happen; what a fused call can
    actually delete is the launch and the surrounding gather/scatter. Four
    changes in the last campaign had a mechanism that fired and a benefit that
    did not follow, so this knob prices the prize before anyone pays for it.

    The two ablations bound it from opposite sides:

      "noattend"  skips the gather, the kernel and the scatter. This is the
                  UPPER BOUND on fusion: a fused call cannot beat not doing the
                  work at all.
      "nokernel"  skips only the flash call, keeping the gather and the scatter.
                  So `noattend - nokernel` is the data movement a fused call
                  removes, and `nokernel`'s own saving is kernel + launch, of
                  which only the launch half is recoverable.

    Measuring just "noattend" would repeat the campaign's recurring error:
    pricing a mechanism and assuming the benefit.

    NUMERICALLY INVALID by construction. Rider rows keep the standard pass's
    output, and under the `seqused_masked` optimisation that pass attended a
    single key for exactly those rows, so their tokens are wrong. (One key is
    still valid attention, so this produces garbage, not NaNs.) Timing only --
    never a correctness, quality or throughput-per-token claim.

    Resolved once and cached, for the same reason as `rungd_layer_limit`: this is
    called per layer per step.
    """
    global _ATTEND_MODE
    if _ATTEND_MODE is None:
        _ATTEND_MODE = _resolve_attend_mode()
    return _ATTEND_MODE


_ATTEND_MODE: str | None = None


def _resolve_attend_mode() -> str:
    mode = os.environ.get("VLLM_KVPIPE_RUNGD_ATTEND_MODE", "full").strip()
    if mode in ("", "full"):
        return "full"
    if mode not in ATTEND_MODES:
        # Loud, because a typo here silently measures the unablated path and the
        # arm would be archived as a real result.
        raise RuntimeError(
            f"VLLM_KVPIPE_RUNGD_ATTEND_MODE={mode!r} is not one of "
            f"{ATTEND_MODES}.")
    if os.environ.get("VLLM_KVPIPE_RUNGD_ATTEND_MODE_UNSAFE", "0") != "1":
        _warn_once(
            "attend_nounsafe",
            f"*** VLLM_KVPIPE_RUNGD_ATTEND_MODE={mode} IGNORED: it produces "
            f"numerically invalid output, so it also requires "
            f"VLLM_KVPIPE_RUNGD_ATTEND_MODE_UNSAFE=1. Running the full rider "
            f"attention pass.")
        return "full"
    _warn_once(
        "attend_on",
        f"*** RUNGD rider attend mode = {mode}: "
        + ("the rider gather, attention call and scatter are ALL SKIPPED"
           if mode == "noattend" else
           "the rider attention KERNEL is skipped, gather and scatter kept")
        + ". OUTPUT IS NUMERICALLY INVALID. Timing measurement only.")
    return mode


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
