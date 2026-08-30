# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which requests arrive with their KV already on the host tier.

A request in this registry is admitted with NO main-pool blocks at all -- not
even for prefill -- because its KV is delivered from outside rather than
computed into HBM and copied out. That is what lets riders be *added* to a full
resident cohort instead of being bought with resident slots (see
`paper/BLOCKER_PREFILL_RESIDENCY.md`).

Module-level because the scheduler and the model runner both need it and sit on
opposite sides of an interface neither owns; the same reason
`hybrid_attn_ctx` is module-level. Safe for the same reason: the hybrid path
forces eager execution and a single engine core process.

The registry says only "KV for this request is expected to arrive from
elsewhere". It does NOT deliver it -- the runner takes delivery, and refuses
loudly if a request is marked here with nothing behind it, because admitting a
request whose KV is nowhere serves silent garbage.
"""
from __future__ import annotations

_PRELOADED: set[str] = set()


def mark_preloaded(req_id: str) -> None:
    """Declare that this request's KV is (or will be) in the host pool."""
    _PRELOADED.add(req_id)


def is_preloaded(req_id: str) -> bool:
    return req_id in _PRELOADED


def discard(req_id: str) -> None:
    """Forget a request. Must be called when it finishes OR is preempted: a
    preempted request recomputes from scratch, so its pool copy is stale and
    re-admitting it as preloaded would serve the previous attempt's KV."""
    _PRELOADED.discard(req_id)


def clear() -> None:
    _PRELOADED.clear()


def count() -> int:
    return len(_PRELOADED)
