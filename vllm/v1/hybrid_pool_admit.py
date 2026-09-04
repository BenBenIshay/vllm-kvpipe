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
# req_id -> prompt key, and the set of keys whose real prefill KV the pool
# actually holds. Together these are the local stand-in for a prefill node's
# "I have KV for this prompt" announcement: the scheduler may only admit a
# request block-free once the WORKER has said the KV exists.
_STASH_KEY: dict[str, str] = {}
_STASH_READY: set[str] = set()


def mark_preloaded(req_id: str) -> None:
    """Declare that this request's KV is (or will be) in the host pool."""
    _PRELOADED.add(req_id)


def is_preloaded(req_id: str) -> bool:
    return req_id in _PRELOADED


def discard(req_id: str) -> None:
    """Forget a request. Must be called when it finishes OR is preempted: a
    preempted request recomputes from scratch, so its pool copy is stale and
    re-admitting it as preloaded would serve the previous attempt's KV.

    The stash itself is keyed by PROMPT and deliberately survives: that is the
    whole point of a prefill node's output outliving the prefill request."""
    _PRELOADED.discard(req_id)
    _STASH_KEY.pop(req_id, None)


def clear() -> None:
    _PRELOADED.clear()
    _STASH_KEY.clear()
    _STASH_READY.clear()


def count() -> int:
    return len(_PRELOADED)


# --------------------------------------------------------- prefill stash

def prompt_key(prompt_token_ids) -> str:
    """Stable identity for a prompt, used to hand prefill KV from the request
    that computed it to a later request that decodes it."""
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    h.update(len(prompt_token_ids).to_bytes(8, "little"))
    for t in prompt_token_ids:
        h.update(int(t).to_bytes(4, "little", signed=True))
    return h.hexdigest()


def set_stash_key(req_id: str, key: str) -> None:
    _STASH_KEY[req_id] = key


def get_stash_key(req_id: str) -> str | None:
    return _STASH_KEY.get(req_id)


def mark_stash_ready(key: str) -> None:
    """Called by the worker once the pool really holds this prompt's KV."""
    _STASH_READY.add(key)


def stash_ready(key: str | None) -> bool:
    return key is not None and key in _STASH_READY


def stash_ready_count() -> int:
    return len(_STASH_READY)
