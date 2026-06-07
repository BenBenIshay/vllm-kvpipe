# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Step-level decode transfer planning (B2): prepare once, thin per-layer hooks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext


def kv_pipelining_step_level_enabled() -> bool:
    """Alias for step-plan (B2). Kept for older env name."""
    return kv_pipelining_step_plan_enabled()


def kv_pipelining_step_plan_enabled() -> bool:
    """B2: prepare once at ``start_load_kv``, thin ``save_kv_layer`` issue."""
    return os.environ.get("VLLM_KVPIPE_STEP_PLAN", "1") == "1"


def kv_pipelining_batched_d2h_enabled() -> bool:
    """B1: one gather + one D2H per layer for conc>1 (vs per-sequence copies)."""
    return os.environ.get("VLLM_KVPIPE_BATCHED_D2H", "1") == "1"


def kv_pipelining_fake_transfer_enabled() -> bool:
    """Diagnostic: infinite-bandwidth simulation for conc=1 thesis benches.

    When enabled (``VLLM_KVPIPE_FAKE_TRANSFER=1``):

    - Runs the **same** staging + prefetch path as production (not ``SKIP_DELTA``).
    - Skips **only** the host<->device bus copies (the D2H/H2D ``copy_`` that
      cross PCIe). This models *infinite bandwidth*, not free work.
    - **Keeps** everything else apples-to-apples with the real path: the
      device-to-device gather (``_gather_n_tokens_to_scratch``), the host-side
      scatter (``_scatter_batched_n1_to_cpu_buf``), per-layer hooks, all CUDA
      events (``wait_stream`` / ``wait_event`` / ``record``), and stream order.
    - Compute stream still **issues** ``wait_event(ready)`` on instant-ready
      events (no PCIe latency; still pays CUDA command-issue + host orchestration
      + gather/scatter cost).

    Invalid for correctness or production.
    """
    return os.environ.get("VLLM_KVPIPE_FAKE_TRANSFER", "0") == "1"


def kv_pipelining_wrap_prefetch_enabled() -> bool:
    """Close rolling prefetch: after layer i save, prefetch (i+K) mod L.

    Replaces bootstrap prefetch of layers 0..K-1 at end-of-forward with
    in-forward wrap at layers L-K..L-1 (more overlap before the next token).
    Set ``VLLM_KVPIPE_WRAP_PREFETCH=0`` to restore bootstrap + open-chain prefetch.
    """
    return os.environ.get("VLLM_KVPIPE_WRAP_PREFETCH", "1") == "1"


def build_prefetch_target_indices(
    num_layers: int,
    k_window: int,
    *,
    wrap: bool,
) -> list[int | None]:
    """Per-layer index to prefetch after that layer's decode save (i -> target).

    Wrap mode: always ``(i + k_window) % num_layers`` (layers L-K..L-1 close
    the ring by prefetching 0..K-1 for the next token). Legacy mode: ``i + K``
    only when in range, else ``None`` (requires bootstrap for layers 0..K-1).
    """
    if k_window <= 0 or num_layers <= 0:
        return [None] * num_layers
    if wrap:
        return [(i + k_window) % num_layers for i in range(num_layers)]
    return [
        i + k_window if i + k_window < num_layers else None
        for i in range(num_layers)
    ]


@dataclass
class DecodeStepTransferPlan:
    """Per-forward plan: slot resolution done once; per-layer issue after attention."""

    n_tokens: int = 0
    use_batched: bool = False
    host_slot: int | None = None
    slot_mapping: torch.Tensor | None = None
    num_actual_tokens: int | None = None
    prepared: bool = False
    layers_issued: int = 0
    _layer_names: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.n_tokens = 0
        self.use_batched = False
        self.host_slot = None
        self.slot_mapping = None
        self.num_actual_tokens = None
        self.prepared = False
        self.layers_issued = 0


def extract_attn_metadata_dict(
    forward_context: ForwardContext,
) -> dict[str, Any] | None:
    attn = forward_context.attn_metadata
    if isinstance(attn, dict):
        return attn
    if isinstance(attn, list) and attn:
        first = attn[0]
        if isinstance(first, dict):
            return first
    return None


def find_step_slot_mapping(
    attn_by_layer: dict[str, Any],
    layer_names: list[str],
) -> tuple[torch.Tensor | None, int | None]:
    """First layer metadata carrying slot_mapping for this forward."""
    for name in layer_names:
        meta = attn_by_layer.get(name)
        if meta is None:
            continue
        sm = getattr(meta, "slot_mapping", None)
        if isinstance(sm, torch.Tensor) and sm.numel() > 0:
            nat = getattr(meta, "num_actual_tokens", None)
            return sm, nat
    for meta in attn_by_layer.values():
        sm = getattr(meta, "slot_mapping", None)
        if isinstance(sm, torch.Tensor) and sm.numel() > 0:
            nat = getattr(meta, "num_actual_tokens", None)
            return sm, nat
    return None, None
