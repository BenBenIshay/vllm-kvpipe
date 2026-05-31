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
