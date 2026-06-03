# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PrefillStagingPipelinedConnector — K+1 tensor pipelined decode.

Designed for P/D (Prefill/Decode disaggregation) where the decode node
only performs decode, receiving pre-computed KV caches from a remote
prefill node.

For single-GPU evaluation, the connector also supports a local prefill
phase that stages real KV outputs to CPU pinned memory, simulating what
the prefill node would send over the network.

Architecture — K+1 Physical GPU Tensors
----------------------------------------
Instead of allocating one GPU tensor per model layer (28 for Qwen3-0.6B),
we allocate only K+1 tensors (e.g. 5 for K=4).  Layers map to tensors
round-robin: ``layer_i → tensor[i % (K+1)]``.

- **K tensors** form the active compute window.
- **1 spare tensor** serves as a landing zone for the next CPU→GPU load,
  preventing data races between load and compute.

This reduces GPU KV memory by ``num_layers / (K+1)`` and allows the
scheduler to admit proportionally more concurrent decode requests.

Decode Pipeline (steady state)
------------------------------
While computing attention for layer ``i`` on ``tensor[i % (K+1)]``:
  1. Load layer ``i+K`` from CPU → ``tensor[(i+K) % (K+1)]`` (the spare)
  2. After attention, stage the new token's KV delta back to CPU

Since ``(i+K) % (K+1) != i % (K+1)`` (for K≥1), the load target is
always a different physical tensor than the one being computed on.

Usage
-----
    vllm serve <model> \\
        --enable-kv-pipelining \\
        --kv-k-window 4 \\
        --kv-transfer-config '{
            "kv_connector": "PrefillStagingPipelinedConnector",
            "kv_role": "kv_both"
        }' \\
        --enable-prefix-caching false
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

import torch

from vllm.attention.layer import AttentionLayerBase
from vllm.config import get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_pipelining import (
    LayerwisePipeliningState,
    PipeliningOrchestrator,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "")
    return v in ("1", "true", "True", "yes", "YES")


def _pipeline_debug_enabled() -> bool:
    """Set ``VLLM_PIPELINE_DEBUG_LOG=1`` for detailed K+1 connector logs."""
    return _env_truthy("VLLM_PIPELINE_DEBUG_LOG")


def _decode_stage_mode() -> str:
    """Decode staging mode: block (default), full, or delta."""
    mode = os.environ.get("VLLM_PIPELINE_DECODE_STAGE_MODE", "block").strip().lower()
    if mode not in {"block", "full", "delta"}:
        return "block"
    return mode


def _layer_names_aligned_with_kv_cache(
    vllm_config: "VllmConfig",
    kv_cache_config: "KVCacheConfig | None",
) -> list[str]:
    """Layer order for K+1 pipelining must match ``get_kv_cache_config`` / tensor
    round-robin (``kv_cache_groups[*].layer_names``), not only
    ``static_forward_context`` (which can include extra ``AttentionLayerBase``
    modules without their own KV tensors).
    """
    if kv_cache_config is not None and kv_cache_config.kv_cache_groups:
        names: list[str] = []
        for group in kv_cache_config.kv_cache_groups:
            names.extend(group.layer_names)
        if names:
            return names
    return list(
        get_layers_from_vllm_config(
            vllm_config, AttentionLayerBase, None
        ).keys()
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass
class PrefillStagingPipelinedConnectorMetadata(KVConnectorMetadata):
    """Metadata sent from scheduler to worker each step."""
    layers_to_prefetch: list[str] = field(default_factory=list)
    has_work: bool = False
    has_decode_work: bool = False
    active_decode_req_ids: list[str] = field(default_factory=list)
    decode_max_seq_len: int | None = None
    decode_max_block_id: int | None = None
    has_prefill_work: bool = False
    # Disjoint inclusive [lo, hi] spans along the physical block dimension for
    # all active requests. Empty => stage/prefetch the full KV tensor (fallback).
    kv_block_runs: list[tuple[int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class PrefillStagingPipelinedConnector(KVConnectorBase_V1):
    """K+1 tensor pipelined decode connector for P/D disaggregation.

    Scheduler side — transparent: lets prefill happen normally (returns 0
    from get_num_new_matched_tokens).

    Worker side —
      Prefill steps: save_kv_layer stages each layer's real KV output to
          CPU pinned memory (for evaluation on a single GPU).
      Decode steps: start_load_kv prefetches from CPU into the shared
          K+1 GPU tensors.  save_kv_layer stages decode-token deltas
          back to CPU.  Only K+1 layers of KV reside on GPU at any moment.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )

        # --- Config ---
        cache_cfg = vllm_config.cache_config
        self._k_window = int(getattr(cache_cfg, "kv_k_window", 4))
        self._block_size = cache_cfg.block_size

        # Layer list: must match K+1 tensor round-robin in kv_cache_utils
        # (kv_cache_groups layer order), not merely all AttentionLayerBase keys.
        fallback_attn = list(
            get_layers_from_vllm_config(
                vllm_config, AttentionLayerBase, None
            ).keys()
        )
        self._layer_names = _layer_names_aligned_with_kv_cache(
            vllm_config, kv_cache_config
        )
        if (
            fallback_attn
            and self._layer_names
            and fallback_attn != self._layer_names
        ):
            logger.warning(
                "PrefillStagingPipelined: KV cache layer order differs from "
                "AttentionLayerBase list (using KV group order). "
                "attn_layers=%d kv_group_layers=%d",
                len(fallback_attn),
                len(self._layer_names),
            )
        self._layer_index: dict[str, int] = {
            name: idx for idx, name in enumerate(self._layer_names)
        }
        self._num_physical_tensors = self._k_window + 1
        self._orchestrator = PipeliningOrchestrator(
            self._layer_names, self._k_window
        )

        # --- Worker-side state ---
        self._kv_caches: dict[str, torch.Tensor] | None = None
        self._worker_state: LayerwisePipeliningState | None = None
        self._forward_context: ForwardContext | None = None
        self._has_decode_work = False
        self._has_prefill_work = False
        self._decode_max_seq_len: int | None = None
        self._decode_max_block_id: int | None = None
        self._kv_block_runs: list[tuple[int, int]] = []

        # Scheduler-side: track block IDs per request for correct decode slicing
        # (same fix as PipelinedDecodeBenchConnector — real prefill uses pool
        # block IDs that can exceed per-sequence block count).
        self._active_req_blocks: dict[str, set[int]] = {}
        # num_prompt_tokens per request — used to detect chunked-prefill steps
        # (running request with num_computed_tokens < num_prompt_tokens must
        # not be classified as decode-only; see build_connector_meta).
        self._req_prompt_tokens: dict[str, int] = {}

        if role == KVConnectorRole.WORKER:
            self._worker_state = LayerwisePipeliningState()

        # Set in register_kv_caches: True when K+1 layout shares physical tensors
        # across layers. If False (one GPU tensor per layer), decode must not
        # prefetch from CPU — that would overwrite correct on-GPU KV with stale
        # or partial CPU copies.
        self._physical_kv_is_shared: bool = True

        # Timing stats
        self._step_count = 0
        self._prefill_steps = 0
        self._decode_steps = 0
        # Per execute_model forward (incremented in start_load_kv when has_work).
        self._dbg_forward_id: int = 0
        self._decode_stage_mode = _decode_stage_mode()

    # ======================================================================
    # Worker-side: KV cache registration
    # ======================================================================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches
        # Log which layers share physical tensors
        tensor_ids: dict[int, list[str]] = {}
        for name, tensor in kv_caches.items():
            tid = tensor.data_ptr()
            tensor_ids.setdefault(tid, []).append(name)
        self._physical_kv_is_shared = len(tensor_ids) < len(kv_caches)
        logger.info(
            "PrefillStagingPipelined: registered %d layers across %d "
            "physical GPU tensors (K+1=%d, shared_across_layers=%s)",
            len(kv_caches),
            len(tensor_ids),
            self._num_physical_tensors,
            self._physical_kv_is_shared,
        )
        # Pre-allocate pinned CPU buffers at full window size before the first
        # sliced prefill ``stage_to_cpu``.  Otherwise a small buffer from the
        # first slice can interact badly with decode ``stage_delta_to_cpu``
        # (see ``LayerwisePipeliningState.ensure_cpu_buffer``).
        if self._worker_state is not None:
            for name, tensor in kv_caches.items():
                self._worker_state.ensure_cpu_buffer(name, tensor)

    # ======================================================================
    # Worker-side: helpers
    # ======================================================================

    @staticmethod
    def _get_block_dim(kv_layer: torch.Tensor) -> int:
        return 1 if kv_layer.shape[0] == 2 else 0

    def _get_max_seq_len(self, attn_metadata: object | None) -> int | None:
        if self._decode_max_seq_len is not None:
            return self._decode_max_seq_len
        if attn_metadata is None:
            return None
        return getattr(attn_metadata, "max_seq_len", None)

    def _kv_span_debug_stats(
        self, kv_layer: torch.Tensor, max_seq_len: int | None
    ) -> dict[str, Any]:
        """Bounds for VLLM_PIPELINE_DEBUG_LOG (physical block-id span)."""
        if kv_layer.ndim < 3:
            return {"ndim": kv_layer.ndim, "shape": tuple(kv_layer.shape)}
        block_dim = self._get_block_dim(kv_layer)
        full = int(kv_layer.shape[block_dim])
        return {
            "block_dim": block_dim,
            "kv_shape": tuple(kv_layer.shape),
            "full_blocks_along_block_dim": full,
            "kv_block_runs": self._kv_block_runs,
            "decode_max_block_id": self._decode_max_block_id,
            "decode_max_seq_len": self._decode_max_seq_len,
            "max_seq_len_arg": max_seq_len,
            "block_size": self._block_size,
            "data_ptr": kv_layer.data_ptr(),
        }

    @staticmethod
    def _contiguous_runs_from_block_ids(
        ids: Iterable[int],
    ) -> list[tuple[int, int]]:
        """Merge physical block ids into inclusive contiguous spans."""
        u = sorted(set(ids))
        if not u:
            return []
        out: list[tuple[int, int]] = []
        lo = hi = u[0]
        for x in u[1:]:
            if x == hi + 1:
                hi = x
            else:
                out.append((lo, hi))
                lo = hi = x
        out.append((lo, hi))
        return out

    def _stage_physical_kv_to_cpu(
        self, layer_name: str, kv_layer: torch.Tensor
    ) -> None:
        runs = self._kv_block_runs
        if not runs:
            self._worker_state.stage_to_cpu(layer_name, kv_layer)
        elif len(runs) == 1:
            lo, hi = runs[0]
            self._worker_state.stage_to_cpu_block_span(
                layer_name, kv_layer, lo, hi
            )
        else:
            self._worker_state.stage_to_cpu_block_runs(
                layer_name, kv_layer, runs
            )

    def _stage_blocks_from_slot_mapping(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_actual_tokens: int | None,
    ) -> bool:
        """Stage only KV blocks that contain updated decode-token slots."""
        valid_slots = slot_mapping
        if num_actual_tokens is not None:
            valid_slots = valid_slots[:num_actual_tokens]
        valid_slots = valid_slots[valid_slots >= 0]
        if valid_slots.numel() == 0:
            return False

        block_ids = torch.div(valid_slots, self._block_size, rounding_mode="floor")
        # NOTE: this host transfer is used only in block-stage mode to compute
        # compact contiguous runs; correctness-first and still cheaper than
        # full active-span staging at higher concurrency.
        uniq = torch.unique(block_ids).to("cpu")
        runs = self._contiguous_runs_from_block_ids(uniq.tolist())
        if not runs:
            return False
        self._worker_state.stage_to_cpu_block_runs(layer_name, kv_layer, runs)
        return True

    def _physical_tensor_prev_occupant(self, layer_name: str) -> str | None:
        """Layer that last used the same K+1 physical GPU tensor, if any."""
        layer_idx = self._layer_index.get(layer_name, -1)
        prev_idx = layer_idx - self._num_physical_tensors
        if prev_idx < 0:
            return None
        return self._layer_names[prev_idx]

    def _prefetch_physical_kv_to_gpu(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        *,
        gpu_wait_save_from_layer: str | None = None,
    ) -> None:
        runs = self._kv_block_runs
        if not runs:
            self._worker_state.prefetch_to_gpu(
                layer_name,
                kv_layer,
                gpu_wait_save_from_layer=gpu_wait_save_from_layer,
            )
        elif len(runs) == 1:
            lo, hi = runs[0]
            self._worker_state.prefetch_to_gpu_block_span(
                layer_name,
                kv_layer,
                lo,
                hi,
                gpu_wait_save_from_layer=gpu_wait_save_from_layer,
            )
        else:
            self._worker_state.prefetch_to_gpu_block_runs(
                layer_name,
                kv_layer,
                runs,
                gpu_wait_save_from_layer=gpu_wait_save_from_layer,
            )

    def _log_pipeline(self, msg: str) -> None:
        if _pipeline_debug_enabled():
            logger.info("%s", msg)

    def _fmt_slot_mapping(self, sm: torch.Tensor | None) -> str:
        if sm is None or not isinstance(sm, torch.Tensor):
            return "None"
        if sm.numel() == 0:
            return "empty"
        mn = int(sm.min().item())
        mx = int(sm.max().item())
        return f"len={sm.numel()} min={mn} max={mx} dtype={sm.dtype}"

    # ======================================================================
    # Worker-side: main hooks
    # ======================================================================

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        if self._worker_state is None:
            return
        if not self.has_connector_metadata():
            return
        metadata = self._get_connector_metadata()
        if not isinstance(
            metadata, PrefillStagingPipelinedConnectorMetadata
        ):
            return
        if not metadata.has_work:
            return

        self._dbg_forward_id += 1
        self._forward_context = forward_context
        self._has_decode_work = metadata.has_decode_work
        self._has_prefill_work = metadata.has_prefill_work
        self._decode_max_seq_len = metadata.decode_max_seq_len
        self._decode_max_block_id = metadata.decode_max_block_id
        self._kv_block_runs = list(metadata.kv_block_runs)

        self._log_pipeline(
            "[pipeline-debug] start_load_kv "
            f"fwd={self._dbg_forward_id} "
            f"has_prefill={self._has_prefill_work} has_decode={self._has_decode_work} "
            f"decode_max_block_id={metadata.decode_max_block_id} "
            f"decode_max_seq_len={metadata.decode_max_seq_len} "
            f"kv_block_runs={metadata.kv_block_runs!r} "
            f"layers_to_prefetch={metadata.layers_to_prefetch!r} "
            f"physical_kv_shared={self._physical_kv_is_shared}"
        )

        # ------------------------------------------------------------------
        # Pure prefill (no decode-phase requests): no CPU→GPU prefetch.
        # Mixed prefill+decode: decode requests still need prefetch below.
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Decode step: prefetch the first K layers from CPU → GPU.
        # ------------------------------------------------------------------
        if not self._has_decode_work or self._kv_caches is None:
            return
        if not self._physical_kv_is_shared:
            # One dedicated GPU tensor per layer — KV is already on GPU; never
            # overwrite it from CPU during decode.
            return

        for layer_name in metadata.layers_to_prefetch:
            kv_layer = self._kv_caches[layer_name]
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata.get(layer_name)
            max_seq_len = self._get_max_seq_len(attn_metadata)
            st = self._kv_span_debug_stats(kv_layer, max_seq_len)
            li = self._layer_index.get(layer_name, -1)
            prev_occ = self._physical_tensor_prev_occupant(layer_name)
            self._log_pipeline(
                "[pipeline-debug] prefetch_batch(start_load_kv) "
                f"fwd={self._dbg_forward_id} layer={layer_name!r} idx={li} "
                f"wait_save_prev_occ={prev_occ!r} {st}"
            )
            self._prefetch_physical_kv_to_gpu(
                layer_name,
                kv_layer,
                gpu_wait_save_from_layer=prev_occ,
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self._worker_state is None:
            return

        # ------------------------------------------------------------------
        # Prefill step: ensure any pending GPU→CPU copy from the previous
        # occupant of this physical tensor has completed before the model
        # writes new KV data into it.
        # ------------------------------------------------------------------
        if self._has_prefill_work:
            if self._physical_kv_is_shared:
                layer_idx = self._layer_index.get(layer_name, -1)
                prev_idx = layer_idx - self._num_physical_tensors
                waited_prev: str | None = None
                if prev_idx >= 0:
                    prev_layer = self._layer_names[prev_idx]
                    self._worker_state.wait_for_layer_save(prev_layer)
                    waited_prev = prev_layer
                self._log_pipeline(
                    "[pipeline-debug] wait_load prefill "
                    f"fwd={self._dbg_forward_id} layer={layer_name!r} idx={layer_idx} "
                    f"wait_save_prev={waited_prev!r}"
                )
            if not self._has_decode_work:
                return

        # ------------------------------------------------------------------
        # Decode step: wait for this layer's CPU→GPU prefetch,
        # then trigger prefetching the next layer(s) in the pipeline.
        # ------------------------------------------------------------------
        if not self._has_decode_work:
            return
        if not self._physical_kv_is_shared:
            return

        cur_idx = self._layer_index.get(layer_name, -1)
        self._log_pipeline(
            "[pipeline-debug] wait_load decode before_wait_layer "
            f"fwd={self._dbg_forward_id} layer={layer_name!r} idx={cur_idx}"
        )

        self._worker_state.wait_for_layer(layer_name)

        if self._forward_context is None or self._kv_caches is None:
            return
        planned = self._orchestrator.plan_prefetch(layer_name)
        self._log_pipeline(
            "[pipeline-debug] wait_load decode plan_prefetch "
            f"fwd={self._dbg_forward_id} after_layer={layer_name!r} -> {planned!r}"
        )
        for next_layer in planned:
            next_idx = self._layer_index.get(next_layer, -1)
            prev_occ_name = self._physical_tensor_prev_occupant(next_layer)

            kv_layer = self._kv_caches[next_layer]
            attn_metadata = self._forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata.get(next_layer)
            max_seq_len = self._get_max_seq_len(attn_metadata)
            st = self._kv_span_debug_stats(kv_layer, max_seq_len)
            self._log_pipeline(
                "[pipeline-debug] prefetch_chain(wait_load) "
                f"fwd={self._dbg_forward_id} next_layer={next_layer!r} idx={next_idx} "
                f"wait_save_prev_occ={prev_occ_name!r} {st}"
            )
            self._prefetch_physical_kv_to_gpu(
                next_layer,
                kv_layer,
                gpu_wait_save_from_layer=prev_occ_name,
            )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        if self._worker_state is None:
            return
        if not self._has_decode_work and not self._has_prefill_work:
            return

        # ------------------------------------------------------------------
        # Prefill step: stage the layer KV to CPU (physical block-id span).
        # ------------------------------------------------------------------
        if self._has_prefill_work:
            max_seq_len = self._get_max_seq_len(attn_metadata)
            st = self._kv_span_debug_stats(kv_layer, max_seq_len)
            self._log_pipeline(
                "[pipeline-debug] save_kv prefill "
                f"fwd={self._dbg_forward_id} layer={layer_name!r} {st}"
            )
            self._stage_physical_kv_to_cpu(layer_name, kv_layer)
            return

        # ------------------------------------------------------------------
        # Pure decode step: stage only the delta (new decode tokens).
        # The full KV is already on CPU from the prefill phase; we only
        # need to update it with the newly generated token's KV.
        # ------------------------------------------------------------------
        if not self._physical_kv_is_shared:
            return

        if self._decode_stage_mode == "full":
            st = self._kv_span_debug_stats(
                kv_layer, self._get_max_seq_len(attn_metadata)
            )
            self._log_pipeline(
                "[pipeline-debug] save_kv decode FULL_STAGE_MODE "
                f"fwd={self._dbg_forward_id} layer={layer_name!r} {st}"
            )
            self._stage_physical_kv_to_cpu(layer_name, kv_layer)
            return

        # Debug: full tensor staging every decode step (slow, high PCIe BW).
        # If outputs match baseline with this set, the bug is likely in
        # stage_delta_to_cpu / _apply_delta_updates scatter.
        if _env_truthy("VLLM_PIPELINE_FULL_DECODE_STAGE"):
            max_seq_len = self._get_max_seq_len(attn_metadata)
            st = self._kv_span_debug_stats(kv_layer, max_seq_len)
            self._log_pipeline(
                "[pipeline-debug] save_kv decode FULL_STAGE "
                f"fwd={self._dbg_forward_id} layer={layer_name!r} {st}"
            )
            self._stage_physical_kv_to_cpu(layer_name, kv_layer)
            return

        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        num_actual_tokens = getattr(
            attn_metadata, "num_actual_tokens", None
        )
        if (
            self._decode_stage_mode == "block"
            and self._worker_state.has_layer(layer_name)
            and isinstance(slot_mapping, torch.Tensor)
        ):
            staged = self._stage_blocks_from_slot_mapping(
                layer_name=layer_name,
                kv_layer=kv_layer,
                slot_mapping=slot_mapping,
                num_actual_tokens=num_actual_tokens,
            )
            if staged:
                st = self._kv_span_debug_stats(
                    kv_layer, self._get_max_seq_len(attn_metadata)
                )
                self._log_pipeline(
                    "[pipeline-debug] save_kv decode BLOCK_STAGE_MODE "
                    f"fwd={self._dbg_forward_id} layer={layer_name!r} "
                    f"slot_mapping={self._fmt_slot_mapping(slot_mapping)} "
                    f"num_actual_tokens={num_actual_tokens!r} {st}"
                )
                return

        if self._decode_stage_mode == "delta" and self._worker_state.has_layer(
            layer_name
        ) and isinstance(slot_mapping, torch.Tensor):
            st = self._kv_span_debug_stats(
                kv_layer, self._get_max_seq_len(attn_metadata)
            )
            self._log_pipeline(
                "[pipeline-debug] save_kv decode DELTA_STAGE_MODE "
                f"fwd={self._dbg_forward_id} layer={layer_name!r} "
                f"slot_mapping={self._fmt_slot_mapping(slot_mapping)} "
                f"num_actual_tokens={num_actual_tokens!r} {st}"
            )
            self._worker_state.stage_delta_to_cpu(
                layer_name,
                kv_layer,
                slot_mapping,
                num_actual_tokens=num_actual_tokens,
            )
            return

        if self._worker_state.has_layer(layer_name) and isinstance(
            slot_mapping, torch.Tensor
        ):
            # block mode can fail to derive non-empty runs (rare edge cases).
            # Fall through to full span staging for correctness.
            pass
        # Fall back to full staging if this is the first time we see the
        # layer in decode mode.
        max_seq_len = self._get_max_seq_len(attn_metadata)
        st = self._kv_span_debug_stats(kv_layer, max_seq_len)
        self._log_pipeline(
            "[pipeline-debug] save_kv decode FALLBACK_FULL "
            f"fwd={self._dbg_forward_id} layer={layer_name!r} {st}"
        )
        self._stage_physical_kv_to_cpu(layer_name, kv_layer)

    def wait_for_save(self) -> None:
        if self._worker_state is None:
            return
        # Decode-only: defer global barriers; prefetch waits per-layer.
        # Prefill / mixed: must drain staging before the next forward.
        decode_only_step = self._has_decode_work and not self._has_prefill_work
        self._worker_state.wait_for_saves(deferred=decode_only_step)

        if self._has_prefill_work:
            self._prefill_steps += 1
        if self._has_decode_work:
            self._decode_steps += 1

        self._forward_context = None
        self._has_decode_work = False
        self._has_prefill_work = False
        self._decode_max_seq_len = None
        self._decode_max_block_id = None
        self._kv_block_runs = []

        self._step_count += 1
        if self._step_count % 100 == 0:
            logger.info(
                "PrefillStagingPipelined step %d: "
                "%d prefill steps, %d decode steps",
                self._step_count,
                self._prefill_steps,
                self._decode_steps,
            )

    # ======================================================================
    # Scheduler-side: transparent (no external token claims)
    # ======================================================================

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Track block IDs for this request so decode can slice the full range."""
        self._req_prompt_tokens[request.request_id] = request.num_prompt_tokens
        block_groups = blocks.get_block_ids()
        all_ids: set[int] = set()
        for group in block_groups:
            all_ids.update(group)
        # Allocation callbacks can arrive multiple times per request lifetime.
        # Merge rather than overwrite so we don't lose older prompt/decode
        # blocks that are still live in the pool.
        existing = self._active_req_blocks.get(request.request_id)
        if existing is None:
            self._active_req_blocks[request.request_id] = all_ids
        else:
            existing.update(all_ids)

    # ======================================================================
    # Scheduler-side: build metadata
    # ======================================================================

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        has_work = scheduler_output.total_num_scheduled_tokens > 0
        # Prefill work: brand-new requests from the waiting queue, OR a
        # running request still processing prompt tokens (chunked prefill).
        # The old check (scheduled_new_reqs only) misclassified chunked
        # prefill as decode-only, running pipelined prefetch during prefill
        # and corrupting KV.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        any_chunked_prefill = False
        for i, req_id in enumerate(cached_reqs.req_ids):
            n_prompt = self._req_prompt_tokens.get(req_id)
            if n_prompt is None:
                continue
            if cached_reqs.num_computed_tokens[i] < n_prompt:
                any_chunked_prefill = True
                break
        has_prefill = (
            len(scheduler_output.scheduled_new_reqs) > 0 or any_chunked_prefill
        )

        # Decode work: any request has finished its prompt and is generating
        # (decode phase).  This must be true even when the same batch still
        # contains chunked-prefill requests — otherwise decode_only stays false,
        # has_decode_work becomes false, and K+1 prefetch / decode hooks are
        # skipped while other sequences are still prefilling (conc>1 bug).
        active_decode_req_ids: list[str] = []
        for i, req_id in enumerate(cached_reqs.req_ids):
            n_prompt = self._req_prompt_tokens.get(req_id)
            if n_prompt is None:
                continue
            if cached_reqs.num_computed_tokens[i] >= n_prompt:
                active_decode_req_ids.append(req_id)
        active_decode_req_ids.sort()
        has_decode_work = len(active_decode_req_ids) > 0

        # Accumulate new block IDs for decode requests (each step can add blocks).
        for i, req_id in enumerate(cached_reqs.req_ids):
            new_blks = cached_reqs.new_block_ids[i]
            if new_blks is not None:
                # For resumed requests, scheduler semantics define new_block_ids
                # as a full replacement of block IDs (not an append delta).
                if req_id in cached_reqs.resumed_req_ids:
                    existing = set()
                    self._active_req_blocks[req_id] = existing
                else:
                    existing = self._active_req_blocks.get(req_id)
                    if existing is None:
                        existing = set()
                        self._active_req_blocks[req_id] = existing
                for group in new_blks:
                    existing.update(group)
        for req_id in scheduler_output.finished_req_ids:
            self._active_req_blocks.pop(req_id, None)
            self._req_prompt_tokens.pop(req_id, None)

        # Compute decode max_seq_len and decode_max_block_id for slicing
        decode_max_seq_len: int | None = None
        decode_max_block_id: int | None = None
        if has_decode_work:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            max_len = 0
            for req_id, num_computed in zip(
                cached_reqs.req_ids,
                cached_reqs.num_computed_tokens,
            ):
                max_len = max(
                    max_len,
                    num_computed + num_scheduled_tokens.get(req_id, 0),
                )
            decode_max_seq_len = max_len if max_len > 0 else None
            max_bid = -1
            for blocks in self._active_req_blocks.values():
                if blocks:
                    max_bid = max(max_bid, max(blocks))
            if max_bid >= 0:
                decode_max_block_id = max_bid

        # Layers to prefetch for decode (first K layers)
        layers_to_prefetch: list[str] = []
        if self._k_window > 0 and has_work and has_decode_work:
            layers_to_prefetch = self._layer_names[: self._k_window]

        kv_block_runs: list[tuple[int, int]] = []
        merged_block_ids: list[int] = []
        for blocks in self._active_req_blocks.values():
            merged_block_ids.extend(blocks)
        if merged_block_ids:
            kv_block_runs = self._contiguous_runs_from_block_ids(
                merged_block_ids
            )

        return PrefillStagingPipelinedConnectorMetadata(
            layers_to_prefetch=layers_to_prefetch,
            has_work=has_work,
            has_decode_work=has_decode_work,
            active_decode_req_ids=active_decode_req_ids,
            decode_max_seq_len=decode_max_seq_len,
            decode_max_block_id=decode_max_block_id,
            has_prefill_work=has_prefill,
            kv_block_runs=kv_block_runs,
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._active_req_blocks.pop(request.request_id, None)
        self._req_prompt_tokens.pop(request.request_id, None)
        return False, None
