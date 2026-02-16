# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PrefillStagingPipelinedConnector: Real-prefill + layer-wise pipelined decode.

Unlike PipelinedDecodeBenchConnector (which fills KV with dummy values),
this connector lets the model run **real prefill** computation.  After each
attention layer finishes prefill, the connector asynchronously stages the
real KV output from GPU to CPU pinned memory.  On subsequent decode steps
it uses the same layer-wise CPU→GPU pipelining as
LayerwisePipelinedConnector.

This enables academically rigorous evaluation of layer-wise KV-cache
pipelining on a single GPU:

  - The prefill output is *real* (not synthetic), so output quality and
    correctness can be validated against a baseline.
  - The decode path exercises the exact same CPU→GPU streaming code that
    a production P/D disaggregation deployment would use.
  - Standard datasets (LongBench, ShareGPT, random) and the vLLM
    benchmark client (`vllm bench serve`) work out of the box.

Two-phase workflow on a single H100
------------------------------------
Phase 1 — Prefill + Stage:
    Send prompts to the server.  The model runs real attention.  After
    each layer's forward pass, ``save_kv_layer`` stages the layer's KV
    from GPU to CPU pinned memory using a CUDA copy-stream.

Phase 2 — Pipelined Decode:
    On the next scheduler step (pure decode), ``start_load_kv`` binds
    K-window GPU buffers and prefetches from CPU.  The model runs
    attention against these window buffers.  ``save_kv_layer`` stages
    new decode-token KV back to CPU.  Only K layers of KV reside on GPU
    at any moment.

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

    # Then use any standard benchmark client:
    vllm bench serve --base-url http://127.0.0.1:8000 \\
        --model <model> --dataset-name random \\
        --random-input-len 512 --random-output-len 128 \\
        --max-concurrency 16
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.attention.layer import AttentionLayerBase
from vllm.config import get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
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

WindowKey = tuple[torch.Size, torch.dtype, torch.device]


# ---------------------------------------------------------------------------
# Window pool (shared with LayerwisePipelinedConnector)
# ---------------------------------------------------------------------------

@dataclass
class WindowPool:
    buffers: list[torch.Tensor]
    layer_to_slot: dict[str, int] = field(default_factory=dict)
    lru: list[str] = field(default_factory=list)

    def bind_layer(self, layer_name: str) -> tuple[torch.Tensor, bool]:
        if layer_name in self.layer_to_slot:
            self._touch(layer_name)
            return self.buffers[self.layer_to_slot[layer_name]], False
        if len(self.layer_to_slot) < len(self.buffers):
            slot = len(self.layer_to_slot)
        else:
            evicted_layer = self.lru.pop(0)
            slot = self.layer_to_slot.pop(evicted_layer)
        self.layer_to_slot[layer_name] = slot
        self.lru.append(layer_name)
        return self.buffers[slot], True

    def _touch(self, layer_name: str) -> None:
        if layer_name in self.lru:
            self.lru.remove(layer_name)
        self.lru.append(layer_name)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass
class PrefillStagingPipelinedConnectorMetadata(KVConnectorMetadata):
    """Metadata sent from scheduler to worker each step."""

    # Pipelining decode fields (identical to LayerwisePipelinedConnector)
    layers_to_prefetch: list[str] = field(default_factory=list)
    has_work: bool = False
    has_decode_work: bool = False
    active_decode_req_ids: list[str] = field(default_factory=list)
    decode_max_seq_len: int | None = None

    # Prefill staging fields
    has_prefill_work: bool = False


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class PrefillStagingPipelinedConnector(KVConnectorBase_V1):
    """Real-prefill + layer-wise pipelined decode connector.

    Scheduler side — transparent: lets prefill happen normally (returns 0
    from get_num_new_matched_tokens) and tracks which steps have prefill
    vs decode work.

    Worker side —
      Prefill steps: save_kv_layer stages each layer's real KV output to
          CPU pinned memory.  The model uses original GPU KV tensors.
      Decode steps: start_load_kv binds window buffers and prefetches
          from CPU.  save_kv_layer stages decode-token deltas back to CPU.
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

        # --- Config knobs ---
        cache_cfg = vllm_config.cache_config
        self._k_window = int(getattr(cache_cfg, "kv_k_window", 4))
        self._block_size = cache_cfg.block_size
        self._free_original_buffers = bool(
            getattr(cache_cfg, "free_original_kv_cache", False)
            or self._kv_transfer_config.get_from_extra_config(
                "free_original_kv_cache", False
            )
        )

        # Layer list (deterministic order)
        self._layer_names: list[str] = list(
            get_layers_from_vllm_config(
                vllm_config, AttentionLayerBase, None
            ).keys()
        )
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
        self._window_pools: dict[WindowKey, WindowPool] = {}
        self._original_kv_cache: dict[
            tuple[str, int], torch.Tensor
        ] = {}
        self._original_kv_cache_meta: dict[
            tuple[str, int], tuple[torch.Size, torch.dtype, torch.device]
        ] = {}
        self._freed_original_buffers = False

        # Pipelining-aware block inflation (optional).
        self._num_blocks_pipelined: int | None = None
        if kv_cache_config is not None:
            phys = getattr(kv_cache_config, "num_physical_blocks", None)
            if phys is not None:
                self._num_blocks_pipelined = kv_cache_config.num_blocks
                logger.info(
                    "PrefillStagingPipelined: block pool inflated to %d "
                    "(physical per-layer tensors: %d blocks)",
                    self._num_blocks_pipelined,
                    phys,
                )

        if role == KVConnectorRole.WORKER:
            self._worker_state = LayerwisePipeliningState()

        # Timing stats
        self._step_count = 0
        self._total_prefill_stage_time = 0.0
        self._total_decode_stage_time = 0.0
        self._prefill_steps = 0
        self._decode_steps = 0

    # ======================================================================
    # Worker-side: KV cache registration
    # ======================================================================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches
        logger.info(
            "PrefillStagingPipelined: registered %d KV cache layers "
            "(%d model layers)",
            len(kv_caches),
            len(self._layer_names),
        )

    # ======================================================================
    # Worker-side: window buffer helpers
    # ======================================================================

    @staticmethod
    def _get_block_dim(kv_layer: torch.Tensor) -> int:
        return 1 if kv_layer.shape[0] == 2 else 0

    def _inflated_shape(self, kv_layer: torch.Tensor) -> torch.Size:
        if self._num_blocks_pipelined is None:
            return kv_layer.shape
        block_dim = self._get_block_dim(kv_layer)
        if self._num_blocks_pipelined <= kv_layer.shape[block_dim]:
            return kv_layer.shape
        new_shape = list(kv_layer.shape)
        new_shape[block_dim] = self._num_blocks_pipelined
        return torch.Size(new_shape)

    def _make_inflated_buffer(self, kv_layer: torch.Tensor) -> torch.Tensor:
        if self._num_blocks_pipelined is None:
            return torch.empty_like(kv_layer)
        block_dim = self._get_block_dim(kv_layer)
        if self._num_blocks_pipelined <= kv_layer.shape[block_dim]:
            return torch.empty_like(kv_layer)
        new_shape = list(kv_layer.shape)
        new_shape[block_dim] = self._num_blocks_pipelined
        return torch.empty(
            new_shape, dtype=kv_layer.dtype, device=kv_layer.device
        )

    def _get_window_buffer(
        self, layer_name: str, kv_layer: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        inflated_shape = self._inflated_shape(kv_layer)
        key = (inflated_shape, kv_layer.dtype, kv_layer.device)
        pool = self._window_pools.get(key)
        if pool is None:
            buffers = [
                self._make_inflated_buffer(kv_layer)
                for _ in range(max(self._k_window, 1))
            ]
            pool = WindowPool(buffers=buffers)
            self._window_pools[key] = pool
        return pool.bind_layer(layer_name)

    def _bind_layer_window(
        self, layer_name: str, forward_context: "ForwardContext"
    ) -> tuple[torch.Tensor, bool]:
        layer = forward_context.no_compile_layers.get(layer_name)
        if layer is None:
            raise ValueError(
                f"Layer {layer_name} not found in forward context"
            )
        ve = forward_context.virtual_engine
        kv_layer = layer.kv_cache[ve]
        original_key = (layer_name, ve)
        if original_key not in self._original_kv_cache:
            self._original_kv_cache[original_key] = kv_layer
            self._original_kv_cache_meta[original_key] = (
                kv_layer.shape,
                kv_layer.dtype,
                kv_layer.device,
            )
        window_kv, is_new_binding = self._get_window_buffer(
            layer_name, kv_layer
        )
        layer.kv_cache[ve] = window_kv
        # Free originals during decode to reclaim GPU memory
        if self._free_original_buffers and self._has_decode_work:
            self._original_kv_cache.pop(original_key, None)
            self._freed_original_buffers = True
        return window_kv, is_new_binding

    def _restore_original_buffers(
        self, forward_context: "ForwardContext | None" = None
    ) -> None:
        """Restore original KV cache tensors for prefill steps."""
        forward_context = forward_context or self._forward_context
        if forward_context is None:
            return
        for (layer_name, ve), kv_layer in list(
            self._original_kv_cache.items()
        ):
            layer = forward_context.no_compile_layers.get(layer_name)
            if layer is not None:
                layer.kv_cache[ve] = kv_layer
        if self._free_original_buffers:
            for (layer_name, ve), meta in self._original_kv_cache_meta.items():
                layer = forward_context.no_compile_layers.get(layer_name)
                if layer is None:
                    continue
                if (layer_name, ve) not in self._original_kv_cache:
                    shape, dtype, device = meta
                    layer.kv_cache[ve] = torch.empty(
                        shape, dtype=dtype, device=device
                    )
        self._original_kv_cache.clear()

    # ======================================================================
    # Worker-side: seq-len slicing helper
    # ======================================================================

    def _get_max_seq_len(self, attn_metadata: object | None) -> int | None:
        if self._decode_max_seq_len is not None:
            return self._decode_max_seq_len
        if attn_metadata is None:
            return None
        return getattr(attn_metadata, "max_seq_len", None)

    def _slice_kv_layer(
        self, kv_layer: torch.Tensor, max_seq_len: int | None
    ) -> torch.Tensor:
        if max_seq_len is None or kv_layer.ndim < 3:
            return kv_layer
        block_size = self._block_size
        max_blocks = cdiv(max_seq_len, block_size)
        block_dim = 1 if kv_layer.shape[0] == 2 else 0
        max_blocks = min(max_blocks, kv_layer.shape[block_dim])
        if max_blocks == kv_layer.shape[block_dim]:
            return kv_layer
        slicer = [slice(None)] * kv_layer.ndim
        slicer[block_dim] = slice(0, max_blocks)
        return kv_layer[tuple(slicer)]

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

        self._forward_context = forward_context
        self._has_decode_work = metadata.has_decode_work
        self._has_prefill_work = metadata.has_prefill_work
        self._decode_max_seq_len = metadata.decode_max_seq_len

        # ------------------------------------------------------------------
        # Prefill step: keep original KV tensors on GPU.
        # The model will run real attention and populate them.
        # save_kv_layer (called after each layer) will stage to CPU.
        # ------------------------------------------------------------------
        if metadata.has_prefill_work and not metadata.has_decode_work:
            # Pure prefill step — ensure original buffers are active.
            self._restore_original_buffers(forward_context)
            return

        # ------------------------------------------------------------------
        # Mixed step (prefill + decode): use original buffers.
        # We'll stage everything during save_kv_layer.
        # ------------------------------------------------------------------
        if metadata.has_prefill_work and metadata.has_decode_work:
            # Keep original buffers — prefill needs them.
            # save_kv_layer will stage full layers to CPU.
            return

        # ------------------------------------------------------------------
        # Pure decode step: bind window buffers + prefetch from CPU.
        # ------------------------------------------------------------------
        if not metadata.has_decode_work:
            return

        for layer_name in metadata.layers_to_prefetch:
            kv_layer, is_new_binding = self._bind_layer_window(
                layer_name, forward_context
            )
            if not is_new_binding:
                continue
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata.get(layer_name)
            max_seq_len = self._get_max_seq_len(attn_metadata)
            kv_view = self._slice_kv_layer(kv_layer, max_seq_len)
            self._worker_state.prefetch_to_gpu(layer_name, kv_view)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self._worker_state is None:
            return
        # During prefill steps, there's nothing to wait for — the model
        # computes KV on GPU natively.
        if self._has_prefill_work and not self._has_decode_work:
            return
        if self._has_prefill_work:
            # Mixed step — no pipelining, just staging after attention.
            return
        # Pure decode step — wait for prefetch and trigger next prefetch.
        self._worker_state.wait_for_layer(layer_name)
        if self._forward_context is None:
            return
        for next_layer in self._orchestrator.plan_prefetch(layer_name):
            kv_layer, is_new = self._bind_layer_window(
                next_layer, self._forward_context
            )
            if not is_new:
                continue
            attn_metadata = self._forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata.get(next_layer)
            max_seq_len = self._get_max_seq_len(attn_metadata)
            kv_view = self._slice_kv_layer(kv_layer, max_seq_len)
            self._worker_state.prefetch_to_gpu(next_layer, kv_view)

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
        # Prefill (or mixed) step: stage the FULL layer KV to CPU.
        # This captures the real attention output from the model.
        # ------------------------------------------------------------------
        if self._has_prefill_work:
            self._worker_state.ensure_cpu_buffer(layer_name, kv_layer)
            max_seq_len = self._get_max_seq_len(attn_metadata)
            kv_view = self._slice_kv_layer(kv_layer, max_seq_len)
            self._worker_state.stage_to_cpu(layer_name, kv_view)
            return

        # ------------------------------------------------------------------
        # Pure decode step: stage only the delta (new decode tokens).
        # ------------------------------------------------------------------
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        num_actual_tokens = getattr(
            attn_metadata, "num_actual_tokens", None
        )
        if self._worker_state.has_layer(layer_name) and isinstance(
            slot_mapping, torch.Tensor
        ):
            self._worker_state.stage_delta_to_cpu(
                layer_name,
                kv_layer,
                slot_mapping,
                num_actual_tokens=num_actual_tokens,
            )
            return
        # Fall back to full staging if this is the first time we see the
        # layer in decode mode.
        self._worker_state.ensure_cpu_buffer(layer_name, kv_layer)
        max_seq_len = self._get_max_seq_len(attn_metadata)
        kv_view = self._slice_kv_layer(kv_layer, max_seq_len)
        self._worker_state.stage_to_cpu(layer_name, kv_view)

    def wait_for_save(self) -> None:
        if self._worker_state is None:
            return
        self._worker_state.wait_for_saves()

        # Track timing stats
        if self._has_prefill_work:
            self._prefill_steps += 1
        if self._has_decode_work:
            self._decode_steps += 1

        self._forward_context = None
        self._has_decode_work = False
        self._has_prefill_work = False
        self._decode_max_seq_len = None
        self._original_kv_cache.clear()

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
        # Return 0 — let prefill happen normally through the model.
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        # No external tokens to handle.
        return

    # ======================================================================
    # Scheduler-side: build metadata
    # ======================================================================

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        has_work = scheduler_output.total_num_scheduled_tokens > 0
        has_prefill = len(scheduler_output.scheduled_new_reqs) > 0

        # Identify decode requests
        decode_req_ids = set(
            scheduler_output.scheduled_cached_reqs.req_ids
        )
        decode_req_ids.update(
            scheduler_output.scheduled_spec_decode_tokens.keys()
        )
        # Only count as "decode work" if there's no prefill mixed in.
        # During mixed steps we keep original GPU tensors for prefill.
        decode_only = not has_prefill
        active_decode_req_ids = (
            sorted(decode_req_ids) if decode_only else []
        )
        has_decode_work = len(active_decode_req_ids) > 0

        # Compute decode max_seq_len for CPU→GPU slice optimization
        decode_max_seq_len: int | None = None
        if has_decode_work:
            cached_reqs = scheduler_output.scheduled_cached_reqs
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

        # Layers to prefetch for decode
        layers_to_prefetch: list[str] = []
        if (
            self._k_window > 0
            and has_work
            and has_decode_work
        ):
            layers_to_prefetch = self._layer_names[: self._k_window]

        return PrefillStagingPipelinedConnectorMetadata(
            layers_to_prefetch=layers_to_prefetch,
            has_work=has_work,
            has_decode_work=has_decode_work,
            active_decode_req_ids=active_decode_req_ids,
            decode_max_seq_len=decode_max_seq_len,
            has_prefill_work=has_prefill,
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None
