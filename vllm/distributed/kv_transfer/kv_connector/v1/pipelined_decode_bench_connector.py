# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PipelinedDecodeBenchConnector: Decode-only P/D pipelining benchmark connector.

Combines the KV-fill logic from DecodeBenchConnector with the layer-wise
CPU→GPU pipelining from LayerwisePipelinedConnector.  This enables
self-contained performance evaluation of decode-only pipelining without
requiring a real remote prefill node or the P/D transfer stack.

Flow for each new request:
  1. Scheduler claims all prompt tokens as externally available (minus one
     for the first decode token).
  2. On the first worker step the connector fills KV caches on GPU with
     dummy values, stages every layer to CPU pinned memory, then switches
     to pipelined window buffers.
  3. All subsequent decode steps use layer-wise CPU→GPU pipelining exactly
     as in the real decode-only P/D disaggregation path.

Usage:
    vllm serve <model> \\
        --decode-only-pipelining \\
        --kv-k-window 4 \\
        --kv-transfer-config '{
            "kv_connector": "PipelinedDecodeBenchConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "fill_mean": 0.015,
                "fill_std": 0.0
            }
        }' \\
        --enable-prefix-caching false
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
# Window pool (identical to LayerwisePipelinedConnector's pool)
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
class PipelinedDecodeBenchConnectorMetadata(KVConnectorMetadata):
    """Metadata sent from scheduler to worker each step."""

    # Requests that need dummy KV fill on this step.
    # req_id -> (block_ids_per_group, num_tokens_to_fill)
    reqs_to_fill: dict[str, tuple[tuple[list[int], ...], int]] = field(
        default_factory=dict
    )

    # Layer-wise pipelining fields (same as LayerwisePipelinedConnector)
    layers_to_prefetch: list[str] = field(default_factory=list)
    has_work: bool = False
    has_decode_work: bool = False
    active_decode_req_ids: list[str] = field(default_factory=list)
    decode_max_seq_len: int | None = None
    decode_max_block_id: int | None = None


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class PipelinedDecodeBenchConnector(KVConnectorBase_V1):
    """Decode-only P/D pipelining benchmark connector.

    Scheduler side  — behaves like DecodeBenchConnector: claims all prompt
    tokens as externally matched and records block allocations so the worker
    can fill them.

    Worker side — on the *first* step for each request, fills KV on GPU
    with dummy values, stages every layer to CPU pinned memory, then
    binds window buffers.  All subsequent decode steps use the same
    layer-wise CPU→GPU pipelining as LayerwisePipelinedConnector.
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

        # --- config knobs ---
        cache_cfg = vllm_config.cache_config
        self._k_window = int(getattr(cache_cfg, "kv_k_window", 4))
        self._block_size = cache_cfg.block_size

        # Layers list (deterministic order)
        self._layer_names: list[str] = list(
            get_layers_from_vllm_config(
                vllm_config, AttentionLayerBase, None
            ).keys()
        )
        self._orchestrator = PipeliningOrchestrator(
            self._layer_names, self._k_window
        )

        # Fill parameters (from extra config)
        self._fill_mean: float = self._kv_transfer_config.get_from_extra_config(
            "fill_mean", 0.015
        )
        self._fill_std: float = self._kv_transfer_config.get_from_extra_config(
            "fill_std", 0.0
        )

        # --- Scheduler-side state ---
        self._filled_requests: set[str] = set()
        self._pending_fills: dict[
            str, tuple[tuple[list[int], ...], int]
        ] = {}
        # Track all block IDs per active request for correct decode slicing
        self._active_req_blocks: dict[str, set[int]] = {}

        # --- Worker-side state ---
        self._kv_caches: dict[str, torch.Tensor] | None = None
        self._worker_state: LayerwisePipeliningState | None = None
        self._forward_context: ForwardContext | None = None
        self._has_decode_work = False
        self._decode_max_seq_len: int | None = None
        self._decode_max_block_id: int | None = None
        self._window_pools: dict[WindowKey, WindowPool] = {}
        self._original_kv_cache: dict[
            tuple[str, int], torch.Tensor
        ] = {}
        self._original_kv_cache_meta: dict[
            tuple[str, int], tuple[torch.Size, torch.dtype, torch.device]
        ] = {}
        self._freed_original_buffers = False

        # Block inflation (decode-only P/D mode)
        self._num_blocks_pipelined: int | None = None
        if kv_cache_config is not None:
            phys = getattr(kv_cache_config, "num_physical_blocks", None)
            if phys is not None:
                self._num_blocks_pipelined = kv_cache_config.num_blocks
            elif getattr(cache_cfg, "decode_only_pipelining", False):
                self._num_blocks_pipelined = kv_cache_config.num_blocks
                logger.info(
                    "PipelinedDecodeBench (decode-only P/D): block pool = %d",
                    self._num_blocks_pipelined,
                )

        if role == KVConnectorRole.WORKER:
            self._worker_state = LayerwisePipeliningState()

        # Timing stats (collected per step, logged periodically)
        self._step_count = 0
        self._total_fill_time = 0.0
        self._total_stage_time = 0.0

    # ======================================================================
    # Worker-side: KV cache registration
    # ======================================================================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches
        logger.info(
            "PipelinedDecodeBench: registered %d KV cache layers",
            len(kv_caches),
        )

    # ======================================================================
    # Worker-side: window buffer helpers (reuse LayerwisePipelined logic)
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
        # Always free originals (decode-only bench never needs them back)
        if self._has_decode_work:
            self._original_kv_cache.pop(original_key, None)
            self._freed_original_buffers = True
        return window_kv, is_new_binding

    # ======================================================================
    # Worker-side: KV fill + stage to CPU
    # ======================================================================

    def _fill_kv_on_gpu(
        self,
        reqs_to_fill: dict[str, tuple[tuple[list[int], ...], int]],
    ) -> None:
        """Fill allocated KV blocks with dummy values on GPU."""
        if not reqs_to_fill or self._kv_caches is None:
            return

        for _req_id, (block_ids_per_group, num_tokens) in reqs_to_fill.items():
            for block_ids in block_ids_per_group:
                if not block_ids:
                    continue
                for layer_name, kv_cache in self._kv_caches.items():
                    block_ids_tensor = torch.tensor(
                        block_ids, dtype=torch.long, device=kv_cache.device
                    )
                    valid_mask = block_ids_tensor < kv_cache.shape[0]
                    valid_ids = block_ids_tensor[valid_mask]
                    if len(valid_ids) == 0:
                        continue
                    block_shape = kv_cache.shape[1:]
                    if self._fill_std > 0:
                        fill = torch.normal(
                            mean=self._fill_mean,
                            std=self._fill_std,
                            size=(len(valid_ids),) + block_shape,
                            dtype=kv_cache.dtype,
                            device=kv_cache.device,
                        )
                    else:
                        fill = torch.full(
                            (len(valid_ids),) + block_shape,
                            self._fill_mean,
                            dtype=kv_cache.dtype,
                            device=kv_cache.device,
                        )
                    kv_cache[valid_ids] = fill

    def _stage_all_layers_to_cpu(self) -> None:
        """Stage every registered KV layer from GPU to CPU pinned memory.

        This must be called BEFORE binding window buffers so that the
        full KV data (in the original per-layer tensors) is captured.
        """
        if self._kv_caches is None or self._worker_state is None:
            return
        for layer_name, kv_cache in self._kv_caches.items():
            # Ensure CPU buffer at the inflated size
            self._worker_state.ensure_cpu_buffer(layer_name, kv_cache)
            self._worker_state.stage_to_cpu(layer_name, kv_cache)
        # Synchronize to make sure all data is on CPU before we
        # free GPU originals and switch to window buffers.
        self._worker_state.wait_for_saves()

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
        """Return a view covering only the blocks in use.

        During decode, uses the tracked max_block_id (the highest block
        index allocated to ANY active sequence) to ensure all scattered
        blocks are included in the transfer.  Falls back to the
        per-sequence max_seq_len estimate when block tracking is
        unavailable.
        """
        if kv_layer.ndim < 3:
            return kv_layer
        block_dim = 1 if kv_layer.shape[0] == 2 else 0

        # Prefer decode_max_block_id (covers all active sequences).
        if self._decode_max_block_id is not None:
            max_blocks = min(
                self._decode_max_block_id + 1,
                kv_layer.shape[block_dim],
            )
        elif max_seq_len is not None:
            max_blocks = min(
                cdiv(max_seq_len, self._block_size),
                kv_layer.shape[block_dim],
            )
        else:
            return kv_layer

        if max_blocks <= 0 or max_blocks == kv_layer.shape[block_dim]:
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
        if not isinstance(metadata, PipelinedDecodeBenchConnectorMetadata):
            return

        if not metadata.has_work:
            return

        self._forward_context = forward_context
        self._has_decode_work = metadata.has_decode_work
        self._decode_max_seq_len = metadata.decode_max_seq_len
        self._decode_max_block_id = metadata.decode_max_block_id

        # --- Phase 1: Fill + stage on first step for new requests ----------
        if metadata.reqs_to_fill:
            t0 = time.perf_counter()
            self._fill_kv_on_gpu(metadata.reqs_to_fill)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            self._stage_all_layers_to_cpu()
            t2 = time.perf_counter()
            self._total_fill_time += t1 - t0
            self._total_stage_time += t2 - t1
            logger.info(
                "PipelinedDecodeBench: filled KV in %.3fs, staged to CPU "
                "in %.3fs (%d requests)",
                t1 - t0,
                t2 - t1,
                len(metadata.reqs_to_fill),
            )

        # --- Phase 2: Bind window buffers + prefetch -----------------------
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
        if self._worker_state is None or not self._has_decode_work:
            return
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        num_actual_tokens = getattr(attn_metadata, "num_actual_tokens", None)
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
        self._worker_state.ensure_cpu_buffer(layer_name, kv_layer)
        max_seq_len = self._get_max_seq_len(attn_metadata)
        kv_view = self._slice_kv_layer(kv_layer, max_seq_len)
        self._worker_state.stage_to_cpu(layer_name, kv_view)

    def wait_for_save(self) -> None:
        if self._worker_state is None:
            return
        self._worker_state.wait_for_saves()
        self._forward_context = None
        self._has_decode_work = False
        self._decode_max_seq_len = None
        self._decode_max_block_id = None
        self._original_kv_cache.clear()

        self._step_count += 1
        if self._step_count % 100 == 0:
            pinned_gb = (
                self._worker_state.pinned_bytes_allocated / 1e9
                if self._worker_state is not None
                else 0.0
            )
            logger.info(
                "PipelinedDecodeBench step %d: cumulative fill=%.3fs, "
                "stage=%.3fs, pinned CPU memory: %.2f GB",
                self._step_count,
                self._total_fill_time,
                self._total_stage_time,
                pinned_gb,
            )

    # ======================================================================
    # Scheduler-side: claim tokens + record blocks (DecodeBench logic)
    # ======================================================================

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        req_id = request.request_id
        if req_id in self._filled_requests:
            return 0, False
        num_uncomputed = request.num_tokens - num_computed_tokens
        num_to_fill = max(0, num_uncomputed - 1)
        if num_to_fill == 0:
            return 0, False
        return num_to_fill, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens == 0:
            return
        req_id = request.request_id
        block_groups = blocks.get_block_ids()
        num_blocks_to_fill = cdiv(num_external_tokens, self._block_size)
        block_ids_per_group = tuple(
            group[:num_blocks_to_fill] for group in block_groups
        )
        self._pending_fills[req_id] = (block_ids_per_group, num_external_tokens)
        self._filled_requests.add(req_id)

        # Track ALL block IDs for this request (for decode slicing).
        all_ids: set[int] = set()
        for group in block_groups:
            all_ids.update(group)
        self._active_req_blocks[req_id] = all_ids

    # ======================================================================
    # Scheduler-side: build metadata
    # ======================================================================

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        # Collect fills
        reqs_to_fill = dict(self._pending_fills)
        self._pending_fills.clear()

        # --- Track block allocations for correct decode slicing ----------
        # Accumulate newly appended blocks for cached (ongoing) requests.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            new_blks = cached_reqs.new_block_ids[i]
            if new_blks is not None:
                existing = self._active_req_blocks.get(req_id)
                if existing is None:
                    existing = set()
                    self._active_req_blocks[req_id] = existing
                for group in new_blks:
                    existing.update(group)

        # Finished requests: clean up.
        for req_id in scheduler_output.finished_req_ids:
            self._active_req_blocks.pop(req_id, None)

        has_work = scheduler_output.total_num_scheduled_tokens > 0
        decode_only = not scheduler_output.scheduled_new_reqs
        decode_req_ids = set(cached_reqs.req_ids)
        decode_req_ids.update(
            scheduler_output.scheduled_spec_decode_tokens.keys()
        )
        active_decode_req_ids = sorted(decode_req_ids) if decode_only else []
        has_decode_work = len(active_decode_req_ids) > 0

        # When there are fills, the FIRST step has both new reqs (fill) and
        # potentially some cached decode reqs.  We treat any step with fills
        # as a "seed step" where the fill+stage dominates; pipelining
        # (layers_to_prefetch) kicks in when there are pure decode steps.
        # However, even on the seed step we set has_decode_work=True so
        # window buffers are bound after the fill/stage completes.
        if reqs_to_fill:
            has_decode_work = True

        decode_max_seq_len: int | None = None
        if has_decode_work:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            max_len = 0
            for req_id, num_computed in zip(
                cached_reqs.req_ids, cached_reqs.num_computed_tokens
            ):
                max_len = max(
                    max_len,
                    num_computed + num_scheduled_tokens.get(req_id, 0),
                )
            # Also account for newly-filled requests
            for req_id, (_, num_tokens) in reqs_to_fill.items():
                max_len = max(max_len, num_tokens + 1)
            decode_max_seq_len = max_len if max_len > 0 else None

        # Compute max block ID across ALL active requests for correct
        # decode-phase slicing (fixes the per-sequence slice bug).
        decode_max_block_id: int | None = None
        if has_decode_work:
            max_bid = -1
            for blocks in self._active_req_blocks.values():
                if blocks:
                    max_bid = max(max_bid, max(blocks))
            if max_bid >= 0:
                decode_max_block_id = max_bid

        layers_to_prefetch: list[str] = []
        if self._k_window > 0 and has_work and has_decode_work:
            layers_to_prefetch = self._layer_names[: self._k_window]

        return PipelinedDecodeBenchConnectorMetadata(
            reqs_to_fill=reqs_to_fill,
            layers_to_prefetch=layers_to_prefetch,
            has_work=has_work,
            has_decode_work=has_decode_work,
            active_decode_req_ids=active_decode_req_ids,
            decode_max_seq_len=decode_max_seq_len,
            decode_max_block_id=decode_max_block_id,
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._filled_requests.discard(request.request_id)
        self._active_req_blocks.pop(request.request_id, None)
        return False, None
