from __future__ import annotations

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
from vllm.v1.kv_cache_pipelining import LayerwisePipeliningState, PipeliningOrchestrator
from vllm.utils.math_utils import cdiv

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


@dataclass
class LayerwisePipelinedConnectorMetadata(KVConnectorMetadata):
    layers_to_prefetch: list[str] = field(default_factory=list)
    has_work: bool = False
    active_req_ids: list[str] = field(default_factory=list)
    has_decode_work: bool = False
    active_decode_req_ids: list[str] = field(default_factory=list)
    decode_max_seq_len: int | None = None


class LayerwisePipelinedConnector(KVConnectorBase_V1):
    """Decode-only, layer-wise KV pipelining connector."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig" | None = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        self._enabled = bool(
            getattr(vllm_config.cache_config, "enable_kv_pipelining", False)
        )
        self._k_window = int(getattr(vllm_config.cache_config, "kv_k_window", 0))
        self._layer_names = list(
            get_layers_from_vllm_config(vllm_config, AttentionLayerBase, None).keys()
        )
        self._orchestrator = PipeliningOrchestrator(self._layer_names, self._k_window)

        self._worker_state: LayerwisePipeliningState | None = None
        self._forward_context: ForwardContext | None = None
        self._has_decode_work = False
        self._decode_max_seq_len: int | None = None
        self._window_pools: dict[WindowKey, WindowPool] = {}
        self._original_kv_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._original_kv_cache_meta: dict[
            tuple[str, int], tuple[torch.Size, torch.dtype, torch.device]
        ] = {}
        self._free_original_buffers = bool(
            getattr(vllm_config.cache_config, "free_original_kv_cache", False)
            or self._kv_transfer_config.get_from_extra_config(
                "free_original_kv_cache", False
            )
        )
        self._freed_original_buffers = False

        # Pipelining-aware block inflation: when num_physical_blocks is set,
        # the block pool has been inflated and window buffers must be sized at
        # the inflated num_blocks (not the physical per-layer tensor size).
        self._num_blocks_pipelined: int | None = None
        self._num_physical_blocks: int | None = None
        if kv_cache_config is not None:
            phys = getattr(kv_cache_config, "num_physical_blocks", None)
            if phys is not None:
                self._num_blocks_pipelined = kv_cache_config.num_blocks
                self._num_physical_blocks = phys
                logger.info(
                    "Connector: block pool inflated to %d "
                    "(physical per-layer tensors: %d blocks)",
                    self._num_blocks_pipelined,
                    self._num_physical_blocks,
                )

        if role == KVConnectorRole.WORKER:
            self._worker_state = LayerwisePipeliningState()

    # ------------------------------------------------------------------
    # Window buffer helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_block_dim(kv_layer: torch.Tensor) -> int:
        """Return the dimension index that corresponds to the num_blocks axis.

        Convention in vLLM:
        - shape[0] == 2  ->  [2(K/V), num_blocks, ...]  ->  block_dim = 1
        - otherwise       ->  [num_blocks, ...]           ->  block_dim = 0
        """
        return 1 if kv_layer.shape[0] == 2 else 0

    def _inflated_shape(self, kv_layer: torch.Tensor) -> torch.Size:
        """Return the shape the window buffer should have (maybe inflated)."""
        if self._num_blocks_pipelined is None:
            return kv_layer.shape
        block_dim = self._get_block_dim(kv_layer)
        if self._num_blocks_pipelined <= kv_layer.shape[block_dim]:
            return kv_layer.shape
        new_shape = list(kv_layer.shape)
        new_shape[block_dim] = self._num_blocks_pipelined
        return torch.Size(new_shape)

    def _make_inflated_buffer(self, kv_layer: torch.Tensor) -> torch.Tensor:
        """Create a window buffer inflated to ``_num_blocks_pipelined``.

        If the block pool was NOT inflated (no pipelining savings), this
        simply returns ``torch.empty_like(kv_layer)`` — the old behaviour.
        """
        if self._num_blocks_pipelined is None:
            return torch.empty_like(kv_layer)

        block_dim = self._get_block_dim(kv_layer)
        current_blocks = kv_layer.shape[block_dim]

        if self._num_blocks_pipelined <= current_blocks:
            return torch.empty_like(kv_layer)

        new_shape = list(kv_layer.shape)
        new_shape[block_dim] = self._num_blocks_pipelined
        return torch.empty(
            new_shape, dtype=kv_layer.dtype, device=kv_layer.device
        )

    def _get_window_buffer(
        self, layer_name: str, kv_layer: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        # Use inflated shape as the pool key so all windows share the same
        # (potentially larger) geometry.
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
            raise ValueError(f"Layer {layer_name} not found in forward context")
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
        window_kv, is_new_binding = self._get_window_buffer(layer_name, kv_layer)
        layer.kv_cache[ve] = window_kv
        if self._free_original_buffers and self._has_decode_work:
            self._original_kv_cache.pop(original_key, None)
            self._freed_original_buffers = True
        return window_kv, is_new_binding

    def _restore_original_buffers(
        self, forward_context: "ForwardContext" | None = None
    ) -> None:
        forward_context = forward_context or self._forward_context
        if forward_context is None:
            return
        for (layer_name, ve), kv_layer in list(self._original_kv_cache.items()):
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
        block_size = self._vllm_config.cache_config.block_size
        max_blocks = cdiv(max_seq_len, block_size)
        if kv_layer.shape[0] == 2:
            block_dim = 1
        else:
            block_dim = 0
        max_blocks = min(max_blocks, kv_layer.shape[block_dim])
        if max_blocks == kv_layer.shape[block_dim]:
            return kv_layer
        slicer = [slice(None)] * kv_layer.ndim
        slicer[block_dim] = slice(0, max_blocks)
        return kv_layer[tuple(slicer)]

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if not self._enabled or self._worker_state is None:
            return
        if not self.has_connector_metadata():
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, LayerwisePipelinedConnectorMetadata):
            return
        if not metadata.has_work or not metadata.has_decode_work:
            if self._free_original_buffers and self._freed_original_buffers:
                logger.warning_once(
                    "KV pipelining freed original KV buffers, but prefill is now "
                    "scheduled. Reinitializing empty KV buffers for prefill."
                )
            self._restore_original_buffers(forward_context)
            return

        self._has_decode_work = metadata.has_decode_work
        self._decode_max_seq_len = metadata.decode_max_seq_len
        self._forward_context = forward_context
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
        if not self._enabled or self._worker_state is None:
            return
        self._worker_state.wait_for_layer(layer_name)
        if self._forward_context is None:
            return
        for next_layer_name in self._orchestrator.plan_prefetch(layer_name):
            kv_layer, is_new_binding = self._bind_layer_window(
                next_layer_name, self._forward_context
            )
            if not is_new_binding:
                continue
            attn_metadata = self._forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata.get(next_layer_name)
            max_seq_len = self._get_max_seq_len(attn_metadata)
            kv_view = self._slice_kv_layer(kv_layer, max_seq_len)
            self._worker_state.prefetch_to_gpu(next_layer_name, kv_view)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        if not self._enabled or self._worker_state is None:
            return
        if not self._has_decode_work:
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
        # For the initial full save, ensure the CPU buffer is created at the
        # FULL window buffer size (which may be inflated by pipelining).  This
        # avoids a costly buffer re-creation when stage_delta_to_cpu is called
        # on the next step with the full-size tensor.
        self._worker_state.ensure_cpu_buffer(layer_name, kv_layer)
        max_seq_len = self._get_max_seq_len(attn_metadata)
        kv_view = self._slice_kv_layer(kv_layer, max_seq_len)
        self._worker_state.stage_to_cpu(layer_name, kv_view)

    def wait_for_save(self):
        if not self._enabled or self._worker_state is None:
            return
        self._worker_state.wait_for_saves()
        self._forward_context = None
        self._has_decode_work = False
        self._decode_max_seq_len = None
        self._original_kv_cache.clear()

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        return

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        if not self._enabled:
            return LayerwisePipelinedConnectorMetadata()
        active_req_ids = (
            [req.req_id for req in scheduler_output.scheduled_new_reqs]
            + list(scheduler_output.scheduled_cached_reqs.req_ids)
        )
        decode_only = not scheduler_output.scheduled_new_reqs
        decode_req_ids = set(scheduler_output.scheduled_cached_reqs.req_ids)
        decode_req_ids.update(scheduler_output.scheduled_spec_decode_tokens.keys())
        active_decode_req_ids = sorted(decode_req_ids) if decode_only else []
        has_decode_work = len(active_decode_req_ids) > 0
        decode_max_seq_len: int | None = None
        if has_decode_work:
            cached_reqs = scheduler_output.scheduled_cached_reqs
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            max_len = 0
            for req_id, num_computed in zip(
                cached_reqs.req_ids, cached_reqs.num_computed_tokens
            ):
                max_len = max(
                    max_len, num_computed + num_scheduled_tokens.get(req_id, 0)
                )
            decode_max_seq_len = max_len if max_len > 0 else None
        has_work = scheduler_output.total_num_scheduled_tokens > 0
        if self._k_window <= 0 or not has_work:
            return LayerwisePipelinedConnectorMetadata(
                has_work=has_work,
                active_req_ids=active_req_ids,
                has_decode_work=has_decode_work,
                active_decode_req_ids=active_decode_req_ids,
                decode_max_seq_len=decode_max_seq_len,
            )
        return LayerwisePipelinedConnectorMetadata(
            layers_to_prefetch=self._layer_names[: self._k_window],
            has_work=has_work,
            active_req_ids=active_req_ids,
            has_decode_work=has_decode_work,
            active_decode_req_ids=active_decode_req_ids,
            decode_max_seq_len=decode_max_seq_len,
        )
