from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_pipelining import LayerwisePipeliningState
from vllm.v1.kv_pipelining_direct_window import DirectKWindowMixin

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class LayerwisePipelinedConnectorMetadata(KVConnectorMetadata):
    layers_to_prefetch: list[str] = field(default_factory=list)
    has_work: bool = False
    active_req_ids: list[str] = field(default_factory=list)
    has_decode_work: bool = False
    active_decode_req_ids: list[str] = field(default_factory=list)
    decode_max_seq_len: int | None = None
    decode_physical_block_ids: list[int] = field(default_factory=list)
    decode_req_blocks: dict[str, list[int]] = field(default_factory=dict)
    finished_req_ids: list[str] = field(default_factory=list)


class LayerwisePipelinedConnector(KVConnectorBase_V1, DirectKWindowMixin):
    """Decode-only, layer-wise KV pipelining connector."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        self._enabled = bool(
            getattr(vllm_config.cache_config, "enable_kv_pipelining", False)
        ) or bool(
            getattr(vllm_config.cache_config, "decode_only_pipelining", False)
        )
        self._init_direct_k_window(vllm_config)
        self._decode_max_seq_len: int | None = None
        self._decode_physical_block_ids: list[int] = []
        self._active_req_blocks: dict[str, set[int]] = {}
        self._worker_registered_reqs: set[str] = set()

        self._num_blocks_pipelined: int | None = None
        if kv_cache_config is not None:
            phys = getattr(kv_cache_config, "num_physical_blocks", None)
            if phys is not None:
                self._num_blocks_pipelined = kv_cache_config.num_blocks
                logger.info(
                    "Connector: block pool inflated to %d "
                    "(physical per-layer tensors: %d blocks)",
                    self._num_blocks_pipelined,
                    phys,
                )
            elif self._decode_only:
                self._num_blocks_pipelined = kv_cache_config.num_blocks
                logger.info(
                    "Connector (decode-only P/D): block pool = %d",
                    self._num_blocks_pipelined,
                )

        if role == KVConnectorRole.WORKER:
            self._worker_state = LayerwisePipeliningState()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches
        max_model_len = self._vllm_config.model_config.max_model_len
        block_size = self._vllm_config.cache_config.block_size
        self.register_kv_caches_direct(
            kv_caches,
            max_model_len=max_model_len,
            block_size=block_size,
            enable_cpu_table=self._decode_only and self._k_window > 0,
        )

    @staticmethod
    def _expand_block_ids(block_ids: set[int] | list[int]) -> list[int]:
        return sorted(int(b) for b in block_ids)

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
        block_dim = 1 if kv_layer.shape[0] == 2 else 0
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
            return

        self._has_decode_work = metadata.has_decode_work
        self._decode_max_seq_len = metadata.decode_max_seq_len
        self._decode_physical_block_ids = metadata.decode_physical_block_ids
        self._forward_context = forward_context
        self._cache_decode_phys_blocks(
            self._decode_physical_block_ids or None
        )

        for req_id in metadata.finished_req_ids:
            if self._worker_state is not None:
                self._worker_state.release_req_blocks(req_id)
            self._worker_registered_reqs.discard(req_id)

        if (
            self._worker_state is not None
            and metadata.decode_req_blocks
            and self._kv_caches
        ):
            dev = next(iter(self._kv_caches.values())).device
            for req_id, blks in metadata.decode_req_blocks.items():
                if req_id not in self._worker_registered_reqs:
                    self._worker_state.register_req_blocks(req_id, blks, dev)
                    self._worker_registered_reqs.add(req_id)

        phys = self._decode_phys_blocks
        for layer_name in metadata.layers_to_prefetch:
            state = self._worker_state._layers.get(layer_name)
            if state is not None and state.ready_event is not None:
                continue
            kv_layer, is_new_binding, evicted_layer = self._bind_layer_direct(
                layer_name, forward_context
            )
            if not is_new_binding:
                continue
            if self._worker_state.uses_cpu_kv_table() and phys:
                layer_idx = self._layer_index.get(layer_name, 0)
                wait_evicted = (
                    None
                    if self._worker_state.uses_delta_scratch()
                    else evicted_layer
                )
                self._worker_state.prefetch_table_blocks_to_gpu(
                    layer_name,
                    kv_layer,
                    phys,
                    gpu_wait_save_from_layer=wait_evicted,
                    layer_idx=layer_idx,
                )
            else:
                attn_metadata = forward_context.attn_metadata
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata.get(layer_name)
                max_seq_len = self._get_max_seq_len(attn_metadata)
                kv_view = self._slice_kv_layer(kv_layer, max_seq_len)
                self._worker_state.prefetch_to_gpu(
                    layer_name,
                    kv_view,
                    gpu_wait_save_from_layer=evicted_layer,
                )

        self._begin_decode_step_plan(forward_context)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._enabled or self._worker_state is None:
            return
        self._worker_state.wait_for_layer(layer_name)

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
            n_tok = self._effective_decode_n_tokens(
                slot_mapping, num_actual_tokens
            )
            host_slot = self._worker_state.resolve_n1_host_slot(
                slot_mapping, num_actual_tokens
            )
            layer_idx = self._layer_index[layer_name]
            if (
                self._use_step_plan
                and self._step_plan.prepared
                and not getattr(self, "_skip_delta_stage", False)
            ):
                self._issue_decode_layer_step(layer_name, kv_layer)
                return
            if n_tok > 1 and self._use_batched_d2h:
                if not self._worker_state._decode_batch_prepared:
                    self._worker_state.prepare_decode_batch(
                        slot_mapping, num_actual_tokens
                    )
                if self._worker_state._decode_batch_active:
                    self._decode_stage_delta_batched_and_prefetch(
                        layer_name, kv_layer
                    )
                else:
                    self._worker_state.stage_delta_to_cpu(
                        layer_name,
                        kv_layer,
                        slot_mapping,
                        num_actual_tokens=num_actual_tokens,
                    )
            elif host_slot is not None:
                self._decode_stage_delta_and_prefetch_n1(
                    layer_name, kv_layer, host_slot
                )
            else:
                self._worker_state.stage_delta_to_cpu(
                    layer_name,
                    kv_layer,
                    slot_mapping,
                    num_actual_tokens=num_actual_tokens,
                )
            if n_tok <= 1 and (
                host_slot is None
                or (
                    num_actual_tokens is not None and num_actual_tokens != 1
                )
            ):
                self._prefetch_after_save_by_index(
                    layer_idx, physical_block_ids=self._decode_phys_blocks
                )
            elif n_tok > 1 and not self._worker_state._decode_batch_active:
                self._prefetch_after_save_by_index(
                    layer_idx, physical_block_ids=self._decode_phys_blocks
                )
            return
        if (
            self._worker_state.uses_cpu_kv_table()
            and self._decode_physical_block_ids
        ):
            self._worker_state.seed_table_from_gpu(
                layer_name,
                kv_layer,
                self._decode_physical_block_ids,
            )
            return
        self._worker_state.ensure_cpu_buffer(layer_name, kv_layer)
        max_seq_len = self._get_max_seq_len(attn_metadata)
        kv_view = self._slice_kv_layer(kv_layer, max_seq_len)
        self._worker_state.stage_to_cpu(layer_name, kv_view)

    def wait_for_save(self) -> None:
        if not self._enabled or self._worker_state is None:
            return
        if self._has_decode_work and self._forward_context is not None:
            self._prefetch_bootstrap_layers()
        self._worker_state.wait_for_saves(deferred=self._decode_only)
        self._clear_decode_step_cache()
        self._forward_context = None
        self._has_decode_work = False
        self._decode_max_seq_len = None
        self._decode_physical_block_ids = []

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        block_groups = blocks.get_block_ids()
        all_ids: set[int] = set()
        for group in block_groups:
            all_ids.update(int(b) for b in group)
        if all_ids:
            self._active_req_blocks[request.request_id] = all_ids

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        if not self._enabled:
            return LayerwisePipelinedConnectorMetadata()

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            new_blks = cached_reqs.new_block_ids[i]
            if new_blks is not None:
                existing = self._active_req_blocks.setdefault(req_id, set())
                for group in new_blks:
                    existing.update(int(b) for b in group)

        for req_id in scheduler_output.finished_req_ids:
            self._active_req_blocks.pop(req_id, None)

        active_req_ids = (
            [req.req_id for req in scheduler_output.scheduled_new_reqs]
            + list(scheduler_output.scheduled_cached_reqs.req_ids)
        )
        decode_req_ids = set(cached_reqs.req_ids)
        decode_req_ids.update(scheduler_output.scheduled_spec_decode_tokens.keys())
        has_decode_work = len(decode_req_ids) > 0 and not scheduler_output.scheduled_new_reqs
        decode_max_seq_len: int | None = None
        if has_decode_work:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            max_len = 0
            for req_id, num_computed in zip(
                cached_reqs.req_ids, cached_reqs.num_computed_tokens
            ):
                max_len = max(
                    max_len, num_computed + num_scheduled_tokens.get(req_id, 0)
                )
            decode_max_seq_len = max_len if max_len > 0 else None

        decode_physical_block_ids: list[int] = []
        decode_req_blocks: dict[str, list[int]] = {}
        if has_decode_work:
            all_blocks: set[int] = set()
            for req_id in decode_req_ids:
                blocks = self._active_req_blocks.get(req_id)
                if blocks:
                    decode_req_blocks[req_id] = self._expand_block_ids(blocks)
                    all_blocks.update(blocks)
            decode_physical_block_ids = self._expand_block_ids(all_blocks)

        has_work = scheduler_output.total_num_scheduled_tokens > 0
        finished = list(scheduler_output.finished_req_ids)
        if self._k_window <= 0 or not has_work:
            return LayerwisePipelinedConnectorMetadata(
                has_work=has_work,
                active_req_ids=active_req_ids,
                has_decode_work=has_decode_work,
                active_decode_req_ids=sorted(decode_req_ids) if has_decode_work else [],
                decode_max_seq_len=decode_max_seq_len,
                decode_physical_block_ids=decode_physical_block_ids,
                decode_req_blocks=decode_req_blocks,
                finished_req_ids=finished,
            )
        return LayerwisePipelinedConnectorMetadata(
            layers_to_prefetch=self._layer_names[: self._k_window],
            has_work=has_work,
            active_req_ids=active_req_ids,
            has_decode_work=has_decode_work,
            active_decode_req_ids=sorted(decode_req_ids) if has_decode_work else [],
            decode_max_seq_len=decode_max_seq_len,
            decode_physical_block_ids=decode_physical_block_ids,
            decode_req_blocks=decode_req_blocks,
            finished_req_ids=finished,
        )
