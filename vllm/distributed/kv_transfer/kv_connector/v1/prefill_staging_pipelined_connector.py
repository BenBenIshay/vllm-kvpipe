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
    has_prefill_work: bool = False


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

        # Layer list (deterministic order)
        self._layer_names: list[str] = list(
            get_layers_from_vllm_config(
                vllm_config, AttentionLayerBase, None
            ).keys()
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

        if role == KVConnectorRole.WORKER:
            self._worker_state = LayerwisePipeliningState()

        # Timing stats
        self._step_count = 0
        self._prefill_steps = 0
        self._decode_steps = 0

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
        logger.info(
            "PrefillStagingPipelined: registered %d layers across %d "
            "physical GPU tensors (K+1=%d)",
            len(kv_caches),
            len(tensor_ids),
            self._num_physical_tensors,
        )

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

    def _slice_kv_layer(
        self, kv_layer: torch.Tensor, max_seq_len: int | None
    ) -> torch.Tensor:
        """Return a view covering only the blocks needed for max_seq_len."""
        if max_seq_len is None or kv_layer.ndim < 3:
            return kv_layer
        block_dim = self._get_block_dim(kv_layer)
        max_blocks = min(
            cdiv(max_seq_len, self._block_size),
            kv_layer.shape[block_dim],
        )
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
        # Prefill step: no prefetching needed.  The model runs real
        # attention and populates the GPU tensors natively.
        # save_kv_layer will stage each layer's KV to CPU.
        # ------------------------------------------------------------------
        if self._has_prefill_work:
            return

        # ------------------------------------------------------------------
        # Pure decode step: prefetch the first K layers from CPU → GPU.
        # ------------------------------------------------------------------
        if not self._has_decode_work or self._kv_caches is None:
            return

        for layer_name in metadata.layers_to_prefetch:
            kv_layer = self._kv_caches[layer_name]
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata.get(layer_name)
            max_seq_len = self._get_max_seq_len(attn_metadata)
            kv_view = self._slice_kv_layer(kv_layer, max_seq_len)
            self._worker_state.prefetch_to_gpu(layer_name, kv_view)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self._worker_state is None:
            return

        # ------------------------------------------------------------------
        # Prefill step: ensure any pending GPU→CPU copy from the previous
        # occupant of this physical tensor has completed before the model
        # writes new KV data into it.
        # ------------------------------------------------------------------
        if self._has_prefill_work:
            layer_idx = self._layer_index.get(layer_name, -1)
            prev_idx = layer_idx - self._num_physical_tensors
            if prev_idx >= 0:
                prev_layer = self._layer_names[prev_idx]
                self._worker_state.wait_for_layer_save(prev_layer)
            return

        # ------------------------------------------------------------------
        # Pure decode step: wait for this layer's CPU→GPU prefetch,
        # then trigger prefetching the next layer(s) in the pipeline.
        # ------------------------------------------------------------------
        if not self._has_decode_work:
            return

        self._worker_state.wait_for_layer(layer_name)

        if self._forward_context is None or self._kv_caches is None:
            return
        for next_layer in self._orchestrator.plan_prefetch(layer_name):
            # Before prefetching into the shared tensor, ensure any
            # in-flight save from the previous occupant has completed.
            next_idx = self._layer_index.get(next_layer, -1)
            prev_occupant_idx = next_idx - self._num_physical_tensors
            if prev_occupant_idx >= 0:
                prev_occupant = self._layer_names[prev_occupant_idx]
                self._worker_state.wait_for_layer_save(prev_occupant)

            kv_layer = self._kv_caches[next_layer]
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
        # Prefill step: stage the layer KV to CPU.
        # Only copy the blocks that are actually in use (up to max_seq_len),
        # not the full shared tensor.  This keeps CPU memory manageable.
        # ------------------------------------------------------------------
        if self._has_prefill_work:
            max_seq_len = self._get_max_seq_len(attn_metadata)
            kv_view = self._slice_kv_layer(kv_layer, max_seq_len)
            self._worker_state.stage_to_cpu(layer_name, kv_view)
            return

        # ------------------------------------------------------------------
        # Pure decode step: stage only the delta (new decode tokens).
        # The full KV is already on CPU from the prefill phase; we only
        # need to update it with the newly generated token's KV.
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
        max_seq_len = self._get_max_seq_len(attn_metadata)
        kv_view = self._slice_kv_layer(kv_layer, max_seq_len)
        self._worker_state.stage_to_cpu(layer_name, kv_view)

    def wait_for_save(self) -> None:
        if self._worker_state is None:
            return
        self._worker_state.wait_for_saves()

        if self._has_prefill_work:
            self._prefill_steps += 1
        if self._has_decode_work:
            self._decode_steps += 1

        self._forward_context = None
        self._has_decode_work = False
        self._has_prefill_work = False
        self._decode_max_seq_len = None

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

        # Layers to prefetch for decode (first K layers)
        layers_to_prefetch: list[str] = []
        if self._k_window > 0 and has_work and has_decode_work:
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
