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
     dummy values, stages every layer to CPU pinned memory, then uses the
     allocator-provided rolling KV tensors directly.
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

import os
import time
from collections import deque
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
from vllm.v1.cpu_kv_table import _kv_layout_blocks_first
from vllm.v1.kv_cache_pipelining import LayerwisePipeliningState
from vllm.v1.kv_pipelining_direct_window import DirectKWindowMixin
from vllm.v1.kv_pipelining_profile import (
    DecodeHookProfile,
    kv_pipelining_profile_enabled,
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
    decode_block_runs: list[tuple[int, int]] = field(default_factory=list)
    decode_physical_block_ids: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class PipelinedDecodeBenchConnector(KVConnectorBase_V1, DirectKWindowMixin):
    """Decode-only P/D pipelining benchmark connector.

    Scheduler side  — behaves like DecodeBenchConnector: claims all prompt
    tokens as externally matched and records block allocations so the worker
    can fill them.

    Worker side — on the *first* step for each request, fills KV on GPU
    with dummy values, stages every layer to CPU pinned memory, then
    binds rolling allocator tensors.  All subsequent decode steps use the same
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
        # This connector is used only for dummy decode-only P/D benchmarking.
        # Treat it as decode-only even if the derived cache flag is not visible
        # yet during early engine construction.
        self._decode_only = True
        self._block_size = cache_cfg.block_size
        self._init_direct_k_window(
            vllm_config, k_window=int(getattr(cache_cfg, "kv_k_window", 4)), decode_only=True
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
        self._decode_block_runs: list[tuple[int, int]] = []
        self._decode_physical_block_ids: list[int] = []
        self._worker_registered_reqs: set[str] = set()

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
        self._total_save_submit_time = 0.0
        self._total_save_submit_calls = 0
        self._skip_delta_stage = (
            os.environ.get("VLLM_KVPIPE_SKIP_DELTA_STAGE", "0") == "1"
        )
        self._fake_transfer = (
            os.environ.get("VLLM_KVPIPE_FAKE_TRANSFER", "0") == "1"
        )

        # --- wait_for_layer_load instrumentation ---
        # Pending CUDA event pairs (layer, step, before_event, after_event).
        # Drained non-blocking each step via Event.query().
        # When enough samples accumulate they get summarised every 100 steps.
        self._wait_log_enabled: bool = (
            os.environ.get("VLLM_KVPIPE_WAIT_LOG", "0") == "1"
        )
        self._profile_save_submit: bool = (
            os.environ.get("VLLM_KVPIPE_PROFILE_SAVE", "0") == "1"
        )
        self._profile_decode: bool = kv_pipelining_profile_enabled()
        if self._profile_decode and self._hook_profile is None:
            self._hook_profile = DecodeHookProfile()
        self._pending_wait_events: deque[
            tuple[str, int, "torch.cuda.Event", "torch.cuda.Event"]
        ] = deque()
        # Per-layer accumulated wait samples in ms since last summary log.
        self._wait_samples_ms: dict[str, list[float]] = {}
        # Across-all-layers samples for quick aggregate summary.
        self._wait_samples_all_ms: list[float] = []

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        max_model_len = self._vllm_config.model_config.max_model_len
        self.register_kv_caches_direct(
            kv_caches,
            max_model_len=max_model_len,
            block_size=self._block_size,
            enable_cpu_table=True,
        )

    def _bind_layer_window(
        self, layer_name: str, forward_context: "ForwardContext"
    ) -> tuple[torch.Tensor, bool, str | None]:
        return self._bind_layer_direct(layer_name, forward_context)

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
                        kv_cache[valid_ids] = fill
                    else:
                        # Constant fill: assign the scalar in-place via
                        # index_fill_ to avoid materializing a large transient
                        # (num_blocks x block_shape) tensor per request/layer.
                        # This is the dummy stand-in for prefill-delivered KV;
                        # the value is irrelevant to decode throughput, but the
                        # transient allocation drove admission-time OOM / TTFT
                        # bursts at high concurrency.
                        kv_cache.index_fill_(0, valid_ids, self._fill_mean)

    @staticmethod
    def _block_runs_from_ids(block_ids: set[int] | list[int]) -> list[tuple[int, int]]:
        if not block_ids:
            return []
        runs: list[tuple[int, int]] = []
        sorted_ids = sorted(int(b) for b in block_ids)
        start = prev = sorted_ids[0]
        for bid in sorted_ids[1:]:
            if bid == prev + 1:
                prev = bid
                continue
            runs.append((start, prev))
            start = prev = bid
        runs.append((start, prev))
        return runs

    @classmethod
    def _block_runs_from_req_fills(
        cls,
        reqs_to_fill: dict[str, tuple[tuple[list[int], ...], int]]
    ) -> list[tuple[int, int]]:
        """Return compact inclusive block-id runs used by the seed requests."""
        block_ids: set[int] = set()
        for block_ids_per_group, _num_tokens in reqs_to_fill.values():
            for group_ids in block_ids_per_group:
                block_ids.update(int(b) for b in group_ids)
        return cls._block_runs_from_ids(block_ids)

    def _stage_all_layers_to_cpu(
        self,
        reqs_to_fill: dict[str, tuple[tuple[list[int], ...], int]],
    ) -> None:
        """Seed the upfront CpuKvTable from GPU dummy fill."""
        if self._kv_caches is None or self._worker_state is None:
            return
        phys_blocks: list[int] = []
        for req_id, (block_ids_per_group, _num_tokens) in reqs_to_fill.items():
            for group in block_ids_per_group:
                phys_blocks.extend(int(b) for b in group)
            if phys_blocks and req_id not in self._worker_registered_reqs:
                dev = next(iter(self._kv_caches.values())).device
                self._worker_state.register_req_blocks(
                    req_id, phys_blocks, dev
                )
                self._worker_registered_reqs.add(req_id)
        phys_blocks = sorted(set(phys_blocks))
        self._decode_physical_block_ids = phys_blocks
        for layer_name, kv_cache in self._kv_caches.items():
            if self._worker_state.uses_cpu_kv_table() and phys_blocks:
                self._worker_state.seed_table_from_gpu(
                    layer_name, kv_cache, phys_blocks
                )
            else:
                block_runs = self._block_runs_from_req_fills(reqs_to_fill)
                if block_runs:
                    self._worker_state.stage_to_cpu_block_runs_compact(
                        layer_name, kv_cache, block_runs
                    )
                else:
                    kv_view = self._slice_kv_layer(
                        kv_cache, self._decode_max_seq_len
                    )
                    self._worker_state.stage_to_cpu(layer_name, kv_view)
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
        block_dim = 0 if _kv_layout_blocks_first(kv_layer) else (
            1 if kv_layer.shape[0] == 2 else 0
        )

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

        prof = self._hook_profile

        self._forward_context = forward_context
        self._has_decode_work = metadata.has_decode_work
        self._decode_max_seq_len = metadata.decode_max_seq_len
        self._decode_max_block_id = metadata.decode_max_block_id
        self._decode_block_runs = metadata.decode_block_runs
        self._decode_physical_block_ids = metadata.decode_physical_block_ids

        phys_for_step = self._decode_physical_block_ids or [
            b
            for lo, hi in self._decode_block_runs
            for b in range(lo, hi + 1)
        ]
        self._cache_decode_phys_blocks(phys_for_step or None)

        # --- Phase 1: Fill + stage on first step for new requests ----------
        if metadata.reqs_to_fill:
            t0 = time.perf_counter()
            self._fill_kv_on_gpu(metadata.reqs_to_fill)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            self._stage_all_layers_to_cpu(metadata.reqs_to_fill)
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

        if prof is not None:
            prof.begin_decode_forward()
        t_start = time.perf_counter() if prof is not None else 0.0

        self._begin_decode_step_plan(forward_context)

        for layer_name in metadata.layers_to_prefetch:
            state = self._worker_state._layers.get(layer_name)
            if state is not None and state.ready_event is not None:
                if prof is not None:
                    prof.fwd.start_prefetch_skipped_ready += 1
                continue
            kv_layer, is_new_binding, evicted_layer = self._bind_layer_window(
                layer_name, forward_context
            )
            if not is_new_binding:
                if prof is not None:
                    prof.fwd.start_prefetch_skipped_bound += 1
                continue
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata.get(layer_name)
            max_seq_len = self._get_max_seq_len(attn_metadata)
            kv_view = self._slice_kv_layer(kv_layer, max_seq_len)
            phys = self._decode_phys_blocks
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
            elif self._decode_block_runs:
                self._worker_state.prefetch_to_gpu_block_runs_compact(
                    layer_name,
                    kv_layer,
                    self._decode_block_runs,
                    gpu_wait_save_from_layer=evicted_layer,
                )
            else:
                self._worker_state.prefetch_to_gpu(
                    layer_name,
                    kv_view,
                    gpu_wait_save_from_layer=evicted_layer,
                )
            if prof is not None:
                prof.fwd.start_prefetch_issued += 1

        if prof is not None:
            prof.fwd.start_load_host_ms += (time.perf_counter() - t_start) * 1000.0

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self._worker_state is None:
            return
        prof = self._hook_profile
        t0 = time.perf_counter() if prof is not None else 0.0
        state = self._worker_state._layers.get(layer_name)
        ready = state.ready_event if state is not None else None
        had_ready = ready is not None
        before_evt = None
        after_evt = None
        if had_ready and (prof is not None or self._wait_log_enabled):
            before_evt = torch.cuda.Event(enable_timing=True)
            after_evt = torch.cuda.Event(enable_timing=True)
            before_evt.record(torch.cuda.current_stream())
            torch.cuda.current_stream().wait_event(ready)
            state.ready_event = None
            after_evt.record(torch.cuda.current_stream())
            if self._wait_log_enabled:
                self._pending_wait_events.append(
                    (layer_name, self._step_count, before_evt, after_evt)
                )
        elif had_ready:
            torch.cuda.current_stream().wait_event(ready)
            state.ready_event = None
        elif self._wait_log_enabled:
            self._worker_state.wait_for_layer(layer_name)
        if prof is not None:
            host_ms = (time.perf_counter() - t0) * 1000.0
            prof.record_wait_load(
                layer_name,
                had_ready=had_ready,
                host_ms=host_ms,
                before_evt=before_evt,
                after_evt=after_evt,
            )

    def _prefetch_layer_after_current_save(self, layer_name: str) -> None:
        self._prefetch_after_save(
            layer_name,
            physical_block_ids=self._decode_phys_blocks,
        )

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
            layer_idx = self._layer_index[layer_name]
            if self._use_step_plan and self._step_plan.prepared:
                if self._skip_delta_stage:
                    prof = self._hook_profile
                    t_save = time.perf_counter() if prof is not None else 0.0
                    self._prefetch_after_save_by_index(
                        layer_idx, physical_block_ids=self._decode_phys_blocks
                    )
                    self._step_plan.layers_issued += 1
                    if prof is not None:
                        prof.fwd.save_calls += 1
                        prof.fwd.save_host_ms += (
                            time.perf_counter() - t_save
                        ) * 1000.0
                    return
                self._issue_decode_layer_step(layer_name, kv_layer)
                return
            prof = self._hook_profile
            t_save = time.perf_counter() if prof is not None else 0.0
            n_tok = self._effective_decode_n_tokens(
                slot_mapping, num_actual_tokens
            )
            host_slot = self._worker_state.resolve_n1_host_slot(
                slot_mapping, num_actual_tokens
            )
            t0 = time.perf_counter() if self._profile_save_submit else 0.0
            if not self._skip_delta_stage:
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
            if self._profile_save_submit:
                self._total_save_submit_time += time.perf_counter() - t0
                self._total_save_submit_calls += 1
            if self._skip_delta_stage or (
                n_tok <= 1
                and (
                    host_slot is None
                    or (
                        num_actual_tokens is not None
                        and num_actual_tokens != 1
                    )
                )
            ) or (
                n_tok > 1 and not self._worker_state._decode_batch_active
            ):
                self._prefetch_after_save_by_index(
                    layer_idx, physical_block_ids=self._decode_phys_blocks
                )
            if prof is not None:
                prof.fwd.save_calls += 1
                prof.fwd.save_host_ms += (time.perf_counter() - t_save) * 1000.0
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
        t0 = time.perf_counter()
        self._worker_state.stage_to_cpu(layer_name, kv_view)
        self._total_save_submit_time += time.perf_counter() - t0
        self._total_save_submit_calls += 1
        self._prefetch_layer_after_current_save(layer_name)

    def wait_for_save(self) -> None:
        if self._worker_state is None:
            return
        prof = self._hook_profile
        t_ws = time.perf_counter() if prof is not None else 0.0
        if self._has_decode_work and self._forward_context is not None:
            t_bs = time.perf_counter() if prof is not None else 0.0
            self._prefetch_bootstrap_layers()
            if prof is not None:
                prof.fwd.bootstrap_host_ms += (time.perf_counter() - t_bs) * 1000.0
        # In the rolling-K decode path, a global end-of-step drain serializes
        # all D2H delta copies and CPU scatter updates.  Correctness only
        # requires targeted ordering: prefetch waits for the previous occupant's
        # save_event on the transfer stream, and CPU-buffer readiness is checked
        # by the prefetch path for the specific layer being loaded.
        self._worker_state.wait_for_saves(deferred=True)
        self._clear_decode_step_cache()
        self._forward_context = None
        had_decode = self._has_decode_work
        self._has_decode_work = False
        self._decode_max_seq_len = None
        self._decode_max_block_id = None
        self._decode_block_runs = []

        self._step_count += 1

        self._drain_wait_events()

        if prof is not None:
            prof.fwd.wait_save_host_ms += (time.perf_counter() - t_ws) * 1000.0
            if had_decode:
                prof.end_decode_forward(
                    step=self._step_count, k_window=self._k_window
                )

        if self._step_count % 100 == 0:
            pinned_gb = (
                self._worker_state.pinned_bytes_allocated / 1e9
                if self._worker_state is not None
                else 0.0
            )
            logger.info(
                "PipelinedDecodeBench step %d: cumulative fill=%.3fs, "
                "stage=%.3fs, save_submit=%.3fs/%d calls "
                "(avg %.3f ms), pinned CPU memory: %.2f GB",
                self._step_count,
                self._total_fill_time,
                self._total_stage_time,
                self._total_save_submit_time,
                self._total_save_submit_calls,
                (
                    1000.0
                    * self._total_save_submit_time
                    / max(self._total_save_submit_calls, 1)
                ),
                pinned_gb,
            )
            self._log_wait_summary()

    def _drain_wait_events(self) -> None:
        """Move any completed CUDA event pairs into ms samples.

        Non-blocking: only events whose ``after`` already finished are
        consumed.  Unfinished pairs stay in the deque and are checked again
        next step.  This keeps the hot path free of synchronous waits.
        """
        if not self._pending_wait_events:
            return
        while self._pending_wait_events:
            layer_name, _step, before_evt, after_evt = (
                self._pending_wait_events[0]
            )
            try:
                if not after_evt.query():
                    break
                elapsed_ms = float(before_evt.elapsed_time(after_evt))
            except Exception:
                self._pending_wait_events.popleft()
                continue
            self._pending_wait_events.popleft()
            self._wait_samples_ms.setdefault(layer_name, []).append(
                elapsed_ms
            )
            self._wait_samples_all_ms.append(elapsed_ms)

    def _log_wait_summary(self) -> None:
        """Log p50/p99/max of `wait_for_layer` stalls since the last summary."""
        if not self._wait_log_enabled:
            return
        if not self._wait_samples_all_ms:
            logger.info(
                "PipelinedDecodeBench wait_for_layer stats: no samples yet "
                "(K=%d, pending=%d).",
                self._k_window,
                len(self._pending_wait_events),
            )
            return
        all_samples = sorted(self._wait_samples_all_ms)
        n = len(all_samples)
        p50 = all_samples[n // 2]
        p99 = all_samples[min(n - 1, max(0, int(0.99 * n) - 1))]
        mx = all_samples[-1]
        mean = sum(all_samples) / n
        # Per-layer p99 (only print the few worst layers).
        per_layer_p99: list[tuple[str, float, int]] = []
        for ln, samples in self._wait_samples_ms.items():
            if not samples:
                continue
            ss = sorted(samples)
            p99_l = ss[min(len(ss) - 1, max(0, int(0.99 * len(ss)) - 1))]
            per_layer_p99.append((ln, p99_l, len(ss)))
        per_layer_p99.sort(key=lambda x: x[1], reverse=True)
        worst = ", ".join(
            f"{ln}:p99={ms:.2f}ms(n={n})"
            for ln, ms, n in per_layer_p99[:4]
        )
        logger.info(
            "PipelinedDecodeBench wait_for_layer (K=%d, n=%d): "
            "mean=%.3f ms, p50=%.3f ms, p99=%.3f ms, max=%.3f ms. "
            "Worst layers: %s",
            self._k_window,
            n,
            mean,
            p50,
            p99,
            mx,
            worst if worst else "(none)",
        )
        # Steady-state expectation: p99 should be ~0 ms when overlap is
        # working.  Warn if p99 > a few ms so the user knows to raise K.
        if p99 > 2.0:
            logger.warning(
                "PipelinedDecodeBench wait_for_layer p99=%.2f ms > 2 ms — "
                "compute is stalling on KV prefetch. Consider increasing "
                "--kv-k-window (current K=%d).",
                p99,
                self._k_window,
            )
        self._wait_samples_ms.clear()
        self._wait_samples_all_ms.clear()

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
        decode_block_runs: list[tuple[int, int]] = []
        if has_decode_work:
            max_bid = -1
            all_active_blocks: set[int] = set()
            for blocks in self._active_req_blocks.values():
                if blocks:
                    max_bid = max(max_bid, max(blocks))
                    all_active_blocks.update(blocks)
            if max_bid >= 0:
                decode_max_block_id = max_bid
            decode_block_runs = self._block_runs_from_ids(all_active_blocks)

        decode_physical_block_ids = [
            b for lo, hi in decode_block_runs for b in range(lo, hi + 1)
        ] if decode_block_runs else []

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
            decode_block_runs=decode_block_runs,
            decode_physical_block_ids=decode_physical_block_ids,
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._filled_requests.discard(request.request_id)
        self._active_req_blocks.pop(request.request_id, None)
        return False, None
