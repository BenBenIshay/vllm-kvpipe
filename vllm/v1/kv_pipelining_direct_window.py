# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared direct K-tensor window binding for decode-only KV pipelining connectors."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from vllm.attention.layer import AttentionLayerBase
from vllm.config import get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.v1.cpu_kv_table import (
    CpuKvTable,
    _block_dim,
    _kv_layout_blocks_first,
    _per_token_shape,
    measure_hbm_kv_bytes,
)
from vllm.v1.kv_cache_pipelining import LayerwisePipeliningState, PipeliningOrchestrator
from vllm.v1.kv_pipelining_profile import (
    DecodeHookProfile,
    kv_pipelining_profile_enabled,
)
from vllm.v1.kv_pipelining_step import (
    DecodeStepTransferPlan,
    build_prefetch_target_indices,
    extract_attn_metadata_dict,
    find_step_slot_mapping,
    kv_pipelining_batched_d2h_enabled,
    kv_pipelining_step_plan_enabled,
    kv_pipelining_wrap_prefetch_enabled,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext

logger = init_logger(__name__)


class DirectKWindowMixin:
    """Mixin: bind layers to allocator rolling K tensors (no connector WindowPool)."""

    _k_window: int
    _decode_only: bool
    _layer_names: list[str]
    _layer_index: dict[str, int]
    _prefetch_target: list[str | None]
    _orchestrator: PipeliningOrchestrator
    _worker_state: LayerwisePipeliningState | None
    _layer_to_slot: dict[str, int]
    _slot_to_layer: dict[int, str]
    _forward_context: ForwardContext | None
    _has_decode_work: bool
    _kv_caches: dict[str, torch.Tensor] | None
    _decode_phys_blocks: list[int] | None
    _scratch_by_slot: dict[int, torch.Tensor]
    _delta_scratch_by_slot: dict[int, torch.Tensor]
    _use_scratch_prefetch: bool
    _use_delta_scratch: bool
    _hook_profile: DecodeHookProfile | None
    _use_step_plan: bool
    _use_batched_d2h: bool
    _use_wrap_prefetch: bool
    _step_plan: DecodeStepTransferPlan

    def _init_direct_k_window(
        self,
        vllm_config: VllmConfig,
        *,
        k_window: int | None = None,
        decode_only: bool | None = None,
    ) -> None:
        cache_cfg = vllm_config.cache_config
        if decode_only is None:
            decode_only = bool(
                getattr(cache_cfg, "decode_only_pipelining", False)
            )
        self._decode_only = decode_only
        self._k_window = int(
            k_window if k_window is not None else getattr(cache_cfg, "kv_k_window", 0)
        )
        self._layer_names = list(
            get_layers_from_vllm_config(
                vllm_config, AttentionLayerBase, None
            ).keys()
        )
        self._layer_index = {
            name: idx for idx, name in enumerate(self._layer_names)
        }
        num_layers = len(self._layer_names)
        self._use_wrap_prefetch = kv_pipelining_wrap_prefetch_enabled()
        self._prefetch_target_idx = build_prefetch_target_indices(
            num_layers,
            self._k_window,
            wrap=self._use_wrap_prefetch,
        )
        self._prefetch_target = [
            self._layer_names[idx] if idx is not None else None
            for idx in self._prefetch_target_idx
        ]
        if self._use_wrap_prefetch and self._k_window > 0:
            logger.info(
                "DirectKWindow: wrap-around prefetch enabled "
                "(layers %d..%d save → prefetch 0..%d); bootstrap off",
                max(0, num_layers - self._k_window),
                num_layers - 1,
                self._k_window - 1,
            )
        self._bootstrap_layer_names = (
            self._layer_names[: self._k_window] if self._k_window > 0 else []
        )
        self._orchestrator = PipeliningOrchestrator(
            self._layer_names, self._k_window
        )
        self._layer_to_slot = {}
        self._slot_to_layer = {}
        self._forward_context = None
        self._has_decode_work = False
        self._kv_caches = None
        self._decode_phys_blocks = None
        self._scratch_by_slot = {}
        self._delta_scratch_by_slot = {}
        self._use_scratch_prefetch = (
            os.environ.get("VLLM_KVPIPE_SCRATCH_PREFETCH", "0") == "1"
        )
        self._use_delta_scratch = (
            os.environ.get("VLLM_KVPIPE_DELTA_SCRATCH", "1") == "1"
        )
        self._hook_profile = (
            DecodeHookProfile() if kv_pipelining_profile_enabled() else None
        )
        self._use_step_plan = kv_pipelining_step_plan_enabled()
        self._use_batched_d2h = kv_pipelining_batched_d2h_enabled()
        self._step_plan = DecodeStepTransferPlan()

    @staticmethod
    def _effective_decode_n_tokens(
        slot_mapping: torch.Tensor,
        num_actual_tokens: int | None,
    ) -> int:
        """Token count for batched vs n=1 paths (per-layer metadata may report 1)."""
        n = int(slot_mapping.numel())
        if num_actual_tokens is not None and num_actual_tokens > 1:
            return num_actual_tokens
        return n

    def _begin_decode_step_plan(self, forward_context: ForwardContext) -> None:
        """B2: resolve slot/batch tables once at step start (before the layer loop)."""
        if (
            not self._use_step_plan
            or self._worker_state is None
            or not self._has_decode_work
            or self._step_plan.prepared
        ):
            return
        attn = extract_attn_metadata_dict(forward_context)
        if attn is None:
            return
        sm, nat = find_step_slot_mapping(attn, self._layer_names)
        if sm is None:
            return
        n_tok, batched, host_slot = self._worker_state.begin_decode_step_plan(
            sm, nat, enable_batched=self._use_batched_d2h
        )
        self._step_plan.n_tokens = n_tok
        self._step_plan.use_batched = batched
        self._step_plan.host_slot = host_slot
        self._step_plan.slot_mapping = sm
        self._step_plan.num_actual_tokens = nat
        self._step_plan.prepared = True

    def _issue_decode_layer_step(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
    ) -> None:
        """B2: stage + prefetch for one layer (plan already prepared)."""
        if self._worker_state is None or not self._has_decode_work:
            return
        import time

        layer_idx = self._layer_index[layer_name]
        prof = self._hook_profile
        t_save = time.perf_counter() if prof is not None else 0.0
        plan = self._step_plan
        if (
            self._use_batched_d2h
            and plan.use_batched
            and self._worker_state._decode_batch_active
        ):
            t_stage = time.perf_counter()
            self._worker_state.stage_decode_batched_layer(
                layer_idx, layer_name, kv_layer
            )
            stage_ms = (time.perf_counter() - t_stage) * 1000.0
            t_pf = time.perf_counter()
            self._prefetch_after_save_by_index(
                layer_idx, physical_block_ids=self._decode_phys_blocks
            )
            pf_ms = (time.perf_counter() - t_pf) * 1000.0
            if prof is not None:
                prof.fwd.save_stage_ms += stage_ms
                prof.fwd.save_prefetch_ms += pf_ms
        elif (
            plan.n_tokens > 1
            and plan.slot_mapping is not None
            and not self._use_batched_d2h
        ):
            t_stage = time.perf_counter()
            self._worker_state.stage_delta_to_cpu(
                layer_name,
                kv_layer,
                plan.slot_mapping,
                plan.num_actual_tokens,
            )
            stage_ms = (time.perf_counter() - t_stage) * 1000.0
            t_pf = time.perf_counter()
            self._prefetch_after_save_by_index(
                layer_idx, physical_block_ids=self._decode_phys_blocks
            )
            pf_ms = (time.perf_counter() - t_pf) * 1000.0
            if prof is not None:
                prof.fwd.save_stage_ms += stage_ms
                prof.fwd.save_prefetch_ms += pf_ms
        elif plan.host_slot is not None:
            slot = self._layer_to_slot.get(layer_name)
            delta_scratch = (
                self._delta_scratch_by_slot.get(slot)
                if slot is not None and self._use_delta_scratch
                else None
            )
            t_stage = time.perf_counter()
            self._worker_state.stage_decode_n1_layer(
                layer_idx,
                layer_name,
                kv_layer,
                plan.host_slot,
                delta_scratch=delta_scratch,
            )
            stage_ms = (time.perf_counter() - t_stage) * 1000.0
            t_pf = time.perf_counter()
            self._prefetch_after_save_by_index(
                layer_idx, physical_block_ids=self._decode_phys_blocks
            )
            pf_ms = (time.perf_counter() - t_pf) * 1000.0
            if prof is not None:
                prof.fwd.save_stage_ms += stage_ms
                prof.fwd.save_prefetch_ms += pf_ms
        plan.layers_issued += 1
        if prof is not None:
            prof.fwd.save_calls += 1
            prof.fwd.save_host_ms += (time.perf_counter() - t_save) * 1000.0

    def register_kv_caches_direct(
        self,
        kv_caches: dict[str, torch.Tensor],
        *,
        max_model_len: int,
        block_size: int,
        enable_cpu_table: bool = True,
    ) -> None:
        """Discover allocator layout and optionally allocate the upfront CPU table."""
        self._kv_caches = kv_caches
        seen_ptrs: set[int] = set()
        ptr_to_slot: dict[int, int] = {}
        total_bytes = 0
        per_tensor_bytes = 0
        self._layer_to_slot.clear()
        self._slot_to_layer.clear()
        for layer_name, t in kv_caches.items():
            ptr = t.data_ptr()
            slot = ptr_to_slot.get(ptr)
            if slot is None:
                slot = len(ptr_to_slot)
                ptr_to_slot[ptr] = slot
                seen_ptrs.add(ptr)
                nbytes = t.numel() * t.element_size()
                total_bytes += nbytes
                per_tensor_bytes = max(per_tensor_bytes, nbytes)
            self._layer_to_slot[layer_name] = slot

        num_layers = len(kv_caches)
        expected_unique = self._k_window if self._decode_only else self._k_window + 1
        logger.info(
            "DirectKWindow: %d layers, unique GPU KV tensors=%d "
            "(expected %d for K=%d decode_only=%s), "
            "allocator GPU KV=%.2f GiB",
            num_layers,
            len(seen_ptrs),
            expected_unique,
            self._k_window,
            self._decode_only,
            total_bytes / (1 << 30),
        )

        # CpuKvTable is for the decode-only (R1) path only. Hybrid CPU-tier
        # streaming uses HybridCpuStreamer (its own host mirror) instead, so we
        # must NOT also create the CpuKvTable in hybrid mode — both would pin a
        # full host copy and blow the host memory cgroup.
        if (
            enable_cpu_table
            and self._worker_state is not None
            and self._k_window > 0
            and self._decode_only
        ):
            table = CpuKvTable.create(
                kv_caches=kv_caches,
                layer_names=self._layer_names,
                k_window=self._k_window,
                max_model_len=max_model_len,
                block_size=block_size,
            )
            self._worker_state.set_cpu_kv_table(table)
            hbm, _, _ = measure_hbm_kv_bytes(kv_caches)
            logger.info(
                "DirectKWindow: CpuKvTable enabled (DDR/HBM ratio=%.2f)",
                table.table_bytes / hbm if hbm else 0.0,
            )
            if self._use_scratch_prefetch:
                self._init_scratch_buffers(kv_caches, table.max_blocks_per_req)
            if self._use_delta_scratch:
                self._init_delta_scratch_buffers(kv_caches)
                self._worker_state.set_delta_scratch_buffers(
                    self._delta_scratch_by_slot, enabled=True
                )
            self._init_batched_delta_scratch(kv_caches)

    def _init_delta_scratch_buffers(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
        """One-token HBM scratch per rolling slot (D2H staging, not prefetch bounce)."""
        self._delta_scratch_by_slot = {}
        ref = kv_caches[self._layer_names[0]]
        nbytes = 0
        for layer_name, main_t in kv_caches.items():
            slot = self._layer_to_slot[layer_name]
            if slot in self._delta_scratch_by_slot:
                continue
            if _kv_layout_blocks_first(main_t):
                token_shape = main_t[0, :, 0].shape
            elif main_t.shape[0] == 2:
                token_shape = main_t[:, 0, 0].shape
            else:
                token_shape = main_t[0, 0].shape
            scratch = torch.empty(
                token_shape, dtype=main_t.dtype, device=main_t.device
            )
            self._delta_scratch_by_slot[slot] = scratch
            nbytes += scratch.numel() * scratch.element_size()
        logger.info(
            "DirectKWindow: delta D2H scratch %d slots, %.2f KiB total",
            len(self._delta_scratch_by_slot),
            nbytes / 1024,
        )

    def _init_batched_delta_scratch(
        self,
        kv_caches: dict[str, torch.Tensor],
        *,
        max_batch_tokens: int = 128,
    ) -> None:
        """Contiguous GPU/CPU slabs for batched one-token D2H (conc>1 decode)."""
        if self._worker_state is None:
            return
        ref = kv_caches[self._layer_names[0]]
        token_shape = _per_token_shape(ref)
        if _kv_layout_blocks_first(ref):
            gpu = torch.empty(
                (max_batch_tokens, *token_shape),
                dtype=ref.dtype,
                device=ref.device,
            )
            cpu = torch.empty(
                (max_batch_tokens, *token_shape),
                dtype=ref.dtype,
                device="cpu",
                pin_memory=True,
            )
        elif ref.shape[0] == 2:
            gpu = torch.empty(
                (2, max_batch_tokens, *token_shape),
                dtype=ref.dtype,
                device=ref.device,
            )
            cpu = torch.empty(
                (2, max_batch_tokens, *token_shape),
                dtype=ref.dtype,
                device="cpu",
                pin_memory=True,
            )
        else:
            gpu = torch.empty(
                (max_batch_tokens, *token_shape),
                dtype=ref.dtype,
                device=ref.device,
            )
            cpu = torch.empty(
                (max_batch_tokens, *token_shape),
                dtype=ref.dtype,
                device="cpu",
                pin_memory=True,
            )
        self._worker_state.set_batched_delta_scratch(
            gpu, cpu, max_tokens=max_batch_tokens
        )
        nbytes = (gpu.numel() + cpu.numel()) * gpu.element_size()
        logger.info(
            "DirectKWindow: batched delta D2H scratch max_tokens=%d, %.2f KiB",
            max_batch_tokens,
            nbytes / 1024,
        )

    def _init_scratch_buffers(
        self,
        kv_caches: dict[str, torch.Tensor],
        max_blocks_per_req: int,
    ) -> None:
        """Compact per-slot GPU buffers for parallel H2D (disjoint from main KV)."""
        self._scratch_by_slot = {}
        ref = kv_caches[self._layer_names[0]]
        bd = _block_dim(ref)
        nbytes = 0
        for layer_name, main_t in kv_caches.items():
            slot = self._layer_to_slot[layer_name]
            if slot in self._scratch_by_slot:
                continue
            shape = list(main_t.shape)
            shape[bd] = min(max_blocks_per_req, int(main_t.shape[bd]))
            scratch = torch.empty(
                shape, dtype=main_t.dtype, device=main_t.device
            )
            self._scratch_by_slot[slot] = scratch
            nbytes += scratch.numel() * scratch.element_size()
        logger.info(
            "DirectKWindow: scratch prefetch buffers %d slots, %.2f MiB total",
            len(self._scratch_by_slot),
            nbytes / (1 << 20),
        )

    def _bind_layer_direct(
        self, layer_name: str, forward_context: ForwardContext
    ) -> tuple[torch.Tensor, bool, str | None]:
        layer = forward_context.no_compile_layers.get(layer_name)
        if layer is None:
            raise ValueError(f"Layer {layer_name} not found in forward context")
        ve = forward_context.virtual_engine
        kv_layer = layer.kv_cache[ve]

        slot = self._layer_to_slot.get(layer_name)
        if slot is None:
            idx = self._layer_index[layer_name]
            slot = idx % max(1, self._k_window)
            self._layer_to_slot[layer_name] = slot

        current = self._slot_to_layer.get(slot)
        if current == layer_name:
            return kv_layer, False, None

        self._slot_to_layer[slot] = layer_name
        return kv_layer, True, current

    def _cache_decode_phys_blocks(
        self, physical_block_ids: list[int] | None
    ) -> list[int] | None:
        if physical_block_ids:
            self._decode_phys_blocks = physical_block_ids
        elif self._decode_phys_blocks:
            return self._decode_phys_blocks
        return physical_block_ids

    def _clear_decode_step_cache(self) -> None:
        self._decode_phys_blocks = None
        self._step_plan.reset()

    def _prefetch_bootstrap_layers(self) -> None:
        """Prefetch first K layers for the next decode step (overlap with scheduling).

        Skipped when wrap-around prefetch is enabled: layers L-K..L-1 already
        issued H2D for 0..K-1 during the same forward.
        """
        if self._use_wrap_prefetch:
            return
        if (
            self._worker_state is None
            or self._forward_context is None
            or not self._bootstrap_layer_names
        ):
            return
        phys = self._decode_phys_blocks
        if not phys or not self._worker_state.uses_cpu_kv_table():
            return
        for layer_name in self._bootstrap_layer_names:
            prof = self._hook_profile
            state = self._worker_state._layers.get(layer_name)
            if state is not None and state.ready_event is not None:
                if prof is not None:
                    prof.fwd.bootstrap_prefetch_skipped_ready += 1
                continue
            kv_layer, is_new, evicted = self._bind_layer_direct(
                layer_name, self._forward_context
            )
            if not is_new:
                continue
            layer_idx = self._layer_index[layer_name]
            wait_evicted = (
                None
                if self._worker_state.uses_delta_scratch()
                else evicted
            )
            self._worker_state.prefetch_table_blocks_to_gpu(
                layer_name,
                kv_layer,
                phys,
                gpu_wait_save_from_layer=wait_evicted,
                layer_idx=layer_idx,
            )
            if prof is not None:
                prof.fwd.bootstrap_prefetch_issued += 1

    def _prefetch_after_save_by_index(
        self,
        layer_idx: int,
        *,
        physical_block_ids: list[int] | None = None,
        kv_view: torch.Tensor | None = None,
    ) -> None:
        if (
            self._worker_state is None
            or self._forward_context is None
            or self._k_window <= 0
        ):
            return
        next_idx = self._prefetch_target_idx[layer_idx]
        if next_idx is None:
            return
        next_layer = self._layer_names[next_idx]
        kv_layer, is_new, evicted = self._bind_layer_direct(
            next_layer, self._forward_context
        )
        prof = self._hook_profile
        if not is_new:
            if prof is not None:
                prof.fwd.rolling_prefetch_skipped_not_new += 1
            return
        phys = self._cache_decode_phys_blocks(physical_block_ids)
        next_slot = self._layer_to_slot.get(next_layer)
        scratch = (
            self._scratch_by_slot.get(next_slot)
            if next_slot is not None
            else None
        )
        use_scratch = (
            self._use_scratch_prefetch
            and scratch is not None
            and evicted is not None
            and phys
            and self._worker_state.uses_cpu_kv_table()
        )
        if use_scratch:
            self._worker_state.prefetch_table_blocks_to_compact_gpu(
                next_layer, scratch, kv_layer
            )
        elif phys and self._worker_state.uses_cpu_kv_table():
            wait_evicted = (
                None
                if self._worker_state.uses_delta_scratch()
                else evicted
            )
            self._worker_state.prefetch_table_blocks_to_gpu(
                next_layer,
                kv_layer,
                phys,
                gpu_wait_save_from_layer=wait_evicted,
                layer_idx=next_idx,
            )
        else:
            dst = kv_view if kv_view is not None else kv_layer
            self._worker_state.prefetch_to_gpu(
                next_layer,
                dst,
                gpu_wait_save_from_layer=evicted,
            )
        if prof is not None:
            prof.fwd.rolling_prefetch_issued += 1

    def _prefetch_after_save(
        self,
        layer_name: str,
        *,
        physical_block_ids: list[int] | None = None,
        kv_view: torch.Tensor | None = None,
    ) -> None:
        layer_idx = self._layer_index.get(layer_name)
        if layer_idx is None:
            return
        self._prefetch_after_save_by_index(
            layer_idx,
            physical_block_ids=physical_block_ids,
            kv_view=kv_view,
        )

    def _decode_stage_delta_batched_and_prefetch(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
    ) -> None:
        """Batched conc>1: one gather+D2H per layer, then prefetch i+K."""
        if self._worker_state is None:
            return
        import time

        layer_idx = self._layer_index[layer_name]
        prof = self._hook_profile
        t_stage = time.perf_counter()
        self._worker_state.stage_decode_batched_layer(
            layer_idx, layer_name, kv_layer
        )
        stage_ms = (time.perf_counter() - t_stage) * 1000.0
        t_pf = time.perf_counter()
        self._prefetch_after_save_by_index(
            layer_idx, physical_block_ids=self._decode_phys_blocks
        )
        pf_ms = (time.perf_counter() - t_pf) * 1000.0
        if prof is not None:
            prof.fwd.save_stage_ms += stage_ms
            prof.fwd.save_prefetch_ms += pf_ms

    def _decode_stage_delta_and_prefetch_n1(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        host_slot: int,
    ) -> None:
        """Stage n==1 D2H and prefetch layer i+K (scratch path when enabled)."""
        if self._worker_state is None:
            return
        import time

        layer_idx = self._layer_index[layer_name]
        prof = self._hook_profile
        t_stage = time.perf_counter()
        slot = self._layer_to_slot.get(layer_name)
        delta_scratch = (
            self._delta_scratch_by_slot.get(slot)
            if slot is not None and self._use_delta_scratch
            else None
        )
        self._worker_state.stage_decode_n1_layer(
            layer_idx,
            layer_name,
            kv_layer,
            host_slot,
            delta_scratch=delta_scratch,
        )
        stage_ms = (time.perf_counter() - t_stage) * 1000.0
        t_pf = time.perf_counter()
        self._prefetch_after_save_by_index(
            layer_idx, physical_block_ids=self._decode_phys_blocks
        )
        pf_ms = (time.perf_counter() - t_pf) * 1000.0
        if prof is not None:
            prof.fwd.save_stage_ms += stage_ms
            prof.fwd.save_prefetch_ms += pf_ms

    def _decode_save_delta_and_prefetch(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        host_slot: int | None,
        *,
        physical_block_ids: list[int] | None = None,
        skip_delta: bool = False,
    ) -> None:
        """Fast path: stage n==1 delta (optional) + prefetch layer i+K."""
        if self._worker_state is None:
            return
        layer_idx = self._layer_index[layer_name]
        if not skip_delta and host_slot is not None:
            self._decode_stage_delta_host_slot(layer_name, kv_layer, host_slot)
        self._prefetch_after_save_by_index(
            layer_idx, physical_block_ids=physical_block_ids
        )
