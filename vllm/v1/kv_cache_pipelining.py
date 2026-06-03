# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
import queue
import threading
from typing import Any

import psutil
import torch

from vllm.logger import init_logger
from vllm.v1.cpu_kv_table import CpuKvTable

logger = init_logger(__name__)


@dataclass
class LayerTransferState:
    cpu_buffer: torch.Tensor
    # Optional compact CPU backing for sparse physical GPU block IDs.
    # When set, cpu_buffer's block dimension is compact row id, and this list
    # maps compact row -> original GPU physical block id.
    compact_block_ids: list[int] | None = None
    compact_block_ids_gpu: torch.Tensor | None = None
    ready_event: torch.cuda.Event | None = None
    save_event: torch.cuda.Event | None = None
    snap_event: torch.cuda.Event | None = None
    # Delta scatter runs on a background thread; prefetch must not read
    # cpu_buffer until the scatter completes (save_event alone is too early).
    cpu_delta_ready: threading.Event = field(default_factory=threading.Event)
    pending_cpu_delta: bool = False
    cpu_delta_buffer: torch.Tensor | None = None
    cpu_slots_buffer: torch.Tensor | None = None
    cpu_delta_ring: list[torch.Tensor] = field(default_factory=list)
    cpu_delta_ring_idx: int = 0

    def __post_init__(self) -> None:
        # No outstanding delta work until stage_delta_to_cpu clears this.
        self.cpu_delta_ready.set()


class LayerwisePipeliningState:
    """Utility for staging KV cache layers between CPU and GPU asynchronously."""

    def __init__(
        self,
        stream: torch.cuda.Stream | None = None,
        max_pinned_bytes: int | None = None,
    ) -> None:
        self._stream = stream or torch.cuda.Stream()
        self._prefetch_stream = torch.cuda.Stream()
        self._layers: dict[str, LayerTransferState] = {}

        # --- Pinned memory budget ---
        if max_pinned_bytes is None:
            max_pinned_bytes = int(
                0.8 * psutil.virtual_memory().total
            )
        self._max_pinned_bytes = max_pinned_bytes
        self._pinned_bytes_allocated = 0

        self._delta_queue: queue.Queue[Any] = queue.Queue()
        self._prefetch_queue: queue.Queue[
            tuple[
                LayerTransferState,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.cuda.Event,
            ]
        ] = queue.Queue()
        self._delta_worker = threading.Thread(
            target=self._apply_delta_updates, daemon=True
        )
        self._delta_worker.start()
        self._prefetch_worker = threading.Thread(
            target=self._apply_prefetch_updates, daemon=True
        )
        self._prefetch_worker.start()
        self._cpu_kv_table: CpuKvTable | None = None
        self._n1_host_slot: int | None = None
        self._decode_n1_active: bool = False
        self._decode_transfer_synced: bool = False
        self._n1_phys: int = 0
        self._n1_off: int = 0
        self._n1_tbl_block: int = -1
        self._save_event_pool: list[torch.cuda.Event] = []
        self._snap_event_pool: list[torch.cuda.Event] = []
        self._ready_event_pool: list[torch.cuda.Event] = []
        self._delta_scratch_by_slot: dict[int, torch.Tensor] = {}
        self._use_delta_scratch: bool = False
        self._batch_gpu_scratch: torch.Tensor | None = None
        self._batch_cpu_scratch: torch.Tensor | None = None
        self._batch_max_tokens: int = 0
        self._decode_batch_prepared: bool = False
        self._decode_batch_active: bool = False
        self._decode_batch_n: int = 0
        self._batch_phys_gpu: torch.Tensor | None = None
        self._batch_off_gpu: torch.Tensor | None = None
        self._batch_tbl_cpu: torch.Tensor | None = None
        self._batch_off_cpu: torch.Tensor | None = None
        self._step_plan_prepared: bool = False

    # ------------------------------------------------------------------
    # Upfront CPU KV table (decode-only P/D)
    # ------------------------------------------------------------------

    def set_cpu_kv_table(self, table: CpuKvTable) -> None:
        """Attach pre-allocated table; per-layer state points at table rows."""
        self._cpu_kv_table = table
        self._pinned_bytes_allocated += table.table_bytes
        n_layers = len(table.layer_names)
        self._save_event_pool = [
            torch.cuda.Event(blocking=False) for _ in range(n_layers)
        ]
        self._snap_event_pool = [
            torch.cuda.Event(blocking=False) for _ in range(n_layers)
        ]
        self._ready_event_pool = [
            torch.cuda.Event(blocking=False) for _ in range(n_layers)
        ]
        for layer_name, idx in table.layer_to_index.items():
            self._layers[layer_name] = LayerTransferState(
                cpu_buffer=table.buffers[idx]
            )
        logger.info(
            "LayerwisePipeliningState: using CpuKvTable (%.2f GiB pinned)",
            table.table_bytes / (1 << 30),
        )

    def uses_cpu_kv_table(self) -> bool:
        return self._cpu_kv_table is not None

    def set_delta_scratch_buffers(
        self, by_slot: dict[int, torch.Tensor], *, enabled: bool = True
    ) -> None:
        self._delta_scratch_by_slot = by_slot
        self._use_delta_scratch = enabled and bool(by_slot)

    def set_batched_delta_scratch(
        self,
        gpu_scratch: torch.Tensor,
        cpu_scratch: torch.Tensor,
        *,
        max_tokens: int,
    ) -> None:
        self._batch_gpu_scratch = gpu_scratch
        self._batch_cpu_scratch = cpu_scratch
        self._batch_max_tokens = max_tokens

    def uses_delta_scratch(self) -> bool:
        return self._use_delta_scratch

    def register_req_blocks(
        self, req_id: str, physical_block_ids: list[int], device: torch.device
    ) -> int:
        if self._cpu_kv_table is None:
            raise RuntimeError("CpuKvTable not initialized")
        return self._cpu_kv_table.register_physical_blocks(
            req_id, physical_block_ids, device=device
        )

    def release_req_blocks(self, req_id: str) -> None:
        if self._cpu_kv_table is not None:
            self._cpu_kv_table.release_req_slot(req_id)

    def seed_table_from_gpu(
        self,
        layer_name: str,
        kv_gpu: torch.Tensor,
        physical_block_ids: list[int],
    ) -> None:
        """Initial full seed of a layer row from GPU physical blocks."""
        if self._cpu_kv_table is None:
            return
        ev = self._cpu_kv_table.stage_blocks_from_gpu(
            layer_name, kv_gpu, physical_block_ids, self._stream
        )
        state = self._layers.get(layer_name)
        if state is None:
            state = LayerTransferState(
                cpu_buffer=self._cpu_kv_table.layer_buffer(layer_name)
            )
            self._layers[layer_name] = state
        state.save_event = ev

    def resolve_n1_host_slot(
        self,
        slot_mapping: torch.Tensor,
        num_actual_tokens: int | None,
    ) -> int | None:
        """Resolve cache slot once per decode step (avoids 24× GPU sync on n==1)."""
        if num_actual_tokens is not None and num_actual_tokens != 1:
            return None
        if slot_mapping.numel() != 1:
            return None
        if self._n1_host_slot is not None:
            return self._n1_host_slot
        slot_val = int(slot_mapping.view(-1).cpu().item())
        if slot_val < 0:
            return None
        self._n1_host_slot = slot_val
        return self._n1_host_slot

    def clear_n1_host_slot(self) -> None:
        self._n1_host_slot = None
        self._decode_n1_active = False
        self._decode_transfer_synced = False
        self._decode_batch_prepared = False
        self._decode_batch_active = False
        self._decode_batch_n = 0
        self._batch_phys_gpu = None
        self._batch_off_gpu = None
        self._batch_tbl_cpu = None
        self._batch_off_cpu = None
        self.clear_step_plan()

    def clear_step_plan(self) -> None:
        self._step_plan_prepared = False

    def begin_decode_step_plan(
        self,
        slot_mapping: torch.Tensor,
        num_actual_tokens: int | None,
        *,
        enable_batched: bool = True,
    ) -> tuple[int, bool, int | None]:
        """Resolve slots once per forward (B2). Returns (n_tokens, batched, host_slot)."""
        if self._step_plan_prepared:
            n = self._decode_batch_n if self._decode_batch_active else (
                1 if self._decode_n1_active else 0
            )
            return (
                n,
                self._decode_batch_active,
                self._n1_host_slot,
            )
        self._step_plan_prepared = True
        n_tok = int(slot_mapping.numel())
        if num_actual_tokens is not None and num_actual_tokens > 1:
            n_tok = num_actual_tokens
        if n_tok > 1 and enable_batched:
            self.prepare_decode_batch(slot_mapping, num_actual_tokens)
            return (self._decode_batch_n, self._decode_batch_active, None)
        host_slot = self.resolve_n1_host_slot(slot_mapping, num_actual_tokens)
        if host_slot is not None:
            self.begin_decode_n1_step(host_slot)
        return (n_tok, False, host_slot)

    def prepare_decode_batch(
        self,
        slot_mapping: torch.Tensor,
        num_actual_tokens: int | None,
    ) -> None:
        """Resolve all cache slots once per decode step (batched conc>1)."""
        if self._decode_batch_prepared:
            return
        self._decode_batch_prepared = True
        table = self._cpu_kv_table
        if table is None:
            return
        sm = slot_mapping
        if num_actual_tokens is not None and num_actual_tokens > 1:
            sm = sm[:num_actual_tokens]
        if sm.numel() == 0:
            return
        valid = sm >= 0
        if not bool(valid.any()):
            return
        if not bool(valid.all()):
            sm = sm[valid]
        n = int(sm.numel())
        if n == 1:
            slot_val = int(sm.view(-1).cpu().item())
            self.begin_decode_n1_step(slot_val)
            return
        if (
            self._batch_gpu_scratch is None
            or self._batch_cpu_scratch is None
            or n > self._batch_max_tokens
        ):
            return
        page = table.page_size
        slots_cpu = sm.detach().cpu().long()
        phys_cpu = torch.div(slots_cpu, page, rounding_mode="floor")
        off_cpu = slots_cpu % page
        flat = table.physical_to_flat[phys_cpu]
        tbl_cpu = torch.div(flat, page, rounding_mode="floor")
        if (tbl_cpu < 0).any():
            return
        device = self._batch_gpu_scratch.device
        self._batch_phys_gpu = phys_cpu.to(device=device, non_blocking=True)
        self._batch_off_gpu = off_cpu.to(device=device, non_blocking=True)
        self._batch_tbl_cpu = tbl_cpu
        self._batch_off_cpu = off_cpu
        self._decode_batch_n = n
        self._decode_batch_active = True
        self.sync_decode_transfer_stream_once()

    def sync_decode_transfer_stream_once(self) -> None:
        """Wait for compute once per decode step (not per layer)."""
        if not self._decode_transfer_synced:
            self._stream.wait_stream(torch.cuda.current_stream())
            self._decode_transfer_synced = True

    def decode_delta_phys(self) -> int | None:
        """Physical block index for the in-flight n==1 D2H (overlap-aware prefetch)."""
        return self._n1_phys if self._decode_n1_active else None

    def begin_decode_n1_step(self, host_slot: int) -> bool:
        """Precompute phys/off/tbl_block once per decode step (n==1)."""
        if self._decode_n1_active:
            return True
        table = self._cpu_kv_table
        if table is None:
            return False
        tbl_block = table.table_block_for_slot(host_slot)
        if tbl_block is None:
            return False
        page = table.page_size
        self._n1_phys = host_slot // page
        self._n1_off = host_slot % page
        self._n1_tbl_block = tbl_block
        self._n1_host_slot = host_slot
        self._decode_n1_active = True
        self.sync_decode_transfer_stream_once()
        return True

    def stage_decode_n1_layer(
        self,
        layer_idx: int,
        layer_name: str,
        kv_layer: torch.Tensor,
        host_slot: int,
        *,
        delta_scratch: torch.Tensor | None = None,
    ) -> None:
        """Batched-style n==1 D2H: pooled events + wait_stream (no per-layer Event alloc)."""
        if self._cpu_kv_table is None or layer_idx >= len(self._save_event_pool):
            return
        if not self._decode_n1_active and not self.begin_decode_n1_step(host_slot):
            return
        save_ev = self._save_event_pool[layer_idx]
        if delta_scratch is not None and self._use_delta_scratch:
            snap_ev = self._snap_event_pool[layer_idx]
            self._cpu_kv_table.enqueue_n1_delta_via_scratch(
                layer_idx,
                kv_layer,
                delta_scratch,
                self._n1_phys,
                self._n1_off,
                self._n1_tbl_block,
                self._stream,
                save_ev,
                snap_ev,
                sync_compute=False,
            )
        else:
            self._cpu_kv_table.enqueue_n1_delta_layer(
                layer_idx,
                kv_layer,
                self._n1_phys,
                self._n1_off,
                self._n1_tbl_block,
                self._stream,
                save_ev,
                sync_compute=False,
            )
        state = self._layers.get(layer_name)
        if state is None:
            return
        state.save_event = save_ev
        if delta_scratch is not None and self._use_delta_scratch:
            state.snap_event = self._snap_event_pool[layer_idx]
        else:
            state.snap_event = None
        state.pending_cpu_delta = False
        state.cpu_delta_ready.set()

    def stage_decode_batched_layer(
        self,
        layer_idx: int,
        layer_name: str,
        kv_layer: torch.Tensor,
        *,
        delta_scratch: torch.Tensor | None = None,
    ) -> None:
        """Batched n>1 D2H: one gather + one copy per layer (not per sequence)."""
        from vllm.v1.kv_pipelining_step import kv_pipelining_fake_transfer_enabled

        table = self._cpu_kv_table
        if (
            table is None
            or not self._decode_batch_active
            or self._batch_phys_gpu is None
            or self._batch_off_gpu is None
            or self._batch_tbl_cpu is None
            or self._batch_off_cpu is None
            or self._batch_gpu_scratch is None
            or self._batch_cpu_scratch is None
            or layer_idx >= len(self._save_event_pool)
        ):
            return
        save_ev = self._save_event_pool[layer_idx]
        snap_ev = (
            self._snap_event_pool[layer_idx]
            if delta_scratch is not None and self._use_delta_scratch
            else None
        )
        table.enqueue_batched_n1_delta_via_scratch(
            layer_idx,
            kv_layer,
            self._batch_gpu_scratch,
            self._batch_cpu_scratch,
            self._batch_phys_gpu,
            self._batch_off_gpu,
            self._batch_tbl_cpu,
            self._batch_off_cpu,
            self._stream,
            save_ev,
            snap_ev,
            sync_compute=False,
            scatter_queue=self._delta_queue,
        )
        state = self._layers.get(layer_name)
        if state is None:
            return
        state.save_event = save_ev
        state.snap_event = snap_ev
        if kv_pipelining_fake_transfer_enabled():
            state.pending_cpu_delta = False
            state.cpu_delta_ready.set()
        else:
            state.pending_cpu_delta = True
            state.cpu_delta_ready.clear()

    def stage_delta_host_slot_to_cpu(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        host_slot: int,
    ) -> None:
        """Stage one decode token using a host-resolved slot (n==1 fast path)."""
        if self._cpu_kv_table is None:
            return
        layer_idx = self._cpu_kv_table.layer_to_index[layer_name]
        self.stage_decode_n1_layer(layer_idx, layer_name, kv_layer, host_slot)

    def prefetch_table_blocks_to_gpu(
        self,
        layer_name: str,
        kv_gpu: torch.Tensor,
        physical_block_ids: list[int] | None,
        *,
        gpu_wait_save_from_layer: str | None = None,
        layer_idx: int | None = None,
    ) -> None:
        if self._cpu_kv_table is None:
            return
        state = self._layers.get(layer_name)
        if state is None:
            return
        if state.ready_event is not None:
            return
        if layer_idx is None:
            layer_idx = self._cpu_kv_table.layer_to_index.get(layer_name, 0)
        use_parallel_prefetch = self._use_delta_scratch and self._decode_n1_active
        pf_stream = self._prefetch_stream if use_parallel_prefetch else self._stream
        wait_save: torch.cuda.Event | None = None
        wait_snap: torch.cuda.Event | None = None
        if gpu_wait_save_from_layer is not None:
            prev = self._layers.get(gpu_wait_save_from_layer)
            if prev is not None:
                if use_parallel_prefetch and prev.snap_event is not None:
                    wait_snap = prev.snap_event
                elif not use_parallel_prefetch and prev.save_event is not None:
                    wait_save = prev.save_event
        ready_ev = (
            self._ready_event_pool[layer_idx]
            if layer_idx < len(self._ready_event_pool)
            else torch.cuda.Event(blocking=False)
        )
        if wait_snap is not None:
            pf_stream.wait_event(wait_snap)
        self._cpu_kv_table.prefetch_layer_to_gpu(
            layer_name,
            kv_gpu,
            physical_block_ids,
            pf_stream,
            wait_save_event=wait_save,
            ready_event=ready_ev,
            delta_phys=None if use_parallel_prefetch else self.decode_delta_phys(),
        )
        state.ready_event = ready_ev

    def prefetch_table_blocks_to_compact_gpu(
        self,
        layer_name: str,
        kv_compact: torch.Tensor,
        kv_main: torch.Tensor,
    ) -> None:
        """H2D into scratch then merge to main on the transfer stream (one ready event)."""
        if self._cpu_kv_table is None:
            return
        state = self._layers.get(layer_name)
        if state is None or state.ready_event is not None:
            return
        self._ensure_cpu_buffer_ready_for_prefetch(layer_name)
        layer_idx = self._cpu_kv_table.layer_to_index.get(layer_name, 0)
        ready_ev = (
            self._ready_event_pool[layer_idx]
            if layer_idx < len(self._ready_event_pool)
            else torch.cuda.Event(blocking=False)
        )
        self._cpu_kv_table.prefetch_table_blocks_to_compact_gpu(
            layer_name,
            kv_compact,
            self._stream,
            ready_event=None,
        )
        self._cpu_kv_table.merge_compact_gpu_to_main(
            kv_compact, kv_main, stream=self._stream
        )
        ready_ev.record(self._stream)
        state.ready_event = ready_ev

    def merge_compact_kv_into_main(
        self,
        layer_name: str,
        kv_main: torch.Tensor,
        kv_compact: torch.Tensor,
    ) -> None:
        """Legacy compute-stream merge (prefer prefetch_table_blocks_to_compact_gpu)."""
        if self._cpu_kv_table is None:
            return
        self._cpu_kv_table.merge_compact_gpu_to_main(kv_compact, kv_main)

    # ------------------------------------------------------------------
    # Pinned memory tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _buf_bytes(t: torch.Tensor) -> int:
        """Return the number of bytes occupied by a tensor."""
        return t.nelement() * t.element_size()

    def _check_pinned_budget(
        self, new_bytes: int, old_bytes: int = 0
    ) -> None:
        """Raise ``RuntimeError`` if the allocation would exceed the budget.

        ``new_bytes`` is the size of the new buffer.  ``old_bytes`` is the
        size of the buffer being replaced (0 if this is a fresh allocation).
        The counter is updated only when the check passes.
        """
        net = new_bytes - old_bytes
        if net <= 0:
            self._pinned_bytes_allocated += net
            return
        projected = self._pinned_bytes_allocated + net
        if projected > self._max_pinned_bytes:
            raise RuntimeError(
                f"KV pipelining pinned CPU memory would exceed 80% of "
                f"system RAM: {self._pinned_bytes_allocated / 1e9:.1f} GB "
                f"allocated + {net / 1e9:.1f} GB requested = "
                f"{projected / 1e9:.1f} GB > "
                f"{self._max_pinned_bytes / 1e9:.1f} GB limit. "
                f"Reduce concurrent requests or use a machine with more "
                f"RAM."
            )
        self._pinned_bytes_allocated = projected

    @property
    def pinned_bytes_allocated(self) -> int:
        """Total pinned CPU bytes currently tracked by this instance."""
        return self._pinned_bytes_allocated

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _get_or_create_state(
        self, layer_name: str, kv_layer: torch.Tensor
    ) -> LayerTransferState:
        state = self._layers.get(layer_name)
        if state is None:
            new_bytes = self._buf_bytes(kv_layer)
            self._check_pinned_budget(new_bytes)
            cpu_buffer = torch.empty_like(
                kv_layer, device="cpu", pin_memory=True
            )
            state = LayerTransferState(cpu_buffer=cpu_buffer)
            self._layers[layer_name] = state
        elif state.cpu_buffer.shape != kv_layer.shape:
            # The buffer needs to grow (e.g. decode allocated a new block
            # beyond what the prefill slice covered).
            # Preserve existing data by copying into the new buffer.
            old_buf = state.cpu_buffer
            bd = self._block_dim(kv_layer)
            if kv_layer.shape[bd] > old_buf.shape[bd]:
                new_bytes = self._buf_bytes(kv_layer)
                old_bytes = self._buf_bytes(old_buf)
                self._check_pinned_budget(new_bytes, old_bytes)
                # Drain pending delta writes so they land in old_buf
                # before we copy it into the new (larger) buffer.
                self._delta_queue.join()
                new_buf = torch.empty_like(
                    kv_layer, device="cpu", pin_memory=True
                )
                old_blocks = old_buf.shape[bd]
                slicer = [slice(None)] * kv_layer.ndim
                slicer[bd] = slice(0, old_blocks)
                new_buf[tuple(slicer)].copy_(old_buf)
                state.cpu_buffer = new_buf
            # If the new shape is smaller, keep the larger existing buffer.
            # (The caller will slice as needed.)
        return state

    def ensure_cpu_buffer(
        self, layer_name: str, kv_layer: torch.Tensor
    ) -> None:
        """Pre-allocate the CPU buffer at ``kv_layer``'s shape.

        When pipelining inflates the window buffer (more blocks than the
        physical per-layer tensor), the first ``stage_to_cpu`` receives a
        *sliced* view.  If the CPU buffer were created at that smaller size,
        the subsequent ``stage_delta_to_cpu`` (which uses the full window
        buffer) would have to re-create it, losing previously staged data.

        Call this once with the **full** window buffer before the first
        ``stage_to_cpu`` to avoid that issue.
        """
        state = self._layers.get(layer_name)
        if state is not None and state.cpu_buffer.shape == kv_layer.shape:
            return  # already the right size
        new_bytes = self._buf_bytes(kv_layer)
        old_bytes = (
            self._buf_bytes(state.cpu_buffer) if state is not None else 0
        )
        self._check_pinned_budget(new_bytes, old_bytes)
        cpu_buffer = torch.empty_like(
            kv_layer, device="cpu", pin_memory=True
        )
        self._layers[layer_name] = LayerTransferState(cpu_buffer=cpu_buffer)

    def has_layer(self, layer_name: str) -> bool:
        if self._cpu_kv_table is not None:
            return layer_name in self._cpu_kv_table.layer_to_index
        return layer_name in self._layers

    @staticmethod
    def _block_dim(kv: torch.Tensor) -> int:
        """Block-dimension index: dim-1 when shape[0]==2, else dim-0."""
        return 1 if kv.shape[0] == 2 else 0

    @staticmethod
    def _page_size(kv: torch.Tensor) -> int:
        if kv.ndim >= 3 and kv.shape[0] == 2:
            return int(kv.shape[2])
        if kv.ndim >= 3 and kv.shape[1] == 2:
            return int(kv.shape[2])
        return int(kv.shape[1])

    @staticmethod
    def _buffer_matches(buf: torch.Tensor | None, ref: torch.Tensor) -> bool:
        return (
            buf is not None
            and buf.shape == ref.shape
            and buf.dtype == ref.dtype
            and buf.device.type == "cpu"
        )

    def _get_delta_buffers(
        self,
        state: LayerTransferState,
        delta: torch.Tensor,
        slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pinned CPU buffers for a delta update.

        Reuse is safe only when no previous CPU scatter is pending for this
        layer; otherwise the background scatter thread may still read the old
        buffer contents.
        """
        if (
            state.pending_cpu_delta
            or not self._buffer_matches(state.cpu_delta_buffer, delta)
            or not self._buffer_matches(state.cpu_slots_buffer, slots)
        ):
            cpu_delta = self._get_cpu_delta_buffer(state, delta)
            cpu_slots = torch.empty_like(slots, device="cpu", pin_memory=True)
            if not state.pending_cpu_delta:
                state.cpu_delta_buffer = cpu_delta
                state.cpu_slots_buffer = cpu_slots
            return cpu_delta, cpu_slots
        assert state.cpu_delta_buffer is not None
        assert state.cpu_slots_buffer is not None
        return state.cpu_delta_buffer, state.cpu_slots_buffer

    def _get_cpu_delta_buffer(
        self,
        state: LayerTransferState,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        ring_size = 8
        if (
            len(state.cpu_delta_ring) != ring_size
            or any(not self._buffer_matches(buf, delta) for buf in state.cpu_delta_ring)
        ):
            state.cpu_delta_ring = [
                torch.empty_like(delta, device="cpu", pin_memory=True)
                for _ in range(ring_size)
            ]
            state.cpu_delta_ring_idx = 0
        buf = state.cpu_delta_ring[state.cpu_delta_ring_idx]
        state.cpu_delta_ring_idx = (state.cpu_delta_ring_idx + 1) % ring_size
        return buf

    def _compact_shape(
        self,
        kv_layer: torch.Tensor,
        block_count: int,
    ) -> torch.Size:
        shape = list(kv_layer.shape)
        shape[self._block_dim(kv_layer)] = block_count
        return torch.Size(shape)

    def _ensure_compact_state(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        block_ids: list[int],
    ) -> LayerTransferState:
        """Ensure a compact CPU buffer for exactly the requested block ids."""
        block_ids = sorted(set(int(b) for b in block_ids))
        state = self._layers.get(layer_name)
        old_ids = state.compact_block_ids if state is not None else None
        if state is not None and old_ids == block_ids:
            return state

        new_shape = self._compact_shape(kv_layer, len(block_ids))
        new_bytes = kv_layer.element_size()
        for dim in new_shape:
            new_bytes *= int(dim)
        old_bytes = self._buf_bytes(state.cpu_buffer) if state is not None else 0
        self._check_pinned_budget(new_bytes, old_bytes)

        new_buf = torch.empty(
            new_shape, dtype=kv_layer.dtype, device="cpu", pin_memory=True
        )
        if state is not None and old_ids:
            old_pos = {bid: i for i, bid in enumerate(old_ids)}
            bd = self._block_dim(new_buf)
            for new_i, bid in enumerate(block_ids):
                old_i = old_pos.get(bid)
                if old_i is None:
                    continue
                dst = [slice(None)] * new_buf.ndim
                src = [slice(None)] * state.cpu_buffer.ndim
                dst[bd] = slice(new_i, new_i + 1)
                src[bd] = slice(old_i, old_i + 1)
                new_buf[tuple(dst)].copy_(state.cpu_buffer[tuple(src)])
        state = LayerTransferState(
            cpu_buffer=new_buf,
            compact_block_ids=block_ids,
            compact_block_ids_gpu=torch.tensor(
                block_ids, dtype=torch.long, device=kv_layer.device
            ),
        )
        self._layers[layer_name] = state
        return state

    @staticmethod
    def _kv_to_delta_flat(kv: torch.Tensor) -> torch.Tensor:
        """View KV cache as a flat layout for per-slot gather/scatter.

        FlashAttention exposes either canonical ``(2, num_blocks, block_size,
        ...)`` or the stride-permuted ``(num_blocks, 2, block_size, ...)``
        layout.  The old ``shape[0] != 2`` MLA heuristic misclassified the
        latter and corrupted delta staging.
        """
        if kv.ndim < 2:
            raise ValueError(f"Unexpected KV tensor rank for delta ops: {kv.shape}")
        if kv.shape[0] == 2:
            nb, ps = kv.shape[1], kv.shape[2]
            return kv.reshape(2, nb * ps, -1)
        if kv.ndim >= 3 and kv.shape[1] == 2:
            nb, ps = kv.shape[0], kv.shape[2]
            return (
                kv.permute(1, 0, *range(2, kv.ndim))
                .contiguous()
                .reshape(2, nb * ps, -1)
            )
        num_pages, page_size = kv.shape[0], kv.shape[1]
        return kv.reshape(num_pages * page_size, -1)

    def stage_to_cpu(self, layer_name: str, kv_layer: torch.Tensor) -> None:
        """Asynchronously stage the layer KV cache to pinned CPU memory.

        If the sequence has grown since the last stage (decode added tokens),
        the CPU buffer is re-allocated to match the new size.
        """
        state = self._layers.get(layer_name)
        if state is None:
            state = self._get_or_create_state(layer_name, kv_layer)

        cpu_buf = state.cpu_buffer
        if cpu_buf.shape != kv_layer.shape:
            bd = self._block_dim(kv_layer)
            if kv_layer.shape[bd] > cpu_buf.shape[bd]:
                # Sequence grew — need a bigger CPU buffer.
                new_bytes = self._buf_bytes(kv_layer)
                old_bytes = self._buf_bytes(cpu_buf)
                self._check_pinned_budget(new_bytes, old_bytes)
                new_buf = torch.empty_like(
                    kv_layer, device="cpu", pin_memory=True
                )
                # Copy old data into the new buffer's leading slice.
                old_blocks = cpu_buf.shape[bd]
                slicer = [slice(None)] * kv_layer.ndim
                slicer[bd] = slice(0, old_blocks)
                new_buf[tuple(slicer)].copy_(cpu_buf)
                state.cpu_buffer = new_buf
                cpu_buf = new_buf
            else:
                # CPU buffer is larger — copy into leading slice.
                slicer = [slice(None)] * kv_layer.ndim
                slicer[bd] = slice(0, kv_layer.shape[bd])
                cpu_buf = cpu_buf[tuple(slicer)]

        # Synchronize: the compute stream wrote kv_layer (e.g. during
        # attention).  Record an event so self._stream waits for that
        # write to complete before reading.
        sync_event = torch.cuda.Event()
        sync_event.record()  # on the current (compute) stream

        with torch.cuda.stream(self._stream):
            self._stream.wait_event(sync_event)
            cpu_buf.copy_(kv_layer, non_blocking=True)
            state.save_event = torch.cuda.Event(blocking=False)
            state.save_event.record(self._stream)

    def _clamp_block_span(
        self, kv_layer: torch.Tensor, lo: int, hi_inclusive: int
    ) -> tuple[int, int]:
        """Clamp ``[lo, hi_inclusive]`` to valid indices along the block dim."""
        bd = self._block_dim(kv_layer)
        n_blocks = int(kv_layer.shape[bd])
        if n_blocks <= 0:
            return 0, -1
        lo = max(0, min(int(lo), n_blocks - 1))
        hi = max(lo, min(int(hi_inclusive), n_blocks - 1))
        return lo, hi

    def stage_to_cpu_block_span(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        block_lo: int,
        block_hi_inclusive: int,
    ) -> None:
        """Stage ``kv_layer[lo:hi+1]`` (block dim) to the same rows of CPU buffer.

        Used when the paged KV pool indexes rows by physical block id; only
        a contiguous id range should be copied, and it must land at matching
        offsets in the pinned buffer (not at row 0).
        """
        lo, hi = self._clamp_block_span(kv_layer, block_lo, block_hi_inclusive)
        if lo > hi:
            return

        state = self._layers.get(layer_name)
        if state is None:
            state = self._get_or_create_state(layer_name, kv_layer)

        cpu_buf = state.cpu_buffer
        if cpu_buf.shape != kv_layer.shape:
            bd = self._block_dim(kv_layer)
            if kv_layer.shape[bd] > cpu_buf.shape[bd]:
                new_bytes = self._buf_bytes(kv_layer)
                old_bytes = self._buf_bytes(cpu_buf)
                self._check_pinned_budget(new_bytes, old_bytes)
                new_buf = torch.empty_like(
                    kv_layer, device="cpu", pin_memory=True
                )
                old_blocks = cpu_buf.shape[bd]
                slicer = [slice(None)] * kv_layer.ndim
                slicer[bd] = slice(0, old_blocks)
                new_buf[tuple(slicer)].copy_(cpu_buf)
                state.cpu_buffer = new_buf
                cpu_buf = new_buf
            else:
                slicer = [slice(None)] * kv_layer.ndim
                slicer[bd] = slice(0, kv_layer.shape[bd])
                cpu_buf = cpu_buf[tuple(slicer)]

        bd = self._block_dim(kv_layer)
        slicer = [slice(None)] * kv_layer.ndim
        slicer[bd] = slice(lo, hi + 1)
        sl = tuple(slicer)

        sync_event = torch.cuda.Event()
        sync_event.record()

        with torch.cuda.stream(self._stream):
            self._stream.wait_event(sync_event)
            cpu_buf[sl].copy_(kv_layer[sl], non_blocking=True)
            state.save_event = torch.cuda.Event(blocking=False)
            state.save_event.record(self._stream)

    def stage_to_cpu_block_runs(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        runs: list[tuple[int, int]],
    ) -> None:
        """Stage several disjoint inclusive block-id spans along the block dim.

        Used when multiple concurrent requests hold non-contiguous physical
        block rows; copying ``[min,max]`` would include unstaged gaps.
        """
        if not runs:
            return
        bd = self._block_dim(kv_layer)
        effective: list[tuple[int, int]] = []
        for lo, hi in runs:
            cl, ch = self._clamp_block_span(kv_layer, lo, hi)
            if cl <= ch:
                effective.append((cl, ch))
        if not effective:
            return

        state = self._layers.get(layer_name)
        if state is None:
            state = self._get_or_create_state(layer_name, kv_layer)

        cpu_buf = state.cpu_buffer
        if cpu_buf.shape != kv_layer.shape:
            if kv_layer.shape[bd] > cpu_buf.shape[bd]:
                new_bytes = self._buf_bytes(kv_layer)
                old_bytes = self._buf_bytes(cpu_buf)
                self._check_pinned_budget(new_bytes, old_bytes)
                new_buf = torch.empty_like(
                    kv_layer, device="cpu", pin_memory=True
                )
                old_blocks = cpu_buf.shape[bd]
                slicer = [slice(None)] * kv_layer.ndim
                slicer[bd] = slice(0, old_blocks)
                new_buf[tuple(slicer)].copy_(cpu_buf)
                state.cpu_buffer = new_buf
                cpu_buf = new_buf
            else:
                slicer = [slice(None)] * kv_layer.ndim
                slicer[bd] = slice(0, kv_layer.shape[bd])
                cpu_buf = cpu_buf[tuple(slicer)]

        sync_event = torch.cuda.Event()
        sync_event.record()

        with torch.cuda.stream(self._stream):
            self._stream.wait_event(sync_event)
            for lo, hi in effective:
                slicer = [slice(None)] * kv_layer.ndim
                slicer[bd] = slice(lo, hi + 1)
                sl = tuple(slicer)
                cpu_buf[sl].copy_(kv_layer[sl], non_blocking=True)
            state.save_event = torch.cuda.Event(blocking=False)
            state.save_event.record(self._stream)

    def stage_to_cpu_block_runs_compact(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        runs: list[tuple[int, int]],
    ) -> None:
        """Stage sparse physical block runs into compact pinned CPU rows."""
        if not runs:
            return
        bd = self._block_dim(kv_layer)
        block_ids: list[int] = []
        for lo, hi in runs:
            cl, ch = self._clamp_block_span(kv_layer, lo, hi)
            if cl <= ch:
                block_ids.extend(range(cl, ch + 1))
        if not block_ids:
            return
        state = self._ensure_compact_state(layer_name, kv_layer, block_ids)
        id_to_compact = {bid: i for i, bid in enumerate(state.compact_block_ids or [])}

        sync_event = torch.cuda.Event()
        sync_event.record()

        with torch.cuda.stream(self._stream):
            self._stream.wait_event(sync_event)
            for lo, hi in runs:
                cl, ch = self._clamp_block_span(kv_layer, lo, hi)
                if cl > ch:
                    continue
                compact_start = id_to_compact.get(cl)
                compact_end = id_to_compact.get(ch)
                if compact_start is None or compact_end is None:
                    continue
                src = [slice(None)] * kv_layer.ndim
                dst = [slice(None)] * state.cpu_buffer.ndim
                src[bd] = slice(cl, ch + 1)
                dst[bd] = slice(compact_start, compact_end + 1)
                state.cpu_buffer[tuple(dst)].copy_(
                    kv_layer[tuple(src)], non_blocking=True
                )
            state.save_event = torch.cuda.Event(blocking=False)
            state.save_event.record(self._stream)

    def stage_delta_to_cpu(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_actual_tokens: int | None = None,
    ) -> None:
        """Asynchronously stage only the updated KV slots to pinned CPU memory."""
        if self._cpu_kv_table is not None:
            from vllm.v1.kv_pipelining_step import kv_pipelining_batched_d2h_enabled

            n_tok = int(slot_mapping.numel())
            if num_actual_tokens is not None and num_actual_tokens > 1:
                n_tok = num_actual_tokens
            if n_tok > 1 and kv_pipelining_batched_d2h_enabled():
                if not self._decode_batch_prepared:
                    self.prepare_decode_batch(slot_mapping, num_actual_tokens)
                if self._decode_batch_active:
                    layer_idx = self._cpu_kv_table.layer_to_index[layer_name]
                    self.stage_decode_batched_layer(
                        layer_idx, layer_name, kv_layer
                    )
                    return
            host_slot = self.resolve_n1_host_slot(slot_mapping, num_actual_tokens)
            if host_slot is not None and (
                num_actual_tokens is None or num_actual_tokens == 1
            ):
                layer_idx = self._cpu_kv_table.layer_to_index[layer_name]
                self.stage_decode_n1_layer(
                    layer_idx, layer_name, kv_layer, host_slot
                )
                return
            self._stage_delta_via_table(
                layer_name,
                kv_layer,
                slot_mapping,
                num_actual_tokens,
                host_slot=host_slot,
            )
            return
        state = self._layers.get(layer_name)
        if state is None or state.compact_block_ids is None:
            state = self._get_or_create_state(layer_name, kv_layer)
        slot_mapping = slot_mapping
        if num_actual_tokens is not None:
            slot_mapping = slot_mapping[:num_actual_tokens]
        if slot_mapping.numel() == 0:
            return
        # Attn metadata may carry padding/sentinel entries (e.g. -1) in
        # slot_mapping. These are invalid cache slots and must never be used
        # for index_select/scatter, especially at higher concurrency where
        # mixed batches often contain padded rows.
        valid_slots = slot_mapping >= 0
        if not bool(valid_slots.any()):
            return
        if not bool(valid_slots.all()):
            slot_mapping = slot_mapping[valid_slots]

        if state.compact_block_ids is not None:
            page_size = self._page_size(kv_layer)
            gpu_blocks = torch.div(slot_mapping, page_size, rounding_mode="floor")
            offsets = slot_mapping % page_size
            ids_gpu = state.compact_block_ids_gpu
            if ids_gpu is None or ids_gpu.numel() == 0:
                return
            # Map original physical block id -> compact CPU row fully on GPU.
            # The active block set is small, so a broadcast compare avoids a
            # per-layer GPU->CPU sync in the decode hot path.
            matches = gpu_blocks.unsqueeze(-1) == ids_gpu.unsqueeze(0)
            compact_blocks = matches.to(torch.long).argmax(dim=-1)
            compact_slots = compact_blocks.to(slot_mapping.dtype) * page_size + offsets

            flat = self._kv_to_delta_flat(kv_layer)
            if flat.shape[0] == 2:
                delta = flat.index_select(1, slot_mapping)
            else:
                delta = flat.index_select(0, slot_mapping)

            sync_event = torch.cuda.Event()
            sync_event.record()

            cpu_delta = torch.empty_like(delta, device="cpu", pin_memory=True)
            with torch.cuda.stream(self._stream):
                self._stream.wait_event(sync_event)
                cpu_delta.copy_(delta, non_blocking=True)
                state.save_event = torch.cuda.Event(blocking=False)
                state.save_event.record(self._stream)
                ready_event = torch.cuda.Event(blocking=False)
                ready_event.record(self._stream)

            state.pending_cpu_delta = True
            state.cpu_delta_ready.clear()
            self._delta_queue.put(
                (
                    layer_name,
                    state.cpu_buffer,
                    compact_slots.detach(),
                    cpu_delta,
                    ready_event,
                )
            )
            return

        flat = self._kv_to_delta_flat(kv_layer)
        if flat.shape[0] == 2:
            delta = flat.index_select(1, slot_mapping)
        else:
            delta = flat.index_select(0, slot_mapping)

        # Synchronize: index_select and slot_mapping live on the compute
        # stream.  Record an event so self._stream waits for them before
        # copying to CPU.
        sync_event = torch.cuda.Event()
        sync_event.record()  # on the current (compute) stream

        cpu_delta, cpu_slots = self._get_delta_buffers(
            state, delta, slot_mapping
        )
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(sync_event)
            cpu_delta.copy_(delta, non_blocking=True)
            cpu_slots.copy_(slot_mapping, non_blocking=True)
            # Mirror stage_to_cpu: wait_for_layer_save / shared K+1 GPU tensors
            # rely on save_event. Decode almost always uses this delta path, not
            # full stage_to_cpu; without recording save_event, prefetch could
            # overwrite a physical tensor before async delta staging finished.
            # Record before ready_event so the background thread still sees
            # ready_event after all copy-stream work above.
            state.save_event = torch.cuda.Event(blocking=False)
            state.save_event.record(self._stream)
            ready_event = torch.cuda.Event(blocking=False)
            ready_event.record(self._stream)

        state.pending_cpu_delta = True
        state.cpu_delta_ready.clear()

        self._delta_queue.put(
            (layer_name, state.cpu_buffer, cpu_slots, cpu_delta, ready_event)
        )

    def _stage_delta_via_table(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_actual_tokens: int | None = None,
        *,
        host_slot: int | None = None,
    ) -> None:
        assert self._cpu_kv_table is not None
        packed = self._cpu_kv_table.stage_delta(
            layer_name,
            kv_layer,
            slot_mapping,
            self._stream,
            num_actual_tokens=num_actual_tokens,
            host_slot=host_slot,
        )
        if packed is None:
            return
        state = self._layers.get(layer_name)
        if state is None:
            buf = self._cpu_kv_table.layer_buffer(layer_name)
            state = LayerTransferState(cpu_buffer=buf)
            self._layers[layer_name] = state

        if isinstance(packed, torch.cuda.Event):
            state.save_event = packed
            state.pending_cpu_delta = False
            state.cpu_delta_ready.set()
            return

        save_event, ready_event, cpu_buf, cpu_slots, cpu_delta = packed
        state.cpu_buffer = cpu_buf
        state.save_event = save_event
        state.pending_cpu_delta = True
        state.cpu_delta_ready.clear()
        page = self._cpu_kv_table.page_size

        def _table_scatter_task() -> None:
            ready_event.synchronize()
            slots_cpu = cpu_slots.contiguous()
            blocks = torch.div(slots_cpu, page, rounding_mode="floor")
            offsets = slots_cpu % page
            with torch.inference_mode():
                if cpu_buf.shape[0] == 2:
                    cpu_buf[:, blocks, offsets] = cpu_delta
                elif cpu_buf.ndim >= 3 and cpu_buf.shape[1] == 2:
                    cpu_buf[blocks, :, offsets] = cpu_delta
                else:
                    cpu_buf[blocks, offsets] = cpu_delta
            state.pending_cpu_delta = False
            state.cpu_delta_ready.set()
            with torch.cuda.stream(self._stream):
                save_event.record(self._stream)

        self._delta_queue.put(_table_scatter_task)

    def prefetch_delta_to_gpu(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_actual_tokens: int | None = None,
    ) -> None:
        """Asynchronously prefetch only the updated KV slots from CPU to GPU."""
        state = self._layers.get(layer_name)
        if state is None:
            return
        if state.ready_event is not None:
            return
        slot_mapping = slot_mapping
        if num_actual_tokens is not None:
            slot_mapping = slot_mapping[:num_actual_tokens]
        if slot_mapping.numel() == 0:
            return
        valid_slots = slot_mapping >= 0
        if not bool(valid_slots.any()):
            return
        if not bool(valid_slots.all()):
            slot_mapping = slot_mapping[valid_slots]

        cpu_slots = torch.empty_like(slot_mapping, device="cpu", pin_memory=True)
        with torch.cuda.stream(self._stream):
            cpu_slots.copy_(slot_mapping, non_blocking=True)
            gpu_slots = slot_mapping.clone()
            slots_ready = torch.cuda.Event(blocking=False)
            slots_ready.record(self._stream)

        self._prefetch_queue.put(
            (state, kv_layer, cpu_slots, gpu_slots, slots_ready)
        )

    def prefetch_to_gpu(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        *,
        gpu_wait_save_from_layer: str | None = None,
    ) -> None:
        """Asynchronously prefetch the layer KV cache back to GPU memory.

        Handles size mismatches between CPU buffer and GPU target:
        - CPU buffer larger: copy only the leading portion that fits.
        - GPU target larger (sequence grew during decode): copy what we
          have and leave the extra GPU blocks for the model to fill.

        ``gpu_wait_save_from_layer``: when reusing a shared physical tensor,
        wait for that layer's async GPU→CPU save on the transfer stream
        instead of blocking the compute stream in the connector.
        """
        state = self._layers.get(layer_name)
        if state is None:
            return
        if state.ready_event is not None:
            return
        self._ensure_cpu_buffer_ready_for_prefetch(layer_name)
        cpu_buf = state.cpu_buffer
        gpu_dst = kv_layer
        if cpu_buf.shape != gpu_dst.shape:
            bd = self._block_dim(gpu_dst)
            common_blocks = min(cpu_buf.shape[bd], gpu_dst.shape[bd])
            slicer = [slice(None)] * gpu_dst.ndim
            slicer[bd] = slice(0, common_blocks)
            cpu_buf = cpu_buf[tuple(slicer)]
            gpu_dst = gpu_dst[tuple(slicer)]
        with torch.cuda.stream(self._stream):
            self._stream_wait_layer_save(gpu_wait_save_from_layer)
            gpu_dst.copy_(cpu_buf, non_blocking=True)
            state.ready_event = torch.cuda.Event(blocking=False)
            state.ready_event.record(self._stream)

    def prefetch_to_gpu_block_span(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        block_lo: int,
        block_hi_inclusive: int,
        *,
        gpu_wait_save_from_layer: str | None = None,
    ) -> None:
        """Prefetch ``cpu_buffer[lo:hi+1]`` into ``kv_layer[lo:hi+1]``."""
        state = self._layers.get(layer_name)
        if state is None:
            return
        if state.ready_event is not None:
            return
        self._ensure_cpu_buffer_ready_for_prefetch(layer_name)

        lo, hi = self._clamp_block_span(kv_layer, block_lo, block_hi_inclusive)
        if lo > hi:
            return

        cpu_buf = state.cpu_buffer
        gpu_dst = kv_layer
        bd = self._block_dim(gpu_dst)
        common_blocks = min(cpu_buf.shape[bd], gpu_dst.shape[bd])
        hi_eff = min(hi, common_blocks - 1)
        if lo > hi_eff:
            return

        slicer = [slice(None)] * gpu_dst.ndim
        slicer[bd] = slice(lo, hi_eff + 1)
        sl = tuple(slicer)

        with torch.cuda.stream(self._stream):
            self._stream_wait_layer_save(gpu_wait_save_from_layer)
            gpu_dst[sl].copy_(cpu_buf[sl], non_blocking=True)
            state.ready_event = torch.cuda.Event(blocking=False)
            state.ready_event.record(self._stream)

    def prefetch_to_gpu_block_runs(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        runs: list[tuple[int, int]],
        *,
        gpu_wait_save_from_layer: str | None = None,
    ) -> None:
        """Prefetch several disjoint block spans from CPU buffer to GPU."""
        if not runs:
            return
        state = self._layers.get(layer_name)
        if state is None:
            return
        if state.ready_event is not None:
            return
        self._ensure_cpu_buffer_ready_for_prefetch(layer_name)

        bd = self._block_dim(kv_layer)
        effective: list[tuple[int, int]] = []
        for lo, hi in runs:
            cl, ch = self._clamp_block_span(kv_layer, lo, hi)
            if cl <= ch:
                effective.append((cl, ch))
        if not effective:
            return

        cpu_buf = state.cpu_buffer
        gpu_dst = kv_layer
        common_blocks = min(cpu_buf.shape[bd], gpu_dst.shape[bd])
        if common_blocks <= 0:
            return

        with torch.cuda.stream(self._stream):
            self._stream_wait_layer_save(gpu_wait_save_from_layer)
            for lo, hi in effective:
                hi_eff = min(hi, common_blocks - 1)
                if lo > hi_eff:
                    continue
                slicer = [slice(None)] * gpu_dst.ndim
                slicer[bd] = slice(lo, hi_eff + 1)
                sl = tuple(slicer)
                gpu_dst[sl].copy_(cpu_buf[sl], non_blocking=True)
            state.ready_event = torch.cuda.Event(blocking=False)
            state.ready_event.record(self._stream)

    def prefetch_to_gpu_block_runs_compact(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        runs: list[tuple[int, int]],
        *,
        gpu_wait_save_from_layer: str | None = None,
    ) -> None:
        """Prefetch compact CPU rows back into sparse physical GPU blocks."""
        if not runs:
            return
        state = self._layers.get(layer_name)
        if state is None:
            return
        if state.ready_event is not None:
            return
        self._ensure_cpu_buffer_ready_for_prefetch(layer_name)
        if state.compact_block_ids is None:
            self.prefetch_to_gpu_block_runs(
                layer_name,
                kv_layer,
                runs,
                gpu_wait_save_from_layer=gpu_wait_save_from_layer,
            )
            return

        bd = self._block_dim(kv_layer)
        id_to_compact = {
            bid: i for i, bid in enumerate(state.compact_block_ids or [])
        }
        effective: list[tuple[int, int, int, int]] = []
        for lo, hi in runs:
            cl, ch = self._clamp_block_span(kv_layer, lo, hi)
            if cl > ch:
                continue
            compact_start = id_to_compact.get(cl)
            compact_end = id_to_compact.get(ch)
            if compact_start is None or compact_end is None:
                continue
            effective.append((cl, ch, compact_start, compact_end))
        if not effective:
            return

        with torch.cuda.stream(self._stream):
            self._stream_wait_layer_save(gpu_wait_save_from_layer)
            for lo, hi, compact_start, compact_end in effective:
                src = [slice(None)] * state.cpu_buffer.ndim
                dst = [slice(None)] * kv_layer.ndim
                src[bd] = slice(compact_start, compact_end + 1)
                dst[bd] = slice(lo, hi + 1)
                kv_layer[tuple(dst)].copy_(
                    state.cpu_buffer[tuple(src)], non_blocking=True
                )
            state.ready_event = torch.cuda.Event(blocking=False)
            state.ready_event.record(self._stream)

    def wait_for_layer(self, layer_name: str) -> None:
        """Ensure the prefetched KV layer is visible to the current stream."""
        state = self._layers.get(layer_name)
        if state is None or state.ready_event is None:
            return
        torch.cuda.current_stream().wait_event(state.ready_event)
        state.ready_event = None

    def wait_for_layer_save(self, layer_name: str) -> None:
        """Wait for a specific layer's async CPU staging to complete.

        This is needed when physical GPU tensors are shared among layers
        (K+1 tensor layout): before a new layer writes to a tensor, any
        in-flight GPU→CPU copy from the tensor's previous occupant must
        finish first.
        """
        state = self._layers.get(layer_name)
        if state is None or state.save_event is None:
            return
        torch.cuda.current_stream().wait_event(state.save_event)
        state.save_event = None

    def wait_for_saves(self, *, deferred: bool = False) -> None:
        """Order the current stream after outstanding async saves.

        Also drains the delta queue so that all background CPU-side scatter
        operations have completed before the next step reads from the CPU
        buffers (e.g. during ``prefetch_to_gpu``).

        When ``deferred=True`` (decode-only steps), do not block the compute
        stream here.  Prefetch paths wait on the transfer stream for prior
        GPU saves and per-layer CPU delta readiness instead.
        """
        self.clear_n1_host_slot()
        if deferred:
            return
        # Wait for GPU→CPU async copies (stage_to_cpu events).
        for state in self._layers.values():
            if state.save_event is not None:
                torch.cuda.current_stream().wait_event(state.save_event)
                state.save_event = None
        # Wait for background delta scatters to finish writing into CPU
        # buffers so that the next step's prefetch reads up-to-date data.
        self._delta_queue.join()
        for state in self._layers.values():
            state.pending_cpu_delta = False
            state.cpu_delta_ready.set()

    def _apply_delta_updates(self) -> None:
        while True:
            item = self._delta_queue.get()
            if callable(item):
                item()
                self._delta_queue.task_done()
                continue
            (
                layer_name,
                cpu_buffer,
                slot_mapping,
                cpu_delta,
                ready_event,
            ) = item
            ready_event.synchronize()
            if slot_mapping.device.type != "cpu":
                slot_mapping = slot_mapping.cpu()
            with torch.inference_mode():
                flat = self._kv_to_delta_flat(cpu_buffer)
                if flat.shape[0] == 2:
                    flat[:, slot_mapping, ...] = cpu_delta
                else:
                    flat[slot_mapping, ...] = cpu_delta
            state = self._layers.get(layer_name)
            if state is not None:
                state.pending_cpu_delta = False
                state.cpu_delta_ready.set()
            self._delta_queue.task_done()

    def _ensure_cpu_buffer_ready_for_prefetch(self, layer_name: str) -> None:
        """Block until a queued delta scatter has finished updating CPU."""
        state = self._layers.get(layer_name)
        if state is None or not state.pending_cpu_delta:
            return
        state.cpu_delta_ready.wait()

    def _stream_wait_layer_save(self, layer_name: str | None) -> None:
        """Wait for a layer's GPU→CPU copy on the transfer stream (not compute)."""
        if not layer_name:
            return
        state = self._layers.get(layer_name)
        if state is None or state.save_event is None:
            return
        self._stream.wait_event(state.save_event)
        state.save_event = None

    def _apply_prefetch_updates(self) -> None:
        while True:
            (
                state,
                kv_layer,
                cpu_slots,
                gpu_slots,
                slots_ready,
            ) = self._prefetch_queue.get()
            slots_ready.synchronize()
            # Wrap in inference_mode so operations on inference tensors
            # (created during the forward pass) are permitted in this thread.
            with torch.inference_mode():
                cpu_flat = self._kv_to_delta_flat(state.cpu_buffer)
                if cpu_flat.shape[0] == 2:
                    cpu_delta = cpu_flat[:, cpu_slots, ...]
                else:
                    cpu_delta = cpu_flat[cpu_slots, ...]

                with torch.cuda.stream(self._stream):
                    gpu_delta = cpu_delta.to(device=kv_layer.device, non_blocking=True)
                    gpu_flat = self._kv_to_delta_flat(kv_layer)
                    if gpu_flat.shape[0] == 2:
                        gpu_flat.index_copy_(1, gpu_slots, gpu_delta)
                    else:
                        gpu_flat.index_copy_(0, gpu_slots, gpu_delta)
                    state.ready_event = torch.cuda.Event(blocking=False)
                    state.ready_event.record(self._stream)

            self._prefetch_queue.task_done()


class PipeliningOrchestrator:
    """Async helper to schedule layer prefetching with a fixed K-window."""

    def __init__(self, layer_names: list[str], k_window: int) -> None:
        self._layer_names = layer_names
        self._k_window = max(k_window, 0)
        self._layer_index = {name: idx for idx, name in enumerate(layer_names)}

    def plan_prefetch(self, current_layer: str) -> list[str]:
        if self._k_window == 0:
            return []
        idx = self._layer_index.get(current_layer)
        if idx is None:
            return []
        start = idx + 1
        end = min(len(self._layer_names), start + self._k_window)
        return self._layer_names[start:end]

    async def wait_and_plan(
        self, layer_queue: asyncio.Queue[str]
    ) -> asyncio.Queue[list[str]]:
        """Consume layer names and produce prefetch plans asynchronously."""
        output: asyncio.Queue[list[str]] = asyncio.Queue()
        while True:
            layer_name = await layer_queue.get()
            await output.put(self.plan_prefetch(layer_name))
            layer_queue.task_done()
        return output
