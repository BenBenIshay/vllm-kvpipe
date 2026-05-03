# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import queue
import threading

import psutil
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class LayerTransferState:
    cpu_buffer: torch.Tensor
    ready_event: torch.cuda.Event | None = None
    save_event: torch.cuda.Event | None = None


class LayerwisePipeliningState:
    """Utility for staging KV cache layers between CPU and GPU asynchronously."""

    def __init__(
        self,
        stream: torch.cuda.Stream | None = None,
        max_pinned_bytes: int | None = None,
    ) -> None:
        self._stream = stream or torch.cuda.Stream()
        self._layers: dict[str, LayerTransferState] = {}

        # --- Pinned memory budget ---
        if max_pinned_bytes is None:
            max_pinned_bytes = int(
                0.8 * psutil.virtual_memory().total
            )
        self._max_pinned_bytes = max_pinned_bytes
        self._pinned_bytes_allocated = 0

        self._delta_queue: queue.Queue[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.cuda.Event,
            ]
        ] = queue.Queue()
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
        return layer_name in self._layers

    @staticmethod
    def _block_dim(kv: torch.Tensor) -> int:
        """Block-dimension index: dim-1 when shape[0]==2, else dim-0."""
        return 1 if kv.shape[0] == 2 else 0

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

    def stage_delta_to_cpu(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_actual_tokens: int | None = None,
    ) -> None:
        """Asynchronously stage only the updated KV slots to pinned CPU memory."""
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

        cpu_delta = torch.empty_like(delta, device="cpu", pin_memory=True)
        cpu_slots = torch.empty_like(slot_mapping, device="cpu", pin_memory=True)
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

        self._delta_queue.put(
            (state.cpu_buffer, cpu_slots, cpu_delta, ready_event)
        )

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

    def prefetch_to_gpu(self, layer_name: str, kv_layer: torch.Tensor) -> None:
        """Asynchronously prefetch the layer KV cache back to GPU memory.

        Handles size mismatches between CPU buffer and GPU target:
        - CPU buffer larger: copy only the leading portion that fits.
        - GPU target larger (sequence grew during decode): copy what we
          have and leave the extra GPU blocks for the model to fill.
        """
        state = self._layers.get(layer_name)
        if state is None:
            return
        if state.ready_event is not None:
            return
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
            gpu_dst.copy_(cpu_buf, non_blocking=True)
            state.ready_event = torch.cuda.Event(blocking=False)
            state.ready_event.record(self._stream)

    def prefetch_to_gpu_block_span(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        block_lo: int,
        block_hi_inclusive: int,
    ) -> None:
        """Prefetch ``cpu_buffer[lo:hi+1]`` into ``kv_layer[lo:hi+1]``."""
        state = self._layers.get(layer_name)
        if state is None:
            return
        if state.ready_event is not None:
            return

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
            gpu_dst[sl].copy_(cpu_buf[sl], non_blocking=True)
            state.ready_event = torch.cuda.Event(blocking=False)
            state.ready_event.record(self._stream)

    def prefetch_to_gpu_block_runs(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        runs: list[tuple[int, int]],
    ) -> None:
        """Prefetch several disjoint block spans from CPU buffer to GPU."""
        if not runs:
            return
        state = self._layers.get(layer_name)
        if state is None:
            return
        if state.ready_event is not None:
            return

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

    def wait_for_saves(self) -> None:
        """Order the current stream after outstanding async saves.

        Also drains the delta queue so that all background CPU-side scatter
        operations have completed before the next step reads from the CPU
        buffers (e.g. during ``prefetch_to_gpu``).
        """
        # Wait for GPU→CPU async copies (stage_to_cpu events).
        for state in self._layers.values():
            if state.save_event is not None:
                torch.cuda.current_stream().wait_event(state.save_event)
                state.save_event = None
        # Wait for background delta scatters to finish writing into CPU
        # buffers so that the next step's prefetch reads up-to-date data.
        self._delta_queue.join()

    def _apply_delta_updates(self) -> None:
        while True:
            cpu_buffer, slot_mapping, cpu_delta, ready_event = (
                self._delta_queue.get()
            )
            ready_event.synchronize()
            # Wrap in inference_mode so in-place updates on inference tensors
            # (created during the forward pass) are permitted in this thread.
            with torch.inference_mode():
                flat = self._kv_to_delta_flat(cpu_buffer)
                if flat.shape[0] == 2:
                    flat[:, slot_mapping, ...] = cpu_delta
                else:
                    flat[slot_mapping, ...] = cpu_delta
            self._delta_queue.task_done()

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
