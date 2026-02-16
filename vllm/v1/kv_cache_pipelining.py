# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import queue
import threading

import torch


@dataclass
class LayerTransferState:
    cpu_buffer: torch.Tensor
    ready_event: torch.cuda.Event | None = None
    save_event: torch.cuda.Event | None = None


class LayerwisePipeliningState:
    """Utility for staging KV cache layers between CPU and GPU asynchronously."""

    def __init__(self, stream: torch.cuda.Stream | None = None) -> None:
        self._stream = stream or torch.cuda.Stream()
        self._layers: dict[str, LayerTransferState] = {}
        self._delta_queue: queue.Queue[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.cuda.Event,
                bool,
            ]
        ] = queue.Queue()
        self._prefetch_queue: queue.Queue[
            tuple[
                LayerTransferState,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.cuda.Event,
                bool,
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

    def _get_or_create_state(
        self, layer_name: str, kv_layer: torch.Tensor
    ) -> LayerTransferState:
        state = self._layers.get(layer_name)
        if state is None or state.cpu_buffer.shape != kv_layer.shape:
            cpu_buffer = torch.empty_like(
                kv_layer, device="cpu", pin_memory=True
            )
            state = LayerTransferState(cpu_buffer=cpu_buffer)
            self._layers[layer_name] = state
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

        with torch.cuda.stream(self._stream):
            cpu_buf.copy_(kv_layer, non_blocking=True)
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

        is_mla = kv_layer.shape[0] != 2
        if is_mla:
            num_pages, page_size = kv_layer.shape[0], kv_layer.shape[1]
            kv_view = kv_layer.reshape(num_pages * page_size, -1)
            delta = kv_view.index_select(0, slot_mapping)
        else:
            num_pages, page_size = kv_layer.shape[1], kv_layer.shape[2]
            kv_view = kv_layer.reshape(2, num_pages * page_size, -1)
            delta = kv_view.index_select(1, slot_mapping)

        cpu_delta = torch.empty_like(delta, device="cpu", pin_memory=True)
        cpu_slots = torch.empty_like(slot_mapping, device="cpu", pin_memory=True)
        with torch.cuda.stream(self._stream):
            cpu_delta.copy_(delta, non_blocking=True)
            cpu_slots.copy_(slot_mapping, non_blocking=True)
            ready_event = torch.cuda.Event(blocking=False)
            ready_event.record(self._stream)

        self._delta_queue.put(
            (state.cpu_buffer, cpu_slots, cpu_delta, ready_event, is_mla)
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

        is_mla = kv_layer.shape[0] != 2
        cpu_slots = torch.empty_like(slot_mapping, device="cpu", pin_memory=True)
        with torch.cuda.stream(self._stream):
            cpu_slots.copy_(slot_mapping, non_blocking=True)
            gpu_slots = slot_mapping.clone()
            slots_ready = torch.cuda.Event(blocking=False)
            slots_ready.record(self._stream)

        self._prefetch_queue.put(
            (state, kv_layer, cpu_slots, gpu_slots, slots_ready, is_mla)
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
        """Order the current stream after outstanding async saves."""
        for state in self._layers.values():
            if state.save_event is not None:
                torch.cuda.current_stream().wait_event(state.save_event)
                state.save_event = None

    def _apply_delta_updates(self) -> None:
        while True:
            cpu_buffer, slot_mapping, cpu_delta, ready_event, is_mla = (
                self._delta_queue.get()
            )
            ready_event.synchronize()
            if is_mla:
                num_pages, page_size = cpu_buffer.shape[0], cpu_buffer.shape[1]
                cpu_view = cpu_buffer.reshape(num_pages * page_size, -1)
                cpu_view[slot_mapping, ...] = cpu_delta
            else:
                num_pages, page_size = cpu_buffer.shape[1], cpu_buffer.shape[2]
                cpu_view = cpu_buffer.reshape(2, num_pages * page_size, -1)
                cpu_view[:, slot_mapping, ...] = cpu_delta
            self._delta_queue.task_done()

    def _apply_prefetch_updates(self) -> None:
        while True:
            (
                state,
                kv_layer,
                cpu_slots,
                gpu_slots,
                slots_ready,
                is_mla,
            ) = self._prefetch_queue.get()
            slots_ready.synchronize()
            if is_mla:
                num_pages, page_size = state.cpu_buffer.shape[0], state.cpu_buffer.shape[1]
                cpu_view = state.cpu_buffer.reshape(num_pages * page_size, -1)
                cpu_delta = cpu_view[cpu_slots, ...]
            else:
                num_pages, page_size = state.cpu_buffer.shape[1], state.cpu_buffer.shape[2]
                cpu_view = state.cpu_buffer.reshape(2, num_pages * page_size, -1)
                cpu_delta = cpu_view[:, cpu_slots, ...]

            with torch.cuda.stream(self._stream):
                gpu_delta = cpu_delta.to(device=kv_layer.device, non_blocking=True)
                if is_mla:
                    kv_view = kv_layer.reshape(num_pages * page_size, -1)
                    kv_view.index_copy_(0, gpu_slots, gpu_delta)
                else:
                    kv_view = kv_layer.reshape(2, num_pages * page_size, -1)
                    kv_view.index_copy_(1, gpu_slots, gpu_delta)
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
