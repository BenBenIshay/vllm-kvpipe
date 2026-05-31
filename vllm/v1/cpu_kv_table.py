# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Upfront CPU KV table for decode-only layer-wise pipelining (P/D decode node)."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import torch

from vllm.utils.math_utils import cdiv

logger = logging.getLogger(__name__)


def measure_hbm_kv_bytes(kv_caches: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    """Return (unique_tensor_bytes, num_unique_tensors, per_tensor_max_bytes)."""
    seen: set[int] = set()
    total = 0
    per_tensor_max = 0
    for t in kv_caches.values():
        ptr = t.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        nbytes = t.numel() * t.element_size()
        total += nbytes
        per_tensor_max = max(per_tensor_max, nbytes)
    return total, len(seen), per_tensor_max


def _block_dim(kv: torch.Tensor) -> int:
    return 1 if kv.ndim >= 1 and kv.shape[0] == 2 else 0


def _page_size(kv: torch.Tensor) -> int:
    if kv.ndim >= 3 and kv.shape[0] == 2:
        return int(kv.shape[2])
    if kv.ndim >= 3 and kv.shape[1] == 2:
        return int(kv.shape[2])
    return int(kv.shape[1])


def _kv_to_delta_flat(kv: torch.Tensor) -> torch.Tensor:
    if kv.ndim < 2:
        raise ValueError(f"Unexpected KV tensor rank: {kv.shape}")
    if kv.shape[0] == 2:
        nb, ps = kv.shape[1], kv.shape[2]
        return kv.reshape(2, nb * ps, -1)
    if kv.ndim >= 3 and kv.shape[1] == 2:
        nb, ps = kv.shape[0], kv.shape[2]
        return kv.permute(1, 0, 2, *range(3, kv.ndim)).reshape(2, nb * ps, -1)
    nb, ps = kv.shape[0], kv.shape[1]
    return kv.reshape(nb * ps, -1)


@dataclass
class CpuKvTable:
    """Pinned CPU table: one buffer per layer, partitioned by request slot."""

    num_layers: int
    k_window: int
    hbm_kv_bytes: int
    table_bytes: int
    block_size: int
    page_size: int
    max_blocks_per_req: int
    num_req_slots: int
    num_blocks_per_layer_row: int
    layer_names: list[str]
    layer_to_index: dict[str, int]
    buffers: list[torch.Tensor]
    physical_to_flat: torch.Tensor
    table_block_gpu: torch.Tensor | None = None
    ordered_phys: list[int] = field(default_factory=list)
    ordered_table_blocks: list[int] = field(default_factory=list)
    req_id_to_slot: dict[str, int] = field(default_factory=dict)
    req_id_to_phys_blocks: dict[str, list[int]] = field(default_factory=dict)
    _free_slots: list[int] = field(default_factory=list)
    _compute_sync: torch.cuda.Event | None = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        *,
        kv_caches: dict[str, torch.Tensor],
        layer_names: list[str],
        k_window: int,
        max_model_len: int,
        block_size: int,
    ) -> CpuKvTable:
        if k_window <= 0:
            raise ValueError("k_window must be positive for CpuKvTable")
        hbm_kv_bytes, num_unique, _per = measure_hbm_kv_bytes(kv_caches)
        num_layers = len(layer_names)
        if num_layers == 0:
            raise ValueError("layer_names must not be empty")
        table_bytes = (num_layers * hbm_kv_bytes) // k_window
        ref_name = layer_names[0]
        ref = kv_caches[ref_name]
        page_size = _page_size(ref)
        bd = _block_dim(ref)
        num_blocks_per_layer_row = int(ref.shape[bd])
        max_blocks_per_req = cdiv(max_model_len, block_size)
        if max_blocks_per_req <= 0:
            max_blocks_per_req = 1
        num_req_slots = max(1, num_blocks_per_layer_row // max_blocks_per_req)
        dtype = ref.dtype

        buffers: list[torch.Tensor] = []
        allocated = 0
        for _ in range(num_layers):
            buf = torch.empty(
                ref.shape, dtype=dtype, device="cpu", pin_memory=True
            )
            buffers.append(buf)
            allocated += buf.numel() * buf.element_size()

        max_phys = num_blocks_per_layer_row
        physical_to_flat = torch.full(
            (max_phys + 1,), -1, dtype=torch.long, device="cpu", pin_memory=True
        )

        layer_to_index = {name: i for i, name in enumerate(layer_names)}
        free_slots = list(range(num_req_slots - 1, -1, -1))

        obj = cls(
            num_layers=num_layers,
            k_window=k_window,
            hbm_kv_bytes=hbm_kv_bytes,
            table_bytes=allocated,
            block_size=block_size,
            page_size=page_size,
            max_blocks_per_req=max_blocks_per_req,
            num_req_slots=num_req_slots,
            num_blocks_per_layer_row=num_blocks_per_layer_row,
            layer_names=layer_names,
            layer_to_index=layer_to_index,
            buffers=buffers,
            physical_to_flat=physical_to_flat,
            _free_slots=free_slots,
        )
        logger.info(
            "CpuKvTable: hbm_kv=%.2f GiB (K=%d unique=%d), "
            "cpu_table=%.2f GiB (%d layers × %.2f MiB/layer), "
            "ratio=%.2fx, req_slots=%d, max_blocks_per_req=%d, "
            "blocks_per_row=%d, page_size=%d",
            hbm_kv_bytes / (1 << 30),
            k_window,
            num_unique,
            allocated / (1 << 30),
            num_layers,
            (allocated / num_layers) / (1 << 20),
            allocated / hbm_kv_bytes if hbm_kv_bytes else 0.0,
            num_req_slots,
            max_blocks_per_req,
            num_blocks_per_layer_row,
            page_size,
        )
        return obj

    def _compute_barrier(self) -> torch.cuda.Event:
        if self._compute_sync is None:
            self._compute_sync = torch.cuda.Event()
        return self._compute_sync

    def reserve_req_slot(self, req_id: str) -> int:
        if req_id in self.req_id_to_slot:
            return self.req_id_to_slot[req_id]
        if not self._free_slots:
            raise RuntimeError(
                f"CpuKvTable: no free request slots (num_req_slots={self.num_req_slots})"
            )
        slot = self._free_slots.pop()
        self.req_id_to_slot[req_id] = slot
        return slot

    def release_req_slot(self, req_id: str) -> None:
        slot = self.req_id_to_slot.pop(req_id, None)
        phys_blocks = self.req_id_to_phys_blocks.pop(req_id, ())
        for phys in phys_blocks:
            if 0 <= phys < self.physical_to_flat.numel():
                self.physical_to_flat[phys] = -1
        if slot is not None:
            self._free_slots.append(slot)
        self.ordered_phys = []
        self.ordered_table_blocks = []

    def ensure_table_block_gpu(self, device: torch.device) -> torch.Tensor:
        n = self.physical_to_flat.numel()
        if (
            self.table_block_gpu is None
            or self.table_block_gpu.device != device
            or self.table_block_gpu.numel() != n
        ):
            self.table_block_gpu = torch.full(
                (n,), -1, dtype=torch.long, device=device
            )
            for phys in range(n):
                flat = int(self.physical_to_flat[phys].item())
                if flat >= 0:
                    self.table_block_gpu[phys] = flat // self.page_size
        return self.table_block_gpu

    def register_physical_blocks(
        self,
        req_id: str,
        physical_block_ids: list[int],
        *,
        device: torch.device | None = None,
    ) -> int:
        req_slot = self.reserve_req_slot(req_id)
        ordered = sorted(set(int(b) for b in physical_block_ids))
        self.req_id_to_phys_blocks[req_id] = ordered
        table_blocks: list[int] = []
        for logical_b, phys in enumerate(ordered):
            if phys < 0 or phys >= self.physical_to_flat.numel():
                continue
            table_block = req_slot * self.max_blocks_per_req + logical_b
            if table_block >= self.num_blocks_per_layer_row:
                break
            self.physical_to_flat[phys] = table_block * self.page_size
            table_blocks.append(table_block)
        self.ordered_phys = ordered
        self.ordered_table_blocks = table_blocks
        if device is not None:
            tbl_gpu = self.ensure_table_block_gpu(device)
            for phys, tb in zip(ordered, table_blocks):
                if 0 <= phys < tbl_gpu.numel():
                    tbl_gpu[phys] = tb
        return req_slot

    def layer_buffer(self, layer_name: str) -> torch.Tensor:
        return self.buffers[self.layer_to_index[layer_name]]

    def table_block_for_slot(self, host_slot: int) -> int | None:
        """CPU-side LUT: global cache slot -> table block index (no GPU sync)."""
        page = self.page_size
        phys = host_slot // page
        if phys < 0 or phys >= self.physical_to_flat.numel():
            return None
        flat = int(self.physical_to_flat[phys].item())
        if flat < 0:
            return None
        return flat // page

    def enqueue_n1_delta_layer(
        self,
        layer_idx: int,
        kv_gpu: torch.Tensor,
        phys: int,
        off: int,
        tbl_block: int,
        stream: torch.cuda.Stream,
        save_ev: torch.cuda.Event,
        *,
        sync_compute: bool = True,
    ) -> None:
        """Enqueue one-token D2H on ``stream`` (caller supplies pooled ``save_ev``)."""
        cpu_buf = self.buffers[layer_idx]
        if sync_compute:
            stream.wait_stream(torch.cuda.current_stream())
        self._copy_n1_token(cpu_buf, kv_gpu, phys, off, tbl_block)
        save_ev.record(stream)

    def enqueue_n1_delta_via_scratch(
        self,
        layer_idx: int,
        kv_gpu: torch.Tensor,
        delta_scratch: torch.Tensor,
        phys: int,
        off: int,
        tbl_block: int,
        stream: torch.cuda.Stream,
        save_ev: torch.cuda.Event,
        snap_ev: torch.cuda.Event,
        *,
        sync_compute: bool = True,
    ) -> None:
        """Snapshot token to scratch (``snap_ev``), then D2H; main free for parallel H2D."""
        cpu_buf = self.buffers[layer_idx]
        if sync_compute:
            stream.wait_stream(torch.cuda.current_stream())
        self._snapshot_n1_token_to_scratch(kv_gpu, delta_scratch, phys, off)
        snap_ev.record(stream)
        self._copy_n1_from_scratch(cpu_buf, delta_scratch, tbl_block, off)
        save_ev.record(stream)

    def enqueue_batched_n1_delta_via_scratch(
        self,
        layer_idx: int,
        kv_gpu: torch.Tensor,
        gpu_scratch: torch.Tensor,
        cpu_scratch: torch.Tensor,
        phys_gpu: torch.Tensor,
        off_gpu: torch.Tensor,
        tbl_block_cpu: torch.Tensor,
        off_cpu: torch.Tensor,
        stream: torch.cuda.Stream,
        save_ev: torch.cuda.Event,
        snap_ev: torch.cuda.Event | None,
        *,
        sync_compute: bool = False,
        scatter_queue: Any | None = None,
    ) -> None:
        """Gather N decode tokens, one D2H, then CPU scatter into the layer row."""
        n = int(phys_gpu.numel())
        if n <= 0:
            return
        if sync_compute:
            stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._gather_n_tokens_to_scratch(kv_gpu, gpu_scratch, phys_gpu, off_gpu, n)
            if snap_ev is not None:
                snap_ev.record(stream)
            if cpu_scratch.shape[0] == 2:
                cpu_scratch[:, :n].copy_(gpu_scratch[:, :n], non_blocking=True)
            else:
                cpu_scratch[:n].copy_(gpu_scratch[:n], non_blocking=True)
            save_ev.record(stream)

        def _finish_scatter() -> None:
            save_ev.synchronize()
            self._scatter_batched_n1_to_cpu_buf(
                self.buffers[layer_idx],
                cpu_scratch,
                tbl_block_cpu,
                off_cpu,
                n,
            )
            with torch.cuda.stream(stream):
                save_ev.record(stream)

        if scatter_queue is not None:
            scatter_queue.put(_finish_scatter)
        else:
            _finish_scatter()

    @staticmethod
    def _gather_n_tokens_to_scratch(
        kv_gpu: torch.Tensor,
        gpu_scratch: torch.Tensor,
        phys: torch.Tensor,
        off: torch.Tensor,
        n: int,
    ) -> None:
        if kv_gpu.shape[0] == 2:
            gathered = kv_gpu[:, phys, off]
            gpu_scratch[:, :n].copy_(gathered, non_blocking=True)
        elif kv_gpu.ndim >= 3 and kv_gpu.shape[1] == 2:
            gathered = kv_gpu[phys, :, off]
            gpu_scratch[:n].copy_(gathered, non_blocking=True)
        else:
            gathered = kv_gpu[phys, off]
            gpu_scratch[:n].copy_(gathered, non_blocking=True)

    @staticmethod
    def _scatter_batched_n1_to_cpu_buf(
        cpu_buf: torch.Tensor,
        cpu_scratch: torch.Tensor,
        tbl_blocks: torch.Tensor,
        offs: torch.Tensor,
        n: int,
    ) -> None:
        blocks = tbl_blocks[:n].long()
        off_slice = offs[:n].long()
        if cpu_buf.shape[0] == 2:
            cpu_buf[:, blocks, off_slice] = cpu_scratch[:, :n]
        elif cpu_buf.ndim >= 3 and cpu_buf.shape[1] == 2:
            cpu_buf[blocks, :, off_slice] = cpu_scratch[:n]
        else:
            cpu_buf[blocks, off_slice] = cpu_scratch[:n]

    @staticmethod
    def _snapshot_n1_token_to_scratch(
        kv_gpu: torch.Tensor,
        delta_scratch: torch.Tensor,
        phys: int,
        off: int,
    ) -> None:
        if kv_gpu.shape[0] == 2:
            delta_scratch.copy_(kv_gpu[:, phys, off], non_blocking=True)
        elif kv_gpu.ndim >= 3 and kv_gpu.shape[1] == 2:
            delta_scratch.copy_(kv_gpu[phys, :, off], non_blocking=True)
        else:
            delta_scratch.copy_(kv_gpu[phys, off], non_blocking=True)

    @staticmethod
    def _copy_n1_from_scratch(
        cpu_buf: torch.Tensor,
        delta_scratch: torch.Tensor,
        tbl_block: int,
        off: int,
    ) -> None:
        if cpu_buf.shape[0] == 2:
            cpu_buf[:, tbl_block, off].copy_(delta_scratch, non_blocking=True)
        elif cpu_buf.ndim >= 3 and cpu_buf.shape[1] == 2:
            cpu_buf[tbl_block, :, off].copy_(delta_scratch, non_blocking=True)
        else:
            cpu_buf[tbl_block, off].copy_(delta_scratch, non_blocking=True)

    def stage_delta_host_slot(
        self,
        layer_name: str,
        kv_gpu: torch.Tensor,
        host_slot: int,
        stream: torch.cuda.Stream,
        *,
        compute_ready: torch.cuda.Event | None = None,
        save_ev: torch.cuda.Event | None = None,
    ) -> torch.cuda.Event | None:
        """D2H one token using host-resolved slot (n==1 decode, no GPU scalar sync)."""
        tbl_block = self.table_block_for_slot(host_slot)
        if tbl_block is None:
            return None
        page = self.page_size
        phys = host_slot // page
        off = host_slot % page
        out_ev = save_ev if save_ev is not None else torch.cuda.Event(blocking=False)
        if compute_ready is not None:
            with torch.cuda.stream(stream):
                stream.wait_event(compute_ready)
                self._copy_n1_token(
                    self.layer_buffer(layer_name), kv_gpu, phys, off, tbl_block
                )
                out_ev.record(stream)
        else:
            layer_idx = self.layer_to_index[layer_name]
            self.enqueue_n1_delta_layer(
                layer_idx, kv_gpu, phys, off, tbl_block, stream, out_ev
            )
        return out_ev

    @staticmethod
    def _copy_n1_token(
        cpu_buf: torch.Tensor,
        kv_gpu: torch.Tensor,
        phys: int,
        off: int,
        tbl_block: int,
    ) -> None:
        if kv_gpu.shape[0] == 2:
            cpu_buf[:, tbl_block, off].copy_(
                kv_gpu[:, phys, off], non_blocking=True
            )
        elif kv_gpu.ndim >= 3 and kv_gpu.shape[1] == 2:
            cpu_buf[tbl_block, :, off].copy_(kv_gpu[phys, :, off], non_blocking=True)
        else:
            cpu_buf[tbl_block, off].copy_(kv_gpu[phys, off], non_blocking=True)

    def stage_blocks_from_gpu(
        self,
        layer_name: str,
        kv_gpu: torch.Tensor,
        physical_block_ids: list[int],
        stream: torch.cuda.Stream,
    ) -> torch.cuda.Event:
        sync_ev = self._compute_barrier()
        sync_ev.record()
        save_ev = torch.cuda.Event(blocking=False)
        cpu_buf = self.layer_buffer(layer_name)
        bd = _block_dim(kv_gpu)
        phys_list = physical_block_ids or self.ordered_phys
        blocks = self.ordered_table_blocks
        with torch.cuda.stream(stream):
            stream.wait_event(sync_ev)
            if (
                len(phys_list) >= 2
                and len(blocks) == len(phys_list)
                and phys_list[-1] - phys_list[0] == len(phys_list) - 1
                and blocks[-1] - blocks[0] == len(blocks) - 1
            ):
                p0, p1 = phys_list[0], phys_list[-1]
                t0, t1 = blocks[0], blocks[-1]
                src = [slice(None)] * kv_gpu.ndim
                dst = [slice(None)] * cpu_buf.ndim
                src[bd] = slice(p0, p1 + 1)
                dst[bd] = slice(t0, t1 + 1)
                cpu_buf[tuple(dst)].copy_(kv_gpu[tuple(src)], non_blocking=True)
            else:
                for phys, tb in zip(phys_list, blocks):
                    if phys < 0 or phys >= kv_gpu.shape[bd]:
                        continue
                    src = [slice(None)] * kv_gpu.ndim
                    dst = [slice(None)] * cpu_buf.ndim
                    src[bd] = slice(phys, phys + 1)
                    dst[bd] = slice(tb, tb + 1)
                    cpu_buf[tuple(dst)].copy_(
                        kv_gpu[tuple(src)], non_blocking=True
                    )
            save_ev.record(stream)
        return save_ev

    def stage_delta(
        self,
        layer_name: str,
        kv_gpu: torch.Tensor,
        slot_mapping: torch.Tensor,
        stream: torch.cuda.Stream,
        num_actual_tokens: int | None = None,
        *,
        host_slot: int | None = None,
    ) -> torch.cuda.Event | None:
        if host_slot is not None:
            return self.stage_delta_host_slot(layer_name, kv_gpu, host_slot, stream)

        if num_actual_tokens is not None:
            slot_mapping = slot_mapping[:num_actual_tokens]
        valid = slot_mapping >= 0
        if not bool(valid.any()):
            return None
        if not bool(valid.all()):
            slot_mapping = slot_mapping[valid]

        n = int(slot_mapping.numel())
        if n == 1:
            slot_val = int(slot_mapping.view(-1).cpu().item())
            return self.stage_delta_host_slot(layer_name, kv_gpu, slot_val, stream)

        page = self.page_size
        cpu_buf = self.layer_buffer(layer_name)
        lut = self.physical_to_flat.to(kv_gpu.device, non_blocking=True)
        gpu_blocks = torch.div(slot_mapping, page, rounding_mode="floor")
        offsets = slot_mapping % page
        base_flat = lut[gpu_blocks]
        table_slots = base_flat + offsets
        if (table_slots < 0).any():
            return None

        if kv_gpu.shape[0] == 2:
            delta = kv_gpu[:, gpu_blocks, offsets]
        elif kv_gpu.ndim >= 3 and kv_gpu.shape[1] == 2:
            delta = kv_gpu[gpu_blocks, :, offsets]
        else:
            delta = kv_gpu[gpu_blocks, offsets]

        sync_ev = self._compute_barrier()
        sync_ev.record()
        save_ev = torch.cuda.Event(blocking=False)
        cpu_delta = torch.empty_like(delta, device="cpu", pin_memory=True)
        cpu_slots = table_slots.to(dtype=torch.long, device="cpu", non_blocking=True)
        ready_event = torch.cuda.Event(blocking=False)
        with torch.cuda.stream(stream):
            stream.wait_event(sync_ev)
            cpu_delta.copy_(delta, non_blocking=True)
            save_ev.record(stream)
            ready_event.record(stream)

        return save_ev, ready_event, cpu_buf, cpu_slots[:n], cpu_delta

    def prefetch_layer_to_gpu(
        self,
        layer_name: str,
        kv_gpu: torch.Tensor,
        physical_block_ids: list[int] | None,
        stream: torch.cuda.Stream,
        *,
        wait_save_event: torch.cuda.Event | None = None,
        ready_event: torch.cuda.Event | None = None,
        delta_phys: int | None = None,
    ) -> torch.cuda.Event:
        ready = ready_event if ready_event is not None else torch.cuda.Event(
            blocking=False
        )
        cpu_buf = self.layer_buffer(layer_name)
        bd = _block_dim(kv_gpu)
        phys_list = physical_block_ids or self.ordered_phys
        blocks = self.ordered_table_blocks
        if not phys_list or len(blocks) != len(phys_list):
            return ready

        def _copy_range(p_start: int, p_end: int, t_start: int, t_end: int) -> None:
            src = [slice(None)] * cpu_buf.ndim
            dst = [slice(None)] * kv_gpu.ndim
            src[bd] = slice(t_start, t_end + 1)
            dst[bd] = slice(p_start, p_end + 1)
            kv_gpu[tuple(dst)].copy_(cpu_buf[tuple(src)], non_blocking=True)

        def _copy_block(phys: int, tb: int) -> None:
            if phys < 0 or phys >= kv_gpu.shape[bd]:
                return
            src = [slice(None)] * cpu_buf.ndim
            dst = [slice(None)] * kv_gpu.ndim
            src[bd] = slice(tb, tb + 1)
            dst[bd] = slice(phys, phys + 1)
            kv_gpu[tuple(dst)].copy_(cpu_buf[tuple(src)], non_blocking=True)

        overlap_phys = (
            delta_phys
            if delta_phys is not None and delta_phys in phys_list
            else None
        )
        if overlap_phys is None:
            if wait_save_event is not None:
                stream.wait_event(wait_save_event)
            if (
                len(phys_list) >= 2
                and phys_list[-1] - phys_list[0] == len(phys_list) - 1
                and blocks[-1] - blocks[0] == len(blocks) - 1
            ):
                _copy_range(phys_list[0], phys_list[-1], blocks[0], blocks[-1])
            else:
                for phys, tb in zip(phys_list, blocks):
                    _copy_block(phys, tb)
        elif (
            len(phys_list) >= 2
            and phys_list[-1] - phys_list[0] == len(phys_list) - 1
            and blocks[-1] - blocks[0] == len(blocks) - 1
        ):
            split_at = phys_list.index(overlap_phys)
            if split_at > 0:
                _copy_range(
                    phys_list[0],
                    phys_list[split_at - 1],
                    blocks[0],
                    blocks[split_at - 1],
                )
            if wait_save_event is not None:
                stream.wait_event(wait_save_event)
            _copy_range(
                overlap_phys,
                phys_list[-1],
                blocks[split_at],
                blocks[-1],
            )
        else:
            for phys, tb in zip(phys_list, blocks):
                if phys != overlap_phys:
                    _copy_block(phys, tb)
            if wait_save_event is not None:
                stream.wait_event(wait_save_event)
            for phys, tb in zip(phys_list, blocks):
                if phys == overlap_phys:
                    _copy_block(phys, tb)
        ready.record(stream)
        return ready

    def prefetch_table_blocks_to_compact_gpu(
        self,
        layer_name: str,
        kv_compact: torch.Tensor,
        stream: torch.cuda.Stream,
        *,
        ready_event: torch.cuda.Event | None = None,
    ) -> torch.cuda.Event:
        """H2D into compact GPU rows 0..n-1 (CPU source uses global table block ids)."""
        ready = ready_event if ready_event is not None else torch.cuda.Event(
            blocking=False
        )
        cpu_buf = self.layer_buffer(layer_name)
        bd = _block_dim(kv_compact)
        blocks = self.ordered_table_blocks
        n = len(blocks)
        if n == 0 or n > kv_compact.shape[bd]:
            return ready
        t0, t1 = blocks[0], blocks[-1]
        if len(blocks) >= 2 and blocks[-1] - blocks[0] == len(blocks) - 1:
            src = [slice(None)] * cpu_buf.ndim
            dst = [slice(None)] * kv_compact.ndim
            src[bd] = slice(t0, t1 + 1)
            dst[bd] = slice(0, n)
            kv_compact[tuple(dst)].copy_(cpu_buf[tuple(src)], non_blocking=True)
        else:
            for local_i, tb in enumerate(blocks):
                if tb < 0:
                    continue
                src = [slice(None)] * cpu_buf.ndim
                dst = [slice(None)] * kv_compact.ndim
                src[bd] = slice(tb, tb + 1)
                dst[bd] = slice(local_i, local_i + 1)
                kv_compact[tuple(dst)].copy_(cpu_buf[tuple(src)], non_blocking=True)
        if ready_event is not None:
            ready.record(stream)
        return ready

    def merge_compact_gpu_to_main(
        self,
        kv_compact: torch.Tensor,
        kv_main: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Copy compact rows 0..n-1 into main physical block rows."""
        bd = _block_dim(kv_main)
        phys_list = self.ordered_phys
        blocks = self.ordered_table_blocks
        n = len(phys_list)
        if n == 0 or len(blocks) != n or n > kv_compact.shape[bd]:
            return
        copy_stream = stream or torch.cuda.current_stream()
        if (
            n >= 2
            and phys_list[-1] - phys_list[0] == n - 1
            and blocks[-1] - blocks[0] == n - 1
        ):
            p0, p1 = phys_list[0], phys_list[-1]
            src = [slice(None)] * kv_compact.ndim
            dst = [slice(None)] * kv_main.ndim
            src[bd] = slice(0, n)
            dst[bd] = slice(p0, p1 + 1)
            with torch.cuda.stream(copy_stream):
                kv_main[tuple(dst)].copy_(
                    kv_compact[tuple(src)], non_blocking=True
                )
        else:
            with torch.cuda.stream(copy_stream):
                for local_i, phys in enumerate(phys_list):
                    if phys < 0 or phys >= kv_main.shape[bd]:
                        continue
                    src = [slice(None)] * kv_compact.ndim
                    dst = [slice(None)] * kv_main.ndim
                    src[bd] = slice(local_i, local_i + 1)
                    dst[bd] = slice(phys, phys + 1)
                    kv_main[tuple(dst)].copy_(
                        kv_compact[tuple(src)], non_blocking=True
                    )
