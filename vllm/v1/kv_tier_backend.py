# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mechanism boundary for request-level KV tiering.

The policy layer (which requests live where, how many riders to admit, when to
migrate) must not depend on *how* a non-resident request's KV is stored and
read. This module is that boundary.

Two implementations are intended:

  ``StagedHostBackend``  -- our rolling k-window: host-pinned mirror, per-layer
      bulk H2D into a compact GPU region, attention reads the region at full
      HBM speed. Runs on any CUDA GPU; this is what we validate on A100.

  ``DirectKVBackend``    -- DirectKV (OSDI'26): KV stays in CPU-pinned memory
      and their SM90a kernel reads it zero-copy over NVLink-C2C. **SM90 only**
      (TMA + FA-3 + CuTe), so it cannot be compiled or run on Ampere.

The boundary is viable because DirectKV's own design already works this way:
their ``forward_decode`` partitions a batch and dispatches disjoint subsets to
different attention implementations in the same step, their kernel accepts an
arbitrary ``req_ids`` subset, and their pool assigns rows per request via a
``req_id -> row`` free-list. See ``paper/DIRECTKV_INTEGRATION_ANALYSIS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass
class TierAttendCtx:
    """Everything a tier needs to run attention for its rows, beyond q."""

    block_table: torch.Tensor      # [n_rows, max_blk] into the tier's region
    seqused_k: torch.Tensor        # [n_rows] valid key length per row
    scale: float
    sliding_window: Any = (-1, -1)
    softcap: float = 0.0
    fa_version: int | None = None
    alibi_slopes: torch.Tensor | None = None
    q_descale: torch.Tensor | None = None
    k_descale: torch.Tensor | None = None
    v_descale: torch.Tensor | None = None
    sinks: torch.Tensor | None = None
    # Loop-invariant across layers; see StagedHostBackend.attend.
    max_seqlen_k: int | None = None
    cu_seqlens_q: "torch.Tensor | None" = None


@runtime_checkable
class KVTierBackend(Protocol):
    """A place a request's KV can live, plus how to attend over it."""

    # -- lifecycle ---------------------------------------------------------
    def alloc(self, req_id: str, num_blocks: int) -> None: ...
    def free(self, req_id: str) -> None: ...

    # -- per-step ----------------------------------------------------------
    def begin_step(
        self, active_req_ids: list[str], seq_lens: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare this step's region. Returns (block_table, seqused_k)."""
        ...

    def prepare_layer(self, layer_idx: int) -> Any:
        """Make layer `layer_idx` readable by attention; returns a handle."""
        ...

    def write_token(self, layer_idx: int, *args: Any) -> None:
        """Append this decode step's K/V for the tier's rows."""
        ...

    def attend(
        self, layer_idx: int, handle: Any, q_sub: torch.Tensor,
        ctx: TierAttendCtx, out: torch.Tensor,
    ) -> None:
        """Attention over this tier's rows, written into `out`.

        The CALLER allocates `out` (from the model's own output tensor), so a
        backend can never change the result's shape/dtype/layout.
        """
        ...

    # -- policy inputs -----------------------------------------------------
    def bytes_per_step(self, seq_lens: list[int]) -> int:
        """KV bytes this tier must move per decode step for these rows.

        Feeds the admission controller's B1 roofline. Zero for a tier whose KV
        is already resident.
        """
        ...


class StagedHostBackend:
    """Rolling k-window over a host-pinned mirror (`RungdHostPool`).

    Thin adapter: the pool already implements the mechanism, this exposes it
    through the boundary so policy code never touches the pool directly.
    """

    name = "staged_host"

    def __init__(self, pool: Any, kv_bytes_per_token: int) -> None:
        self._pool = pool
        self._b_tok = int(kv_bytes_per_token)

    # -- lifecycle
    def alloc(self, req_id: str, num_blocks: int) -> None:
        self._pool.ensure_req(req_id, num_blocks)

    def free(self, req_id: str) -> None:
        self._pool.free_req(req_id)

    # -- per-step
    def begin_step(
        self, active_req_ids: list[str], seq_lens: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._pool.build_step(active_req_ids, seq_lens)

    def prepare_layer(self, layer_idx: int) -> Any:
        return self._pool.stream_layer(layer_idx)

    def write_token(self, layer_idx: int, *args: Any) -> None:
        self._pool.write_token(layer_idx, *args)

    def attend(
        self, layer_idx: int, handle: Any, q_sub: torch.Tensor,
        ctx: TierAttendCtx, out: torch.Tensor,
    ) -> None:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        # `prepare_layer` returns the pool's (key_slot, value_slot). They are
        # separate contiguous tensors rather than one [2, W, ...] to unbind:
        # the mirrors are block-major so their runs DMA contiguously, and a
        # block-major slot cannot be sliced into what flash needs.
        k_slot, v_slot = handle
        m = q_sub.shape[0]
        # Both of these are fixed for the step; the caller supplies them so
        # this does not redo a device reduction (with a .item(), so a full
        # pipeline drain) and an allocation on every one of 64 layers.
        cu = ctx.cu_seqlens_q
        if cu is None or cu.numel() != m + 1:
            cu = torch.arange(0, m + 1, device=q_sub.device, dtype=torch.int32)
        max_seqlen_k = ctx.max_seqlen_k
        if max_seqlen_k is None:
            max_seqlen_k = int(ctx.seqused_k.max())
        flash_attn_varlen_func(
            q=q_sub, k=k_slot, v=v_slot, out=out, cu_seqlens_q=cu,
            max_seqlen_q=1, seqused_k=ctx.seqused_k,
            max_seqlen_k=max_seqlen_k, softmax_scale=ctx.scale,
            causal=True, alibi_slopes=ctx.alibi_slopes,
            window_size=ctx.sliding_window, block_table=ctx.block_table,
            softcap=ctx.softcap, fa_version=ctx.fa_version,
            q_descale=ctx.q_descale, k_descale=ctx.k_descale,
            v_descale=ctx.v_descale, s_aux=ctx.sinks,
        )

    # -- policy inputs
    def bytes_per_step(self, seq_lens: list[int]) -> int:
        return self._b_tok * int(sum(seq_lens))


class ResidentBackend:
    """The HBM tier. KV is already in the paged cache; nothing to move.

    Exists so the policy layer can treat both tiers uniformly and so
    `bytes_per_step` is 0 by construction rather than by special-casing.
    """

    name = "resident"

    def alloc(self, req_id: str, num_blocks: int) -> None:
        return None

    def free(self, req_id: str) -> None:
        return None

    def begin_step(self, active_req_ids, seq_lens):
        raise NotImplementedError("resident rows use the standard attention path")

    def prepare_layer(self, layer_idx: int) -> Any:
        return None

    def write_token(self, layer_idx: int, *args: Any) -> None:
        return None

    def attend(self, layer_idx, handle, q_sub, ctx, out):
        raise NotImplementedError("resident rows use the standard attention path")

    def bytes_per_step(self, seq_lens: list[int]) -> int:
        return 0


class DirectKVBackend:
    """DirectKV (OSDI'26) as the host tier — SM90a only, UNTESTED.

    Written against the boundary so the GH200 trip is a swap rather than a
    rewrite. **Deliberately not exercised**: their kernel needs TMA + FA-3 +
    CuTe (SM90a) and this cluster has neither an SM90 GPU nor `nvcc`, so none
    of this can be compiled or run here. Treat every line as unreviewed-by-
    execution until it runs on Hopper.

    Mapping onto their code (see `paper/DIRECTKV_INTEGRATION_ANALYSIS.md`):
      alloc/free   -> DirectKVRequestPool.alloc/free (per-request row free-list)
      prepare_layer-> no-op; their kernel reads CPU-pinned rows directly
      attend       -> _directkv_fwd over an arbitrary `req_ids` subset, which is
                      exactly how their forward_decode already dispatches
                      seqlen buckets
    """

    name = "directkv"

    def __init__(self, pool: Any, kv_bytes_per_token: int) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("DirectKVBackend requires CUDA")
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            raise RuntimeError(
                "DirectKVBackend requires SM90 (Hopper): their kernel uses TMA, "
                "FlashAttention-3 and CuTe. Use StagedHostBackend on Ampere."
            )
        self._pool = pool
        self._b_tok = int(kv_bytes_per_token)

    def alloc(self, req_id: str, num_blocks: int) -> None:
        self._pool.alloc(req_id)

    def free(self, req_id: str) -> None:
        self._pool.free(req_id)

    def begin_step(self, active_req_ids, seq_lens):
        raise NotImplementedError("DirectKVBackend: not implemented until Hopper")

    def prepare_layer(self, layer_idx: int) -> Any:
        return None            # their kernel reads host rows in place

    def write_token(self, layer_idx: int, *args: Any) -> None:
        raise NotImplementedError("DirectKVBackend: not implemented until Hopper")

    def attend(self, layer_idx, handle, q_sub, ctx, out):
        raise NotImplementedError("DirectKVBackend: not implemented until Hopper")

    def bytes_per_step(self, seq_lens: list[int]) -> int:
        return self._b_tok * int(sum(seq_lens))
