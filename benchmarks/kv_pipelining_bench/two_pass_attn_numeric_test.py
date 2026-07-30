#!/usr/bin/env python3
"""Numeric foundation test for the hybrid two-pass (dual-region) decode attention.

The hybrid capacity win needs ONE weight-shared forward whose *attention* reads
two KV sources: HBM rows attend a full-N persistent pool, CPU rows attend a
rolling-K pool (different physical tensor, different block table). This test
validates the two invariants that the two-pass FlashAttentionImpl subclass will
rely on, in ISOLATION (no server, real GPU numerics, bf16 + GQA + causal, decode
shapes), so any bug is in the attention recipe itself, not the vLLM plumbing.

INVARIANT A (split is lossless over ONE pool):
    single-pass flash_attn_varlen over all N decode rows
      == two-pass at boundary h (rows[0:h] then rows[h:]) over the SAME pool,
    bitwise. This is what Step 2 (subclass, both cohorts still resident) needs.

INVARIANT B (two-region stitch is correct):
    running pass1 over KV-region-A/block-table-A into out[0:h] and pass2 over a
    DIFFERENT KV-region-B/block-table-B into out[h:] equals the two cohorts
    computed independently — i.e. writing each cohort into its row-slice of one
    shared output buffer, with rebased cu_seqlens for pass2, is correct. This is
    the exact recipe the capacity kernel uses.

Run on the GPU node:
  srun --jobid=<J> --overlap --mem=0 --gres=gpu:1 bash -lc \
    'cd /mnt/beegfs/benishay/vllm-kvpipe && .venv/bin/python \
     benchmarks/kv_pipelining_bench/two_pass_attn_numeric_test.py'
"""
from __future__ import annotations

import torch

from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
)

DEV = "cuda"
DT = torch.bfloat16
torch.manual_seed(0)

# Qwen2.5-0.5B-ish decode shapes.
NUM_HEADS = 14
NUM_KV_HEADS = 2
HEAD_SIZE = 64
BLOCK_SIZE = 16
SCALE = HEAD_SIZE ** -0.5
FA_VER = get_flash_attn_version()


def _make_pool(num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A paged K and V cache: [num_blocks, block_size, num_kv_heads, head_size]."""
    k = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, device=DEV, dtype=DT)
    v = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, device=DEV, dtype=DT)
    return k, v


def _block_table(seq_lens: list[int], base_block: int, max_blocks: int) -> torch.Tensor:
    """Contiguous distinct physical blocks per request (no sharing), int32."""
    bt = torch.zeros(len(seq_lens), max_blocks, device=DEV, dtype=torch.int32)
    nxt = base_block
    for i, sl in enumerate(seq_lens):
        nblk = (sl + BLOCK_SIZE - 1) // BLOCK_SIZE
        for j in range(nblk):
            bt[i, j] = nxt
            nxt += 1
    return bt


def _decode_attn(q, k, v, seq_lens, block_table, out):
    """One decode flash call: 1 query token per request, seqused_k = history."""
    nreq = len(seq_lens)
    cu_q = torch.arange(0, nreq + 1, device=DEV, dtype=torch.int32)  # 1 tok/req
    seqused_k = torch.tensor(seq_lens, device=DEV, dtype=torch.int32)
    flash_attn_varlen_func(
        q=q, k=k, v=v, out=out,
        cu_seqlens_q=cu_q, max_seqlen_q=1,
        seqused_k=seqused_k, max_seqlen_k=max(seq_lens),
        softmax_scale=SCALE, causal=True,
        block_table=block_table, fa_version=FA_VER,
    )


def invariant_a_split_lossless() -> bool:
    """single-pass over N == two-pass at boundary h, same pool. Bitwise."""
    seq_lens = [7, 31, 16, 48, 5, 33, 20, 9]   # mixed histories
    n = len(seq_lens)
    h = 3                                       # split boundary
    max_blocks = max((s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in seq_lens)
    nb = sum((s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in seq_lens) + 4
    k, v = _make_pool(nb)
    bt = _block_table(seq_lens, base_block=1, max_blocks=max_blocks)  # block 0 reserved
    q = torch.randn(n, NUM_HEADS, HEAD_SIZE, device=DEV, dtype=DT)

    out_single = torch.empty(n, NUM_HEADS, HEAD_SIZE, device=DEV, dtype=DT)
    _decode_attn(q, k, v, seq_lens, bt, out_single)

    out_two = torch.empty(n, NUM_HEADS, HEAD_SIZE, device=DEV, dtype=DT)
    # pass 1: rows [0:h]
    _decode_attn(q[:h], k, v, seq_lens[:h], bt[:h], out_two[:h])
    # pass 2: rows [h:] — sliced q/block_table/seq_lens, output written in place
    _decode_attn(q[h:], k, v, seq_lens[h:], bt[h:], out_two[h:])

    same = torch.equal(out_single, out_two)
    maxdiff = (out_single.float() - out_two.float()).abs().max().item()
    print(f"[A] split-lossless (same pool): bitwise_equal={same} max|diff|={maxdiff:.3e}")
    return same


def invariant_b_two_region_stitch() -> bool:
    """Two DIFFERENT KV pools/block-tables stitched into one output buffer ==
    each cohort computed independently. Bitwise."""
    hbm_seq = [40, 12, 55, 8]        # HBM cohort histories
    cpu_seq = [16, 33, 7]            # CPU cohort histories (different pool)
    n_hbm, n_cpu = len(hbm_seq), len(cpu_seq)
    n = n_hbm + n_cpu

    mb_h = max((s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in hbm_seq)
    mb_c = max((s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in cpu_seq)
    nb_h = sum((s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in hbm_seq) + 2
    nb_c = sum((s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in cpu_seq) + 2

    k_h, v_h = _make_pool(nb_h)      # HBM region
    k_c, v_c = _make_pool(nb_c)      # rolling region (distinct tensor)
    bt_h = _block_table(hbm_seq, base_block=1, max_blocks=mb_h)
    bt_c = _block_table(cpu_seq, base_block=1, max_blocks=mb_c)

    q = torch.randn(n, NUM_HEADS, HEAD_SIZE, device=DEV, dtype=DT)
    q_h, q_c = q[:n_hbm], q[n_hbm:]

    # Reference: each cohort computed independently into its own buffer.
    ref_h = torch.empty(n_hbm, NUM_HEADS, HEAD_SIZE, device=DEV, dtype=DT)
    ref_c = torch.empty(n_cpu, NUM_HEADS, HEAD_SIZE, device=DEV, dtype=DT)
    _decode_attn(q_h, k_h, v_h, hbm_seq, bt_h, ref_h)
    _decode_attn(q_c, k_c, v_c, cpu_seq, bt_c, ref_c)

    # Fused: one output buffer, two passes into row-slices (the real recipe).
    out = torch.empty(n, NUM_HEADS, HEAD_SIZE, device=DEV, dtype=DT)
    _decode_attn(q_h, k_h, v_h, hbm_seq, bt_h, out[:n_hbm])   # HBM pass -> [0:n_hbm)
    _decode_attn(q_c, k_c, v_c, cpu_seq, bt_c, out[n_hbm:])   # CPU pass -> [n_hbm:)

    ok_h = torch.equal(out[:n_hbm], ref_h)
    ok_c = torch.equal(out[n_hbm:], ref_c)
    print(f"[B] two-region stitch: hbm_slice_ok={ok_h} cpu_slice_ok={ok_c}")
    return ok_h and ok_c


def main() -> None:
    print(f"device={torch.cuda.get_device_name()} dtype={DT} fa_version={FA_VER}")
    a = invariant_a_split_lossless()
    b = invariant_b_two_region_stitch()
    print("=" * 60)
    if a and b:
        print("TWO-PASS NUMERIC PASS: split is lossless AND two-region stitch is "
              "bitwise-correct -> the dual-region attention recipe is sound.")
    else:
        print(f"TWO-PASS NUMERIC FAIL: invariant_a={a} invariant_b={b} -> recipe bug.")


if __name__ == "__main__":
    main()
