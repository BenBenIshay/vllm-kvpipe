#!/usr/bin/env python3
"""Standalone test for RungdHostPool (rung-d host-block-id KV subsystem).

Simulates multiple CPU-tier requests decoding with their KV managed entirely by
the host pool (own block-id space): seed prefill from resident, grow with new
host blocks, write each decode token, stream per layer into the k_window rolling
region. Validates against a resident reference that:
  (1) the rolling slot reconstructs each request's full KV history bitwise;
  (2) flash attention over the compacted rolling region == attention over the
      resident reference;
  (3) it stays correct AFTER the resident blocks are overwritten (simulating the
      block freed + reused by another request) — the point of the host id space.

Run on the GPU node:
  srun --jobid=<J> --overlap --mem=0 --gres=gpu:1 bash -lc \
    'cd /mnt/beegfs/benishay/vllm-kvpipe && .venv/bin/python \
     benchmarks/kv_pipelining_bench/rungd_host_pool_test.py'
"""
from __future__ import annotations

import torch

from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
)
from vllm.v1.hybrid_rungd_pool import RungdHostPool

DEV = "cuda"
DT = torch.bfloat16
torch.manual_seed(0)

NUM_LAYERS = 6
NBLK, BS, NKV, HD = 64, 16, 2, 8
NUM_HEADS = 8
K_WINDOW = 3
SCALE = HD ** -0.5
FA = get_flash_attn_version()


def paged(kv_hist: torch.Tensor, seqlen: int):
    """[2, seq, NKV, HD] -> paged (k,v) [nb, BS, NKV, HD]."""
    nb = (seqlen + BS - 1) // BS
    kk = torch.zeros(nb, BS, NKV, HD, device=DEV, dtype=DT)
    vv = torch.zeros(nb, BS, NKV, HD, device=DEV, dtype=DT)
    for i in range(seqlen):
        kk[i // BS, i % BS] = kv_hist[0, i]
        vv[i // BS, i % BS] = kv_hist[1, i]
    return kk, vv


def main() -> None:
    reqs = {
        "A": {"res_blocks": [1, 2, 3], "prefill": 40},   # 3 blocks
        "B": {"res_blocks": [5, 6], "prefill": 28},      # 2 blocks
    }
    T = 12

    res = [torch.randn(2, NBLK, BS, NKV, HD, device=DEV, dtype=DT)
           for _ in range(NUM_LAYERS)]
    # Ground-truth per-(req,layer) KV history, seeded from resident prefill.
    truth = {r: [] for r in reqs}
    for r, info in reqs.items():
        for L in range(NUM_LAYERS):
            kv = torch.zeros(2, info["prefill"], NKV, HD, device=DEV, dtype=DT)
            for i in range(info["prefill"]):
                b, off = info["res_blocks"][i // BS], i % BS
                kv[:, i] = res[L][:, b, off]
            truth[r].append(kv)

    pool = RungdHostPool(
        layer_names=[f"l{i}" for i in range(NUM_LAYERS)],
        kv_shape=(2, NBLK, BS, NKV, HD), block_size=BS, k_window=K_WINDOW,
        h_max_blocks=32, w_max_blocks=16, device=DEV, dtype=DT,
    )

    seq = {r: info["prefill"] for r, info in reqs.items()}  # KV length so far
    content_ok = True
    attn_ok = True

    for step in range(T):
        active = list(reqs)
        for r in active:
            nblocks = (seq[r] + 1 + BS - 1) // BS   # room for the new token
            pool.ensure_req(r, nblocks)
            if r not in pool.req_seeded:
                for L in range(NUM_LAYERS):
                    pool.seed_layer(r, f"l{L}", res[L], reqs[r]["res_blocks"])
                pool.mark_seeded(r)

        seqused_list = [seq[r] + 1 for r in active]  # incl. the new token
        bt, seqused = pool.build_step(active, seqused_list)

        if step == 1:  # simulate the resident blocks being freed + reused
            for L in range(NUM_LAYERS):
                for r in active:
                    for b in reqs[r]["res_blocks"]:
                        res[L][:, b] = torch.randn_like(res[L][:, b])

        for L in range(NUM_LAYERS):
            slot = pool.stream_layer(L)
            new_k = torch.randn(len(active), NKV, HD, device=DEV, dtype=DT)
            new_v = torch.randn(len(active), NKV, HD, device=DEV, dtype=DT)
            comp, offs, hbs = [], [], []
            for ri, r in enumerate(active):
                p = seq[r]  # new token position
                c, off = pool.compact_of(r, p)
                comp.append(c); offs.append(off)
                hbs.append(pool.req_blocks[r][p // BS])
                # append to ground truth
                col = torch.stack([new_k[ri], new_v[ri]])[:, None]  # [2,1,NKV,HD]
                truth[r][L] = torch.cat([truth[r][L], col], dim=1)
            pool.write_token(L, comp, offs, hbs, new_k, new_v)

            kslot, vslot = slot   # (key_slot, value_slot) from the split mirrors
            # (1) content check
            for ri, r in enumerate(active):
                for pos in range(seq[r] + 1):
                    c, off = pool.compact_of(r, pos)
                    if not torch.equal(kslot[c, off], truth[r][L][0, pos]):
                        content_ok = False
                    if not torch.equal(vslot[c, off], truth[r][L][1, pos]):
                        content_ok = False

            # (2/3) attention check
            q = torch.randn(len(active), NUM_HEADS, HD, device=DEV, dtype=DT)
            out_pool = torch.empty(len(active), NUM_HEADS, HD, device=DEV, dtype=DT)
            cu = torch.arange(0, len(active) + 1, device=DEV, dtype=torch.int32)
            flash_attn_varlen_func(
                q=q, k=kslot, v=vslot, out=out_pool, cu_seqlens_q=cu, max_seqlen_q=1,
                seqused_k=seqused.to(DEV), max_seqlen_k=int(seqused.max()),
                softmax_scale=SCALE, causal=True, block_table=bt.to(DEV), fa_version=FA,
            )
            for ri, r in enumerate(active):
                kk, vv = paged(truth[r][L], seq[r] + 1)
                oref = torch.empty(1, NUM_HEADS, HD, device=DEV, dtype=DT)
                flash_attn_varlen_func(
                    q=q[ri:ri+1], k=kk, v=vv, out=oref,
                    cu_seqlens_q=torch.tensor([0, 1], device=DEV, dtype=torch.int32),
                    max_seqlen_q=1,
                    seqused_k=torch.tensor([seq[r] + 1], device=DEV, dtype=torch.int32),
                    max_seqlen_k=seq[r] + 1, softmax_scale=SCALE, causal=True,
                    block_table=torch.arange((seq[r] + 1 + BS - 1) // BS, device=DEV,
                                             dtype=torch.int32)[None],
                    fa_version=FA,
                )
                if not torch.equal(out_pool[ri], oref[0]):
                    attn_ok = False

        for r in active:
            seq[r] += 1

    print(f"[1] rolling-slot content == truth (bitwise): {content_ok}")
    print(f"[2/3] attn(pool) == attn(resident ref), incl. after block reuse: {attn_ok}")
    print("=" * 64)
    if content_ok and attn_ok:
        print("RUNGD HOST-POOL PASS: multi-request seed+grow+write+stream is "
              "bitwise-correct and immune to resident block reuse -> the host "
              "block-id reclaim subsystem is sound.")
    else:
        print(f"RUNGD HOST-POOL FAIL: content_ok={content_ok} attn_ok={attn_ok}")


if __name__ == "__main__":
    main()
