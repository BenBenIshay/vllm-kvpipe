#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare decode-attention kernels on identical shapes: FA2 vs FlashInfer.

Why this matters: every number we have (the eff_kv degradation curve, the
two-source tax, the B=32 anomaly) was measured on vLLM's auto-selected
FLASH_ATTN backend = FA2 on A100. num_splits turned out to be FA3-only, so we
cannot tune FA2's splitting. If FlashInfer's paged-decode kernel is materially
faster -- or lacks the B=32 anomaly -- then our conclusions describe FA2, not
the hardware, and must be re-derived on the better kernel.
"""

from __future__ import annotations

import argparse
import itertools
import json

import torch

from vllm.vllm_flash_attn import flash_attn_varlen_func


def timeit(fn, iters=60, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--q-heads", type=int, default=64)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--batches", default="4,8,13,16,24,32,48,64")
    p.add_argument("--contexts", default="4096,8192")
    p.add_argument("--out", default="backend_compare.json")
    args = p.parse_args()

    dev, dtype = "cuda", torch.float16
    scale = args.head_dim ** -0.5
    batches = [int(x) for x in args.batches.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]

    import flashinfer
    print(f"device {torch.cuda.get_device_name(0)} | flashinfer "
          f"{flashinfer.__version__}")
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)

    print(f"\n{'B':>4} {'S':>6} {'FA2':>10} {'FlashInfer':>12} "
          f"{'FI vs FA2':>10}  {'bytes':>8} {'FA2 eff':>8} {'FI eff':>8}")
    PEAK = 1935e9
    kv_per_tok = 2 * args.kv_heads * args.head_dim * 2   # K+V, fp16
    results = []

    for S, B in itertools.product(contexts, batches):
        pages_per_seq = (S + args.page_size - 1) // args.page_size
        n_pages = pages_per_seq * B

        # ---- FA2 layout: separate k/v, (n_pages, page_size, kv_heads, dim)
        kc = torch.randn(n_pages + 8, args.page_size, args.kv_heads,
                         args.head_dim, dtype=dtype, device=dev)
        vc = torch.randn_like(kc)
        bt = torch.arange(n_pages, dtype=torch.int32,
                          device=dev).view(B, pages_per_seq)
        q = torch.randn(B, args.q_heads, args.head_dim, dtype=dtype, device=dev)
        out = torch.empty_like(q)
        seq = torch.full((B,), S, dtype=torch.int32, device=dev)
        cu = torch.arange(B + 1, dtype=torch.int32, device=dev)

        t_fa = timeit(lambda: flash_attn_varlen_func(
            q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=1,
            seqused_k=seq, max_seqlen_k=S, softmax_scale=scale,
            causal=True, block_table=bt))

        # ---- FlashInfer layout: combined (n_pages, 2, page_size, kv_h, dim)
        t_fi = None
        try:
            kv = torch.randn(n_pages, 2, args.page_size, args.kv_heads,
                             args.head_dim, dtype=dtype, device=dev)
            indptr = torch.arange(0, (B + 1) * pages_per_seq, pages_per_seq,
                                  dtype=torch.int32, device=dev)
            indices = torch.arange(n_pages, dtype=torch.int32, device=dev)
            last = torch.full((B,), args.page_size, dtype=torch.int32,
                              device=dev)
            wrap = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
            wrap.plan(indptr, indices, last, args.q_heads, args.kv_heads,
                      args.head_dim, args.page_size,
                      q_data_type=dtype, data_type=dtype)
            t_fi = timeit(lambda: wrap.run(q, kv))
            del kv
        except Exception as ex:
            print(f"  B={B} S={S} FlashInfer FAILED: {str(ex)[:90]}")

        by = B * S * kv_per_tok
        e_fa = by / (t_fa / 1e3) / PEAK
        e_fi = by / (t_fi / 1e3) / PEAK if t_fi else float("nan")
        rel = f"{100*(t_fi/t_fa-1):+.1f}%" if t_fi else "n/a"
        print(f"{B:>4} {S:>6} {t_fa:>9.3f}m {t_fi if t_fi else float('nan'):>11.3f}m "
              f"{rel:>10}  {by/1e9:>7.2f}G {e_fa:>8.2f} {e_fi:>8.2f}")
        results.append(dict(ctx=S, batch=B, fa2_ms=t_fa, fi_ms=t_fi,
                            kv_bytes=by, fa2_eff=e_fa, fi_eff=e_fi))
        del kc, vc
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
