#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure the TWO-SOURCE ATTENTION TAX.

Direction A ("mixture") serves a decode batch from two KV sources: some
requests resident in HBM, some streamed from host memory. Even ignoring all
transfer cost, splitting one attention call into two costs something:
  - two kernel launches instead of one
  - each call has fewer CTAs -> possibly a worse FlashDecoding split regime
  - reduced occupancy / tail effects per call

That cost is INTERCONNECT-INDEPENDENT, so measuring it on A100 transfers
directly to GH200 -- and it is the piece that decides whether the mixture's
structural advantage (+C/u requests) survives at all.

Isolation: **all KV stays in HBM in both arms.** No host memory, no transfer,
no pool. The only difference is whether the same rows are attended by one
flash_attn_varlen_func call or two. Anything we measure is pure structure.

  arm A (single): one call over B rows
  arm B (split):  two calls over B1 and B2 rows, B1 + B2 = B

Reported per decode step (x num_layers), since that is how it shows up in ITL.
"""

from __future__ import annotations

import argparse
import itertools
import json

import torch

from vllm.vllm_flash_attn import flash_attn_varlen_func

try:
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
except Exception:
    unified_attention = None


def build_cache(n_blocks, block_size, n_kv_heads, head_dim, dtype, dev):
    k = torch.randn(n_blocks, block_size, n_kv_heads, head_dim,
                    dtype=dtype, device=dev)
    v = torch.randn_like(k)
    return k, v


def time_call(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters   # ms per call-group


def capture_graph(fn):
    """Capture fn into a CUDA graph — vLLM runs decode under FULL cudagraphs,
    so launch overhead is NOT part of the real tax. Without this the split arm
    is unfairly charged for its extra kernel launch."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def main():
    p = argparse.ArgumentParser()
    # Qwen2.5-32B decode shape by default: 64 layers, 40 q heads, 8 kv heads
    p.add_argument("--q-heads", type=int, default=40)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=64)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--batches", default="8,16,22,32,48")
    p.add_argument("--contexts", default="2048,4096")
    p.add_argument("--splits", default="0.25,0.5",
                   help="fraction of the batch in the SECOND (host-tier) group")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out", default="two_source_tax.json")
    p.add_argument("--cuda-graph", action="store_true",
                   help="capture both arms in CUDA graphs (vLLM decode default)")
    p.add_argument("--backend", default="fa2", choices=["fa2", "triton"],
                   help="decode kernel; triton is faster at large batch on A100")
    args = p.parse_args()

    dev, dtype = "cuda", torch.float16
    batches = [int(x) for x in args.batches.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]
    splits = [float(x) for x in args.splits.split(",")]
    scale = args.head_dim ** -0.5

    print(f"two-source attention tax | q_heads={args.q_heads} "
          f"kv_heads={args.kv_heads} head_dim={args.head_dim} "
          f"layers={args.layers} block={args.block_size}")
    print(f"{'B':>4} {'S':>6} {'split':>11} {'single':>9} {'two-call':>9} "
          f"{'tax/layer':>10} {'tax/step':>10} {'tax %':>7}")

    results = []
    for S, B in itertools.product(contexts, batches):
        blocks_per_seq = (S + args.block_size - 1) // args.block_size
        n_blocks = blocks_per_seq * B + 8
        kc, vc = build_cache(n_blocks, args.block_size, args.kv_heads,
                             args.head_dim, dtype, dev)
        # each sequence gets its own distinct blocks (no sharing)
        block_table = torch.arange(
            blocks_per_seq * B, dtype=torch.int32, device=dev
        ).view(B, blocks_per_seq)
        q = torch.randn(B, args.q_heads, args.head_dim, dtype=dtype, device=dev)
        out = torch.empty_like(q)
        seqused = torch.full((B,), S, dtype=torch.int32, device=dev)
        cu_q = torch.arange(B + 1, dtype=torch.int32, device=dev)

        def call(lo, hi):
            n = hi - lo
            if args.backend == "triton":
                unified_attention(
                    q=q[lo:hi], k=kc, v=vc, out=out[lo:hi],
                    cu_seqlens_q=cu_q[: n + 1], max_seqlen_q=1,
                    seqused_k=seqused[lo:hi], max_seqlen_k=S,
                    softmax_scale=scale, causal=True,
                    window_size=(-1, -1), block_table=block_table[lo:hi],
                    softcap=0.0, q_descale=None, k_descale=None,
                    v_descale=None,
                )
            else:
                flash_attn_varlen_func(
                    q=q[lo:hi], k=kc, v=vc, out=out[lo:hi],
                    cu_seqlens_q=cu_q[: n + 1], max_seqlen_q=1,
                    seqused_k=seqused[lo:hi], max_seqlen_k=S,
                    softmax_scale=scale, causal=True,
                    block_table=block_table[lo:hi],
                )

        if args.cuda_graph:
            g1 = capture_graph(lambda: call(0, B))
            single = time_call(g1.replay, args.iters)
        else:
            single = time_call(lambda: call(0, B), args.iters)

        for frac in splits:
            b2 = max(1, int(round(B * frac)))
            b1 = B - b2
            if b1 < 1:
                continue

            def two():
                call(0, b1)
                call(b1, B)

            if args.cuda_graph:
                g2 = capture_graph(two)
                split_t = time_call(g2.replay, args.iters)
            else:
                split_t = time_call(two, args.iters)
            tax_layer = split_t - single
            tax_step = tax_layer * args.layers
            pct = 100.0 * tax_layer / single
            print(f"{B:>4} {S:>6} {b1:>4}+{b2:<6} {single:>8.3f}m "
                  f"{split_t:>8.3f}m {tax_layer:>9.3f}m {tax_step:>9.2f}m "
                  f"{pct:>6.1f}%")
            results.append(dict(batch=B, ctx=S, b1=b1, b2=b2,
                                single_ms=single, split_ms=split_t,
                                tax_per_layer_ms=tax_layer,
                                tax_per_step_ms=tax_step, tax_pct=pct))
        del kc, vc
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    if results:
        pcts = [r["tax_pct"] for r in results]
        print(f"\ntax over all configs: min {min(pcts):.1f}%  "
              f"median {sorted(pcts)[len(pcts)//2]:.1f}%  max {max(pcts):.1f}%")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
