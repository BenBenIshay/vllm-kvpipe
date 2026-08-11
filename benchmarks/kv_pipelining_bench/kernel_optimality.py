#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Are we measuring the best available decode-attention kernel?

The two-source tax result is only meaningful if BOTH arms run the best kernel
config available. Two specific risks:

  1. num_splits: FlashDecoding's split-KV count is chosen by a heuristic
     (num_splits=0 => auto). A bad auto choice would make the single-call arm
     artificially slow -- which is exactly the shape of the B=32 anomaly we saw
     both in-engine and in the microbenchmark. If a tuned num_splits removes
     the anomaly, "splitting is faster" was a heuristic bug, not a property.

  2. Backend: vLLM auto-selected FLASH_ATTN out of
     (FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION). FlashInfer's paged
     decode kernel is a credible alternative on this shape.

Both arms of any tax comparison must use each one's OWN best config, or the
comparison is rigged.
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
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--batches", default="4,8,13,16,24,32,48")
    p.add_argument("--contexts", default="4096,8192")
    p.add_argument("--splits", default="0,1,2,4,8,16,32")
    p.add_argument("--out", default="kernel_optimality.json")
    args = p.parse_args()

    dev, dtype = "cuda", torch.float16
    scale = args.head_dim ** -0.5
    batches = [int(x) for x in args.batches.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]
    nsplits = [int(x) for x in args.splits.split(",")]

    try:
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
        from vllm.vllm_flash_attn import is_fa_version_supported
        print(f"FA version vLLM selects: {get_flash_attn_version()}   "
              f"FA2 ok={is_fa_version_supported(2)} "
              f"FA3 ok={is_fa_version_supported(3)}")
    except Exception as ex:
        print(f"(fa version probe failed: {ex})")
    print(f"device: {torch.cuda.get_device_name(0)}\n")

    results = []
    for S, B in itertools.product(contexts, batches):
        bps = (S + args.block_size - 1) // args.block_size
        kc = torch.randn(bps * B + 8, args.block_size, args.kv_heads,
                         args.head_dim, dtype=dtype, device=dev)
        vc = torch.randn_like(kc)
        bt = torch.arange(bps * B, dtype=torch.int32, device=dev).view(B, bps)
        q = torch.randn(B, args.q_heads, args.head_dim, dtype=dtype, device=dev)
        out = torch.empty_like(q)
        seq = torch.full((B,), S, dtype=torch.int32, device=dev)
        cu = torch.arange(B + 1, dtype=torch.int32, device=dev)

        row = {"ctx": S, "batch": B, "by_split": {}}
        for ns in nsplits:
            def f(ns=ns):
                flash_attn_varlen_func(
                    q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=1,
                    seqused_k=seq, max_seqlen_k=S, softmax_scale=scale,
                    causal=True, block_table=bt, num_splits=ns)
            try:
                row["by_split"][ns] = timeit(f)
            except Exception as ex:
                row["by_split"][ns] = None
                print(f"  B={B} S={S} num_splits={ns} FAILED: "
                      f"{str(ex)[:60]}")

        ok = {k: v for k, v in row["by_split"].items() if v}
        auto = ok.get(0)
        best_ns = min(ok, key=ok.get)
        row["auto_ms"], row["best_ms"], row["best_split"] = auto, ok[best_ns], best_ns
        gain = 100 * (auto / ok[best_ns] - 1) if auto else 0
        flag = "  <<< AUTO IS BAD" if gain > 8 else ""
        print(f"B={B:>3} S={S:>5}  auto(0)={auto:.3f}ms  "
              f"best=num_splits{best_ns}:{ok[best_ns]:.3f}ms  "
              f"auto is {gain:+.1f}% off best{flag}")
        print("      " + "  ".join(
            f"ns{k}={v:.3f}" for k, v in sorted(ok.items())))
        results.append(row)
        del kc, vc
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
