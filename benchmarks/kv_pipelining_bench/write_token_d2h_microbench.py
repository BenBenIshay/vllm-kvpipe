#!/usr/bin/env python3
"""`write_token`'s device->host copy is a pipeline drain, not a copy.

Measured in the engine (profile_host_tier.py, Qwen-32B, ctx 2048, all-rider):

    riders   stream_layer GPU   write_token CPU
       1          23.5 ms           25.6 ms
       2          45.8 ms           46.8 ms
       3          67.9 ms           68.0 ms

~21 ms of CPU per rider per step to append ONE token's KV -- a few KB spread
over 64 layers, or 0.33 ms per layer to move ~4 KB. No copy of that size costs
that. What costs it is this, in `RungdHostPool.write_token`:

    k_cpu, v_cpu = keys.to("cpu"), values.to("cpu")

a BLOCKING device->host copy, twice per layer, 128 times per step. Each one
waits for the CUDA pipeline to drain. It also destroys the overlap the whole
design depends on: `_launch_prefetch` puts layer L+1's transfer on a side
stream, and then layer L's write_token synchronises the main stream and
serialises everything behind it. This is the most likely explanation for the
3.3x microbench-vs-engine gap that was recorded as unexplained, and for async
double-buffering having measured worthless when the gather was blamed.

The point of measuring it HERE rather than in the engine is that a blocking
copy on an IDLE GPU is nearly free -- the cost only appears when there is
queued work to drain. So this benchmark queues realistic work first; without
that it would measure nothing and wrongly exonerate the line.

Run: srun ... .venv/bin/python \
       benchmarks/kv_pipelining_bench/write_token_d2h_microbench.py
"""

from __future__ import annotations

import argparse

import torch

DT = torch.float16


def queue_work(busy: torch.Tensor, rounds: int) -> None:
    """Enqueue GPU work so a blocking copy has a pipeline to drain."""
    for _ in range(rounds):
        busy.mul_(1.0001).add_(1e-6)


def bench(n_riders: int, layers: int, nkv: int, hd: int, h_blocks: int,
          bs: int, busy_rounds: int, iters: int) -> dict:
    dev = "cuda"
    # Pinned host mirrors, exactly as the pool allocates them.
    hk = [torch.empty((h_blocks, bs, nkv, hd), dtype=DT, pin_memory=True)
          for _ in range(layers)]
    keys = [torch.randn(n_riders, nkv, hd, device=dev, dtype=DT)
            for _ in range(layers)]
    pos = torch.arange(n_riders, dtype=torch.long) * bs
    busy = torch.randn(4096, 4096, device=dev, dtype=DT)

    def blocking() -> None:
        for li in range(layers):
            k_cpu = keys[li].to("cpu")                      # <-- the drain
            hk[li].view(h_blocks * bs, nkv, hd).index_copy_(0, pos, k_cpu)

    def nonblocking() -> None:
        for li in range(layers):
            flat = hk[li].view(h_blocks * bs, nkv, hd)
            for r in range(n_riders):
                flat[pos[r]:pos[r] + 1].copy_(keys[li][r:r + 1],
                                              non_blocking=True)

    out = {}
    for name, fn in (("blocking", blocking), ("non_blocking", nonblocking)):
        for _ in range(2):                                   # warm up
            queue_work(busy, busy_rounds)
            fn()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        import time
        torch.cuda.synchronize()
        t0.record()
        wall0 = time.perf_counter()
        for _ in range(iters):
            queue_work(busy, busy_rounds)                    # the pipeline
            fn()
        wall = (time.perf_counter() - wall0) * 1000.0 / iters
        t1.record()
        torch.cuda.synchronize()
        out[name] = {"cpu_ms_per_step": round(wall, 2),
                     "gpu_ms_per_step": round(t0.elapsed_time(t1) / iters, 2)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--riders", default="1,2,3")
    ap.add_argument("--layers", type=int, default=64)      # Qwen-32B
    ap.add_argument("--nkv", type=int, default=8)
    ap.add_argument("--hd", type=int, default=128)
    ap.add_argument("--h-blocks", type=int, default=1600)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--busy-rounds", type=int, default=40,
                    help="GPU work queued per step; a blocking copy is nearly "
                         "free on an idle GPU, so without this the benchmark "
                         "measures nothing")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    print(f"layers={args.layers} nkv={args.nkv} hd={args.hd} "
          f"({args.layers * args.nkv * args.hd * 2 * 2 / 1024:.1f} KiB of K+V "
          f"per rider per step)")
    print(f"{'riders':>6} {'blocking cpu':>13} {'async cpu':>10} "
          f"{'speedup':>8}")
    for r in [int(x) for x in args.riders.split(",")]:
        res = bench(r, args.layers, args.nkv, args.hd, args.h_blocks,
                    args.block_size, args.busy_rounds, args.iters)
        b = res["blocking"]["cpu_ms_per_step"]
        a = res["non_blocking"]["cpu_ms_per_step"]
        print(f"{r:>6} {b:>13} {a:>10} {b / a if a else 0:>7.1f}x")
    print()
    print("The engine sees ~21 ms of write_token CPU per rider per step. If "
          "the blocking column is the same order, the line is the cost and "
          "the async form is the fix; if it is far smaller, the drain is being "
          "provoked by something this benchmark does not reproduce and the "
          "engine needs a trace instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
