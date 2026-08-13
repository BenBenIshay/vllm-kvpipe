#!/usr/bin/env python3
"""Where do the 24 ms per `stream_layer` call actually go?

Profiling the host tier at Qwen-32B / ctx 2048 / 4 requests put `stream_layer`
at 72% of wall (10.9 s of 15.2 s) over 448 calls = 24.4 ms each. Each call moves
one layer's KV for the cohort: 512 blocks x 64 KiB = 32 MiB. At the measured
PCIe rate of 25 GB/s that is 1.3 ms, so the path is running ~19x slower than the
link. Turning on the async double-buffering (`VLLM_KVPIPE_RUNGD_ASYNC=1`)
changed nothing (10988 ms vs 10940 ms), so the cost is NOT the H2D failing to
overlap -- it is inside the call.

`stream_layer` does two things:
    1. `_gather_into`  : host mirror -> pinned staging buffer, on the CPU
    2. H2D             : staging buffer -> the GPU rolling slot

But the host mirror is ALREADY pinned, and with per-request contiguous extents
the active blocks form a few long runs. So step 1 may be a pure waste: we could
issue the H2D straight from the host mirror's run slices and skip the CPU copy
entirely.

This times the three variants at the real shape, with no engine involved:

    gather+copy   what the code does today
    gather only   the CPU-side staging copy alone
    direct        H2D from the pinned mirror's runs, no staging  <-- proposed

Run: srun ... .venv/bin/python benchmarks/kv_pipelining_bench/stream_layer_microbench.py
"""

from __future__ import annotations

import argparse

import torch

MiB = 1024 * 1024


def timed(fn, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=512,
                    help="active cohort blocks (4 reqs x 2048 tok / 16)")
    ap.add_argument("--h-max", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--runs", type=int, default=4,
                    help="contiguous runs the active set forms (= n requests "
                         "with per-request extents)")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    dev = torch.device("cuda")
    dt = torch.float16
    W, H, bs, nkv, hd = (args.blocks, args.h_max, args.block_size,
                         args.kv_heads, args.head_dim)

    host = torch.empty((2, H, bs, nkv, hd), dtype=dt, device="cpu",
                       pin_memory=True)
    stage = torch.empty((2, W, bs, nkv, hd), dtype=dt, device="cpu",
                        pin_memory=True)
    slot = torch.empty((2, W, bs, nkv, hd), dtype=dt, device=dev)

    per_block = 2 * bs * nkv * hd * 2
    total = W * per_block
    print(f"cohort {W} blocks x {per_block // 1024} KiB = {total / MiB:.0f} MiB "
          f"per layer, {args.runs} contiguous run(s)")

    # Active set as `runs` equal contiguous extents, like per-request extents.
    ln = W // args.runs
    runs = [(i * ln, i * ln, ln) for i in range(args.runs)]
    idx = torch.tensor([hs + j for hs, _, l in runs for j in range(l)],
                       dtype=torch.long)

    def gather_into():
        for hs, cs, l in runs:
            stage.narrow(1, cs, l).copy_(host.narrow(1, hs, l))

    def h2d():
        slot.copy_(stage, non_blocking=True)

    def gather_plus_copy():           # what stream_layer does today
        gather_into()
        h2d()

    def direct():                     # proposed: skip the CPU staging entirely
        for hs, cs, l in runs:
            slot.narrow(1, cs, l).copy_(host.narrow(1, hs, l),
                                        non_blocking=True)

    def gather_index_select():        # the fragmented fallback path
        torch.index_select(host, 1, idx, out=stage)

    # BLOCK-MAJOR variant. The mirror is [2, H, ...], so a run of blocks is two
    # strided chunks (one for K, one for V) -- which is why the direct H2D above
    # is SLOWER than staging. Lay the mirror out block-first, [H, 2, ...], and a
    # run becomes one contiguous byte range: it can go straight from pinned host
    # to the device with no CPU copy at all.
    host_bm = torch.empty((H, 2, bs, nkv, hd), dtype=dt, device="cpu",
                          pin_memory=True)
    slot_bm = torch.empty((W, 2, bs, nkv, hd), dtype=dt, device=dev)

    def direct_block_major():
        for hs, cs, l in runs:
            slot_bm.narrow(0, cs, l).copy_(host_bm.narrow(0, hs, l),
                                           non_blocking=True)

    results = {}
    for name, fn in (("gather+copy (today)", gather_plus_copy),
                     ("  gather only (CPU)", gather_into),
                     ("  h2d only", h2d),
                     ("  index_select gather", gather_index_select),
                     ("direct H2D (proposed)", direct),
                     ("direct H2D block-major", direct_block_major)):
        ms = timed(fn, args.iters)
        gbps = total / (ms / 1e3) / 1e9
        results[name.strip()] = (ms, gbps)
        print(f"{name:>24}: {ms:7.2f} ms   {gbps:6.2f} GB/s")

    today = results["gather+copy (today)"][0]
    best_name, (best_ms, best_gbps) = min(results.items(), key=lambda kv: kv[1][0])
    print()
    print(f"fastest: {best_name} at {best_ms:.2f} ms ({best_gbps:.1f} GB/s); "
          f"{today / best_ms:.1f}x vs today's {today:.2f} ms")
    print(f"per DECODE STEP (64 layers): {today * 64 / 1000:.3f}s -> "
          f"{best_ms * 64 / 1000:.3f}s")
    print("compute/step for this model is ~45 ms, so the transfer path must "
          f"reach <=0.70 ms/layer to hide; best here is {best_ms:.2f} ms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
