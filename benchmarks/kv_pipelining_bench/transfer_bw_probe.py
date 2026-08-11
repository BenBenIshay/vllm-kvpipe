#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What host->device bandwidth can a rider cohort ACTUALLY sustain?

rider_count = ITL * X_eff / (S * k), so X_eff multiplies the entire value of
direction A. Our roofline assumes 25.1 GB/s on A100/PCIe4; this measures what
the real access granularities deliver.

We stream into a ROLLING ONE-LAYER WINDOW (the k-window design), so the device
buffer is one layer's worth of the rider cohort's KV -- not the whole cohort.
The question is how bandwidth varies with COPY GRANULARITY:

  per-layer chunk   : one bulk copy per layer          <- the design we want
  per-block chunk   : one copy per KV block            <- paged granularity
  per-rider-block   : one copy per rider per block     <- finer
  per-row           : tiny copies                      <- naive rung-d today

Plus an overlap test: does the transfer hide under concurrent GPU compute?
"""

from __future__ import annotations

import argparse
import json

import torch


def timed(fn, iters, warmup=3):
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
    p.add_argument("--riders", type=int, default=5)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--layers", type=int, default=80)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--out", default="transfer_bw.json")
    args = p.parse_args()

    dev = "cuda"
    k_tok_layer = 2 * args.kv_heads * args.head_dim * 2     # K+V fp16, 1 layer
    per_layer = args.riders * args.ctx * k_tok_layer        # all riders, 1 layer
    step_total = per_layer * args.layers                    # full step traffic
    n_blocks = args.ctx // args.block_size

    print(f"rider cohort: {args.riders} reqs x {args.ctx} tok x {args.layers} L"
          f"  = {step_total/1e9:.2f} GB per decode step")
    print(f"rolling window (1 layer, all riders) = {per_layer/1e6:.1f} MB\n")

    host = torch.empty(per_layer, dtype=torch.uint8, pin_memory=True)
    devbuf = torch.empty(per_layer, dtype=torch.uint8, device=dev)
    results = {}

    grans = [
        ("per-layer bulk", per_layer, "the design we want"),
        ("per-block", per_layer // n_blocks, "paged granularity"),
        ("per-rider-block", per_layer // (n_blocks * args.riders), "finer"),
        ("per-row (4KB)", 4096, "naive rung-d today"),
    ]
    print(f"{'granularity':<18} {'chunk':>10} {'copies':>8} {'ms/layer':>10} "
          f"{'GB/s':>8}  note")
    for name, chunk, note in grans:
        chunk = max(4096, int(chunk))
        ncopy = per_layer // chunk
        if ncopy > 20000:                    # keep the probe bounded
            ncopy = 20000
        def f(chunk=chunk, ncopy=ncopy):
            for i in range(ncopy):
                o = i * chunk
                devbuf[o:o + chunk].copy_(host[o:o + chunk], non_blocking=True)
        it = max(3, args.iters // (4 if ncopy > 1000 else 1))
        t = timed(f, it)
        # scale to a full layer if we capped the copy count
        t_layer = t * ((per_layer // chunk) / ncopy)
        gbs = per_layer / (t_layer / 1e3) / 1e9
        results[name] = dict(chunk_bytes=chunk, gbs=gbs, ms_per_layer=t_layer)
        print(f"{name:<18} {chunk/1e3:>9.1f}K {per_layer//chunk:>8} "
              f"{t_layer:>9.2f}m {gbs:>7.2f}  {note}")

    # overlap: does a per-layer copy hide under concurrent compute?
    side = torch.cuda.Stream()
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    def compute():
        for _ in range(20):
            torch.mm(a, a)
    t_c = timed(compute, args.iters)
    def overlapped():
        with torch.cuda.stream(side):
            devbuf.copy_(host, non_blocking=True)
        compute()
        torch.cuda.current_stream().wait_stream(side)
    t_o = timed(overlapped, args.iters)
    t_copy = results["per-layer bulk"]["ms_per_layer"]
    hidden = 100 * (1 - max(0.0, t_o - t_c) / t_copy)
    print(f"\noverlap: compute {t_c:.2f} ms | compute+copy {t_o:.2f} ms | "
          f"copy alone {t_copy:.2f} ms  -> {hidden:.0f}% of the transfer HIDDEN")
    results["overlap"] = dict(compute_ms=t_c, overlapped_ms=t_o,
                              copy_ms=t_copy, hidden_pct=hidden)

    best = results["per-layer bulk"]["gbs"]
    print(f"\nroofline assumes 25.1 GB/s -> measured per-layer bulk "
          f"{best:.2f} GB/s = {100*best/25.1:.0f}% of assumption")
    print(f"step transfer time at that rate: "
          f"{step_total/(best*1e9)*1e3:.1f} ms for {step_total/1e9:.2f} GB")

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
