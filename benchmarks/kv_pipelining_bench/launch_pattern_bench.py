#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What does the host path's LAUNCH PATTERN cost, apart from its bytes?

WHY. The residual `F` is attributed to "roughly 640 extra kernel launches per
step, about ten per layer across 64 layers at 0.19 ms per layer". Two things
about that number have never been tested:

1. It was inferred from an operator trace at ONE rider count, while the actual
   H2D pattern is `2 * L * R` memcpys per step -- two per contiguous run per
   layer, and the run count is the active rider count. So the launch cost
   should scale with riders, not be fixed.

2. Every measurement in this repo ran with `enforce_eager=True`. For a story
   whose residual is hundreds of small launches per step, nobody has ever asked
   how much of it a CUDA graph would remove.

This benchmark answers both without an engine and without touching the feature.
It replays the pattern -- `2 * L * R` pinned H2D copies of a given size, plus
the ~11 small per-layer ops the rider pass performs -- and times it eager
against graph-captured.

WHAT IT IS FOR, precisely. Two numbers the projection needs:

  * **per-launch cost vs transfer size.** This is the independent check on the
    in-engine `minbytes` arm, which assumes a memcpy's launch overhead does not
    depend on its size. If that assumption fails, this sweep is what shows it,
    and then `minbytes` is a bound rather than a measurement.

  * **the graph-removable fraction** -- an UPPER BOUND on how much of `F` a
    CUDA graph could take away. A bound, not a plan: the hybrid path forces
    eager (`hybrid_attn_ctx.py`), so making it capturable is a design change
    and explicitly out of scope here.

It also gives the C2C-relevant asymmetry directly: hold the launch count fixed
and shrink the bytes 17x, and whatever time remains is what a faster
interconnect cannot buy back.

PORTABILITY IS THE POINT. No vLLM, no model, no checkpoint -- torch and a GPU.
It runs in under a minute, so it is exactly the artifact to hand to anyone with
a GH200: the one measurement that would close the projection's biggest
unknown, which is what a Grace host's launch cost actually is.

Usage:
    python launch_pattern_bench.py --layers 64 --riders 1,2,4,8,16 \
        --bytes-per-copy 8192,65536,524288,2097152 --out launch_pattern.json
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
import sys

print = functools.partial(print, flush=True)  # noqa: A001


def time_call(fn, iters: int, warmup: int = 20) -> float:
    """Median ms per call, CUDA-event timed. Copied idiom from
    gh200_preflight.measure_h2d_gbps / two_source_tax.time_call so the numbers
    are comparable with the rest of the repo."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    return statistics.median(samples)


def build_pattern(layers: int, riders: int, bytes_per_copy: int,
                  small_ops: int, device) -> tuple:
    """Pinned host mirrors + device slots + the closure that replays one step.

    Shapes mirror `RungdHostPool._issue_runs`: per layer, one contiguous H2D
    per rider per tensor (K and V), straight out of pinned memory. The
    `small_ops` are stand-ins for the per-layer rider-pass ops that are
    independent of rider count (index_select of the query rows, the output
    allocation, the scatter back, the seqused mask).
    """
    import torch
    n = max(1, bytes_per_copy // 2)  # fp16 elements
    host_k = [torch.empty(riders * n, dtype=torch.float16).pin_memory()
              for _ in range(layers)]
    host_v = [torch.empty(riders * n, dtype=torch.float16).pin_memory()
              for _ in range(layers)]
    dev_k = [torch.empty(riders * n, dtype=torch.float16, device=device)
             for _ in range(layers)]
    dev_v = [torch.empty(riders * n, dtype=torch.float16, device=device)
             for _ in range(layers)]
    # Small per-layer tensors for the rider-pass stand-in ops.
    q = torch.empty(max(riders, 1), 512, dtype=torch.float16, device=device)
    idx = torch.arange(max(riders, 1), device=device)

    def step():
        for li in range(layers):
            hk, hv = host_k[li], host_v[li]
            dk, dv = dev_k[li], dev_v[li]
            for r in range(riders):
                lo = r * n
                dk.narrow(0, lo, n).copy_(hk.narrow(0, lo, n),
                                          non_blocking=True)
                dv.narrow(0, lo, n).copy_(hv.narrow(0, lo, n),
                                          non_blocking=True)
            for _ in range(small_ops):
                q.index_select(0, idx)

    return step, (host_k, host_v, dev_k, dev_v, q, idx)


def graph_capture(step, device):
    """Capture the pattern into a CUDA graph, or return None if it cannot be."""
    import torch
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        return g
    except Exception as e:                                     # noqa: BLE001
        print(f"  graph capture failed: {type(e).__name__}: {e}")
        return None


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", type=int, default=64)
    p.add_argument("--riders", default="1,2,4,8,16")
    p.add_argument("--bytes-per-copy", default="8192,65536,524288,2097152",
                   help="bytes per contiguous H2D. At block 16 / 8 kv heads / "
                        "head dim 128 / fp16 one block is 256 KiB, so a rider "
                        "at ctx 2048 is 128 blocks = 32 MiB per layer; the "
                        "small end is where launch overhead dominates.")
    p.add_argument("--small-ops", type=int, default=11,
                   help="per-layer rider-pass ops independent of rider count "
                        "(the ~11*L term)")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--out", default="launch_pattern.json")
    a = p.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("FATAL: no CUDA device")
        return 2
    dev = torch.device("cuda")
    riders = [int(x) for x in a.riders.split(",") if x]
    sizes = [int(x) for x in a.bytes_per_copy.split(",") if x]

    print("=" * 74)
    print(f" LAUNCH PATTERN   {torch.cuda.get_device_name(0)}")
    print(f" layers {a.layers}  riders {riders}  small_ops/layer {a.small_ops}")
    print(f" torch {torch.__version__}")
    print("=" * 74)

    rows = []
    for size in sizes:
        for r in riders:
            step, keep = build_pattern(a.layers, r, size, a.small_ops, dev)
            launches = 2 * a.layers * r
            eager_ms = time_call(step, a.iters)
            gbytes = launches * size / 1e9

            g = graph_capture(step, dev)
            graph_ms = time_call(g.replay, a.iters) if g is not None else None

            # Per-launch cost, with the transfer time the link would need
            # removed. Uses the eager arm, since that is what the engine does.
            row = {
                "bytes_per_copy": size, "riders": r, "layers": a.layers,
                "h2d_launches": launches,
                "gb_per_step": gbytes,
                "eager_ms": eager_ms,
                "graph_ms": graph_ms,
                "eager_us_per_launch": eager_ms * 1e3 / launches,
                "graph_removable_ms": (eager_ms - graph_ms)
                if graph_ms is not None else None,
                "graph_removable_frac": ((eager_ms - graph_ms) / eager_ms)
                if graph_ms is not None and eager_ms > 0 else None,
                "implied_gbps": (gbytes / (eager_ms / 1e3))
                if eager_ms > 0 else None,
            }
            rows.append(row)
            gm = f"{graph_ms:8.3f}" if graph_ms is not None else "       -"
            gf = (f"{row['graph_removable_frac'] * 100:5.1f}%"
                  if row["graph_removable_frac"] is not None else "     -")
            print(f"  {size:>9}B r={r:<3} launches={launches:<5} "
                  f"eager={eager_ms:8.3f} graph={gm} removable={gf} "
                  f"us/launch={row['eager_us_per_launch']:6.2f} "
                  f"GB/s={row['implied_gbps'] or 0:6.2f}")
            del step, keep, g
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- verdicts
    print("\n" + "=" * 74)
    print(" IS PER-LAUNCH COST INDEPENDENT OF TRANSFER SIZE?  (concern C3)")
    print(" If it is, the in-engine `minbytes` arm measures the launch cost.")
    print(" If it is not, `minbytes` is a BOUND and this sweep is the truth.")
    for r in riders:
        per = [(x["bytes_per_copy"], x["eager_us_per_launch"])
               for x in rows if x["riders"] == r]
        if len(per) < 2:
            continue
        lo, hi = per[0][1], per[-1][1]
        print(f"  r={r:<3} us/launch {lo:6.2f} at {per[0][0]}B -> "
              f"{hi:6.2f} at {per[-1][0]}B   "
              f"({'size-INDEPENDENT' if hi <= 1.25 * lo else 'size-DEPENDENT'})")

    print("\n LAUNCH COST vs RIDER COUNT  (is the overhead fixed, as the model"
          " assumes, or per-rider?)")
    small = min(sizes)
    per_r = [(x["riders"], x["eager_ms"]) for x in rows
             if x["bytes_per_copy"] == small]
    if len(per_r) >= 2:
        for rr, ms in per_r:
            print(f"  r={rr:<3} {ms:8.3f} ms at {small}B/copy "
                  f"({ms / rr:6.3f} ms per rider)")
        print("  -> a FIXED cost would be flat down this column; it is the "
              "2*L*R structure if it rises roughly linearly.")

    ok = [x for x in rows if x["graph_removable_frac"] is not None]
    if ok:
        best = max(x["graph_removable_frac"] for x in ok)
        print(f"\n CUDA-GRAPH UPPER BOUND: at most "
              f"{best * 100:.1f}% of this pattern's time is launch overhead a "
              f"graph could remove. A BOUND, not a plan -- the hybrid path "
              f"forces eager, and making it capturable is a design change.")
    else:
        print("\n CUDA-GRAPH UPPER BOUND: not obtained (capture failed on "
              "every configuration); report as unmeasured, do not assume.")

    out = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "layers": a.layers, "riders": riders, "sizes": sizes,
        "small_ops_per_layer": a.small_ops, "iters": a.iters,
        "note": "engine-free replay of the rung-d H2D launch pattern; "
                "portable to any CUDA box including GH200",
        "rows": rows,
    }
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.abspath(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
