#!/usr/bin/env python3
"""Analyze a vLLM torch (Kineto) trace to localize the orchestration tax.

Splits the profiled decode window into:
  - GPU-busy time   (union of device kernel/memcpy/memset intervals)
  - GPU-idle time   (span - busy)  -> host/launch-bound bubbles
  - kernel count + top kernels by total duration
  - host (CPU) op time on the worker process

Decisive question: does R3 spend its extra wall-time as GPU-idle gaps
(host/launch-bound -> DPU-addressable) or as extra GPU-kernel-active time
(extra device work)?

Usage:
  analyze_trace.py TRACE.json[.gz] [--label R3]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict


def _load(path: str) -> dict:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def _union(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    total += ce - cs
    return total


DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--label", default="")
    ap.add_argument("--steps", type=int, default=0,
                    help="known #decode steps profiled (for per-step normalization)")
    args = ap.parse_args()

    data = _load(args.trace)
    evs = data.get("traceEvents", [])

    # Identify device (GPU) tids: events whose cat is a device category.
    dev_tids: set = set()
    for e in evs:
        if e.get("cat") in DEVICE_CATS and e.get("dur") is not None:
            dev_tids.add((e.get("pid"), e.get("tid")))

    dev_intervals: list[tuple[float, float]] = []
    kern_dur: dict[str, float] = defaultdict(float)
    kern_cnt: dict[str, int] = defaultdict(int)
    dmin = float("inf")
    dmax = float("-inf")
    n_dev = 0
    for e in evs:
        cat = e.get("cat")
        if cat in DEVICE_CATS and e.get("dur") is not None and "ts" in e:
            ts = float(e["ts"]); dur = float(e["dur"])
            if dur <= 0:
                continue
            dev_intervals.append((ts, ts + dur))
            name = e.get("name", "?")
            kern_dur[name] += dur
            kern_cnt[name] += 1
            n_dev += 1
            dmin = min(dmin, ts); dmax = max(dmax, ts + dur)

    if n_dev == 0:
        print("No device events found. Categories present:")
        cats = defaultdict(int)
        for e in evs:
            cats[e.get("cat")] += 1
        for c, n in sorted(cats.items(), key=lambda x: -x[1])[:20]:
            print(f"  {c}: {n}")
        return 1

    span_us = dmax - dmin
    busy_us = _union(list(dev_intervals))
    idle_us = span_us - busy_us

    lbl = f"[{args.label}] " if args.label else ""
    print(f"{lbl}device events: {n_dev}")
    print(f"{lbl}GPU timeline span : {span_us/1000:9.3f} ms")
    print(f"{lbl}GPU busy (union)  : {busy_us/1000:9.3f} ms  ({100*busy_us/span_us:5.1f}%)")
    print(f"{lbl}GPU idle (gaps)   : {idle_us/1000:9.3f} ms  ({100*idle_us/span_us:5.1f}%)")
    if args.steps > 0:
        print(f"{lbl}per-step ({args.steps} steps): "
              f"span {span_us/args.steps/1000:.3f} ms | "
              f"busy {busy_us/args.steps/1000:.3f} ms | "
              f"idle {idle_us/args.steps/1000:.3f} ms")

    print(f"\n{lbl}top kernels by total GPU time:")
    for name in sorted(kern_dur, key=lambda k: -kern_dur[k])[:15]:
        print(f"  {kern_dur[name]/1000:8.3f} ms  x{kern_cnt[name]:5d}  {name[:70]}")

    # Host (CPU) op time on worker pid: cat == 'cpu_op'
    cpu_dur: dict[str, float] = defaultdict(float)
    cpu_cnt: dict[str, int] = defaultdict(int)
    for e in evs:
        if e.get("cat") in ("cpu_op", "user_annotation") and e.get("dur"):
            cpu_dur[e.get("name", "?")] += float(e["dur"])
            cpu_cnt[e.get("name", "?")] += 1
    if cpu_dur:
        print(f"\n{lbl}top host cpu_op by total time:")
        for name in sorted(cpu_dur, key=lambda k: -cpu_dur[k])[:15]:
            print(f"  {cpu_dur[name]/1000:8.3f} ms  x{cpu_cnt[name]:5d}  {name[:70]}")

    # Distribution of key sync/control ops across 5 time-quantiles of the GPU
    # timeline (front-loaded => admission artifact; uniform => per-step steady).
    watch = ["aten::item", "aten::_local_scalar_dense", "Memcpy HtoD (Pageable -> Device)",
             "aten::copy_", "aten::select", "aten::_to_copy",
             "unified_attention_with_output"]
    nb = 5
    bins = {w: [0] * nb for w in watch}
    for e in evs:
        nm = e.get("name")
        if nm in bins and "ts" in e:
            q = min(nb - 1, max(0, int(nb * (float(e["ts"]) - dmin) / max(1.0, span_us))))
            bins[nm][q] += 1
    print(f"\n{lbl}op counts per timeline-quantile (Q0=start .. Q4=end):")
    print(f"  {'op':40s}  {'Q0':>8s} {'Q1':>8s} {'Q2':>8s} {'Q3':>8s} {'Q4':>8s}  total")
    for w in watch:
        b = bins[w]
        print(f"  {w[:40]:40s}  " + " ".join(f"{x:8d}" for x in b) + f"  {sum(b)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
