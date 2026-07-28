#!/usr/bin/env python3
"""Aggregate Python call-stacks behind copy_/to ops in a WITH_STACK Kineto trace.

Pinpoints which source lines launch the steady-state HtoD memcpy storm.
Usage: stack_sources.py TRACE.json[.gz] [--tail-frac 0.4]
"""
from __future__ import annotations
import argparse, gzip, json, sys
from collections import defaultdict


def load(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--tail-frac", type=float, default=0.4)
    ap.add_argument("--names", default="aten::copy_,aten::to,aten::_to_copy,aten::index_put_,aten::index,aten::index_select")
    args = ap.parse_args()
    d = load(args.trace)
    evs = d["traceEvents"]
    names = set(args.names.split(","))

    # time window from kernel events
    kts = [float(e["ts"]) for e in evs if e.get("cat") == "kernel" and "ts" in e]
    tmin, tmax = min(kts), max(kts)
    cut = tmin + (1 - args.tail_frac) * (tmax - tmin)

    # inspect what stack field exists
    sample = None
    for e in evs:
        if e.get("name") in names and e.get("cat") == "cpu_op":
            sample = e
            break
    if sample is not None:
        print("sample cpu_op keys:", list(sample.keys()))
        print("sample args keys:", list((sample.get("args") or {}).keys()))

    # aggregate by last vllm frame(s)
    by_leaf = defaultdict(int)
    by_name = defaultdict(int)
    total = 0
    for e in evs:
        if e.get("cat") != "cpu_op" or e.get("name") not in names:
            continue
        if "ts" in e and float(e["ts"]) < cut:
            continue
        total += 1
        by_name[e["name"]] += 1
        args_ = e.get("args") or {}
        stack = args_.get("Call stack") or args_.get("call_stack") or ""
        frames = [f for f in stack.split(";") if f.strip()]
        # pick last frame mentioning vllm
        leaf = None
        for f in reversed(frames):
            if "vllm/v1" in f or "kv_" in f or "cpu_kv_table" in f or "pipelin" in f:
                leaf = f.strip()
                break
        if leaf is None and frames:
            leaf = frames[-1].strip()
        by_leaf[leaf or "<no-stack>"] += 1

    print(f"\nsteady-window cpu_op (tail {args.tail_frac}): total={total}")
    print("by op name:")
    for n, c in sorted(by_name.items(), key=lambda x: -x[1]):
        print(f"  {c:7d}  {n}")
    print("\ntop launching source frames:")
    for leaf, c in sorted(by_leaf.items(), key=lambda x: -x[1])[:25]:
        print(f"  {c:7d}  {leaf[:130]}")


if __name__ == "__main__":
    sys.exit(main())
