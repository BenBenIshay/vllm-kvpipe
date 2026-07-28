#!/usr/bin/env python3
"""Experiment B2: prefix-cache / KV-block management CPU cost.

Closed-loop non-streaming load with a LONG prompt prefix, two arms:
  - shared : every request shares the SAME long prefix -> cache hits, block reuse
             (minimal hashing/alloc after warmup)
  - unique : every request has a UNIQUE long prefix -> every block hashed, stored,
             and evicted under pressure -> heavy KV-block management

Both run with prefix caching ON. The unique-minus-shared delta isolates the
hashing + cache-map + alloc/free/evict CPU (which runs on the scheduler step
thread). Server pinned to cores 2-7, client to 0-1.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_baseline_sweep import Sampler  # noqa: E402

_FILLER = ("The quarterly distributed-systems report analyzes throughput, "
           "latency, and cache behavior across heterogeneous clusters. ")


def make_prefix(target_tokens: int, unique: bool) -> str:
    head = f"SESSION-{random.randint(0, 10**12)}-{random.random()} " if unique else ""
    body = _FILLER * (max(1, (target_tokens * 4) // len(_FILLER)))
    return (head + body)[: target_tokens * 4]


def one_request(port, model, prefix, max_tokens, timeout):
    body = {"model": model, "prompt": prefix, "max_tokens": max_tokens,
            "temperature": 0.0, "ignore_eos": True, "stream": False}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        d = json.loads(resp.read())
    return int(d.get("usage", {}).get("completion_tokens", 0))


def run(port, model, prefix_tokens, max_tokens, C, duration, unique, timeout,
        shared_prefix):
    stop = threading.Event()
    counts = [0]
    reqs = [0]
    lock = threading.Lock()

    def worker():
        local_t = local_r = 0
        while not stop.is_set():
            pfx = make_prefix(prefix_tokens, True) if unique else shared_prefix
            try:
                local_t += one_request(port, model, pfx, max_tokens, timeout)
                local_r += 1
            except Exception:
                pass
        with lock:
            counts[0] += local_t
            reqs[0] += local_r

    pool = [threading.Thread(target=worker, daemon=True) for _ in range(C)]
    t0 = time.time()
    for t in pool:
        t.start()
    time.sleep(duration)
    stop.set()
    for t in pool:
        t.join(timeout=timeout)
    wall = time.time() - t0
    return counts[0] / wall, reqs[0] / wall, t0, t0 + wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--root-pid", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefix-tokens", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--concurrencies", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"client affinity cores: {len(os.sched_getaffinity(0))}")
    shared_prefix = make_prefix(args.prefix_tokens, False)
    sampler = Sampler(args.root_pid, interval=0.4)
    sampler.start()
    rows = []
    print(f"{'arm':8s} {'C':>4} {'tok/s':>8} {'req/s':>7} {'tree':>6} "
          f"{'busy':>6} {'worker':>7} {'GPU%':>6}")
    for C in args.concurrencies:
        for unique in (False, True):
            arm = "unique" if unique else "shared"
            run(args.port, args.model, args.prefix_tokens, args.max_tokens, C,
                2.0, unique, args.timeout, shared_prefix)  # warmup
            tps, rps, t0, t1 = run(args.port, args.model, args.prefix_tokens,
                                   args.max_tokens, C, args.duration, unique,
                                   args.timeout, shared_prefix)
            lo, hi = t0 + 0.15 * (t1 - t0), t1 - 0.15 * (t1 - t0)
            win = [s for s in sampler.samples if lo <= s["wall"] <= hi]
            if len(win) >= 2:
                dt = win[-1]["wall"] - win[0]["wall"]
                tree = (win[-1]["cpu"] - win[0]["cpu"]) / dt
                keys = set(win[0]["threads"]) & set(win[-1]["threads"])
                deltas = sorted((win[-1]["threads"][k] - win[0]["threads"][k]) / dt
                                for k in keys)
                busy = deltas[-1] if deltas else float("nan")
                worker = max(0.0, sum(deltas) - busy)
                sm = [s["sm"] for s in win if s["sm"] == s["sm"]]
                gpu = sum(sm) / len(sm) if sm else float("nan")
            else:
                tree = busy = worker = gpu = float("nan")
            r = {"arm": arm, "C": C, "tok_s": round(tps, 1),
                 "req_s": round(rps, 1), "tree": round(tree, 2),
                 "busy": round(busy, 2), "worker": round(worker, 2),
                 "gpu_sm": round(gpu, 1)}
            rows.append(r)
            print(f"{arm:8s} {C:>4} {r['tok_s']:>8} {r['req_s']:>7} "
                  f"{r['tree']:>6} {r['busy']:>6} {r['worker']:>7} {r['gpu_sm']:>6}")
        print("-" * 56)

    sampler.stop()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
