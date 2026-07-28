#!/usr/bin/env python3
"""Experiment B1: scheduling single-thread ceiling (clean, non-streaming).

Closed-loop load: C worker threads each fire NON-streaming completion requests
back-to-back for a fixed duration (minimal client CPU -> client doesn't contend).
Measures sustained server throughput + server process-tree cores + busiest thread
(EngineCore step loop) + GPU SM%. Shows whether the single-threaded scheduler caps
throughput while the GPU still has headroom.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_baseline_sweep import Sampler  # noqa: E402
from decode_only_bench import _generate_prompt  # noqa: E402


def one_request(port, model, prompt, max_tokens, timeout):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "ignore_eos": True, "stream": False}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        d = json.loads(resp.read())
    return int(d.get("usage", {}).get("completion_tokens", 0))


def run_closed_loop(port, model, prompt, max_tokens, C, duration, timeout):
    stop = threading.Event()
    counts = [0]
    lock = threading.Lock()

    def worker():
        local = 0
        while not stop.is_set():
            try:
                local += one_request(port, model, prompt, max_tokens, timeout)
            except Exception:
                pass
        with lock:
            counts[0] += local

    pool = [threading.Thread(target=worker, daemon=True) for _ in range(C)]
    t0 = time.time()
    for t in pool:
        t.start()
    time.sleep(duration)
    stop.set()
    for t in pool:
        t.join(timeout=timeout)
    wall = time.time() - t0
    return counts[0] / wall, t0, t0 + wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--root-pid", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--concurrencies", type=int, nargs="+",
                    default=[16, 32, 64, 128, 256, 512])
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"client affinity cores: {len(os.sched_getaffinity(0))}")
    prompt = _generate_prompt(args.input_len)
    sampler = Sampler(args.root_pid, interval=0.4)
    sampler.start()
    rows = []
    print(f"{'C':>5} {'tok/s':>9} {'tree':>6} {'busy':>6} {'worker':>7} {'GPU%':>6}")
    for C in args.concurrencies:
        # warmup
        run_closed_loop(args.port, args.model, prompt, args.max_tokens, C,
                        2.0, args.timeout)
        tps, t0, t1 = run_closed_loop(args.port, args.model, prompt,
                                      args.max_tokens, C, args.duration,
                                      args.timeout)
        # steady window = middle 70%
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
        r = {"C": C, "tok_s": round(tps, 1), "tree": round(tree, 2),
             "busy": round(busy, 2), "worker": round(worker, 2),
             "gpu_sm": round(gpu, 1)}
        rows.append(r)
        print(f"{C:>5} {r['tok_s']:>9} {r['tree']:>6} {r['busy']:>6} "
              f"{r['worker']:>7} {r['gpu_sm']:>6}")

    sampler.stop()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
