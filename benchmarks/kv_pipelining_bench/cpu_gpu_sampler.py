#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CPU + GPU sampler for the KV-pipelining baseline study
======================================================

Continuously samples the **whole vLLM server process tree** (API server +
EngineCore child + any workers) and the GPU, writing one timestamped CSV row
per interval. Purpose: measure how many host CPU cores the *production* vLLM
baseline burns on scheduling / KV-block (HBM) management while the A100 is at
peak decode throughput.

Columns
-------
- ``ts``                wall-clock epoch seconds (float)
- ``elapsed_s``         seconds since sampler start
- ``n_procs``           # matched processes in the tree this tick
- ``total_cores``       sum of (user+system) CPU-time deltas / wall delta over
                        the whole tree  (== "cores busy"; 1.0 == one full core)
- ``top_proc_cores``    busiest single process (cores)
- ``top_thread_cores``  busiest single *thread* across the tree (cores) — if
                        this pegs ~1.0 while total is low we are GIL/single-
                        thread bound (scheduler hot loop)
- ``gpu_util``          GPU SM utilization %  (nvidia-smi utilization.gpu)
- ``gpu_mem_util``      GPU memory-controller utilization %
- ``gpu_mem_used_mib``  HBM used (MiB)

The matched process set is rediscovered every tick (so the EngineCore child is
picked up whenever it spawns) by taking every process whose cmdline contains
``--match`` PLUS all of their recursive descendants.

Run::

    python cpu_gpu_sampler.py --match api_server --port 9012 \
        --interval 0.5 --out /tmp/run/cpu_gpu.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import psutil


def find_tree(match: str, port: str | None) -> dict[int, psutil.Process]:
    """Return {pid: Process} for every matching root plus all descendants."""
    roots: list[psutil.Process] = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = p.info["cmdline"] or []
            joined = " ".join(cl)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if match not in joined:
            continue
        if port is not None and port not in joined:
            # Allow EngineCore/worker children that don't carry the port; they
            # are caught as descendants of the api_server root which does.
            if "api_server" in joined or "EngineCore" in joined:
                pass
            else:
                continue
        roots.append(p)
    tree: dict[int, psutil.Process] = {}
    for r in roots:
        try:
            tree[r.pid] = r
            for c in r.children(recursive=True):
                tree[c.pid] = c
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return tree


def query_gpu(gpu_index: int) -> tuple[float, float, float]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        line = out.stdout.strip().splitlines()[0]
        u, m, used = (x.strip() for x in line.split(","))
        return float(u), float(m), float(used)
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return -1.0, -1.0, -1.0


def proc_cpu_seconds(p: psutil.Process) -> float:
    t = p.cpu_times()
    return float(t.user) + float(t.system)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="api_server")
    ap.add_argument("--port", default=None)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--duration", type=float, default=0.0, help="0 = until killed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fh = open(args.out, "w", buffering=1)
    fh.write(
        "ts,elapsed_s,n_procs,total_cores,top_proc_cores,top_thread_cores,"
        "gpu_util,gpu_mem_util,gpu_mem_used_mib\n"
    )

    prev_proc: dict[int, float] = {}
    prev_thread: dict[tuple[int, int], float] = {}
    t_start = time.monotonic()
    last_wall = time.monotonic()
    last_print = 0.0

    while True:
        now = time.monotonic()
        dt = now - last_wall
        if dt <= 0:
            time.sleep(args.interval)
            continue

        tree = find_tree(args.match, args.port)
        cur_proc: dict[int, float] = {}
        cur_thread: dict[tuple[int, int], float] = {}
        per_proc_cores: dict[int, float] = {}
        top_thread = 0.0

        for pid, p in tree.items():
            try:
                cur_proc[pid] = proc_cpu_seconds(p)
                if pid in prev_proc:
                    per_proc_cores[pid] = max(0.0, (cur_proc[pid] - prev_proc[pid]) / dt)
                for th in p.threads():
                    key = (pid, th.id)
                    cum = float(th.user_time) + float(th.system_time)
                    cur_thread[key] = cum
                    if key in prev_thread:
                        c = max(0.0, (cum - prev_thread[key]) / dt)
                        if c > top_thread:
                            top_thread = c
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        total_cores = sum(per_proc_cores.values())
        top_proc = max(per_proc_cores.values()) if per_proc_cores else 0.0
        gu, gmu, gmem = query_gpu(args.gpu_index)

        elapsed = now - t_start
        fh.write(
            f"{time.time():.3f},{elapsed:.3f},{len(tree)},{total_cores:.3f},"
            f"{top_proc:.3f},{top_thread:.3f},{gu:.0f},{gmu:.0f},{gmem:.0f}\n"
        )

        if now - last_print >= 2.0:
            print(
                f"[{elapsed:6.1f}s] procs={len(tree):2d} "
                f"cores={total_cores:5.2f} top_proc={top_proc:4.2f} "
                f"top_thr={top_thread:4.2f} gpu={gu:3.0f}% mem={gmem/1024:5.1f}GiB",
                flush=True,
            )
            last_print = now

        prev_proc = cur_proc
        prev_thread = cur_thread
        last_wall = now

        if args.duration and elapsed >= args.duration:
            break
        time.sleep(args.interval)

    fh.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
