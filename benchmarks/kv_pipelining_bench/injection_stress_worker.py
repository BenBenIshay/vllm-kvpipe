#!/usr/bin/env python3
"""
Background stress for mem / PCIe / HBM interference next to vLLM bench.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import torch

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))
from kv_sizing import (  # noqa: E402
    align_numel,
    numel_from_chunk_mib_fp16,
    numel_from_kv_block_one_layer,
)


def _pace_sleep(chunk_bytes: float, target_bps: float, elapsed: float) -> None:
    if target_bps <= 0:
        return
    want = chunk_bytes / target_bps
    slp = want - elapsed
    if slp > 0:
        time.sleep(slp)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument(
        "mode",
        choices=["h2d", "h2d-async", "d2h-async", "h2d-d2h-mix", "hbm-parallel"],
    )
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--target-gbps", type=float, default=0.0)
    p.add_argument("--h2d-gbps", type=float, default=0.0)
    p.add_argument("--d2h-gbps", type=float, default=0.0)
    p.add_argument("--chunk-mib", type=int, default=256)
    p.add_argument("--kv-tokens", type=int, default=0)
    p.add_argument("--n-kv-heads", type=int, default=0)
    p.add_argument("--head-dim", type=int, default=0)
    p.add_argument("--sync-every", type=int, default=1)
    p.add_argument("--concurrent", type=int, default=1)
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--tensor-mib", type=int, default=64)
    return p.parse_args()


def _resolve_numel(args) -> int:
    if int(args.kv_tokens) > 0 and int(args.n_kv_heads) > 0 and int(args.head_dim) > 0:
        return align_numel(
            numel_from_kv_block_one_layer(args.kv_tokens, args.n_kv_heads, args.head_dim)
        )
    return numel_from_chunk_mib_fp16(args.chunk_mib)


def _h2d_one_stream(seconds: float, numel: int, target_bps: float, sync_every: int, asyncish: bool) -> float:
    host = torch.randn(numel, dtype=torch.float16, pin_memory=True)
    gpu = torch.empty(numel, dtype=torch.float16, device="cuda")
    st = torch.cuda.Stream()
    bytes_chunk = float(numel * 2)
    se = max(1, sync_every)
    if asyncish and se == 1:
        se = 32
    moved = 0.0
    pending = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        t1 = time.time()
        with torch.cuda.stream(st):
            gpu.copy_(host, non_blocking=True)
        pending += 1
        if pending >= se:
            st.synchronize()
            moved += bytes_chunk * float(pending)
            pending = 0
        _pace_sleep(bytes_chunk, target_bps, time.time() - t1)
    if pending:
        st.synchronize()
        moved += bytes_chunk * float(pending)
    return moved


def _d2h_one_stream(seconds: float, numel: int, target_bps: float, sync_every: int, asyncish: bool) -> float:
    gpu = torch.randn(numel, dtype=torch.float16, device="cuda")
    host = torch.empty(numel, dtype=torch.float16, pin_memory=True)
    st = torch.cuda.Stream()
    bytes_chunk = float(numel * 2)
    se = max(1, sync_every)
    if asyncish and se == 1:
        se = 32
    moved = 0.0
    pending = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        t1 = time.time()
        with torch.cuda.stream(st):
            host.copy_(gpu, non_blocking=True)
        pending += 1
        if pending >= se:
            st.synchronize()
            moved += bytes_chunk * float(pending)
            pending = 0
        _pace_sleep(bytes_chunk, target_bps, time.time() - t1)
    if pending:
        st.synchronize()
        moved += bytes_chunk * float(pending)
    return moved


def _thread_runner(fn: Callable[..., float], out: list[float], idx: int, fargs: tuple) -> None:
    out[idx] = fn(*fargs)


def run_h2d_family(asyncish: bool, seconds: float, target_gbps: float, numel: int, sync_every: int, concurrent: int) -> None:
    target_bps = max(0.0, float(target_gbps)) * 1e9
    conc = max(1, int(concurrent))
    parts = [0.0] * conc
    threads = []
    t0 = time.time()
    for i in range(conc):
        th = threading.Thread(
            target=_thread_runner,
            args=(_h2d_one_stream, parts, i, (seconds, numel, target_bps, sync_every, asyncish)),
        )
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    moved = sum(parts)
    dur = time.time() - t0
    ach = (moved / 1e9) / max(dur, 1e-6)
    kind = "h2d_async" if asyncish else "h2d"
    se = max(1, int(sync_every))
    if asyncish and se == 1:
        se = 32
    print(
        f"mode={kind} target_gbps={target_gbps} achieved_gbps={ach:.4f} "
        f"concurrent={conc} numel={numel} sync_every={se} duration={dur:.3f}"
    )


def run_d2h_async(seconds: float, target_gbps: float, numel: int, sync_every: int, concurrent: int) -> None:
    target_bps = max(0.0, float(target_gbps)) * 1e9
    conc = max(1, int(concurrent))
    parts = [0.0] * conc
    threads = []
    t0 = time.time()
    for i in range(conc):
        th = threading.Thread(
            target=_thread_runner,
            args=(_d2h_one_stream, parts, i, (seconds, numel, target_bps, sync_every, True)),
        )
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    moved = sum(parts)
    dur = time.time() - t0
    ach = (moved / 1e9) / max(dur, 1e-6)
    se = max(1, int(sync_every))
    if se == 1:
        se = 32
    print(
        f"mode=d2h_async target_gbps={target_gbps} achieved_gbps={ach:.4f} "
        f"concurrent={conc} numel={numel} sync_every={se} duration={dur:.3f}"
    )


def run_h2d_d2h_mix(seconds: float, h2d_gbps: float, d2h_gbps: float, numel: int, sync_every: int) -> None:
    h2d_bps = max(0.0, float(h2d_gbps)) * 1e9
    d2h_bps = max(0.0, float(d2h_gbps)) * 1e9
    se = max(1, int(sync_every))
    if se == 1:
        se = 32
    h2d_m = [0.0]
    d2h_m = [0.0]
    t0 = time.time()
    th1 = threading.Thread(
        target=_thread_runner,
        args=(_h2d_one_stream, h2d_m, 0, (seconds, numel, h2d_bps, se, True)),
    )
    th2 = threading.Thread(
        target=_thread_runner,
        args=(_d2h_one_stream, d2h_m, 0, (seconds, numel, d2h_bps, se, True)),
    )
    th1.start()
    th2.start()
    th1.join()
    th2.join()
    dur = time.time() - t0
    h2d_ach = (h2d_m[0] / 1e9) / max(dur, 1e-6)
    d2h_ach = (d2h_m[0] / 1e9) / max(dur, 1e-6)
    print(
        f"mode=h2d_d2h_mix h2d_target_gbps={h2d_gbps} d2h_target_gbps={d2h_gbps} "
        f"h2d_achieved_gbps={h2d_ach:.4f} d2h_achieved_gbps={d2h_ach:.4f} "
        f"numel={numel} duration={dur:.3f}"
    )


def _hbm_thread(stream, a: torch.Tensor, b: torch.Tensor, stop: threading.Event) -> None:
    with torch.cuda.stream(stream):
        while not stop.is_set():
            torch.add(a, b, out=a)


def run_hbm_parallel(seconds: float, parallel: int, tensor_mib: int) -> None:
    n = int(tensor_mib) * 1024 * 1024 // 2
    stop = threading.Event()
    threads: list[threading.Thread] = []
    streams: list[torch.cuda.Stream] = []
    t0 = time.time()
    for _ in range(max(1, int(parallel))):
        s = torch.cuda.Stream()
        a = torch.randn(n, device="cuda", dtype=torch.float16)
        b = torch.randn(n, device="cuda", dtype=torch.float16) * 0.01
        streams.append(s)
        th = threading.Thread(target=_hbm_thread, args=(s, a, b, stop), daemon=True)
        th.start()
        threads.append(th)
    time.sleep(seconds)
    stop.set()
    for th in threads:
        th.join(timeout=30.0)
    for s in streams:
        s.synchronize()
    dur = time.time() - t0
    print(
        f"mode=hbm_parallel parallel_streams={int(parallel)} tensor_mib={tensor_mib} "
        f"duration={dur:.3f}"
    )


def main() -> None:
    args = _parse()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    numel = _resolve_numel(args)
    if args.mode in ("h2d", "h2d-async"):
        run_h2d_family(args.mode == "h2d-async", args.seconds, args.target_gbps, numel, args.sync_every, args.concurrent)
    elif args.mode == "d2h-async":
        run_d2h_async(args.seconds, args.target_gbps, numel, args.sync_every, args.concurrent)
    elif args.mode == "h2d-d2h-mix":
        run_h2d_d2h_mix(args.seconds, args.h2d_gbps, args.d2h_gbps, numel, args.sync_every)
    else:
        run_hbm_parallel(args.seconds, args.parallel, args.tensor_mib)


if __name__ == "__main__":
    main()
