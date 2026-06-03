#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Decode HBM headroom probe — validates the "capacity vs bandwidth" use-case plan.

This script does **not** need vLLM or a model. It runs a small CUDA experiment:

1. **Solo memory-bound kernel** — streaming reads/writes on large tensors (decode-like
   arithmetic intensity: low FLOPs per byte). Reports achieved **effective GB/s**
   toward device memory.

2. **Solo H2D DMA** — pinned host → device copies (PCIe path), reports achieved GB/s.

3. **Overlap** — same memory-bound kernel on one stream + concurrent ``copy_`` on
   another stream. If the kernel slows down materially when DMA runs, **HBM (and
   memory subsystem) contention** between SMs and copy engines is present — i.e.
   less "free" headroom for overlapping KV prefetch with decode.

Interpretation (for your KV DDR offload story):

- If **solo kernel GB/s** is **well below** published peak HBM for your GPU, the
  box may be **latency/occupancy** limited, not saturated — extra DMA may be cheap.
- If **solo kernel GB/s** is **high** and **overlap** drops it sharply, you are in
  a **bandwidth-contended** regime; schedule/throttle prefetch.

**Optional:** ``--print-peak-hbm-gbps`` uses a small built-in table for common SKUs
(nvidia-smi name) so you can compute **utilization = achieved / peak** manually.

**Pair with full serving tests:** After this micro-probe, run
``benchmarks/kv_pipelining/capacity_test.py`` or ``kv_pipelining_bench/decode_only_bench.py``
for OOM / concurrency limits on real decode.

Usage::

    python benchmarks/kv_pipelining/decode_hbm_headroom_probe.py
    python benchmarks/kv_pipelining/decode_hbm_headroom_probe.py --tensor-mb 256 \\
        --dma-mb-per-iter 32 --iters 200
    python benchmarks/kv_pipelining/decode_hbm_headroom_probe.py --quick --json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from typing import Any

# Approximate peak HBM (device) GB/s — literature / product briefs; for utilization
# estimates only. Keys are substrings matched against nvidia-smi product name (lower).
_PEAK_HBM_GBPS_BY_NAME_SUBSTR: tuple[tuple[str, float], ...] = (
    ("h100", 3350.0),
    ("h200", 4800.0),
    ("b200", 8000.0),  # varies by config; order-of-magnitude
    ("a100", 2039.0),
    ("a6000", 768.0),
    ("rtx 4090", 1008.0),
    ("l40s", 864.0),
    ("l40", 864.0),
)


def _gpu_product_name() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.strip().split("\n")[0].strip().lower()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, IndexError):
        return ""


def _lookup_peak_hbm_gbps(product_name: str) -> float | None:
    ln = product_name.lower()
    for key, gbps in _PEAK_HBM_GBPS_BY_NAME_SUBSTR:
        if key in ln:
            return gbps
    return None


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for this probe.", file=sys.stderr)
        sys.exit(2)


def _memory_bound_kernel_iters(
    x: Any,
    y: Any,
    z: Any,
    stream: Any,
    iters: int,
) -> None:
    import torch

    with torch.cuda.stream(stream):
        for _ in range(iters):
            torch.add(x, y, out=z)


def _dma_iters(
    host_buf: Any,
    dev_buf: Any,
    stream: Any,
    iters: int,
) -> None:
    import torch

    with torch.cuda.stream(stream):
        for _ in range(iters):
            dev_buf.copy_(host_buf, non_blocking=True)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    itemsize = torch.tensor([], dtype=dtype, device="cpu").element_size()
    numel = max(1, int(args.tensor_mb * 1024 * 1024 // itemsize))

    x = torch.randn(numel, device=device, dtype=dtype)
    y = torch.randn(numel, device=device, dtype=dtype)
    z = torch.empty_like(x)

    bytes_moved_per_iter = 3 * numel * itemsize  # add: 2 loads + 1 store

    compute_stream = torch.cuda.Stream(device=device)
    copy_stream = torch.cuda.Stream(device=device)

    dma_bytes = int(args.dma_mb_per_iter * 1024 * 1024)
    host_buf = None
    dev_buf = None
    if dma_bytes > 0:
        # Pin host, device buffer for H2D
        host_buf = torch.empty(dma_bytes, dtype=torch.uint8, pin_memory=True)
        dev_buf = torch.empty(dma_bytes, dtype=torch.uint8, device=device)
        host_buf.random_(0, 255)

    def sync():
        torch.cuda.synchronize(device)

    # Warmup
    for _ in range(args.warmup):
        _memory_bound_kernel_iters(x, y, z, compute_stream, 1)
        if dma_bytes > 0:
            _dma_iters(host_buf, dev_buf, copy_stream, 1)
    sync()

    # 1) Solo compute
    sync()
    t0 = time.perf_counter()
    _memory_bound_kernel_iters(x, y, z, compute_stream, args.iters)
    sync()
    t_compute_solo = time.perf_counter() - t0
    gbps_compute_solo = (
        (bytes_moved_per_iter * args.iters) / t_compute_solo / 1e9
    )

    # 2) Solo DMA (if enabled)
    gbps_dma_solo = None
    if dma_bytes > 0:
        sync()
        t0 = time.perf_counter()
        _dma_iters(host_buf, dev_buf, copy_stream, args.iters)
        sync()
        t_dma_solo = time.perf_counter() - t0
        gbps_dma_solo = (dma_bytes * args.iters) / t_dma_solo / 1e9

    # 3) Overlap: interleave same counts as the tight loop
    gbps_compute_overlap = None
    if dma_bytes > 0:
        sync()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            with torch.cuda.stream(copy_stream):
                dev_buf.copy_(host_buf, non_blocking=True)
            with torch.cuda.stream(compute_stream):
                torch.add(x, y, out=z)
        sync()
        t_overlap = time.perf_counter() - t0
        gbps_compute_overlap = (
            (bytes_moved_per_iter * args.iters) / t_overlap / 1e9
        )

    product = _gpu_product_name()
    peak = _lookup_peak_hbm_gbps(product) if getattr(args, "print_peak", False) else None

    out: dict[str, Any] = {
        "gpu_name": product or None,
        "tensor_mb": args.tensor_mb,
        "dtype": args.dtype,
        "numel": numel,
        "iters": args.iters,
        "dma_mb_per_iter": args.dma_mb_per_iter,
        "gbps_memory_bound_kernel_solo": round(gbps_compute_solo, 2),
        "gbps_h2d_dma_solo": round(gbps_dma_solo, 2) if gbps_dma_solo else None,
        "gbps_memory_bound_kernel_under_overlap": (
            round(gbps_compute_overlap, 2) if gbps_compute_overlap else None
        ),
        "peak_hbm_gbps_lookup": round(peak, 1) if peak else None,
    }

    if peak and gbps_compute_solo > 0:
        out["estimated_hbm_utilization_kernel_solo"] = round(
            100.0 * gbps_compute_solo / peak, 1
        )

    if gbps_compute_overlap and gbps_compute_solo > 0:
        out["kernel_effective_bw_ratio_overlap_vs_solo"] = round(
            gbps_compute_overlap / gbps_compute_solo, 3
        )

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Micro-benchmark: memory-bound kernel vs concurrent H2D DMA (HBM contention probe)",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--tensor-mb",
        type=float,
        default=128.0,
        help="Per-tensor size (x,y,z) in MiB; larger => more HBM pressure",
    )
    p.add_argument(
        "--dma-mb-per-iter",
        type=float,
        default=32.0,
        help="Pinned H2D bytes per iteration in overlap section (0 = no DMA)",
    )
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    p.add_argument(
        "--print-peak",
        action="store_true",
        help="Match GPU name to a rough peak HBM GB/s table (utilization estimate)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Small tensors and few iters for smoke test",
    )
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.tensor_mb = min(args.tensor_mb, 32.0)
        args.iters = min(args.iters, 40)
        args.warmup = min(args.warmup, 5)
        args.dma_mb_per_iter = min(args.dma_mb_per_iter, 8.0)

    _require_cuda()

    result = run_probe(args)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("Decode HBM headroom probe (synthetic)")
    print("=" * 60)
    print(f"  GPU:           {result.get('gpu_name')}")
    print(f"  Tensor size:   {result['tensor_mb']} MiB × 3 (add)")
    print(f"  Iterations:    {result['iters']}")
    print(f"  DMA per iter:  {result['dma_mb_per_iter']} MiB (H2D, pinned)")
    if result.get("peak_hbm_gbps_lookup"):
        print(
            f"  Peak HBM (ref): ~{result['peak_hbm_gbps_lookup']} GB/s "
            f"(rough table; verify for your SKU)"
        )
    print()
    print(
        f"  Memory-bound kernel (solo):     {result['gbps_memory_bound_kernel_solo']:.2f} GB/s effective"
    )
    if result.get("gbps_h2d_dma_solo") is not None:
        print(
            f"  H2D DMA (solo):                 {result['gbps_h2d_dma_solo']:.2f} GB/s"
        )
    if result.get("gbps_memory_bound_kernel_under_overlap") is not None:
        print(
            f"  Memory-bound kernel (overlap): {result['gbps_memory_bound_kernel_under_overlap']:.2f} GB/s effective"
        )
        ratio = result.get("kernel_effective_bw_ratio_overlap_vs_solo")
        if ratio is not None:
            pct = 100.0 * (1.0 - ratio)
            print(
                f"  Overlap vs solo (kernel BW):    {ratio:.3f}x  (~{pct:.1f}% effective BW loss)"
            )
    if result.get("estimated_hbm_utilization_kernel_solo") is not None:
        print(
            f"  Est. HBM util (solo kernel):   {result['estimated_hbm_utilization_kernel_solo']:.1f}% of ref peak"
        )
    print()
    print("Interpretation:")
    print("  - Low solo utilization + small overlap loss => room for pipelined KV prefetch.")
    print("  - High solo utilization + large overlap loss => throttle/chunk DMA with decode.")
    print()
    print("Next: run benchmarks/kv_pipelining/capacity_test.py for OOM/concurrency limits.")


if __name__ == "__main__":
    main()
