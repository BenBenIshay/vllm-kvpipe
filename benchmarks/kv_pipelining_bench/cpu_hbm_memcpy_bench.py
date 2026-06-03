#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU (pinned) → GPU HBM host-to-device copy bandwidth microbenchmark.

Use after serving benchmarks to compare raw PCIe/NVLink transfer headroom vs
end-to-end scheduling limits (qualitative; access patterns differ from GEMM/KV).

Writes CSV + optional PNG under ``--result-dir``.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch


def run_once(num_bytes: int, trials: int, warmup: int) -> tuple[float, float]:
    """Return (median_gb_s, stdev_gb_s) over successful trials."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device("cuda:0")
    bytes_per_elem = 2  # fp16
    n = num_bytes // bytes_per_elem
    host = torch.randn(n, dtype=torch.float16, pin_memory=True)
    gpu = torch.empty(n, dtype=torch.float16, device=device)

    rates: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        gpu.copy_(host, non_blocking=True)
    torch.cuda.synchronize()

    for _ in range(trials):
        start.record()
        gpu.copy_(host, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        if ms <= 0:
            continue
        gb_s = (num_bytes / 1e9) / (ms / 1000.0)
        rates.append(gb_s)

    if len(rates) < 2:
        med = rates[0] if rates else 0.0
        return med, 0.0
    return float(statistics.median(rates)), float(statistics.stdev(rates))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sizes-mib",
        type=float,
        nargs="+",
        default=[64, 128, 256, 512, 1024, 2048],
        help="Transfer sizes in MiB",
    )
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--result-dir",
        type=Path,
        default=Path("/tmp/hbm_memcpy_bench"),
    )
    args = p.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.result_dir / "h2d_bandwidth.csv"
    rows: list[dict[str, float | int]] = []

    print("MiB\tGB/s (median)\tstd")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["size_mib", "size_bytes", "median_gb_s", "stdev_gb_s"],
        )
        w.writeheader()
        for mib in args.sizes_mib:
            nbytes = int(mib * 1024 * 1024)
            med, sd = run_once(nbytes, args.trials, args.warmup)
            print(f"{mib:g}\t{med:.3f}\t{sd:.3f}")
            row = {
                "size_mib": int(mib) if mib == int(mib) else mib,
                "size_bytes": nbytes,
                "median_gb_s": round(med, 4),
                "stdev_gb_s": round(sd, 4),
            }
            rows.append(row)
            w.writerow(row)

    # Simple PNG if matplotlib available
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        xs = [r["size_mib"] for r in rows]
        ys = [r["median_gb_s"] for r in rows]
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("Transfer size (MiB)")
        ax.set_ylabel("H2D bandwidth (GB/s)")
        ax.set_title("Pinned CPU → GPU copy (torch.cuda)")
        ax.grid(True, alpha=0.3)
        png = args.result_dir / "h2d_bandwidth.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {png}")
    except Exception as e:
        print(f"(skip plot: {e})")

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
