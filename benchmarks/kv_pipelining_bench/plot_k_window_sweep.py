#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot mean ITL vs k_window from run_k_window_sweep.sh CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--K", type=int, default=1, help="Concurrency row to plot.")
    args = ap.parse_args()

    rows: list[dict[str, str]] = []
    with args.csv.open() as f:
        for row in csv.DictReader(f):
            if int(row["K"]) == args.K:
                rows.append(row)
    if not rows:
        raise SystemExit(f"No rows for K={args.K}")

    kws = sorted({int(r["k_window"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for mode, ax, ylabel in [
        ("baseline", axes[0], "Baseline mean ITL (ms)"),
        ("real_prefill", axes[1], "Feature mean ITL (ms)"),
    ]:
        sub = [r for r in rows if r["mode"] == mode]
        by_kw = {int(r["k_window"]): float(r["mean_itl_ms"]) for r in sub if r["mean_itl_ms"]}
        ys = [by_kw.get(k) for k in kws]
        ax.plot(kws, ys, "o-", linewidth=2, markersize=8)
        ax.set_xlabel("k_window")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{mode} @ K={args.K}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"k_window sweep (mean ITL, concurrency K={args.K})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
