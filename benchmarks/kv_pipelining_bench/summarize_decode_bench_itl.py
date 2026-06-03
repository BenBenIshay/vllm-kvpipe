#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print mean ITL from decode_only_bench results JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _mean_itl_ms(path: Path) -> list[tuple[int, float | None]]:
    data = json.loads(path.read_text())
    rows: list[tuple[int, float | None]] = []
    for cell in data.get("throughput") or []:
        K = int(cell["concurrency"])
        m = cell.get("concurrent_mean_itl_s_mean")
        rows.append((K, float(m) * 1000.0 if m is not None else None))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path)
    args = ap.parse_args()
    for p in sorted(args.result_dir.glob("results_*.json")):
        mode = p.stem.replace("results_", "")
        print(f"\n{mode}:")
        for K, ms in _mean_itl_ms(p):
            print(f"  K={K:3d}  mean_ITL_ms={ms:.2f}" if ms else f"  K={K}  (no data)")


if __name__ == "__main__":
    main()
