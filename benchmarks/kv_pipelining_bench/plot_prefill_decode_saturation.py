#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate saturation-point graphs for decode and prefill sweeps.

Inputs:
- Decode: results.jsonl from baseline_itl_load_sweep.py
- Prefill: summary.csv from prefill_saturation_benchmark.sh

Outputs:
- decode_saturation.png
- prefill_saturation.png
- saturation_points.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def first_consolidation_conc(
    concs: list[int],
    tok_s: list[float],
    *,
    rel_gain_threshold: float = 0.05,
    consecutive: int = 2,
) -> int:
    """Return first conc where gain is below threshold for N consecutive steps."""
    if len(concs) <= 1:
        return concs[0]
    low_gain_run = 0
    for i in range(1, len(concs)):
        prev = tok_s[i - 1]
        cur = tok_s[i]
        if prev <= 0:
            gain = 1.0
        else:
            gain = (cur - prev) / prev
        if gain < rel_gain_threshold:
            low_gain_run += 1
            if low_gain_run >= consecutive:
                return concs[i - consecutive + 1]
        else:
            low_gain_run = 0
    return concs[-1]


def load_decode(jsonl_path: Path) -> dict[tuple[int, int], dict[int, float]]:
    rows = [json.loads(x) for x in jsonl_path.read_text().splitlines() if x.strip()]
    grouped: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for r in rows:
        key = (
            int(r["input_len"]),
            int(r["output_len"]),
            int(r["concurrency"]),
        )
        grouped[key].append(float(r["throughput_tok_per_s"]))
    out: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    for (inp, out_len, conc), vals in grouped.items():
        out[(inp, out_len)][conc] = median(vals)
    return out


def load_prefill(csv_path: Path) -> dict[int, dict[int, float]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    with csv_path.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            inp = int(r["input_len"])
            conc = int(r["concurrency"])
            tok_s = float(r["input_tok_s"])
            grouped[(inp, conc)].append(tok_s)
    out: dict[int, dict[int, float]] = defaultdict(dict)
    for (inp, conc), vals in grouped.items():
        out[inp][conc] = median(vals)
    return out


def plot_decode(
    decode: dict[tuple[int, int], dict[int, float]],
    out_png: Path,
    sat_table: list[dict[str, Any]],
) -> None:
    plt.figure(figsize=(11, 7))
    for (inp, out_len), by_conc in sorted(decode.items()):
        concs = sorted(by_conc)
        vals = [by_conc[c] for c in concs]
        sat = first_consolidation_conc(concs, vals)
        sat_y = by_conc[sat]
        plt.plot(concs, vals, marker="o", linewidth=2, label=f"{inp}/{out_len}")
        plt.scatter([sat], [sat_y], s=70, marker="X")
        sat_table.append(
            {
                "phase": "decode",
                "req_len": f"{inp}/{out_len}",
                "saturation_concurrency": sat,
                "saturation_tok_s": round(sat_y, 4),
            }
        )
    plt.xscale("log", base=2)
    plt.xticks([1, 4, 8, 16, 64, 128, 192, 256, 384, 512], [1, 4, 8, 16, 64, 128, 192, 256, 384, 512])
    plt.xlabel("Concurrency")
    plt.ylabel("Throughput (tok/s)")
    plt.title("Decode saturation by request length (median across trials)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="in/out tokens")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_prefill(
    prefill: dict[int, dict[int, float]],
    out_png: Path,
    sat_table: list[dict[str, Any]],
) -> None:
    plt.figure(figsize=(11, 7))
    for inp, by_conc in sorted(prefill.items()):
        concs = sorted(by_conc)
        vals = [by_conc[c] for c in concs]
        sat = first_consolidation_conc(concs, vals)
        sat_y = by_conc[sat]
        plt.plot(concs, vals, marker="o", linewidth=2, label=f"{inp}")
        plt.scatter([sat], [sat_y], s=70, marker="X")
        sat_table.append(
            {
                "phase": "prefill",
                "req_len": str(inp),
                "saturation_concurrency": sat,
                "saturation_tok_s": round(sat_y, 4),
            }
        )
    plt.xscale("log", base=2)
    plt.xticks([1, 4, 8, 16, 64, 128, 192, 256, 384, 512], [1, 4, 8, 16, 64, 128, 192, 256, 384, 512])
    plt.xlabel("Concurrency")
    plt.ylabel("Input token throughput (tok/s)")
    plt.title("Prefill saturation by request length")
    plt.grid(True, alpha=0.3)
    plt.legend(title="input tokens")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decode-jsonl", type=Path, required=True)
    p.add_argument("--prefill-summary-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    ns = p.parse_args()

    ns.out_dir.mkdir(parents=True, exist_ok=True)
    decode = load_decode(ns.decode_jsonl)
    prefill = load_prefill(ns.prefill_summary_csv)

    sat_rows: list[dict[str, Any]] = []
    plot_decode(decode, ns.out_dir / "decode_saturation.png", sat_rows)
    plot_prefill(prefill, ns.out_dir / "prefill_saturation.png", sat_rows)

    with (ns.out_dir / "saturation_points.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "req_len",
                "saturation_concurrency",
                "saturation_tok_s",
            ],
        )
        w.writeheader()
        w.writerows(sat_rows)

    print(f"wrote {ns.out_dir / 'decode_saturation.png'}")
    print(f"wrote {ns.out_dir / 'prefill_saturation.png'}")
    print(f"wrote {ns.out_dir / 'saturation_points.csv'}")


if __name__ == "__main__":
    main()

