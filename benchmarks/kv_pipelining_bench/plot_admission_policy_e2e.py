#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot the stock-vs-policy E2E comparison from bench_admission_policy_e2e.

Inputs (default discovered relative to ``--root``):

  raw_results.jsonl  one row per request with ``arm``, ``output_tokens``,
                     ``ttft_s``, ``mean_itl_s``, ``p99_itl_s``, ``refused``, ...
  summary.json       per-arm aggregates (already written by the bench)

Outputs (written under ``--root/plots/``):

  paper_admission_policy_tradeoff.png   2x2 paper figure
    (top-left)  raw vs SLO-compliant tok/s, grouped by arm
    (top-right) SLO compliance rate (% of completed) by arm
    (bot-left)  per-request p99 ITL CDF, stock vs policy, with SLO line
    (bot-right) refused / completed counts by arm

  policy_e2e_summary.csv  flat CSV for downstream tables.

The SLO budget is read from ``summary.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARM_ORDER = ["stock", "policy"]
COLORS = {"stock": "#888888", "policy": "#1f77b4"}


def _load(root: Path) -> tuple[dict[str, Any], list[dict]]:
    summary = json.loads((root / "summary.json").read_text())
    rows: list[dict] = []
    raw_path = root / "raw_results.jsonl"
    if raw_path.is_file():
        with raw_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return summary, rows


def _panel_throughput(ax, summary):
    arms = [a for a in ARM_ORDER if a in summary["summaries"]]
    raw = [summary["summaries"][a].get("raw_throughput_tok_s", 0.0) for a in arms]
    slo = [summary["summaries"][a].get("slo_compliant_throughput_tok_s", 0.0) for a in arms]
    x = list(range(len(arms)))
    width = 0.35
    ax.bar([xi - width / 2 for xi in x], raw, width=width, label="raw", color="#cccccc",
           edgecolor="#555")
    ax.bar([xi + width / 2 for xi in x], slo, width=width, label="SLO-compliant",
           color=[COLORS[a] for a in arms], edgecolor="#222")
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("tok/s")
    ax.set_title("Throughput: raw vs SLO-compliant")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for xi, (r, s) in enumerate(zip(raw, slo)):
        ax.text(xi - width / 2, r, f"{r:.0f}", ha="center", va="bottom", fontsize=7)
        ax.text(xi + width / 2, s, f"{s:.0f}", ha="center", va="bottom", fontsize=7)


def _panel_compliance(ax, summary):
    arms = [a for a in ARM_ORDER if a in summary["summaries"]]
    pct = [summary["summaries"][a].get("slo_compliant_pct_of_completed", 0.0) for a in arms]
    ax.bar(arms, pct, color=[COLORS[a] for a in arms], edgecolor="#222")
    ax.set_ylim(0, 105)
    ax.set_ylabel("% completed within SLO")
    ax.set_title("SLO compliance rate")
    ax.grid(axis="y", alpha=0.3)
    for i, p in enumerate(pct):
        ax.text(i, p + 1, f"{p:.1f}%", ha="center", va="bottom", fontsize=8)


def _panel_itl_cdf(ax, rows, slo_p99_ms: float):
    ax.set_title("Per-request p99 ITL CDF")
    ax.set_xlabel("p99 ITL (ms)")
    ax.set_ylabel("F(x)")
    has_data = False
    for arm in ARM_ORDER:
        ms = [r["p99_itl_s"] * 1000.0 for r in rows
              if r.get("arm") == arm and r.get("p99_itl_s") is not None]
        if not ms:
            continue
        ms.sort()
        y = [(i + 1) / len(ms) for i in range(len(ms))]
        ax.plot(ms, y, label=f"{arm} (n={len(ms)})", color=COLORS[arm], linewidth=2)
        has_data = True
    if slo_p99_ms > 0:
        ax.axvline(slo_p99_ms, color="red", linestyle="--", alpha=0.6,
                   label=f"SLO {slo_p99_ms:.0f} ms")
    if has_data:
        ax.set_xscale("log")
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)


def _panel_admit_counts(ax, summary):
    arms = [a for a in ARM_ORDER if a in summary["summaries"]]
    completed = [summary["summaries"][a].get("completed", 0) for a in arms]
    refused = [summary["summaries"][a].get("refused", 0) for a in arms]
    failed = [summary["summaries"][a].get("failed_other", 0) for a in arms]
    ax.bar(arms, completed, label="completed", color="#5cb85c", edgecolor="#222")
    ax.bar(arms, refused, bottom=completed, label="refused (policy)",
           color="#f0ad4e", edgecolor="#222")
    ax.bar(arms, failed, bottom=[c + r for c, r in zip(completed, refused)],
           label="other errors", color="#d9534f", edgecolor="#222")
    ax.set_ylabel("# requests")
    ax.set_title("Outcome distribution")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def _write_summary_csv(root: Path, summary: dict) -> Path:
    out = root / "policy_e2e_summary.csv"
    fields = [
        "arm", "n_requests", "completed", "refused", "failed_other",
        "slo_compliant_completions", "slo_compliant_pct_of_completed",
        "raw_throughput_tok_s", "slo_compliant_throughput_tok_s",
        "p50_p99_itl_ms", "p99_p99_itl_ms", "p50_ttft_ms", "p99_ttft_ms",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for arm, s in summary["summaries"].items():
            row = {k: s.get(k) for k in fields if k != "arm"}
            row["arm"] = arm
            w.writerow(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="Output directory of bench_admission_policy_e2e.py.")
    ap.add_argument("--out-prefix", default="paper_admission_policy_tradeoff",
                    help="Output PNG basename (without extension).")
    args = ap.parse_args()

    root = args.root
    if not (root / "summary.json").is_file():
        raise SystemExit(f"summary.json not found under {root}")
    summary, rows = _load(root)
    slo_p99_ms = float(summary.get("args", {}).get("slo_p99_itl_ms", 0.0))

    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _panel_throughput(axes[0][0], summary)
    _panel_compliance(axes[0][1], summary)
    _panel_itl_cdf(axes[1][0], rows, slo_p99_ms)
    _panel_admit_counts(axes[1][1], summary)
    rate = summary.get("args", {}).get("request_rate", "?")
    model = summary.get("args", {}).get("model", "?")
    fig.suptitle(
        f"SLO-aware admission policy: stock vs policy "
        f"({model}, rate={rate} req/s, SLO p99 ITL={slo_p99_ms:.0f} ms)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = plots_dir / f"{args.out_prefix}.png"
    fig.savefig(out_png, dpi=160)
    print(f"Wrote {out_png}")

    csv_out = _write_summary_csv(root, summary)
    print(f"Wrote {csv_out}")


if __name__ == "__main__":
    main()
