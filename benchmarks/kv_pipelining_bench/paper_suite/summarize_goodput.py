#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize run_c6_goodput.sh output into a (SLO x rate x arm) goodput grid.

Reads each cell's summary.json (written by bench_admission_policy_e2e.py) and
emits goodput_summary.csv + GOODPUT_RESULTS.md. Primary metric:
slo_compliant_throughput_tok_s (goodput); also reports raw throughput, completed,
refused, and tail p99 ITL per arm.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from statistics import median

ARMS = ("stock", "policy")
_SLO_RE = re.compile(r"slo_([0-9.]+)")
_RATE_RE = re.compile(r"rate_([0-9.]+)")


def _find_cells(root: Path) -> list[tuple[float, float, Path]]:
    cells: list[tuple[float, float, Path]] = []
    for summary in root.glob("slo_*/rate_*/summary.json"):
        slo_m = _SLO_RE.search(str(summary.parent.parent.name))
        rate_m = _RATE_RE.search(str(summary.parent.name))
        if not slo_m or not rate_m:
            continue
        cells.append((float(slo_m.group(1)), float(rate_m.group(1)), summary))
    return sorted(cells, key=lambda c: (c[0], c[1]))


def _load(summary: Path) -> dict | None:
    try:
        return json.loads(summary.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _fmt(v) -> str:
    return f"{v:.1f}" if isinstance(v, (int, float)) else "—"


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/c6_goodput")
    print(f"Results root: {root}\n")

    cells = _find_cells(root)
    if not cells:
        print("No summary.json cells found (slo_*/rate_*/summary.json).")
        return

    rows: list[dict] = []
    print(
        f"{'SLO':>6} {'rate':>6} {'arm':>7} {'goodput':>9} {'raw_tps':>9} "
        f"{'compl':>6} {'refus':>6} {'p99ITL':>8}"
    )
    print("-" * 64)
    for slo, rate, summary in cells:
        data = _load(summary)
        if data is None:
            continue
        per_arm = data.get("summaries", {})
        row = {"slo_ms": slo, "rate_req_s": rate}
        for arm in ARMS:
            s = per_arm.get(arm) or {}
            goodput = s.get("slo_compliant_throughput_tok_s")
            raw = s.get("raw_throughput_tok_s")
            compl = s.get("completed")
            refus = s.get("refused")
            p99 = s.get("p99_p99_itl_ms")
            row[f"{arm}_goodput_tok_s"] = goodput
            row[f"{arm}_raw_tok_s"] = raw
            row[f"{arm}_completed"] = compl
            row[f"{arm}_refused"] = refus
            row[f"{arm}_p99_itl_ms"] = p99
            print(
                f"{slo:>6.0f} {rate:>6.1f} {arm:>7} {_fmt(goodput):>9} "
                f"{_fmt(raw):>9} {_fmt(compl):>6} {_fmt(refus):>6} {_fmt(p99):>8}"
            )
        # goodput gain of policy over stock
        sp = row.get("policy_goodput_tok_s")
        st = row.get("stock_goodput_tok_s")
        row["policy_goodput_gain_pct"] = (
            round(100.0 * (sp - st) / st, 1)
            if isinstance(sp, (int, float)) and isinstance(st, (int, float)) and st > 0
            else None
        )
        rows.append(row)

    # CSV
    if rows:
        fields = list(rows[0].keys())
        with (root / "goodput_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    # Markdown
    md = [
        "# C6: SLO-aware scheduler goodput",
        "",
        f"Results: `{root}`",
        "",
        "Goodput = SLO-compliant output tokens/s (requests whose p99 ITL <= SLO).",
        "Policy arm = transfer-aware admission ON; stock = admit everything.",
        "",
        "| SLO ms | rate | stock goodput | policy goodput | gain % | stock refused | policy refused |",
        "|--------|------|---------------|----------------|--------|---------------|----------------|",
    ]
    for r in rows:
        md.append(
            f"| {r['slo_ms']:.0f} | {r['rate_req_s']:.1f} | "
            f"{_fmt(r.get('stock_goodput_tok_s'))} | {_fmt(r.get('policy_goodput_tok_s'))} | "
            f"{_fmt(r.get('policy_goodput_gain_pct'))} | {_fmt(r.get('stock_refused'))} | "
            f"{_fmt(r.get('policy_refused'))} |"
        )
    gains = [r["policy_goodput_gain_pct"] for r in rows
             if isinstance(r.get("policy_goodput_gain_pct"), (int, float))]
    if gains:
        md += ["", f"Median policy goodput gain over stock: **{median(gains):.1f}%** "
                   f"(across {len(gains)} cells)."]
    md += [
        "",
        "## Paper figure",
        "Plot goodput (SLO-compliant tok/s) vs offered request rate, one line per arm,",
        "one panel per SLO. Expect stock to collapse past saturation while policy holds",
        "goodput by refusing/queuing admissions that would breach the SLO.",
    ]
    (root / "GOODPUT_RESULTS.md").write_text("\n".join(md) + "\n")

    print(f"\nWrote {root / 'goodput_summary.csv'}")
    print(f"Wrote {root / 'GOODPUT_RESULTS.md'}")


if __name__ == "__main__":
    main()
