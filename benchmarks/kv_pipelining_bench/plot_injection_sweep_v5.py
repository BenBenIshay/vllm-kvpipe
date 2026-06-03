#!/usr/bin/env python3
"""Plot injection_sweep_v5 results.

Produces per (model, phase):
  1. Line plots: SLO-limited tok/s vs request_rate, one line per scenario family.
  2. Heatmap: SLO-limited tok/s grid  (request_rate × injection_bw/scenario).
  3. Baseline delta table: % change vs no-injection baseline per scenario.

Reads injection_sweep_v5.jsonl from --root (default: injection_sweep_v5/ next
to this file, or $KV_SWEEP_OUT).

Usage::
  python plot_injection_sweep_v5.py [--root /path/to/injection_sweep_v5]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

THIS = Path(__file__).resolve().parent
ROOT_DEFAULT = Path(__file__).resolve().parent / "injection_sweep_v5"

MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]
MODEL_SHORT = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _scenario_family(tag: str) -> str:
    if tag.startswith("h2d"):
        return "H2D"
    if tag.startswith("d2h"):
        return "D2H"
    if tag.startswith("hbm"):
        return "HBM"
    return "baseline"


def _scenario_x(tag: str) -> float:
    """Numeric x-axis value: BW for H2D/D2H, parallel count for HBM, 0 for baseline."""
    m = re.match(r"(?:h2d|d2h)_(\d+)_c\d+", tag)
    if m:
        return float(m.group(1))
    m = re.match(r"hbm(\d+)", tag)
    if m:
        return float(m.group(1))
    return 0.0


COLORS = {
    "baseline": "#555555",
    "H2D":      "#1f77b4",
    "D2H":      "#ff7f0e",
    "HBM":      "#2ca02c",
}
MARKERS = {"baseline": "x", "H2D": "o", "D2H": "s", "HBM": "^"}


def plot_lines_by_rate(rows: list[dict], out: Path, title: str) -> None:
    """SLO-limited tok/s vs request_rate, one line per scenario."""
    by_scenario: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        tag = r.get("injection_scenario", "")
        rate = float(r.get("request_rate", 0) or 0)
        slt = r.get("slo_limited_tok_s")
        if slt is not None:
            by_scenario[tag].append((rate, float(slt)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for tag in sorted(by_scenario):
        pts = sorted(by_scenario[tag])
        fam = _scenario_family(tag)
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker=MARKERS.get(fam, "o"), linewidth=2,
                color=COLORS.get(fam, "gray"),
                label=tag, alpha=0.85)
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("SLO-limited output tok/s")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=170)
    plt.close(fig)


def plot_heatmap(rows: list[dict], out: Path, title: str) -> None:
    """Heatmap: rows = request_rates, cols = injection scenarios."""
    rates = sorted({float(r.get("request_rate", 0) or 0) for r in rows})
    tags = sorted({r.get("injection_scenario", "") for r in rows})
    if not rates or not tags:
        return

    data = np.full((len(rates), len(tags)), np.nan)
    for r in rows:
        rate = float(r.get("request_rate", 0) or 0)
        tag = r.get("injection_scenario", "")
        slt = r.get("slo_limited_tok_s")
        if slt is not None and rate in rates and tag in tags:
            ri = rates.index(rate)
            ci = tags.index(tag)
            data[ri, ci] = float(slt)

    fig, ax = plt.subplots(figsize=(max(8, len(tags) * 0.7), max(4, len(rates) * 0.6)))
    im = ax.imshow(data, aspect="auto", cmap="YlGn", interpolation="nearest")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rates)))
    ax.set_yticklabels([f"{r:g}" for r in rates])
    ax.set_xlabel("Injection scenario")
    ax.set_ylabel("Request rate (req/s)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="SLO-limited tok/s")

    # Annotate cells with values
    for ri in range(len(rates)):
        for ci in range(len(tags)):
            v = data[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="black")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=170)
    plt.close(fig)


def plot_delta_from_baseline(rows: list[dict], out: Path, title: str) -> None:
    """% change in SLO-limited tok/s vs baseline_no_worker, per (rate, scenario)."""
    baseline: dict[float, float] = {}
    for r in rows:
        if r.get("injection_scenario") == "baseline_no_worker":
            rate = float(r.get("request_rate", 0) or 0)
            slt = r.get("slo_limited_tok_s")
            if slt is not None:
                baseline[rate] = float(slt)

    rates = sorted({float(r.get("request_rate", 0) or 0) for r in rows})
    tags = sorted({r.get("injection_scenario", "") for r in rows
                   if r.get("injection_scenario") != "baseline_no_worker"})
    if not rates or not tags or not baseline:
        return

    data = np.full((len(rates), len(tags)), np.nan)
    for r in rows:
        tag = r.get("injection_scenario", "")
        if tag == "baseline_no_worker":
            continue
        rate = float(r.get("request_rate", 0) or 0)
        slt = r.get("slo_limited_tok_s")
        b = baseline.get(rate)
        if slt is not None and b and b > 0 and rate in rates and tag in tags:
            ri = rates.index(rate)
            ci = tags.index(tag)
            data[ri, ci] = 100.0 * (float(slt) - b) / b

    fig, ax = plt.subplots(figsize=(max(8, len(tags) * 0.7), max(4, len(rates) * 0.6)))
    vmax = max(5.0, float(np.nanmax(np.abs(data)))) if not np.all(np.isnan(data)) else 20.0
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rates)))
    ax.set_yticklabels([f"{r:g}" for r in rates])
    ax.set_xlabel("Injection scenario")
    ax.set_ylabel("Request rate (req/s)")
    ax.set_title(f"{title}\n% change vs baseline (negative = degradation)")
    plt.colorbar(im, ax=ax, label="Δ SLO-limited tok/s (%)")

    for ri in range(len(rates)):
        for ci in range(len(tags)):
            v = data[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:+.1f}%", ha="center", va="center",
                        fontsize=7, color="black")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT,
                    help="Directory containing injection_sweep_v5.jsonl")
    args = ap.parse_args()

    results = args.root / "injection_sweep_v5.jsonl"
    rows = read_jsonl(results)
    if not rows:
        print(f"No rows in {results}; nothing to plot.")
        return

    plots_root = args.root / "plots"

    # Group by (model, phase)
    by_mp: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        model = r.get("model", "")
        phase = r.get("phase", "")
        by_mp[(model, phase)].append(r)

    for (model, phase), phase_rows in sorted(by_mp.items()):
        ms = MODEL_SHORT.get(model, model)
        out_dir = plots_root / model.replace("/", "__") / phase
        title_base = f"{ms} — {phase}"

        plot_lines_by_rate(
            phase_rows,
            out_dir / "lines_slo_limited_tok_s_vs_rate.png",
            f"{title_base}: SLO-limited tok/s vs request rate",
        )
        plot_heatmap(
            phase_rows,
            out_dir / "heatmap_slo_limited_tok_s.png",
            f"{title_base}: SLO-limited tok/s heatmap (rate × scenario)",
        )
        plot_delta_from_baseline(
            phase_rows,
            out_dir / "heatmap_delta_from_baseline.png",
            f"{title_base}: Δ SLO-limited tok/s vs no-injection baseline",
        )
        print(f"Wrote plots to {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
