#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SLO-limited tok/s vs PCIe injection BW (continuous curves).

For each (model, phase) writes one PNG with two subplots:

  Left:  H2D PCIe injection BW sweep (0, 16, 56 Gbps)
  Right: D2H PCIe injection BW sweep (0, 16, 56 Gbps)

A separate line per request_rate makes it easy to see how the BW degradation
shape changes as the system approaches saturation.

Reads the combined injection_sweep_v5 JSONL produced by run_injection_sweep_v5.

Usage::

  python plot_slo_vs_bw_curve.py --jsonl <path> --out-dir <dir>
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

MODEL_SHORT = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B",
}

_BW_RE = re.compile(r"^(h2d|d2h)_(\d+)_c(\d+)$")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _classify(scenario: str) -> tuple[str | None, int | None]:
    """Return (direction, gbps) for h2d/d2h scenarios. Baseline -> ('baseline', 0)."""
    if scenario == "baseline_no_worker":
        return "baseline", 0
    m = _BW_RE.match(scenario)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def _curve(rows: list[dict], direction: str) -> dict[float, list[tuple[int, float]]]:
    """Return {rate: [(gbps, slo_tok_s), ...]} for the given direction.

    Each rate's list begins with the (0, baseline) point and is extended with
    that direction's measured (gbps, slo_tok_s) cells.
    """
    curve: dict[float, list[tuple[int, float]]] = defaultdict(list)
    baselines: dict[float, float] = {}
    for r in rows:
        rate = float(r.get("request_rate", 0) or 0)
        slt = r.get("slo_limited_tok_s")
        if slt is None or float(slt) <= 0:
            continue
        d, gbps = _classify(str(r.get("injection_scenario", "")))
        if d == "baseline":
            baselines[rate] = float(slt)
        elif d == direction and gbps is not None:
            curve[rate].append((gbps, float(slt)))

    for rate, base in baselines.items():
        curve[rate].append((0, base))

    for rate in curve:
        curve[rate] = sorted(curve[rate], key=lambda p: p[0])
    return curve


def _plot_panel(ax: plt.Axes, curve: dict[float, list[tuple[int, float]]],
                title: str) -> None:
    cmap = plt.get_cmap("viridis")
    rates = sorted(curve.keys())
    for idx, rate in enumerate(rates):
        pts = curve[rate]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = cmap(0.15 + 0.7 * (idx / max(1, len(rates) - 1)))
        ax.plot(xs, ys, marker="o", linewidth=2, color=color,
                label=f"r={rate:g} req/s")
    ax.set_xlabel("Injection BW (Gbps)")
    ax.set_ylabel("SLO-limited tok/s")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 16, 32, 56])
    ax.legend(fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=Path(
            "/data/ben/kv_phase_a_internal/open_loop_eval_v1/"
            "injection_sweep_v5_combined/injection_sweep_v5.jsonl"
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/data/ben/kv_phase_a_internal/open_loop_eval_v1/"
            "injection_sweep_v5_combined/plots"
        ),
    )
    args = ap.parse_args()

    rows = _read_jsonl(args.jsonl)
    if not rows:
        raise SystemExit(f"No rows in {args.jsonl}")

    by_mp: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_mp[(str(r.get("model", "")), str(r.get("phase", "")))].append(r)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for (model, phase), phase_rows in sorted(by_mp.items()):
        h2d = _curve(phase_rows, "h2d")
        d2h = _curve(phase_rows, "d2h")
        if not h2d and not d2h:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        ms = MODEL_SHORT.get(model, model)
        _plot_panel(axes[0], h2d, "H2D PCIe (CPU -> GPU)")
        _plot_panel(axes[1], d2h, "D2H PCIe (GPU -> CPU)")
        fig.suptitle(f"{ms} - {phase}: SLO-limited tok/s vs PCIe injection BW",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        slug = model.replace("/", "__")
        out = args.out_dir / f"slo_vs_bw_curves_{slug}_{phase}.png"
        fig.savefig(str(out), dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
