#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paper figure: 2x2 panel of SLO-limited tok/s vs interference scenario at the
saturation operating point.

Rows are models (Llama-3.1-8B, Qwen2.5-14B); columns are phases (prefill, decode).
Each panel is a bar plot at the saturation rate for that (model, phase) cell:

  prefill  -> request_rate = 16 req/s (highest measured saturation rate)
  decode   -> request_rate = 128 req/s (highest measured plateau rate)

Bars are scenarios from injection_sweep_v5.jsonl:

  baseline_no_worker, h2d_16, h2d_56, d2h_16, d2h_56, hbm1, hbm2, hbm4

The baseline bar is annotated with absolute SLO-tok/s; injection bars are
annotated with the percent delta vs the baseline (negative = degradation).

Usage::

  python plot_injection_paper_figure.py \
      --jsonl /data/ben/kv_phase_a_internal/open_loop_eval_v1/\
injection_sweep_v5_combined/injection_sweep_v5.jsonl \
      --out paper_2x2_slo_vs_scenario.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS_ORDER: Sequence[str] = (
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
)
PHASES_ORDER: Sequence[str] = ("prefill", "decode")

MODEL_SHORT = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B",
}

# Saturation operating point per phase. The prefill workload (i=4096, o=1)
# saturates by r=16; the decode workload (1024/1024) reaches a plateau
# around r=128 (see length_dependent_slo_tok_s_combined_v3.png).
SAT_RATE = {"prefill": 16.0, "decode": 128.0}

SCENARIO_ORDER: Sequence[str] = (
    "baseline_no_worker",
    "h2d_16_c1",
    "h2d_56_c1",
    "d2h_16_c1",
    "d2h_56_c1",
    "hbm1",
    "hbm2",
    "hbm4",
)
SCENARIO_LABEL = {
    "baseline_no_worker": "no inj.",
    "h2d_16_c1": "H2D 16",
    "h2d_56_c1": "H2D 56",
    "d2h_16_c1": "D2H 16",
    "d2h_56_c1": "D2H 56",
    "hbm1": "HBM x1",
    "hbm2": "HBM x2",
    "hbm4": "HBM x4",
}
SCENARIO_FAMILY = {
    "baseline_no_worker": "baseline",
    "h2d_16_c1": "H2D",
    "h2d_56_c1": "H2D",
    "d2h_16_c1": "D2H",
    "d2h_56_c1": "D2H",
    "hbm1": "HBM",
    "hbm2": "HBM",
    "hbm4": "HBM",
}
FAMILY_COLOR = {
    "baseline": "#4d4d4d",
    "H2D": "#1f77b4",
    "D2H": "#ff7f0e",
    "HBM": "#2ca02c",
}


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _index(rows: list[dict]) -> dict[tuple[str, str, float, str], dict]:
    """Index rows by (model, phase, request_rate, injection_scenario)."""
    idx: dict[tuple[str, str, float, str], dict] = {}
    for r in rows:
        key = (
            str(r.get("model", "")),
            str(r.get("phase", "")),
            float(r.get("request_rate", 0) or 0),
            str(r.get("injection_scenario", "")),
        )
        slt = r.get("slo_limited_tok_s")
        if slt is None or float(slt) <= 0:
            continue
        idx[key] = r
    return idx


def _panel(
    ax: plt.Axes,
    indexed: dict[tuple[str, str, float, str], dict],
    model: str,
    phase: str,
) -> None:
    rate = SAT_RATE[phase]
    base_row = indexed.get((model, phase, rate, "baseline_no_worker"))
    if base_row is None:
        ax.text(0.5, 0.5, "no baseline", ha="center", va="center",
                transform=ax.transAxes)
        return
    baseline = float(base_row["slo_limited_tok_s"])

    ys: list[float] = []
    deltas: list[float | None] = []
    families: list[str] = []
    labels: list[str] = []
    for scen in SCENARIO_ORDER:
        row = indexed.get((model, phase, rate, scen))
        if row is None:
            continue
        v = float(row["slo_limited_tok_s"])
        ys.append(v)
        labels.append(SCENARIO_LABEL[scen])
        families.append(SCENARIO_FAMILY[scen])
        if scen == "baseline_no_worker":
            deltas.append(None)
        else:
            deltas.append(100.0 * (v - baseline) / baseline)

    xs = np.arange(len(ys))
    colors = [FAMILY_COLOR[f] for f in families]
    bars = ax.bar(xs, ys, color=colors, edgecolor="black", linewidth=0.6)

    # Reference dashed line at baseline level for visual comparison.
    ax.axhline(baseline, color=FAMILY_COLOR["baseline"], linestyle="--",
               linewidth=0.8, alpha=0.7)

    # Annotate each bar with absolute value (baseline) or % delta (others).
    ymax = max(ys) if ys else 1.0
    for bar, v, d in zip(bars, ys, deltas):
        x = bar.get_x() + bar.get_width() / 2
        if d is None:
            label = f"{v:.0f}"
        else:
            label = f"{d:+.1f}%"
        ax.text(x, v + 0.02 * ymax, label, ha="center", va="bottom",
                fontsize=7.5)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("SLO-limited tok/s")
    ax.set_title(f"{MODEL_SHORT.get(model, model)} - {phase} (r={rate:g})",
                 fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, ymax * 1.18)


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
        help="Combined injection sweep JSONL (default: standard repo location).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/data/ben/kv_phase_a_internal/open_loop_eval_v1/"
            "injection_sweep_v5_combined/plots/paper_2x2_slo_vs_scenario.png"
        ),
        help="Output PNG path.",
    )
    ap.add_argument("--llama-prefill-rate", type=float, default=None,
                    help="Override Llama prefill saturation rate.")
    ap.add_argument("--llama-decode-rate", type=float, default=None,
                    help="Override Llama decode saturation rate.")
    ap.add_argument("--qwen-prefill-rate", type=float, default=None,
                    help="Override Qwen prefill saturation rate.")
    ap.add_argument("--qwen-decode-rate", type=float, default=None,
                    help="Override Qwen decode saturation rate.")
    args = ap.parse_args()

    if not args.jsonl.is_file():
        raise SystemExit(f"JSONL not found: {args.jsonl}")
    rows = _read_jsonl(args.jsonl)
    if not rows:
        raise SystemExit(f"No rows in {args.jsonl}")
    indexed = _index(rows)

    # Allow per-cell rate overrides if provided.
    rate_override = {
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "prefill"): args.llama_prefill_rate,
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "decode"): args.llama_decode_rate,
        ("Qwen/Qwen2.5-14B-Instruct", "prefill"): args.qwen_prefill_rate,
        ("Qwen/Qwen2.5-14B-Instruct", "decode"): args.qwen_decode_rate,
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=False)
    for r_idx, model in enumerate(MODELS_ORDER):
        for c_idx, phase in enumerate(PHASES_ORDER):
            ax = axes[r_idx][c_idx]
            override = rate_override.get((model, phase))
            if override is not None:
                SAT_RATE[phase] = float(override)
            _panel(ax, indexed, model, phase)

    fig.suptitle(
        "SLO-limited tok/s vs PCIe / HBM interference at saturation",
        fontsize=12,
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in
        ("baseline", "H2D", "D2H", "HBM")
    ]
    fig.legend(handles, ["baseline", "H2D PCIe", "D2H PCIe", "HBM contention"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01),
               frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
