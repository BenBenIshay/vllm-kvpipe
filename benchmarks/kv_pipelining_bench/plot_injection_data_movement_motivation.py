#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paper figures: extra data movement during decode hurts TBT / SLO throughput.

**Purpose (motivation, not feature calibration):** Show that injecting host↔device
or HBM-style memory traffic while vLLM is decoding on an H100 degrades decode
p99 ITL (TBT) and SLO-limited tok/s. This supports the argument that a
layer-KV system—which adds CPU↔GPU KV traffic—needs SLO-aware admission control.

This script does **not** compare feature-measured Gbps to the injection x-axis.

Outputs (under ``--out-dir``):

  ``injection_motivation_p99_itl_vs_bw_{model}_decode.png``
    2×2: p99 ITL vs PCIe Gbps (H2D / D2H) at sub-sat (r=16) and sat (r=64).

  ``injection_motivation_scenarios_{model}_decode_r{rate}.png``
    Bar chart: baseline vs H2D/D2H/HBM scenarios at a chosen rate.

  ``injection_motivation_combined_{model}_decode_r{rate}.png``
    2×2 summary: SLO tok/s + p99 ITL vs BW at r=16 (motivation slide).

Usage::

  python plot_injection_data_movement_motivation.py \\
    --jsonl /data/ben/.../injection_sweep_v5_combined/injection_sweep_v5.jsonl
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

SCENARIO_ORDER = [
    "baseline_no_worker",
    "h2d_16_c1",
    "h2d_56_c1",
    "d2h_16_c1",
    "d2h_56_c1",
    "hbm1",
    "hbm2",
    "hbm4",
]
SCENARIO_LABEL = {
    "baseline_no_worker": "baseline",
    "h2d_16_c1": "H2D 16",
    "h2d_56_c1": "H2D 56",
    "d2h_16_c1": "D2H 16",
    "d2h_56_c1": "D2H 56",
    "hbm1": "HBM×1",
    "hbm2": "HBM×2",
    "hbm4": "HBM×4",
}


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
    if scenario == "baseline_no_worker":
        return "baseline", 0
    m = _BW_RE.match(scenario)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def _curve_metric(
    rows: list[dict],
    direction: str,
    metric: str,
) -> dict[float, list[tuple[int, float]]]:
    """{rate: [(gbps, metric_value), ...]} for h2d or d2h."""
    curve: dict[float, list[tuple[int, float]]] = defaultdict(list)
    baselines: dict[float, float] = {}
    for r in rows:
        rate = float(r.get("request_rate", 0) or 0)
        val = r.get(metric)
        if val is None:
            continue
        val = float(val)
        d, gbps = _classify(str(r.get("injection_scenario", "")))
        if d == "baseline":
            baselines[rate] = val
        elif d == direction and gbps is not None:
            curve[rate].append((gbps, val))
    for rate, base in baselines.items():
        curve[rate].append((0, base))
    for rate in curve:
        curve[rate] = sorted(curve[rate], key=lambda p: p[0])
    return curve


def _plot_bw_panel(
    ax: plt.Axes,
    curve: dict[float, list[tuple[int, float]]],
    *,
    title: str,
    ylabel: str,
    rates: list[float],
) -> None:
    cmap = plt.get_cmap("viridis")
    plotted = [rt for rt in rates if rt in curve and curve[rt]]
    for idx, rate in enumerate(plotted):
        pts = curve[rate]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = cmap(0.15 + 0.7 * (idx / max(1, len(plotted) - 1)))
        ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=f"r={rate:g} req/s")
    ax.set_xlabel("Injected PCIe BW (Gbps); 0 = no worker")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 16, 32, 56])
    if plotted:
        ax.legend(fontsize=7, loc="best")


def _plot_p99_itl_vs_bw(
    phase_rows: list[dict],
    model: str,
    out_dir: Path,
    *,
    rates_sub: list[float],
    rates_sat: list[float],
) -> Path:
    h2d = _curve_metric(phase_rows, "h2d", "p99_itl_ms")
    d2h = _curve_metric(phase_rows, "d2h", "p99_itl_ms")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ms = MODEL_SHORT.get(model, model)
    _plot_bw_panel(axes[0][0], h2d, title=f"Sub-sat — H2D (rates {rates_sub})",
                   ylabel="p99 ITL (ms) = TBT tail", rates=rates_sub)
    _plot_bw_panel(axes[0][1], d2h, title="Sub-sat — D2H",
                   ylabel="p99 ITL (ms)", rates=rates_sub)
    _plot_bw_panel(axes[1][0], h2d, title=f"Saturation — H2D (rates {rates_sat})",
                   ylabel="p99 ITL (ms)", rates=rates_sat)
    _plot_bw_panel(axes[1][1], d2h, title="Saturation — D2H",
                   ylabel="p99 ITL (ms)", rates=rates_sat)
    fig.suptitle(
        f"{ms} decode: extra host↔device traffic increases p99 TBT\n"
        "(vanilla vLLM + injection worker; motivates SLO-aware admission for layer-KV)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    slug = model.replace("/", "__")
    out = out_dir / f"injection_motivation_p99_itl_vs_bw_{slug}_decode.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_combined_motivation(
    phase_rows: list[dict],
    model: str,
    out_dir: Path,
    *,
    rate: float,
) -> Path:
    """2×2 at one rate: SLO tok/s and p99 ITL for H2D and D2H."""
    sub = [r for r in phase_rows if float(r.get("request_rate", 0) or 0) == rate]
    slo_h2d = _curve_metric(sub, "h2d", "slo_limited_tok_s")
    slo_d2h = _curve_metric(sub, "d2h", "slo_limited_tok_s")
    itl_h2d = _curve_metric(sub, "h2d", "p99_itl_ms")
    itl_d2h = _curve_metric(sub, "d2h", "p99_itl_ms")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ms = MODEL_SHORT.get(model, model)
    _plot_bw_panel(axes[0][0], slo_h2d, title="SLO-limited tok/s — H2D",
                   ylabel="SLO-limited tok/s", rates=[rate])
    _plot_bw_panel(axes[0][1], slo_d2h, title="SLO-limited tok/s — D2H",
                   ylabel="SLO-limited tok/s", rates=[rate])
    _plot_bw_panel(axes[1][0], itl_h2d, title="p99 ITL (TBT) — H2D",
                   ylabel="p99 ITL (ms)", rates=[rate])
    _plot_bw_panel(axes[1][1], itl_d2h, title="p99 ITL (TBT) — D2H",
                   ylabel="p99 ITL (ms)", rates=[rate])
    fig.suptitle(
        f"{ms} decode @ r={rate:g} req/s: data movement degrades throughput and TBT",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    slug = model.replace("/", "__")
    out = out_dir / f"injection_motivation_combined_{slug}_decode_r{int(rate)}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_scenario_bars(
    phase_rows: list[dict],
    model: str,
    out_dir: Path,
    *,
    rate: float,
) -> Path:
    sub = [r for r in phase_rows
           if float(r.get("request_rate", 0) or 0) == rate]
    by_scn: dict[str, dict] = {}
    for r in sub:
        scn = str(r.get("injection_scenario", ""))
        if scn in SCENARIO_ORDER:
            by_scn[scn] = r
    labels, slo, itl = [], [], []
    for scn in SCENARIO_ORDER:
        if scn not in by_scn:
            continue
        labels.append(SCENARIO_LABEL.get(scn, scn))
        slo.append(float(by_scn[scn].get("slo_limited_tok_s") or 0))
        itl.append(float(by_scn[scn].get("p99_itl_ms") or 0))
    if not labels:
        raise ValueError(f"No scenarios at rate={rate}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = range(len(labels))
    axes[0].bar(x, slo, color="#5cb85c", edgecolor="#333")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylabel("SLO-limited tok/s")
    axes[0].set_title("Throughput under SLO")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, itl, color="#d9534f", edgecolor="#333")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("p99 ITL (ms)")
    axes[1].set_title("Decode tail latency (TBT)")
    axes[1].grid(axis="y", alpha=0.3)

    ms = MODEL_SHORT.get(model, model)
    fig.suptitle(
        f"{ms} decode @ r={rate:g} req/s: PCIe + HBM contention vs baseline",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    slug = model.replace("/", "__")
    out = out_dir / f"injection_motivation_scenarios_{slug}_decode_r{int(rate)}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
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
            "injection_sweep_v5_combined/plots/motivation"
        ),
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=[
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
        ],
    )
    ap.add_argument("--rate-sub", type=float, default=16.0,
                    help="Sub-saturation decode rate for scenario bars and combined panel.")
    ap.add_argument("--rate-sat", type=float, default=64.0)
    args = ap.parse_args()

    rows = _read_jsonl(args.jsonl)
    if not rows:
        raise SystemExit(f"No rows in {args.jsonl}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        dec = [r for r in rows
               if r.get("phase") == "decode" and r.get("model") == model]
        if not dec:
            print(f"[skip] no decode rows for {model}")
            continue
        p1 = _plot_p99_itl_vs_bw(
            dec, model, args.out_dir,
            rates_sub=[args.rate_sub],
            rates_sat=[args.rate_sat, 128.0],
        )
        print(f"Wrote {p1}")
        p2 = _plot_combined_motivation(dec, model, args.out_dir, rate=args.rate_sub)
        print(f"Wrote {p2}")
        p3 = _plot_scenario_bars(dec, model, args.out_dir, rate=args.rate_sub)
        print(f"Wrote {p3}")


if __name__ == "__main__":
    main()
