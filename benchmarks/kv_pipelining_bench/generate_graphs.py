#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate publication-quality graphs for KV Cache Pipelining evaluation.

Produces:
  1. ITL vs Concurrency (baseline / LMCache / pipelined_inflated per K)
  2. SLO Attainment curves
  3. GPU Memory bar chart
  4. Max Capacity bar chart
  5. GPU Utilization timeline
  6. Cost Efficiency comparison
  7. K-sweep ITL heatmap
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Try matplotlib — fall back gracefully
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, generating text report only")

# Try numpy
try:
    import numpy as np

    HAS_NP = True
except ImportError:
    HAS_NP = False

# =========================================================================
# Style
# =========================================================================
COLORS = {
    "baseline": "#2196F3",
    "lmcache": "#FF9800",
    "pipelined_inflated": "#4CAF50",
    "pipelined_inflated_k6": "#66BB6A",
    "pipelined_inflated_k8": "#43A047",
    "pipelined_inflated_k10": "#2E7D32",
}
MARKERS = {
    "baseline": "o",
    "lmcache": "s",
    "pipelined_inflated": "^",
    "pipelined_inflated_k6": "v",
    "pipelined_inflated_k8": "D",
    "pipelined_inflated_k10": "P",
}

# H100 NVL cloud cost ($/hour) — for cost efficiency computation
H100_COST_PER_HOUR = 3.50


def set_style():
    if not HAS_MPL:
        return
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


# =========================================================================
# Data loading
# =========================================================================


def load_json_results(result_dir: str, pattern: str = "results_*.json") -> dict:
    """Load all JSON result files from a directory."""
    results = {}
    path = Path(result_dir)
    if not path.exists():
        return results
    for f in sorted(path.glob(pattern)):
        try:
            with open(f) as fh:
                data = json.load(fh)
            # Extract mode and concurrency from filename
            # e.g., results_pipelined_inflated_conc64.json
            name = f.stem  # results_pipelined_inflated_conc64
            results[name] = data
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: could not load {f}: {e}")
    return results


def extract_metric(results: dict, mode: str, conc: int, metric: str) -> float | None:
    """Extract a specific metric from results."""
    key = f"results_{mode}_conc{conc}"
    r = results.get(key, {})
    return r.get(metric)


def load_gpu_csv(csv_path: str) -> list[dict]:
    """Load GPU monitoring CSV."""
    rows = []
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except (FileNotFoundError, IOError):
        pass
    return rows


def parse_server_capacity(log_file: str) -> dict:
    """Extract capacity info from server startup log."""
    info = {}
    try:
        with open(log_file) as f:
            for line in f:
                m = re.search(
                    r"Maximum concurrency for ([\d,]+) tokens.*?:\s*([\d.]+)x",
                    line,
                )
                if m:
                    info["max_concurrency_reported"] = float(m.group(2))
                m = re.search(
                    r"Available KV cache memory:\s*([\d.]+)\s*GiB", line
                )
                if m:
                    info["kv_cache_memory_gib"] = float(m.group(1))
    except FileNotFoundError:
        pass
    return info


# =========================================================================
# Metric computation
# =========================================================================


def compute_slo_attainment(results: dict, mode: str, conc: int,
                           thresholds_ms: list[float] = None) -> dict:
    """Compute SLO attainment from per-request ITL data.

    Since vllm bench serve JSON includes mean_itl_ms and p99_itl_ms,
    we estimate SLO attainment from these percentile values.
    """
    if thresholds_ms is None:
        thresholds_ms = [10, 15, 20, 50]

    key = f"results_{mode}_conc{conc}"
    r = results.get(key, {})
    mean_itl = r.get("mean_itl_ms")
    p99_itl = r.get("p99_itl_ms")
    p50_itl = mean_itl  # approximate

    if mean_itl is None:
        return {}

    slo = {}
    for t in thresholds_ms:
        # Estimate: if p99 < threshold, ~99% of tokens meet SLO
        # if mean < threshold < p99, estimate based on position
        if p99_itl is not None and p99_itl <= t:
            slo[f"slo_{t}ms"] = 99.0
        elif mean_itl <= t:
            if p99_itl is not None and p99_itl > 0:
                # Linear interpolation estimate between mean (~50th) and p99
                ratio = (t - mean_itl) / (p99_itl - mean_itl)
                pct = min(50.0 + ratio * 49.0, 99.0)
                slo[f"slo_{t}ms"] = round(pct, 1)
            else:
                slo[f"slo_{t}ms"] = 95.0
        else:
            # mean > threshold — most tokens miss SLO
            if p99_itl is not None and p99_itl > 0:
                ratio = t / mean_itl
                slo[f"slo_{t}ms"] = round(max(ratio * 50.0, 0), 1)
            else:
                slo[f"slo_{t}ms"] = 0.0
    return slo


def compute_cost_efficiency(results: dict, mode: str, conc: int) -> dict:
    """Compute cost efficiency metrics."""
    key = f"results_{mode}_conc{conc}"
    r = results.get(key, {})
    throughput = r.get("output_throughput", 0)

    if throughput <= 0:
        return {}

    cost_per_second = H100_COST_PER_HOUR / 3600
    tokens_per_dollar = throughput / cost_per_second if cost_per_second > 0 else 0
    cost_per_million = (cost_per_second / throughput) * 1e6 if throughput > 0 else 0

    return {
        "tokens_per_gpu_second": throughput,
        "tokens_per_dollar": round(tokens_per_dollar, 0),
        "cost_per_million_tokens": round(cost_per_million, 4),
    }


# =========================================================================
# Graph generation
# =========================================================================


def plot_itl_vs_concurrency(
    all_data: dict[str, dict],
    concurrency_levels: list[int],
    model_name: str,
    output_dir: str,
):
    """Plot Mean ITL vs Concurrency for all modes."""
    if not HAS_MPL:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for label, data in all_data.items():
        color = COLORS.get(label, "#999999")
        marker = MARKERS.get(label, "o")
        x, y_mean, y_p99 = [], [], []
        for conc in concurrency_levels:
            mean_itl = extract_metric(data, _mode_from_label(label), conc, "mean_itl_ms")
            p99_itl = extract_metric(data, _mode_from_label(label), conc, "p99_itl_ms")
            if mean_itl is not None:
                x.append(conc)
                y_mean.append(mean_itl)
                y_p99.append(p99_itl if p99_itl else mean_itl)

        if x:
            display_name = _display_name(label)
            ax1.plot(x, y_mean, marker=marker, color=color, label=display_name,
                     linewidth=2, markersize=8)
            ax2.plot(x, y_p99, marker=marker, color=color, label=display_name,
                     linewidth=2, markersize=8)

    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("Mean ITL (ms)")
    ax1.set_title(f"Mean ITL vs Concurrency — {model_name}")
    ax1.legend()
    ax1.set_xticks(concurrency_levels)

    ax2.set_xlabel("Concurrency")
    ax2.set_ylabel("P99 ITL (ms)")
    ax2.set_title(f"P99 ITL vs Concurrency — {model_name}")
    ax2.legend()
    ax2.set_xticks(concurrency_levels)

    plt.tight_layout()
    out = os.path.join(output_dir, f"itl_vs_concurrency_{_short_model(model_name)}.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def plot_throughput_vs_concurrency(
    all_data: dict[str, dict],
    concurrency_levels: list[int],
    model_name: str,
    output_dir: str,
):
    """Plot Throughput vs Concurrency."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, data in all_data.items():
        color = COLORS.get(label, "#999999")
        marker = MARKERS.get(label, "o")
        x, y = [], []
        for conc in concurrency_levels:
            tput = extract_metric(data, _mode_from_label(label), conc, "output_throughput")
            if tput is not None:
                x.append(conc)
                y.append(tput)

        if x:
            ax.plot(x, y, marker=marker, color=color, label=_display_name(label),
                    linewidth=2, markersize=8)

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Output Throughput (tokens/s)")
    ax.set_title(f"Throughput vs Concurrency — {model_name}")
    ax.legend()
    ax.set_xticks(concurrency_levels)

    plt.tight_layout()
    out = os.path.join(output_dir, f"throughput_vs_concurrency_{_short_model(model_name)}.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def plot_slo_attainment(
    all_data: dict[str, dict],
    concurrency_levels: list[int],
    model_name: str,
    output_dir: str,
    slo_threshold_ms: float = 15.0,
):
    """Plot SLO attainment at a given threshold across concurrency levels."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, data in all_data.items():
        color = COLORS.get(label, "#999999")
        marker = MARKERS.get(label, "o")
        x, y = [], []
        mode = _mode_from_label(label)
        for conc in concurrency_levels:
            slo = compute_slo_attainment(data, mode, conc, [slo_threshold_ms])
            val = slo.get(f"slo_{slo_threshold_ms}ms")
            if val is not None:
                x.append(conc)
                y.append(val)

        if x:
            ax.plot(x, y, marker=marker, color=color, label=_display_name(label),
                    linewidth=2, markersize=8)

    ax.set_xlabel("Concurrency")
    ax.set_ylabel(f"SLO Attainment (% tokens < {slo_threshold_ms}ms ITL)")
    ax.set_title(f"SLO Attainment @ {slo_threshold_ms}ms — {model_name}")
    ax.set_ylim(0, 105)
    ax.axhline(y=99, color="red", linestyle="--", alpha=0.5, label="99% target")
    ax.legend()
    ax.set_xticks(concurrency_levels)

    plt.tight_layout()
    out = os.path.join(output_dir, f"slo_attainment_{_short_model(model_name)}.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def plot_memory_comparison(
    capacities: dict[str, dict],
    model_name: str,
    output_dir: str,
):
    """Bar chart of GPU KV memory and max capacity."""
    if not HAS_MPL:
        return

    labels = list(capacities.keys())
    mem_vals = [capacities[l].get("kv_mem_gib", 0) for l in labels]
    cap_vals = [capacities[l].get("max_conc", 0) for l in labels]
    colors = [COLORS.get(l, "#999999") for l in labels]
    display_names = [_display_name(l) for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars1 = ax1.bar(display_names, mem_vals, color=colors, edgecolor="white", linewidth=1.5)
    ax1.set_ylabel("KV Cache Memory (GiB)")
    ax1.set_title(f"GPU KV Memory — {model_name}")
    for bar, val in zip(bars1, mem_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontweight="bold")

    bars2 = ax2.bar(display_names, cap_vals, color=colors, edgecolor="white", linewidth=1.5)
    ax2.set_ylabel("Max Concurrent Requests")
    ax2.set_title(f"Serving Capacity — {model_name}")
    for bar, val in zip(bars2, cap_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(output_dir, f"memory_capacity_{_short_model(model_name)}.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def plot_cost_efficiency(
    all_data: dict[str, dict],
    concurrency_levels: list[int],
    model_name: str,
    output_dir: str,
):
    """Plot cost per million tokens across concurrency levels."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, data in all_data.items():
        color = COLORS.get(label, "#999999")
        marker = MARKERS.get(label, "o")
        x, y = [], []
        mode = _mode_from_label(label)
        for conc in concurrency_levels:
            cost = compute_cost_efficiency(data, mode, conc)
            val = cost.get("cost_per_million_tokens")
            if val is not None and val > 0:
                x.append(conc)
                y.append(val)

        if x:
            ax.plot(x, y, marker=marker, color=color, label=_display_name(label),
                    linewidth=2, markersize=8)

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Cost per Million Tokens ($)")
    ax.set_title(f"Cost Efficiency — {model_name} (H100 @ ${H100_COST_PER_HOUR}/hr)")
    ax.legend()
    ax.set_xticks(concurrency_levels)

    plt.tight_layout()
    out = os.path.join(output_dir, f"cost_efficiency_{_short_model(model_name)}.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def plot_gpu_utilization(
    gpu_csv_files: dict[str, str],
    model_name: str,
    output_dir: str,
):
    """Plot GPU utilization timeline from CSV files."""
    if not HAS_MPL:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for label, csv_path in gpu_csv_files.items():
        rows = load_gpu_csv(csv_path)
        if not rows:
            continue

        color = COLORS.get(label, "#999999")
        timestamps = list(range(len(rows)))
        gpu_util = []
        mem_used = []
        for r in rows:
            try:
                gpu_util.append(float(r.get("gpu_util_pct", "0").strip()))
            except (ValueError, AttributeError):
                gpu_util.append(0)
            try:
                mem_used.append(float(r.get("mem_used_mib", "0").strip()) / 1024)
            except (ValueError, AttributeError):
                mem_used.append(0)

        ax1.plot(timestamps, gpu_util, color=color, label=_display_name(label),
                 alpha=0.8, linewidth=1)
        ax2.plot(timestamps, mem_used, color=color, label=_display_name(label),
                 alpha=0.8, linewidth=1)

    ax1.set_ylabel("GPU Utilization (%)")
    ax1.set_title(f"GPU Utilization Timeline — {model_name}")
    ax1.legend()
    ax1.set_ylim(0, 105)

    ax2.set_ylabel("GPU Memory Used (GiB)")
    ax2.set_xlabel("Time (seconds)")
    ax2.legend()

    plt.tight_layout()
    out = os.path.join(output_dir, f"gpu_utilization_{_short_model(model_name)}.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def plot_k_sweep_heatmap(
    result_root: str,
    model_prefix: str,
    model_name: str,
    k_values: list[int],
    concurrency_levels: list[int],
    output_dir: str,
):
    """Heatmap of ITL across K values and concurrency levels."""
    if not HAS_MPL:
        return

    data = []
    for k in k_values:
        row = []
        d = os.path.join(result_root, f"{model_prefix}_ksweep_k{k}", "random")
        for conc in concurrency_levels:
            fpath = os.path.join(d, f"results_pipelined_inflated_conc{conc}.json")
            try:
                with open(fpath) as f:
                    r = json.load(f)
                    row.append(r.get("mean_itl_ms", float("nan")))
            except (FileNotFoundError, json.JSONDecodeError):
                row.append(float("nan"))
        data.append(row)

    if not any(any(not (v != v) for v in row) for row in data):  # all NaN
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if HAS_NP:
        arr = np.array(data)
        im = ax.imshow(arr, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(concurrency_levels)))
        ax.set_xticklabels([str(c) for c in concurrency_levels])
        ax.set_yticks(range(len(k_values)))
        ax.set_yticklabels([f"K={k}" for k in k_values])

        # Add text annotations
        for i in range(len(k_values)):
            for j in range(len(concurrency_levels)):
                val = arr[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=11, fontweight="bold",
                            color="white" if val > 15 else "black")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mean ITL (ms)")
    else:
        # Fallback: simple table
        ax.text(0.5, 0.5, "numpy required for heatmap",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("K Window")
    ax.set_title(f"ITL Heatmap (K × Concurrency) — {model_name}")

    plt.tight_layout()
    out = os.path.join(output_dir, f"k_sweep_heatmap_{_short_model(model_name)}.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# =========================================================================
# Text report
# =========================================================================


def generate_text_report(
    result_root: str,
    prev_results: str,
    output_dir: str,
):
    """Generate a comprehensive text report."""
    report_lines = []
    rpt = report_lines.append

    rpt("=" * 100)
    rpt("  KV CACHE PIPELINING — COMPREHENSIVE EVALUATION REPORT v2")
    rpt("  K-sweep + LMCache + SLO Attainment + Cost Efficiency")
    rpt("=" * 100)
    rpt("")

    for model_prefix, model_name, baseline_itl_64 in [
        ("llama8b", "Llama-3.1-8B", 8.65),
        ("qwen14b", "Qwen2.5-14B", 15.03),
    ]:
        rpt(f"\n{'=' * 100}")
        rpt(f"  MODEL: {model_name}")
        rpt(f"{'=' * 100}")

        # K-sweep results
        rpt(f"\n  K-SWEEP RESULTS (pipelined_inflated, random 512→128)")
        rpt(f"  {'K':>4s}  {'conc=64':>12s}  {'conc=128':>12s}  {'conc=256':>12s}  "
            f"{'Δ vs base@64':>14s}")
        rpt(f"  {'─' * 4}  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 14}")

        for k in [6, 8, 10]:
            d = os.path.join(result_root, f"{model_prefix}_ksweep_k{k}", "random")
            vals = []
            for conc in [64, 128, 256]:
                try:
                    with open(os.path.join(d, f"results_pipelined_inflated_conc{conc}.json")) as f:
                        r = json.load(f)
                        vals.append(f"{r.get('mean_itl_ms', -1):.2f}ms")
                except:
                    vals.append("FAIL")

            # Compute delta vs baseline at conc=64
            try:
                with open(os.path.join(d, "results_pipelined_inflated_conc64.json")) as f:
                    r = json.load(f)
                    itl = r.get("mean_itl_ms", 0)
                    delta = ((itl - baseline_itl_64) / baseline_itl_64) * 100
                    delta_str = f"{delta:+.1f}%"
            except:
                delta_str = "N/A"

            rpt(f"  K={k:>2d}  {vals[0]:>12s}  {vals[1]:>12s}  {vals[2]:>12s}  "
                f"{delta_str:>14s}")

        # LMCache results
        rpt(f"\n  LMCACHE RESULTS")
        lmc_dir = os.path.join(result_root, f"{model_prefix}_lmcache", "random")
        for conc in [64, 128]:
            try:
                with open(os.path.join(lmc_dir, f"results_lmcache_conc{conc}.json")) as f:
                    r = json.load(f)
                    rpt(f"  conc={conc}: ITL={r.get('mean_itl_ms', -1):.2f}ms, "
                        f"P99={r.get('p99_itl_ms', -1):.2f}ms, "
                        f"tput={r.get('output_throughput', -1):.0f} tok/s")
            except:
                rpt(f"  conc={conc}: FAILED or not available")

        # Capacity sweep
        best_k_file = os.path.join(result_root, f"best_k_{model_prefix}.txt")
        best_k = "?"
        try:
            with open(best_k_file) as f:
                best_k = f.read().strip()
        except:
            pass

        rpt(f"\n  CAPACITY SWEEP (best K={best_k})")
        cap_dir = os.path.join(result_root, f"{model_prefix}_capacity_k{best_k}", "random")
        for conc in [64, 128, 256, 384, 512]:
            try:
                with open(os.path.join(cap_dir, f"results_pipelined_inflated_conc{conc}.json")) as f:
                    r = json.load(f)
                    failed = r.get("failed", 0)
                    status = "✓" if failed == 0 else f"✗ ({failed} failed)"
                    rpt(f"  conc={conc}: {status}, ITL={r.get('mean_itl_ms', -1):.2f}ms, "
                        f"tput={r.get('output_throughput', -1):.0f} tok/s")
            except:
                rpt(f"  conc={conc}: not tested or failed")

        # SLO + Cost summary
        rpt(f"\n  SLO ATTAINMENT & COST (baseline vs pipelined_inflated K={best_k})")
        rpt(f"  {'Mode':<25s}  {'conc':>5s}  {'ITL':>8s}  {'SLO<15ms':>10s}  "
            f"{'SLO<20ms':>10s}  {'$/Mtok':>10s}")
        rpt(f"  {'─' * 25}  {'─' * 5}  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 10}")

        # Baseline from prev results
        base_dir = os.path.join(result_root, f"{model_prefix}_random_baseline", "random")
        for conc in [64]:
            try:
                with open(os.path.join(base_dir, f"results_baseline_conc{conc}.json")) as f:
                    r_base = json.load(f)
                    base_results = {f"results_baseline_conc{conc}": r_base}
                    slo = compute_slo_attainment(base_results, "baseline", conc, [15, 20])
                    cost = compute_cost_efficiency(base_results, "baseline", conc)
                    rpt(f"  {'baseline':<25s}  {conc:>5d}  "
                        f"{r_base.get('mean_itl_ms', -1):>7.2f}  "
                        f"{slo.get('slo_15ms', 'N/A'):>9}%  "
                        f"{slo.get('slo_20ms', 'N/A'):>9}%  "
                        f"${cost.get('cost_per_million_tokens', 'N/A'):>8}")
            except:
                pass

        # Pipelined inflated (best K) from K-sweep results
        pipe_dir = os.path.join(result_root, f"{model_prefix}_ksweep_k{best_k}", "random")
        for conc in [64]:
            try:
                with open(os.path.join(pipe_dir, f"results_pipelined_inflated_conc{conc}.json")) as f:
                    r_pipe = json.load(f)
                    pipe_results = {f"results_pipelined_inflated_conc{conc}": r_pipe}
                    slo = compute_slo_attainment(pipe_results, "pipelined_inflated", conc, [15, 20])
                    cost = compute_cost_efficiency(pipe_results, "pipelined_inflated", conc)
                    rpt(f"  {f'pipelined_inflated K={best_k}':<25s}  {conc:>5d}  "
                        f"{r_pipe.get('mean_itl_ms', -1):>7.2f}  "
                        f"{slo.get('slo_15ms', 'N/A'):>9}%  "
                        f"{slo.get('slo_20ms', 'N/A'):>9}%  "
                        f"${cost.get('cost_per_million_tokens', 'N/A'):>8}")
            except:
                pass

        # LMCache comparison
        lmc_slo_dir = os.path.join(result_root, f"{model_prefix}_lmcache", "random")
        for conc in [64]:
            try:
                with open(os.path.join(lmc_slo_dir, f"results_lmcache_conc{conc}.json")) as f:
                    r_lmc = json.load(f)
                    lmc_results = {f"results_lmcache_conc{conc}": r_lmc}
                    slo = compute_slo_attainment(lmc_results, "lmcache", conc, [15, 20])
                    cost = compute_cost_efficiency(lmc_results, "lmcache", conc)
                    rpt(f"  {'lmcache':<25s}  {conc:>5d}  "
                        f"{r_lmc.get('mean_itl_ms', -1):>7.2f}  "
                        f"{slo.get('slo_15ms', 'N/A'):>9}%  "
                        f"{slo.get('slo_20ms', 'N/A'):>9}%  "
                        f"${cost.get('cost_per_million_tokens', 'N/A'):>8}")
            except:
                pass

    rpt(f"\n{'=' * 100}")
    rpt("  Report generated from: " + result_root)
    rpt(f"{'=' * 100}")

    report_text = "\n".join(report_lines)
    report_path = os.path.join(result_root, "REPORT_v2.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(report_text)
    print(f"\n  Report saved to: {report_path}")


# =========================================================================
# Helpers
# =========================================================================


def _mode_from_label(label: str) -> str:
    """Convert display label to mode name used in result files."""
    if label.startswith("pipelined_inflated_k"):
        return "pipelined_inflated"
    return label


def _display_name(label: str) -> str:
    """Pretty display name for legends."""
    names = {
        "baseline": "Baseline (all KV on GPU)",
        "lmcache": "LMCache (CPU offload)",
        "pipelined_inflated": "Ours (pipelined inflated)",
        "pipelined_inflated_k6": "Ours K=6",
        "pipelined_inflated_k8": "Ours K=8",
        "pipelined_inflated_k10": "Ours K=10",
    }
    return names.get(label, label)


def _short_model(model_name: str) -> str:
    """Short model name for filenames."""
    if "8B" in model_name or "8b" in model_name or "llama" in model_name.lower():
        return "llama8b"
    if "14B" in model_name or "14b" in model_name or "qwen" in model_name.lower():
        return "qwen14b"
    return model_name.replace("/", "_").replace(" ", "_")


# =========================================================================
# Main
# =========================================================================


def main():
    p = argparse.ArgumentParser(description="Generate evaluation graphs")
    p.add_argument("--result-root", required=True,
                   help="Root directory with v2 eval results")
    p.add_argument("--prev-results", default="/tmp/kv_full_eval",
                   help="Directory with previous (v1) baseline results")
    p.add_argument("--output-dir", default=None,
                   help="Output directory for graphs (default: result-root/graphs)")
    args = p.parse_args()

    output_dir = args.output_dir or os.path.join(args.result_root, "graphs")
    os.makedirs(output_dir, exist_ok=True)

    if HAS_MPL:
        set_style()

    # ── Generate graphs per model ──
    for model_prefix, model_name, conc_levels in [
        ("llama8b", "Llama-3.1-8B-Instruct", [1, 4, 16, 64, 128, 256]),
        ("qwen14b", "Qwen2.5-14B-Instruct", [1, 4, 16, 64, 128, 256]),
    ]:
        print(f"\n{'=' * 60}")
        print(f"  Generating graphs for {model_name}")
        print(f"{'=' * 60}")

        # Collect all results for this model
        all_data = {}

        # Baseline from previous run
        base_dir = os.path.join(args.result_root, f"{model_prefix}_random_baseline", "random")
        if not os.path.exists(base_dir):
            base_dir = os.path.join(args.prev_results, f"{model_prefix}_random", "random")
        base_results = load_json_results(base_dir)
        if base_results:
            all_data["baseline"] = base_results
            print(f"  Loaded baseline: {len(base_results)} files")

        # LMCache
        lmc_dir = os.path.join(args.result_root, f"{model_prefix}_lmcache", "random")
        lmc_results = load_json_results(lmc_dir)
        if lmc_results:
            all_data["lmcache"] = lmc_results
            print(f"  Loaded LMCache: {len(lmc_results)} files")

        # K-sweep results
        for k in [6, 8, 10]:
            k_dir = os.path.join(args.result_root, f"{model_prefix}_ksweep_k{k}", "random")
            k_results = load_json_results(k_dir)
            if k_results:
                all_data[f"pipelined_inflated_k{k}"] = k_results
                print(f"  Loaded K={k}: {len(k_results)} files")

        # Best K capacity sweep
        best_k_file = os.path.join(args.result_root, f"best_k_{model_prefix}.txt")
        best_k = None
        try:
            with open(best_k_file) as f:
                best_k = f.read().strip()
        except:
            pass

        if best_k:
            cap_dir = os.path.join(args.result_root, f"{model_prefix}_capacity_k{best_k}", "random")
            cap_results = load_json_results(cap_dir)
            if cap_results:
                all_data["pipelined_inflated"] = cap_results
                print(f"  Loaded capacity K={best_k}: {len(cap_results)} files")

        if not all_data:
            print(f"  No data found for {model_name}, skipping graphs")
            continue

        # Determine which concurrency levels have data
        available_conc = set()
        for data in all_data.values():
            for key in data:
                m = re.search(r"conc(\d+)", key)
                if m:
                    available_conc.add(int(m.group(1)))
        conc_sorted = sorted(available_conc)
        print(f"  Concurrency levels with data: {conc_sorted}")

        # 1. ITL vs Concurrency
        plot_itl_vs_concurrency(all_data, conc_sorted, model_name, output_dir)

        # 2. Throughput vs Concurrency
        plot_throughput_vs_concurrency(all_data, conc_sorted, model_name, output_dir)

        # 3. SLO Attainment
        plot_slo_attainment(all_data, conc_sorted, model_name, output_dir, slo_threshold_ms=15.0)

        # 4. Cost Efficiency
        plot_cost_efficiency(all_data, conc_sorted, model_name, output_dir)

        # 5. K-sweep heatmap
        plot_k_sweep_heatmap(
            args.result_root, model_prefix, model_name,
            k_values=[6, 8, 10], concurrency_levels=[64, 128, 256],
            output_dir=output_dir,
        )

        # 6. GPU Utilization timelines
        gpu_csvs = {}
        for k in [6, 8, 10]:
            csv_path = os.path.join(args.result_root, f"{model_prefix}_ksweep_k{k}", "gpu_util.csv")
            if os.path.exists(csv_path):
                gpu_csvs[f"pipelined_inflated_k{k}"] = csv_path
        lmc_csv = os.path.join(args.result_root, f"{model_prefix}_lmcache", "gpu_util.csv")
        if os.path.exists(lmc_csv):
            gpu_csvs["lmcache"] = lmc_csv
        if gpu_csvs:
            plot_gpu_utilization(gpu_csvs, model_name, output_dir)

        # 7. Memory & Capacity comparison
        capacities = {}
        # Baseline
        base_log = os.path.join(args.result_root, f"{model_prefix}_random_baseline", "random", "server_baseline.log")
        if not os.path.exists(base_log):
            base_log = os.path.join(args.prev_results, f"{model_prefix}_random", "random", "server_baseline.log")
        base_cap = parse_server_capacity(base_log) if os.path.exists(base_log) else {}
        if base_cap:
            capacities["baseline"] = {
                "kv_mem_gib": base_cap.get("kv_cache_memory_gib", 0),
                "max_conc": base_cap.get("max_concurrency_reported", 0),
            }

        # Best K inflated
        if best_k:
            infl_log = os.path.join(args.result_root, f"{model_prefix}_capacity_k{best_k}", "random", "server_pipelined_inflated.log")
            if not os.path.exists(infl_log):
                # Try k-sweep dir
                infl_log = os.path.join(args.result_root, f"{model_prefix}_ksweep_k{best_k}", "random", "server_pipelined_inflated.log")
            infl_cap = parse_server_capacity(infl_log) if os.path.exists(infl_log) else {}
            if infl_cap:
                capacities["pipelined_inflated"] = {
                    "kv_mem_gib": infl_cap.get("kv_cache_memory_gib", 0),
                    "max_conc": infl_cap.get("max_concurrency_reported", 0),
                }

        if capacities:
            plot_memory_comparison(capacities, model_name, output_dir)

    # ── Text report ──
    generate_text_report(args.result_root, args.prev_results, output_dir)

    print(f"\n  All graphs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
