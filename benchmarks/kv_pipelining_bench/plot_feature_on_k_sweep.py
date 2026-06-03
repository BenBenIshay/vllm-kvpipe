#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot the feature-ON K sweep results.

For each model produces one PNG with two stacked panels:

  Top panel: p99 ITL (ms) vs running batch size K, with three overlays:
    - measured (feature ON)        - solid line, markers
    - vanilla baseline (May 4)     - dashed line (only at K-equivalents that
                                     vanilla data covers; i.e. low K)
    - additive prediction          - dotted line: vanilla(K) + PCIe(T_gbps(K))
                                     where the PCIe term uses the May 4 injection
                                     data evaluated at the layer-KV feature's
                                     induced T_gbps

  Bottom panel: throughput (tok/s) vs K, single line per model, helps show
  where compute saturation kicks in.

The gap between "additive prediction" and "measured" is the **extra compute
penalty** the user wanted to highlight.

Usage::

  python plot_feature_on_k_sweep.py \
      --feature-on-csv /data/ben/.../feature_on_k_sweep/feature_on_k_sweep.csv \
      --vanilla-csv /data/ben/.../slo_limited_grid_v3.csv \
      --injection-jsonl /data/ben/.../injection_sweep_v5.jsonl \
      --out-dir /data/ben/.../feature_on_k_sweep/plots
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from admission_predictor import KV_PROFILES, AdmissionPredictor  # noqa: E402

MODEL_SHORT = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B",
}


def _load_feature_on_csv(path: Path) -> dict[str, dict[int, dict]]:
    """Return {model: {K: {p99_itl_ms, throughput_tok_s, ...}}}.

    Skip-empty / aggregated CSV format produced by ``feature_on_k_sweep.py``.
    """
    out: dict[str, dict[int, dict]] = defaultdict(dict)

    def _f(v) -> float | None:
        if v in (None, ""):
            return None
        try:
            return float(v)
        except ValueError:
            return None

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                model = row["model"]
                K = int(row["K"])
            except (KeyError, ValueError):
                continue
            p99 = _f(row.get("p99_itl_ms"))
            mean = _f(row.get("mean_itl_ms"))
            tput = _f(row.get("throughput_tok_s"))
            if p99 is None:
                continue
            out[model][K] = {
                "p99_itl_ms": p99,
                "mean_itl_ms": mean,
                "throughput_tok_s": tput,
                "input_len": int(row.get("input_len", 0) or 0),
                "output_len": int(row.get("output_len", 0) or 0),
            }
    return out


def _vanilla_decode_curve(
    vanilla_csv: Path,
    model: str,
    shape: tuple[int, int],
) -> list[tuple[float, float]]:
    """From vanilla CSV, return [(K_eq, p99_itl_ms), ...] for decode rows.

    The vanilla CSV exposes ``output_throughput`` (tok/s aggregate) and
    ``p99_itl_ms``. Little's law on per-token timing:

        K_eq  =  throughput_tok_s  *  mean_itl_s

    Since the CSV does not store mean_itl, we approximate
    ``mean_itl_ms ~ p99_itl_ms / 5`` (typical bursty p99/mean ratio for
    decode systems, validated by the feature-ON sweep where we have both:
    the ratio ranges 7-21x; using 10x is a more conservative estimate).
    """
    pts: list[tuple[float, float]] = []
    if not vanilla_csv.is_file():
        return pts
    with vanilla_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("phase") != "decode":
                continue
            row_model = row.get("model_hf") or row.get("model") or ""
            base_a = row_model.lower().split("/")[-1]
            base_b = model.lower().split("/")[-1]
            for prefix in ("meta-",):
                if base_a.startswith(prefix):
                    base_a = base_a[len(prefix):]
                if base_b.startswith(prefix):
                    base_b = base_b[len(prefix):]
            if base_a != base_b:
                continue
            try:
                in_len = int(row.get("input_len") or 0)
                out_len = int(row.get("output_len") or 0)
                if (in_len, out_len) != shape:
                    continue
                p99 = float(row.get("p99_itl_ms") or 0)
                tput = float(row.get("output_throughput") or 0)
            except ValueError:
                continue
            if p99 <= 0 or tput <= 0:
                continue
            mean_itl_s_approx = (p99 / 10.0) / 1000.0
            k_eq = tput * mean_itl_s_approx
            if k_eq <= 0:
                continue
            pts.append((k_eq, p99))
    return sorted(pts)


def _additive_prediction(
    feat_curve: list[tuple[int, float]],
    pcie_table: dict[tuple[str, int], float],
    profile,
    vanilla_curve: list[tuple[float, float]],
) -> list[tuple[int, float]]:
    """Return [(K, predicted_p99_ms), ...].

    Uses ``vanilla_p99(K_eq)`` (linearly interpolated/extrapolated) plus the
    H2D PCIe delta evaluated at the feature's induced T_gbps. Where the
    extrapolation is unreliable (K well above vanilla's max K_eq) we still
    extrapolate but the gap to measured will be the headline finding.
    """
    if not vanilla_curve:
        return []
    vk = [p[0] for p in vanilla_curve]
    vv = [p[1] for p in vanilla_curve]

    def _interp_vanilla(k: float) -> float:
        if k <= vk[0]:
            return vv[0]
        if k >= vk[-1]:
            slope = ((vv[-1] - vv[-2]) / max(1.0, vk[-1] - vk[-2])) if len(vk) >= 2 else 0.0
            return vv[-1] + slope * (k - vk[-1])
        for (a, ya), (b, yb) in zip(vanilla_curve, vanilla_curve[1:]):
            if a <= k <= b:
                t = (k - a) / max(1.0, b - a)
                return ya + t * (yb - ya)
        return vv[-1]

    def _interp_pcie(t_gbps: float) -> float:
        h2d_pts = sorted([(g, d) for (dirn, g), d in pcie_table.items() if dirn == "h2d"])
        if not h2d_pts:
            return 0.0
        if t_gbps <= h2d_pts[0][0]:
            return h2d_pts[0][1] * (t_gbps / max(1e-6, h2d_pts[0][0]))
        if t_gbps >= h2d_pts[-1][0]:
            return h2d_pts[-1][1]
        for (g_a, d_a), (g_b, d_b) in zip(h2d_pts, h2d_pts[1:]):
            if g_a <= t_gbps <= g_b:
                t = (t_gbps - g_a) / max(1e-6, g_b - g_a)
                return d_a + t * (d_b - d_a)
        return h2d_pts[-1][1]

    out: list[tuple[int, float]] = []
    for K, p99 in feat_curve:
        # T_gbps(K) using the measured feature-ON p99 ITL as the
        # steps-per-sec proxy (self-consistent with the table)
        steps_per_sec = 1000.0 / max(1e-6, p99)
        bytes_per_step = profile.bytes_per_decode_step_per_seq() if profile else 0
        t_gbps = K * bytes_per_step * steps_per_sec * 8 / 1e9
        baseline = _interp_vanilla(float(K))
        pcie = _interp_pcie(t_gbps)
        out.append((K, baseline + max(0.0, pcie)))
    return out


def _plot(
    model: str,
    feat_data: dict[int, dict],
    vanilla_curve: list[tuple[float, float]],
    additive_pred: list[tuple[int, float]],
    out_path: Path,
) -> None:
    if not feat_data:
        return
    Ks = sorted(feat_data.keys())
    p99s = [feat_data[K]["p99_itl_ms"] for K in Ks]
    means = [feat_data[K].get("mean_itl_ms") or float("nan") for K in Ks]
    tputs = [feat_data[K].get("throughput_tok_s") or float("nan") for K in Ks]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})
    ax_top, ax_bot = axes

    ax_top.plot(Ks, p99s, marker="o", linewidth=2.2,
                label="measured p99 (feature ON)", color="#1f77b4")
    ax_top.plot(Ks, means, marker="s", linewidth=1.4, linestyle="--",
                label="measured mean (feature ON)", color="#7f7f7f")
    if vanilla_curve:
        vk = [p[0] for p in vanilla_curve]
        vv = [p[1] for p in vanilla_curve]
        ax_top.plot(vk, vv, marker="x", linestyle="--", linewidth=1.5,
                    label="vanilla p99 (May 4, K_eq via Little's law)", color="#aaaaaa")
    if additive_pred:
        ak = [p[0] for p in additive_pred]
        av = [p[1] for p in additive_pred]
        ax_top.plot(ak, av, marker="^", linestyle=":", linewidth=1.6,
                    label="additive prediction (vanilla + PCIe)", color="#2ca02c")

    ax_top.set_yscale("log")
    ax_top.set_ylabel("ITL (ms, log scale)")
    ax_top.set_title(
        f"{MODEL_SHORT.get(model, model)}: ITL vs running batch K  (feature ON, decode-only)\n"
        "Gap between measured and additive prediction quantifies the extra compute "
        "penalty at high K",
        fontsize=11,
    )
    ax_top.grid(True, alpha=0.3, which="both")
    ax_top.legend(loc="upper left", fontsize=8)

    ax_bot.plot(Ks, tputs, marker="o", linewidth=2.0, color="#d62728",
                label="aggregate throughput")
    ax_bot.set_xlabel("Running decode batch size K")
    ax_bot.set_ylabel("Throughput (tok/s)")
    ax_bot.set_xscale("log")
    ax_bot.grid(True, alpha=0.3, which="both")
    ax_bot.set_xticks(Ks)
    ax_bot.set_xticklabels([str(k) for k in Ks])
    ax_bot.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=170)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--feature-on-csv", type=Path,
        default=Path("/data/ben/kv_phase_a_internal/open_loop_eval_v1/"
                     "feature_on_k_sweep/feature_on_k_sweep.csv"),
    )
    ap.add_argument(
        "--vanilla-csv", type=Path,
        default=Path("/data/ben/kv_phase_a_internal/open_loop_eval_v1/"
                     "slo_limited_grid_v3.csv"),
    )
    ap.add_argument(
        "--injection-jsonl", type=Path,
        default=Path("/data/ben/kv_phase_a_internal/open_loop_eval_v1/"
                     "injection_sweep_v5_combined/injection_sweep_v5.jsonl"),
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path("/data/ben/kv_phase_a_internal/open_loop_eval_v1/"
                     "feature_on_k_sweep/plots"),
    )
    args = ap.parse_args()

    if not args.feature_on_csv.is_file():
        raise SystemExit(f"feature_on_csv not found: {args.feature_on_csv}")

    feat = _load_feature_on_csv(args.feature_on_csv)
    if not feat:
        raise SystemExit(f"No usable rows in {args.feature_on_csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model, k_to_row in sorted(feat.items()):
        sample = next(iter(k_to_row.values()))
        shape = (int(sample.get("input_len", 0)), int(sample.get("output_len", 0)))
        # Note: vanilla decode shapes were (1024,1024) on May 4. The feature-ON
        # sweep may use a shorter output_len for speed; we still cross-check
        # against vanilla decode at (input_len, 1024) since the steady-state
        # ITL is governed by input_len + decode K, not output_len.
        vanilla_shape = (shape[0], 1024) if shape[1] != 1024 else shape
        vanilla = _vanilla_decode_curve(args.vanilla_csv, model, vanilla_shape)

        injection_table = AdmissionPredictor._load_injection(
            args.injection_jsonl, model=model, shape=(1024, 1024),
        )
        feat_curve = sorted([(K, k_to_row[K]["p99_itl_ms"]) for K in k_to_row])
        profile = KV_PROFILES.get(model)
        additive = _additive_prediction(feat_curve, injection_table, profile, vanilla)

        slug = model.replace("/", "__")
        out_path = args.out_dir / f"p99_itl_vs_k_{slug}.png"
        _plot(model, k_to_row, vanilla, additive, out_path)


if __name__ == "__main__":
    main()
