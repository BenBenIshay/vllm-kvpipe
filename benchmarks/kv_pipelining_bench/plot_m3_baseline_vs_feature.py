#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot M3 baseline vs real_prefill decode ITL at matched concurrencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_p99_ms(path: Path) -> dict[int, float]:
    data = json.loads(path.read_text())
    out: dict[int, float] = {}
    for cell in data.get("throughput") or []:
        K = int(cell["concurrency"])
        p99_s = cell.get("concurrent_global_p99_itl_s")
        mean_s = cell.get("concurrent_mean_itl_s_mean")
        if p99_s is not None:
            out[K] = float(p99_s) * 1000.0
        elif mean_s is not None:
            out[K] = float(mean_s) * 1000.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/data/ben/kv_phase_a_internal/open_loop_eval_v1/m3_baseline_vs_feature"
        ),
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    baseline = _load_p99_ms(args.root / "baseline/results_baseline.json")
    feature = _load_p99_ms(args.root / "real_prefill/results_real_prefill.json")
    ks = sorted(set(baseline) & set(feature))
    if not ks:
        raise SystemExit("No overlapping K values")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = range(len(ks))
    w = 0.35
    b_p99 = [baseline[k] for k in ks]
    f_p99 = [feature[k] for k in ks]
    ax[0].bar([i - w / 2 for i in x], b_p99, width=w, label="baseline (GPU KV)", color="#5cb85c")
    ax[0].bar([i + w / 2 for i in x], f_p99, width=w, label="real_prefill (layer-KV)", color="#1f77b4")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([str(k) for k in ks])
    ax[0].set_xlabel("Concurrency K")
    ax[0].set_ylabel("Global p99 ITL (ms)")
    ax[0].set_title("Decode tail latency")
    ax[0].legend(fontsize=8)
    ax[0].grid(axis="y", alpha=0.3)

    ratio = [feature[k] / max(baseline[k], 1e-6) for k in ks]
    ax[1].bar(x, ratio, color="#d9534f", edgecolor="#333")
    ax[1].axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([str(k) for k in ks])
    ax[1].set_xlabel("Concurrency K")
    ax[1].set_ylabel("Feature / baseline p99 ITL")
    ax[1].set_title("Layer-KV overhead factor")
    ax[1].grid(axis="y", alpha=0.3)

    fig.suptitle("M3: vanilla GPU-KV vs layer-KV pipelined decode (1024/256)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = args.out or (args.root / "m3_baseline_vs_feature_p99.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
