#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot open-loop sweep JSONL from open_loop_rate_sweep.py.

For each model subdir under --root, reads prefill_rate_sweep.jsonl and
decode_rate_sweep.jsonl and writes PNGs to ``plots/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def plot_lines(
    rows: list[dict],
    *,
    xkey: str,
    ykey: str,
    series_key: str,
    title: str,
    ylabel: str,
    out: Path,
) -> None:
    by_series: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        if r.get("exit_code", -1) != 0:
            continue
        if r.get(ykey) is None:
            continue
        sk = str(r.get(series_key, "?"))
        by_series.setdefault(sk, []).append((float(r[xkey]), float(r[ykey])))

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in sorted(by_series):
        pts = sorted(by_series[name])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", linewidth=2, label=name)

    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title=series_key, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/data/ben/kv_phase_a_internal/open_loop_eval_v1"),
    )
    args = ap.parse_args()

    for sub in sorted(args.root.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.startswith(".") or sub.name == "microbench_cpu_hbm":
            continue
        pref = load_jsonl(sub / "prefill_rate_sweep.jsonl")
        dec = load_jsonl(sub / "decode_rate_sweep.jsonl")
        if pref:
            # annotate series as input_len
            for r in pref:
                r["_series"] = f"input={r.get('input_len')}"
            plot_lines(
                pref,
                xkey="request_rate",
                ykey="p99_ttft_ms",
                series_key="_series",
                title=f"{sub.name}: prefill p99 TTFT vs rate",
                ylabel="p99 TTFT (ms)",
                out=sub / "plots" / "prefill_p99_ttft_vs_rate.png",
            )
            plot_lines(
                pref,
                xkey="request_rate",
                ykey="output_throughput",
                series_key="_series",
                title=f"{sub.name}: prefill output tok/s vs rate",
                ylabel="Output tok/s",
                out=sub / "plots" / "prefill_output_tok_s_vs_rate.png",
            )
        if dec:
            for r in dec:
                r["_series"] = f"i{r.get('input_len')}_o{r.get('output_len')}"
            plot_lines(
                dec,
                xkey="request_rate",
                ykey="p99_itl_ms",
                series_key="_series",
                title=f"{sub.name}: decode p99 ITL vs rate",
                ylabel="p99 ITL (ms)",
                out=sub / "plots" / "decode_p99_itl_vs_rate.png",
            )
            plot_lines(
                dec,
                xkey="request_rate",
                ykey="output_throughput",
                series_key="_series",
                title=f"{sub.name}: decode output tok/s vs rate",
                ylabel="Output tok/s",
                out=sub / "plots" / "decode_output_tok_s_vs_rate.png",
            )
        print(f"Processed {sub}")


if __name__ == "__main__":
    main()
