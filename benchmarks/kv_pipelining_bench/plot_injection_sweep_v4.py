#!/usr/bin/env python3
"""SLO-limited effective tok/s for injection_sweep_v4."""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("KV_SWEEP_OUT", str(THIS / "injection_sweep_v4")))
RESULT = ROOT / "injection_sweep_results.jsonl"
SLO_CSV = Path(
    os.environ.get(
        "KV_SLO_CSV",
        "/data/ben/kv_phase_a_internal/open_loop_eval_v1/length_dependent_slo_definitions_v2.csv",
    )
)
HF_TO_TAB = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B-Instruct",
}


def load_slo_map():
    m = {}
    with SLO_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            m[(r["model"].strip(), r["phase"].strip(), r["shape"].strip())] = float(
                r["slo_ms"]
            )
    return m


def row_shape(r):
    il, ol = int(r["input_len"]), int(r["output_len"])
    if r.get("phase") == "prefill" or ol <= 1:
        return str(il)
    return f"{il}/{ol}"


def slo_limited(slo_by_shape, r):
    p = (r.get("phase") or "").strip() or (
        "prefill" if int(r.get("output_len", 0)) <= 1 else "decode"
    )
    mname = HF_TO_TAB.get(r["model"])
    if mname is None:
        slo = float(r["slo_ms"])
    else:
        slo = slo_by_shape.get((mname, p, row_shape(r)), float(r["slo_ms"]))
    kind = (r.get("slo_kind") or ("itl" if p == "decode" else "ttft")).strip()
    lat = r.get("p99_itl_ms") if kind == "itl" else r.get("p99_ttft_ms")
    if lat is None:
        return 0.0
    raw = float(r.get("output_throughput") or 0.0)
    latf = float(lat)
    if latf <= 0:
        return 0.0
    return raw * min(1.0, slo / latf)


def _h2d_x(tag: str) -> float:
    m = re.match(r"^h2d_(\d+)_c(\d+)$", tag)
    return float(m.group(1)) if m else 0.0


def _d2h_x(tag: str) -> float:
    m = re.match(r"^d2h_(\d+)_c(\d+)$", tag)
    return float(m.group(1)) if m else 0.0


def _hbm_x(tag: str) -> int:
    m = re.match(r"^hbm(\d+)$", tag)
    return int(m.group(1)) if m else -1


def main():
    if not RESULT.exists():
        raise SystemExit(f"missing {RESULT}")

    rows = [json.loads(x) for x in RESULT.read_text().splitlines() if x.strip()]
    slo_by = load_slo_map()
    for r in rows:
        r["slo_limited_tok_s"] = slo_limited(slo_by, r)

    by = {}
    for r in rows:
        by.setdefault(r["workload"], []).append(r)

    n = len(by)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.8 * max(1, n)))
    if n == 1:
        axes = [axes]
    for i, (wk, rs) in enumerate(sorted(by.items())):
        ax0, ax1 = axes[i]
        h2d = sorted([r for r in rs if r.get("injection_mode") == "h2d_async"], key=lambda r: _h2d_x(str(r.get("injection_scenario", ""))))
        d2h = sorted([r for r in rs if r.get("injection_mode") == "d2h_async"], key=lambda r: _d2h_x(str(r.get("injection_scenario", ""))))
        hbm = sorted([r for r in rs if r.get("injection_mode") == "hbm_parallel"], key=lambda r: _hbm_x(str(r.get("injection_scenario", ""))))

        if h2d:
            ax0.plot([_h2d_x(str(r["injection_scenario"])) for r in h2d], [r["slo_limited_tok_s"] for r in h2d], "o-", color="tab:blue", label="H2D async")
        if d2h:
            ax0.plot([_d2h_x(str(r["injection_scenario"])) for r in d2h], [r["slo_limited_tok_s"] for r in d2h], "s--", color="tab:orange", label="D2H async")
        ax0.set_title(f"{wk} PCIe")
        ax0.set_xlabel("Target GB/s")
        ax0.set_ylabel("SLO-lim tok/s")
        ax0.grid(alpha=0.3)
        ax0.legend(fontsize=7, loc="best")

        if hbm:
            ax1.plot([_hbm_x(str(r["injection_scenario"])) for r in hbm], [r["slo_limited_tok_s"] for r in hbm], "o-", color="tab:green", label="HBM streams")
        mix = [r for r in rs if r.get("injection_mode") == "h2d_d2h_mix"]
        for r in mix:
            ax1.scatter([0.5], [r["slo_limited_tok_s"]], marker="x", color="red", s=50)
        ax1.set_title(f"{wk} on-GPU + mix")
        ax1.set_xlabel("HBM stream count (mix as red x)")
        ax1.set_ylabel("SLO-lim tok/s")
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=7, loc="best")

    fig.suptitle("SLO-limited = raw * min(1, SLO/p99)", y=1.01, fontsize=10)
    fig.tight_layout()
    out = ROOT / "injection_slo_limited_v4.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)

    csvp = ROOT / "injection_slo_limited_v4.csv"
    keys = sorted({k for r in rows for k in r.keys()})
    with csvp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(out, csvp)


if __name__ == "__main__":
    main()
