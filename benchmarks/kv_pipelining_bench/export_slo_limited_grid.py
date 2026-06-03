#!/usr/bin/env python3
"""Build slo_limited_grid_v3.csv and regenerate SLO-limited tok/s plots.

Reads {model_slug}/{prefill,decode}_rate_sweep.jsonl from --root,
joins against length_dependent_slo_definitions_v2.csv, and emits:

  slo_limited_grid_v3.csv            — all rows with slo_limited_tok_s
  length_dependent_slo_definitions_v2.csv  — regenerated (saturated rows only)
  length_dependent_slo_tok_s_combined_v3.png

The SLO ms definition: for each (model, phase, shape) take
  1.25 × p99_ms at the lowest request rate for which a saturated row exists
  (the reference-rate anchor).

SLO-limited tok/s (continuous): output_throughput * min(1.0, slo_ms / p99_ms).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DEFAULT = Path("/data/ben/kv_phase_a_internal/open_loop_eval_v1")
MODELS = [
    ("meta-llama__Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct",
     "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("Qwen__Qwen2.5-14B-Instruct", "Qwen2.5-14B-Instruct",
     "Qwen/Qwen2.5-14B-Instruct"),
]


def read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _shape(phase: str, il: int, ol: int) -> str:
    return str(il) if (phase == "prefill" or ol <= 1) else f"{il}/{ol}"


def build_slo_definitions(
    all_rows: list[dict], factor: float = 1.25
) -> dict[tuple[str, str, str], float]:
    """Return {(model_tab, phase, shape): slo_ms} from saturated rows only."""
    # Collect per-(model, phase, shape) the minimum rate that has a valid
    # saturated row and its p99.
    anchor: dict[tuple[str, str, str], tuple[float, float]] = {}  # key → (rate, p99)
    for r in all_rows:
        if r.get("exit_code", -1) != 0:
            continue
        # Accept saturated rows; also accept rows where key is absent (legacy data)
        if r.get("saturated") is False:
            continue
        phase = (r.get("phase") or "").replace("_open_loop", "")
        if phase not in ("prefill", "decode"):
            continue
        il = int(r.get("input_len", 0) or 0)
        ol = int(r.get("output_len", 0) or 0)
        rate = float(r.get("request_rate", 0) or 0)
        p99 = r.get("p99_ttft_ms") if phase == "prefill" else r.get("p99_itl_ms")
        if p99 is None:
            continue
        mname = r.get("model_tab") or r.get("model_hf", "")
        sh = _shape(phase, il, ol)
        key = (mname, phase, sh)
        if key not in anchor or rate < anchor[key][0]:
            anchor[key] = (rate, float(p99))
    return {k: round(v[1] * factor, 4) for k, v in anchor.items()}


def slo_limited(row: dict, slo_ms: float) -> float | None:
    phase = (row.get("phase") or "").replace("_open_loop", "")
    p99 = row.get("p99_ttft_ms") if phase == "prefill" else row.get("p99_itl_ms")
    raw = row.get("output_throughput")
    if p99 is None or raw is None:
        return None
    p99f = float(p99)
    if p99f <= 0:
        return None
    return float(raw) * min(1.0, slo_ms / p99f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    ap.add_argument("--slo-factor", type=float, default=1.25,
                    help="Multiplier applied to reference p99 to get SLO ms.")
    args = ap.parse_args()
    root = args.root

    # ── Collect all rows from both models ────────────────────────────────────
    all_rows: list[dict] = []
    for slug, tab, hf_id in MODELS:
        mdir = root / slug
        for phase, jsonl in [("prefill", "prefill_rate_sweep.jsonl"),
                              ("decode",  "decode_rate_sweep.jsonl")]:
            for r in read_jsonl(mdir / jsonl):
                r["model_tab"] = tab
                r["model_hf"] = hf_id
                all_rows.append(r)

    # ── Build SLO definitions ────────────────────────────────────────────────
    slo_map = build_slo_definitions(all_rows, factor=args.slo_factor)

    # Write updated SLO CSV
    slo_csv_path = root / "length_dependent_slo_definitions_v2.csv"
    slo_rows = []
    for (mname, phase, shape), slo_ms in sorted(slo_map.items()):
        slo_rows.append({
            "model": mname, "phase": phase, "shape": shape,
            "slo_ms": slo_ms,
        })
    with slo_csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "phase", "shape", "slo_ms"])
        w.writeheader()
        w.writerows(slo_rows)
    print(f"Wrote {slo_csv_path}  ({len(slo_rows)} shapes)")

    # ── Compute slo_limited_tok_s for every row ──────────────────────────────
    grid_rows = []
    for r in all_rows:
        if r.get("exit_code", -1) != 0:
            continue
        phase = (r.get("phase") or "").replace("_open_loop", "")
        if phase not in ("prefill", "decode"):
            continue
        il = int(r.get("input_len", 0) or 0)
        ol = int(r.get("output_len", 0) or 0)
        sh = _shape(phase, il, ol)
        mname = r.get("model_tab", "")
        slo_ms = slo_map.get((mname, phase, sh))
        slt = slo_limited(r, slo_ms) if slo_ms else None
        grid_rows.append({
            "model": mname,
            "model_hf": r.get("model_hf", ""),
            "phase": phase,
            "input_len": il,
            "output_len": ol,
            "shape": sh,
            "request_rate": r.get("request_rate"),
            "output_throughput": r.get("output_throughput"),
            "p99_ttft_ms": r.get("p99_ttft_ms"),
            "p99_itl_ms": r.get("p99_itl_ms"),
            "slo_ms": slo_ms,
            "slo_limited_tok_s": round(slt, 4) if slt is not None else None,
            "saturated": r.get("saturated"),
            "tel_sm_mean": r.get("tel_sm_mean"),
            "tel_sm_p95": r.get("tel_sm_p95"),
            "tel_pwr_p95_w": r.get("tel_pwr_p95_w"),
        })

    grid_csv_path = root / "slo_limited_grid_v3.csv"
    if grid_rows:
        with grid_csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
            w.writeheader()
            w.writerows(grid_rows)
    print(f"Wrote {grid_csv_path}  ({len(grid_rows)} rows)")

    # ── Per-model per-phase plots ─────────────────────────────────────────────
    for slug, tab, _ in MODELS:
        mdir = root / slug
        out = mdir / "plots_slo_per_length_v3"
        out.mkdir(exist_ok=True)

        mrows = [r for r in grid_rows if r["model"] == tab]
        for phase, lat_key in [("prefill", "p99_ttft_ms"), ("decode", "p99_itl_ms")]:
            phase_rows = [r for r in mrows if r["phase"] == phase]
            shapes = sorted({r["shape"] for r in phase_rows})

            # Raw tok/s
            fig, ax = plt.subplots(figsize=(9, 5))
            for sh in shapes:
                pts = sorted(
                    [(float(r["request_rate"]), float(r["output_throughput"]))
                     for r in phase_rows if r["shape"] == sh
                     and r.get("output_throughput") is not None],
                    key=lambda t: t[0],
                )
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            marker="o", linewidth=2, label=sh)
            ax.set_title(f"{tab}: {phase} raw tok/s vs rate")
            ax.set_xlabel("Request rate (req/s)")
            ax.set_ylabel("Output tok/s")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / f"{phase}_raw_tok_s_vs_rate.png", dpi=170)
            plt.close(fig)

            # p99 latency
            fig, ax = plt.subplots(figsize=(9, 5))
            for sh in shapes:
                pts = sorted(
                    [(float(r["request_rate"]), float(r[lat_key]))
                     for r in phase_rows if r["shape"] == sh
                     and r.get(lat_key) is not None],
                    key=lambda t: t[0],
                )
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            marker="o", linewidth=2, label=sh)
            lat_label = "p99 TTFT (ms)" if phase == "prefill" else "p99 ITL (ms)"
            ax.set_title(f"{tab}: {phase} {lat_label} vs rate")
            ax.set_xlabel("Request rate (req/s)")
            ax.set_ylabel(lat_label)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / f"{phase}_p99_lat_vs_rate.png", dpi=170)
            plt.close(fig)

            # SLO-limited tok/s (continuous)
            fig, ax = plt.subplots(figsize=(9, 5))
            for sh in shapes:
                pts = sorted(
                    [(float(r["request_rate"]), float(r["slo_limited_tok_s"]))
                     for r in phase_rows if r["shape"] == sh
                     and r.get("slo_limited_tok_s") is not None],
                    key=lambda t: t[0],
                )
                if pts:
                    slo_val = next(
                        (r["slo_ms"] for r in phase_rows if r["shape"] == sh
                         and r.get("slo_ms") is not None), None)
                    lbl = f"{sh} (SLO={slo_val:.0f}ms)" if slo_val else sh
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            marker="o", linewidth=2, label=lbl)
            ax.set_title(f"{tab}: {phase} SLO-limited tok/s vs rate")
            ax.set_xlabel("Request rate (req/s)")
            ax.set_ylabel("SLO-limited output tok/s")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / f"{phase}_slo_limited_tok_s_vs_rate.png", dpi=170)
            plt.close(fig)

    # ── Combined 2×4 panel (v3) ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for ridx, (slug, tab, _) in enumerate(MODELS):
        p = root / slug / "plots_slo_per_length_v3"
        panels = [
            ("Prefill raw tok/s",        "prefill_raw_tok_s_vs_rate.png"),
            ("Prefill p99 TTFT",         "prefill_p99_lat_vs_rate.png"),
            ("Decode raw tok/s",         "decode_raw_tok_s_vs_rate.png"),
            ("Decode p99 ITL",           "decode_p99_lat_vs_rate.png"),
        ]
        for cidx, (title, fname) in enumerate(panels):
            ip = p / fname
            if ip.is_file():
                axes[ridx, cidx].imshow(plt.imread(str(ip)))
            else:
                axes[ridx, cidx].text(0.5, 0.5, "no data",
                                      ha="center", va="center", transform=axes[ridx, cidx].transAxes)
            axes[ridx, cidx].set_title(f"{tab}\n{title}", fontsize=9)
            axes[ridx, cidx].axis("off")

    fig.suptitle("Raw tok/s + p99 latency (v3, H100 saturation gated)", fontsize=12, y=1.01)
    fig.tight_layout()
    comb = root / "raw_and_extended_latency_combined_v3.png"
    fig.savefig(str(comb), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {comb}")

    # SLO combined
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ridx, (slug, tab, _) in enumerate(MODELS):
        p = root / slug / "plots_slo_per_length_v3"
        for cidx, (ph, fname) in enumerate([
            ("Prefill", "prefill_slo_limited_tok_s_vs_rate.png"),
            ("Decode",  "decode_slo_limited_tok_s_vs_rate.png"),
        ]):
            ip = p / fname
            if ip.is_file():
                axes[ridx, cidx].imshow(plt.imread(str(ip)))
            else:
                axes[ridx, cidx].text(0.5, 0.5, "no data",
                                      ha="center", va="center", transform=axes[ridx, cidx].transAxes)
            axes[ridx, cidx].set_title(f"{tab} — {ph}", fontsize=11)
            axes[ridx, cidx].axis("off")

    fig.suptitle("SLO-limited tok/s (continuous, length-dependent SLO, H100-saturated cells only)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    comb_slo = root / "length_dependent_slo_tok_s_combined_v3.png"
    fig.savefig(str(comb_slo), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {comb_slo}")


if __name__ == "__main__":
    main()
