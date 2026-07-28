#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare hide-boundary measurements to hybrid roofline predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add bench dir for hybrid_kv_model imports.
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from hybrid_kv_model import (  # noqa: E402
    LLAMA8B_R0_ITL_MS,
    b_off_max,
    bytes_per_token,
    hide_budget_tok_slots,
    itl_at_batch,
    resolve_model,
)

X_EFF = 25.1
S_CTX = 512
ITL_SLO_MS = 2.0


def _load_r1(root: Path, conc: int) -> float | None:
    p = root / "short" / f"conc{conc}" / "R1" / "results_pipelined.json"
    if not p.is_file():
        return None
    row = json.loads(p.read_text()).get("throughput", [{}])[0]
    itl = row.get("concurrent_mean_itl_steady_s_mean") or row.get(
        "concurrent_mean_itl_s_mean"
    )
    return float(itl) * 1000.0 if itl else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "hide_root",
        type=Path,
        default=Path("/mnt/beegfs/benishay/paper_runs/20260705/hybrid_kv/hide_boundary"),
        nargs="?",
    )
    ap.add_argument("--x-eff", type=float, default=X_EFF)
    ap.add_argument("--slo-ms", type=float, default=ITL_SLO_MS)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/mnt/beegfs/benishay/paper_runs/20260705/hybrid_kv/THEORY_VS_MEASURED.md"),
    )
    args = ap.parse_args()

    _, prof = resolve_model("meta-llama/Llama-3.1-8B-Instruct")
    b_tok = bytes_per_token(prof)

    lines = [
        "# Theory vs Measured — Hybrid KV Roofline (Llama-8B, A100)",
        "",
        f"**X_eff:** {args.x_eff} GB/s (pinned H2D microbench)",
        f"**S:** {S_CTX} tokens | **SLO:** R1−R0 ≤ {args.slo_ms} ms",
        "",
        "## Hide-boundary (all-R1 pipelined, real PCIe)",
        "",
        "| conc | R1 ITL | R0 ITL | Δ ms | pred B_off | hidden? |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]

    max_hidden = 0
    shape = args.hide_root / "short"
    concs = sorted(
        int(d.name.replace("conc", ""))
        for d in shape.iterdir()
        if d.is_dir() and d.name.startswith("conc")
    ) if shape.is_dir() else []

    for c in concs:
        r1 = _load_r1(args.hide_root, c)
        r0 = itl_at_batch(c, LLAMA8B_R0_ITL_MS)
        if r1 is None:
            lines.append(f"| {c} | — | {r0:.2f} | — | — | — |")
            continue
        delta = r1 - r0
        pred = b_off_max(S_CTX, r1, args.x_eff, b_tok)
        hidden = delta <= args.slo_ms
        if hidden:
            max_hidden = c
        lines.append(
            f"| {c} | {r1:.2f} | {r0:.2f} | {delta:+.2f} | {pred} | "
            f"{'yes' if hidden else 'no'} |"
        )

    lines.extend(
        [
            "",
            f"**Measured max hidden conc:** {max_hidden}",
            "",
            "## Hybrid tier prediction @ B_hbm=49",
            "",
        ]
    )

    b_hbm = 49
    itl_hbm = itl_at_batch(b_hbm, LLAMA8B_R0_ITL_MS)
    theta = hide_budget_tok_slots(itl_hbm, args.x_eff, b_tok)
    off_pred = b_off_max(S_CTX, itl_hbm, args.x_eff, b_tok)
    lines.append(f"- Θ @ B_hbm={b_hbm}: {theta:.0f} tok-slots")
    lines.append(f"- Predicted B_off_max: **{off_pred}**")
    lines.append(f"- Measured max all-R1 hidden: **{max_hidden}**")
    if max_hidden > 0:
        err = abs(max_hidden - off_pred) / max(off_pred, 1) * 100
        lines.append(f"- Relative error vs B_off at B_hbm: {err:.0f}%")
    lines.extend(
        [
            "",
            "## Composite throughput bound (hybrid target)",
            "",
            "If HBM tier runs at R0 ITL and CPU tier at R1 ITL (hidden):",
            "",
            "```",
            f"tok/s_hybrid ≈ B_hbm/ITL_r0(B_hbm) + B_off/ITL_r1(B_off)",
            "```",
            "",
        ]
    )

    if max_hidden > 0:
        r1_off = _load_r1(args.hide_root, max_hidden) or itl_at_batch(
            max_hidden, LLAMA8B_R0_ITL_MS
        )
        tput_hbm = b_hbm / (itl_hbm / 1000.0)
        tput_off = off_pred / (r1_off / 1000.0) if off_pred > 0 else 0
        tput_hybrid = tput_hbm + tput_off
        tput_r1_total = (b_hbm + off_pred) / (r1_off / 1000.0) if off_pred else 0
        lines.append(f"| Mode | Batch | tok/s (est.) |")
        lines.append(f"|---|---:|---:|")
        lines.append(f"| R0 only | {b_hbm} | {tput_hbm:.0f} |")
        lines.append(
            f"| Hybrid (target) | {b_hbm}+{off_pred} | **{tput_hybrid:.0f}** |"
        )
        lines.append(
            f"| All-R1 @ {b_hbm+off_pred} | {b_hbm+off_pred} | "
            f"{tput_r1_total:.0f} (if ran) |"
        )

    lines.extend(
        [
            "",
            "## Status",
            "",
            "- Mixed-batch dual KV path in gpu_model_runner: **pending**",
            "- Hybrid admission + derived CpuKvTable cap: **implemented**",
            "",
        ]
    )

    text = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
    print(text)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
