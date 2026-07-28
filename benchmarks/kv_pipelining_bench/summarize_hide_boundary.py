#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize R1 hide-boundary sweep vs R0 ITL and hybrid roofline predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from hybrid_kv_model import (  # noqa: E402
    LLAMA8B_R0_ITL_MS,
    b_off_max,
    bytes_per_token,
    itl_at_batch,
    resolve_model,
)

X_EFF_GBS = 25.1
S_CTX = 512
ITL_SLO_MS = 2.0


def _load_r1(cell: Path) -> dict | None:
    p = cell / "results_pipelined.json"
    if not p.is_file():
        return None
    data = json.loads(p.read_text())
    tp = data.get("throughput") or []
    if not tp:
        return None
    row = tp[0]
    itl = row.get("concurrent_mean_itl_steady_s_mean")
    if itl is None:
        itl = row.get("concurrent_mean_itl_s_mean")
    if itl is None:
        return None
    conc = int(row.get("concurrency", 0))
    ok = int(row.get("successes", 0))
    return {"itl_ms": itl * 1000.0, "conc": conc, "ok": ok}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--slo-ms", type=float, default=ITL_SLO_MS)
    ap.add_argument("--x-eff", type=float, default=X_EFF_GBS)
    args = ap.parse_args()
    root = args.root
    _, prof = resolve_model("meta-llama/Llama-3.1-8B-Instruct")
    b_tok = bytes_per_token(prof)

    lines: list[str] = []
    lines.append("R1 HIDE-BOUNDARY SWEEP (short S=512, real PCIe)")
    lines.append("=" * 72)
    lines.append(f"Data root : {root}")
    lines.append(f"X_eff     : {X_EFF_GBS} GB/s | SLO delta: {args.slo_ms} ms vs R0")
    lines.append("")
    lines.append(
        "+------+--------+--------+----------+----------+-----------+---------+"
    )
    lines.append(
        "| conc | R1 ITL | R0 ITL | R1-R0 ms | R1-R0 %  | pred B_off| hidden? |"
    )
    lines.append(
        "+------+--------+--------+----------+----------+-----------+---------+"
    )

    max_hidden = 0
    shape_dir = root / "short"
    if not shape_dir.is_dir():
        print(f"No short/ under {root}", file=sys.stderr)
        sys.exit(1)

    concs = sorted(
        int(d.name.replace("conc", ""))
        for d in shape_dir.iterdir()
        if d.is_dir() and d.name.startswith("conc")
    )
    for c in concs:
        r1 = _load_r1(shape_dir / f"conc{c}" / "R1")
        if r1 is None:
            lines.append(f"| {c:4d} |   FAIL |        |          |          |           |         |")
            continue
        r0 = itl_at_batch(c, LLAMA8B_R0_ITL_MS)
        delta = r1["itl_ms"] - r0
        pct = 100.0 * delta / r0 if r0 else 0.0
        pred = b_off_max(S_CTX, r1["itl_ms"], args.x_eff, b_tok)
        hidden = delta <= args.slo_ms
        if hidden:
            max_hidden = c
        hid = "YES" if hidden else "NO"
        lines.append(
            f"| {c:4d} | {r1['itl_ms']:6.2f} | {r0:6.2f} | {delta:+8.2f} | "
            f"{pct:+8.1f}% | {pred:9d} | {hid:7s} |"
        )
    lines.append(
        "+------+--------+--------+----------+----------+-----------+---------+"
    )
    lines.append("")
    lines.append(f"Max R1 conc within SLO vs R0: {max_hidden}")
    lines.append(f"Roofline B_off_max @ ITL={itl_at_batch(max_hidden, LLAMA8B_R0_ITL_MS):.1f}ms: {b_off_max(S_CTX, itl_at_batch(max_hidden, LLAMA8B_R0_ITL_MS), args.x_eff, b_tok)}")

    text = "\n".join(lines)
    print(text)
    out = root / "HIDE_BOUNDARY.txt"
    out.write_text(text + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
