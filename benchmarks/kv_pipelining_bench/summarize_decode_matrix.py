#!/usr/bin/env python3
"""Summarize R0/R3 decode matrix results into DECODE_ENGINE_MATRIX-style tables."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LONG_CONC = [1, 4, 8, 16, 32, 64]
SHORT_CONC = [16, 64, 256, 512]


def _load_row(cell_dir: Path, arm: str) -> dict | None:
    if arm == "R0":
        p = cell_dir / "results_baseline.json"
    else:
        p = cell_dir / "results_pipelined.json"
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
    conc = int(row.get("concurrency", 0))
    ok = int(row.get("successes", 0))
    fail = int(row.get("failures", 0))
    if itl is None or ok == 0:
        return None
    return {
        "itl_ms": itl * 1000.0,
        "tok_s": conc / itl if itl > 0 else 0.0,
        "ok": ok,
        "fail": fail,
        "conc": conc,
    }


def _fmt_table(shape: str, input_len: int, root: Path, out_lines: list[str]) -> None:
    concs = LONG_CONC if shape == "long" else SHORT_CONC
    out_lines.append(f"\nLlama-3.1-8B / {shape}  (S={input_len})")
    out_lines.append(
        "+------+--------+--------+------------------+----------+----------+-----------+"
    )
    out_lines.append(
        "| conc | R0 ITL | R3 ITL | R3-R0 ITL        | R0 tok/s | R3 tok/s | tok/s     |"
    )
    out_lines.append(
        "|      |  (ms)  |  (ms)  | (ms / %)         | (B/ITL)  | (B/ITL)  | loss      |"
    )
    out_lines.append(
        "+------+--------+--------+------------------+----------+----------+-----------+"
    )
    for c in concs:
        r0 = _load_row(root / shape / f"conc{c}" / "R0", "R0")
        r3 = _load_row(root / shape / f"conc{c}" / "R3", "R3")
        if r0 is None or r3 is None:
            out_lines.append(
                f"| {c:4d} |   --   |   --   | MISSING          |    --    |    --    |    --     |"
            )
            continue
        delta = r3["itl_ms"] - r0["itl_ms"]
        pct = (delta / r0["itl_ms"] * 100.0) if r0["itl_ms"] > 0 else 0.0
        loss = (r3["tok_s"] - r0["tok_s"]) / r0["tok_s"] * 100.0 if r0["tok_s"] else 0.0
        out_lines.append(
            f"| {c:4d} | {r0['itl_ms']:6.2f} | {r3['itl_ms']:6.2f} | "
            f"{delta:+6.2f} ({pct:+5.1f}%) | {r0['tok_s']:8.0f} | {r3['tok_s']:8.0f} | "
            f"{loss:+6.1f}%    |"
        )
    out_lines.append(
        "+------+--------+--------+------------------+----------+----------+-----------+"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()
    root = args.root.resolve()
    lines = [
        "DECODE ENGINE MATRIX -- R0 vs R3 (prefix caching ON both arms)",
        "=" * 72,
        f"Data root : {root}",
        "Protocol  : --prefix-caching-on, --itl-steady-skip 1, --skip-isolated-latency",
        "Arms      : R0 baseline | R3 fake transfer (VLLM_KVPIPE_FAKE_TRANSFER=1)",
    ]
    _fmt_table("long", 8192, root, lines)
    _fmt_table("short", 512, root, lines)
    text = "\n".join(lines) + "\n"
    out_path = args.output or (root / "DECODE_ENGINE_MATRIX.txt")
    out_path.write_text(text)
    print(text)
    if args.output is None:
        print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
