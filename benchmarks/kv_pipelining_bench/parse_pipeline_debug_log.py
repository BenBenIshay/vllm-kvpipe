#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize PrefillStaging connector debug lines from a server log."""
from __future__ import annotations

import argparse
import ast
import re
from collections import Counter
from pathlib import Path


def _extract_dict(line: str) -> dict | None:
    m = re.search(r"\{.*\}", line)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(0))
    except (SyntaxError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--max-forwards", type=int, default=5,
                    help="Only summarize first N forward ids with decode save lines.")
    args = ap.parse_args()
    text = args.log.read_text(errors="replace")
    lines = text.splitlines()

    save_modes = Counter()
    prefetch_layers = 0
    forwards_seen: set[int] = set()
    block_runs_samples: list[tuple] = []
    fallback_full = 0

    fwd_re = re.compile(r"fwd=(\d+)")
    for line in lines:
        if "[pipeline-debug]" not in line:
            continue
        m = fwd_re.search(line)
        fwd = int(m.group(1)) if m else -1
        if "save_kv decode BLOCK_STAGE_MODE" in line:
            save_modes["BLOCK_STAGE"] += 1
            if fwd >= 0:
                forwards_seen.add(fwd)
        elif "save_kv decode DELTA_STAGE_MODE" in line:
            save_modes["DELTA_STAGE"] += 1
        elif "save_kv decode FALLBACK_FULL" in line or "FULL_STAGE" in line:
            save_modes["FULL_OR_FALLBACK"] += 1
            fallback_full += 1
        elif "prefetch_batch" in line or "prefetch_chain" in line:
            prefetch_layers += 1
            if "kv_block_runs" in line and len(block_runs_samples) < 20:
                d = _extract_dict(line)
                if d and "kv_block_runs" in d:
                    block_runs_samples.append((fwd, d.get("kv_block_runs")))

    print(f"Log: {args.log}")
    print(f"Decode save modes: {dict(save_modes)}")
    print(f"Prefetch debug lines: {prefetch_layers}")
    print(f"Forwards with block-stage saves: {len(forwards_seen)}")
    if fallback_full:
        print("WARNING: FULL/FALLBACK staging seen — expect high PCIe and bad ITL.")
    if block_runs_samples:
        print("\nSample kv_block_runs (forward, runs):")
        for fwd, runs in block_runs_samples[: args.max_forwards]:
            span = sum(hi - lo + 1 for lo, hi in runs) if runs else 0
            print(f"  fwd={fwd}: runs={runs!r}  total_blocks≈{span}")


if __name__ == "__main__":
    main()
