#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Find a good K (kv-k-window) for pipelined dummy-KV decode
==========================================================

Runs :mod:`pipelined_decode_fast_bench` once per K value (pipelined-only),
starts from a reference K (default **6**), and picks the K that maximizes
decode throughput at a given concurrency while keeping **p99 ITL** within a
relative tolerance of the reference K (proxy for “same TBT”).

This does **not** compare baseline vs pipelined — use when the GPU is shared
and only relative tuning across K is meaningful.

Example (full K list, default concurrencies from fast bench)::

    python benchmarks/kv_pipelining_bench/find_optimal_k.py \\
        --model /data/ben/models/Meta-Llama-3.1-8B-Instruct \\
        --k-values 6 5 4 3 2 \\
        --measure-concurrency 32 \\
        --p99-itl-relative-tol 0.05 \\
        --result-dir /tmp/kv_k_sweep

**Limited GPU / time** — use ``--lite`` (fixed descending K sweep 6→4→2, lower
concurrency, shorter 256/64 token workload; same dummy-KV decode-only path as
``pipelined_decode_fast_bench.py``)::

    python benchmarks/kv_pipelining_bench/find_optimal_k.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --lite \\
        --result-dir /tmp/kv_k_sweep_lite

Each K is written under ``result_dir/k_<K>/`` unless ``--flat-result-dir`` is set.
Requires a free port and enough GPU memory for the model.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from pipelined_decode_fast_bench import (  # noqa: E402
    apply_auto_gpu_memory,
    build_parser,
    run_benchmark,
)


def _throughput_at_conc(summary: dict[str, Any], conc: int) -> float | None:
    for row in summary.get("throughput", []):
        if row.get("concurrency") == conc and row.get("failures", 1) == 0:
            return float(row.get("throughput_tok_per_s", 0.0))
    return None


def _p99_itl_ms(summary: dict[str, Any]) -> float | None:
    lat = summary.get("latency") or {}
    if lat.get("error"):
        return None
    v = lat.get("p99_itl_s_mean")
    if v is None:
        return None
    return float(v) * 1000.0


def _mean_itl_ms(summary: dict[str, Any]) -> float | None:
    lat = summary.get("latency") or {}
    if lat.get("error"):
        return None
    v = lat.get("mean_itl_s_mean")
    if v is None:
        return None
    return float(v) * 1000.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep K and recommend best under latency tolerance",
        parents=[build_parser()],
        conflict_handler="resolve",
    )
    # Remove duplicate --model from parent: parent required=False; we add logic
    p.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[6, 5, 4, 3, 2],
        help="K values to try (kv-k-window), high to low by default",
    )
    p.add_argument(
        "--measure-concurrency",
        type=int,
        default=32,
        help="Concurrency used to read throughput / compare latency",
    )
    p.add_argument(
        "--p99-itl-relative-tol",
        type=float,
        default=0.05,
        help="Allow p99 ITL up to (1+tol) * ref p99 (ref = first K in list)",
    )
    p.add_argument(
        "--require-all-k-success",
        action="store_true",
        help="If set, skip K when throughput fails at measure-concurrency",
    )
    p.add_argument(
        "--flat-result-dir",
        action="store_true",
        help="Write all K into the same result-dir (overwrites; default is k_<K>/ subdirs)",
    )
    p.add_argument(
        "--lite",
        action="store_true",
        help=(
            "Resource-constrained preset: K sweep 6 4 2 (high→low), "
            "concurrency 4 8 16, measure-concurrency 16, input/output 256/64. "
            "Do not combine with explicit --k-values/--concurrency you care about—"
            "this flag replaces those for the sweep."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.lite:
        args.k_values = [6, 4, 2]
        args.concurrency = [4, 8, 16]
        args.measure_concurrency = 16
        args.input_len = 256
        args.output_len = 64
    if not args.model:
        print("error: --model is required", file=sys.stderr)
        sys.exit(2)

    result_base = Path(args.result_dir)
    result_base.mkdir(parents=True, exist_ok=True)

    if args.auto_gpu_memory:
        apply_auto_gpu_memory(args)

    k_list = list(args.k_values)
    ref_k = k_list[0]

    rows: list[dict[str, Any]] = []
    ref_p99: float | None = None

    for k in k_list:
        subdir = result_base if args.flat_result_dir else result_base / f"k_{k}"
        ns = deepcopy(args)
        ns.k_window = k
        ns.modes = ["pipelined"]
        ns.with_baseline = False
        ns.result_dir = subdir
        # Only run one concurrency if user passed many; we only need measure
        # point + maybe warmup range — keep full list for max_successful stats
        if args.measure_concurrency not in args.concurrency:
            ns.concurrency = sorted(set(list(args.concurrency) + [args.measure_concurrency]))

        print(f"\n{'#' * 70}\n# K = {k}\n{'#' * 70}")
        all_out = run_benchmark(ns)
        pip = all_out.get("pipelined", {})
        if pip.get("error"):
            rows.append({"k": k, "error": pip.get("error")})
            continue
        if pip.get("aborted"):
            rows.append(
                {
                    "k": k,
                    "error": "gpu_memory_abort",
                    "abort_phase": pip.get("abort_phase"),
                    "abort_reason": pip.get("abort_reason"),
                }
            )
            print(
                "\n[K-sweep] GPU memory abort — stopping K sweep.",
                file=sys.stderr,
                flush=True,
            )
            break

        p99_ms = _p99_itl_ms(pip)
        mean_ms = _mean_itl_ms(pip)
        tput = _throughput_at_conc(pip, args.measure_concurrency)
        if k == ref_k and p99_ms is not None:
            ref_p99 = p99_ms

        rows.append(
            {
                "k": k,
                "p99_itl_ms": p99_ms,
                "mean_itl_ms": mean_ms,
                "throughput_tok_s_at_conc": tput,
                "measure_concurrency": args.measure_concurrency,
                "max_successful_concurrency": pip.get("max_successful_concurrency"),
            }
        )

    # Pick best K
    if ref_p99 is None:
        print("\nCould not establish reference p99 ITL; check failures.")
        sys.exit(1)

    tol = args.p99_itl_relative_tol
    best_k: int | None = None
    best_tput = -1.0

    for row in rows:
        if row.get("error"):
            continue
        p99 = row.get("p99_itl_ms")
        tput = row.get("throughput_tok_s_at_conc")
        if p99 is None or tput is None:
            if args.require_all_k_success:
                continue
            continue
        if p99 > ref_p99 * (1.0 + tol):
            continue
        if tput > best_tput:
            best_tput = tput
            best_k = row["k"]

    report = {
        "reference_k": ref_k,
        "reference_p99_itl_ms": ref_p99,
        "relative_tolerance": tol,
        "measure_concurrency": args.measure_concurrency,
        "rows": rows,
        "recommended_k": best_k,
        "recommended_throughput_tok_s": best_tput if best_k is not None else None,
    }

    out_json = result_base / "optimal_k_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("K sweep summary")
    print("=" * 70)
    print(f"  Reference K     : {ref_k}  (p99 ITL ≈ {ref_p99:.3f} ms)")
    print(f"  Tolerance       : {tol*100:.1f}% on p99 ITL")
    print(f"  At concurrency  : {args.measure_concurrency}")
    print()
    for row in rows:
        if row.get("error"):
            print(f"  K={row['k']}: ERROR {row['error']}")
            continue
        print(
            f"  K={row['k']}: p99_itl={row['p99_itl_ms']:.3f} ms  "
            f"tput={row.get('throughput_tok_s_at_conc')} tok/s  "
            f"max_ok_conc={row.get('max_successful_concurrency')}"
        )
    print()
    if best_k is not None:
        print(f"  Recommended K   : {best_k}  (best throughput within p99 ITL bound)")
    else:
        print("  Recommended K   : none (all K violated tolerance or failed)")
    print(f"\n  Wrote {out_json}")


if __name__ == "__main__":
    main()
