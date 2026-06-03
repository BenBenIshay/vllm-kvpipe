#!/usr/bin/env python3
"""Regenerate results_table.md / .csv with latency columns in milliseconds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def format_table(headers: list[str], rows_text: list[list[str]]) -> str:
    all_rows = [headers] + rows_text
    widths = [max(len(all_rows[i][c]) for i in range(len(all_rows))) for c in range(len(headers))]

    def pad(c: int, text: str) -> str:
        w = widths[c]
        return text.ljust(w) if c == 0 else text.rjust(w)

    lines: list[str] = []
    lines.append("| " + " | ".join(pad(i, headers[i]) for i in range(len(headers))) + " |")
    lines.append("|" + "|".join("-" * (widths[i] + 2) for i in range(len(headers))) + "|")
    for tr in rows_text:
        lines.append("| " + " | ".join(pad(i, tr[i]) for i in range(len(headers))) + " |")
    return "\n".join(lines)


def ms_str(sec: float | None) -> str:
    if sec is None:
        return "—"
    ms = float(sec) * 1000.0
    if ms >= 10000:
        return f"{ms:.1f}"
    return f"{ms:.2f}"


def regen_abc(src_jsonl: Path, out_md: Path, out_csv: Path) -> None:
    rows = [json.loads(l) for l in src_jsonl.read_text().splitlines() if l.strip()]
    HEAD = ["wave", "run", "K", "in", "out", "conc", "tok/s", "TTFT_ms", "mean_ITL_ms", "p99_ITL_ms", "wall_s", "ok"]

    def row_abc(r: dict) -> list[str]:
        ok = f"{r.get('throughput_successes', 0)}/{r.get('throughput_successes', 0) + r.get('throughput_failures', 0)}"
        return [
            r.get("wave", ""),
            str(r.get("run_id", "")),
            str(r.get("k_window", "")),
            str(r.get("input_len", "")),
            str(r.get("output_len", "")),
            str(r.get("concurrency", "")),
            f"{r.get('throughput_tok_per_s', 0):.2f}",
            ms_str(r.get("ttft_s_mean")),
            ms_str(r.get("mean_itl_s_mean")),
            ms_str(r.get("p99_itl_s_mean")),
            f"{r.get('wall_s', 0):.1f}",
            ok,
        ]

    by_wave: dict[str, list] = {}
    for r in rows:
        by_wave.setdefault(r["wave"], []).append(r)

    lines: list[str] = []
    lines.append("# Phase A sweep results (readable tables)")
    lines.append("")
    lines.append(f"Source: `results.jsonl` — {len(rows)} runs.")
    lines.append("")
    lines.append(
        "Latency columns (**TTFT_ms**, **mean_ITL_ms**, **p99_ITL_ms**) are **milliseconds** "
        "(isolated single-request streaming)."
    )
    lines.append("")
    lines.append(f"CSV: `{out_csv.name}`")
    lines.append("")
    lines.append("Columns are padded for monospace alignment.")
    lines.append("")
    titles = {
        "A": "## Wave A — 512×512, vary K and concurrency",
        "B": "## Wave B — concurrency 64, input×output×K grid",
        "C": "## Wave C — 1024×4096, concurrency 128",
    }
    for w in ["A", "B", "C"]:
        if w not in by_wave:
            continue
        rs = sorted(by_wave[w], key=lambda x: (x["run_id"], x.get("concurrency") or 0))
        lines.append(titles[w])
        lines.append("")
        lines.append(format_table(HEAD, [row_abc(r) for r in rs]))
        lines.append("")
    out_md.write_text("\n".join(lines).rstrip() + "\n")

    csv_cols = [
        "wave",
        "run_id",
        "k_window",
        "input_len",
        "output_len",
        "concurrency",
        "throughput_tok_per_s",
        "ttft_ms",
        "mean_itl_ms",
        "p99_itl_ms",
        "wall_s",
        "throughput_successes",
        "throughput_failures",
        "exit_code",
        "result_dir",
    ]
    with open(out_csv, "w", newline="") as f:
        wp = csv.DictWriter(f, fieldnames=csv_cols)
        wp.writeheader()
        for r in rows:
            wp.writerow(
                {
                    "wave": r.get("wave"),
                    "run_id": r.get("run_id"),
                    "k_window": r.get("k_window"),
                    "input_len": r.get("input_len"),
                    "output_len": r.get("output_len"),
                    "concurrency": r.get("concurrency"),
                    "throughput_tok_per_s": r.get("throughput_tok_per_s"),
                    "ttft_ms": round(float(r["ttft_s_mean"]) * 1000, 4) if r.get("ttft_s_mean") is not None else "",
                    "mean_itl_ms": round(float(r["mean_itl_s_mean"]) * 1000, 4)
                    if r.get("mean_itl_s_mean") is not None
                    else "",
                    "p99_itl_ms": round(float(r["p99_itl_s_mean"]) * 1000, 4)
                    if r.get("p99_itl_s_mean") is not None
                    else "",
                    "wall_s": r.get("wall_s"),
                    "throughput_successes": r.get("throughput_successes"),
                    "throughput_failures": r.get("throughput_failures"),
                    "exit_code": r.get("exit_code"),
                    "result_dir": r.get("result_dir", ""),
                }
            )


def regen_itl(src_jsonl: Path, out_md: Path, out_csv: Path) -> None:
    rows = [json.loads(l) for l in src_jsonl.read_text().splitlines() if l.strip()]
    HEAD = ["run", "K", "in", "out", "conc", "tok/s", "mITL_ms", "p99_req_ms", "p99_glob_ms", "TTFT_ms", "n_itl", "ok"]

    def row_itl(r: dict) -> list[str]:
        ok = f"{r.get('throughput_successes', 0)}/{r.get('throughput_successes', 0) + r.get('throughput_failures', 0)}"
        return [
            str(r.get("sweep_run_id", "")),
            str(r.get("k_window", "")),
            str(r.get("input_len", "")),
            str(r.get("output_len", "")),
            str(r.get("concurrency", "")),
            f"{r.get('throughput_tok_per_s', 0):.2f}",
            ms_str(r.get("concurrent_mean_itl_s")),
            ms_str(r.get("concurrent_mean_p99_itl_s")),
            ms_str(r.get("concurrent_global_p99_itl_s")),
            ms_str(r.get("concurrent_ttft_s_mean")),
            str(r.get("concurrent_itl_sample_count", "") or ""),
            ok,
        ]

    lines: list[str] = []
    lines.append("# ITL load sweep (concurrent streaming ITL)")
    lines.append("")
    lines.append(f"Source: `results.jsonl` — {len(rows)} rows.")
    lines.append("")
    lines.append("**All latency columns are milliseconds:** `mITL_ms`, `p99_req_ms`, `p99_glob_ms`, `TTFT_ms`.")
    lines.append("")
    lines.append(f"CSV: `{out_csv.name}`")
    lines.append("")
    lines.append("Columns are padded for monospace alignment.")
    lines.append("")
    by_k: dict[int, list] = {}
    for r in rows:
        by_k.setdefault(r["k_window"], []).append(r)
    for k in sorted(by_k.keys(), reverse=True):
        rs = sorted(by_k[k], key=lambda x: (x["sweep_run_id"], x["concurrency"]))
        lines.append(f"## K = {k}")
        lines.append("")
        lines.append(format_table(HEAD, [row_itl(r) for r in rs]))
        lines.append("")
    out_md.write_text("\n".join(lines).rstrip() + "\n")

    cols = [
        "sweep_run_id",
        "k_window",
        "input_len",
        "output_len",
        "concurrency",
        "throughput_tok_per_s",
        "mean_itl_ms",
        "p99_req_ms",
        "p99_glob_ms",
        "ttft_ms",
        "concurrent_itl_sample_count",
        "throughput_successes",
        "throughput_failures",
        "wall_session_s",
        "exit_code",
        "result_dir",
    ]
    with open(out_csv, "w", newline="") as f:
        wo = csv.DictWriter(f, fieldnames=cols)
        wo.writeheader()
        for r in rows:

            def to_ms(key: str):
                v = r.get(key)
                return round(float(v) * 1000, 6) if v is not None else ""

            wo.writerow(
                {
                    "sweep_run_id": r.get("sweep_run_id"),
                    "k_window": r.get("k_window"),
                    "input_len": r.get("input_len"),
                    "output_len": r.get("output_len"),
                    "concurrency": r.get("concurrency"),
                    "throughput_tok_per_s": r.get("throughput_tok_per_s"),
                    "mean_itl_ms": to_ms("concurrent_mean_itl_s"),
                    "p99_req_ms": to_ms("concurrent_mean_p99_itl_s"),
                    "p99_glob_ms": to_ms("concurrent_global_p99_itl_s"),
                    "ttft_ms": to_ms("concurrent_ttft_s_mean"),
                    "concurrent_itl_sample_count": r.get("concurrent_itl_sample_count"),
                    "throughput_successes": r.get("throughput_successes"),
                    "throughput_failures": r.get("throughput_failures"),
                    "wall_session_s": round(r.get("wall_s", 0), 2),
                    "exit_code": r.get("exit_code"),
                    "result_dir": r.get("result_dir", ""),
                }
            )


def regen_baseline(src_jsonl: Path, out_md: Path, out_csv: Path) -> None:
    rows = [json.loads(l) for l in src_jsonl.read_text().splitlines() if l.strip()]
    if not rows:
        out_md.write_text("# Baseline ITL load sweep\n\n(empty results.jsonl)\n")
        out_csv.write_text("")
        return
    # Keep baseline markdown in the same column layout as ITL tables:
    # run | K | in | out | conc | tok/s | mITL_ms | p99_req_ms | p99_glob_ms | TTFT_ms | n_itl | ok
    # Map trial rows to synthetic run ids (1..12) so 3 trials x 4 io-pairs
    # renders as 48 rows in a single "K = baseline" section.
    HEAD = ["run", "K", "in", "out", "conc", "tok/s", "mITL_ms", "p99_req_ms", "p99_glob_ms", "TTFT_ms", "n_itl", "ok"]

    def row_bl(r: dict) -> list[str]:
        trial = int(r.get("trial") or 1)
        io_run = int(r.get("sweep_run_id") or 0)
        synthetic_run = (trial - 1) * 4 + io_run
        ok = f"{r.get('throughput_successes', 0)}/{r.get('throughput_successes', 0) + r.get('throughput_failures', 0)}"
        return [
            str(synthetic_run),
            "baseline",
            str(r.get("input_len", "")),
            str(r.get("output_len", "")),
            str(r.get("concurrency", "")),
            f"{r.get('throughput_tok_per_s', 0):.2f}",
            ms_str(r.get("concurrent_mean_itl_s")),
            ms_str(r.get("concurrent_mean_p99_itl_s")),
            ms_str(r.get("concurrent_global_p99_itl_s")),
            ms_str(r.get("concurrent_ttft_s_mean")),
            str(r.get("concurrent_itl_sample_count", "") or ""),
            ok,
        ]

    lines: list[str] = []
    lines.append("# ITL load sweep (concurrent streaming ITL)")
    lines.append("")
    lines.append(f"Source: `results.jsonl` — {len(rows)} rows.")
    lines.append("")
    lines.append("**All latency columns are milliseconds:** `mITL_ms`, `p99_req_ms`, `p99_glob_ms`, `TTFT_ms`.")
    lines.append("")
    lines.append(f"CSV: `{out_csv.name}`")
    lines.append("")
    lines.append("Columns are padded for monospace alignment.")
    lines.append("")
    rs = sorted(
        rows,
        key=lambda x: (
            int(x.get("trial") or 1),
            int(x.get("sweep_run_id") or 0),
            int(x.get("concurrency") or 0),
        ),
    )
    lines.append("## K = baseline")
    lines.append("")
    lines.append(format_table(HEAD, [row_bl(r) for r in rs]))
    lines.append("")
    out_md.write_text("\n".join(lines).rstrip() + "\n")

    cols = [
        "sweep_run_id",
        "trial",
        "input_len",
        "output_len",
        "concurrency",
        "throughput_tok_per_s",
        "mean_itl_ms",
        "p99_req_ms",
        "p99_glob_ms",
        "ttft_ms",
        "concurrent_itl_sample_count",
        "throughput_successes",
        "throughput_failures",
        "wall_session_s",
        "exit_code",
        "result_dir",
    ]
    with open(out_csv, "w", newline="") as f:
        wo = csv.DictWriter(f, fieldnames=cols)
        wo.writeheader()
        for r in rows:

            def to_ms(key: str):
                v = r.get(key)
                return round(float(v) * 1000, 6) if v is not None else ""

            wo.writerow(
                {
                    "sweep_run_id": r.get("sweep_run_id"),
                    "trial": r.get("trial"),
                    "input_len": r.get("input_len"),
                    "output_len": r.get("output_len"),
                    "concurrency": r.get("concurrency"),
                    "throughput_tok_per_s": r.get("throughput_tok_per_s"),
                    "mean_itl_ms": to_ms("concurrent_mean_itl_s"),
                    "p99_req_ms": to_ms("concurrent_mean_p99_itl_s"),
                    "p99_glob_ms": to_ms("concurrent_global_p99_itl_s"),
                    "ttft_ms": to_ms("concurrent_ttft_s_mean"),
                    "concurrent_itl_sample_count": r.get("concurrent_itl_sample_count"),
                    "throughput_successes": r.get("throughput_successes"),
                    "throughput_failures": r.get("throughput_failures"),
                    "wall_session_s": round(r.get("wall_s", 0), 2),
                    "exit_code": r.get("exit_code"),
                    "result_dir": r.get("result_dir", ""),
                }
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--abc-jsonl",
        type=Path,
        default=Path("/data/ben/kv_phase_a_internal/sweep_full_ABC_v11/results.jsonl"),
    )
    p.add_argument(
        "--itl-jsonl",
        type=Path,
        default=Path("/data/ben/kv_phase_a_internal/itl_load_v1/results.jsonl"),
    )
    p.add_argument(
        "--baseline-jsonl",
        type=Path,
        default=Path("/data/ben/kv_phase_a_internal/baseline_itl_load_v1/results.jsonl"),
    )
    p.add_argument("--abc-out-dir", type=Path, default=None)
    p.add_argument("--itl-out-dir", type=Path, default=None)
    p.add_argument("--baseline-out-dir", type=Path, default=None)
    ns = p.parse_args()
    abc_dir = ns.abc_out_dir or ns.abc_jsonl.parent
    itl_dir = ns.itl_out_dir or ns.itl_jsonl.parent
    baseline_dir = ns.baseline_out_dir or ns.baseline_jsonl.parent
    regen_abc(ns.abc_jsonl, abc_dir / "results_table.md", abc_dir / "results_table.csv")
    regen_itl(ns.itl_jsonl, itl_dir / "results_table.md", itl_dir / "results_table.csv")
    if ns.baseline_jsonl.is_file():
        regen_baseline(
            ns.baseline_jsonl,
            baseline_dir / "results_table.md",
            baseline_dir / "results_table.csv",
        )
        print(f"Wrote {baseline_dir / 'results_table.md'}")
    else:
        print(f"skip baseline table (missing {ns.baseline_jsonl})", flush=True)
    print(f"Wrote {abc_dir / 'results_table.md'}")
    print(f"Wrote {itl_dir / 'results_table.md'}")


if __name__ == "__main__":
    main()
