#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-ON K sweep driver.

For each (model, repeat) launches ``decode_only_bench.py --modes real_prefill
--streaming-throughput`` over the requested list of concurrencies, parses the
resulting ``results_real_prefill.json``, and emits one JSONL row per (model,
K, repeat) plus an aggregated CSV consumable by ``admission_predictor.py``.

The wrapped bench uses the layer-KV pipelined connector
``PrefillStagingPipelinedConnector`` and passes
``--enable-kv-pipelining --kv-k-window K --free-original-kv-cache`` to the API
server, so the K sweep observes the full feature-ON behavior (compute + PCIe).

Usage::

    python feature_on_k_sweep.py \
        --models meta-llama/Meta-Llama-3.1-8B-Instruct Qwen/Qwen2.5-14B-Instruct \
        --k-values 16 32 64 128 192 256 384 \
        --input-len 1024 --output-len 1024 \
        --k-window 4 --repeats 2 \
        --out /data/ben/kv_phase_a_internal/open_loop_eval_v1/feature_on_k_sweep

The sweep is resumable: cells whose JSONL row already exists with successful
data are skipped (override with ``--no-skip-existing``).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

_THIS_DIR = Path(__file__).resolve().parent
_DECODE_ONLY_BENCH = _THIS_DIR / "decode_only_bench.py"


def _model_slug(model: str) -> str:
    return model.replace("/", "__")


def _free_port(start: int = 8200, attempts: int = 50) -> int:
    """Return a free TCP port on localhost starting at ``start``."""
    for offset in range(attempts):
        port = start + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port in range [{start}, {start + attempts})")


def _load_existing_rows(jsonl: Path) -> list[dict]:
    rows: list[dict] = []
    if not jsonl.is_file():
        return rows
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _completed_keys(rows: Iterable[dict]) -> set[tuple[str, int, int, int, int]]:
    """Return {(model, input_len, output_len, K, repeat)} for rows with valid p99 ITL."""
    out: set[tuple[str, int, int, int, int]] = set()
    for r in rows:
        if r.get("p99_itl_ms") is None:
            continue
        try:
            key = (
                str(r["model"]),
                int(r["input_len"]),
                int(r["output_len"]),
                int(r["K"]),
                int(r["repeat"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        out.add(key)
    return out


def _run_decode_only_bench(
    *,
    python: str,
    model: str,
    port: int,
    input_len: int,
    output_len: int,
    concurrencies: list[int],
    k_window: int,
    gpu_mem_util: float,
    max_model_len: int,
    result_dir: Path,
    log_file: Path,
    request_timeout: int,
    startup_timeout: int,
    warmup_requests: int,
    extra_env: dict[str, str] | None = None,
) -> int:
    """Invoke decode_only_bench.py with --modes real_prefill --streaming-throughput.

    Returns the subprocess exit code. Output JSON is written to
    ``result_dir/results_real_prefill.json``.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python, str(_DECODE_ONLY_BENCH),
        "--model", model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--concurrency", *[str(c) for c in concurrencies],
        "--k-window", str(k_window),
        "--modes", "real_prefill",
        "--result-dir", str(result_dir),
        "--streaming-throughput",
        "--startup-timeout", str(startup_timeout),
        "--request-timeout", str(request_timeout),
        "--warmup-requests", str(warmup_requests),
    ]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(f"\n[CMD] {' '.join(cmd)}", flush=True)
    with log_file.open("w") as lf:
        lf.write(f"# {' '.join(cmd)}\n")
        for k in ("HF_HOME", "HF_HUB_CACHE", "HF_TOKEN", "CUDA_VISIBLE_DEVICES"):
            if k in env:
                show = env[k] if k != "HF_TOKEN" else f"<{len(env[k])} chars>"
                lf.write(f"# env {k}={show}\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def _parse_throughput_results(
    bench_json: Path,
    *,
    model: str,
    input_len: int,
    output_len: int,
    k_window: int,
    repeat: int,
) -> list[dict]:
    """Parse decode_only_bench results_real_prefill.json into one row per K.

    Each row carries the columns expected by ``admission_predictor.from_csv``.
    """
    if not bench_json.is_file():
        return []
    try:
        data = json.loads(bench_json.read_text())
    except json.JSONDecodeError:
        return []

    rows: list[dict] = []
    throughput = data.get("throughput") or []
    latency_block = data.get("latency") or {}
    single_p99_itl_ms = (
        float(latency_block["p99_itl_s_mean"]) * 1000.0
        if latency_block.get("p99_itl_s_mean") is not None
        else None
    )
    single_mean_itl_ms = (
        float(latency_block["mean_itl_s_mean"]) * 1000.0
        if latency_block.get("mean_itl_s_mean") is not None
        else None
    )
    single_ttft_ms = (
        float(latency_block["ttft_s_mean"]) * 1000.0
        if latency_block.get("ttft_s_mean") is not None
        else None
    )

    for cell in throughput:
        K = int(cell.get("concurrency", 0))
        if K <= 0:
            continue
        # Streaming-throughput cells contain both per-request and pooled-global
        # p99 ITL. Use the pooled-global one ("concurrent_global_p99_itl_s")
        # which reflects worst-case TBT across all in-flight requests --
        # exactly what the admission policy must protect.
        gp99 = cell.get("concurrent_global_p99_itl_s")
        if gp99 is None:
            gp99 = cell.get("concurrent_mean_p99_itl_s")
        p99_itl_ms = float(gp99) * 1000.0 if gp99 is not None else None
        mean_itl_ms = (
            float(cell["concurrent_mean_itl_s_mean"]) * 1000.0
            if cell.get("concurrent_mean_itl_s_mean") is not None
            else None
        )
        ttft_ms = (
            float(cell["concurrent_ttft_s_mean"]) * 1000.0
            if cell.get("concurrent_ttft_s_mean") is not None
            else None
        )
        rows.append({
            "model": model,
            "input_len": input_len,
            "output_len": output_len,
            "K": K,
            "k_window": k_window,
            "repeat": repeat,
            "p99_itl_ms": p99_itl_ms,
            "mean_itl_ms": mean_itl_ms,
            "ttft_ms": ttft_ms,
            "throughput_tok_s": float(cell.get("throughput_tok_per_s", 0.0)),
            "successes": int(cell.get("successes", 0)),
            "failures": int(cell.get("failures", 0)),
            "elapsed_s": float(cell.get("elapsed_s", 0.0) or 0.0),
            "single_request_p99_itl_ms": single_p99_itl_ms,
            "single_request_mean_itl_ms": single_mean_itl_ms,
            "single_request_ttft_ms": single_ttft_ms,
        })
    return rows


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _aggregate_csv(jsonl: Path, csv_path: Path) -> None:
    """Aggregate per-row JSONL into a CSV averaged across repeats."""
    rows = _load_existing_rows(jsonl)
    if not rows:
        return
    grouped: dict[tuple[str, int, int, int, int], list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("p99_itl_ms") is None:
            continue
        key = (
            str(r["model"]),
            int(r["input_len"]),
            int(r["output_len"]),
            int(r["K"]),
            int(r.get("k_window", 0) or 0),
        )
        grouped[key].append(r)

    out_rows: list[dict] = []

    def _avg(rs: list[dict], field: str) -> float | None:
        vs = [float(r[field]) for r in rs if r.get(field) is not None]
        return (sum(vs) / len(vs)) if vs else None

    for key, rs in sorted(grouped.items()):
        model, in_len, out_len, K, kw = key
        out_rows.append({
            "model": model,
            "input_len": in_len,
            "output_len": out_len,
            "K": K,
            "k_window": kw,
            "n_repeats": len(rs),
            "p99_itl_ms": _avg(rs, "p99_itl_ms"),
            "mean_itl_ms": _avg(rs, "mean_itl_ms"),
            "ttft_ms": _avg(rs, "ttft_ms"),
            "throughput_tok_s": _avg(rs, "throughput_tok_s"),
            "single_request_p99_itl_ms": _avg(rs, "single_request_p99_itl_ms"),
        })

    if not out_rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n[AGG] Wrote {csv_path} ({len(out_rows)} rows)", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--models", nargs="+",
        default=[
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
        ],
    )
    ap.add_argument(
        "--k-values", nargs="+", type=int,
        default=[16, 32, 64, 128, 192, 256, 384],
        help="Concurrency levels (running decode batch sizes K) to sweep.",
    )
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=1024)
    ap.add_argument("--k-window", type=int, default=4,
                    help="--kv-k-window passed to the server.")
    ap.add_argument("--repeats", type=int, default=2,
                    help="Number of repeat runs per (model, K) for noise estimate.")
    ap.add_argument(
        "--out", type=Path,
        default=Path("/data/ben/kv_phase_a_internal/open_loop_eval_v1/feature_on_k_sweep"),
        help="Output root: results JSONL/CSV + per-cell logs go here.",
    )
    ap.add_argument(
        "--python", default=sys.executable,
        help="Python interpreter used to launch decode_only_bench (must have vLLM).",
    )
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--gpu-id", type=str, default=None,
                    help="CUDA_VISIBLE_DEVICES override for the server.")
    ap.add_argument("--port-base", type=int, default=8200)
    ap.add_argument("--startup-timeout", type=int, default=600)
    ap.add_argument("--request-timeout", type=int, default=600)
    ap.add_argument("--warmup-requests", type=int, default=2)
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="If set, do not skip cells whose JSONL row already exists.")
    ap.add_argument("--hf-home", type=str, default=os.environ.get("HF_HOME"),
                    help="HF_HOME for the bench subprocess. If unset and HF_HUB_CACHE "
                         "also unset, defaults to /data/ben/huggingface.")
    ap.add_argument("--hf-hub-cache", type=str, default=os.environ.get("HF_HUB_CACHE"),
                    help="HF_HUB_CACHE for the bench subprocess. Defaults to "
                         "/data/ben/huggingface (where the May 4 paper models live).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out / "feature_on_k_sweep.jsonl"
    csv_path = args.out / "feature_on_k_sweep.csv"
    logs_dir = args.out / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_rows(jsonl_path)
    completed = _completed_keys(existing) if not args.no_skip_existing else set()
    print(f"[INFO] Existing JSONL rows: {len(existing)}")
    print(f"[INFO] Completed keys (model x K x repeat): {len(completed)}")

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        print(f"[INFO] CUDA_VISIBLE_DEVICES={args.gpu_id}")

    sys.path.insert(0, str(_THIS_DIR))
    from hf_env import apply_hf_token_from_user_file  # noqa: WPS433
    apply_hf_token_from_user_file()
    extra_env: dict[str, str] = {}
    hub_cache = args.hf_hub_cache or "/data/ben/huggingface"
    extra_env["HF_HUB_CACHE"] = hub_cache
    if args.hf_home:
        extra_env["HF_HOME"] = args.hf_home
    if "HF_TOKEN" in os.environ:
        extra_env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    print(f"[INFO] HF_HUB_CACHE={hub_cache}")
    if "HF_HOME" in extra_env:
        print(f"[INFO] HF_HOME={extra_env['HF_HOME']}")
    if "HF_TOKEN" in extra_env:
        print("[INFO] HF_TOKEN: set")

    total_runs = 0
    skipped_runs = 0
    for model in args.models:
        for repeat in range(args.repeats):
            ks_to_run = [
                K for K in args.k_values
                if (model, args.input_len, args.output_len, K, repeat) not in completed
            ]
            if not ks_to_run:
                print(f"[SKIP] {model} repeat={repeat}: all K cells already complete.")
                skipped_runs += 1
                continue
            slug = _model_slug(model)
            cell_dir = args.out / slug / f"repeat_{repeat}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            log_file = logs_dir / f"{slug}__repeat_{repeat}.log"

            port = _free_port(start=args.port_base)
            t0 = time.monotonic()
            print(
                f"\n=== {model} | repeat={repeat} | "
                f"K={ks_to_run} | port={port} ===",
                flush=True,
            )
            rc = _run_decode_only_bench(
                python=args.python,
                model=model,
                port=port,
                input_len=args.input_len,
                output_len=args.output_len,
                concurrencies=ks_to_run,
                k_window=args.k_window,
                gpu_mem_util=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                result_dir=cell_dir,
                log_file=log_file,
                request_timeout=args.request_timeout,
                startup_timeout=args.startup_timeout,
                warmup_requests=args.warmup_requests,
                extra_env=extra_env,
            )
            wall = time.monotonic() - t0
            print(f"[DONE] {model} repeat={repeat} rc={rc} wall={wall:.1f}s", flush=True)

            bench_json = cell_dir / "results_real_prefill.json"
            new_rows = _parse_throughput_results(
                bench_json,
                model=model,
                input_len=args.input_len,
                output_len=args.output_len,
                k_window=args.k_window,
                repeat=repeat,
            )
            new_rows = [r for r in new_rows if r["K"] in ks_to_run]
            if not new_rows:
                print(f"[WARN] No usable rows from {bench_json}; check log {log_file}.")
            else:
                _append_jsonl(jsonl_path, new_rows)
                print(f"[WROTE] {len(new_rows)} rows -> {jsonl_path}")
            total_runs += 1

    if total_runs == 0:
        print("\nNo runs executed (all cells already done?). Re-running aggregation only.")

    _aggregate_csv(jsonl_path, csv_path)
    print(
        f"\nSweep complete. ran={total_runs} skipped_full_repeats={skipped_runs} "
        f"jsonl={jsonl_path} csv={csv_path}"
    )


if __name__ == "__main__":
    main()
