#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
P/D Decode-Only Evaluation — TTFT / E2E / P99 E2E Focus
=========================================================

Publication-quality benchmark for layer-wise KV-cache pipelining in a
Prefill/Decode disaggregation scenario.  Focuses on the metrics that
demonstrate the *capacity* advantage of pipelined decode:

  TTFT      — Time To First Token.  Includes queue wait time.  Baseline
              queues requests beyond its KV capacity (~135 for Llama-8B
              on H100), so TTFT spikes.  Pipelined serves all
              immediately with its inflated block pool.

  E2E       — End-to-End latency per request (includes queue + prefill +
              all decode tokens).  Captures the full user experience.

  P99 E2E   — Tail E2E latency.  The cleanest metric for publication:
              shows that queued requests have dramatically worse P99.

  Throughput — Total output tokens/s across all requests.  Pipelined
              continues to climb past baseline's capacity wall.

This is the "DistServe / Splitwise" style evaluation (OSDI'24, ISCA'24).

Design
------
  - Uses PipelinedDecodeBenchConnector (dummy KV fill) for pipelined
    mode, standard vLLM for baseline.
  - Pushes N requests simultaneously (closed-loop, request_rate=inf)
    at concurrency levels 64, 128, 256, 384, 512.
  - Starts concurrency sweep at 64 (skip low concurrency where both
    modes behave identically).
  - Saves raw JSON results for later graph generation.

Usage
------
    # Quick test:
    python benchmarks/kv_pipelining_bench/eval_pd_decode.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --k-window 5

    # Full evaluation:
    python benchmarks/kv_pipelining_bench/eval_pd_decode.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --k-window 5 \\
        --concurrency 64 128 256 384 512 \\
        --num-prompts 200

    # Two-model comparison:
    python benchmarks/kv_pipelining_bench/eval_pd_decode.py \\
        --model Qwen/Qwen2.5-14B-Instruct \\
        --k-window 5 \\
        --concurrency 64 128 256 384 512
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# =========================================================================
# Constants
# =========================================================================

DEFAULT_PORT = 8199

# Metrics we focus on for P/D evaluation (TTFT + E2E, not ITL/TPOT)
PD_METRICS = [
    ("mean_ttft_ms", "Mean TTFT (ms)"),
    ("median_ttft_ms", "Median TTFT (ms)"),
    ("p99_ttft_ms", "P99 TTFT (ms)"),
    ("mean_e2el_ms", "Mean E2E (ms)"),
    ("median_e2el_ms", "Median E2E (ms)"),
    ("p99_e2el_ms", "P99 E2E (ms)"),
    ("mean_itl_ms", "Mean ITL (ms)"),
    ("p99_itl_ms", "P99 ITL (ms)"),
    ("output_throughput", "Output tput (tok/s)"),
    ("request_throughput", "Req tput (req/s)"),
    ("completed", "Completed"),
    ("failed", "Failed"),
]


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P/D decode-only evaluation (TTFT/E2E focus)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model / hardware
    p.add_argument("--model", required=True,
                   help="HuggingFace model name or path")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument(
        "--enforce-eager",
        dest="enforce_eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pass --enforce-eager to vLLM for both modes (matches validate_qa.py; "
            "avoids Inductor eliding KV connector hooks)."
        ),
    )

    # Pipelining params
    p.add_argument("--k-window", type=int, default=5,
                   help="K layers kept on GPU during pipelined decode")
    p.add_argument(
        "--inflate-kv-pool",
        dest="inflate_kv_pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For pipelined mode: pass --inflate-kv-pool (capacity / P/D story). "
            "Use --no-inflate-kv-pool to match validate_qa.py and reduce CPU "
            "pinned memory (recommended if the engine exits under load)."
        ),
    )

    # Evaluation scope
    p.add_argument(
        "--modes", nargs="+",
        default=["baseline", "pipelined"],
        choices=["baseline", "pipelined"],
        help="Server modes to compare",
    )
    p.add_argument("--concurrency", type=int, nargs="+",
                   default=[64, 128, 256, 384, 512],
                   help="Concurrency levels to test")
    p.add_argument("--num-prompts", type=int, default=200,
                   help="Number of requests per benchmark run")
    p.add_argument("--random-input-len", type=int, default=512)
    p.add_argument("--random-output-len", type=int, default=128)

    # Timing
    p.add_argument("--startup-timeout", type=int, default=600)
    p.add_argument("--warmup-requests", type=int, default=3)
    p.add_argument("--cooldown", type=int, default=10)

    # Output
    p.add_argument("--result-dir", default="/tmp/kv_pd_eval")

    return p.parse_args()


# =========================================================================
# Server management
# =========================================================================

def _cleanup_gpu_and_port(port: int) -> None:
    """Kill any leftover processes on the target port."""
    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"],
                       capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    time.sleep(2)


def _wait_for_server(port: int, timeout: int) -> bool:
    """Poll /health until the server is ready."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(2)
    return False


def _kill_server(proc: subprocess.Popen) -> None:
    """Gracefully stop a server process group."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        except Exception:
            pass


def _parse_server_capacity(log_file: str) -> dict:
    """Extract capacity info from server startup log."""
    info: dict[str, Any] = {}
    try:
        with open(log_file) as f:
            for line in f:
                m = re.search(
                    r"Maximum concurrency for ([\d,]+) tokens.*?:\s*([\d.]+)x",
                    line,
                )
                if m:
                    info["max_concurrency_reported"] = float(m.group(2))
                    info["tokens_per_request"] = int(
                        m.group(1).replace(",", "")
                    )
                m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", line)
                if m:
                    info["kv_cache_tokens"] = int(
                        m.group(1).replace(",", "")
                    )
                m = re.search(
                    r"Available KV cache memory:\s*([\d.]+)\s*GiB", line
                )
                if m:
                    info["kv_cache_memory_gib"] = float(m.group(1))
    except FileNotFoundError:
        pass
    return info


def launch_server(
    args: argparse.Namespace,
    mode: str,
    max_num_seqs: int,
    log_file: str,
) -> subprocess.Popen:
    """Launch a vLLM server in the specified mode."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--disable-log-requests",
        "--dtype", "auto",
        "--max-num-seqs", str(max_num_seqs),
    ]
    if getattr(args, "enforce_eager", True):
        cmd.append("--enforce-eager")

    if mode == "baseline":
        # Standard vLLM: all KV on GPU.
        pass

    elif mode == "pipelined":
        # PrefillStagingPipelinedConnector: real prefill, KV staged
        # to CPU, then K+1 pipelined decode with inflated block pool.
        # The inflated pool gives ~num_layers/(K+1) more capacity.
        kv_config = {
            "kv_connector": "PrefillStagingPipelinedConnector",
            "kv_role": "kv_both",
        }
        cmd += [
            "--enable-kv-pipelining",
            "--kv-k-window", str(args.k_window),
            "--kv-transfer-config", json.dumps(kv_config),
            "--no-enable-prefix-caching",
        ]
        if getattr(args, "inflate_kv_pool", True):
            cmd.append("--inflate-kv-pool")

    else:
        raise ValueError(f"Unknown mode: {mode}")

    log_fh = open(log_file, "w")
    proc = subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT, preexec_fn=os.setsid,
    )
    return proc


# =========================================================================
# Client: vllm bench serve wrapper
# =========================================================================

def _run_bench_serve(
    port: int,
    model: str,
    extra_args: list[str],
    result_file: str,
    num_prompts: int,
    max_concurrency: int | None,
    warmup: int,
    timeout: int = 1800,
) -> dict | None:
    """Run ``vllm bench serve`` and parse the JSON result."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "openai",
        "--port", str(port),
        "--model", model,
        "--num-prompts", str(num_prompts),
        "--request-rate", "inf",
        "--save-result",
        "--result-filename", result_file,
        "--num-warmups", str(warmup),
        "--seed", "42",
    ]
    if max_concurrency is not None:
        cmd += ["--max-concurrency", str(max_concurrency)]
    cmd += extra_args

    print(f"    $ {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )

    if result.returncode != 0:
        print(f"    [ERROR] bench serve failed (rc={result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-15:]:
                print(f"      {line}")
        return None

    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"      {line}")

    if os.path.exists(result_file):
        with open(result_file) as f:
            return json.load(f)
    return None


# =========================================================================
# Per-mode evaluation
# =========================================================================

def run_mode_evaluation(
    args: argparse.Namespace,
    mode: str,
    result_dir: Path,
) -> dict:
    """Run all concurrency levels for a mode, return results."""
    max_conc = max(args.concurrency)
    log_file = str(result_dir / f"server_{mode}.log")

    _cleanup_gpu_and_port(args.port)

    print(f"\n  Launching server [{mode}] ...")
    proc = launch_server(args, mode, max_num_seqs=max_conc,
                         log_file=log_file)

    try:
        print(f"  Waiting for server (timeout={args.startup_timeout}s) ...")
        if not _wait_for_server(args.port, args.startup_timeout):
            print(f"  [FAIL] Server failed to start. Check: {log_file}")
            return {"mode": mode, "error": "startup_failed"}

        server_info = _parse_server_capacity(log_file)
        print(f"  [OK] Server ready. Info: {server_info}")

        results_per_conc: dict[str, dict] = {}

        for conc in sorted(args.concurrency):
            print(f"\n  --- {mode} / concurrency={conc} ---")
            result_file = str(
                result_dir / f"results_{mode}_conc{conc}.json"
            )
            extra_args = [
                "--dataset-name", "random",
                "--random-input-len", str(args.random_input_len),
                "--random-output-len", str(args.random_output_len),
            ]

            result = _run_bench_serve(
                port=args.port, model=args.model, extra_args=extra_args,
                result_file=result_file, num_prompts=args.num_prompts,
                max_concurrency=conc, warmup=args.warmup_requests,
            )

            if result is not None:
                results_per_conc[str(conc)] = result
                ttft = result.get("mean_ttft_ms")
                p99_e2e = result.get("p99_e2el_ms")
                e2e = result.get("mean_e2el_ms")
                tput = result.get("output_throughput")
                completed = result.get("completed", "?")
                failed = result.get("failed", "?")
                print(
                    f"    -> completed={completed}, failed={failed}",
                    end="",
                )
                if ttft is not None:
                    print(f", TTFT={ttft:.1f}ms", end="")
                if e2e is not None:
                    print(f", E2E={e2e:.1f}ms", end="")
                if p99_e2e is not None:
                    print(f", P99_E2E={p99_e2e:.1f}ms", end="")
                if tput is not None:
                    print(f", tput={tput:.0f} tok/s", end="")
                print()
            else:
                print(f"    -> [FAIL] No results")
                results_per_conc[str(conc)] = {"error": "failed"}

        return {
            "mode": mode,
            "server_info": server_info,
            "concurrency_results": results_per_conc,
        }
    finally:
        print(f"  Stopping {mode} server ...")
        _kill_server(proc)
        time.sleep(args.cooldown)


# =========================================================================
# Comparison reporting
# =========================================================================

def print_pd_comparison(
    modes: list[str],
    all_results: dict[str, dict],
    concurrency_levels: list[int],
) -> None:
    """Print P/D-focused comparison tables."""
    print(f"\n{'=' * 100}")
    print("  P/D DECODE EVALUATION — TTFT / E2E / P99 E2E COMPARISON")
    print(f"{'=' * 100}")

    for conc in sorted(concurrency_levels):
        ckey = str(conc)
        print(f"\n  concurrency = {conc}")

        # Header
        header = f"  {'Metric':<25s}"
        for m in modes:
            header += f"  {m:>14s}"
        if "baseline" in modes and "pipelined" in modes:
            header += f"  {'Delta':>12s}"
        print(header)
        print(f"  {'─' * 25}", end="")
        for _ in modes:
            print(f"  {'─' * 14}", end="")
        if "baseline" in modes and "pipelined" in modes:
            print(f"  {'─' * 12}", end="")
        print()

        for metric_key, metric_label in PD_METRICS:
            line = f"  {metric_label:<25s}"
            values: dict[str, Any] = {}
            for m in modes:
                cr = (all_results.get(m, {})
                      .get("concurrency_results", {})
                      .get(ckey, {}))
                val = cr.get(metric_key)
                values[m] = val
                if val is not None:
                    if isinstance(val, float):
                        line += f"  {val:>14.2f}"
                    else:
                        line += f"  {val:>14}"
                else:
                    line += f"  {'N/A':>14s}"

            # Delta: pipelined vs baseline
            if "baseline" in modes and "pipelined" in modes:
                bv = values.get("baseline")
                pv = values.get("pipelined")
                if (
                    bv is not None
                    and pv is not None
                    and isinstance(bv, (int, float))
                    and bv != 0
                ):
                    pct = ((pv - bv) / abs(bv)) * 100
                    line += f"  {pct:>+10.1f}%"
                else:
                    line += f"  {'—':>12s}"

            print(line)


def print_summary(
    args: argparse.Namespace,
    all_results: dict[str, dict],
) -> None:
    """Print key findings for paper."""
    print(f"\n{'=' * 100}")
    print("  KEY FINDINGS FOR PUBLICATION")
    print(f"{'=' * 100}")

    # For each concurrency level, show the TTFT and capacity advantage
    for conc in sorted(args.concurrency):
        ckey = str(conc)
        b = all_results.get("baseline", {}).get("concurrency_results", {}).get(ckey, {})
        p = all_results.get("pipelined", {}).get("concurrency_results", {}).get(ckey, {})

        if not b or not p:
            continue

        b_ttft = b.get("mean_ttft_ms")
        p_ttft = p.get("mean_ttft_ms")
        b_p99e2e = b.get("p99_e2el_ms")
        p_p99e2e = p.get("p99_e2el_ms")
        b_tput = b.get("output_throughput")
        p_tput = p.get("output_throughput")
        b_failed = b.get("failed", 0)
        p_failed = p.get("failed", 0)
        b_completed = b.get("completed", 0)
        p_completed = p.get("completed", 0)

        print(f"\n  Concurrency = {conc}:")

        if b_failed > 0 and p_failed == 0:
            print(
                f"    ** Baseline FAILED {b_failed} requests "
                f"(completed {b_completed}), "
                f"pipelined completed ALL {p_completed} **"
            )

        if b_ttft is not None and p_ttft is not None and b_ttft > 0:
            ratio = b_ttft / p_ttft if p_ttft > 0 else float("inf")
            print(
                f"    TTFT: baseline={b_ttft:.1f}ms vs "
                f"pipelined={p_ttft:.1f}ms "
                f"(baseline {ratio:.1f}x slower)"
            )

        if b_p99e2e is not None and p_p99e2e is not None and p_p99e2e > 0:
            ratio = b_p99e2e / p_p99e2e if p_p99e2e > 0 else float("inf")
            print(
                f"    P99 E2E: baseline={b_p99e2e:.1f}ms vs "
                f"pipelined={p_p99e2e:.1f}ms "
                f"(baseline {ratio:.1f}x worse)"
            )

        if b_tput is not None and p_tput is not None:
            print(
                f"    Throughput: baseline={b_tput:.0f} vs "
                f"pipelined={p_tput:.0f} tok/s"
            )

    # Server capacity comparison
    b_info = all_results.get("baseline", {}).get("server_info", {})
    p_info = all_results.get("pipelined", {}).get("server_info", {})
    b_cap = b_info.get("max_concurrency_reported")
    p_cap = p_info.get("max_concurrency_reported")
    if b_cap and p_cap:
        print(
            f"\n  Server-reported capacity: "
            f"baseline={b_cap:.0f}x vs pipelined={p_cap:.0f}x "
            f"({p_cap/b_cap:.1f}x improvement)"
        )


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  P/D Decode-Only Evaluation (TTFT / E2E / P99 Focus)")
    print("  DistServe/Splitwise-style capacity benchmark")
    print("=" * 80)
    print(f"  Model             : {args.model}")
    print(f"  Max model len     : {args.max_model_len}")
    print(f"  GPU mem util      : {args.gpu_memory_utilization}")
    print(f"  K-window          : {args.k_window}")
    print(f"  Inflate KV pool   : {getattr(args, 'inflate_kv_pool', True)}")
    print(f"  Modes             : {args.modes}")
    print(f"  Concurrency       : {args.concurrency}")
    print(f"  Num prompts       : {args.num_prompts}")
    print(f"  Input/Output len  : {args.random_input_len}/{args.random_output_len}")
    print(f"  Result dir        : {args.result_dir}")

    all_results: dict[str, dict] = {}

    for mode in args.modes:
        print(f"\n{'#' * 80}")
        print(f"# MODE: {mode.upper()}")
        print(f"{'#' * 80}")

        all_results[mode] = run_mode_evaluation(args, mode, result_dir)

    # Print comparison
    print_pd_comparison(args.modes, all_results, args.concurrency)
    if "baseline" in args.modes and "pipelined" in args.modes:
        print_summary(args, all_results)

    # Save results
    final = {
        "config": {
            "model": args.model,
            "max_model_len": args.max_model_len,
            "k_window": args.k_window,
            "inflate_kv_pool": getattr(args, "inflate_kv_pool", True),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": getattr(args, "enforce_eager", True),
            "modes": args.modes,
            "concurrency": args.concurrency,
            "num_prompts": args.num_prompts,
            "request_rate": "inf",
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
        },
        "results": all_results,
    }

    summary_path = result_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(final, f, indent=2, default=str)

    print(f"\n  Full results saved to: {result_dir}/")
    print(f"  Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
