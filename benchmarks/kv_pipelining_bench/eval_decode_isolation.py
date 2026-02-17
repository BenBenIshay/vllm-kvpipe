#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Decode-Isolated KV Cache Pipelining Evaluation
===============================================

Publication-quality benchmark for layer-wise KV-cache pipelining in a
Prefill/Decode disaggregation scenario.  All modes perform **real model
prefill** — the only difference is how KV caches are managed during the
decode phase.

This benchmark is designed for single-GPU academic evaluation on H100.
It emulates P/D disaggregation by running prefill locally, then
measuring decode-only performance under different KV management
strategies.

Modes (server configurations)
------------------------------
  baseline     Standard vLLM: real prefill, all KV stays on GPU during
               decode.  Gold standard for decode latency but limited by
               GPU KV memory for concurrent capacity.

  pipelined    PrefillStagingPipelinedConnector: real prefill, KV staged
               to CPU pinned memory, then loaded K layers at a time
               during decode via K+1 rotating GPU tensors.  Only K+1
               physical tensors on GPU → ~num_layers/(K+1) more decode
               capacity.

  offloading   vLLM native CPU offloading (OffloadingConnector): real
               prefill, cold KV blocks evicted to CPU, loaded back on
               demand.  Existing SOTA KV memory optimization.

Workloads (top-tier conference benchmarks)
-------------------------------------------
  random       Synthetic fixed-length prompts (vLLM standard).
               Used in: OSDI'24, SOSP'23, MLSys benchmarks.
  longbench    Long-context QA from THUDM/LongBench.
               Used in: NeurIPS'23, ICLR'24 long-context evaluations.
  multi_round  Emulated multi-turn QA with growing context.
               Used in: DistServe (OSDI'24), Splitwise (ISCA'24).

Metrics (decode-focused, apple-to-apple)
-----------------------------------------
  ITL          Inter-token latency, mean + P99 (ms) — pure decode metric
  TPOT         Time per output token (ms) — pure decode metric
  Throughput   Output tokens/second
  Max Conc     Maximum concurrent decode requests before failure

Tests
------
  1. Decode latency comparison at fixed concurrency levels
  2. Throughput scaling with increasing concurrency
  3. Capacity sweep: max concurrent requests before OOM/failure

Usage
------
    # Quick test with random workload:
    python benchmarks/kv_pipelining_bench/eval_decode_isolation.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --workloads random \\
        --modes baseline pipelined \\
        --k-window 4

    # Full paper evaluation:
    python benchmarks/kv_pipelining_bench/eval_decode_isolation.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --workloads random longbench multi_round \\
        --modes baseline pipelined offloading \\
        --k-window 4 \\
        --concurrency 1 4 16 32 64 128 \\
        --capacity-test

    # Two-model comparison:
    python benchmarks/kv_pipelining_bench/eval_decode_isolation.py \\
        --model Qwen/Qwen2.5-14B-Instruct \\
        --workloads random longbench \\
        --modes baseline pipelined \\
        --k-window 4 \\
        --concurrency 1 4 16 32 \\
        --capacity-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# =========================================================================
# Constants
# =========================================================================

DEFAULT_PORT = 8199
LONGBENCH_HF_DATASET = "THUDM/LongBench"

# =========================================================================
# CLI
# =========================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decode-isolated KV pipelining evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model / hardware
    p.add_argument("--model", required=True,
                   help="HuggingFace model name or path")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)

    # Pipelining params
    p.add_argument("--k-window", type=int, default=4,
                   help="K layers kept on GPU during pipelined decode")

    # Offloading params
    p.add_argument("--offloading-size-gb", type=float, default=40.0,
                   help="CPU memory budget for native offloading (GB)")

    # Evaluation scope
    p.add_argument(
        "--workloads", nargs="+",
        default=["random"],
        choices=["random", "longbench", "multi_round"],
        help="Workloads to evaluate",
    )
    p.add_argument(
        "--modes", nargs="+", default=["baseline", "pipelined"],
        choices=["baseline", "pipelined", "pipelined_inflated", "offloading"],
        help="Server modes to compare",
    )

    # Workload params
    p.add_argument("--num-prompts", type=int, default=200,
                   help="Number of requests per benchmark run")
    p.add_argument("--concurrency", type=int, nargs="+",
                   default=[1, 4, 16, 64],
                   help="Max concurrency levels to test")
    p.add_argument("--request-rate", type=float, default=float("inf"),
                   help="Requests/s (inf = closed-loop)")

    # Random workload
    p.add_argument("--random-input-len", type=int, default=512)
    p.add_argument("--random-output-len", type=int, default=128)

    # LongBench workload
    p.add_argument("--longbench-min-input", type=int, default=2000,
                   help="Minimum input tokens for LongBench samples")
    p.add_argument("--longbench-max-input", type=int, default=None,
                   help="Maximum input tokens (default: max-model-len - 256)")
    p.add_argument("--longbench-output-len", type=int, default=128)

    # Multi-round workload
    p.add_argument("--multi-round-turns", type=int, default=5)
    p.add_argument("--multi-round-sessions", type=int, default=20)
    p.add_argument("--multi-round-input-len", type=int, default=256)
    p.add_argument("--multi-round-output-len", type=int, default=128)

    # Capacity test
    p.add_argument("--capacity-test", action="store_true",
                   help="Run capacity sweep to find max concurrent requests")
    p.add_argument("--capacity-input-len", type=int, default=512,
                   help="Input length for capacity test")
    p.add_argument("--capacity-output-len", type=int, default=128,
                   help="Output length for capacity test")
    p.add_argument("--capacity-start", type=int, default=4,
                   help="Starting concurrency for capacity sweep")
    p.add_argument("--capacity-step", type=int, default=4,
                   help="Concurrency step for capacity sweep")
    p.add_argument("--capacity-max", type=int, default=512,
                   help="Maximum concurrency to try in capacity sweep")

    # Timing
    p.add_argument("--startup-timeout", type=int, default=600)
    p.add_argument("--warmup-requests", type=int, default=3)
    p.add_argument("--cooldown", type=int, default=10)

    # Output
    p.add_argument("--result-dir", default="/tmp/kv_pipelining_decode_eval")

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
    info = {}
    try:
        with open(log_file) as f:
            for line in f:
                # "Maximum concurrency for 4,096 tokens per request: 187.33x"
                m = re.search(
                    r"Maximum concurrency for ([\d,]+) tokens.*?:\s*([\d.]+)x",
                    line,
                )
                if m:
                    info["max_concurrency_reported"] = float(m.group(2))
                    info["tokens_per_request"] = int(
                        m.group(1).replace(",", "")
                    )
                # "GPU KV cache size: 767,296 tokens"
                m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", line)
                if m:
                    info["kv_cache_tokens"] = int(
                        m.group(1).replace(",", "")
                    )
                # "Available KV cache memory: 81.96 GiB"
                m = re.search(
                    r"Available KV cache memory:\s*([\d.]+)\s*GiB", line
                )
                if m:
                    info["kv_cache_memory_gib"] = float(m.group(1))
                # num_layers from config log
                m = re.search(r"num_hidden_layers.*?(\d+)", line)
                if m:
                    info["num_layers"] = int(m.group(1))
    except FileNotFoundError:
        pass
    return info


def launch_server(
    args: argparse.Namespace,
    mode: str,
    max_num_seqs: int,
    log_file: str,
) -> subprocess.Popen:
    """Launch a vLLM server in the specified mode.

    All modes perform real model prefill.  The difference is only in
    KV cache management during decode.
    """
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

    if mode == "baseline":
        # Standard vLLM: real prefill, all KV on GPU.
        # No KV connector — vanilla vLLM decode.
        pass  # no extra flags

    elif mode == "pipelined":
        # Our solution: real prefill + K+1 tensor pipelined decode.
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

    elif mode == "pipelined_inflated":
        # Our solution with block pool inflation for capacity testing.
        # Uses same total GPU memory as baseline but with K+1 tensors,
        # allowing ~num_layers/(K+1) more concurrent decode requests.
        kv_config = {
            "kv_connector": "PrefillStagingPipelinedConnector",
            "kv_role": "kv_both",
        }
        cmd += [
            "--enable-kv-pipelining",
            "--kv-k-window", str(args.k_window),
            "--inflate-kv-pool",
            "--kv-transfer-config", json.dumps(kv_config),
            "--no-enable-prefix-caching",
        ]

    elif mode == "offloading":
        # vLLM native CPU offloading (SOTA comparison).
        cmd += [
            "--kv-offloading-backend", "native",
            "--kv-offloading-size", str(args.offloading_size_gb),
            "--disable-hybrid-kv-cache-manager",
        ]

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
    request_rate: float,
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
        "--request-rate", str(request_rate),
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
# Workload runners
# =========================================================================

def run_random_workload(
    args: argparse.Namespace,
    mode: str,
    max_concurrency: int,
    result_dir: Path,
) -> dict | None:
    """Random synthetic workload — vLLM standard benchmark."""
    result_file = str(result_dir / f"results_{mode}_conc{max_concurrency}.json")
    extra_args = [
        "--dataset-name", "random",
        "--random-input-len", str(args.random_input_len),
        "--random-output-len", str(args.random_output_len),
    ]
    return _run_bench_serve(
        port=args.port, model=args.model, extra_args=extra_args,
        result_file=result_file, num_prompts=args.num_prompts,
        request_rate=args.request_rate, max_concurrency=max_concurrency,
        warmup=args.warmup_requests,
    )


def _prepare_longbench_jsonl(
    args: argparse.Namespace, result_dir: Path
) -> str:
    """Download LongBench, filter by length, write JSONL for custom dataset."""
    jsonl_path = str(result_dir / "longbench_samples.jsonl")
    if os.path.exists(jsonl_path):
        print(f"    Using cached LongBench JSONL: {jsonl_path}")
        return jsonl_path

    print("    Downloading LongBench dataset ...")
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise RuntimeError(
            "'datasets' package required for LongBench. "
            "Install: pip install datasets"
        )

    subsets = ["qasper", "multifieldqa_en", "narrativeqa", "2wikimqa"]
    max_input = args.longbench_max_input or (args.max_model_len - 256)
    min_input = args.longbench_min_input

    samples = []
    for subset in subsets:
        try:
            ds = load_dataset(
                "THUDM/LongBench", subset, split="test",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"    Warning: could not load LongBench/{subset}: {e}")
            continue
        for row in ds:
            context = row.get("context", "")
            question = row.get("input", "")
            prompt = (
                f"Read the following passage and answer the question.\n\n"
                f"Passage:\n{context}\n\n"
                f"Question: {question}\n\nAnswer:"
            )
            approx_tokens = len(prompt.split())
            if min_input <= approx_tokens <= max_input:
                samples.append({
                    "prompt": prompt,
                    "output_tokens": args.longbench_output_len,
                })
        print(f"    {subset}: {len(samples)} samples so far")

    if not samples:
        print("    Warning: no LongBench samples matched filters")
        return jsonl_path

    with open(jsonl_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"    Wrote {len(samples)} LongBench samples to {jsonl_path}")
    return jsonl_path


def run_longbench_workload(
    args: argparse.Namespace,
    mode: str,
    max_concurrency: int,
    result_dir: Path,
) -> dict | None:
    """Long-context QA from THUDM/LongBench."""
    jsonl_path = _prepare_longbench_jsonl(args, result_dir)
    if not os.path.exists(jsonl_path):
        print("    [SKIP] LongBench JSONL not available")
        return None

    result_file = str(result_dir / f"results_{mode}_conc{max_concurrency}.json")
    extra_args = [
        "--dataset-name", "custom",
        "--dataset-path", jsonl_path,
        "--custom-output-len", str(args.longbench_output_len),
    ]
    return _run_bench_serve(
        port=args.port, model=args.model, extra_args=extra_args,
        result_file=result_file, num_prompts=min(args.num_prompts, 100),
        request_rate=args.request_rate, max_concurrency=max_concurrency,
        warmup=min(args.warmup_requests, 1),
    )


def run_multi_round_workload(
    args: argparse.Namespace,
    mode: str,
    max_concurrency: int,
    result_dir: Path,
) -> dict | None:
    """Emulated multi-turn QA with growing context."""
    all_turn_results = []

    for turn in range(1, args.multi_round_turns + 1):
        turn_input_len = args.multi_round_input_len * turn
        if turn_input_len > args.max_model_len - args.multi_round_output_len:
            print(f"    Skipping turn {turn}: exceeds max-model-len")
            break

        num_prompts = args.multi_round_sessions
        result_file = str(
            result_dir
            / f"results_{mode}_conc{max_concurrency}_turn{turn}.json"
        )
        extra_args = [
            "--dataset-name", "random",
            "--random-input-len", str(turn_input_len),
            "--random-output-len", str(args.multi_round_output_len),
        ]
        print(
            f"\n    --- Turn {turn}/{args.multi_round_turns} "
            f"(input_len={turn_input_len}) ---"
        )

        turn_result = _run_bench_serve(
            port=args.port, model=args.model, extra_args=extra_args,
            result_file=result_file, num_prompts=num_prompts,
            request_rate=args.request_rate, max_concurrency=max_concurrency,
            warmup=1 if turn == 1 else 0,
        )
        if turn_result is not None:
            turn_result["turn"] = turn
            turn_result["turn_input_len"] = turn_input_len
            all_turn_results.append(turn_result)

    if not all_turn_results:
        return None

    aggregated = _aggregate_multi_round(all_turn_results)
    agg_file = str(
        result_dir / f"results_{mode}_conc{max_concurrency}_aggregated.json"
    )
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    return aggregated


def _aggregate_multi_round(turn_results: list[dict]) -> dict:
    """Aggregate per-turn results into a single summary."""
    total_output = sum(r.get("total_output_tokens", 0) for r in turn_results)
    total_duration = sum(r.get("duration", 0) for r in turn_results)
    completed = sum(r.get("completed", 0) for r in turn_results)
    failed = sum(r.get("failed", 0) for r in turn_results)

    def _vals(key):
        return [r[key] for r in turn_results if key in r and r[key] is not None]

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "workload": "multi_round",
        "num_turns": len(turn_results),
        "completed": completed,
        "failed": failed,
        "duration": total_duration,
        "total_output_tokens": total_output,
        "output_throughput": (
            total_output / total_duration if total_duration > 0 else 0
        ),
        "request_throughput": (
            completed / total_duration if total_duration > 0 else 0
        ),
        "mean_ttft_ms": _mean(_vals("mean_ttft_ms")),
        "mean_itl_ms": _mean(_vals("mean_itl_ms")),
        "mean_tpot_ms": _mean(_vals("mean_tpot_ms")),
        "mean_e2el_ms": _mean(_vals("mean_e2el_ms")),
        "p99_itl_ms": _mean(_vals("p99_itl_ms")),
        "p99_ttft_ms": _mean(_vals("p99_ttft_ms")),
        "per_turn": turn_results,
    }


WORKLOAD_RUNNERS = {
    "random": run_random_workload,
    "longbench": run_longbench_workload,
    "multi_round": run_multi_round_workload,
}


# =========================================================================
# Capacity sweep
# =========================================================================

def run_capacity_sweep(
    args: argparse.Namespace,
    mode: str,
    result_dir: Path,
) -> dict:
    """Binary-search for the maximum concurrency that completes without
    failures.  Uses short runs (fewer prompts) for speed."""

    print(f"\n  === Capacity sweep for [{mode}] ===")

    log_file = str(result_dir / f"server_capacity_{mode}.log")
    # Use a high max_num_seqs so the server doesn't cap us
    max_num_seqs = min(args.capacity_max, 1024)

    _cleanup_gpu_and_port(args.port)
    proc = launch_server(args, mode, max_num_seqs=max_num_seqs,
                         log_file=log_file)

    try:
        print(f"  Waiting for server (timeout={args.startup_timeout}s) ...")
        if not _wait_for_server(args.port, args.startup_timeout):
            print(f"  [FAIL] Server failed to start. Check: {log_file}")
            return {"mode": mode, "error": "startup_failed",
                    "max_concurrency": 0}

        # Parse server capacity info from log
        server_info = _parse_server_capacity(log_file)
        print(f"  [OK] Server ready. Info: {server_info}")

        # Sweep concurrency: start low, increase until failure
        max_ok = 0
        first_fail = None
        conc = args.capacity_start
        results_per_conc = {}

        while conc <= args.capacity_max:
            print(f"\n  Trying concurrency={conc} ...")

            # Use a small number of prompts for capacity test
            num_prompts = max(conc * 2, 20)
            result_file = str(
                result_dir / f"capacity_{mode}_conc{conc}.json"
            )
            extra_args = [
                "--dataset-name", "random",
                "--random-input-len", str(args.capacity_input_len),
                "--random-output-len", str(args.capacity_output_len),
            ]

            result = _run_bench_serve(
                port=args.port, model=args.model, extra_args=extra_args,
                result_file=result_file, num_prompts=num_prompts,
                request_rate=float("inf"), max_concurrency=conc,
                warmup=1, timeout=300,
            )

            if result is not None:
                failed = result.get("failed", 0)
                completed = result.get("completed", 0)
                if failed == 0 and completed > 0:
                    max_ok = conc
                    results_per_conc[str(conc)] = result
                    tput = result.get("output_throughput", 0)
                    itl = result.get("mean_itl_ms", 0)
                    print(
                        f"    ✓ conc={conc}: completed={completed}, "
                        f"throughput={tput:.0f} tok/s, ITL={itl:.2f}ms"
                    )
                else:
                    print(
                        f"    ✗ conc={conc}: completed={completed}, "
                        f"failed={failed}"
                    )
                    if first_fail is None:
                        first_fail = conc
                    # Don't break — try a few more in case it's transient
                    if conc > first_fail + args.capacity_step * 2:
                        break
            else:
                print(f"    ✗ conc={conc}: bench serve failed entirely")
                if first_fail is None:
                    first_fail = conc
                if conc > (first_fail or conc) + args.capacity_step * 2:
                    break

            conc += args.capacity_step

        print(f"\n  [{mode}] Max successful concurrency: {max_ok}")
        if first_fail:
            print(f"  [{mode}] First failure at concurrency: {first_fail}")

        return {
            "mode": mode,
            "max_concurrency": max_ok,
            "first_failure": first_fail,
            "server_info": server_info,
            "results_per_conc": results_per_conc,
        }

    finally:
        print(f"  Stopping {mode} server ...")
        _kill_server(proc)
        time.sleep(args.cooldown)


# =========================================================================
# Per-workload evaluation orchestrator
# =========================================================================

def run_workload_evaluation(
    args: argparse.Namespace,
    workload: str,
    mode: str,
    workload_dir: Path,
) -> dict:
    """Run all concurrency levels for a (workload, mode) pair."""
    max_conc = max(args.concurrency)
    log_file = str(workload_dir / f"server_{mode}.log")

    _cleanup_gpu_and_port(args.port)

    print(f"\n  Launching server [{mode}] ...")
    proc = launch_server(args, mode, max_num_seqs=max_conc,
                         log_file=log_file)

    try:
        print(f"  Waiting for server (timeout={args.startup_timeout}s) ...")
        if not _wait_for_server(args.port, args.startup_timeout):
            print(f"  [FAIL] Server failed to start. Check: {log_file}")
            return {
                "mode": mode, "workload": workload,
                "error": "startup_failed",
            }

        server_info = _parse_server_capacity(log_file)
        print(f"  [OK] Server ready. Info: {server_info}")

        results_per_conc = {}
        runner = WORKLOAD_RUNNERS[workload]

        for conc in sorted(args.concurrency):
            print(
                f"\n  --- {workload} / {mode} / concurrency={conc} ---"
            )
            result = runner(args, mode, conc, workload_dir)
            if result is not None:
                results_per_conc[str(conc)] = result
                itl = result.get("mean_itl_ms")
                tpot = result.get("mean_tpot_ms")
                tput = result.get("output_throughput")
                p99 = result.get("p99_itl_ms")
                completed = result.get("completed", "?")
                failed = result.get("failed", "?")
                print(f"    → completed={completed}, failed={failed}", end="")
                if itl is not None:
                    print(f", ITL={itl:.2f}ms", end="")
                if p99 is not None:
                    print(f", P99={p99:.2f}ms", end="")
                if tpot is not None:
                    print(f", TPOT={tpot:.2f}ms", end="")
                if tput is not None:
                    print(f", tput={tput:.0f} tok/s", end="")
                print()
            else:
                print(f"    → [FAIL] No results")
                results_per_conc[str(conc)] = {"error": "failed"}

        return {
            "mode": mode,
            "workload": workload,
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

# Decode-focused metrics (TTFT excluded — it conflates prefill timing)
DECODE_METRICS = [
    ("mean_itl_ms", "Mean ITL (ms)"),
    ("p99_itl_ms", "P99 ITL (ms)"),
    ("mean_tpot_ms", "Mean TPOT (ms)"),
    ("output_throughput", "Output tput (tok/s)"),
    ("request_throughput", "Req tput (req/s)"),
    ("completed", "Completed"),
    ("failed", "Failed"),
]

# Full metrics including TTFT for reference
ALL_METRICS = [
    ("mean_ttft_ms", "Mean TTFT (ms)"),
    ("p99_ttft_ms", "P99 TTFT (ms)"),
] + DECODE_METRICS


def print_decode_comparison(
    workload: str,
    modes: list[str],
    all_results: dict[str, dict],
    concurrency_levels: list[int],
) -> None:
    """Print decode-focused comparison table for a workload."""
    print(f"\n{'=' * 100}")
    print(f"  DECODE COMPARISON: {workload.upper()}")
    print(f"  (All modes perform real prefill; metrics below are decode-only)")
    print(f"{'=' * 100}")

    for conc in sorted(concurrency_levels):
        ckey = str(conc)
        print(f"\n  concurrency = {conc}")

        # Build header
        header = f"  {'Metric':<25s}"
        for m in modes:
            header += f"  {m:>14s}"
        if "baseline" in modes and "pipelined" in modes:
            header += f"  {'Δ pipe/base':>12s}"
        if "offloading" in modes and "pipelined" in modes:
            header += f"  {'Δ pipe/offl':>12s}"
        print(header)
        print(f"  {'─' * 25}", end="")
        for _ in modes:
            print(f"  {'─' * 14}", end="")
        if "baseline" in modes and "pipelined" in modes:
            print(f"  {'─' * 12}", end="")
        if "offloading" in modes and "pipelined" in modes:
            print(f"  {'─' * 12}", end="")
        print()

        for metric_key, metric_label in DECODE_METRICS:
            line = f"  {metric_label:<25s}"
            values = {}
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
            # Delta: pipelined vs offloading
            if "offloading" in modes and "pipelined" in modes:
                ov = values.get("offloading")
                pv = values.get("pipelined")
                if (
                    ov is not None
                    and pv is not None
                    and isinstance(ov, (int, float))
                    and ov != 0
                ):
                    pct = ((pv - ov) / abs(ov)) * 100
                    line += f"  {pct:>+10.1f}%"
                else:
                    line += f"  {'—':>12s}"

            print(line)


def print_capacity_comparison(
    modes: list[str],
    capacity_results: dict[str, dict],
) -> None:
    """Print capacity sweep comparison."""
    print(f"\n{'=' * 80}")
    print(f"  CAPACITY COMPARISON (max concurrent decode requests)")
    print(f"{'=' * 80}")

    header = f"  {'Mode':<15s}  {'Max Conc':>10s}  {'1st Fail':>10s}  {'KV Mem (GiB)':>12s}"
    print(header)
    print(f"  {'─' * 15}  {'─' * 10}  {'─' * 10}  {'─' * 12}")

    baseline_max = None
    for m in modes:
        cr = capacity_results.get(m, {})
        max_c = cr.get("max_concurrency", "N/A")
        first_f = cr.get("first_failure", "N/A")
        kv_mem = cr.get("server_info", {}).get("kv_cache_memory_gib", "N/A")

        if m == "baseline" and isinstance(max_c, (int, float)):
            baseline_max = max_c

        line = f"  {m:<15s}  {str(max_c):>10s}  {str(first_f):>10s}"
        if isinstance(kv_mem, float):
            line += f"  {kv_mem:>12.2f}"
        else:
            line += f"  {str(kv_mem):>12s}"
        print(line)

    # Show improvement ratio
    if baseline_max and baseline_max > 0:
        print()
        for m in modes:
            if m == "baseline":
                continue
            mc = capacity_results.get(m, {}).get("max_concurrency", 0)
            if mc > 0:
                ratio = mc / baseline_max
                print(
                    f"  {m} serves {ratio:.1f}× more concurrent "
                    f"requests than baseline"
                )


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  Decode-Isolated KV Cache Pipelining Evaluation")
    print("  (Apple-to-apple: all modes perform real prefill)")
    print("=" * 80)
    print(f"  Model             : {args.model}")
    print(f"  Max model len     : {args.max_model_len}")
    print(f"  GPU mem util      : {args.gpu_memory_utilization}")
    print(f"  K-window          : {args.k_window}")
    print(f"  Modes             : {args.modes}")
    print(f"  Workloads         : {args.workloads}")
    print(f"  Concurrency       : {args.concurrency}")
    print(f"  Num prompts       : {args.num_prompts}")
    print(f"  Request rate      : {args.request_rate}")
    print(f"  Capacity test     : {args.capacity_test}")
    print(f"  Result dir        : {args.result_dir}")

    # =================================================================
    # Phase 1: Decode latency & throughput comparison
    # =================================================================

    all_summaries: dict[str, dict[str, dict]] = {}

    for workload in args.workloads:
        print(f"\n{'#' * 80}")
        print(f"# WORKLOAD: {workload.upper()}")
        print(f"{'#' * 80}")

        workload_dir = result_dir / workload
        workload_dir.mkdir(parents=True, exist_ok=True)
        all_summaries[workload] = {}

        for mode in args.modes:
            summary = run_workload_evaluation(
                args, workload, mode, workload_dir,
            )
            all_summaries[workload][mode] = summary

        # Print decode-focused comparison
        print_decode_comparison(
            workload, args.modes,
            all_summaries[workload], args.concurrency,
        )

    # =================================================================
    # Phase 2: Capacity sweep
    # =================================================================

    capacity_results: dict[str, dict] = {}
    if args.capacity_test:
        print(f"\n{'#' * 80}")
        print(f"# CAPACITY SWEEP")
        print(f"{'#' * 80}")

        cap_dir = result_dir / "capacity"
        cap_dir.mkdir(parents=True, exist_ok=True)

        for mode in args.modes:
            capacity_results[mode] = run_capacity_sweep(
                args, mode, cap_dir,
            )

        print_capacity_comparison(args.modes, capacity_results)

    # =================================================================
    # Final summary
    # =================================================================

    print(f"\n{'=' * 100}")
    print("  OVERALL SUMMARY (decode-only metrics)")
    print(f"{'=' * 100}")

    # Compact decode metrics table
    ref_conc = str(args.concurrency[0])

    header = f"  {'Workload':<15s}"
    for m in args.modes:
        header += (
            f"  {m + ' ITL':>14s}  {m + ' P99':>12s}"
            f"  {m + ' tput':>12s}"
        )
    print(header)
    print(f"  {'─' * 15}", end="")
    for _ in args.modes:
        print(f"  {'─' * 14}  {'─' * 12}  {'─' * 12}", end="")
    print()

    summary_table: dict[str, dict] = {}
    for workload in args.workloads:
        summary_table[workload] = {}
        line = f"  {workload:<15s}"
        for m in args.modes:
            cr = (all_summaries.get(workload, {})
                  .get(m, {})
                  .get("concurrency_results", {})
                  .get(ref_conc, {}))
            itl = cr.get("mean_itl_ms")
            p99 = cr.get("p99_itl_ms")
            tput = cr.get("output_throughput")
            summary_table[workload][m] = {
                "mean_itl_ms": itl,
                "p99_itl_ms": p99,
                "output_throughput": tput,
            }
            line += f"  {itl:>14.2f}" if itl is not None else f"  {'N/A':>14s}"
            line += f"  {p99:>12.2f}" if p99 is not None else f"  {'N/A':>12s}"
            line += f"  {tput:>12.0f}" if tput is not None else f"  {'N/A':>12s}"
        print(line)

    # Key findings
    if "baseline" in args.modes and "pipelined" in args.modes:
        print(f"\n  Key findings (concurrency={ref_conc}):")
        for workload in args.workloads:
            b = summary_table[workload].get("baseline", {})
            p = summary_table[workload].get("pipelined", {})
            b_itl = b.get("mean_itl_ms")
            p_itl = p.get("mean_itl_ms")
            b_tput = b.get("output_throughput")
            p_tput = p.get("output_throughput")
            if b_itl and p_itl and b_itl > 0:
                overhead = ((p_itl - b_itl) / b_itl) * 100
                print(
                    f"    [{workload}] Decode ITL overhead: {overhead:+.1f}%"
                )
            if b_tput and p_tput and b_tput > 0:
                delta = ((p_tput - b_tput) / b_tput) * 100
                print(
                    f"    [{workload}] Throughput change: {delta:+.1f}%"
                )

    if capacity_results:
        print(f"\n  Capacity results:")
        for m in args.modes:
            mc = capacity_results.get(m, {}).get("max_concurrency", "?")
            print(f"    [{m}] Max concurrent requests: {mc}")

    # =================================================================
    # Save results
    # =================================================================

    final = {
        "config": {
            "model": args.model,
            "max_model_len": args.max_model_len,
            "k_window": args.k_window,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "modes": args.modes,
            "workloads": args.workloads,
            "concurrency": args.concurrency,
            "num_prompts": args.num_prompts,
            "request_rate": str(args.request_rate),
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
        },
        "workload_results": all_summaries,
        "summary_table": summary_table,
    }
    if capacity_results:
        final["capacity_results"] = capacity_results

    summary_path = result_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\n  Full results saved to: {result_dir}/")
    print(f"  Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
