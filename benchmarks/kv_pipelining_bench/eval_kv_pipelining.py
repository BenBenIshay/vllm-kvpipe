#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
KV Cache Pipelining — Comprehensive Evaluation Benchmark
=========================================================

End-to-end evaluation of layer-wise KV-cache pipelining against the standard
vLLM decode baseline.  The benchmark launches vLLM servers in different
configurations, sends realistic workloads via ``vllm bench serve``, and
produces a structured comparison of decode-only performance metrics.

Modes (server configurations)
-----------------------------
  baseline     Standard vLLM decode-only inference via DecodeBenchConnector
               (all KV on GPU, dummy-fill to simulate P/D disaggregation).
  pipelined    Real prefill + layer-wise CPU→GPU pipelined decode via
               PrefillStagingPipelinedConnector.  Only k_window layers of
               KV reside on GPU at any moment.

Workloads (client request patterns)
------------------------------------
  random       Synthetic prompts of fixed token length.  Equivalent to
               ``vllm bench serve --dataset-name random``.
  sharegpt     Real multi-turn QA conversations from the ShareGPT dataset.
               Captures realistic prompt/output length distributions.
  longbench    Long-context QA from the LongBench HuggingFace dataset.
               Tests performance with 2k–8k+ token prompts.
  multi_round  Emulated multi-round chat: the benchmark issues multiple
               sequential turns per "conversation", reusing the chat
               history.  Tests KV re-use and decode under growing context.

Metrics (decode-focused)
------------------------
  - TTFT  — time to first token (ms)
  - TPOT  — time per output token excluding first (ms)
  - ITL   — inter-token latency, mean + p99 (ms)
  - E2EL  — end-to-end request latency (ms)
  - Output throughput (tok/s)
  - Request throughput (req/s)
  - Max concurrency before failure

Usage
-----
    # Quick random workload test:
    python benchmarks/kv_pipelining_bench/eval_kv_pipelining.py \\
        --model meta-llama/Meta-Llama-3.1-8B \\
        --workloads random \\
        --modes baseline pipelined \\
        --k-window 4

    # Full evaluation with all workloads:
    python benchmarks/kv_pipelining_bench/eval_kv_pipelining.py \\
        --model meta-llama/Meta-Llama-3.1-8B \\
        --workloads random sharegpt longbench multi_round \\
        --modes baseline pipelined \\
        --k-window 4 \\
        --concurrency 1 4 16 64

    # ShareGPT with local dataset file:
    python benchmarks/kv_pipelining_bench/eval_kv_pipelining.py \\
        --model meta-llama/Meta-Llama-3.1-8B \\
        --workloads sharegpt \\
        --sharegpt-path /data/ShareGPT_V3_unfiltered.json \\
        --modes baseline pipelined

Output
------
  Results are saved to ``--result-dir`` (default /tmp/kv_pipelining_eval/):
    <result-dir>/
      <workload>/
        results_<mode>.json     — raw vllm bench serve output
        server_<mode>.log       — server stdout/stderr
      summary.json              — aggregated cross-mode comparison
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
SHAREGPT_HF_DATASET = "ShareGPT/ShareGPT_V3_unfiltered_cleaned_split"
LONGBENCH_HF_DATASET = "THUDM/LongBench"


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KV cache pipelining comprehensive evaluation",
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
                   help="Number of KV layers kept on GPU during decode")

    # Evaluation scope
    p.add_argument(
        "--workloads", nargs="+",
        default=["random", "sharegpt", "longbench", "multi_round"],
        choices=["random", "sharegpt", "longbench", "multi_round"],
        help="Workloads to evaluate",
    )
    p.add_argument(
        "--modes", nargs="+", default=["baseline", "pipelined"],
        choices=["baseline", "pipelined"],
        help="Server modes to compare",
    )

    # Workload params
    p.add_argument("--num-prompts", type=int, default=200,
                   help="Number of requests per workload run")
    p.add_argument("--concurrency", type=int, nargs="+",
                   default=[1, 4, 16, 64],
                   help="Max concurrency levels to test")
    p.add_argument("--request-rate", type=float, default=float("inf"),
                   help="Requests/s (inf = send all immediately)")

    # Random workload
    p.add_argument("--random-input-len", type=int, default=512)
    p.add_argument("--random-output-len", type=int, default=128)

    # ShareGPT workload
    p.add_argument("--sharegpt-path", default=None,
                   help="Path to ShareGPT JSON (auto-downloads if omitted)")
    p.add_argument("--sharegpt-output-len", type=int, default=None,
                   help="Override output length for ShareGPT")

    # LongBench workload
    p.add_argument("--longbench-min-input", type=int, default=2000,
                   help="Minimum input tokens for LongBench samples")
    p.add_argument("--longbench-max-input", type=int, default=None,
                   help="Maximum input tokens (default: --max-model-len - 256)")
    p.add_argument("--longbench-output-len", type=int, default=128)

    # Multi-round workload
    p.add_argument("--multi-round-turns", type=int, default=5,
                   help="Number of conversation turns per session")
    p.add_argument("--multi-round-sessions", type=int, default=20,
                   help="Number of concurrent conversation sessions")
    p.add_argument("--multi-round-input-len", type=int, default=256,
                   help="Approximate tokens per user turn")
    p.add_argument("--multi-round-output-len", type=int, default=128,
                   help="Max output tokens per turn")

    # Timing
    p.add_argument("--startup-timeout", type=int, default=600,
                   help="Server startup timeout in seconds")
    p.add_argument("--warmup-requests", type=int, default=3)
    p.add_argument("--cooldown", type=int, default=8,
                   help="Seconds to wait between server restarts")

    # Correctness
    p.add_argument("--correctness", action="store_true",
                   help="Run correctness validation (compare outputs at T=0)")
    p.add_argument("--correctness-samples", type=int, default=5,
                   help="Number of prompts for correctness check")

    # Output
    p.add_argument("--result-dir", default="/tmp/kv_pipelining_eval")

    return p.parse_args()


# =========================================================================
# Server management
# =========================================================================

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

    if mode == "baseline":
        kv_config = {
            "kv_connector": "DecodeBenchConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "fill_mean": 0.015,
                "fill_std": 0.0,
            },
        }
        cmd += ["--kv-transfer-config", json.dumps(kv_config)]

    elif mode == "pipelined":
        kv_config = {
            "kv_connector": "PrefillStagingPipelinedConnector",
            "kv_role": "kv_both",
        }
        cmd += [
            "--enable-kv-pipelining",
            "--kv-k-window", str(args.k_window),
            "--free-original-kv-cache",
            "--kv-transfer-config", json.dumps(kv_config),
            "--enable-prefix-caching", "false",
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    log_fh = open(log_file, "w")
    proc = subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT, preexec_fn=os.setsid,
    )
    return proc


# =========================================================================
# Client: wrappers around ``vllm bench serve``
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
) -> dict | None:
    """
    Run ``vllm bench serve`` as a subprocess and parse the JSON result.
    """
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
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min max
    )

    if result.returncode != 0:
        print(f"    [ERROR] bench serve failed (rc={result.returncode})")
        if result.stderr:
            # Print last 20 lines of stderr
            lines = result.stderr.strip().split("\n")
            for line in lines[-20:]:
                print(f"      {line}")
        return None

    # Print the benchmark output
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"      {line}")

    # Load the saved result JSON
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
    """Random synthetic workload using vLLM's built-in random dataset."""
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


def run_sharegpt_workload(
    args: argparse.Namespace,
    mode: str,
    max_concurrency: int,
    result_dir: Path,
) -> dict | None:
    """ShareGPT workload — real multi-turn conversations."""
    result_file = str(result_dir / f"results_{mode}_conc{max_concurrency}.json")
    extra_args = [
        "--dataset-name", "sharegpt",
    ]
    if args.sharegpt_path:
        extra_args += ["--dataset-path", args.sharegpt_path]
    else:
        extra_args += ["--dataset-path", SHAREGPT_HF_DATASET]
    if args.sharegpt_output_len is not None:
        extra_args += ["--sharegpt-output-len", str(args.sharegpt_output_len)]
    return _run_bench_serve(
        port=args.port, model=args.model, extra_args=extra_args,
        result_file=result_file, num_prompts=args.num_prompts,
        request_rate=args.request_rate, max_concurrency=max_concurrency,
        warmup=args.warmup_requests,
    )


def _prepare_longbench_jsonl(args: argparse.Namespace, result_dir: Path) -> str:
    """
    Download LongBench from HuggingFace, filter samples by token length,
    and write a JSONL file compatible with ``--dataset-name custom``.

    Returns path to the generated JSONL file.
    """
    jsonl_path = str(result_dir / "longbench_samples.jsonl")
    if os.path.exists(jsonl_path):
        print(f"    Using cached LongBench JSONL: {jsonl_path}")
        return jsonl_path

    print(f"    Downloading LongBench dataset from HuggingFace ...")
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "The 'datasets' package is required for LongBench. "
            "Install it: pip install datasets"
        )

    # LongBench subsets with long contexts suitable for our evaluation
    subsets = ["qasper", "multifieldqa_en", "narrativeqa", "2wikimqa"]
    max_input = args.longbench_max_input or (args.max_model_len - 256)
    min_input = args.longbench_min_input

    samples = []
    for subset in subsets:
        try:
            ds = load_dataset("THUDM/LongBench", subset, split="test",
                              trust_remote_code=True)
        except Exception as e:
            print(f"    Warning: could not load LongBench/{subset}: {e}")
            continue

        for row in ds:
            context = row.get("context", "")
            question = row.get("input", "")
            prompt = (
                f"Read the following passage and answer the question.\n\n"
                f"Passage:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )
            # Rough token estimate (will be properly counted by tokenizer)
            approx_tokens = len(prompt.split())
            if min_input <= approx_tokens <= max_input:
                samples.append({
                    "prompt": prompt,
                    "output_tokens": args.longbench_output_len,
                })
        print(f"    Loaded {subset}: {len(samples)} total samples so far")

    if not samples:
        print("    Warning: no LongBench samples matched length filters")
        # Fall back to any available sample
        return jsonl_path

    # Write JSONL
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
    """
    LongBench workload — long-context QA.

    Downloads THUDM/LongBench from HuggingFace, converts to JSONL, and
    uses ``--dataset-name custom`` to benchmark.  Contexts range from
    2k–8k+ tokens.
    """
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
    """
    Multi-round conversation workload.

    Emulates multi-turn chat: for each session, the benchmark issues
    sequential turns.  Each turn builds on the previous conversation context,
    so the model sees growing prompt length (simulating KV reuse in chat).

    We use vLLM's standard random dataset but structure the calls as
    sequential turn batches.  Each "turn" is a separate bench run with
    increasing input lengths.
    """
    all_turn_results = []

    for turn in range(1, args.multi_round_turns + 1):
        # Input length grows with each turn (simulating chat history)
        turn_input_len = args.multi_round_input_len * turn
        if turn_input_len > args.max_model_len - args.multi_round_output_len:
            print(f"    Skipping turn {turn}: input would exceed max-model-len")
            break

        num_prompts = args.multi_round_sessions
        result_file = str(
            result_dir / f"results_{mode}_conc{max_concurrency}_turn{turn}.json"
        )
        extra_args = [
            "--dataset-name", "random",
            "--random-input-len", str(turn_input_len),
            "--random-output-len", str(args.multi_round_output_len),
        ]
        print(f"\n    --- Turn {turn}/{args.multi_round_turns} "
              f"(input_len={turn_input_len}) ---")

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

    # Aggregate multi-round results
    aggregated = _aggregate_multi_round(all_turn_results)
    agg_file = str(
        result_dir / f"results_{mode}_conc{max_concurrency}_aggregated.json"
    )
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    return aggregated


def _aggregate_multi_round(turn_results: list[dict]) -> dict:
    """Aggregate per-turn results into a single summary."""
    total_input = sum(r.get("total_input_tokens", 0) for r in turn_results)
    total_output = sum(r.get("total_output_tokens", 0) for r in turn_results)
    total_duration = sum(r.get("duration", 0) for r in turn_results)
    completed = sum(r.get("completed", 0) for r in turn_results)
    failed = sum(r.get("failed", 0) for r in turn_results)

    # Weighted average of per-turn metrics
    ttft_vals = [r.get("mean_ttft_ms", 0) for r in turn_results if "mean_ttft_ms" in r]
    itl_vals = [r.get("mean_itl_ms", 0) for r in turn_results if "mean_itl_ms" in r]
    tpot_vals = [r.get("mean_tpot_ms", 0) for r in turn_results if "mean_tpot_ms" in r]
    e2el_vals = [r.get("mean_e2el_ms", 0) for r in turn_results if "mean_e2el_ms" in r]

    p99_itl_vals = [
        r.get("p99_itl_ms", 0) for r in turn_results if "p99_itl_ms" in r
    ]
    p99_ttft_vals = [
        r.get("p99_ttft_ms", 0) for r in turn_results if "p99_ttft_ms" in r
    ]

    def _safe_mean(vals: list) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "workload": "multi_round",
        "num_turns": len(turn_results),
        "completed": completed,
        "failed": failed,
        "duration": total_duration,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "output_throughput": total_output / total_duration if total_duration > 0 else 0,
        "request_throughput": completed / total_duration if total_duration > 0 else 0,
        "mean_ttft_ms": _safe_mean(ttft_vals),
        "mean_itl_ms": _safe_mean(itl_vals),
        "mean_tpot_ms": _safe_mean(tpot_vals),
        "mean_e2el_ms": _safe_mean(e2el_vals),
        "p99_itl_ms": _safe_mean(p99_itl_vals),
        "p99_ttft_ms": _safe_mean(p99_ttft_vals),
        "per_turn": turn_results,
    }


WORKLOAD_RUNNERS = {
    "random": run_random_workload,
    "sharegpt": run_sharegpt_workload,
    "longbench": run_longbench_workload,
    "multi_round": run_multi_round_workload,
}


# =========================================================================
# Correctness validation
# =========================================================================

def _send_completion(port: int, model: str, prompt: str,
                     max_tokens: int) -> str | None:
    """Send a non-streaming completion and return the generated text."""
    url = f"http://127.0.0.1:{port}/v1/completions"
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["text"]
    except Exception as e:
        print(f"    [ERROR] Completion failed: {e}")
        return None


def run_correctness_check(
    args: argparse.Namespace,
    result_dir: Path,
    num_samples: int = 5,
) -> dict:
    """
    Validate that pipelined mode produces identical output to baseline.

    Launches both servers sequentially, sends the same prompts at
    temperature=0, and compares outputs token-by-token.
    """
    print(f"\n{'#' * 80}")
    print(f"# CORRECTNESS VALIDATION")
    print(f"{'#' * 80}")

    # Generate a set of fixed test prompts
    prompts = [
        "The capital of France is",
        "In a galaxy far, far away, there lived a",
        "The quickest way to sort an array in Python is to",
        "According to the laws of thermodynamics,",
        "Once upon a time in a land of endless mountains,",
    ][:num_samples]
    max_tokens = 50

    results = {"prompts": [], "match": True, "details": []}

    for mode in ["baseline", "pipelined"]:
        log_file = str(result_dir / f"server_correctness_{mode}.log")
        print(f"\n  Launching {mode} server for correctness check ...")
        proc = launch_server(args, mode, max_num_seqs=4, log_file=log_file)

        try:
            if not _wait_for_server(args.port, args.startup_timeout):
                print(f"  [FAIL] {mode} server failed to start")
                results["error"] = f"{mode}_startup_failed"
                return results
            print(f"  [OK] {mode} server ready")

            # Send each prompt and record output
            for i, prompt in enumerate(prompts):
                output = _send_completion(args.port, args.model, prompt,
                                          max_tokens)
                if len(results["prompts"]) <= i:
                    results["prompts"].append({
                        "prompt": prompt,
                        "outputs": {},
                    })
                results["prompts"][i]["outputs"][mode] = output
        finally:
            _kill_server(proc)
            time.sleep(args.cooldown)

    # Compare outputs
    print(f"\n  Comparing outputs:")
    all_match = True
    for i, entry in enumerate(results["prompts"]):
        baseline_out = entry["outputs"].get("baseline", "")
        pipelined_out = entry["outputs"].get("pipelined", "")
        match = (baseline_out == pipelined_out)
        if not match:
            all_match = False
        status = "MATCH" if match else "MISMATCH"
        detail = {
            "prompt_idx": i,
            "match": match,
            "baseline_output": baseline_out[:200] if baseline_out else None,
            "pipelined_output": pipelined_out[:200] if pipelined_out else None,
        }
        results["details"].append(detail)
        prompt_preview = entry["prompt"][:60]
        print(f"    [{status}] prompt {i}: \"{prompt_preview}...\"")
        if not match:
            print(f"      baseline : {(baseline_out or '')[:100]}")
            print(f"      pipelined: {(pipelined_out or '')[:100]}")

    results["match"] = all_match
    results["num_matched"] = sum(1 for d in results["details"] if d["match"])
    results["num_total"] = len(results["details"])

    if all_match:
        print(f"\n  ✓ All {len(prompts)} outputs match!")
    else:
        matched = results["num_matched"]
        print(f"\n  ✗ {matched}/{len(prompts)} outputs match")

    # Save results
    with open(result_dir / "correctness_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# =========================================================================
# Orchestrator: per workload × mode × concurrency
# =========================================================================

def run_evaluation(
    args: argparse.Namespace,
    workload: str,
    mode: str,
    workload_dir: Path,
) -> dict:
    """Run all concurrency levels for a given (workload, mode) pair."""
    max_conc = max(args.concurrency)
    log_file = str(workload_dir / f"server_{mode}.log")

    print(f"\n  Launching server [{mode}] ...")
    proc = launch_server(args, mode, max_num_seqs=max_conc, log_file=log_file)

    try:
        print(f"  Waiting for server (timeout={args.startup_timeout}s) ...")
        if not _wait_for_server(args.port, args.startup_timeout):
            print(f"  [FAIL] Server failed to start. Check: {log_file}")
            return {"mode": mode, "workload": workload, "error": "startup_failed"}

        print(f"  [OK] Server ready")

        results_per_conc = {}
        runner = WORKLOAD_RUNNERS[workload]

        for conc in sorted(args.concurrency):
            print(f"\n  --- {workload} / {mode} / concurrency={conc} ---")
            result = runner(args, mode, conc, workload_dir)
            if result is not None:
                results_per_conc[str(conc)] = result
                # Print quick summary
                ttft = result.get("mean_ttft_ms")
                itl = result.get("mean_itl_ms")
                tput = result.get("output_throughput")
                completed = result.get("completed", "?")
                failed = result.get("failed", "?")
                print(f"    -> completed={completed}, failed={failed}", end="")
                if ttft is not None:
                    print(f", TTFT={ttft:.1f}ms", end="")
                if itl is not None:
                    print(f", ITL={itl:.2f}ms", end="")
                if tput is not None:
                    print(f", throughput={tput:.1f} tok/s", end="")
                print()
            else:
                print(f"    -> [FAIL] No results")
                results_per_conc[str(conc)] = {"error": "bench_serve_failed"}

        return {
            "mode": mode,
            "workload": workload,
            "concurrency_results": results_per_conc,
        }
    finally:
        print(f"  Stopping {mode} server ...")
        _kill_server(proc)
        time.sleep(args.cooldown)


# =========================================================================
# Comparison report
# =========================================================================

METRIC_LABELS = [
    ("mean_ttft_ms", "Mean TTFT (ms)"),
    ("p99_ttft_ms", "P99 TTFT (ms)"),
    ("mean_itl_ms", "Mean ITL (ms)"),
    ("p99_itl_ms", "P99 ITL (ms)"),
    ("mean_tpot_ms", "Mean TPOT (ms)"),
    ("mean_e2el_ms", "Mean E2EL (ms)"),
    ("output_throughput", "Output tput (tok/s)"),
    ("request_throughput", "Req tput (req/s)"),
    ("completed", "Completed"),
    ("failed", "Failed"),
]


def print_comparison(
    workload: str,
    modes: list[str],
    all_results: dict[str, dict],
    concurrency_levels: list[int],
) -> None:
    """Print a per-concurrency comparison table for a workload."""
    print(f"\n{'=' * 80}")
    print(f"  COMPARISON: {workload.upper()}")
    print(f"{'=' * 80}")

    for conc in sorted(concurrency_levels):
        conc_key = str(conc)
        print(f"\n  concurrency = {conc}")
        header = f"  {'metric':<25s}"
        for m in modes:
            header += f"  {m:>15s}"
        if len(modes) == 2:
            header += f"  {'delta':>12s}"
        print(header)
        print(f"  {'-' * 25}", end="")
        for _ in modes:
            print(f"  {'-' * 15}", end="")
        if len(modes) == 2:
            print(f"  {'-' * 12}", end="")
        print()

        for metric_key, metric_label in METRIC_LABELS:
            line = f"  {metric_label:<25s}"
            values = []
            for m in modes:
                r = all_results.get(m, {}).get("concurrency_results", {})
                cr = r.get(conc_key, {})
                val = cr.get(metric_key)
                values.append(val)
                if val is not None:
                    if isinstance(val, float):
                        line += f"  {val:>15.2f}"
                    else:
                        line += f"  {val:>15}"
                else:
                    line += f"  {'N/A':>15s}"

            # Compute delta (pipelined vs baseline)
            if len(modes) == 2 and all(v is not None for v in values):
                base, pipe = values
                if isinstance(base, (int, float)) and base != 0:
                    pct = ((pipe - base) / abs(base)) * 100
                    line += f"  {pct:>+10.1f}%"
                else:
                    line += f"  {'—':>12s}"
            elif len(modes) == 2:
                line += f"  {'—':>12s}"

            print(line)


def print_multi_round_detail(
    modes: list[str],
    all_results: dict[str, dict],
    conc: int,
) -> None:
    """Print per-turn breakdown for multi_round workload."""
    conc_key = str(conc)
    print(f"\n  Per-turn breakdown (concurrency={conc}):")

    # Find max turns across modes
    max_turns = 0
    for m in modes:
        cr = all_results.get(m, {}).get("concurrency_results", {}).get(conc_key, {})
        turns = cr.get("per_turn", [])
        max_turns = max(max_turns, len(turns))

    if max_turns == 0:
        return

    header = f"    {'turn':<6s}  {'input_len':<10s}"
    for m in modes:
        header += f"  {m + ' TTFT':>14s}  {m + ' ITL':>12s}"
    print(header)

    for t_idx in range(max_turns):
        vals = {}
        input_len = "?"
        for m in modes:
            cr = all_results.get(m, {}).get("concurrency_results", {}).get(conc_key, {})
            turns = cr.get("per_turn", [])
            if t_idx < len(turns):
                vals[m] = turns[t_idx]
                input_len = str(turns[t_idx].get("turn_input_len", "?"))
        line = f"    {t_idx + 1:<6d}  {input_len:<10s}"
        for m in modes:
            tr = vals.get(m, {})
            ttft = tr.get("mean_ttft_ms")
            itl = tr.get("mean_itl_ms")
            line += f"  {ttft:>14.1f}" if ttft is not None else f"  {'N/A':>14s}"
            line += f"  {itl:>12.2f}" if itl is not None else f"  {'N/A':>12s}"
        print(line)


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  KV Cache Pipelining — Comprehensive Evaluation")
    print("=" * 80)
    print(f"  Model           : {args.model}")
    print(f"  Max model len   : {args.max_model_len}")
    print(f"  GPU mem util    : {args.gpu_memory_utilization}")
    print(f"  K-window        : {args.k_window}")
    print(f"  Modes           : {args.modes}")
    print(f"  Workloads       : {args.workloads}")
    print(f"  Concurrency     : {args.concurrency}")
    print(f"  Num prompts     : {args.num_prompts}")
    print(f"  Request rate    : {args.request_rate}")
    print(f"  Result dir      : {args.result_dir}")

    # ---- Correctness validation (optional, runs first) ----
    correctness_results = None
    if args.correctness:
        correctness_dir = result_dir / "correctness"
        correctness_dir.mkdir(parents=True, exist_ok=True)
        correctness_results = run_correctness_check(
            args, correctness_dir, args.correctness_samples,
        )

    all_summaries: dict[str, dict[str, dict]] = {}
    # all_summaries[workload][mode] = { mode, workload, concurrency_results }

    for workload in args.workloads:
        print(f"\n{'#' * 80}")
        print(f"# WORKLOAD: {workload.upper()}")
        print(f"{'#' * 80}")

        workload_dir = result_dir / workload
        workload_dir.mkdir(parents=True, exist_ok=True)
        all_summaries[workload] = {}

        for mode in args.modes:
            summary = run_evaluation(args, workload, mode, workload_dir)
            all_summaries[workload][mode] = summary

        # Print comparison table for this workload
        print_comparison(
            workload, args.modes,
            all_summaries[workload], args.concurrency,
        )

        # Special multi-round per-turn detail
        if workload == "multi_round":
            for conc in args.concurrency:
                print_multi_round_detail(
                    args.modes, all_summaries[workload], conc,
                )

    # ---- Final summary across all workloads ----
    print(f"\n{'=' * 80}")
    print("  OVERALL SUMMARY")
    print(f"{'=' * 80}")

    # For single-concurrency summary, use the first concurrency level
    ref_conc = str(args.concurrency[0])

    summary_table: dict[str, dict] = {}
    for workload in args.workloads:
        summary_table[workload] = {}
        for mode in args.modes:
            cr = (all_summaries.get(workload, {})
                  .get(mode, {})
                  .get("concurrency_results", {})
                  .get(ref_conc, {}))
            summary_table[workload][mode] = {
                "mean_ttft_ms": cr.get("mean_ttft_ms"),
                "mean_itl_ms": cr.get("mean_itl_ms"),
                "output_throughput": cr.get("output_throughput"),
                "completed": cr.get("completed"),
                "failed": cr.get("failed"),
            }

    # Print compact table
    header = f"  {'workload':<15s}"
    for m in args.modes:
        header += f"  {m + ' TTFT':>14s}  {m + ' ITL':>12s}  {m + ' tput':>12s}"
    print(header)

    for workload in args.workloads:
        line = f"  {workload:<15s}"
        for m in args.modes:
            s = summary_table[workload].get(m, {})
            ttft = s.get("mean_ttft_ms")
            itl = s.get("mean_itl_ms")
            tput = s.get("output_throughput")
            line += f"  {ttft:>14.1f}" if ttft is not None else f"  {'N/A':>14s}"
            line += f"  {itl:>12.2f}" if itl is not None else f"  {'N/A':>12s}"
            line += f"  {tput:>12.1f}" if tput is not None else f"  {'N/A':>12s}"
        print(line)

    # Key findings
    if len(args.modes) == 2 and "baseline" in args.modes and "pipelined" in args.modes:
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
                print(f"    [{workload}] ITL overhead: {overhead:+.1f}%")
            if b_tput and p_tput and b_tput > 0:
                delta = ((p_tput - b_tput) / b_tput) * 100
                print(f"    [{workload}] Throughput delta: {delta:+.1f}%")

    # Save everything
    final_output = {
        "config": {
            "model": args.model,
            "max_model_len": args.max_model_len,
            "k_window": args.k_window,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "modes": args.modes,
            "workloads": args.workloads,
            "concurrency": args.concurrency,
            "num_prompts": args.num_prompts,
            "request_rate": args.request_rate,
        },
        "results": all_summaries,
        "summary_table": summary_table,
    }
    if correctness_results is not None:
        final_output["correctness"] = {
            "all_match": correctness_results.get("match", False),
            "num_matched": correctness_results.get("num_matched", 0),
            "num_total": correctness_results.get("num_total", 0),
        }
    summary_path = result_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(final_output, f, indent=2, default=str)
    print(f"\n  Full results saved to: {result_dir}/")
    print(f"  Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
