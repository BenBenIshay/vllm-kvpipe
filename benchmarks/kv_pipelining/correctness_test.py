#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Correctness validation for KV-cache pipelining.

Workflow
--------
1.  Launch a **baseline** vLLM server (no pipelining).
2.  Send a fixed set of prompts with deterministic decoding (temperature=0).
3.  Record generated tokens.
4.  Tear down the server.
5.  Launch a **pipelined** vLLM server (--enable-kv-pipelining, K-window).
6.  Re-send the same prompts.
7.  Compare outputs token-by-token.

Usage
-----
    python correctness_test.py                        # defaults
    python correctness_test.py --model meta-llama/Meta-Llama-3.1-8B \
        --k-window 32 --max-tokens 128 --num-prompts 20

The script manages server lifecycle automatically and prints a clear
PASS / FAIL verdict at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Prompts – diverse enough to exercise different decode lengths / patterns.
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "Write a short Python function that computes the Fibonacci sequence.",
    "Translate the following English sentence to French: 'The quick brown fox jumps over the lazy dog.'",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "What are the main differences between TCP and UDP?",
    "Describe how a transformer neural network processes a sentence.",
    "List the planets in our solar system in order from the sun.",
    "Write a haiku about machine learning.",
    "Explain the concept of KV-cache in large language model inference.",
    "What is the time complexity of merge sort and why?",
    "Describe the water cycle in four steps.",
    "What are three benefits of using asynchronous I/O in systems programming?",
    "Give a brief history of the Python programming language.",
    "Explain what CUDA streams are and why they matter for GPU programming.",
    "Write a SQL query that finds the top 5 customers by total order value.",
    "What is the difference between latency and throughput?",
    "Describe the architecture of a typical LLM serving system.",
    "Explain the CAP theorem in distributed systems.",
    "What is pipelining in computer architecture?",
    "Write a short story about a robot learning to cook.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KV-pipelining correctness test")
    p.add_argument(
        "--model",
        default=os.environ.get("MODEL", "meta-llama/Meta-Llama-3.1-8B"),
        help="HuggingFace model id",
    )
    p.add_argument("--port", type=int, default=8199, help="vLLM server port")
    p.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.75,
        help="GPU memory fraction for vLLM",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens to generate per prompt",
    )
    p.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Number of prompts to use (default: all 20)",
    )
    p.add_argument(
        "--k-window",
        type=int,
        default=32,
        help="K-window size for pipelined mode",
    )
    p.add_argument(
        "--free-original-kv-cache",
        action="store_true",
        default=False,
        help="Also pass --free-original-kv-cache in pipelined mode",
    )
    p.add_argument(
        "--startup-timeout",
        type=int,
        default=300,
        help="Seconds to wait for vLLM server to become ready",
    )
    p.add_argument(
        "--result-dir",
        type=str,
        default="/tmp/kv_pipelining_correctness",
        help="Directory to store result JSON files",
    )
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        default=False,
        help="Skip baseline run and load from a previous result file",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Server lifecycle helpers
# ---------------------------------------------------------------------------

def _wait_for_server(port: int, timeout: int) -> bool:
    """Poll the /health endpoint until the server is ready."""
    import urllib.request
    import urllib.error

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


def _launch_server(
    model: str,
    port: int,
    max_model_len: int,
    gpu_mem_util: float,
    pipelined: bool = False,
    k_window: int = 32,
    free_original: bool = False,
    log_file: str | None = None,
) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--disable-log-requests",
        "--dtype", "auto",
        "--enforce-eager",  # disable cuda graph for determinism during test
    ]
    if pipelined:
        kv_config = {
            "kv_connector": "LayerwisePipelinedConnector",
            "kv_role": "kv_both",
        }
        cmd += [
            "--enable-kv-pipelining",
            "--kv-k-window", str(k_window),
            "--kv-transfer-config", json.dumps(kv_config),
            "--enable-prefix-caching", "false",
        ]
        if free_original:
            cmd += ["--free-original-kv-cache"]

    log_fh = open(log_file, "w") if log_file else subprocess.DEVNULL
    mode_label = "pipelined" if pipelined else "baseline"
    print(f"  → Starting {mode_label} server ...")
    print(f"    cmd: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc


def _kill_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _get_model_name(port: int) -> str:
    """Fetch the actual served model name from /v1/models."""
    import urllib.request
    url = f"http://127.0.0.1:{port}/v1/models"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    return data["data"][0]["id"]


def _query_completions(
    port: int,
    prompts: list[str],
    max_tokens: int,
) -> list[dict]:
    """Send prompts to the OpenAI-compatible /v1/completions endpoint."""
    import urllib.request

    model_name = _get_model_name(port)
    url = f"http://127.0.0.1:{port}/v1/completions"
    results = []
    for i, prompt in enumerate(prompts):
        body = json.dumps({
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
        }).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())

        choice = data["choices"][0]
        results.append({
            "prompt_index": i,
            "prompt": prompt,
            "text": choice["text"],
            "finish_reason": choice["finish_reason"],
            "usage": data.get("usage"),
        })
        comp_tokens = data.get("usage", {}).get("completion_tokens", "?")
        print(f"    prompt {i+1}/{len(prompts)} done "
              f"({choice['finish_reason']}, {comp_tokens} tokens)")
    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _compare_results(
    baseline: list[dict],
    pipelined: list[dict],
) -> tuple[bool, list[str]]:
    """Compare two result sets.  Returns (all_match, list_of_diffs)."""
    diffs: list[str] = []
    if len(baseline) != len(pipelined):
        diffs.append(
            f"Result count mismatch: baseline={len(baseline)}, "
            f"pipelined={len(pipelined)}"
        )
        return False, diffs

    all_match = True
    for i, (b, p) in enumerate(zip(baseline, pipelined)):
        if b["text"] != p["text"]:
            all_match = False
            # Show first divergence point
            b_tokens = b["text"].split()
            p_tokens = p["text"].split()
            first_diff = -1
            for j, (bt, pt) in enumerate(zip(b_tokens, p_tokens)):
                if bt != pt:
                    first_diff = j
                    break
            if first_diff == -1:
                first_diff = min(len(b_tokens), len(p_tokens))
            diffs.append(
                f"Prompt {i}: text mismatch starting at word ~{first_diff}\n"
                f"  baseline ({len(b['text'])} chars): "
                f"{textwrap.shorten(b['text'], 200)}\n"
                f"  pipelined ({len(p['text'])} chars): "
                f"{textwrap.shorten(p['text'], 200)}"
            )
    return all_match, diffs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    prompts = DEFAULT_PROMPTS[: args.num_prompts] if args.num_prompts else DEFAULT_PROMPTS
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("KV-Cache Pipelining Correctness Test")
    print("=" * 70)
    print(f"  Model:            {args.model}")
    print(f"  Max model len:    {args.max_model_len}")
    print(f"  Max tokens:       {args.max_tokens}")
    print(f"  Num prompts:      {len(prompts)}")
    print(f"  K-window:         {args.k_window}")
    print(f"  Free orig KV:     {args.free_original_kv_cache}")
    print(f"  Port:             {args.port}")
    print(f"  GPU mem util:     {args.gpu_memory_utilization}")
    print(f"  Result dir:       {result_dir}")
    print()

    # ---- Phase 1: Baseline ----
    if args.skip_baseline:
        baseline_file = result_dir / "baseline_results.json"
        if not baseline_file.exists():
            print(f"  ✗ --skip-baseline set but {baseline_file} not found.")
            sys.exit(1)
        print("─" * 70)
        print("Phase 1: BASELINE (loaded from previous run)")
        print("─" * 70)
        with open(baseline_file) as f:
            baseline_results = json.load(f)
        print(f"  ✓ Loaded {len(baseline_results)} baseline results\n")
    else:
        print("─" * 70)
        print("Phase 1: BASELINE (no pipelining)")
        print("─" * 70)
        baseline_proc = _launch_server(
            model=args.model,
            port=args.port,
            max_model_len=args.max_model_len,
            gpu_mem_util=args.gpu_memory_utilization,
            pipelined=False,
            log_file=str(result_dir / "server_baseline.log"),
        )
        try:
            print(f"  Waiting for server (timeout={args.startup_timeout}s) ...")
            if not _wait_for_server(args.port, args.startup_timeout):
                print("  ✗ Baseline server failed to start. Check logs:")
                print(f"    {result_dir / 'server_baseline.log'}")
                _kill_server(baseline_proc)
                sys.exit(1)
            print("  ✓ Server ready. Querying completions ...")
            baseline_results = _query_completions(args.port, prompts, args.max_tokens)
        finally:
            print("  Stopping baseline server ...")
            _kill_server(baseline_proc)
            time.sleep(3)  # let GPU memory fully release

        with open(result_dir / "baseline_results.json", "w") as f:
            json.dump(baseline_results, f, indent=2)
        print(f"  ✓ Baseline results saved ({len(baseline_results)} prompts)\n")

    # ---- Phase 2: Pipelined ----
    print("─" * 70)
    print(f"Phase 2: PIPELINED (K={args.k_window}, "
          f"free_original={args.free_original_kv_cache})")
    print("─" * 70)
    pipelined_proc = _launch_server(
        model=args.model,
        port=args.port,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_memory_utilization,
        pipelined=True,
        k_window=args.k_window,
        free_original=args.free_original_kv_cache,
        log_file=str(result_dir / "server_pipelined.log"),
    )
    try:
        print(f"  Waiting for server (timeout={args.startup_timeout}s) ...")
        if not _wait_for_server(args.port, args.startup_timeout):
            print("  ✗ Pipelined server failed to start. Check logs:")
            print(f"    {result_dir / 'server_pipelined.log'}")
            _kill_server(pipelined_proc)
            sys.exit(1)
        print("  ✓ Server ready. Querying completions ...")
        pipelined_results = _query_completions(args.port, prompts, args.max_tokens)
    finally:
        print("  Stopping pipelined server ...")
        _kill_server(pipelined_proc)

    with open(result_dir / "pipelined_results.json", "w") as f:
        json.dump(pipelined_results, f, indent=2)
    print(f"  ✓ Pipelined results saved ({len(pipelined_results)} prompts)\n")

    # ---- Phase 3: Comparison ----
    print("─" * 70)
    print("Phase 3: COMPARISON")
    print("─" * 70)
    all_match, diffs = _compare_results(baseline_results, pipelined_results)

    summary = {
        "model": args.model,
        "k_window": args.k_window,
        "free_original_kv_cache": args.free_original_kv_cache,
        "max_tokens": args.max_tokens,
        "num_prompts": len(prompts),
        "all_match": all_match,
        "num_mismatches": len(diffs),
        "diffs": diffs,
    }
    with open(result_dir / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    if all_match:
        print("  ✅ PASS — All outputs match between baseline and pipelined mode!")
    else:
        print(f"  ❌ FAIL — {len(diffs)} mismatch(es) detected:")
        for d in diffs:
            print(f"    {d}")

    print()
    print(f"Full results in: {result_dir}/")
    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
