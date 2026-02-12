#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
KV-Cache Pipelining Capacity Test

Measures the max number of concurrent requests each mode can handle
before OOM, demonstrating that pipelined + free-original-kv-cache
allows serving more requests with the same GPU memory.

Approach:
  1. Start a vLLM server (baseline or pipelined).
  2. Ramp up concurrent requests until the server fails / OOMs.
  3. Record the max successful concurrency.
  4. Compare across modes.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KV-pipelining capacity test")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    p.add_argument("--port", type=int, default=8199)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--input-len", type=int, default=512,
                    help="Approx input length in tokens")
    p.add_argument("--output-len", type=int, default=128,
                    help="Max output tokens per request")
    p.add_argument("--start-concurrency", type=int, default=4)
    p.add_argument("--end-concurrency", type=int, default=128)
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--k-window", type=int, default=32)
    p.add_argument("--startup-timeout", type=int, default=300)
    p.add_argument("--result-dir", default="/tmp/kv_pipelining_capacity")
    p.add_argument("--modes", nargs="+",
                    default=["baseline", "pipelined", "pipelined_free"],
                    help="Modes to test")
    return p.parse_args()


def _wait_for_server(port: int, timeout: int) -> bool:
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
    model: str, port: int, max_model_len: int, gpu_mem_util: float,
    max_num_seqs: int, mode: str = "baseline", k_window: int = 32,
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
        "--max-num-seqs", str(max_num_seqs),
    ]
    if mode in ("pipelined", "pipelined_free"):
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
        if mode == "pipelined_free":
            cmd += ["--free-original-kv-cache"]

    log_fh = open(log_file, "w") if log_file else subprocess.DEVNULL
    proc = subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT, preexec_fn=os.setsid,
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


def _get_model_name(port: int) -> str:
    url = f"http://127.0.0.1:{port}/v1/models"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    return data["data"][0]["id"]


def _generate_prompt(target_tokens: int) -> str:
    """Generate a prompt of approximately target_tokens length."""
    # ~4 chars per token as rough estimate
    base = ("The following is a detailed technical analysis of system "
            "performance metrics in large-scale distributed computing "
            "environments. ")
    repeat_count = max(1, (target_tokens * 4) // len(base))
    return (base * repeat_count)[:target_tokens * 4]


def _send_request(port: int, model_name: str, prompt: str,
                   max_tokens: int, timeout: int = 120) -> dict | None:
    """Send a single completion request. Returns result or None on failure."""
    url = f"http://127.0.0.1:{port}/v1/completions"
    body = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        return {
            "success": True,
            "completion_tokens": data.get("usage", {}).get("completion_tokens"),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _test_concurrency(port: int, model_name: str, concurrency: int,
                       input_len: int, output_len: int) -> dict:
    """Send `concurrency` simultaneous requests and report results."""
    prompt = _generate_prompt(input_len)
    results = []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_send_request, port, model_name, prompt, output_len)
            for _ in range(concurrency)
        ]
        for f in as_completed(futures):
            results.append(f.result())
    elapsed = time.monotonic() - t0

    successes = sum(1 for r in results if r and r.get("success"))
    failures = concurrency - successes
    errors = [r.get("error", "unknown") for r in results
              if r and not r.get("success")]

    return {
        "concurrency": concurrency,
        "successes": successes,
        "failures": failures,
        "elapsed_s": round(elapsed, 2),
        "errors": errors[:3],  # keep first 3 error messages
    }


def run_mode(args: argparse.Namespace, mode: str, result_dir: Path) -> dict:
    """Run a full concurrency sweep for a given mode."""
    print(f"\n{'='*70}")
    print(f"MODE: {mode.upper()}")
    print(f"{'='*70}")

    max_conc = args.end_concurrency
    log_file = str(result_dir / f"server_{mode}.log")

    proc = _launch_server(
        model=args.model,
        port=args.port,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_memory_utilization,
        max_num_seqs=max_conc,
        mode=mode,
        k_window=args.k_window,
        log_file=log_file,
    )

    try:
        print(f"  Waiting for server (timeout={args.startup_timeout}s) ...")
        if not _wait_for_server(args.port, args.startup_timeout):
            print(f"  ✗ Server failed to start. Check: {log_file}")
            return {"mode": mode, "max_concurrency": 0, "error": "startup_failed"}

        model_name = _get_model_name(args.port)
        print(f"  ✓ Server ready ({model_name}). Starting sweep ...")

        sweep_results = []
        max_successful = 0

        for conc in range(args.start_concurrency,
                          args.end_concurrency + 1,
                          args.step):
            result = _test_concurrency(
                args.port, model_name, conc,
                args.input_len, args.output_len,
            )
            sweep_results.append(result)

            if result["failures"] == 0:
                max_successful = conc
                print(f"    conc={conc:3d}: ✓ all {result['successes']} OK "
                      f"({result['elapsed_s']}s)")
            else:
                print(f"    conc={conc:3d}: ✗ {result['failures']} failures "
                      f"({result['elapsed_s']}s)")
                if result["errors"]:
                    print(f"             first error: {result['errors'][0][:100]}")
                # If we get failures, try one more step then stop
                if conc > max_successful + args.step:
                    print(f"    Stopping sweep — failures persisting.")
                    break

        summary = {
            "mode": mode,
            "max_successful_concurrency": max_successful,
            "sweep_results": sweep_results,
        }
        with open(result_dir / f"sweep_{mode}.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n  ⟹  Max successful concurrency for {mode}: {max_successful}")
        return summary

    finally:
        print(f"  Stopping {mode} server ...")
        _kill_server(proc)
        time.sleep(5)  # let GPU memory fully release


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("KV-Cache Pipelining Capacity Test")
    print("=" * 70)
    print(f"  Model:        {args.model}")
    print(f"  Input len:    {args.input_len}")
    print(f"  Output len:   {args.output_len}")
    print(f"  Concurrency:  {args.start_concurrency}-{args.end_concurrency} "
          f"(step={args.step})")
    print(f"  K-window:     {args.k_window}")
    print(f"  GPU mem util: {args.gpu_memory_utilization}")
    print(f"  Modes:        {args.modes}")

    all_summaries = {}
    for mode in args.modes:
        summary = run_mode(args, mode, result_dir)
        all_summaries[mode] = summary

    # Final comparison
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for mode, summary in all_summaries.items():
        max_conc = summary.get("max_successful_concurrency", 0)
        print(f"  {mode:20s}: max concurrency = {max_conc}")

    baseline_max = all_summaries.get("baseline", {}).get(
        "max_successful_concurrency", 0)
    for mode in ["pipelined", "pipelined_free"]:
        if mode in all_summaries:
            pipe_max = all_summaries[mode].get(
                "max_successful_concurrency", 0)
            if baseline_max > 0:
                improvement = ((pipe_max - baseline_max) / baseline_max) * 100
                print(f"  {mode} vs baseline: "
                      f"{'+' if improvement > 0 else ''}{improvement:.1f}%")

    with open(result_dir / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nFull results in: {result_dir}/")


if __name__ == "__main__":
    main()
