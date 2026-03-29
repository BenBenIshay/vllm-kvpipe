#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Decode-Only P/D Pipelining Benchmark
=====================================

Self-contained performance evaluation of layer-wise KV cache pipelining in a
decode-only setting.  No remote prefill node or P/D transfer stack required.

The benchmark works in two phases per request:
  Phase 1 (seed): Run the prompt through the model.  The
      PipelinedDecodeBenchConnector fills KV caches on GPU with dummy
      values, stages them to CPU pinned memory, and frees the per-layer
      GPU tensors -- exactly the state a real P/D decode node would be in
      after receiving KV from a prefill node.
  Phase 2 (decode): Generate output tokens using layer-wise CPU->GPU
      pipelining through window buffers.  Only k_window layers reside
      on GPU at any moment.

Modes
-----
  baseline       : Standard vLLM decode (all KV on GPU via DecodeBenchConnector).
  pipelined      : Decode-only pipelining via PipelinedDecodeBenchConnector
                   (dummy KV fill → CPU → pipelined decode).
  real_prefill   : Real model prefill + pipelined decode via
                   PrefillStagingPipelinedConnector.  The model runs actual
                   attention during prefill.  KV is staged to CPU after each
                   layer, then decode uses layer-wise CPU→GPU pipelining.
                   Suitable for correctness validation and realistic latency
                   measurement on academic datasets (LongBench, ShareGPT, etc).

K sweep (optimal ``kv-k-window``)
---------------------------------
  For **dummy-KV pipelined decode only**, use
  ``benchmarks/kv_pipelining_bench/find_optimal_k.py`` (wraps
  ``pipelined_decode_fast_bench``), which sweeps **decreasing K** and writes
  ``optimal_k_report.json``. Use ``--lite`` on a small GPU or tight time budget.

Metrics
-------
  - Time-to-first-token (TTFT)
  - Inter-token latency (ITL) -- mean and p99
  - End-to-end latency
  - Throughput (tokens/s)
  - Max successful concurrency before OOM

Usage
-----
    # Dummy-fill benchmark (quick, measures decode pipelining overhead):
    python benchmarks/kv_pipelining_bench/decode_only_bench.py \
        --model meta-llama/Meta-Llama-3.1-8B \
        --k-window 4 \
        --input-len 512 \
        --output-len 128 \
        --concurrency 1 4 16 64 \
        --modes baseline pipelined

    # Real-prefill benchmark (for paper; correctness + realistic latency):
    python benchmarks/kv_pipelining_bench/decode_only_bench.py \
        --model meta-llama/Meta-Llama-3.1-8B \
        --k-window 4 \
        --input-len 512 \
        --output-len 128 \
        --concurrency 1 4 16 64 \
        --modes baseline real_prefill

    # All three modes:
    python benchmarks/kv_pipelining_bench/decode_only_bench.py \
        --model meta-llama/Meta-Llama-3.1-8B \
        --k-window 4 \
        --input-len 512 \
        --output-len 128 \
        --concurrency 1 4 16 64 \
        --modes baseline pipelined real_prefill
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decode-only P/D pipelining benchmark",
    )
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    p.add_argument("--port", type=int, default=8199)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument(
        "--input-len", type=int, default=512,
        help="Prompt length in tokens (KV filled via dummy values or real prefill)",
    )
    p.add_argument(
        "--output-len", type=int, default=128,
        help="Max output tokens per request",
    )
    p.add_argument(
        "--concurrency", type=int, nargs="+", default=[1, 4, 16, 64],
        help="Concurrency levels to test",
    )
    p.add_argument("--k-window", type=int, default=4)
    p.add_argument("--startup-timeout", type=int, default=300)
    p.add_argument("--request-timeout", type=int, default=180)
    p.add_argument("--result-dir", default="/tmp/kv_pipelining_decode_bench")
    p.add_argument(
        "--modes", nargs="+", default=["baseline", "pipelined"],
        help="Modes to test (baseline, pipelined, real_prefill)",
    )
    p.add_argument(
        "--dataset", default="random",
        choices=["random", "sharegpt", "longbench"],
        help="Dataset for real_prefill mode (random generates synthetic prompts)",
    )
    p.add_argument(
        "--dataset-path", default=None,
        help="Path to dataset file (for sharegpt/longbench modes)",
    )
    p.add_argument(
        "--warmup-requests", type=int, default=2,
        help="Number of warmup requests before measurement",
    )
    return p.parse_args()


# =========================================================================
# Server management
# =========================================================================

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
    model: str,
    port: int,
    max_model_len: int,
    gpu_mem_util: float,
    max_num_seqs: int,
    mode: str,
    k_window: int,
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

    if mode == "baseline":
        # Standard decode bench: fills KV on GPU, no pipelining.
        kv_config = {
            "kv_connector": "DecodeBenchConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"fill_mean": 0.015, "fill_std": 0.0},
        }
        cmd += [
            "--kv-transfer-config", json.dumps(kv_config),
        ]
    elif mode == "pipelined":
        # Decode-only pipelining: fill KV -> stage to CPU -> pipeline.
        kv_config = {
            "kv_connector": "PipelinedDecodeBenchConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"fill_mean": 0.015, "fill_std": 0.0},
        }
        cmd += [
            "--decode-only-pipelining",
            "--kv-k-window", str(k_window),
            "--kv-transfer-config", json.dumps(kv_config),
            "--enable-prefix-caching", "false",
        ]
    elif mode == "real_prefill":
        # Real prefill + pipelined decode: model computes actual attention
        # during prefill, KV staged to CPU, decode via pipelining.
        kv_config = {
            "kv_connector": "PrefillStagingPipelinedConnector",
            "kv_role": "kv_both",
        }
        cmd += [
            "--enable-kv-pipelining",
            "--kv-k-window", str(k_window),
            "--free-original-kv-cache",
            "--kv-transfer-config", json.dumps(kv_config),
            "--enable-prefix-caching", "false",
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

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


# =========================================================================
# Request generation
# =========================================================================

def _generate_prompt(target_tokens: int) -> str:
    """Generate a prompt of approximately target_tokens length."""
    base = (
        "The following is a detailed technical analysis of system "
        "performance metrics in large-scale distributed computing "
        "environments. "
    )
    repeat_count = max(1, (target_tokens * 4) // len(base))
    return (base * repeat_count)[: target_tokens * 4]


# =========================================================================
# Streaming request (to capture TTFT and ITL)
# =========================================================================

def _send_streaming_request(
    port: int,
    model_name: str,
    prompt: str,
    max_tokens: int,
    timeout: int = 180,
) -> dict:
    """Send a streaming completions request and measure TTFT / ITL."""
    url = f"http://127.0.0.1:{port}/v1/completions"
    body = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
    )

    token_times: list[float] = []
    t_start = time.perf_counter()
    ttft: float | None = None
    total_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            buf = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buf += chunk
                # Parse SSE events
                while b"\n\n" in buf:
                    event, buf = buf.split(b"\n\n", 1)
                    for line in event.split(b"\n"):
                        line = line.strip()
                        if line.startswith(b"data: "):
                            data_str = line[6:].decode()
                            if data_str.strip() == "[DONE]":
                                continue
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            choices = data.get("choices", [])
                            if choices:
                                text = choices[0].get("text", "")
                                if text:
                                    now = time.perf_counter()
                                    if ttft is None:
                                        ttft = now - t_start
                                    token_times.append(now)
                                    total_tokens += 1

        t_end = time.perf_counter()
        e2e = t_end - t_start

        # Inter-token latencies
        itls = []
        for i in range(1, len(token_times)):
            itls.append(token_times[i] - token_times[i - 1])

        mean_itl = sum(itls) / len(itls) if itls else 0.0
        p99_itl = sorted(itls)[int(len(itls) * 0.99)] if itls else 0.0

        return {
            "success": True,
            "ttft_s": ttft,
            "mean_itl_s": mean_itl,
            "p99_itl_s": p99_itl,
            "e2e_s": e2e,
            "total_tokens": total_tokens,
            "tokens_per_sec": total_tokens / e2e if e2e > 0 else 0,
        }
    except Exception as e:
        return {"success": False, "error": str(e)[:200]}


def _send_request(
    port: int, model_name: str, prompt: str, max_tokens: int, timeout: int
) -> dict:
    """Non-streaming request for concurrency/throughput tests."""
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
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        e2e = time.perf_counter() - t0
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        return {
            "success": True,
            "e2e_s": e2e,
            "completion_tokens": completion_tokens,
            "tokens_per_sec": completion_tokens / e2e if e2e > 0 else 0,
        }
    except Exception as e:
        return {"success": False, "error": str(e)[:200]}


# =========================================================================
# Benchmark driver
# =========================================================================

def _run_latency_test(
    port: int,
    model_name: str,
    input_len: int,
    output_len: int,
    request_timeout: int,
    warmup: int,
) -> dict:
    """Measure single-request latency with streaming."""
    prompt = _generate_prompt(input_len)

    # Warmup
    for _ in range(warmup):
        _send_streaming_request(
            port, model_name, prompt, output_len, request_timeout
        )

    # Measurement (5 runs)
    results = []
    for _ in range(5):
        r = _send_streaming_request(
            port, model_name, prompt, output_len, request_timeout
        )
        if r["success"]:
            results.append(r)

    if not results:
        return {"error": "all latency requests failed"}

    return {
        "num_runs": len(results),
        "ttft_s_mean": sum(r["ttft_s"] for r in results) / len(results),
        "mean_itl_s_mean": (
            sum(r["mean_itl_s"] for r in results) / len(results)
        ),
        "p99_itl_s_mean": (
            sum(r["p99_itl_s"] for r in results) / len(results)
        ),
        "e2e_s_mean": sum(r["e2e_s"] for r in results) / len(results),
        "tokens_per_sec_mean": (
            sum(r["tokens_per_sec"] for r in results) / len(results)
        ),
    }


def _run_throughput_test(
    port: int,
    model_name: str,
    input_len: int,
    output_len: int,
    concurrency: int,
    request_timeout: int,
) -> dict:
    """Measure throughput at a given concurrency level."""
    prompt = _generate_prompt(input_len)
    results = []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _send_request, port, model_name, prompt,
                output_len, request_timeout,
            )
            for _ in range(concurrency)
        ]
        for f in as_completed(futures):
            results.append(f.result())
    elapsed = time.monotonic() - t0

    successes = sum(1 for r in results if r.get("success"))
    failures = concurrency - successes
    total_tokens = sum(
        r.get("completion_tokens", 0) for r in results if r.get("success")
    )
    errors = [r.get("error", "?") for r in results if not r.get("success")]

    return {
        "concurrency": concurrency,
        "successes": successes,
        "failures": failures,
        "elapsed_s": round(elapsed, 3),
        "total_output_tokens": total_tokens,
        "throughput_tok_per_s": (
            round(total_tokens / elapsed, 2) if elapsed > 0 else 0
        ),
        "errors": errors[:3],
    }


def run_mode(args: argparse.Namespace, mode: str, result_dir: Path) -> dict:
    """Run the full benchmark for a given mode."""
    print(f"\n{'=' * 70}")
    print(f"MODE: {mode.upper()}")
    print(f"{'=' * 70}")

    max_conc = max(args.concurrency)
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
            print(f"  [FAIL] Server failed to start. Check: {log_file}")
            return {"mode": mode, "error": "startup_failed"}

        model_name = _get_model_name(args.port)
        print(f"  [OK] Server ready ({model_name})")

        summary: dict = {"mode": mode}

        # ---- Latency test (single request, streaming) ----
        print("\n  --- Latency test (single-request, streaming) ---")
        latency = _run_latency_test(
            args.port, model_name,
            args.input_len, args.output_len,
            args.request_timeout, args.warmup_requests,
        )
        summary["latency"] = latency
        if "error" not in latency:
            print(
                f"    TTFT           : "
                f"{latency['ttft_s_mean']*1000:.1f} ms"
            )
            print(
                f"    Mean ITL       : "
                f"{latency['mean_itl_s_mean']*1000:.2f} ms"
            )
            print(
                f"    P99 ITL        : "
                f"{latency['p99_itl_s_mean']*1000:.2f} ms"
            )
            print(
                f"    E2E latency    : "
                f"{latency['e2e_s_mean']*1000:.1f} ms"
            )
            print(
                f"    Throughput     : "
                f"{latency['tokens_per_sec_mean']:.1f} tok/s"
            )
        else:
            print(f"    [FAIL] {latency['error']}")

        # ---- Throughput test (various concurrency) ----
        print("\n  --- Throughput test ---")
        throughput_results = []
        max_successful_conc = 0
        for conc in sorted(args.concurrency):
            result = _run_throughput_test(
                args.port, model_name,
                args.input_len, args.output_len,
                conc, args.request_timeout,
            )
            throughput_results.append(result)

            if result["failures"] == 0:
                max_successful_conc = conc
                print(
                    f"    conc={conc:3d}: [OK] {result['successes']} ok, "
                    f"{result['throughput_tok_per_s']} tok/s "
                    f"({result['elapsed_s']}s)"
                )
            else:
                print(
                    f"    conc={conc:3d}: [FAIL] {result['failures']} failures "
                    f"({result['elapsed_s']}s)"
                )
                if result["errors"]:
                    print(f"             {result['errors'][0][:100]}")

        summary["throughput"] = throughput_results
        summary["max_successful_concurrency"] = max_successful_conc

        # Save per-mode results
        with open(result_dir / f"results_{mode}.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    finally:
        print(f"  Stopping {mode} server ...")
        _kill_server(proc)
        time.sleep(5)


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Decode-Only P/D Pipelining Benchmark")
    print("=" * 70)
    print(f"  Model         : {args.model}")
    print(f"  Input len     : {args.input_len} tokens")
    print(f"  Output len    : {args.output_len} tokens")
    print(f"  Concurrency   : {args.concurrency}")
    print(f"  K-window      : {args.k_window}")
    print(f"  GPU mem util  : {args.gpu_memory_utilization}")
    print(f"  Modes         : {args.modes}")
    print(f"  Warmup        : {args.warmup_requests} requests")

    all_summaries = {}
    for mode in args.modes:
        summary = run_mode(args, mode, result_dir)
        all_summaries[mode] = summary

    # ---- Final comparison ----
    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print("=" * 70)
    print(f"  {'metric':<25s}", end="")
    for mode in args.modes:
        print(f"  {mode:>15s}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for _ in args.modes:
        print(f"  {'-'*15}", end="")
    print()

    metrics = [
        ("TTFT (ms)", "latency", "ttft_s_mean", 1000),
        ("Mean ITL (ms)", "latency", "mean_itl_s_mean", 1000),
        ("P99 ITL (ms)", "latency", "p99_itl_s_mean", 1000),
        ("E2E latency (ms)", "latency", "e2e_s_mean", 1000),
        ("Single-req tok/s", "latency", "tokens_per_sec_mean", 1),
        ("Max concurrency", None, "max_successful_concurrency", 1),
    ]
    for label, section, key, scale in metrics:
        print(f"  {label:<25s}", end="")
        for mode in args.modes:
            s = all_summaries.get(mode, {})
            if section:
                val = s.get(section, {}).get(key)
            else:
                val = s.get(key)
            if val is not None:
                print(f"  {val * scale:>15.2f}", end="")
            else:
                print(f"  {'N/A':>15s}", end="")
        print()

    # Throughput comparison at each concurrency level
    print("\n  Throughput (tok/s) by concurrency:")
    for conc in sorted(args.concurrency):
        print(f"    conc={conc:<3d}:", end="")
        for mode in args.modes:
            s = all_summaries.get(mode, {})
            for r in s.get("throughput", []):
                if r.get("concurrency") == conc:
                    if r["failures"] == 0:
                        print(
                            f"  {r['throughput_tok_per_s']:>12.1f}", end=""
                        )
                    else:
                        print(f"  {'FAIL':>12s}", end="")
                    break
            else:
                print(f"  {'N/A':>12s}", end="")
        print()

    # Improvement summary
    baseline = all_summaries.get("baseline", {})
    b_itl = baseline.get("latency", {}).get("mean_itl_s_mean")
    b_conc = baseline.get("max_successful_concurrency", 0)

    print("\n  Key findings:")
    for pmode in ("pipelined", "real_prefill"):
        p = all_summaries.get(pmode, {})
        p_itl = p.get("latency", {}).get("mean_itl_s_mean")
        p_conc = p.get("max_successful_concurrency", 0)
        if b_itl and p_itl and b_itl > 0:
            overhead = ((p_itl - b_itl) / b_itl) * 100
            print(f"    [{pmode}] ITL overhead: {overhead:+.1f}%")
        if b_conc > 0 and p_conc > 0:
            improvement = ((p_conc - b_conc) / b_conc) * 100
            print(
                f"    [{pmode}] Max concurrency improvement: "
                f"{improvement:+.1f}%"
            )

    with open(result_dir / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n  Full results: {result_dir}/")


if __name__ == "__main__":
    main()
