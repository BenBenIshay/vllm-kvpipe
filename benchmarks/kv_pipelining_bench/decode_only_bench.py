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
from typing import Any

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
from gpu_preflight import check_gpu_free_fraction, format_snapshot  # noqa: E402
from hf_env import apply_hf_token_from_user_file, hf_token_file_path  # noqa: E402


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
    p.add_argument(
        "--streaming-throughput",
        action="store_true",
        help=(
            "Use concurrent streaming completions for the throughput sweep "
            "(measures tok/s plus mean/p99 ITL and TTFT under load). "
            "Better for seeing decode behavior as the system approaches its limit."
        ),
    )
    p.add_argument(
        "--enforce-eager",
        dest="enforce_eager",
        action="store_true",
        default=True,
        help=(
            "Pass --enforce-eager to the API server (default: on). "
            "Avoids CUDA-graph dummy_run assertion with KV bench connectors on some builds."
        ),
    )
    p.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="Disable --enforce-eager for max throughput (may fail to start with bench connectors).",
    )
    p.add_argument(
        "--require-gpu-free-fraction",
        type=float,
        default=0.85,
        help=(
            "Abort unless at least this fraction of GPU VRAM is free before "
            "starting servers (default: 0.85). Also requires free >= "
            "--gpu-memory-utilization."
        ),
    )
    p.add_argument(
        "--skip-gpu-preflight",
        action="store_true",
        help="Skip GPU memory preflight (not recommended for published numbers).",
    )
    p.add_argument(
        "--prefix-caching-on",
        action="store_true",
        help=(
            "Leave vLLM default prefix caching enabled (for external comparison "
            "rows only; do not mix with primary PC=OFF results)."
        ),
    )
    p.add_argument(
        "--ignore-eos",
        action="store_true",
        help=(
            "Pass ignore_eos=true to completion requests. Useful for synthetic "
            "dummy-KV decode where random KV can emit EOS before output-len."
        ),
    )
    p.add_argument(
        "--itl-steady-skip",
        type=int,
        default=1,
        help=(
            "When computing steady-state ITL, drop the first N inter-chunk "
            "gaps per request. Default 1: the first gap (after TTFT) often "
            "includes pipelined decode pipeline fill that baseline does not "
            "pay as inter-token time (baseline KV is already on GPU). "
            "TTFT always includes prefill + first-token work including initial "
            "k_window H2D for the feature path."
        ),
    )
    return p.parse_args()


def _itl_stats(
    itls: list[float], steady_skip: int
) -> dict[str, Any]:
    """Mean/p99 over all gaps, plus steady-state after dropping first N gaps."""
    if not itls:
        return {
            "mean_itl_s": 0.0,
            "p99_itl_s": 0.0,
            "mean_itl_steady_s": 0.0,
            "p99_itl_steady_s": 0.0,
            "itl_steady_skip": steady_skip,
            "itl_steady_sample_count": 0,
        }

    def _mean_p99(samples: list[float]) -> tuple[float, float]:
        mean_v = sum(samples) / len(samples)
        srt = sorted(samples)
        p99_idx = max(0, min(len(srt) - 1, int((len(srt) - 1) * 0.99)))
        return mean_v, srt[p99_idx]

    mean_itl, p99_itl = _mean_p99(itls)
    skip = max(0, min(steady_skip, len(itls)))
    steady = itls[skip:]
    if steady:
        mean_steady, p99_steady = _mean_p99(steady)
    else:
        mean_steady, p99_steady = mean_itl, p99_itl

    return {
        "mean_itl_s": mean_itl,
        "p99_itl_s": p99_itl,
        "mean_itl_steady_s": mean_steady,
        "p99_itl_steady_s": p99_steady,
        "itl_steady_skip": skip,
        "itl_steady_sample_count": len(steady),
    }


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
    enforce_eager: bool = True,
    prefix_caching_on: bool = False,
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
    if enforce_eager:
        cmd.append("--enforce-eager")

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
        ]
        if not prefix_caching_on:
            cmd.append("--no-enable-prefix-caching")
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
        ]
        if not prefix_caching_on:
            cmd.append("--no-enable-prefix-caching")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    log_fh = open(log_file, "w") if log_file else subprocess.DEVNULL
    proc = subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT, preexec_fn=os.setsid,
    )
    return proc


def _kill_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def _get_model_name(port: int, retries: int = 20, retry_s: float = 1.0) -> str:
    """Read model id from OpenAI models endpoint with startup retries."""
    url = f"http://127.0.0.1:{port}/v1/models"
    last_err: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = resp.read().decode().strip()
            if not raw:
                raise ValueError("empty /v1/models response")
            data = json.loads(raw)
            models = data.get("data") if isinstance(data, dict) else None
            if not models:
                raise ValueError(f"missing model list in /v1/models: {data!r}")
            mid = models[0].get("id")
            if not mid:
                raise ValueError(f"missing model id in /v1/models: {models[0]!r}")
            return str(mid)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(retry_s)
    raise RuntimeError(
        f"failed to get model name from {url} after {retries} retries"
    ) from last_err


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
    itl_steady_skip: int = 1,
    ignore_eos: bool = False,
) -> dict:
    """Send a streaming completions request and measure TTFT / ITL.

    TTFT covers prefill + first output token (including feature-path initial
    k_window H2D before layer 0). Steady ITL drops the first *itl_steady_skip*
    inter-chunk gaps so pipeline fill is not compared to baseline decode ITL.
    """
    url = f"http://127.0.0.1:{port}/v1/completions"
    body_dict: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        body_dict["ignore_eos"] = True
    body = json.dumps(body_dict).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
    )

    token_times: list[float] = []
    t_start = time.perf_counter()
    ttft: float | None = None
    chunk_events = 0
    usage_completion_tokens: int | None = None
    finish_reason: str | None = None

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
                            usage = data.get("usage")
                            if isinstance(usage, dict):
                                ct = usage.get("completion_tokens")
                                if ct is not None:
                                    usage_completion_tokens = int(ct)
                            choices = data.get("choices", [])
                            if choices:
                                ch0 = choices[0]
                                fr = ch0.get("finish_reason")
                                if fr:
                                    finish_reason = str(fr)
                                text = ch0.get("text", "")
                                if text:
                                    now = time.perf_counter()
                                    if ttft is None:
                                        ttft = now - t_start
                                    token_times.append(now)
                                    chunk_events += 1

        t_end = time.perf_counter()
        e2e = t_end - t_start

        total_tokens = chunk_events
        if usage_completion_tokens is not None:
            total_tokens = max(total_tokens, usage_completion_tokens)

        if max_tokens > 0 and total_tokens == 0:
            return {
                "success": False,
                "error": "streaming produced 0 tokens (decode may not have run)",
            }
        if (
            max_tokens > 0
            and usage_completion_tokens is not None
            and usage_completion_tokens < int(max_tokens * 0.9)
        ):
            return {
                "success": False,
                "error": (
                    f"short completion: usage.completion_tokens="
                    f"{usage_completion_tokens} < 0.9*{max_tokens} "
                    f"(finish_reason={finish_reason!r})"
                ),
                "completion_tokens": usage_completion_tokens,
                "chunk_events": chunk_events,
                "finish_reason": finish_reason,
            }

        # Inter-token latencies (gaps between streaming chunk arrivals).
        itls = [
            token_times[i] - token_times[i - 1]
            for i in range(1, len(token_times))
        ]
        stats = _itl_stats(itls, itl_steady_skip)

        return {
            "success": True,
            "ttft_s": ttft,
            "mean_itl_s": stats["mean_itl_s"],
            "p99_itl_s": stats["p99_itl_s"],
            "mean_itl_steady_s": stats["mean_itl_steady_s"],
            "p99_itl_steady_s": stats["p99_itl_steady_s"],
            "itl_steady_skip": stats["itl_steady_skip"],
            "itl_steady_sample_count": stats["itl_steady_sample_count"],
            "e2e_s": e2e,
            "total_tokens": total_tokens,
            "completion_tokens": usage_completion_tokens or total_tokens,
            "chunk_events": chunk_events,
            "finish_reason": finish_reason,
            "tokens_per_sec": total_tokens / e2e if e2e > 0 else 0,
            "itl_samples": itls,
        }
    except Exception as e:
        return {"success": False, "error": str(e)[:200]}


def _validate_completion_json(
    data: dict[str, Any],
    *,
    min_expected_completion_tokens: int,
) -> tuple[bool, str | None]:
    """Reject HTTP-200 bodies that are empty errors or zero-token completions.

    vLLM may return HTTP 200 with an ``error`` object, or 200 with
    ``choices``/``usage`` that indicate no tokens were generated (e.g. dead
    engine). Those must not count as successful throughput.
    """
    if data.get("error") is not None:
        err = data["error"]
        if isinstance(err, dict):
            return False, str(err.get("message", err))[:300]
        return False, str(err)[:300]
    choices = data.get("choices")
    if not choices:
        return False, "no choices in completion response"
    ch0 = choices[0]
    text = (ch0.get("text") or "").strip()
    usage = data.get("usage") or {}
    ct = usage.get("completion_tokens")
    if ct is None:
        ct = 0
    # Some responses omit usage; fall back to token_ids length if present.
    if ct == 0:
        tids = ch0.get("token_ids")
        if isinstance(tids, list) and len(tids) > 0:
            ct = len(tids)
    if min_expected_completion_tokens > 0 and ct == 0 and not text:
        fr = ch0.get("finish_reason")
        return (
            False,
            f"empty completion (completion_tokens=0, finish_reason={fr!r})",
        )
    return True, None


def _send_request(
    port: int,
    model_name: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    ignore_eos: bool = False,
) -> dict:
    """Non-streaming request for concurrency/throughput tests."""
    url = f"http://127.0.0.1:{port}/v1/completions"
    body_dict: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if ignore_eos:
        body_dict["ignore_eos"] = True
    body = json.dumps(body_dict).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            data = json.loads(raw)
        e2e = time.perf_counter() - t0
        ok, err = _validate_completion_json(
            data,
            min_expected_completion_tokens=max_tokens,
        )
        if not ok:
            return {"success": False, "error": err or "invalid completion"}
        usage = data.get("usage", {})
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        if completion_tokens == 0:
            ch0 = (data.get("choices") or [{}])[0]
            tids = ch0.get("token_ids")
            if isinstance(tids, list) and len(tids) > 0:
                completion_tokens = len(tids)
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
    itl_steady_skip: int,
    ignore_eos: bool,
) -> dict:
    """Measure single-request latency with streaming."""
    prompt = _generate_prompt(input_len)

    # Warmup
    for _ in range(warmup):
        _send_streaming_request(
            port, model_name, prompt, output_len, request_timeout,
            itl_steady_skip=itl_steady_skip,
            ignore_eos=ignore_eos,
        )

    # Measurement (5 runs)
    results = []
    for _ in range(5):
        r = _send_streaming_request(
            port, model_name, prompt, output_len, request_timeout,
            itl_steady_skip=itl_steady_skip,
            ignore_eos=ignore_eos,
        )
        if r["success"] and r.get("ttft_s") is not None:
            results.append(r)

    if not results:
        return {"error": "all latency requests failed"}

    def _avg(key: str) -> float:
        return sum(float(r[key]) for r in results) / len(results)

    return {
        "num_runs": len(results),
        "ttft_s_mean": _avg("ttft_s"),
        "mean_itl_s_mean": _avg("mean_itl_s"),
        "p99_itl_s_mean": _avg("p99_itl_s"),
        "mean_itl_steady_s_mean": _avg("mean_itl_steady_s"),
        "p99_itl_steady_s_mean": _avg("p99_itl_steady_s"),
        "itl_steady_skip": itl_steady_skip,
        "e2e_s_mean": _avg("e2e_s"),
        "tokens_per_sec_mean": _avg("tokens_per_sec"),
    }


def _run_throughput_test(
    port: int,
    model_name: str,
    input_len: int,
    output_len: int,
    concurrency: int,
    request_timeout: int,
    ignore_eos: bool,
) -> dict:
    """Measure throughput at a given concurrency level."""
    prompt = _generate_prompt(input_len)
    results = []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _send_request, port, model_name, prompt,
                output_len, request_timeout, ignore_eos,
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


def _maybe_profile(port: int, action: str) -> None:
    """Diagnostic: wrap the timed concurrent wave with vLLM torch-profiler.

    Enabled only when env ``KVPIPE_PROFILE_WAVE=1`` (server must be launched with
    ``VLLM_TORCH_PROFILER_DIR`` set). ``action`` is "start" or "stop". Failures are
    swallowed so a missing profiler endpoint never breaks the benchmark.
    """
    if os.environ.get("KVPIPE_PROFILE_WAVE", "0") != "1":
        return
    path = "/start_profile" if action == "start" else "/stop_profile"
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", data=b"", method="POST"
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp.read()
    except Exception as e:  # noqa: BLE001
        print(f"  [profile] {action} failed: {str(e)[:120]}")


def _run_throughput_test_streaming(
    port: int,
    model_name: str,
    input_len: int,
    output_len: int,
    concurrency: int,
    request_timeout: int,
    itl_steady_skip: int,
    ignore_eos: bool,
) -> dict:
    """Throughput with **concurrent** streaming requests — ITL measured under load.

    Each request records inter-chunk ITLs; we aggregate:
    - ``concurrent_mean_itl_s_mean``: mean of per-request mean ITL
    - ``concurrent_mean_itl_steady_s_mean``: same after dropping first N gaps
    - ``concurrent_mean_p99_itl_s``: mean of per-request p99 ITL
    - ``concurrent_global_p99_itl_s``: p99 over all ITL samples (pooled)
    - ``concurrent_ttft_s_mean``: mean TTFT across successful requests
    """
    prompt = _generate_prompt(input_len)
    results: list[dict[str, Any]] = []
    _maybe_profile(port, "start")
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _send_streaming_request,
                port,
                model_name,
                prompt,
                output_len,
                request_timeout,
                itl_steady_skip,
                ignore_eos,
            )
            for _ in range(concurrency)
        ]
        for f in as_completed(futures):
            results.append(f.result())
    elapsed = time.monotonic() - t0
    _maybe_profile(port, "stop")

    oks = [r for r in results if r.get("success")]
    successes = len(oks)
    failures = concurrency - successes
    total_tokens = sum(
        int(r.get("completion_tokens", r.get("total_tokens", 0)) or 0) for r in oks
    )
    errors = [r.get("error", "?") for r in results if not r.get("success")]

    mean_itls = [float(r["mean_itl_s"]) for r in oks if r.get("mean_itl_s") is not None]
    mean_itls_steady = [
        float(r["mean_itl_steady_s"]) for r in oks
        if r.get("mean_itl_steady_s") is not None
    ]
    p99_itls = [float(r["p99_itl_s"]) for r in oks if r.get("p99_itl_s") is not None]
    p99_itls_steady = [
        float(r["p99_itl_steady_s"]) for r in oks
        if r.get("p99_itl_steady_s") is not None
    ]
    ttfts = [float(r["ttft_s"]) for r in oks if r.get("ttft_s") is not None]

    all_itl: list[float] = []
    all_itl_steady: list[float] = []
    skip = max(0, itl_steady_skip)
    for r in oks:
        samples = r.get("itl_samples")
        if isinstance(samples, list) and samples:
            floats = [float(x) for x in samples]
            all_itl.extend(floats)
            all_itl_steady.extend(floats[skip:])

    def _global_p99(samples: list[float]) -> float | None:
        if len(samples) >= 2:
            srt = sorted(samples)
            gidx = max(0, min(len(srt) - 1, int((len(srt) - 1) * 0.99)))
            return srt[gidx]
        if len(samples) == 1:
            return samples[0]
        return None

    global_p99 = _global_p99(all_itl)
    global_p99_steady = _global_p99(all_itl_steady)

    out: dict[str, Any] = {
        "concurrency": concurrency,
        "successes": successes,
        "failures": failures,
        "elapsed_s": round(elapsed, 3),
        "total_output_tokens": total_tokens,
        "throughput_tok_per_s": (
            round(total_tokens / elapsed, 2) if elapsed > 0 else 0
        ),
        "errors": errors[:3],
        "itl_steady_skip": itl_steady_skip,
        "concurrent_mean_itl_s_mean": (
            round(sum(mean_itls) / len(mean_itls), 6) if mean_itls else None
        ),
        "concurrent_mean_itl_steady_s_mean": (
            round(sum(mean_itls_steady) / len(mean_itls_steady), 6)
            if mean_itls_steady
            else None
        ),
        "concurrent_mean_p99_itl_s": (
            round(sum(p99_itls) / len(p99_itls), 6) if p99_itls else None
        ),
        "concurrent_mean_p99_itl_steady_s_mean": (
            round(sum(p99_itls_steady) / len(p99_itls_steady), 6)
            if p99_itls_steady
            else None
        ),
        "concurrent_global_p99_itl_s": (
            round(global_p99, 6) if global_p99 is not None else None
        ),
        "concurrent_global_p99_itl_steady_s": (
            round(global_p99_steady, 6) if global_p99_steady is not None else None
        ),
        "concurrent_ttft_s_mean": (
            round(sum(ttfts) / len(ttfts), 6) if ttfts else None
        ),
        "concurrent_itl_sample_count": len(all_itl),
        "concurrent_itl_steady_sample_count": len(all_itl_steady),
    }
    return out


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
        enforce_eager=args.enforce_eager,
        prefix_caching_on=args.prefix_caching_on,
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
            args.itl_steady_skip,
            args.ignore_eos,
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
            if latency.get("mean_itl_steady_s_mean") is not None:
                print(
                    f"    Mean ITL steady: "
                    f"{latency['mean_itl_steady_s_mean']*1000:.2f} ms "
                    f"(skip first {latency.get('itl_steady_skip', 0)} gaps)"
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
        tp_label = (
            "Throughput test (streaming under load)"
            if args.streaming_throughput
            else "Throughput test (batch completions)"
        )
        print(f"\n  --- {tp_label} ---")
        throughput_results = []
        max_successful_conc = 0
        for conc in sorted(args.concurrency):
            if args.streaming_throughput:
                result = _run_throughput_test_streaming(
                    args.port, model_name,
                    args.input_len, args.output_len,
                    conc, args.request_timeout,
                    args.itl_steady_skip,
                    args.ignore_eos,
                )
            else:
                result = _run_throughput_test(
                    args.port, model_name,
                    args.input_len, args.output_len,
                    conc, args.request_timeout,
                    args.ignore_eos,
                )
            throughput_results.append(result)

            if result["failures"] == 0:
                max_successful_conc = conc
                line = (
                    f"    conc={conc:3d}: [OK] {result['successes']} ok, "
                    f"{result['throughput_tok_per_s']} tok/s "
                    f"({result['elapsed_s']}s)"
                )
                if args.streaming_throughput:
                    mitl = result.get("concurrent_mean_itl_steady_s_mean")
                    if mitl is None:
                        mitl = result.get("concurrent_mean_itl_s_mean")
                    pitl = result.get("concurrent_mean_p99_itl_s")
                    g99 = result.get("concurrent_global_p99_itl_steady_s")
                    if g99 is None:
                        g99 = result.get("concurrent_global_p99_itl_s")
                    ttft = result.get("concurrent_ttft_s_mean")
                    if mitl is not None:
                        line += f" | mean_ITL_steady {mitl*1000:.2f}ms"
                    if pitl is not None:
                        line += f" p99_ITL~ {pitl*1000:.2f}ms"
                    if g99 is not None:
                        line += f" global_p99_ITL {g99*1000:.2f}ms"
                    if ttft is not None:
                        line += f" mean_TTFT {ttft*1000:.1f}ms"
                print(line)
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

    if not args.skip_gpu_preflight:
        snap = check_gpu_free_fraction(
            min_free_fraction=args.require_gpu_free_fraction,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        print("GPU preflight OK")
        print(f"  {format_snapshot(snap).replace(chr(10), chr(10) + '  ')}")
        print()

    if apply_hf_token_from_user_file():
        print(f"  HF auth      : token from {hf_token_file_path()}")

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
    print(f"  ITL steady-skip: {args.itl_steady_skip} gaps (after TTFT)")
    print(f"  Prefix caching: {'ON (comparison)' if args.prefix_caching_on else 'OFF'}")
    print(f"  Ignore EOS    : {args.ignore_eos}")
    print(
        f"  Throughput    : "
        f"{'streaming (ITL under load)' if args.streaming_throughput else 'batch'}"
    )
    print(f"  Enforce eager : {args.enforce_eager}")

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
