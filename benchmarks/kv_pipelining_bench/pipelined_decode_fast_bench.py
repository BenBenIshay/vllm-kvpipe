#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fast decode-only benchmark (dummy KV) — pipelined by default
============================================================

Uses **PipelinedDecodeBenchConnector** in decode-only mode
(``--decode-only-pipelining``) for dummy-KV layer-wise CPU↔GPU pipelining.
Optional **DecodeBenchConnector** baseline (``--with-baseline``). Suitable for
tuning **K** (``--kv-k-window``) and scheduler limits. Default
**--inflate-kv-pool** enlarges the block pool for the rolling K-tensor layout.

Optional `--with-baseline` runs **DecodeBenchConnector** in a second pass for
same-day comparison (only when the GPU is idle enough to be meaningful).

Example::

    python benchmarks/kv_pipelining_bench/pipelined_decode_fast_bench.py \\
        --model /data/ben/models/Meta-Llama-3.1-8B-Instruct \\
        --k-window 6 \\
        --concurrency 8 16 32 \\
        --result-dir /tmp/kv_fast_bench_run1

Programmatic use::

    from pathlib import Path
    from argparse import Namespace
    from pipelined_decode_fast_bench import run_benchmark

    summary = run_benchmark(Namespace(
        model=\"/data/ben/models/...\", k_window=6,
        concurrency=[8, 16], modes=[\"pipelined\"],
        result_dir=Path(\"/tmp/out\"), ...
    ))
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

# Reuse client + measurement helpers from decode_only_bench.py
_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

import decode_only_bench as dob  # noqa: E402
from hf_env import apply_hf_token_from_user_file, hf_token_file_path  # noqa: E402


def query_gpu_memory_mib(gpu_index: int = 0) -> tuple[int, int, int]:
    """Return (total_mib, used_mib, free_mib) for one GPU via nvidia-smi."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return 0, 0, 0
        line = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return 0, 0, 0
        total, used, free = (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])))
        return total, used, free
    except (OSError, ValueError, subprocess.TimeoutExpired, IndexError):
        return 0, 0, 0


def apply_auto_gpu_memory(ns: argparse.Namespace) -> None:
    """Set ``ns.gpu_memory_utilization`` from nvidia-smi when ``auto_gpu_memory`` is True."""
    if not getattr(ns, "auto_gpu_memory", True):
        return
    total_mib, used_mib, free_mib = query_gpu_memory_mib(ns.gpu_index)
    if total_mib > 0:
        ns.gpu_memory_utilization = compute_gpu_memory_utilization(
            total_mib,
            free_mib,
            getattr(ns, "free_memory_fraction", 0.6),
        )
        print(
            f"  [gpu] GPU {ns.gpu_index}: total={total_mib} MiB used={used_mib} MiB "
            f"free={free_mib} MiB → --gpu-memory-utilization {ns.gpu_memory_utilization:.4f} "
            f"(auto: {getattr(ns, 'free_memory_fraction', 0.6)} * free/total, clamped)"
        )
    else:
        print(
            "  [gpu] nvidia-smi unavailable; using --gpu-memory-utilization "
            f"{getattr(ns, 'gpu_memory_utilization', 0.8):.2f}",
            file=sys.stderr,
        )


def clamp_gpu_mem_for_pipelined_window_buffers(ns: argparse.Namespace) -> None:
    """Optionally cap ``gpu_memory_utilization`` for legacy WindowPool runs.

    The target rolling-tensor implementation does not allocate a second
    K-window pool, so capacity measurements should normally disable this cap
    and use ``--inflate-kv-pool``.  The cap remains useful for legacy/debug
    experiments that still stack connector-owned window buffers.

    Default: **tiered by max concurrency** (higher conc → stricter cap). Override
    with ``--pipelined-gpu-mem-cap 0.7`` or disable with ``--no-pipelined-gpu-mem-cap``.
    """
    if getattr(ns, "no_pipelined_gpu_mem_cap", False):
        return
    modes = ns.modes or []
    if "pipelined" not in modes and "hybrid" not in modes:
        return
    override = getattr(ns, "pipelined_gpu_mem_cap", None)
    if getattr(ns, "inflate_kv_pool", False) and override is None:
        return
    if override is not None:
        cap = float(override)
    else:
        max_c = max(ns.concurrency)
        # Leave headroom for K-window + runtime; high concurrency needs a lower cap.
        if max_c >= 128:
            cap = 0.62
        elif max_c >= 64:
            cap = 0.65
        elif max_c >= 32:
            cap = 0.68
        else:
            cap = 0.74
    if ns.gpu_memory_utilization > cap:
        print(
            "  [pipelined] clamping --gpu-memory-utilization "
            f"{ns.gpu_memory_utilization:.3f} -> {cap} "
            "(KV pool + K-window balance; --pipelined-gpu-mem-cap to override; "
            "--no-pipelined-gpu-mem-cap to disable)",
            flush=True,
        )
        ns.gpu_memory_utilization = cap

    # vLLM startup requires free HBM >= target fraction of total (see
    # request_memory in gpu_worker). Tiered caps can still exceed free memory
    # if other processes hold VRAM — clamp again to ~92% of (free/total).
    total_mib, _used_mib, free_mib = query_gpu_memory_mib(ns.gpu_index)
    if total_mib > 0 and free_mib > 0:
        max_safe = max(0.12, min(0.92, (free_mib / total_mib) * 0.92))
        if ns.gpu_memory_utilization > max_safe:
            print(
                "  [pipelined] clamping --gpu-memory-utilization "
                f"{ns.gpu_memory_utilization:.3f} -> {max_safe:.3f} "
                f"(fit free HBM: {free_mib}/{total_mib} MiB)",
                flush=True,
            )
            ns.gpu_memory_utilization = max_safe


def compute_gpu_memory_utilization(
    total_mib: int,
    free_mib: int,
    free_fraction: float,
    util_min: float = 0.25,
    util_max: float = 0.95,
) -> float:
    """Map available free memory to vLLM ``--gpu-memory-utilization`` (fraction of total).

    Uses ``free_fraction * (free_mib / total_mib)`` so we target using a fraction
    of *currently free* HBM (e.g. 80% of free), expressed as a fraction of total
    GPU memory — then clamp to ``[util_min, util_max]``.
    """
    if total_mib <= 0 or free_mib < 0:
        return 0.80
    raw = free_fraction * (free_mib / total_mib)
    return max(util_min, min(util_max, raw))


class GpuMemoryMonitor:
    """Background thread: when GPU HBM crosses the threshold, abort the benchmark."""

    def __init__(
        self,
        gpu_index: int,
        interval_s: float,
        warn_used_frac: float,
        abort_event: threading.Event,
        *,
        start_delay_s: float = 0.0,
        on_abort: Callable[[], None] | None = None,
    ) -> None:
        self._gpu_index = gpu_index
        self._interval_s = interval_s
        self._warn_used_frac = warn_used_frac
        self._abort_event = abort_event
        self._on_abort = on_abort
        self._start_delay_s = start_delay_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fired = False

    @property
    def aborted(self) -> bool:
        return self._abort_event.is_set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gpu-mem-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(
                timeout=max(self._interval_s + 1.0, self._start_delay_s + 5.0)
            )

    def _loop(self) -> None:
        # Skip load/compile spikes: vLLM legitimately uses most HBM during warmup.
        if self._start_delay_s > 0:
            if self._stop.wait(self._start_delay_s):
                return
        while not self._stop.wait(self._interval_s):
            total, used, _free = query_gpu_memory_mib(self._gpu_index)
            if total <= 0:
                continue
            frac = used / total
            if frac >= self._warn_used_frac:
                if self._fired:
                    continue
                self._fired = True
                print(
                    f"[gpu-mem] ABORT GPU {self._gpu_index}: {used}/{total} MiB "
                    f"({frac * 100:.1f}% used) ≥ threshold — stopping benchmark",
                    file=sys.stderr,
                    flush=True,
                )
                self._abort_event.set()
                if self._on_abort is not None:
                    try:
                        self._on_abort()
                    except Exception as e:
                        print(f"[gpu-mem] on_abort error: {e}", file=sys.stderr, flush=True)
                self._stop.set()
                break


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fast pipelined decode benchmark (dummy KV; baseline optional)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="HF id or local path (e.g. under /data/ben/...)",
    )
    p.add_argument("--port", type=int, default=8199)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU index for nvidia-smi and memory monitor",
    )
    p.add_argument(
        "--auto-gpu-memory",
        dest="auto_gpu_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Query nvidia-smi before starting: set --gpu-memory-utilization to "
            "free_memory_fraction × (free/total), clamped (default: on; "
            "see --free-memory-fraction, default 0.6)"
        ),
    )
    p.add_argument(
        "--free-memory-fraction",
        type=float,
        default=0.6,
        help=(
            "With --auto-gpu-memory: utilization ≈ this × (free_MiB/total_MiB), clamped "
            "(default 0.6 = budget 60% of free memory; lower = less KV pressure)"
        ),
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.80,
        help="Used only when --no-auto-gpu-memory (default 0.80)",
    )
    p.add_argument(
        "--no-pipelined-gpu-mem-cap",
        action="store_true",
        help=(
            "Do not cap --gpu-memory-utilization for pipelined mode "
            "(default: tiered cap by max concurrency)"
        ),
    )
    p.add_argument(
        "--pipelined-gpu-mem-cap",
        type=float,
        default=None,
        dest="pipelined_gpu_mem_cap",
        metavar="FRAC",
        help=(
            "Fixed gpu_memory_utilization cap for pipelined mode (overrides tiered "
            "default). Example: 0.70"
        ),
    )
    p.add_argument(
        "--inflate-kv-pool",
        dest="inflate_kv_pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pipelined mode: pass --inflate-kv-pool so the KV block pool is inflated "
            "for the rolling K-tensor layout. This is the capacity mode: same "
            "HBM KV budget, larger block pool. Default: true."
        ),
    )
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Optional cap for max tokens per iteration (scheduler)",
    )
    p.add_argument(
        "--with-baseline",
        action="store_true",
        help="Also run DecodeBenchConnector (all KV on GPU) after pipelined",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help='Modes to run. Default: ["pipelined"] or ["pipelined","baseline"] '
        "if --with-baseline",
    )
    p.add_argument("--k-window", type=int, default=6, dest="k_window")
    p.add_argument("--hybrid-b-hbm-max", type=int, default=49)
    p.add_argument("--hybrid-seq-tokens", type=int, default=512)
    p.add_argument("--hybrid-pcie-gbps", type=float, default=25.1)
    p.add_argument("--hybrid-b-tok", type=int, default=131072)
    p.add_argument("--hybrid-itl-ms", type=float, default=0.0)
    p.add_argument(
        "--enforce-eager",
        dest="enforce_eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pass --enforce-eager to vLLM (default: on). Required for stable "
            "startup with PipelinedDecodeBenchConnector: --no-enforce-eager can "
            "hit CUDA graph capture errors (dummy_run mode mismatch) on some builds."
        ),
    )
    p.add_argument(
        "--memory-monitor-interval",
        type=float,
        default=1.0,
        help="Seconds between GPU memory checks (0 disables monitor thread)",
    )
    p.add_argument(
        "--memory-warn-used-frac",
        type=float,
        default=0.995,
        help=(
            "Abort when used_HBM/total_HBM exceeds this (default 0.995). "
            "Lower values (e.g. 0.92) often false-positive during normal vLLM decode."
        ),
    )
    p.add_argument(
        "--memory-monitor-start-delay",
        type=float,
        default=90.0,
        help=(
            "Seconds to wait after server start before first memory poll (default 90). "
            "Skips load/compile/cudagraph spikes that briefly fill HBM."
        ),
    )
    p.add_argument(
        "--memory-abort-on-warn",
        dest="memory_abort_on_warn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On threshold: set abort and stop benchmark (default: on)",
    )
    p.add_argument(
        "--memory-kill-server-on-abort",
        dest="memory_kill_server_on_abort",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On abort: SIGTERM the vLLM server to unblock immediately (default: on)",
    )
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request ignore_eos so fixed decode lengths are honored (default: true)",
    )
    p.add_argument(
        "--prefix-caching-on",
        dest="prefix_caching_on",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Leave vLLM prefix caching enabled (default: on). Use "
            "--no-prefix-caching-on to disable on both baseline and pipelined."
        ),
    )
    p.add_argument(
        "--itl-steady-skip",
        type=int,
        default=1,
        help="Number of initial ITL gaps to skip in steady-state ITL summaries",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[8, 16, 32],
    )
    p.add_argument("--startup-timeout", type=int, default=600)
    p.add_argument("--request-timeout", type=int, default=300)
    p.add_argument("--warmup-requests", type=int, default=2)
    p.add_argument(
        "--concurrent-itl-throughput",
        dest="concurrent_itl_throughput",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Throughput phase: use concurrent streaming completions and report ITL "
            "under load (mean/p99 per request + global p99 over all gaps). "
            "Default: on. Use --no-concurrent-itl-throughput for legacy non-streaming "
            "HTTP completions (tok/s only)."
        ),
    )
    p.add_argument(
        "--skip-isolated-latency",
        dest="skip_isolated_latency",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip the single-request streaming latency phase (TTFT / ITL from one "
            "request only). Use when you only care about ITL from the concurrent "
            "throughput phase."
        ),
    )
    p.add_argument(
        "--result-dir",
        type=Path,
        default=Path("/tmp/kv_pipelining_fast_bench"),
    )
    return p


def launch_server(
    *,
    model: str,
    port: int,
    max_model_len: int,
    gpu_mem_util: float,
    max_num_seqs: int,
    mode: str,
    k_window: int,
    enforce_eager: bool,
    max_num_batched_tokens: int | None,
    log_file: str | None,
    inflate_kv_pool: bool = False,
    prefix_caching_on: bool = True,
    hybrid_b_hbm_max: int = 49,
    hybrid_seq_tokens: int = 512,
    hybrid_pcie_gbps: float = 25.1,
    hybrid_b_tok: int = 131072,
    hybrid_itl_ms: float = 0.0,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_mem_util),
        "--disable-log-requests",
        "--dtype",
        "auto",
        "--max-num-seqs",
        str(max_num_seqs),
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")
    if max_num_batched_tokens is not None:
        cmd += ["--max-num-batched-tokens", str(max_num_batched_tokens)]

    if mode == "vanilla":
        # Plain vLLM: NO kv-transfer-config -> no connector -> CUDA graphs allowed
        # (real prefill + decode). Used to price eager-vs-graph host launch cost.
        pass
    elif mode == "baseline":
        kv_config = {
            "kv_connector": "DecodeBenchConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"fill_mean": 0.015, "fill_std": 0.0},
        }
        cmd += ["--kv-transfer-config", json.dumps(kv_config)]
        if not prefix_caching_on:
            cmd.append("--no-enable-prefix-caching")
    elif mode == "pipelined":
        kv_config = {
            "kv_connector": "PipelinedDecodeBenchConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"fill_mean": 0.015, "fill_std": 0.0},
        }
        cmd += [
            "--enable-kv-pipelining",
            "--decode-only-pipelining",
            "--kv-k-window",
            str(k_window),
            "--kv-transfer-config",
            json.dumps(kv_config),
        ]
        if not prefix_caching_on:
            cmd.append("--no-enable-prefix-caching")
        if inflate_kv_pool:
            cmd.append("--inflate-kv-pool")
    elif mode == "hybrid":
        kv_config = {
            "kv_connector": "PipelinedDecodeBenchConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"fill_mean": 0.015, "fill_std": 0.0},
        }
        hybrid_b_hbm = int(hybrid_b_hbm_max)
        hybrid_seq = int(hybrid_seq_tokens)
        hybrid_pcie = float(hybrid_pcie_gbps)
        hybrid_b_tok_val = int(hybrid_b_tok)
        hybrid_itl = float(hybrid_itl_ms)
        cmd += [
            "--enable-kv-pipelining",
            "--kv-k-window",
            str(k_window),
            "--kv-transfer-config",
            json.dumps(kv_config),
            "--enable-transfer-aware-admission",
            "--enable-hybrid-kv-tiers",
            "--hybrid-b-hbm-max",
            str(hybrid_b_hbm),
            "--hybrid-seq-tokens",
            str(hybrid_seq),
            "--hybrid-pcie-gbps",
            str(hybrid_pcie),
            "--hybrid-b-tok",
            str(hybrid_b_tok_val),
        ]
        if hybrid_itl > 0:
            cmd += ["--hybrid-itl-ms", str(hybrid_itl)]
        if not prefix_caching_on:
            cmd.append("--no-enable-prefix-caching")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    log_fh = open(log_file, "w") if log_file else subprocess.DEVNULL
    return subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def _aborted_summary(
    mode: str,
    ns: argparse.Namespace,
    latency: Any,
    throughput_results: list[Any],
    max_ok: int,
    phase: str,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "k_window": ns.k_window,
        "latency": latency
        if latency
        else {"error": "aborted", "abort_phase": phase},
        "throughput": throughput_results,
        "max_successful_concurrency": max_ok,
        "aborted": True,
        "abort_reason": "gpu_memory_threshold",
        "abort_phase": phase,
    }


def run_mode(ns: argparse.Namespace, mode: str, result_dir: Path) -> dict[str, Any]:
    print(f"\n{'=' * 70}\nMODE: {mode.upper()}\n{'=' * 70}")

    max_conc = max(ns.concurrency)
    log_file = str(result_dir / f"server_{mode}.log")

    proc = launch_server(
        model=ns.model,
        port=ns.port,
        max_model_len=ns.max_model_len,
        gpu_mem_util=ns.gpu_memory_utilization,
        max_num_seqs=max_conc,
        mode=mode,
        k_window=ns.k_window,
        enforce_eager=ns.enforce_eager,
        max_num_batched_tokens=ns.max_num_batched_tokens,
        log_file=log_file,
        inflate_kv_pool=bool(
            getattr(ns, "inflate_kv_pool", False) and mode == "pipelined"
        ),
        prefix_caching_on=bool(getattr(ns, "prefix_caching_on", True)),
        hybrid_b_hbm_max=int(getattr(ns, "hybrid_b_hbm_max", 49)),
        hybrid_seq_tokens=int(getattr(ns, "hybrid_seq_tokens", 512)),
        hybrid_pcie_gbps=float(getattr(ns, "hybrid_pcie_gbps", 25.1)),
        hybrid_b_tok=int(getattr(ns, "hybrid_b_tok", 131072)),
        hybrid_itl_ms=float(getattr(ns, "hybrid_itl_ms", 0.0)),
    )

    abort_mem = threading.Event()
    monitor: GpuMemoryMonitor | None = None
    if (
        getattr(ns, "memory_monitor_interval", 0)
        and ns.memory_monitor_interval > 0
        and getattr(ns, "memory_abort_on_warn", True)
    ):
        kill = bool(getattr(ns, "memory_kill_server_on_abort", True))

        def _kill_proc() -> None:
            if kill and proc.poll() is None:
                dob._kill_server(proc)

        monitor = GpuMemoryMonitor(
            gpu_index=ns.gpu_index,
            interval_s=ns.memory_monitor_interval,
            warn_used_frac=ns.memory_warn_used_frac,
            abort_event=abort_mem,
            start_delay_s=getattr(ns, "memory_monitor_start_delay", 0.0),
            on_abort=_kill_proc if kill else None,
        )
        monitor.start()

    try:
        print(f"  Waiting for server (timeout={ns.startup_timeout}s) ...")
        if not dob._wait_for_server(ns.port, ns.startup_timeout):
            print(f"  [FAIL] Server did not become healthy. See {log_file}")
            return {"mode": mode, "error": "startup_failed"}

        model_name = dob._get_model_name(ns.port)
        print(f"  [OK] {model_name}")

        if abort_mem.is_set():
            summary = _aborted_summary(mode, ns, None, [], 0, "gpu_memory_before_latency")
            with open(result_dir / f"results_{mode}.json", "w") as f:
                json.dump(summary, f, indent=2)
            return summary

        use_stream_itl = getattr(ns, "concurrent_itl_throughput", True)
        if getattr(ns, "skip_isolated_latency", False):
            latency = {
                "skipped": True,
                "reason": "skip_isolated_latency",
            }
        else:
            latency = dob._run_latency_test(
                ns.port,
                model_name,
                ns.input_len,
                ns.output_len,
                ns.request_timeout,
                ns.warmup_requests,
                ns.itl_steady_skip,
                ns.ignore_eos,
            )
        if abort_mem.is_set():
            summary = _aborted_summary(mode, ns, latency, [], 0, "gpu_memory_after_latency")
            with open(result_dir / f"results_{mode}.json", "w") as f:
                json.dump(summary, f, indent=2)
            return summary

        throughput_results = []
        max_ok = 0
        for conc in sorted(ns.concurrency):
            if abort_mem.is_set():
                print("  [ABORT] Skipping remaining concurrency levels (GPU memory)", flush=True)
                break
            if use_stream_itl:
                r = dob._run_throughput_test_streaming(
                    ns.port,
                    model_name,
                    ns.input_len,
                    ns.output_len,
                    conc,
                    ns.request_timeout,
                    ns.itl_steady_skip,
                    ns.ignore_eos,
                )
            else:
                r = dob._run_throughput_test(
                    ns.port,
                    model_name,
                    ns.input_len,
                    ns.output_len,
                    conc,
                    ns.request_timeout,
                    ns.ignore_eos,
                )
            throughput_results.append(r)
            if r["failures"] == 0:
                max_ok = conc
            tag = "OK" if r["failures"] == 0 else "FAIL"
            if use_stream_itl:
                mitl = r.get("concurrent_mean_itl_s_mean")
                pitl = r.get("concurrent_mean_p99_itl_s")
                gitl = r.get("concurrent_global_p99_itl_s")
                extra = ""
                if mitl is not None and pitl is not None and gitl is not None:
                    extra = (
                        f" mean_ITL={mitl:.4f}s p99_req={pitl:.4f}s "
                        f"p99_global={gitl:.4f}s"
                    )
                print(
                    f"  conc={conc:4d} [{tag}] "
                    f"{r['throughput_tok_per_s']} tok/s{extra} "
                    f"ok={r['successes']}/{conc}"
                )
            else:
                print(
                    f"  conc={conc:4d} [{tag}] "
                    f"{r['throughput_tok_per_s']} tok/s "
                    f"ok={r['successes']}/{conc}"
                )
            if abort_mem.is_set():
                print("  [ABORT] Stopping after this level (GPU memory)", flush=True)
                break

        summary: dict[str, Any] = {
            "mode": mode,
            "k_window": ns.k_window,
            "latency": latency,
            "throughput": throughput_results,
            "max_successful_concurrency": max_ok,
            "aborted": abort_mem.is_set(),
            "abort_reason": "gpu_memory_threshold" if abort_mem.is_set() else None,
        }
        with open(result_dir / f"results_{mode}.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary
    finally:
        if monitor is not None:
            monitor.stop()
        print(f"  Stopping server ({mode}) ...")
        dob._kill_server(proc)
        time.sleep(3)


def run_benchmark(ns: argparse.Namespace) -> dict[str, Any]:
    """Run all requested modes; return {mode: summary}."""
    result_dir = Path(ns.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    total_mib, used_mib, free_mib = query_gpu_memory_mib(ns.gpu_index)
    manifest: dict[str, Any] = {
        "argv": sys.argv,
        "model": ns.model,
        "k_window": ns.k_window,
        "concurrency": list(ns.concurrency),
        "inflate_kv_pool": getattr(ns, "inflate_kv_pool", False),
        "pipelined_gpu_mem_cap": getattr(ns, "pipelined_gpu_mem_cap", None),
        "enforce_eager": ns.enforce_eager,
        "gpu_index": ns.gpu_index,
        "auto_gpu_memory": ns.auto_gpu_memory,
        "free_memory_fraction": getattr(ns, "free_memory_fraction", 0.6),
        "gpu_memory_utilization": ns.gpu_memory_utilization,
        "nvidia_smi_mib_at_start": {
            "total": total_mib,
            "used": used_mib,
            "free": free_mib,
        },
        "max_num_batched_tokens": ns.max_num_batched_tokens,
        "input_len": ns.input_len,
        "output_len": ns.output_len,
        "memory_monitor_interval": getattr(ns, "memory_monitor_interval", 1.0),
        "memory_warn_used_frac": getattr(ns, "memory_warn_used_frac", 0.995),
        "memory_monitor_start_delay": getattr(ns, "memory_monitor_start_delay", 90.0),
        "memory_abort_on_warn": getattr(ns, "memory_abort_on_warn", True),
        "memory_kill_server_on_abort": getattr(ns, "memory_kill_server_on_abort", True),
        "concurrent_itl_throughput": getattr(ns, "concurrent_itl_throughput", True),
        "skip_isolated_latency": getattr(ns, "skip_isolated_latency", False),
        "prefix_caching_on": getattr(ns, "prefix_caching_on", True),
    }
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_BENCH_DIR.parent.parent),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if rev.returncode == 0:
            manifest["git_head"] = rev.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass

    with open(result_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    out: dict[str, Any] = {}
    for mode in ns.modes:
        out[mode] = run_mode(ns, mode, result_dir)
        if isinstance(out[mode], dict) and out[mode].get("aborted"):
            print(
                "\n[bench] GPU memory abort — skipping remaining modes.",
                file=sys.stderr,
                flush=True,
            )
            break

    with open(result_dir / "summary_all.json", "w") as f:
        json.dump(out, f, indent=2)

    return out


def main() -> None:
    p = build_parser()
    ns = p.parse_args()
    if apply_hf_token_from_user_file():
        print(f"  HF auth       : token from {hf_token_file_path()}", flush=True)
    if not ns.model:
        p.error("--model is required (HF id or path under e.g. /data/ben/...)")

    if ns.modes is None:
        ns.modes = ["pipelined", "baseline"] if ns.with_baseline else ["pipelined"]

    if ns.auto_gpu_memory:
        apply_auto_gpu_memory(ns)
    else:
        total_mib, _u, free_mib = query_gpu_memory_mib(ns.gpu_index)
        print(
            f"  [gpu] GPU {ns.gpu_index}: total={total_mib} MiB free={free_mib} MiB "
            f"(manual --gpu-memory-utilization {ns.gpu_memory_utilization})"
        )

    clamp_gpu_mem_for_pipelined_window_buffers(ns)

    print("=" * 70)
    print("pipelined_decode_fast_bench (dummy KV)")
    print("=" * 70)
    print(f"  Model          : {ns.model}")
    print(f"  K (kv-k-window): {ns.k_window}")
    print(f"  Modes          : {ns.modes}")
    print(f"  Concurrency    : {ns.concurrency}")
    print(
        f"  Throughput ITL : {getattr(ns, 'concurrent_itl_throughput', True)} "
        f"(concurrent streaming gaps; --no-concurrent-itl-throughput = tok/s only)"
    )
    print(
        f"  Isolated latency: {not getattr(ns, 'skip_isolated_latency', False)} "
        f"(single-request TTFT/ITL; --skip-isolated-latency to skip)"
    )
    print(
        f"  Inflate pool   : {getattr(ns, 'inflate_kv_pool', False)} "
        f"(capacity mode; --no-inflate-kv-pool for HBM-saving debug runs)"
    )
    print(f"  Enforce eager  : {ns.enforce_eager}")
    print(
        f"  Prefix caching : {getattr(ns, 'prefix_caching_on', True)} "
        f"(--no-prefix-caching-on to disable both arms)"
    )
    print(f"  GPU mem util   : {ns.gpu_memory_utilization:.4f}")
    print(f"  Result dir     : {ns.result_dir}")

    run_benchmark(ns)
    print(f"\nDone. Results under: {ns.result_dir}/")


if __name__ == "__main__":
    main()
