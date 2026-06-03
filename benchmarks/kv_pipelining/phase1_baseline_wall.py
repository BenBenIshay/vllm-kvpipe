#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Phase 1 — Baseline vLLM wall characterization (stock server, then optional CPU↔GPU bench)

**Purpose:** Compare **stock vLLM** (all KV on GPU, no KV connector) against the optional
**PipelinedDecodeBenchConnector** path that **imitates layer-wise CPU staging** so you can
measure operational impact (TPOT/ITL, throughput) after recording the baseline.

**Data / quota:** By default all HF caches and run outputs go under ``/data/ben/vllm_bench``
(override with ``--data-root`` or ``VLLM_BENCH_DATA_ROOT``). This avoids ``$HOME`` disk quota
when downloading weights.

**Sweep modes:**

1. **Fixed list:** ``--concurrency 8 16 32 ...`` (default traditional sweep).
2. **Until KV pressure:** ``--until-kv-pressure`` increases concurrency (geometric or +step)
   until streaming requests **fail** (scheduler cannot allocate KV blocks / admission backs up /
   client errors). First failing level is recorded and the sweep stops.

**Metrics:** For steady decode, **ITL** inter-token gaps ≈ **TPOT** / **TBT** (time per output
token between chunks). Results include ``tpot_*`` / ``tbt_*`` aliases next to ``pooled_itl_*``.

**GPU / memory timeline:** Pass ``--with-nsys`` to wrap the **vLLM server** with
``nsys profile`` (Nsight Systems). Reports are written next to JSON results as
``nsys_<phase>.nsys-rep`` — open in the Nsight Systems GUI for CUDA timelines, copies,
and kernel correlation (not the HTTP client process).

**nvidia-smi:** ``--gpu-extended-metrics`` adds utilization / clocks / power snapshots.
These complement but do not replace Nsight for memory traffic analysis.

Use the **same Python** that has vLLM/PyTorch (e.g. ``/path/to/vllm/.venv/bin/python``).
System ``python3`` often lacks ``torch``, so the spawned server exits immediately.

Usage (recommended, stock baseline then CPU-push compare)::

    export VLLM_BENCH_DATA_ROOT=/data/ben/vllm_bench   # optional; default path uses /data/ben

    # A) Stock vLLM only — sweep until requests fail (KV / scheduler pressure)
    python benchmarks/kv_pipelining/phase1_baseline_wall.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --input-len 512 --output-len 512 \\
        --until-kv-pressure --conc-start 8 --conc-max 512 --conc-mult 2 \\
        --phases baseline

    # B) Baseline + pipelined bench connector (same concurrency plan, second server)
    python benchmarks/kv_pipelining/phase1_baseline_wall.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --input-len 512 --output-len 512 \\
        --until-kv-pressure --conc-start 8 --conc-max 256 --conc-mult 2 \\
        --phases both --k-window 4 --pipelined-gpu-mem-cap 0.72

    # C) Same as (A) but record Nsight Systems timeline for the **server** (CUDA / memory)
    python benchmarks/kv_pipelining/phase1_baseline_wall.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --input-len 512 --output-len 512 \\
        --until-kv-pressure --conc-start 8 --conc-max 512 --conc-mult 2 \\
        --phases baseline --with-nsys

    # Default report paths with --phases both: <result-dir>/nsys_baseline.nsys-rep and
    # nsys_pipelined.nsys-rep. For a custom --nsys-output-prefix, use separate runs or
    # distinct prefixes so the second phase does not overwrite the first.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

# Reuse streaming + prompt helpers from decode_only_bench
_KPB = Path(__file__).resolve().parent.parent / "kv_pipelining_bench"
if str(_KPB) not in sys.path:
    sys.path.insert(0, str(_KPB))

from decode_only_bench import (  # noqa: E402
    _generate_prompt,
    _get_model_name,
    _send_streaming_request,
    _wait_for_server,
)

try:
    from hf_env import apply_hf_token_from_user_file, hf_token_file_path  # noqa: E402
except ImportError:
    apply_hf_token_from_user_file = lambda: False  # noqa: E731
    hf_token_file_path = lambda: ""  # noqa: E731

# Default storage: user's large volume (not $HOME) — avoids HF cache quota on home.
def _default_data_root() -> Path:
    env = os.environ.get("VLLM_BENCH_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    return Path("/data/ben/vllm_bench")


def _server_env(download_dir: str | None) -> dict[str, str]:
    """Env for api_server (and forked EngineCore): avoid $HOME quota + hf_xet flakes."""
    env = dict(os.environ)
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    if download_dir:
        root = Path(download_dir).resolve()
        env["HF_HOME"] = str(root / "hf_home")
        env["HF_HUB_CACHE"] = str(root / "hub")
        env["TRANSFORMERS_CACHE"] = str(root / "transformers")
    return env


def _percentile_sorted(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = max(0, min(len(sorted_vals) - 1, int(round((len(sorted_vals) - 1) * q))))
    return sorted_vals[idx]


ITL_PERCENTILE_MIN_GAP_S = 1e-4


def _wrap_server_with_nsys_profile(
    inner_cmd: list[str],
    *,
    nsys_bin: str,
    output_prefix: Path,
    trace: str,
    extra_argv: list[str],
) -> list[str]:
    """Wrap the api_server command with `nsys profile` so CUDA work is captured."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    out = str(output_prefix.resolve())
    prefix: list[str] = [
        nsys_bin,
        "profile",
        "-t",
        trace,
        "-o",
        out,
        "--force-overwrite=true",
        *extra_argv,
        "--",
    ]
    return prefix + inner_cmd


def _collect_nsys_stats_from_report(
    rep_path: Path,
    result_dir: Path,
    phase_tag: str,
) -> dict[str, Any]:
    """
    After the server exits, run `nsys stats` to pull built-in JSON reports into the
    result bundle (full JSON + paths to on-disk sidecars).
    """
    result: dict[str, Any] = {"report_file": str(rep_path.resolve())}
    if not rep_path.is_file():
        result["error"] = (
            f"Nsight report missing: {rep_path} "
            "(profiling wrapper may have failed before creating the session)."
        )
        return result

    nsys = shutil.which("nsys")
    if not nsys:
        result["error"] = "`nsys` not found for nsys stats post-processing."
        return result

    export_dir = result_dir / "nsys_stats_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    report_names = (
        "cuda_gpu_sum",
        "cuda_gpu_mem_time_sum",
        "cuda_gpu_mem_size_sum",
        "cuda_api_sum",
    )
    exports: dict[str, Any] = {}

    for rname in report_names:
        base = export_dir / f"{phase_tag}_{rname}"
        cmd = [
            nsys,
            "stats",
            str(rep_path.resolve()),
            "--force-export",
            "true",
            "--report",
            rname,
            "--format",
            "json",
            "--output",
            str(base),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            exports[rname] = {"error": str(exc)}
            continue

        json_paths = [
            Path(str(base) + ".json"),
            base.with_suffix(".json"),
            base if base.suffix == ".json" else Path(str(base) + ".json"),
        ]
        json_paths.extend(sorted(export_dir.glob(f"{phase_tag}_{rname}*.json")))
        parsed: Any = None
        json_file: Path | None = None
        for jp in json_paths:
            if jp.is_file():
                try:
                    parsed = json.loads(jp.read_text(encoding="utf-8", errors="replace"))
                    json_file = jp
                    break
                except (json.JSONDecodeError, OSError):
                    parsed = None

        if parsed is None and proc.stdout.strip():
            try:
                parsed = json.loads(proc.stdout)
                json_file = None
            except json.JSONDecodeError:
                parsed = {"_raw_stdout_preview": proc.stdout[:8000]}

        if parsed is None:
            exports[rname] = {
                "error": proc.stderr.strip()
                or proc.stdout.strip()
                or f"exit code {proc.returncode}",
                "exit_code": proc.returncode,
            }
            continue

        exports[rname] = {
            "json_path": str(json_file) if json_file is not None else None,
            "data": parsed,
        }

    result["exports"] = exports
    result["note"] = (
        "Structured CUDA/API/mem summaries from `nsys stats`. Open .nsys-rep in the GUI "
        "for full timeline; these tables are convenient for automated reports."
    )
    return result


def _nvidia_smi_memory_mib() -> dict[str, Any] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        if len(parts) >= 3:
            return {
                "memory_used_mib": float(parts[0]),
                "memory_total_mib": float(parts[1]),
                "sm_util_percent": float(parts[2]),
            }
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, ValueError):
        pass
    return None


def _nvidia_smi_extended_metrics() -> dict[str, Any] | None:
    """Best-effort: util, mem util, clocks, power. Not a substitute for HBM BW profiling."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,"
                "clocks.current.memory,clocks.current.graphics,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        keys = [
            "utilization_gpu_pct",
            "utilization_memory_pct",
            "memory_used_mib",
            "memory_total_mib",
            "clocks_memory_mhz",
            "clocks_graphics_mhz",
            "power_draw_w",
            "temperature_gpu_c",
        ]
        if len(parts) >= len(keys):
            row: dict[str, Any] = {}
            for k, v in zip(keys, parts[: len(keys)], strict=False):
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            row["sampled_at_s"] = time.time()
            return row
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, ValueError):
        pass
    return None


def _launch_pipelined_decode_bench_server(
    model: str,
    port: int,
    max_model_len: int,
    gpu_mem_util: float,
    max_num_seqs: int,
    k_window: int,
    log_file: str | None,
    enforce_eager: bool,
    pipelined_mem_cap: float = 0.72,
    download_dir: str | None = None,
    nsys_bin: str | None = None,
    nsys_output_prefix: Path | None = None,
    nsys_trace: str = "cuda,nvtx,osrt,cublas,cudnn",
    nsys_extra_argv: list[str] | None = None,
) -> subprocess.Popen:
    kv_config = {
        "kv_connector": "PipelinedDecodeBenchConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"fill_mean": 0.015, "fill_std": 0.0},
    }

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
        str(min(float(gpu_mem_util), float(pipelined_mem_cap))),
        "--disable-log-requests",
        "--dtype",
        "auto",
        "--max-num-seqs",
        str(max_num_seqs),
        "--decode-only-pipelining",
        "--kv-k-window",
        str(k_window),
        "--kv-transfer-config",
        json.dumps(kv_config),
        "--no-enable-prefix-caching",
    ]
    if download_dir:
        cmd.extend(["--download-dir", download_dir])
    if enforce_eager:
        cmd.append("--enforce-eager")
    if nsys_bin and nsys_output_prefix is not None:
        cmd = _wrap_server_with_nsys_profile(
            cmd,
            nsys_bin=nsys_bin,
            output_prefix=nsys_output_prefix,
            trace=nsys_trace,
            extra_argv=list(nsys_extra_argv or ()),
        )
    log_fh = open(log_file, "w") if log_file else subprocess.DEVNULL
    return subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env=_server_env(download_dir),
    )


def _launch_baseline_server(
    model: str,
    port: int,
    max_model_len: int,
    gpu_mem_util: float,
    max_num_seqs: int,
    log_file: str | None,
    enforce_eager: bool,
    download_dir: str | None = None,
    nsys_bin: str | None = None,
    nsys_output_prefix: Path | None = None,
    nsys_trace: str = "cuda,nvtx,osrt,cublas,cudnn",
    nsys_extra_argv: list[str] | None = None,
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
    if download_dir:
        cmd.extend(["--download-dir", download_dir])
    if enforce_eager:
        cmd.append("--enforce-eager")
    if nsys_bin and nsys_output_prefix is not None:
        cmd = _wrap_server_with_nsys_profile(
            cmd,
            nsys_bin=nsys_bin,
            output_prefix=nsys_output_prefix,
            trace=nsys_trace,
            extra_argv=list(nsys_extra_argv or ()),
        )
    log_fh = open(log_file, "w") if log_file else subprocess.DEVNULL
    return subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env=_server_env(download_dir),
    )


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


def _add_decode_latency_aliases(row: dict[str, Any]) -> dict[str, Any]:
    """TPOT/TBT aliases: steady decode step ≈ ITL between streamed chunks."""
    for src, dst in (
        ("pooled_itl_s_mean_filtered", "tpot_decode_s_mean_filtered"),
        ("pooled_itl_s_mean", "tpot_decode_s_mean"),
        ("pooled_itl_s_p50", "tpot_decode_s_p50"),
        ("pooled_itl_s_p95", "tpot_decode_s_p95"),
        ("pooled_itl_s_p99", "tpot_decode_s_p99"),
    ):
        if src in row and row[src] is not None:
            row[dst] = row[src]
    for src, dst in (
        ("pooled_itl_s_mean_filtered", "tbt_decode_s_mean_filtered"),
        ("pooled_itl_s_p50", "tbt_decode_s_p50"),
        ("pooled_itl_s_p95", "tbt_decode_s_p95"),
        ("pooled_itl_s_p99", "tbt_decode_s_p99"),
    ):
        if src in row and row[src] is not None:
            row[dst] = row[src]
    row["metrics_note"] = (
        "tpot_decode_* / tbt_decode_* mirror pooled_itl_* (decode ITL ≈ TPOT/TBT)"
    )
    return row


def run_concurrent_streaming_decode_metrics(
    port: int,
    model_name: str,
    input_len: int,
    output_len: int,
    concurrency: int,
    request_timeout: int,
) -> dict[str, Any]:
    prompt = _generate_prompt(input_len)
    results: list[dict[str, Any]] = []
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
            )
            for _ in range(concurrency)
        ]
        for f in as_completed(futures):
            results.append(f.result())
    elapsed = time.monotonic() - t0

    oks = [r for r in results if r.get("success")]
    successes = len(oks)
    failures = concurrency - successes
    total_tokens = sum(int(r.get("total_tokens", 0) or 0) for r in oks)
    errors = [r.get("error", "?") for r in results if not r.get("success")]

    all_itl: list[float] = []
    for r in oks:
        samples = r.get("itl_samples")
        if isinstance(samples, list) and samples:
            all_itl.extend(float(x) for x in samples)

    all_itl.sort()
    n = len(all_itl)
    pooled_mean = sum(all_itl) / n if n else None
    pooled_p50_raw = _percentile_sorted(all_itl, 0.50)
    pooled_p95_raw = _percentile_sorted(all_itl, 0.95)
    pooled_p99_raw = _percentile_sorted(all_itl, 0.99)

    meaningful = [x for x in all_itl if x >= ITL_PERCENTILE_MIN_GAP_S]
    if len(meaningful) < max(1, n // 20):
        meaningful = list(all_itl)
    meaningful.sort()
    nm = len(meaningful)
    pooled_mean_filtered = sum(meaningful) / nm if nm else None
    pooled_p50 = _percentile_sorted(meaningful, 0.50) if nm else None
    pooled_p95 = _percentile_sorted(meaningful, 0.95) if nm else None
    pooled_p99 = _percentile_sorted(meaningful, 0.99) if nm else None

    mean_itls = [
        float(r["mean_itl_s"]) for r in oks if r.get("mean_itl_s") is not None
    ]
    mean_of_per_request_mean_itl = (
        sum(mean_itls) / len(mean_itls) if mean_itls else None
    )

    row: dict[str, Any] = {
        "concurrency": concurrency,
        "successes": successes,
        "failures": failures,
        "elapsed_s": round(elapsed, 3),
        "total_output_tokens": total_tokens,
        "throughput_tok_per_s": (
            round(total_tokens / elapsed, 2) if elapsed > 0 else 0.0
        ),
        "errors": errors[:10],
        "pooled_itl_s_mean": round(pooled_mean, 6) if pooled_mean is not None else None,
        "pooled_itl_s_mean_filtered": (
            round(pooled_mean_filtered, 6)
            if pooled_mean_filtered is not None
            else None
        ),
        "pooled_itl_s_p50_raw": (
            round(pooled_p50_raw, 6) if pooled_p50_raw is not None else None
        ),
        "pooled_itl_s_p95_raw": (
            round(pooled_p95_raw, 6) if pooled_p95_raw is not None else None
        ),
        "pooled_itl_s_p99_raw": (
            round(pooled_p99_raw, 6) if pooled_p99_raw is not None else None
        ),
        "pooled_itl_s_p50": round(pooled_p50, 6) if pooled_p50 is not None else None,
        "pooled_itl_s_p95": round(pooled_p95, 6) if pooled_p95 is not None else None,
        "pooled_itl_s_p99": round(pooled_p99, 6) if pooled_p99 is not None else None,
        "pooled_itl_percentile_min_gap_s": ITL_PERCENTILE_MIN_GAP_S,
        "pooled_itl_meaningful_sample_count": nm,
        "pooled_itl_sample_count": n,
        "mean_of_request_mean_itl_s": (
            round(mean_of_per_request_mean_itl, 6)
            if mean_of_per_request_mean_itl is not None
            else None
        ),
        "note": "ITL = inter-token latency (decode); TTFT omitted from primary table",
    }
    return _add_decode_latency_aliases(row)


def _conc_iterator_until_pressure(
    start: int,
    conc_max: int,
    mult: float | None,
    step: int | None,
) -> Iterator[int]:
    c = start
    while c <= conc_max:
        yield c
        if step is not None and step > 0:
            c = c + step
        elif mult is not None and mult > 1.0:
            c = max(c + 1, int(math.ceil(c * mult)))
        else:
            c = c * 2


def _parse_args() -> argparse.Namespace:
    dr = _default_data_root()
    p = argparse.ArgumentParser(
        description=(
            "Phase 1 wall: stock vLLM decode metrics until KV pressure; optional pipelined CPU push"
        ),
    )
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--port", type=int, default=8220)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=512)
    p.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[8, 16, 32, 48, 64, 96, 128],
        help="Concurrency levels when not using --until-kv-pressure.",
    )
    p.add_argument(
        "--until-kv-pressure",
        action="store_true",
        help=(
            "Increase concurrency until some streaming requests fail (KV block "
            "allocation / backlog / HTTP errors). Use --conc-start/--conc-max and "
            "either --conc-mult or --conc-step."
        ),
    )
    p.add_argument("--conc-start", type=int, default=8)
    p.add_argument("--conc-max", type=int, default=512)
    p.add_argument(
        "--conc-mult",
        type=float,
        default=2.0,
        help="Geometric growth for --until-kv-pressure (ignored if --conc-step set).",
    )
    p.add_argument(
        "--conc-step",
        type=int,
        default=0,
        help="If >0, add this each step instead of multiplying by --conc-mult.",
    )
    p.add_argument("--request-timeout", type=int, default=600)
    p.add_argument("--startup-timeout", type=int, default=900)
    p.add_argument(
        "--data-root",
        type=Path,
        default=dr,
        help=f"Root for HF cache + default runs (default: {dr}). Env: VLLM_BENCH_DATA_ROOT.",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Subdirectory under data-root/runs (default: timestamp).",
    )
    p.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Explicit output directory (overrides --data-root/--run-name).",
    )
    p.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="HF + vLLM weights cache (default: <data-root>/hf_cache).",
    )
    p.add_argument(
        "--enforce-eager",
        dest="enforce_eager",
        action="store_true",
        default=False,
        help="Pass --enforce-eager to server.",
    )
    p.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="Allow CUDA graphs when supported.",
    )
    p.add_argument(
        "--gpu-snapshot",
        action="store_true",
        help="nvidia-smi memory snapshot before/after each concurrency step.",
    )
    p.add_argument(
        "--gpu-extended-metrics",
        action="store_true",
        help="Sample utilization/memory clocks/power each step (not HBM BW — see docstring).",
    )
    p.add_argument(
        "--phases",
        choices=("baseline", "pipelined", "both"),
        default="baseline",
        help=(
            "baseline=stock vLLM only; pipelined=PipelinedDecodeBenchConnector only; "
            "both=baseline then pipelined compare."
        ),
    )
    p.add_argument(
        "--pipelined",
        action="store_true",
        help=(
            "[Deprecated] same as --phases pipelined — kept for backward compatibility."
        ),
    )
    p.add_argument("--k-window", type=int, default=4)
    p.add_argument(
        "--pipelined-gpu-mem-cap",
        type=float,
        default=0.72,
        help="Cap GPU memory util when running pipelined connector (window buffers).",
    )
    p.add_argument(
        "--continue-after-failure",
        action="store_true",
        help=(
            "With --until-kv-pressure, keep increasing concurrency after failures "
            "(default: stop after first failing level)."
        ),
    )
    p.add_argument(
        "--with-nsys",
        dest="with_nsys",
        action="store_true",
        help=(
            "Wrap the vLLM api_server with `nsys profile` (Nsight Systems). "
            "Captures CUDA/kernel/memory timeline for the server process; report "
            "path is recorded in results JSON."
        ),
    )
    p.add_argument(
        "--nsys-trace",
        default="cuda,nvtx,osrt,cublas,cudnn",
        help="Arguments for nsys profile -t (trace categories).",
    )
    p.add_argument(
        "--nsys-output-prefix",
        type=Path,
        default=None,
        help=(
            "Output path prefix for nsys -o (file will be <prefix>.nsys-rep). "
            "Default: <result-dir>/nsys_<baseline|pipelined>."
        ),
    )
    p.add_argument(
        "--nsys-extra",
        type=str,
        default="",
        help=(
            "Extra CLI tokens for `nsys profile` before `--` "
            '(e.g. \'--capture-range=none\' — use shell-style quoting).'
        ),
    )
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, str | None]:
    data_root: Path = args.data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.result_dir is not None:
        result_dir = args.result_dir.expanduser().resolve()
    else:
        result_dir = (data_root / "runs" / run_name).resolve()

    result_dir.mkdir(parents=True, exist_ok=True)

    if args.download_dir is not None:
        dl_path = args.download_dir.expanduser().resolve()
    else:
        dl_path = data_root / "hf_cache"
    dl_path.mkdir(parents=True, exist_ok=True)
    dl = str(dl_path)

    return result_dir, dl


def _max_num_seqs_for_server(args: argparse.Namespace) -> int:
    if args.until_kv_pressure:
        return max(args.conc_max, args.conc_start)
    return max(args.concurrency)


def _concurrency_list_for_fixed(args: argparse.Namespace) -> list[int]:
    return sorted(args.concurrency)


def _run_single_phase(
    *,
    args: argparse.Namespace,
    result_dir: Path,
    dl: str | None,
    pipelined: bool,
    max_num_seqs: int,
    phase_tag: str,
) -> tuple[list[dict[str, Any]], Path]:
    ns_bin = shutil.which("nsys") if args.with_nsys else None
    ns_prefix: Path | None = None
    ns_extra = (
        shlex.split(args.nsys_extra.strip())
        if args.with_nsys and args.nsys_extra.strip()
        else []
    )
    if args.with_nsys:
        if ns_bin is None:
            print(
                "ERROR: --with-nsys requires Nsight Systems CLI `nsys` on PATH "
                "(CUDA toolkit / NVIDIA Nsight Systems).",
                file=sys.stderr,
            )
            sys.exit(2)
        if args.nsys_output_prefix is not None:
            ns_prefix = args.nsys_output_prefix.expanduser().resolve()
        else:
            ns_prefix = result_dir / f"nsys_{phase_tag}"

    if pipelined:
        log_path = result_dir / f"server_{phase_tag}.log"
        title = (
            "Pipelined decode bench (CPU↔GPU imitation — PipelinedDecodeBenchConnector)"
        )
        phase_id = "pipelined_decode_bench_connector"
        enforce_eager = True if pipelined else bool(args.enforce_eager)
        proc = _launch_pipelined_decode_bench_server(
            model=args.model,
            port=args.port,
            max_model_len=args.max_model_len,
            gpu_mem_util=args.gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            k_window=args.k_window,
            log_file=str(log_path),
            enforce_eager=enforce_eager,
            pipelined_mem_cap=args.pipelined_gpu_mem_cap,
            download_dir=dl,
            nsys_bin=ns_bin,
            nsys_output_prefix=ns_prefix,
            nsys_trace=args.nsys_trace,
            nsys_extra_argv=ns_extra,
        )
    else:
        log_path = result_dir / f"server_{phase_tag}.log"
        title = "Baseline — stock vLLM (all KV on GPU, no KV connector)"
        phase_id = "baseline_stock_vllm"
        proc = _launch_baseline_server(
            model=args.model,
            port=args.port,
            max_model_len=args.max_model_len,
            gpu_mem_util=args.gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            log_file=str(log_path),
            enforce_eager=bool(args.enforce_eager),
            download_dir=dl,
            nsys_bin=ns_bin,
            nsys_output_prefix=ns_prefix,
            nsys_trace=args.nsys_trace,
            nsys_extra_argv=ns_extra,
        )

    rows: list[dict[str, Any]] = []
    stop_on_fail = args.until_kv_pressure and not args.continue_after_failure

    try:
        print("=" * 72)
        print(title)
        print("=" * 72)
        print(f"  log: {log_path}")
        print(f"  max_num_seqs (server): {max_num_seqs}")
        if args.with_nsys and ns_prefix is not None:
            print(f"  Nsight Systems: server wrapped with nsys profile")
            print(f"    report prefix: {ns_prefix}  (→ {ns_prefix}.nsys-rep)")
            print(f"    trace -t:      {args.nsys_trace}")

        print(f"Waiting for server ...")
        if not _wait_for_server(args.port, args.startup_timeout):
            print("ERROR: server did not become healthy. See log.")
            sys.exit(1)
        model_name = _get_model_name(args.port)
        print(f"Ready: {model_name}\n")

        if args.until_kv_pressure:
            step = args.conc_step if args.conc_step > 0 else None
            mult = None if step else float(args.conc_mult)
            for conc in _conc_iterator_until_pressure(
                args.conc_start, args.conc_max, mult, step
            ):
                snap_b = _nvidia_smi_memory_mib() if args.gpu_snapshot else None
                ext_b = _nvidia_smi_extended_metrics() if args.gpu_extended_metrics else None
                row = run_concurrent_streaming_decode_metrics(
                    args.port,
                    model_name,
                    args.input_len,
                    args.output_len,
                    conc,
                    args.request_timeout,
                )
                snap_a = _nvidia_smi_memory_mib() if args.gpu_snapshot else None
                ext_a = _nvidia_smi_extended_metrics() if args.gpu_extended_metrics else None
                if args.gpu_snapshot:
                    row["nvidia_smi_before"] = snap_b
                    row["nvidia_smi_after"] = snap_a
                if args.gpu_extended_metrics:
                    row["nvidia_smi_extended_before"] = ext_b
                    row["nvidia_smi_extended_after"] = ext_a
                row["sweep_mode"] = "until_kv_pressure"
                rows.append(row)

                if row["failures"] == 0:
                    m_f = row.get("pooled_itl_s_mean_filtered") or row.get(
                        "pooled_itl_s_mean"
                    )
                    print(
                        f"  conc={conc:4d}  ok  tok/s={row['throughput_tok_per_s']:8.1f}  "
                        f"TPOT p50 ms={((row.get('tpot_decode_s_p50') or 0))*1000:.2f}  "
                        f"n_itl={row['pooled_itl_sample_count']}"
                    )
                else:
                    print(
                        f"  conc={conc:4d}  FAIL {row['failures']}/{conc}  "
                        f"errors: {row['errors'][:2]}"
                    )
                    row["kv_pressure_or_client_errors"] = True
                    if stop_on_fail:
                        break
        else:
            for conc in _concurrency_list_for_fixed(args):
                snap_b = _nvidia_smi_memory_mib() if args.gpu_snapshot else None
                ext_b = _nvidia_smi_extended_metrics() if args.gpu_extended_metrics else None
                row = run_concurrent_streaming_decode_metrics(
                    args.port,
                    model_name,
                    args.input_len,
                    args.output_len,
                    conc,
                    args.request_timeout,
                )
                snap_a = _nvidia_smi_memory_mib() if args.gpu_snapshot else None
                ext_a = _nvidia_smi_extended_metrics() if args.gpu_extended_metrics else None
                if args.gpu_snapshot:
                    row["nvidia_smi_before"] = snap_b
                    row["nvidia_smi_after"] = snap_a
                if args.gpu_extended_metrics:
                    row["nvidia_smi_extended_before"] = ext_b
                    row["nvidia_smi_extended_after"] = ext_a
                row["sweep_mode"] = "fixed_concurrency_list"
                rows.append(row)

                if row["failures"] == 0:
                    m_f = row.get("pooled_itl_s_mean_filtered") or row.get(
                        "pooled_itl_s_mean"
                    )
                    print(
                        f"  conc={conc:4d}  ok  tok/s={row['throughput_tok_per_s']:8.1f}  "
                        f"TPOT p50 ms={((row.get('tpot_decode_s_p50') or 0))*1000:.2f}  "
                        f"n_itl={row['pooled_itl_sample_count']}"
                    )
                else:
                    print(
                        f"  conc={conc:4d}  FAIL {row['failures']}/{conc}  "
                        f"errors: {row['errors'][:2]}"
                    )

    finally:
        print("\nStopping server ...")
        _kill_server(proc)
        time.sleep(2)

    nsys_stats_bundle: dict[str, Any] | None = None
    if args.with_nsys and ns_prefix is not None:
        rep_file = Path(str(ns_prefix) + ".nsys-rep")
        print(
            f"\nNsight: exporting tables via `nsys stats` ← {rep_file.name} ...",
            flush=True,
        )
        nsys_stats_bundle = _collect_nsys_stats_from_report(
            rep_file, result_dir, phase_tag
        )
        if nsys_stats_bundle.get("error"):
            print(f"  warning: {nsys_stats_bundle['error']}", flush=True)
        elif isinstance(nsys_stats_bundle.get("exports"), dict):
            ok = sum(
                1
                for v in nsys_stats_bundle["exports"].values()
                if isinstance(v, dict) and "data" in v
            )
            print(f"  nsys stats: merged {ok} report(s) into results JSON.", flush=True)

    summary: dict[str, Any] = {
        "phase": phase_id,
        "pipelined_mode": pipelined,
        "config": {
            "model": args.model,
            "max_model_len": args.max_model_len,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": True if pipelined else bool(args.enforce_eager),
            "download_dir": dl,
            "data_root": str(args.data_root),
            "until_kv_pressure": bool(args.until_kv_pressure),
            "conc_start": args.conc_start,
            "conc_max": args.conc_max,
            "conc_mult": args.conc_mult,
            "conc_step": args.conc_step,
            "k_window": args.k_window if pipelined else None,
            "pipelined_gpu_mem_cap": args.pipelined_gpu_mem_cap if pipelined else None,
            "with_nsys": bool(args.with_nsys),
            "nsys_trace": args.nsys_trace if args.with_nsys else None,
            "nsys_extra": args.nsys_extra.strip() or None if args.with_nsys else None,
        },
        "rows": rows,
        "nsight_systems": (
            {
                "enabled": True,
                "cli_profile": "nsys profile (wraps api_server)",
                "cli_stats": (
                    "nsys stats — cuda_gpu_sum, cuda_gpu_mem_time_sum, "
                    "cuda_gpu_mem_size_sum, cuda_api_sum → JSON in results + "
                    f"{result_dir / 'nsys_stats_export'}"
                ),
                "report_path": f"{ns_prefix}.nsys-rep" if ns_prefix is not None else None,
                "report_prefix": str(ns_prefix) if ns_prefix is not None else None,
                "stats_tables": nsys_stats_bundle,
                "note": (
                    "Timeline: open .nsys-rep in Nsight Systems GUI. Structured tables "
                    "from `nsys stats` are embedded under stats_tables for automated reporting."
                ),
            }
            if args.with_nsys
            else {"enabled": False}
        ),
        "profiling_hint": {
            "hbm_bandwidth": (
                "With --with-nsys, use the .nsys-rep timeline (Memory / CUDA / NVTX) "
                "to analyze copies and kernels. `stats_tables` summarizes CUDA/mem/API time; "
                "for per-kernel DRAM counters use Nsight Compute on selected kernels."
            ),
            "gpu_extended_metrics_in_json": (
                "utilization.gpu, utilization.memory, memory clocks — useful for "
                "correlation, not DRAM bandwidth."
            ),
        },
    }
    out_json = result_dir / f"results_{phase_tag}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")

    csv_path = result_dir / f"results_{phase_tag}.csv"
    if rows:
        flat_keys = [
            "concurrency",
            "successes",
            "failures",
            "throughput_tok_per_s",
            "tpot_decode_s_p50",
            "tpot_decode_s_p95",
            "tpot_decode_s_p99",
            "pooled_itl_s_mean_filtered",
            "pooled_itl_sample_count",
            "elapsed_s",
            "total_output_tokens",
            "kv_pressure_or_client_errors",
        ]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in flat_keys})
        print(f"Wrote {csv_path}")

    return rows, out_json


def main() -> None:
    args = _parse_args()
    if args.pipelined:
        args.phases = "pipelined"

    result_dir, dl = _resolve_paths(args)
    if apply_hf_token_from_user_file():
        print(f"HF token: {hf_token_file_path()}")

    max_num_seqs = _max_num_seqs_for_server(args)

    print(f"Data root:     {args.data_root}")
    print(f"HF / weights:  {dl}")
    print(f"Result dir:    {result_dir}")
    print(f"Phases:        {args.phases}")
    print(f"Model:         {args.model}")
    print(f"in/out tokens: {args.input_len} / {args.output_len}")
    print(f"GPU util:      {args.gpu_memory_utilization}")
    if args.until_kv_pressure:
        print(
            f"Sweep: until pressure  start={args.conc_start} max={args.conc_max} "
            f"mult={args.conc_mult} step={args.conc_step}"
        )
    else:
        print(f"Concurrency:   {sorted(args.concurrency)}")
    print()

    baseline_rows: list[dict[str, Any]] = []
    pipelined_rows: list[dict[str, Any]] = []

    if args.phases in ("baseline", "both"):
        baseline_rows, _ = _run_single_phase(
            args=args,
            result_dir=result_dir,
            dl=dl,
            pipelined=False,
            max_num_seqs=max_num_seqs,
            phase_tag="baseline",
        )

    if args.phases in ("pipelined", "both"):
        pipelined_rows, _ = _run_single_phase(
            args=args,
            result_dir=result_dir,
            dl=dl,
            pipelined=True,
            max_num_seqs=max_num_seqs,
            phase_tag="pipelined",
        )

    if args.phases == "both" and baseline_rows and pipelined_rows:
        by_c: dict[str, Any] = {}
        for br in baseline_rows:
            k = str(br["concurrency"])
            by_c[k] = {"baseline": br, "pipelined": None}
        for pr in pipelined_rows:
            k = str(pr["concurrency"])
            if k not in by_c:
                by_c[k] = {"baseline": None, "pipelined": None}
            by_c[k]["pipelined"] = pr
        combined = {
            "result_dir": str(result_dir),
            "by_concurrency": by_c,
            "note": (
                "Compare tpot_decode_s_p50 (or pooled_itl_s_p50) baseline vs pipelined "
                "for operational impact of CPU-push imitation."
            ),
        }
        cmp_path = result_dir / "compare_baseline_vs_pipelined.json"
        with open(cmp_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nWrote {cmp_path}")


if __name__ == "__main__":
    main()
