#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ITL-under-load sweep (pipelined decode) — concurrent streaming throughput.

Measures **inter-token (chunk-gap) latency while N requests run together**, not
isolated single-request ITL. Grid (default):

- K ∈ {5, 4, 3}
- (input, output) ∈ {(512,512), (1024,1024), (2048,2048), (4096,4096)}
- concurrency levels per server session: 1, 16, 128, 256

Each of the 12 sessions runs ``pipelined_decode_fast_bench`` once with four
throughput steps. HF cache: ``HF_HOME=<result-root>/hf_hub_workspace``.

Example::

  PYTHONPATH=/path/to/vllm python benchmarks/kv_pipelining_bench/itl_load_sweep.py \\
    --result-root /data/ben/kv_phase_a_internal/itl_load_v1 \\
    --model Qwen/Qwen2.5-0.5B-Instruct
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[misc, assignment]


def _resolve_repo_and_bench() -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    if (here / "pipelined_decode_fast_bench.py").is_file():
        return here.parents[2], here
    root = Path(os.environ.get("VLLM_REPO_ROOT", "/home/ben/vllm")).resolve()
    return root, root / "benchmarks/kv_pipelining_bench"


_REPO_ROOT, _BENCH_DIR = _resolve_repo_and_bench()
_FAST_BENCH = _BENCH_DIR / "pipelined_decode_fast_bench.py"

if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
from hf_env import apply_hf_token_from_user_file, hf_token_file_path  # noqa: E402

# Import shared sweep helpers from phase_a_k_sweep (same directory)
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "phase_a_k_sweep", _BENCH_DIR / "phase_a_k_sweep.py"
)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load phase_a_k_sweep")
_phase = importlib.util.module_from_spec(_spec)
sys.modules["phase_a_k_sweep"] = _phase
_spec.loader.exec_module(_phase)

MemoryGuard = _phase.MemoryGuard
effective_gpu_guard_frac = _phase.effective_gpu_guard_frac
max_model_len_for = _phase.max_model_len_for
_preflight_gpu = _phase._preflight_gpu
query_gpu_memory_mib = _phase.query_gpu_memory_mib

DEFAULT_RESULT_ROOT = Path("/data/ben/kv_phase_a_internal")

PURPOSE_NOTE = (
    "ITL under concurrent load: streaming throughput phase with per-request and "
    "global p99 ITL (not isolated single-request latency)."
)


@dataclass
class ItlSweepRun:
    run_id: int
    k_window: int
    input_len: int
    output_len: int
    concurrencies: list[int]


def iter_runs() -> list[ItlSweepRun]:
    conc = [1, 16, 128, 256]
    runs: list[ItlSweepRun] = []
    rid = 0
    for k in (5, 4, 3):
        for inp, out in (
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
        ):
            rid += 1
            runs.append(
                ItlSweepRun(
                    run_id=rid,
                    k_window=k,
                    input_len=inp,
                    output_len=out,
                    concurrencies=conc,
                )
            )
    return runs


def build_cmd(
    *,
    python: Path,
    run: ItlSweepRun,
    result_dir: Path,
    port: int,
    model: str,
    gpu_mem_util: float,
    gpu_index: int,
    startup_timeout: int,
    request_timeout: int,
) -> list[str]:
    mml = max_model_len_for(run.input_len, run.output_len)
    conc = [str(c) for c in run.concurrencies]
    return [
        str(python),
        str(_FAST_BENCH),
        "--model",
        model,
        "--no-auto-gpu-memory",
        "--gpu-memory-utilization",
        str(gpu_mem_util),
        "--gpu-index",
        str(gpu_index),
        "--k-window",
        str(run.k_window),
        "--input-len",
        str(run.input_len),
        "--output-len",
        str(run.output_len),
        "--concurrency",
        *conc,
        "--max-model-len",
        str(mml),
        "--result-dir",
        str(result_dir),
        "--port",
        str(port),
        "--startup-timeout",
        str(startup_timeout),
        "--request-timeout",
        str(request_timeout),
        "--memory-warn-used-frac",
        "0.995",
        "--memory-monitor-start-delay",
        "120",
        "--modes",
        "pipelined",
        "--concurrent-itl-throughput",
        "--skip-isolated-latency",
        "--warmup-requests",
        "0",
    ]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def parse_summary(result_dir: Path) -> dict[str, Any] | None:
    p = result_dir / "summary_all.json"
    if not p.is_file():
        return None
    with open(p) as f:
        return json.load(f)


def flatten_results(
    run: ItlSweepRun,
    summary: dict[str, Any] | None,
    *,
    wall_s: float,
    exit_code: int,
    guard_reason: str | None,
    result_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = {
        "phase": "itl_load",
        "purpose_note": PURPOSE_NOTE,
        "sweep_run_id": run.run_id,
        "k_window": run.k_window,
        "input_len": run.input_len,
        "output_len": run.output_len,
        "wall_s": wall_s,
        "exit_code": exit_code,
        "mem_guard_abort": guard_reason,
        "result_dir": str(result_dir),
    }
    pip = (summary or {}).get("pipelined") or {}
    lat = pip.get("latency") or {}
    if isinstance(lat, dict) and lat.get("skipped"):
        base["isolated_latency_skipped"] = True
    for tr in pip.get("throughput") or []:
        row = {
            **base,
            "concurrency": tr.get("concurrency"),
            "throughput_tok_per_s": tr.get("throughput_tok_per_s"),
            "throughput_successes": tr.get("successes"),
            "throughput_failures": tr.get("failures"),
            "concurrent_mean_itl_s": tr.get("concurrent_mean_itl_s_mean"),
            "concurrent_mean_p99_itl_s": tr.get("concurrent_mean_p99_itl_s"),
            "concurrent_global_p99_itl_s": tr.get("concurrent_global_p99_itl_s"),
            "concurrent_ttft_s_mean": tr.get("concurrent_ttft_s_mean"),
            "concurrent_itl_sample_count": tr.get("concurrent_itl_sample_count"),
            "bench_aborted": pip.get("aborted", False),
            "bench_abort_reason": pip.get("abort_reason"),
        }
        rows.append(row)
    if not rows:
        rows.append({**base, "concurrency": None, "note": "no_throughput_rows"})
    return rows


def _is_under_data_ben(path: Path) -> bool:
    try:
        p = path.resolve()
    except OSError:
        return False
    anchor = Path("/data/ben").resolve()
    return p == anchor or str(p).startswith(str(anchor) + os.sep)


def run_one(
    run: ItlSweepRun,
    *,
    python: Path,
    result_root: Path,
    port: int,
    model: str,
    gpu_mem_util: float,
    gpu_index: int,
    startup_timeout: int,
    request_timeout: int,
    gpu_guard_frac: float,
    cpu_guard_pct: float,
    guard_poll_s: float,
    guard_start_delay_s: float,
    skip_gpu_preflight: bool,
    min_free_gpu_mib: int,
    preflight_retries: int,
    preflight_retry_s: float,
    log_print: Any,
) -> int:
    max_conc = max(run.concurrencies) if run.concurrencies else 0
    eff_guard = effective_gpu_guard_frac(gpu_guard_frac, max_conc)
    if eff_guard != gpu_guard_frac:
        log_print(
            f"[sweep] max_concurrency={max_conc} → parent GPU guard frac "
            f"{gpu_guard_frac} → {eff_guard}",
            flush=True,
        )

    if not skip_gpu_preflight:
        _preflight_gpu(
            gpu_index,
            min_free_gpu_mib,
            retries=preflight_retries,
            retry_s=preflight_retry_s,
            log_fn=log_print,
            label="between-runs",
        )

    subdir = result_root / (
        f"run_{run.run_id:03d}_k{run.k_window}_i{run.input_len}_o{run.output_len}"
    )
    subdir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(
        python=python,
        run=run,
        result_dir=subdir,
        port=port,
        model=model,
        gpu_mem_util=gpu_mem_util,
        gpu_index=gpu_index,
        startup_timeout=startup_timeout,
        request_timeout=request_timeout,
    )
    log_print(
        f"\n{'=' * 70}\n"
        f"ITL-load run {run.run_id}: K={run.k_window} "
        f"in={run.input_len} out={run.output_len} conc={run.concurrencies}\n"
        f"  dir={subdir}\n"
        f"{'=' * 70}",
        flush=True,
    )
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    hf_home = result_root / "hf_hub_workspace"
    hf_home.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(hf_home)
    if apply_hf_token_from_user_file(env):
        log_print(f"[sweep] HF_TOKEN set from {hf_token_file_path()}", flush=True)

    proc_holder: dict[str, subprocess.Popen[Any] | None] = {"proc": None}
    guard = MemoryGuard(
        gpu_index=gpu_index,
        gpu_used_frac_max=eff_guard,
        cpu_percent_max=cpu_guard_pct,
        poll_interval_s=guard_poll_s,
        start_delay_s=guard_start_delay_s,
        proc_holder=proc_holder,
        log_fn=log_print,
    )
    guard.start()
    t0 = time.monotonic()
    exit_code = -1
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_REPO_ROOT),
            env=env,
            preexec_fn=os.setsid,
            stdout=None,
            stderr=None,
        )
        proc_holder["proc"] = proc
        exit_code = proc.wait()
    finally:
        guard.stop()
        if proc_holder.get("proc") is not None and proc_holder["proc"].poll() is None:
            try:
                os.killpg(os.getpgid(proc_holder["proc"].pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        proc_holder["proc"] = None

    wall_s = time.monotonic() - t0
    summary = parse_summary(subdir)
    rows = flatten_results(
        run,
        summary,
        wall_s=wall_s,
        exit_code=exit_code,
        guard_reason=guard.abort_reason,
        result_dir=subdir,
    )
    jsonl = result_root / "results.jsonl"
    for row in rows:
        append_jsonl(jsonl, row)
    log_print(
        f"[sweep] finished run {run.run_id} exit={exit_code} wall={wall_s:.1f}s "
        f"guard={guard.abort_reason!r}",
        flush=True,
    )
    return exit_code


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT / "itl_load_sweep",
        help="Output directory (HF cache = <root>/hf_hub_workspace)",
    )
    p.add_argument(
        "--allow-result-root-outside-data-ben",
        action="store_true",
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--base-port", type=int, default=8320)
    p.add_argument("--gpu-mem-util", type=float, default=0.80, dest="gpu_mem_util")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--startup-timeout", type=int, default=2400)
    p.add_argument("--request-timeout", type=int, default=1800)
    p.add_argument("--gpu-guard-frac", type=float, default=0.985)
    p.add_argument("--cpu-guard-percent", type=float, default=92.0)
    p.add_argument("--guard-poll-s", type=float, default=2.0)
    p.add_argument("--guard-start-delay-s", type=float, default=150.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-fail", action="store_true")
    p.add_argument("--skip-gpu-preflight", action="store_true")
    p.add_argument("--no-inter-run-preflight", action="store_true")
    p.add_argument("--min-free-gpu-mib", type=int, default=10_000)
    p.add_argument("--preflight-retries", type=int, default=5)
    p.add_argument("--preflight-retry-s", type=float, default=5.0)
    ns = p.parse_args()

    if not _FAST_BENCH.is_file():
        print(f"error: missing {_FAST_BENCH}", file=sys.stderr)
        sys.exit(2)

    root = ns.result_root
    if not ns.allow_result_root_outside_data_ben and not _is_under_data_ben(root):
        print(
            "error: --result-root must be under /data/ben (or pass "
            "--allow-result-root-outside-data-ben)",
            file=sys.stderr,
        )
        sys.exit(2)

    runs = iter_runs()
    print(f"[sweep] {len(runs)} server sessions × {len(runs[0].concurrencies)} "
          f"conc levels = {len(runs) * len(runs[0].concurrencies)} result rows", flush=True)

    if ns.dry_run:
        for r in runs:
            print(
                f"  run {r.run_id} K={r.k_window} {r.input_len}x{r.output_len} "
                f"conc={r.concurrencies}",
                flush=True,
            )
        return

    total, used = query_gpu_memory_mib(ns.gpu_index)
    if total > 0 and not ns.skip_gpu_preflight:
        free = total - used
        print(
            f"[sweep] preflight start: OK (free={free} MiB, used={used}/{total})",
            flush=True,
        )

    root.mkdir(parents=True, exist_ok=True)
    with open(root / "sweep_note.json", "w") as f:
        json.dump(
            {
                "grid": "K∈{5,4,3} × I/O∈{(512²)…(4096²)} × conc∈{1,16,128,256}",
                "metrics": "concurrent streaming ITL + tok/s per throughput step",
                "model": ns.model,
            },
            f,
            indent=2,
        )

    def log_print(*a: Any, **kw: Any) -> None:
        print(*a, **kw)

    port = ns.base_port
    failed = 0
    for run in runs:
        if not ns.skip_gpu_preflight and not ns.no_inter_run_preflight:
            _preflight_gpu(
                ns.gpu_index,
                ns.min_free_gpu_mib,
                retries=ns.preflight_retries,
                retry_s=ns.preflight_retry_s,
                log_fn=log_print,
                label="between-runs",
            )
        rc = run_one(
            run,
            python=ns.python,
            result_root=root,
            port=port,
            model=ns.model,
            gpu_mem_util=ns.gpu_mem_util,
            gpu_index=ns.gpu_index,
            startup_timeout=ns.startup_timeout,
            request_timeout=ns.request_timeout,
            gpu_guard_frac=ns.gpu_guard_frac,
            cpu_guard_pct=ns.cpu_guard_percent,
            guard_poll_s=ns.guard_poll_s,
            guard_start_delay_s=ns.guard_start_delay_s,
            skip_gpu_preflight=True,
            min_free_gpu_mib=ns.min_free_gpu_mib,
            preflight_retries=ns.preflight_retries,
            preflight_retry_s=ns.preflight_retry_s,
            log_print=log_print,
        )
        if rc != 0:
            failed += 1
            if ns.stop_on_fail:
                print(f"[sweep] abort (--stop-on-fail) rc={rc}", file=sys.stderr)
                sys.exit(rc)

    print(f"\n[sweep] complete. Results: {root / 'results.jsonl'}", flush=True)
    if failed:
        print(f"[sweep] warning: {failed} non-zero exit(s)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
