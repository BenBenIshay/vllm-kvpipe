#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copy may live under /data/ben when $HOME repo hits disk quota; run with:
#   python /data/ben/kv_phase_a_internal/phase_a_k_sweep.py --wave A ...
"""Phase A K×load sweep — internal R&D only.

Hugging Face cache: ``HF_HOME`` is set to ``<result-root>/hf_hub_workspace`` so
weights stay on the same filesystem as ``--result-root`` (use ``/data/ben/...``).

Concurrency vs parent GPU guard: vLLM uses ``max_num_seqs`` on the order of the
requested max concurrency. The KV block pool scales with that slot count; with
``--gpu-mem-util`` (e.g. 0.8), **nvidia-smi can report ~99% HBM used in normal
operation** — not necessarily a leak. The sweep raises the parent HBM fraction
ceiling for higher concurrency (see ``effective_gpu_guard_frac``).
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
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any


def _resolve_repo_and_bench() -> tuple[Path, Path]:
    """Support running from repo or from a copy under ``/data/ben/...``."""
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

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[misc, assignment]

PHASE_A_PURPOSE = (
    "Internal R&D sweep for adaptive-K design (dummy pipelined decode). "
    "Not for paper publication."
)

DEFAULT_RESULT_ROOT = Path("/data/ben/kv_phase_a_internal")


@dataclass
class SweepRun:
    wave: str
    run_id: int
    k_window: int
    input_len: int
    output_len: int
    concurrencies: list[int]


def query_gpu_memory_mib(gpu_index: int = 0) -> tuple[int, int]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return 0, 0
        parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
        if len(parts) < 2:
            return 0, 0
        return int(float(parts[0])), int(float(parts[1]))
    except (OSError, ValueError, subprocess.TimeoutExpired, IndexError):
        return 0, 0


class MemoryGuard:
    def __init__(
        self,
        *,
        gpu_index: int,
        gpu_used_frac_max: float,
        cpu_percent_max: float,
        poll_interval_s: float,
        start_delay_s: float,
        proc_holder: dict[str, subprocess.Popen[Any] | None],
        log_fn: Any,
    ) -> None:
        self._gpu_index = gpu_index
        self._gpu_max = gpu_used_frac_max
        self._cpu_max = cpu_percent_max
        self._interval = poll_interval_s
        self._start_delay_s = start_delay_s
        self._holder = proc_holder
        self._log = log_fn
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.abort_reason: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mem-guard")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self._interval + 2.0, self._start_delay_s + 5.0))

    def _kill_tree(self, reason: str) -> None:
        if self.abort_reason is not None:
            return
        self.abort_reason = reason
        self._log(f"[mem-guard] ABORT: {reason}", flush=True)
        p = self._holder.get("proc")
        if p is not None and p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError) as e:
                self._log(f"[mem-guard] killpg: {e}", flush=True)
        self._stop.set()

    def _loop(self) -> None:
        if self._start_delay_s > 0 and self._stop.wait(self._start_delay_s):
            return
        while not self._stop.wait(self._interval):
            p = self._holder.get("proc")
            if p is not None and p.poll() is not None:
                return
            total, used = query_gpu_memory_mib(self._gpu_index)
            if total > 0:
                frac = used / total
                if frac >= self._gpu_max:
                    self._kill_tree(
                        f"GPU HBM {used}/{total} MiB ({frac * 100:.1f}% ≥ "
                        f"{self._gpu_max * 100:.0f}%)"
                    )
                    return
            if psutil is not None:
                vm = psutil.virtual_memory()
                if vm.percent >= self._cpu_max:
                    self._kill_tree(
                        f"system RAM {vm.percent:.1f}% ≥ {self._cpu_max:.0f}% "
                        f"({vm.used / (1024**3):.1f} GiB used)"
                    )
                    return


def max_model_len_for(input_len: int, output_len: int) -> int:
    need = input_len + output_len + 256
    for cap in (4096, 8192, 16384, 32768, 65536):
        if cap >= need:
            return cap
    return 65536


def iter_wave_runs(wave: str) -> list[SweepRun]:
    wave = wave.upper()
    runs: list[SweepRun] = []
    rid = 0
    if wave == "A":
        # One concurrency per run: vLLM uses max_num_seqs=max(concurrency), so
        # [1,16,128] in a single run reserves 128 slots at the top (was 256; OOM risk).
        #
        # High c: KV pool can push nvidia-smi used/total to ~99% — parent MemoryGuard
        # uses effective_gpu_guard_frac (see run_one).
        for k in (6, 4, 2):
            for c in (1, 16, 128):
                rid += 1
                runs.append(
                    SweepRun(
                        wave="A",
                        run_id=rid,
                        k_window=k,
                        input_len=512,
                        output_len=512,
                        concurrencies=[c],
                    )
                )
    elif wave == "B":
        conc = [64]
        for inp, out, k in product(
            (256, 512, 1024, 4096),
            (512, 1024, 4096),
            (6, 5, 4, 3, 2),
        ):
            rid += 1
            runs.append(
                SweepRun(
                    wave="B",
                    run_id=rid,
                    k_window=k,
                    input_len=inp,
                    output_len=out,
                    concurrencies=conc,
                )
            )
    elif wave == "C":
        # Wave C: stress K-window + data movement (large I/O, high concurrency).
        conc = [128]
        for k in (6, 4, 2):
            rid += 1
            runs.append(
                SweepRun(
                    wave="C",
                    run_id=rid,
                    k_window=k,
                    input_len=1024,
                    output_len=4096,
                    concurrencies=conc,
                )
            )
    else:
        raise ValueError(f"unknown wave {wave!r} (use A, B, C, or ALL)")
    return runs


def effective_gpu_guard_frac(base: float, max_concurrency: int) -> float:
    """Parent-process HBM used/total ceiling that avoids false aborts.

    vLLM reserves KV blocks for up to ``max_num_seqs`` concurrent sequences
    (driven by ``--max-num-seqs`` / scheduler limits tied to load). Under high
    concurrency and ``--gpu-memory-utilization``, **most of HBM can be used
    legitimately**; ``nvidia-smi`` may show ~99% used while the server is healthy.

    We raise the guard fraction for higher max concurrency so the parent does not
    SIGTERM the bench thinking the GPU is "stuck full" when vLLM is simply using
    its configured pool.
    """
    if max_concurrency >= 128:
        return max(base, 0.9995)
    if max_concurrency >= 64:
        return max(base, 0.995)
    if max_concurrency >= 16:
        return max(base, 0.99)
    return base


def _is_under_data_ben(path: Path) -> bool:
    """Require sweep artifacts (HF cache, results) on /data/ben to avoid home quota."""
    try:
        p = path.resolve()
    except OSError:
        return False
    anchor = Path("/data/ben").resolve()
    return p == anchor or str(p).startswith(str(anchor) + os.sep)


def build_fast_bench_cmd(
    *,
    python: Path,
    run: SweepRun,
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
        "--no-concurrent-itl-throughput",
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
    run: SweepRun,
    summary: dict[str, Any] | None,
    *,
    wall_s: float,
    exit_code: int,
    guard_reason: str | None,
    result_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = {
        "phase": "A_internal",
        "purpose_note": PHASE_A_PURPOSE,
        "wave": run.wave,
        "run_id": run.run_id,
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
    if isinstance(lat, dict) and not lat.get("error"):
        base_lat = {
            "ttft_s_mean": lat.get("ttft_s_mean"),
            "mean_itl_s_mean": lat.get("mean_itl_s_mean"),
            "p99_itl_s_mean": lat.get("p99_itl_s_mean"),
        }
    else:
        base_lat = {
            "ttft_s_mean": None,
            "mean_itl_s_mean": None,
            "p99_itl_s_mean": None,
            "latency_error": lat.get("error") if isinstance(lat, dict) else str(lat),
        }
    for tr in pip.get("throughput") or []:
        row = {
            **base,
            **base_lat,
            "concurrency": tr.get("concurrency"),
            "throughput_tok_per_s": tr.get("throughput_tok_per_s"),
            "throughput_successes": tr.get("successes"),
            "throughput_failures": tr.get("failures"),
            "bench_aborted": pip.get("aborted", False),
            "bench_abort_reason": pip.get("abort_reason"),
        }
        rows.append(row)
    if not rows:
        rows.append({**base, **base_lat, "concurrency": None, "note": "no_throughput_rows"})
    return rows


def _preflight_gpu(
    gpu_index: int,
    min_free_mib: int,
    *,
    retries: int,
    retry_s: float,
    log_fn: Any,
    label: str,
) -> None:
    last_free = 0
    for attempt in range(retries):
        total, used = query_gpu_memory_mib(gpu_index)
        if total <= 0:
            return
        free = total - used
        last_free = free
        if free >= min_free_mib:
            if attempt > 0:
                log_fn(
                    f"[sweep] preflight {label}: OK after retry "
                    f"(free={free} MiB, used={used}/{total})",
                    flush=True,
                )
            else:
                log_fn(
                    f"[sweep] preflight {label}: OK (free={free} MiB, used={used}/{total})",
                    flush=True,
                )
            return
        if attempt < retries - 1:
            log_fn(
                f"[sweep] preflight {label}: GPU {gpu_index} only ~{free} MiB free "
                f"({used}/{total} MiB); attempt {attempt + 1}/{retries}, sleeping {retry_s:.1f}s "
                "(CUDA teardown or stray GPU processes)",
                flush=True,
            )
            time.sleep(retry_s)
    print(
        f"[sweep] error: preflight {label}: GPU {gpu_index} has only ~{last_free} MiB free "
        f"(need ≥{min_free_mib} MiB). Clear leaked allocations (nvidia-smi, "
        f"kill stray processes, or reboot).",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(3)


def run_one(
    run: SweepRun,
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
            f"{gpu_guard_frac} → {eff_guard} (vLLM KV pool scales with concurrency; "
            f"high HBM usage can be normal)",
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

    subdir = result_root / f"wave_{run.wave}_run_{run.run_id:03d}_k{run.k_window}_i{run.input_len}_o{run.output_len}"
    subdir.mkdir(parents=True, exist_ok=True)
    cmd = build_fast_bench_cmd(
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
        f"WAVE {run.wave} run {run.run_id}: K={run.k_window} "
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
        "--wave",
        choices=["A", "B", "C", "ALL"],
        default="A",
        help="A/B/C grids or ALL (= A then B then C)",
    )
    p.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Sweep output directory; HF cache = <result-root>/hf_hub_workspace. "
        "Must be under /data/ben unless --allow-result-root-outside-data-ben",
    )
    p.add_argument(
        "--allow-result-root-outside-data-ben",
        action="store_true",
        help="Allow --result-root outside /data/ben (e.g. /tmp for CI)",
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--base-port", type=int, default=8310)
    p.add_argument("--gpu-mem-util", type=float, default=0.80, dest="gpu_mem_util")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--startup-timeout", type=int, default=1800)
    p.add_argument("--request-timeout", type=int, default=600)
    p.add_argument(
        "--gpu-guard-frac",
        type=float,
        default=0.985,
        help="Base parent guard: kill if HBM used/total ≥ effective frac after start delay "
        "(effective frac is raised for high concurrency — see effective_gpu_guard_frac)",
    )
    p.add_argument("--cpu-guard-percent", type=float, default=92.0)
    p.add_argument("--guard-poll-s", type=float, default=2.0)
    p.add_argument(
        "--guard-start-delay-s",
        type=float,
        default=150.0,
        help="Skip parent GPU/RAM checks for this long (compile/load spike)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-fail", action="store_true")
    p.add_argument(
        "--skip-gpu-preflight",
        action="store_true",
        help="Skip all GPU free-HBM preflight checks (start and between runs)",
    )
    p.add_argument(
        "--no-inter-run-preflight",
        action="store_true",
        help="Skip preflight between runs (still runs initial preflight unless --skip-gpu-preflight)",
    )
    p.add_argument(
        "--min-free-gpu-mib",
        type=int,
        default=10_000,
        help="Preflight: require at least this many MiB free before each run",
    )
    p.add_argument(
        "--preflight-retries",
        type=int,
        default=5,
        help="Retries if GPU is temporarily full after previous run (CUDA teardown delay)",
    )
    p.add_argument(
        "--preflight-retry-s",
        type=float,
        default=5.0,
        help="Seconds to sleep between preflight retries",
    )
    ns = p.parse_args()

    if psutil is None:
        print("[sweep] warning: psutil missing; CPU guard off", file=sys.stderr, flush=True)
    if not _FAST_BENCH.is_file():
        print(f"error: missing {_FAST_BENCH}", file=sys.stderr)
        sys.exit(2)

    root = ns.result_root
    if not ns.allow_result_root_outside_data_ben and not _is_under_data_ben(root):
        print(
            "error: --result-root must be under /data/ben (HF cache + results avoid home "
            "quota). Example: --result-root /data/ben/kv_phase_a_internal/my_sweep\n"
            "Override with --allow-result-root-outside-data-ben for CI/tmp only.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)

    inter_run = not ns.no_inter_run_preflight

    if not ns.skip_gpu_preflight and not ns.dry_run:
        _preflight_gpu(
            ns.gpu_index,
            ns.min_free_gpu_mib,
            retries=ns.preflight_retries,
            retry_s=ns.preflight_retry_s,
            log_fn=print,
            label="start",
        )

    waves = ["A", "B", "C"] if ns.wave == "ALL" else [ns.wave]
    all_runs: list[SweepRun] = []
    for w in waves:
        all_runs.extend(iter_wave_runs(w))

    root.mkdir(parents=True, exist_ok=True)
    hf_ws = root / "hf_hub_workspace"
    note = {
        "purpose": PHASE_A_PURPOSE,
        "waves": waves,
        "n_runs": len(all_runs),
        "model": ns.model,
        "gpu_mem_util": ns.gpu_mem_util,
        "hf_home": str(hf_ws),
        "gpu_guard_frac_base": ns.gpu_guard_frac,
        "concurrency_gpu_guard_note": (
            "effective_gpu_guard_frac(base, max_conc) raises the parent HBM cap when "
            "max_concurrency is large; vLLM KV reservation can legitimately use ~99% HBM."
        ),
        "cpu_guard_percent": ns.cpu_guard_percent,
        "guard_poll_s": ns.guard_poll_s,
        "guard_start_delay_s": ns.guard_start_delay_s,
        "preflight_min_free_mib": ns.min_free_gpu_mib,
        "preflight_between_runs": inter_run and not ns.skip_gpu_preflight,
        "psutil_available": psutil is not None,
        "runs": [asdict(r) for r in all_runs],
    }
    with open(root / "sweep_note.json", "w") as f:
        json.dump(note, f, indent=2)
    print(json.dumps(note, indent=2, default=str))
    if ns.dry_run:
        return

    for i, run in enumerate(all_runs):
        port = ns.base_port + (i % 80)
        # Between-run preflight: skip first run (already checked at start), unless
        # --no-inter-run-preflight or --skip-gpu-preflight
        do_preflight_this_iteration = (
            not ns.skip_gpu_preflight
            and inter_run
            and i > 0
        )
        code = run_one(
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
            skip_gpu_preflight=ns.skip_gpu_preflight or not do_preflight_this_iteration,
            min_free_gpu_mib=ns.min_free_gpu_mib,
            preflight_retries=ns.preflight_retries,
            preflight_retry_s=ns.preflight_retry_s,
            log_print=print,
        )
        if code != 0 and ns.stop_on_fail:
            print(f"[sweep] stopping after exit {code}", flush=True)
            sys.exit(code)

    print(f"\n[sweep] complete. Results: {root / 'results.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
