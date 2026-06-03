#!/usr/bin/env python3
"""Run v5 interference sweep — reduced scenario set, manifest-driven workloads.

Changes vs v4
-------------
- BW=0 → no worker process (was: uncapped copy loop due to _pace_sleep short-circuit).
  This makes BW=0 a true "no injection" baseline.  Post-run, verify that BW=0 cells
  match the corresponding vanilla open-loop rows within noise.
- Scenario set reduced to 12 per workload (no mix, no H2D concurrency variants):
    baseline (no worker), H2D ×4 (8/16/32/56 Gbps), D2H ×4, HBM parallel ×3.
- Worker duration shortened to --worker-seconds (default 120) for grid scale.
- Bench num_prompts is rate-aware: max(--min-prompts, ceil(--steady-seconds × rate)).
- Workloads come from --workloads-json manifest (see --help); defaults to the two
  representative shapes selected for this run.
- One server per model (batches all rate/shape/scenario combos before teardown).
- nvidia-smi dmon sidecar per cell; summary stats stored in output JSONL.
- Output: injection_sweep_v5/ directory with injection_sweep_v5.jsonl + .csv.

Key BW=0 fix explanation
------------------------
In run_injection_sweep_v4.py, scenarios h2d_0_c1 and d2h_0_c1 spawn the worker
with --target-gbps 0.  In injection_stress_worker.py::_pace_sleep:

    def _pace_sleep(chunk_bytes, target_bps, elapsed):
        if target_bps <= 0:   # <-- short-circuits when target_bps == 0
            return

This means the copy loop runs with NO throttling — effectively at maximum PCIe
bandwidth.  So "BW=0" in v4 actually means "MAX BW", which is the worst case,
not the baseline.  This file fixes the issue by mapping target_gbps=0 to wcmd=None
(no worker process), identical to the existing hbm0 treatment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/ben/vllm")
THIS = Path(__file__).resolve().parent
PY = str(ROOT / ".venv/bin/python")
WORKER = THIS / "injection_stress_worker.py"

OUT_DEFAULT = Path(os.environ.get(
    "KV_SWEEP_OUT",
    str(THIS / "injection_sweep_v5"),
))
SLO_CSV = Path(os.environ.get(
    "KV_SLO_CSV",
    "/data/ben/kv_phase_a_internal/open_loop_eval_v1/length_dependent_slo_definitions_v2.csv",
))

HF_TO_TAB = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B-Instruct",
}

# Default representative workloads (one per model × phase, each replayed at
# all 5 request rates).
_DEFAULT_WORKLOADS = [
    # Llama prefill: input_len=4096
    {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
     "input_len": 4096, "output_len": 1,
     "request_rate": 16.0, "slo_kind": "ttft"},
    {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
     "input_len": 4096, "output_len": 1,
     "request_rate": 32.0, "slo_kind": "ttft"},
    {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
     "input_len": 4096, "output_len": 1,
     "request_rate": 64.0, "slo_kind": "ttft"},
    {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
     "input_len": 4096, "output_len": 1,
     "request_rate": 128.0, "slo_kind": "ttft"},
    {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
     "input_len": 4096, "output_len": 1,
     "request_rate": 256.0, "slo_kind": "ttft"},
    # Llama decode: (1024, 1024)
    {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
     "input_len": 1024, "output_len": 1024,
     "request_rate": 16.0, "slo_kind": "itl"},
    {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
     "input_len": 1024, "output_len": 1024,
     "request_rate": 32.0, "slo_kind": "itl"},
    {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
     "input_len": 1024, "output_len": 1024,
     "request_rate": 64.0, "slo_kind": "itl"},
    {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
     "input_len": 1024, "output_len": 1024,
     "request_rate": 128.0, "slo_kind": "itl"},
    {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
     "input_len": 1024, "output_len": 1024,
     "request_rate": 256.0, "slo_kind": "itl"},
    # Qwen prefill: input_len=4096
    {"model": "Qwen/Qwen2.5-14B-Instruct",
     "input_len": 4096, "output_len": 1,
     "request_rate": 16.0, "slo_kind": "ttft"},
    {"model": "Qwen/Qwen2.5-14B-Instruct",
     "input_len": 4096, "output_len": 1,
     "request_rate": 32.0, "slo_kind": "ttft"},
    {"model": "Qwen/Qwen2.5-14B-Instruct",
     "input_len": 4096, "output_len": 1,
     "request_rate": 64.0, "slo_kind": "ttft"},
    {"model": "Qwen/Qwen2.5-14B-Instruct",
     "input_len": 4096, "output_len": 1,
     "request_rate": 128.0, "slo_kind": "ttft"},
    {"model": "Qwen/Qwen2.5-14B-Instruct",
     "input_len": 4096, "output_len": 1,
     "request_rate": 256.0, "slo_kind": "ttft"},
    # Qwen decode: (1024, 1024)
    {"model": "Qwen/Qwen2.5-14B-Instruct",
     "input_len": 1024, "output_len": 1024,
     "request_rate": 16.0, "slo_kind": "itl"},
    {"model": "Qwen/Qwen2.5-14B-Instruct",
     "input_len": 1024, "output_len": 1024,
     "request_rate": 32.0, "slo_kind": "itl"},
    {"model": "Qwen/Qwen2.5-14B-Instruct",
     "input_len": 1024, "output_len": 1024,
     "request_rate": 64.0, "slo_kind": "itl"},
    {"model": "Qwen/Qwen2.5-14B-Instruct",
     "input_len": 1024, "output_len": 1024,
     "request_rate": 128.0, "slo_kind": "itl"},
    {"model": "Qwen/Qwen2.5-14B-Instruct",
     "input_len": 1024, "output_len": 1024,
     "request_rate": 256.0, "slo_kind": "itl"},
]


# ---------------------------------------------------------------------------
# SLO helpers
# ---------------------------------------------------------------------------

def load_slo_map(csv_path: Path) -> dict[tuple[str, str, str], float]:
    m: dict[tuple[str, str, str], float] = {}
    if not csv_path.is_file():
        return m
    with csv_path.open(newline="") as f:
        for r in csv.DictReader(f):
            m[(r["model"].strip(), r["phase"].strip(), r["shape"].strip())] = float(r["slo_ms"])
    return m


def shape_slo_ms(slo_map: dict, model_hf: str, phase: str, il: int, ol: int) -> float | None:
    mname = HF_TO_TAB.get(model_hf)
    if mname is None:
        return None
    shape = str(il) if (phase == "prefill" or ol <= 1) else f"{il}/{ol}"
    return slo_map.get((mname, phase, shape))


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

def killpg(p: subprocess.Popen | None) -> None:
    if p and p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def wait_health(port: int, timeout: int = 1800) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=4) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def model_id_from_server(port: int) -> str:
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=8) as r:
                return json.loads(r.read().decode())["data"][0]["id"]
        except Exception:
            time.sleep(1)
    raise RuntimeError("model id unavailable")


def parse_worker_log(path: Path) -> dict[str, str]:
    d: dict[str, str] = {}
    if not path.exists():
        return d
    for line in path.read_text().strip().splitlines():
        for tok in line.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                d[k] = v
    return d


# ---------------------------------------------------------------------------
# nvidia-smi dmon sidecar (reused from open_loop_rate_sweep.py)
# ---------------------------------------------------------------------------

def _parse_dmon_csv(path: Path) -> dict[str, float | None]:
    sm_vals: list[float] = []
    pwr_vals: list[float] = []
    fb_vals: list[float] = []
    header_cols: list[str] = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            # dmon emits TWO header lines: first=field names, second=units.
            # Use only the FIRST (#) line; subsequent ones are units ("Idx W C ...").
            if not header_cols:
                header_cols = line.lstrip("#").split()
            continue
        parts = line.split()
        if not parts or not header_cols:
            continue
        row = {header_cols[i]: parts[i] for i in range(min(len(header_cols), len(parts)))}
        def _f(k: str) -> float | None:
            v = row.get(k, "-")
            try:
                return float(v) if v != "-" else None
            except ValueError:
                return None
        sm = _f("sm"); pwr = _f("pwr"); fb = _f("fb")
        if sm is not None:
            sm_vals.append(sm)
        if pwr is not None:
            pwr_vals.append(pwr)
        if fb is not None:
            fb_vals.append(fb)

    def _p95(vals: list[float]) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        return s[max(0, int(math.ceil(0.95 * len(s))) - 1)]

    return {
        "tel_sm_mean": round(sum(sm_vals) / len(sm_vals), 2) if sm_vals else None,
        "tel_sm_p95":  round(_p95(sm_vals) or 0, 2) if sm_vals else None,
        "tel_pwr_p95_w": round(_p95(pwr_vals) or 0, 2) if pwr_vals else None,
        "tel_fb_peak_mib": round(max(fb_vals), 1) if fb_vals else None,
        "tel_n_samples": len(sm_vals),
    }


class DmonSidecar:
    def __init__(self, gpu_id: int, out_csv: Path) -> None:
        self.gpu_id = gpu_id
        self.out_csv = out_csv
        self._proc: subprocess.Popen | None = None
        self._fh = None

    def start(self) -> None:
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.out_csv, "w")
        self._proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-i", str(self.gpu_id),
             "-s", "pucvmt", "-d", "1"],
            stdout=self._fh, stderr=subprocess.DEVNULL,
        )

    def stop(self) -> dict[str, float | None]:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill(); self._proc.wait()
        if self._fh:
            self._fh.close()
        return _parse_dmon_csv(self.out_csv)


# ---------------------------------------------------------------------------
# Scenario builder — 12 scenarios, BW=0 fixed
# ---------------------------------------------------------------------------

def kv_args_for_model(model_id: str) -> list[str]:
    if "llama" in model_id.lower():
        return ["--kv-tokens", "16", "--n-kv-heads", "8", "--head-dim", "128"]
    if "qwen" in model_id.lower():
        return ["--kv-tokens", "16", "--n-kv-heads", "2", "--head-dim", "128"]
    return ["--chunk-mib", "128"]


def scenarios_for(
    model_name: str, worker_seconds: float,
    h2d_bws: tuple[int, ...] = (8, 16, 32, 56),
    d2h_bws: tuple[int, ...] = (8, 16, 32, 56),
    hbm_levels: tuple[int, ...] = (1, 2, 4),
) -> list[dict]:
    """Return scenario list. BW=0 → wcmd=None (no worker, true baseline).

    Default = 12 scenarios (1 baseline + 4 H2D + 4 D2H + 3 HBM).
    Customize by passing different bw/level tuples.
    """
    kv = kv_args_for_model(model_name)
    wsec = str(worker_seconds)
    out = []

    # Baseline: no injection worker at all
    out.append({"tag": "baseline_no_worker", "mode": "baseline", "wcmd": None})

    for bw in h2d_bws:
        out.append({
            "tag": f"h2d_{bw}_c1",
            "mode": "h2d_async",
            "wcmd": [PY, str(WORKER), "h2d-async", *kv,
                     "--seconds", wsec, "--target-gbps", str(bw), "--concurrent", "1"],
        })

    for bw in d2h_bws:
        out.append({
            "tag": f"d2h_{bw}_c1",
            "mode": "d2h_async",
            "wcmd": [PY, str(WORKER), "d2h-async", *kv,
                     "--seconds", wsec, "--target-gbps", str(bw), "--concurrent", "1"],
        })

    for n in hbm_levels:
        out.append({
            "tag": f"hbm{n}",
            "mode": "hbm_parallel",
            "wcmd": [PY, str(WORKER), "hbm-parallel",
                     "--seconds", wsec, "--parallel", str(n), "--tensor-mib", "64"],
        })

    return out


# ---------------------------------------------------------------------------
# Bench runner
# ---------------------------------------------------------------------------

def run_bench(
    w: dict, mid: str, port: int, outj: Path,
    label: str, num_prompts: int, num_warmups: int,
    env: dict, gpu_id: int,
    tel_csv: Path,
) -> tuple[subprocess.CompletedProcess, dict]:
    cmd = [
        PY, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "openai", "--host", "127.0.0.1", "--port", str(port),
        "--dataset-name", "random", "--model", mid,
        "--input-len", str(w["input_len"]),
        "--output-len", str(w["output_len"]),
        "--request-rate", str(w["request_rate"]),
        "--num-prompts", str(num_prompts),
        "--num-warmups", str(num_warmups),
        "--ignore-eos", "--disable-tqdm",
        "--percentile-metrics", "ttft,tpot,itl",
        "--metric-percentiles", "50,95,99",
        "--save-result",
        "--result-filename", outj.name,
        "--result-dir", str(outj.parent),
        "--label", label,
    ]
    bench_env = {**env, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    sidecar = DmonSidecar(gpu_id=gpu_id, out_csv=tel_csv)
    sidecar.start()
    rr = subprocess.run(cmd, capture_output=True, text=True,
                        timeout=7200, env=bench_env, cwd=str(ROOT))
    tel = sidecar.stop()
    return rr, tel


# ---------------------------------------------------------------------------
# SLO-limited tok/s
# ---------------------------------------------------------------------------

def slo_limited_tok_s(row: dict, slo_ms: float | None) -> float | None:
    if slo_ms is None:
        return None
    kind = row.get("slo_kind", "itl")
    lat = row.get("p99_itl_ms") if kind == "itl" else row.get("p99_ttft_ms")
    raw = row.get("output_throughput")
    if lat is None or raw is None or float(lat) <= 0:
        return None
    return float(raw) * min(1.0, slo_ms / float(lat))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT,
                    help="Output directory (default: injection_sweep_v5/ next to this file).")
    ap.add_argument("--workloads-json", type=Path, default=None,
                    help="JSON file with list of workload dicts "
                         "(model, input_len, output_len, request_rate, slo_kind). "
                         "If omitted, uses built-in default 20-workload grid.")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--worker-seconds", type=float, default=120.0,
                    help="Duration of each injection_stress_worker run.")
    ap.add_argument("--steady-seconds", type=float, default=60.0,
                    help="Minimum steady-state window for bench; "
                         "num_prompts = max(--min-prompts, ceil(steady_seconds x rate)).")
    ap.add_argument("--min-prompts", type=int, default=200)
    ap.add_argument("--max-prompts-cap", type=int, default=2000,
                    help="Hard cap on num_prompts per cell. Prevents multi-hour queue "
                         "drains at high rates above server saturation. Default 2000.")
    ap.add_argument("--base-warmups", type=int, default=5)
    ap.add_argument("--max-warmups", type=int, default=60)
    ap.add_argument("--warmup-scale-rate", type=float, default=32.0)
    ap.add_argument("--h2d-bws", type=int, nargs="+", default=[8, 16, 32, 56],
                    help="H2D bandwidths (Gbps) for injection scenarios.")
    ap.add_argument("--d2h-bws", type=int, nargs="+", default=[8, 16, 32, 56],
                    help="D2H bandwidths (Gbps) for injection scenarios.")
    ap.add_argument("--hbm-levels", type=int, nargs="+", default=[1, 2, 4],
                    help="HBM parallel-tensor levels for injection scenarios.")
    ap.add_argument("--startup-timeout", type=int, default=1800)
    ap.add_argument("--smoke", action="store_true",
                    help="Quick test: 2 workloads, 2 scenarios, short durations.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("VLLM_REPO_ROOT", str(ROOT))
    env.setdefault("PYTHONPATH", str(ROOT))
    env.setdefault("HUGGINGFACE_HUB_CACHE", "/data/ben/huggingface")
    env.setdefault("HF_HOME", "/data/ben/hf_home")
    env.setdefault("VLLM_CACHE_ROOT", "/data/ben/vllm_cache")
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data/ben/torch_cache/inductor")
    env.setdefault("TORCH_COMPILE_CACHE_DIR", "/data/ben/torch_cache/compile")
    env.setdefault("XDG_CACHE_HOME", "/data/ben/xdg_cache")
    ht = Path.home() / "vllm/hf_token"
    if "HF_TOKEN" not in env and ht.exists():
        env["HF_TOKEN"] = ht.read_text().strip()

    slo_map = load_slo_map(SLO_CSV)

    # Load workloads
    if args.workloads_json and args.workloads_json.is_file():
        workloads = json.loads(args.workloads_json.read_text())
    else:
        workloads = _DEFAULT_WORKLOADS

    if args.smoke:
        # Two representative workloads, shorten timings
        workloads = [workloads[0], workloads[5]]
        args.worker_seconds = 15.0
        args.steady_seconds = 15.0
        args.min_prompts = 20
        args.base_warmups = 2
        args.max_warmups = 5

    # Attach SLO ms to each workload
    for w in workloads:
        ph = "prefill" if w["output_len"] <= 1 else "decode"
        slo = shape_slo_ms(slo_map, w["model"], ph, w["input_len"], w["output_len"])
        w["slo_ms"] = slo
        w["phase"] = ph
        if "name" not in w:
            safe_m = w["model"].replace("/", "_").replace("-", "_")
            w["name"] = (f"{safe_m}_{ph}_{w['input_len']}_{w['output_len']}"
                         f"_r{w['request_rate']:g}")

    results_path = args.out / "injection_sweep_v5.jsonl"
    results_path.write_text("")  # fresh run
    all_rows: list[dict] = []

    # Group workloads by model so we only start each server once
    from collections import defaultdict
    by_model: dict[str, list[dict]] = defaultdict(list)
    for w in workloads:
        by_model[w["model"]].append(w)

    for model, mworkloads in by_model.items():
        mslug = model.replace("/", "__")
        port = 8890
        mdir = args.out / mslug
        mdir.mkdir(parents=True, exist_ok=True)
        tel_dir = mdir / "telemetry"
        tel_dir.mkdir(parents=True, exist_ok=True)

        # Determine max_model_len from all workloads for this model
        max_mml = 8192
        for w in mworkloads:
            need = w["input_len"] + w["output_len"] + 256
            rounded = ((need + 1023) // 1024) * 1024
            max_mml = max(max_mml, min(rounded, 32768))

        slog_path = mdir / "api_server.log"
        slog = open(slog_path, "w")
        srv_cmd = [
            PY, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(port),
            "--max-model-len", str(max_mml),
            "--gpu-memory-utilization", "0.90",
            "--max-num-seqs", "512",
            "--disable-log-requests",
            "--dtype", "auto",
            "--no-enable-prefix-caching",
        ]
        if "Qwen" in model:
            srv_cmd.append("--trust-remote-code")
        srv_env = {**env, "CUDA_VISIBLE_DEVICES": str(args.gpu_id)}
        srv = subprocess.Popen(
            srv_cmd, stdout=slog, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid, env=srv_env, cwd=str(ROOT),
        )
        print(f"\n{'='*60}")
        print(f"Server started: {model}  port={port}  max_model_len={max_mml}")

        worker_proc: subprocess.Popen | None = None
        try:
            if not wait_health(port, args.startup_timeout):
                raise RuntimeError(f"server health timeout for {model}")
            mid = model_id_from_server(port)
            print(f"Server ready: model_id={mid}")

            for w in mworkloads:
                wdir = args.out / mslug / w["name"]
                wdir.mkdir(parents=True, exist_ok=True)
                scens = scenarios_for(
                    model, args.worker_seconds,
                    h2d_bws=tuple(args.h2d_bws),
                    d2h_bws=tuple(args.d2h_bws),
                    hbm_levels=tuple(args.hbm_levels),
                )
                if args.smoke:
                    scens = scens[:2]

                # Warm up server before first scenario of this workload
                num_p0 = max(args.min_prompts // 5, 20)
                warmup_cmd = [
                    PY, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
                    "--backend", "openai", "--host", "127.0.0.1", "--port", str(port),
                    "--dataset-name", "random", "--model", mid,
                    "--input-len", str(w["input_len"]),
                    "--output-len", str(w["output_len"]),
                    "--request-rate", str(w["request_rate"]),
                    "--num-prompts", str(num_p0),
                    "--num-warmups", "3",
                    "--ignore-eos", "--disable-tqdm",
                ]
                subprocess.run(warmup_cmd, capture_output=True, text=True,
                               timeout=3600, env={**env, "CUDA_VISIBLE_DEVICES": str(args.gpu_id)},
                               cwd=str(ROOT))

                for sc in scens:
                    worker_proc = None
                    wlog_path: Path | None = None
                    if sc["wcmd"] is not None:
                        wlog_path = wdir / f'worker_{sc["tag"]}.log'
                        with open(wlog_path, "w", buffering=1) as wf:
                            worker_proc = subprocess.Popen(
                                sc["wcmd"], stdout=wf, stderr=subprocess.STDOUT,
                                preexec_fn=os.setsid, env=env, cwd=str(ROOT),
                            )
                        time.sleep(2.0)  # let worker reach steady throughput

                    num_p = max(args.min_prompts, math.ceil(args.steady_seconds * w["request_rate"]))
                    if args.max_prompts_cap is not None:
                        num_p = min(num_p, args.max_prompts_cap)
                    num_wm = args.max_warmups if w["request_rate"] >= args.warmup_scale_rate else args.base_warmups
                    label = f'{w["name"]}_v5_{sc["tag"]}'
                    outj = wdir / f"{label}.json"
                    tel_csv = tel_dir / f"{label}_dmon.csv"

                    print(f"  [{w['name']}] scenario={sc['tag']}  "
                          f"rate={w['request_rate']}  "
                          f"num_prompts={num_p}  warmups={num_wm}")

                    rr, tel = run_bench(
                        w, mid, port, outj, label, num_p, num_wm,
                        env, args.gpu_id, tel_csv,
                    )

                    row: dict = {
                        "workload": w["name"],
                        "model": model,
                        "phase": w["phase"],
                        "input_len": w["input_len"],
                        "output_len": w["output_len"],
                        "request_rate": w["request_rate"],
                        "injection_scenario": sc["tag"],
                        "injection_mode": sc["mode"],
                        "exit_code": rr.returncode,
                        "result_json": str(outj),
                        "slo_kind": w["slo_kind"],
                        "slo_ms": w.get("slo_ms"),
                        "bw0_is_baseline": (sc["tag"] == "baseline_no_worker"),
                    }
                    # Worker telemetry
                    if wlog_path and wlog_path.exists():
                        plog = parse_worker_log(wlog_path)
                        for k in ("achieved_gbps", "h2d_achieved_gbps",
                                  "d2h_achieved_gbps", "concurrent", "numel"):
                            if plog.get(k):
                                row[f"worker_{k}"] = plog[k]
                    # Bench metrics + GPU telemetry
                    if outj.exists():
                        try:
                            j = json.loads(outj.read_text())
                            for k in ("output_throughput", "p99_itl_ms", "p99_ttft_ms",
                                      "p99_tpot_ms", "mean_itl_ms", "mean_ttft_ms",
                                      "completed", "failed"):
                                if k in j:
                                    row[k] = j[k]
                        except (json.JSONDecodeError, OSError):
                            pass
                    row.update({k: v for k, v in tel.items() if v is not None})

                    # SLO-limited tok/s
                    row["slo_limited_tok_s"] = slo_limited_tok_s(row, w.get("slo_ms"))

                    with results_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    all_rows.append(row)

                    print(f"    done  rc={rr.returncode}  "
                          f"tok/s={row.get('output_throughput')}  "
                          f"slo_limited={row.get('slo_limited_tok_s')}  "
                          f"sm_mean={row.get('tel_sm_mean')}%")

                    if worker_proc:
                        killpg(worker_proc)
                        worker_proc = None

        finally:
            if worker_proc:
                killpg(worker_proc)
            killpg(srv)
            slog.close()

    # Write CSV. Cells with active workers carry extra worker_* telemetry
    # not present on baseline rows; build the fieldnames as the union over
    # all rows so DictWriter does not raise ValueError on unseen keys.
    if all_rows:
        csv_path = args.out / "injection_sweep_v5.csv"
        seen: dict[str, None] = {}
        for row in all_rows:
            for key in row.keys():
                if key not in seen:
                    seen[key] = None
        fieldnames = list(seen.keys())
        with csv_path.open("w", newline="") as f:
            w_csv = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w_csv.writeheader()
            w_csv.writerows(all_rows)
        print(f"\nWrote {csv_path}")

    print(f"WROTE {results_path}")


if __name__ == "__main__":
    main()
