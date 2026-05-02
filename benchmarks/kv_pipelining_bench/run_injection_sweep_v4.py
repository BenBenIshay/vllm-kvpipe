#!/usr/bin/env python3
"""Run v4 interference matrix for layer-KV proxy transport."""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/ben/vllm")
THIS = Path(__file__).resolve().parent
PY = str(ROOT / ".venv/bin/python")
WORKER = THIS / "injection_stress_worker.py"
OUT = Path(os.environ.get("KV_SWEEP_OUT", str(THIS / "injection_sweep_v4")))
OUT.mkdir(parents=True, exist_ok=True)
SLO_CSV = Path(
    os.environ.get(
        "KV_SLO_CSV",
        "/data/ben/kv_phase_a_internal/open_loop_eval_v1/length_dependent_slo_definitions_v2.csv",
    )
)
HF_TO_TAB = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B-Instruct",
}


def load_slo_map():
    m = {}
    with SLO_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            m[(r["model"].strip(), r["phase"].strip(), r["shape"].strip())] = float(
                r["slo_ms"]
            )
    return m


_SLO = load_slo_map()


def shape_slo_ms(model_hf: str, phase: str, il: int, ol: int) -> float:
    mdl = HF_TO_TAB[model_hf]
    shape = str(il) if (phase == "prefill" or ol <= 1) else f"{il}/{ol}"
    return _SLO[(mdl, phase, shape)]


def kv_args_for_model(model_id: str) -> list[str]:
    if "llama" in model_id.lower():
        return ["--kv-tokens", "16", "--n-kv-heads", "8", "--head-dim", "128"]
    if "qwen" in model_id.lower():
        return ["--kv-tokens", "16", "--n-kv-heads", "2", "--head-dim", "128"]
    return ["--chunk-mib", "128"]


def killpg(p):
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


def run_bench(w: dict, mid: str, port: int, outj: Path, label: str, num_prompts: int, num_warmups: int, env: dict) -> subprocess.CompletedProcess:
    cmd = [
        PY, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "openai", "--host", "127.0.0.1", "--port", str(port),
        "--dataset-name", "random", "--model", mid,
        "--input-len", str(w["input_len"]), "--output-len", str(w["output_len"]),
        "--request-rate", str(w["request_rate"]),
        "--num-prompts", str(num_prompts), "--num-warmups", str(num_warmups),
        "--ignore-eos", "--disable-tqdm", "--percentile-metrics", "ttft,tpot,itl",
        "--metric-percentiles", "50,95,99", "--save-result",
        "--result-filename", outj.name, "--result-dir", str(outj.parent),
        "--label", label,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workloads", choices=["all", "llama", "qwen"], default="all")
    args = ap.parse_args()

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

    workloads = [
        {"name": "llama_decode_1024_512_r8", "model": "meta-llama/Meta-Llama-3.1-8B-Instruct", "input_len": 1024, "output_len": 512, "request_rate": 8.0, "slo_kind": "itl"},
        {"name": "qwen_prefill_1024_1_r160", "model": "Qwen/Qwen2.5-14B-Instruct", "input_len": 1024, "output_len": 1, "request_rate": 160.0, "slo_kind": "ttft"},
    ]
    for w in workloads:
        ph = "decode" if w["output_len"] > 1 else "prefill"
        w["slo_ms"] = shape_slo_ms(w["model"], ph, int(w["input_len"]), int(w["output_len"]))

    if args.workloads == "llama":
        wlist = [workloads[0]]
    elif args.workloads == "qwen":
        wlist = [workloads[1]]
    else:
        wlist = workloads if not args.smoke else [workloads[0]]

    wsec = 12.0 if args.smoke else 420.0
    num_p = 8 if args.smoke else 120
    warm = 2 if args.smoke else 5
    warm0 = 20 if args.smoke else 60

    def scenarios_for(model_name: str):
        kv = kv_args_for_model(model_name)
        if args.smoke:
            return [
                {"tag": "h2d_16_c1", "mode": "h2d_async", "wcmd": [PY, str(WORKER), "h2d-async", *kv, "--seconds", str(wsec), "--target-gbps", "16", "--concurrent", "1"]},
                {"tag": "d2h_8_c1", "mode": "d2h_async", "wcmd": [PY, str(WORKER), "d2h-async", *kv, "--seconds", str(wsec), "--target-gbps", "8", "--concurrent", "1"]},
                {"tag": "mix_4_4", "mode": "h2d_d2h_mix", "wcmd": [PY, str(WORKER), "h2d-d2h-mix", *kv, "--seconds", str(wsec), "--h2d-gbps", "4", "--d2h-gbps", "4"]},
                {"tag": "hbm2", "mode": "hbm_parallel", "wcmd": [PY, str(WORKER), "hbm-parallel", "--seconds", str(wsec), "--parallel", "2", "--tensor-mib", "64"]},
            ]
        out = []
        for bw in (0, 8, 16, 24, 32, 40, 48, 56):
            out.append({"tag": f"h2d_{bw}_c1", "mode": "h2d_async", "wcmd": [PY, str(WORKER), "h2d-async", *kv, "--seconds", str(wsec), "--target-gbps", str(bw), "--concurrent", "1"]})
        out.extend([
            {"tag": "h2d_32_c2", "mode": "h2d_async", "wcmd": [PY, str(WORKER), "h2d-async", *kv, "--seconds", str(wsec), "--target-gbps", "32", "--concurrent", "2"]},
            {"tag": "h2d_32_c4", "mode": "h2d_async", "wcmd": [PY, str(WORKER), "h2d-async", *kv, "--seconds", str(wsec), "--target-gbps", "32", "--concurrent", "4"]},
        ])
        for bw in (0, 8, 16, 32):
            out.append({"tag": f"d2h_{bw}_c1", "mode": "d2h_async", "wcmd": [PY, str(WORKER), "d2h-async", *kv, "--seconds", str(wsec), "--target-gbps", str(bw), "--concurrent", "1"]})
        for a, b in ((4, 4), (8, 8), (16, 8)):
            out.append({"tag": f"mix_{a}_{b}", "mode": "h2d_d2h_mix", "wcmd": [PY, str(WORKER), "h2d-d2h-mix", *kv, "--seconds", str(wsec), "--h2d-gbps", str(a), "--d2h-gbps", str(b)]})
        for n in (0, 1, 2, 4):
            if n == 0:
                out.append({"tag": "hbm0", "mode": "hbm_parallel", "wcmd": None})
            else:
                out.append({"tag": f"hbm{n}", "mode": "hbm_parallel", "wcmd": [PY, str(WORKER), "hbm-parallel", "--seconds", str(wsec), "--parallel", str(n), "--tensor-mib", "64"]})
        return out

    rows: list[dict] = []
    results = OUT / "injection_sweep_results.jsonl"
    results.write_text("")
    for wi, w in enumerate(wlist, start=1):
        port = 8885 + wi
        wdir = OUT / w["name"]
        wdir.mkdir(parents=True, exist_ok=True)
        slog = open(wdir / "api_server.log", "w")
        srv_cmd = [PY, "-m", "vllm.entrypoints.openai.api_server", "--model", w["model"], "--port", str(port), "--max-model-len", "8192", "--gpu-memory-utilization", "0.90", "--max-num-seqs", "512", "--disable-log-requests", "--dtype", "auto"]
        if "Qwen" in w["model"]:
            srv_cmd.append("--trust-remote-code")
        srv = subprocess.Popen(srv_cmd, stdout=slog, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env, cwd=str(ROOT))
        worker = None
        try:
            if not wait_health(port):
                raise RuntimeError("server health timeout")
            mid = model_id_from_server(port)
            subprocess.run([PY, "-m", "vllm.entrypoints.cli.main", "bench", "serve", "--backend", "openai", "--host", "127.0.0.1", "--port", str(port), "--dataset-name", "random", "--model", mid, "--input-len", str(w["input_len"]), "--output-len", str(w["output_len"]), "--request-rate", str(w["request_rate"]), "--num-prompts", str(warm0), "--num-warmups", "3", "--ignore-eos", "--disable-tqdm"], capture_output=True, text=True, timeout=3600, env=env, cwd=str(ROOT))
            for sc in scenarios_for(w["model"]):
                worker = None
                wlog = None
                if sc["wcmd"] is not None:
                    wlog = wdir / f'worker_{sc["tag"]}.log'
                    with open(wlog, "w", buffering=1) as wf:
                        worker = subprocess.Popen(sc["wcmd"], stdout=wf, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env, cwd=str(ROOT))
                    time.sleep(2.0)
                label = f'{w["name"]}_v4_{sc["tag"]}'
                outj = wdir / f"{label}.json"
                rr = run_bench(w, mid, port, outj, label, num_p, warm, env)
                row = {
                    "workload": w["name"], "model": w["model"], "phase": "decode" if w["output_len"] > 1 else "prefill",
                    "input_len": w["input_len"], "output_len": w["output_len"], "injection_scenario": sc["tag"], "injection_mode": sc["mode"],
                    "exit_code": rr.returncode, "result_json": str(outj), "slo_kind": w["slo_kind"], "slo_ms": w["slo_ms"], "smoke": bool(args.smoke),
                }
                if wlog and wlog.exists():
                    plog = parse_worker_log(wlog)
                    for k in ("achieved_gbps", "h2d_achieved_gbps", "d2h_achieved_gbps", "concurrent", "numel"):
                        if plog.get(k):
                            row[f"worker_{k}"] = plog[k]
                if outj.exists():
                    j = json.loads(outj.read_text())
                    row["output_throughput"] = j.get("output_throughput")
                    row["p99_itl_ms"] = j.get("p99_itl_ms")
                    row["p99_ttft_ms"] = j.get("p99_ttft_ms")
                with results.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                rows.append(row)
                print("done", w["name"], sc["tag"], "rc", rr.returncode)
                if worker:
                    killpg(worker); worker = None
        finally:
            if worker:
                killpg(worker)
            killpg(srv)
            slog.close()

    if rows:
        with (OUT / "injection_sweep_results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("WROTE", results)


if __name__ == "__main__":
    main()
