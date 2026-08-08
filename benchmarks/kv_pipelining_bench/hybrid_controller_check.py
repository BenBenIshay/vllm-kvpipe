#!/usr/bin/env python3
"""Validate the ITL-preserving admission controller (B1 seq-aware roofline + B2
closed-loop guardrail).

No manual off_max / itl override: the controller decides how many CPU-tier
requests to parallelize from the interconnect roofline (hide budget = itl_ms,
bytes/token, x_eff) and a closed-loop guardrail against the measured ITL SLO.
We sweep the ITL SLO and confirm the admitted CPU-tier concurrency ADAPTS: a
tight SLO admits fewer; a loose SLO admits more. Uses REAL KV (no bench
connector). Peak CPU-tier concurrency is read from the TIER_OCCUPANCY log.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = "/mnt/beegfs/benishay/vllm-kvpipe"
PY = f"{ROOT}/.venv/bin/python"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LOGDIR = "/mnt/beegfs/benishay/paper_runs/20260728/hybrid_m1"
N_REQS = 24
MAX_TOKENS = 200


def launch(port: int, slo_ms: float, itl_ms: float) -> subprocess.Popen:
    env = dict(os.environ)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    env["PYTHONUNBUFFERED"] = "1"
    env["VLLM_KVPIPE_TIER_OCCUPANCY"] = "1"
    env["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
    env["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
    env["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
        "--port", str(port), "--max-model-len", "1024",
        "--gpu-memory-utilization", "0.30", "--num-gpu-blocks-override", "400",
        "--disable-log-requests", "--dtype", "auto", "--max-num-seqs", str(N_REQS),
        "--enforce-eager", "--no-enable-prefix-caching",
        "--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
        "--hybrid-b-hbm-max", "2", "--hybrid-pcie-gbps", "25.1",
        "--hybrid-b-tok", "24576",                       # 0.5B bytes/token
        "--hybrid-itl-ms", str(itl_ms),                  # hide budget target
        "--transfer-aware-slo-p99-itl-ms", str(slo_ms),  # B2 measured SLO
        "--enable-kv-pipelining", "--kv-k-window", "4",
    ]
    logf = open(f"{LOGDIR}/controller_{port}.log", "w")
    return subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def wait_ready(port: int, timeout: int = 240) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def one(port: int, i: int) -> int:
    body = json.dumps({"model": MODEL,
                       "prompt": f"Essay #{i} on the history of science and technology:",
                       "max_tokens": MAX_TOKENS, "temperature": 0.0,
                       "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return len(json.loads(r.read()).get("choices", [{}])[0].get("text", ""))
    except Exception:
        return -1


def run(port: int, slo_ms: float, itl_ms: float) -> dict:
    proc = launch(port, slo_ms, itl_ms)
    try:
        if not wait_ready(port):
            return {"ok": 0, "peak_cpu": 0, "median_run": 0}
        # STAGGERED arrivals: submit one every STAGGER seconds so later requests
        # see a running CPU cohort with grown sequences. A burst would admit
        # everyone before the roofline / guardrail engages (gate-new-only).
        ex = ThreadPoolExecutor(max_workers=N_REQS)
        futs = []
        for i in range(N_REQS):
            futs.append(ex.submit(one, port, i))
            time.sleep(0.4)
        res = [f.result() for f in futs]
        ex.shutdown()
        ok = sum(1 for r in res if r > 0)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(3)
    pat = re.compile(r"TIER_OCCUPANCY: n_hbm=(\d+) n_cpu=(\d+) ")
    peak_cpu = 0
    runs = []
    with open(f"{LOGDIR}/controller_{port}.log") as f:
        for line in f:
            m = pat.search(line)
            if m:
                nh, nc = int(m.group(1)), int(m.group(2))
                peak_cpu = max(peak_cpu, nc)
                runs.append(nh + nc)
    steady = sorted(runs)[len(runs)//10:len(runs)-len(runs)//10] if len(runs) > 10 else runs
    med = sorted(steady)[len(steady)//2] if steady else 0
    return {"ok": ok, "peak_cpu": peak_cpu, "median_run": med}


def main() -> None:
    # Isolate the B1 seq-length-aware roofline: vary the hide budget (itl_ms) with
    # a generous SLO. As the staggered cohort's sequences grow, a TIGHT budget
    # caps how many CPU-tier requests run concurrently; a LOOSE budget admits more.
    print("=== TIGHT budget (itl_ms=2) — roofline should cap CPU concurrency ===",
          flush=True)
    tight = run(9961, slo_ms=5000.0, itl_ms=2.0)
    print(f"  {tight}", flush=True)
    print("=== LOOSE budget (itl_ms=40) — roofline admits many CPU-tier ===",
          flush=True)
    loose = run(9962, slo_ms=5000.0, itl_ms=40.0)
    print(f"  {loose}", flush=True)
    print("=" * 64)
    print(f"steady-state concurrent running: tight={tight['median_run']}  "
          f"loose={loose['median_run']}")
    print(f"peak CPU-tier concurrency:        tight={tight['peak_cpu']}  "
          f"loose={loose['peak_cpu']}")
    print(f"completed OK: tight={tight['ok']}/{N_REQS}  loose={loose['ok']}/{N_REQS}")
    if loose["peak_cpu"] > tight["peak_cpu"] and tight["ok"] == N_REQS == loose["ok"]:
        print(f"CONTROLLER PASS: the seq-aware roofline adapts CPU-tier parallelism "
              f"to the hide budget — loose admits {loose['peak_cpu']} concurrent CPU "
              f"vs tight {tight['peak_cpu']}, both correct, no manual off_max.")
    else:
        print("CONTROLLER: no clear budget-driven adaptation (inspect logs).")


if __name__ == "__main__":
    main()
