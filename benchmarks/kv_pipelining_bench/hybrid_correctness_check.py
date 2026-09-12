#!/usr/bin/env python3
"""Rung-(c) correctness harness for hybrid CPU-tier streaming.

The decode bench fakes KV with a CONSTANT fill (fill_std=0 -> all KV = 0.015),
so greedy decode is deterministic. Therefore a request's output must be
IDENTICAL whether it is served HBM-resident (b_hbm_max>=1) or CPU-tier streamed
(b_hbm_max=0). If the streaming moves/reads the wrong bytes, the outputs diverge.

Launches the hybrid server twice (resident ref vs streamed), sends the same
greedy prompt, and diffs the output. Usage: python correctness_check.py
"""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from bench_teardown import terminate_engine  # noqa: E402
import json, os, subprocess, sys, time, urllib.request, urllib.error, signal

ROOT = "/mnt/beegfs/benishay/vllm-kvpipe"
PY = f"{ROOT}/.venv/bin/python"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The quick brown fox jumps over the lazy dog and then"
MAX_TOKENS = 24


def launch(port: int, mode: str, b_hbm_max: int = 8, stream: bool = False) -> subprocess.Popen:
    env = dict(os.environ)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    env["VLLM_KVPIPE_HYBRID_STREAM"] = "1" if stream else "0"
    env["PYTHONUNBUFFERED"] = "1"
    kv_cfg = {
        "kv_connector": "PipelinedDecodeBenchConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"fill_mean": 0.015, "fill_std": 0.0},
    }
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL, "--port", str(port),
        "--max-model-len", "1024", "--gpu-memory-utilization", "0.15",
        "--disable-log-requests", "--dtype", "auto", "--max-num-seqs", "8",
        "--enforce-eager",
    ]
    if mode == "vanilla":
        pass  # plain vLLM, no connector
    elif mode == "pipelined":  # R1 (all-CPU rolling, known-correct)
        cmd += ["--enable-kv-pipelining", "--decode-only-pipelining",
                "--kv-k-window", "4", "--inflate-kv-pool",
                "--kv-transfer-config", json.dumps(kv_cfg)]
    elif mode == "hybrid":
        cmd += ["--enable-kv-pipelining", "--kv-k-window", "4",
                "--kv-transfer-config", json.dumps(kv_cfg),
                "--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
                "--hybrid-b-hbm-max", str(b_hbm_max),
                "--hybrid-pcie-gbps", "25.1", "--hybrid-b-tok", "131072"]
    logf = open(f"/mnt/beegfs/benishay/paper_runs/20260728/hybrid_m1/cc_server_{port}.log", "w")
    return subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def wait_ready(port: int, timeout: int = 180) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def complete(port: int) -> str:
    body = json.dumps({
        "model": MODEL, "prompt": PROMPT, "max_tokens": MAX_TOKENS,
        "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return d["choices"][0]["text"]


def run_cfg(port: int, mode: str, b_hbm_max: int = 8, stream: bool = False) -> str:
    proc = launch(port, mode, b_hbm_max, stream)
    try:
        if not wait_ready(port):
            return "<server-did-not-start>"
        return complete(port)
    finally:
        # SIGTERM alone drained nova06/07 on 2026-09-06: a wedged
        # engine ignores it and the survivor is what SLURM cannot
        # kill. terminate_engine verifies the GROUP is empty and
        # escalates. See bench_teardown.py.
        terminate_engine(proc, label=__name__)


def main():
    # Clean isolation: SAME cpu-tier (b_hbm_max=0), toggle ONLY streaming.
    # Zero confound -> any difference is the streaming path alone.
    cfgs = [
        ("cpu-tier RESIDENT (b_hbm_max=0, stream OFF)", 9891, "hybrid", 0, False),
        ("cpu-tier STREAMED (b_hbm_max=0, stream ON)", 9892, "hybrid", 0, True),
    ]
    out = {}
    for label, port, mode, bhm, stream in cfgs:
        print(f"=== {label} ===", flush=True)
        txt = run_cfg(port, mode, bhm, stream)
        out[label] = txt
        print(repr(txt), flush=True)
    print("=" * 60)
    resident = out[cfgs[0][0]]
    streamed = out[cfgs[1][0]]
    if resident.startswith("<server") or streamed.startswith("<server"):
        print("INCONCLUSIVE: a server did not start.")
    elif resident == streamed:
        print("RUNG-C PASS: streaming preserves KV (streamed == resident, "
              "same cpu-tier, only streaming toggled).")
    else:
        print("RUNG-C FAIL: streaming corrupts KV (streamed != resident with "
              "ONLY streaming toggled -> the bug is in the streaming path).")


if __name__ == "__main__":
    main()
