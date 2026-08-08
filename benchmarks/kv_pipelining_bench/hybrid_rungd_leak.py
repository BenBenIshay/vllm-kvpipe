#!/usr/bin/env python3
"""Regression test for the rung-d host-pool lifecycle (free_req on finish).

Forces a SMALL host pool (VLLM_KVPIPE_RUNGD_HMAX) and sends many requests
SEQUENTIALLY (one finishes before the next). Without free_req, finished
requests' host blocks leak and the pool exhausts after ~h_max/blocks_per_req
requests -> later requests degrade (garbage) + a 'pool could not manage'
warning. With free_req wired, blocks recycle and every request is served
identically. Uses REAL KV (no bench connector), all CPU-tier (b_hbm_max=0).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request

ROOT = "/mnt/beegfs/benishay/vllm-kvpipe"
PY = f"{ROOT}/.venv/bin/python"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LOGDIR = "/mnt/beegfs/benishay/paper_runs/20260728/hybrid_m1"
PORT = 9971
LOG = f"{LOGDIR}/rungd_leak_{PORT}.log"
N = 30           # sequential requests (>> h_max / blocks_per_req)
H_MAX = 40       # tiny pool: leaks would exhaust after ~13 requests
PROMPT = "Explain the theory of general relativity in a few sentences:"


def main() -> None:
    env = dict(os.environ)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    env["PYTHONUNBUFFERED"] = "1"
    env["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
    env["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
    env["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
    env["VLLM_KVPIPE_RUNGD_HMAX"] = str(H_MAX)
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
        "--port", str(PORT), "--max-model-len", "512",
        "--gpu-memory-utilization", "0.30", "--disable-log-requests",
        "--dtype", "auto", "--max-num-seqs", "8", "--enforce-eager",
        "--no-enable-prefix-caching",
        "--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
        "--hybrid-b-hbm-max", "0", "--hybrid-pcie-gbps", "25.1",
        "--hybrid-b-tok", "24576", "--hybrid-itl-ms", "100000",
        "--enable-kv-pipelining", "--kv-k-window", "4",
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=open(LOG, "w"),
                            stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    outs = []
    try:
        t0 = time.time()
        while time.time() - t0 < 240:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(2)
        for i in range(N):
            body = json.dumps({"model": MODEL, "prompt": PROMPT, "max_tokens": 40,
                               "temperature": 0.0, "stream": False}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    outs.append(json.loads(r.read())["choices"][0]["text"])
            except Exception as e:
                outs.append(f"<err:{e}>")
        time.sleep(1)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(3)

    warns = 0
    with open(LOG) as f:
        for line in f:
            if "pool could not manage" in line:
                warns += 1
    ref = outs[0] if outs else ""
    identical = sum(1 for o in outs if o == ref)
    print("=" * 64)
    print(f"requests: {len(outs)}   identical-to-first: {identical}/{len(outs)}")
    print(f"'pool could not manage' warnings: {warns}")
    print(f"first output: {ref[:60]!r}")
    print(f"last  output: {outs[-1][:60]!r}" if outs else "  (none)")
    print("=" * 64)
    if identical == N and warns == 0 and ref and not ref.startswith("<err"):
        print(f"LEAK TEST PASS: all {N} sequential requests served identically "
              f"under a {H_MAX}-block pool — host blocks recycle on finish "
              "(free_req wired); no exhaustion.")
    else:
        print("LEAK TEST FAIL: pool exhaustion / divergence — free_req not "
              "recycling blocks (inspect log).")


if __name__ == "__main__":
    main()
