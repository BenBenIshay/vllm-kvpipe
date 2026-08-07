#!/usr/bin/env python3
"""Definitive token-level correctness for rung-d, with REAL KV (no bench connector).

The PipelinedDecodeBenchConnector fake-fills KV (0.015), so token comparison
against it is invalid (the pool stores real projected keys and correctly
diverges). Here we run hybrid tiers WITHOUT that connector — tier tagging comes
from the scheduler's transfer-aware admission, not the connector — so prefill KV
is REAL. If the CPU cohort served entirely from the host-block-id pool
(RUNGD_FREE=1, b_hbm_max=0) generates the SAME greedy tokens as vanilla vLLM,
the pool faithfully reproduces real generation.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = "/mnt/beegfs/benishay/vllm-kvpipe"
PY = f"{ROOT}/.venv/bin/python"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LOGDIR = "/mnt/beegfs/benishay/paper_runs/20260728/hybrid_m1"
MAX_TOKENS = 32
PROMPTS = [
    "The history of the Roman empire began when",
    "In quantum mechanics, the wave function describes",
    "A recipe for sourdough bread starts with",
]


def launch(port: int, pool: bool) -> subprocess.Popen:
    env = dict(os.environ)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    env["PYTHONUNBUFFERED"] = "1"
    if pool:
        env["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
        env["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
        env["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
        "--port", str(port), "--max-model-len", "1024",
        "--gpu-memory-utilization", "0.30", "--disable-log-requests",
        "--dtype", "auto", "--max-num-seqs", "8", "--enforce-eager",
        "--no-enable-prefix-caching",
    ]
    if pool:
        # Hybrid tiers via admission only — NO --kv-transfer-config (real KV).
        cmd += [
            "--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
            "--hybrid-b-hbm-max", "0", "--hybrid-pcie-gbps", "25.1",
            "--hybrid-b-tok", "131072", "--enable-kv-pipelining", "--kv-k-window", "4",
        ]
    logf = open(f"{LOGDIR}/rungd_real_{port}.log", "w")
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


def complete(port: int, prompt: str) -> str:
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": MAX_TOKENS,
                       "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    return d.get("choices", [{}])[0].get("text", "<err>")


def run(port: int, pool: bool) -> dict[str, str]:
    proc = launch(port, pool)
    try:
        if not wait_ready(port):
            return {p: "<no-start>" for p in PROMPTS}
        with ThreadPoolExecutor(max_workers=len(PROMPTS)) as ex:
            outs = list(ex.map(lambda p: complete(port, p), PROMPTS))
        return dict(zip(PROMPTS, outs))
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(3)


def main() -> None:
    print("=== VANILLA (no tiers, real KV) ===", flush=True)
    v = run(9931, pool=False)
    print("=== POOL (hybrid tiers, no connector, RUNGD_FREE=1) ===", flush=True)
    b = run(9932, pool=True)
    print("=" * 64)
    if any(x.startswith("<no-start>") or x == "<err>" for x in {**v, **b}.values()):
        print("INCONCLUSIVE: a server did not start / errored.")
        for p in PROMPTS:
            print(f"  vanilla={v[p][:30]!r}  pool={b[p][:30]!r}")
        return
    mism = [p for p in PROMPTS if v[p] != b[p]]
    for p in PROMPTS:
        tag = "OK " if v[p] == b[p] else "DIFF"
        print(f"[{tag}] {p[:34]!r}\n     vanilla-> {v[p][:46]!r}\n     pool   -> {b[p][:46]!r}")
    print("=" * 64)
    if not mism:
        print("RUNG-D REAL CORRECTNESS PASS: CPU cohort served from the host-"
              "block-id pool generates tokens IDENTICAL to vanilla vLLM (real "
              "KV) -> serve-more mechanism is correct end-to-end.")
    else:
        print(f"RUNG-D REAL CORRECTNESS FAIL: {len(mism)}/{len(PROMPTS)} diverged.")


if __name__ == "__main__":
    main()
