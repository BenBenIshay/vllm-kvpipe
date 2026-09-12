#!/usr/bin/env python3
"""Token-level correctness gate for rung-d serve-more (host-block-id pool).

The pool path skips CPU resident writes (slot_mapping=-1) so the =2 verify (which
references resident) can't be used. Instead: run two DISTINCT greedy prompts and
compare per-prompt outputs between
  A) all-resident (b_hbm_max=2, no rung-d), and
  B) mixed with the pool (b_hbm_max=1, VLLM_KVPIPE_HYBRID_RUNGD_FREE=1) — one
     prompt is served CPU-tier entirely from the host-block-id pool.
Distinct prompts have batching-invariant greedy trajectories, so any per-prompt
divergence is the pool path's fault. Match => the CPU cohort served from the
pool (resident writes skipped) is token-identical to residency.
"""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from bench_teardown import terminate_engine  # noqa: E402

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
]


def launch(port: int, b_hbm_max: int, rungd_free: bool) -> subprocess.Popen:
    env = dict(os.environ)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    env["VLLM_KVPIPE_HYBRID_STREAM"] = "0"
    env["VLLM_KVPIPE_HYBRID_RUNGD"] = "1" if rungd_free else "0"
    env["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1" if rungd_free else "0"
    env["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1" if rungd_free else "0"
    env["PYTHONUNBUFFERED"] = "1"
    kv_cfg = {"kv_connector": "PipelinedDecodeBenchConnector", "kv_role": "kv_both",
              "kv_connector_extra_config": {"fill_mean": 0.015, "fill_std": 0.0}}
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
        "--port", str(port), "--max-model-len", "1024",
        "--gpu-memory-utilization", "0.15", "--disable-log-requests",
        "--dtype", "auto", "--max-num-seqs", "8", "--enforce-eager",
        "--no-enable-prefix-caching", "--enable-kv-pipelining", "--kv-k-window", "4",
        "--kv-transfer-config", json.dumps(kv_cfg),
        "--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
        "--hybrid-b-hbm-max", str(b_hbm_max), "--hybrid-pcie-gbps", "25.1",
        "--hybrid-b-tok", "131072",
    ]
    logf = open(f"{LOGDIR}/rungd_corr_{port}.log", "w")
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
        return json.loads(r.read())["choices"][0]["text"]


def run(port: int, b_hbm_max: int, rungd_free: bool) -> dict[str, str]:
    proc = launch(port, b_hbm_max, rungd_free)
    try:
        if not wait_ready(port):
            return {p: "<no-start>" for p in PROMPTS}
        with ThreadPoolExecutor(max_workers=len(PROMPTS)) as ex:
            outs = list(ex.map(lambda p: complete(port, p), PROMPTS))
        return dict(zip(PROMPTS, outs))
    finally:
        # SIGTERM alone drained nova06/07 on 2026-09-06: a wedged
        # engine ignores it and the survivor is what SLURM cannot
        # kill. terminate_engine verifies the GROUP is empty and
        # escalates. See bench_teardown.py.
        terminate_engine(proc, label=__name__)


def main() -> None:
    # Compare the SAME cohort (all CPU-tier, b_hbm_max=0) served resident vs via
    # the pool — toggling only rung-d. This cancels the bench connector's CPU-
    # cohort fake-fill (identical in both), so any per-prompt divergence is the
    # pool path alone. (Comparing CPU-vs-HBM would be a confound: the connector
    # treats the CPU cohort's KV differently.)
    print("=== A: CPU-tier RESIDENT (b_hbm_max=0, no rung-d) ===", flush=True)
    a = run(9921, b_hbm_max=0, rungd_free=False)
    print("=== B: CPU-tier POOL (b_hbm_max=0, RUNGD_FREE=1) ===", flush=True)
    b = run(9922, b_hbm_max=0, rungd_free=True)
    print("=" * 64)
    if any(v.startswith("<no-start>") for v in {**a, **b}.values()):
        print("INCONCLUSIVE: a server did not start.")
        return
    mism = [p for p in PROMPTS if a[p] != b[p]]
    for p in PROMPTS:
        tag = "OK " if a[p] == b[p] else "DIFF"
        print(f"[{tag}] {p[:36]!r} -> {b[p][:44]!r}")
    print("=" * 64)
    if not mism:
        print("RUNG-D POOL CORRECTNESS PASS: CPU-tier served from the host-block-"
              "id pool (resident writes skipped) is token-identical to residency.")
    else:
        print(f"RUNG-D POOL CORRECTNESS FAIL: {len(mism)}/{len(PROMPTS)} prompts "
              "diverged -> pool path bug.")


if __name__ == "__main__":
    main()
