#!/usr/bin/env python3
"""Step-1 in-server check for the hybrid two-pass (dual-region) decode attention.

Validates that running attention as two tier-passes over the SAME resident pool
(VLLM_KVPIPE_HYBRID_TWOPASS=1) is bitwise-equivalent to the standard single pass
(=0), end-to-end through the real server — i.e. the runner's HBM/CPU row split +
the impl's gather/scatter plumbing are correct. No memory saving yet; that is
Step 2 (CPU pass -> rolling region).

Method: fire N *distinct* greedy prompts concurrently at a hybrid server with
--hybrid-b-hbm-max = N/2, so admission tags ~half HBM and ~half CPU and the
decode steps are genuinely mixed (the only steps where two-pass activates).
Distinct prompts each have a batching-invariant greedy trajectory, so any
per-prompt divergence between the two runs is the two-pass path's fault.

Run on the GPU node:
  srun --jobid=<J> --overlap --mem=0 --gres=gpu:1 bash -lc \
    'cd /mnt/beegfs/benishay/vllm-kvpipe && HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface \
     .venv/bin/python benchmarks/kv_pipelining_bench/hybrid_twopass_check.py'
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
MAX_TOKENS = 24
B_HBM_MAX = 4

PROMPTS = [
    "The history of the Roman empire began when",
    "In quantum mechanics, the wave function describes",
    "A recipe for sourdough bread starts with",
    "The fastest land animal in the world is the",
    "To compute the derivative of a polynomial you",
    "Climate scientists measure ocean temperature using",
    "The capital city of Australia is called",
    "When debugging a segmentation fault you should",
]


def launch(port: int, twopass: bool) -> subprocess.Popen:
    env = dict(os.environ)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    env["VLLM_KVPIPE_HYBRID_STREAM"] = "0"  # resident; isolate the two-pass path
    env["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1" if twopass else "0"
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
        "--disable-log-requests", "--dtype", "auto", "--max-num-seqs", "16",
        "--enforce-eager", "--no-enable-prefix-caching",
        "--enable-kv-pipelining", "--kv-k-window", "4",
        "--kv-transfer-config", json.dumps(kv_cfg),
        "--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
        "--hybrid-b-hbm-max", str(B_HBM_MAX),
        "--hybrid-pcie-gbps", "25.1",
    ]
    logf = open(f"{LOGDIR}/twopass_server_{port}.log", "w")
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
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "max_tokens": MAX_TOKENS,
        "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["text"]


def run(port: int, twopass: bool) -> dict[str, str]:
    proc = launch(port, twopass)
    try:
        if not wait_ready(port):
            return {p: "<no-start>" for p in PROMPTS}
        # Fire all prompts concurrently so the batch is genuinely mixed.
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
    print(f"=== single-pass (TWOPASS=0), b_hbm_max={B_HBM_MAX} ===", flush=True)
    base = run(9901, twopass=False)
    print(f"=== two-pass  (TWOPASS=1), b_hbm_max={B_HBM_MAX} ===", flush=True)
    two = run(9902, twopass=True)
    print("=" * 60)
    if any(v.startswith("<no-start>") for v in {**base, **two}.values()):
        print("INCONCLUSIVE: a server did not start.")
        return
    mism = [p for p in PROMPTS if base[p] != two[p]]
    for p in PROMPTS:
        tag = "OK " if base[p] == two[p] else "DIFF"
        print(f"[{tag}] {p[:40]!r} -> {two[p][:50]!r}")
    print("=" * 60)
    if not mism:
        print(f"TWO-PASS STEP-1 PASS: all {len(PROMPTS)} prompts identical "
              "single-pass vs two-pass (same pool) -> split/gather/scatter is "
              "correct end-to-end.")
    else:
        print(f"TWO-PASS STEP-1 FAIL: {len(mism)}/{len(PROMPTS)} prompts diverged "
              "-> bug in the two-pass plumbing.")


if __name__ == "__main__":
    main()
