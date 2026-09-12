#!/usr/bin/env python3
"""Definitive in-process bitwise check of the hybrid two-pass decode.

Launches ONE hybrid server with VLLM_KVPIPE_HYBRID_TWOPASS=2 (self-verify): for
every mixed decode step the impl computes the standard single pass AND the two
pass on the SAME inputs and logs max|diff| + mismatched_rows per layer. Immune
to cross-run batching nondeterminism (both paths see identical tensors). Sends a
few concurrent distinct prompts to force mixed batches, then parses the server
log and reports the worst diff seen.
"""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from bench_teardown import terminate_engine  # noqa: E402

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
PORT = 9903
LOG = f"{LOGDIR}/twopass_verify_server_{PORT}.log"
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


def main() -> None:
    env = dict(os.environ)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    env["VLLM_KVPIPE_HYBRID_STREAM"] = "0"
    env["VLLM_KVPIPE_HYBRID_TWOPASS"] = "2"  # self-verify
    env["PYTHONUNBUFFERED"] = "1"
    kv_cfg = {
        "kv_connector": "PipelinedDecodeBenchConnector", "kv_role": "kv_both",
        "kv_connector_extra_config": {"fill_mean": 0.015, "fill_std": 0.0},
    }
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
        "--port", str(PORT), "--max-model-len", "1024",
        "--gpu-memory-utilization", "0.15", "--disable-log-requests",
        "--dtype", "auto", "--max-num-seqs", "16", "--enforce-eager",
        "--no-enable-prefix-caching", "--enable-kv-pipelining", "--kv-k-window", "4",
        "--kv-transfer-config", json.dumps(kv_cfg),
        "--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
        "--hybrid-b-hbm-max", "4", "--hybrid-pcie-gbps", "25.1",
        "--hybrid-b-tok", "131072",
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=open(LOG, "w"),
                            stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    try:
        t0 = time.time()
        while time.time() - t0 < 240:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(2)

        def one(p: str) -> None:
            body = json.dumps({"model": MODEL, "prompt": p, "max_tokens": 24,
                               "temperature": 0.0, "stream": False}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=180).read()

        with ThreadPoolExecutor(max_workers=len(PROMPTS)) as ex:
            list(ex.map(one, PROMPTS))
        time.sleep(2)
    finally:
        # SIGTERM alone drained nova06/07 on 2026-09-06: a wedged
        # engine ignores it and the survivor is what SLURM cannot
        # kill. terminate_engine verifies the GROUP is empty and
        # escalates. See bench_teardown.py.
        terminate_engine(proc, label=__name__)

    # Parse verify lines.
    pat = re.compile(r"max\|diff\|=([0-9.eE+-]+) mismatched_rows=(\d+)/(\d+)")
    worst = 0.0
    total_lines = 0
    bad_lines = 0
    total_bad_rows = 0
    with open(LOG) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            total_lines += 1
            diff = float(m.group(1))
            bad = int(m.group(2))
            worst = max(worst, diff)
            if bad > 0:
                bad_lines += 1
                total_bad_rows += bad
    print("=" * 60)
    print(f"verify layer-steps checked: {total_lines}")
    print(f"worst max|diff| across all layer-steps: {worst:.3e}")
    print(f"layer-steps with any mismatched row: {bad_lines}")
    print(f"total mismatched rows: {total_bad_rows}")
    if total_lines == 0:
        print("INCONCLUSIVE: no verify lines — two-pass never activated "
              "(no mixed decode step formed).")
    elif worst == 0.0 and total_bad_rows == 0:
        print("TWO-PASS VERIFY PASS: two-pass == single-pass BITWISE on every "
              "mixed decode layer-step -> plumbing is correct; token-level "
              "diffs are batching nondeterminism, not a two-pass bug.")
    else:
        print("TWO-PASS VERIFY FAIL: two-pass diverges from single-pass on "
              "identical inputs -> real plumbing bug to fix.")


if __name__ == "__main__":
    main()
