#!/usr/bin/env python3
"""Isolate the long-prefill divergence found by hybrid_reclaim_correctness.py.

That test found 1 of 6 host-tier requests generating different tokens from
vanilla, with the vanilla-vs-vanilla control clean, so the difference is real.
It does not say WHICH mechanism is at fault. Four arms, one variable at a time:

  vanilla        no tiers at all                          -- the reference
  host_only      host tier, pool-managed, NO reclaim      -- the configuration
                 (RUNGD_FREE=1, RECLAIM unset)               validated before
                                                             today, but only
                                                             ever at ~10-token
                                                             prompts
  reclaim_big    reclaim ON, pool large enough that the   -- reclaim without
                 freed blocks are never handed out again     block reuse
  reclaim_small  reclaim ON, tight pool, blocks recycled  -- the failing case

Reading the result:
  host_only diverges too  -> the defect PREDATES the reclaim and lives in the
                             rung-d pool path at long sequences; the reclaim is
                             exonerated (and the earlier PASSes were too short
                             to see it).
  host_only clean,
  reclaim_big diverges    -> the reclaim itself loses KV.
  both clean,
  reclaim_small diverges  -> something still READS the resident blocks after
                             they are freed and reused.

Run with --prompts to narrow to the failing prompt once identified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = "/mnt/beegfs/benishay/vllm-kvpipe"
PY = f"{ROOT}/.venv/bin/python"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LOGDIR = "/mnt/beegfs/benishay/paper_runs/20260813"
MAX_TOKENS = 48
MAX_MODEL_LEN = 1024
GPU_MEM = "0.30"

_BODIES = [
    "the Roman republic collapsed into empire under a general who crossed a river",
    "a wave function assigns a complex amplitude to every configuration of a system",
    "sourdough fermentation depends on a stable culture of yeast and lactobacilli",
    "the transatlantic cable required insulating a conductor against seawater",
    "glacial moraines record the farthest advance of ice across a valley floor",
    "a compiler lowers a typed syntax tree into a sequence of machine operations",
]
ALL_PROMPTS = [
    ("Consider the following account. " + (body + " ") * 24
     + "Summarise the account in one sentence:")
    for body in _BODIES
]

# arm -> (host tier?, reclaim?, kv blocks)
ARMS = {
    "vanilla":       (False, False, 192),
    "host_only":     (True,  False, 192),
    "reclaim_big":   (True,  True,  2048),
    "reclaim_small": (True,  True,  192),
}


def launch(port: int, arm: str, n_reqs: int) -> subprocess.Popen:
    host_tier, reclaim, blocks = ARMS[arm]
    env = dict(os.environ)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    env["PYTHONUNBUFFERED"] = "1"
    for k in ("VLLM_KVPIPE_HYBRID_RUNGD", "VLLM_KVPIPE_HYBRID_RUNGD_FREE",
              "VLLM_KVPIPE_HYBRID_TWOPASS", "VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM"):
        env.pop(k, None)
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
        "--port", str(port), "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", GPU_MEM,
        "--num-gpu-blocks-override", str(blocks),
        "--dtype", "auto", "--max-num-seqs", str(max(n_reqs, 1)),
        "--enforce-eager", "--no-enable-prefix-caching", "--disable-log-requests",
    ]
    if host_tier:
        env["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
        env["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
        env["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
        if reclaim:
            env["VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM"] = "1"
        cmd += [
            "--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
            "--hybrid-b-hbm-max", "0", "--hybrid-pcie-gbps", "25.1",
            "--hybrid-b-tok", "131072", "--hybrid-itl-ms", "100000",
            "--enable-kv-pipelining", "--kv-k-window", "4",
        ]
    os.makedirs(LOGDIR, exist_ok=True)
    logf = open(f"{LOGDIR}/isolate_{arm}.log", "w")
    return subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def wait_ready(port: int, timeout: int = 300) -> bool:
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
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["choices"][0]["text"]


def run(port: int, arm: str, prompts: list[str]) -> list[str]:
    proc = launch(port, arm, len(prompts))
    try:
        if not wait_ready(port):
            return ["<no-start>"] * len(prompts)
        with ThreadPoolExecutor(max_workers=len(prompts)) as ex:
            return list(ex.map(lambda p: complete(port, p), prompts))
    except Exception as e:                                   # noqa: BLE001
        return [f"<err: {e}>"] * len(prompts)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(4)


def reclaim_count(arm: str) -> int:
    try:
        with open(f"{LOGDIR}/isolate_{arm}.log") as f:
            return sum(1 for line in f
                       if re.search(r"RECLAIMED \d+ main-pool blocks", line))
    except OSError:
        return 0


def first_diff(a: str, b: str) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"char {i}: {a[max(0, i - 25):i + 25]!r} vs {b[max(0, i - 25):i + 25]!r}"
    return f"prefix (len {len(a)} vs {len(b)})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="",
                    help="comma-separated prompt indices (default: all)")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()

    idx = ([int(x) for x in args.prompts.split(",")] if args.prompts
           else list(range(len(ALL_PROMPTS))))
    prompts = [ALL_PROMPTS[i] for i in idx]
    arms = args.arms.split(",")
    print(f"prompts {idx}, arms {arms}", flush=True)

    results: dict[str, list[str]] = {}
    for i, arm in enumerate(arms):
        print(f"=== {arm} ===", flush=True)
        results[arm] = run(9950 + i, arm, prompts)
        if ARMS[arm][1]:
            print(f"    (reclaim events: {reclaim_count(arm)})", flush=True)

    with open(f"{LOGDIR}/isolate_outputs.json", "w") as f:
        json.dump({"prompt_indices": idx, "results": results}, f, indent=2)

    ref = results.get("vanilla")
    print("=" * 70)
    if ref is None:
        print("no vanilla arm -- nothing to compare against")
        return 2
    for arm in arms:
        if arm == "vanilla":
            continue
        got = results[arm]
        bad = [j for j in range(len(prompts)) if got[j] != ref[j]]
        status = "CLEAN" if not bad else f"DIVERGED on prompts {[idx[j] for j in bad]}"
        print(f"{arm:>14}: {status}")
        for j in bad:
            print(f"                {first_diff(ref[j], got[j])}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
