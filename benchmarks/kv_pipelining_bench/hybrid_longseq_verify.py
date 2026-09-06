#!/usr/bin/env python3
"""Bitwise check of the host-tier pool at LONG sequences.

`hybrid_reclaim_isolate.py` found that one of six host-tier requests generates
different tokens from vanilla with prompts of ~326 tokens, and that it does so
with the reclaim OFF -- so the effect predates the reclaim and lives in the
rung-d pool path itself. Every previous bitwise gate for that path used ~10
token prompts (one block per request); this is the first look at 20+ blocks.

Two very different things produce that symptom:

  (a) the pool LOSES or CORRUPTES KV at long sequences -- a real defect;
  (b) the pool reproduces the same attention, but the CPU pass is a separate
      flash call over a different-shaped batch, so its reductions land in a
      different order and the last bits differ. Greedy decoding then flips on
      a near-tie and diverges from there.

`VLLM_KVPIPE_HYBRID_TWOPASS=2` separates them. For every step it computes the
standard single pass over the RESIDENT blocks into a scratch buffer, runs the
pool override, and reports max|diff| between them -- same process, same inputs,
so batching nondeterminism cannot contaminate it.

  max|diff| == 0   -> (b): the pool is exact; the token divergence is reduction
                     order in a separate kernel call, inherent to two-pass.
  max|diff| >> eps -> (a): the pool is losing KV. Fix before anything else.

TWO SETTINGS ARE LOAD-BEARING, and getting either wrong fakes a failure:

  * the reclaim MUST be off -- it frees the resident blocks the reference pass
    reads;
  * `VLLM_KVPIPE_HYBRID_RUNGD_NOWRITE=0` MUST be set. By default the host tier
    SKIPS the resident write for its decode tokens (their KV lives in the pool),
    so resident holds prefill only and the reference would attend a sequence
    missing every generated token. A first run of this script left it at the
    default and reported max|diff|=9.6 on 1152/1152 layer-steps -- which cannot
    be right, since 5 of 6 prompts are token-identical to vanilla. The resident
    write must stay on for resident to be a valid reference.
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
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = "/mnt/beegfs/benishay/vllm-kvpipe"
PY = f"{ROOT}/.venv/bin/python"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LOGDIR = "/mnt/beegfs/benishay/paper_runs/20260813"
PORT = 9960
LOG = f"{LOGDIR}/longseq_verify_{PORT}.log"
MAX_TOKENS = 48

_BODIES = [
    "the Roman republic collapsed into empire under a general who crossed a river",
    "a wave function assigns a complex amplitude to every configuration of a system",
    "sourdough fermentation depends on a stable culture of yeast and lactobacilli",
    "the transatlantic cable required insulating a conductor against seawater",
    "glacial moraines record the farthest advance of ice across a valley floor",
    "a compiler lowers a typed syntax tree into a sequence of machine operations",
]
PROMPTS = [
    ("Consider the following account. " + (body + " ") * 24
     + "Summarise the account in one sentence:")
    for body in _BODIES
]


def main() -> int:
    env = dict(os.environ)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    env["PYTHONUNBUFFERED"] = "1"
    env["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
    env["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
    env["VLLM_KVPIPE_HYBRID_TWOPASS"] = "2"          # self-verify
    env.pop("VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM", None)  # reference needs resident
    env["VLLM_KVPIPE_HYBRID_RUNGD_NOWRITE"] = "0"      # ...and needs it COMPLETE

    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
        "--port", str(PORT), "--max-model-len", "1024",
        "--gpu-memory-utilization", "0.30",
        "--num-gpu-blocks-override", "512",
        "--dtype", "auto", "--max-num-seqs", str(len(PROMPTS)), "--enforce-eager",
        "--no-enable-prefix-caching", "--disable-log-requests",
        "--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
        "--hybrid-b-hbm-max", "0", "--hybrid-pcie-gbps", "25.1",
        "--hybrid-b-tok", "131072", "--hybrid-itl-ms", "100000",
        "--enable-kv-pipelining", "--kv-k-window", "4",
    ]
    os.makedirs(LOGDIR, exist_ok=True)
    proc = subprocess.Popen(cmd, env=env, stdout=open(LOG, "w"),
                            stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    try:
        t0 = time.time()
        while time.time() - t0 < 300:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(2)

        def one(p: str) -> None:
            body = json.dumps({"model": MODEL, "prompt": p,
                               "max_tokens": MAX_TOKENS, "temperature": 0.0,
                               "stream": False}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=900).read()

        with ThreadPoolExecutor(max_workers=len(PROMPTS)) as ex:
            list(ex.map(one, PROMPTS))
        time.sleep(3)
    finally:
        # SIGTERM alone drained nova06/07 on 2026-09-06: a wedged
        # engine ignores it and the survivor is what SLURM cannot
        # kill. terminate_engine verifies the GROUP is empty and
        # escalates. See bench_teardown.py.
        terminate_engine(proc, label=__name__)

    pat = re.compile(r"max\|diff\|=([0-9.eE+-]+) mismatched_rows=(\d+)/(\d+)")
    worst, checked, bad_steps, bad_rows = 0.0, 0, 0, 0
    with open(LOG) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            checked += 1
            worst = max(worst, float(m.group(1)))
            if int(m.group(2)) > 0:
                bad_steps += 1
                bad_rows += int(m.group(2))

    print("=" * 68)
    print(f"prompts: {len(PROMPTS)} x ~326 tokens, {MAX_TOKENS} generated")
    print(f"verify layer-steps checked : {checked}")
    print(f"worst max|diff|            : {worst:.3e}")
    print(f"layer-steps with a bad row : {bad_steps}")
    print(f"total mismatched rows      : {bad_rows}")
    print("=" * 68)
    if checked == 0:
        print("INCONCLUSIVE: no verify lines -- the pool override never ran.")
        return 2
    if worst == 0.0 and bad_rows == 0:
        print("LONG-SEQ POOL VERIFY PASS: at ~326-token sequences the host-tier "
              "pool reproduces resident attention BITWISE. The pool does not "
              "lose KV, so the token divergence against vanilla is reduction "
              "order in a separately-shaped flash call, not a data defect.")
        return 0
    print("LONG-SEQ POOL VERIFY FAIL: the pool does NOT reproduce resident "
          "attention at long sequences -- a real defect in the pool path, "
          "independent of the reclaim. Fix this before measuring anything.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
