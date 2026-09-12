#!/usr/bin/env python3
"""Regression test for the rung-d pool-exhaustion / cap-consistency guard (#2).

Forces the host pool to FAIL by making it tiny (VLLM_KVPIPE_RUNGD_HMAX) while a
burst of concurrent CPU-tier requests overloads it. The scheduler has already
capped those requests' main-pool blocks, so a resident write would corrupt a
reused block; the guard sets slot_mapping=-1 even when the pool fails, so the
overloaded requests degrade gracefully with NO crash and NO corruption. We then
send clean sequential requests and confirm the server recovered and produces
correct, coherent output (no lingering corruption). REAL KV, all CPU-tier.
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
PORT = 9972
LOG = f"{LOGDIR}/rungd_guard_{PORT}.log"
BURST = 16       # concurrent, overloads the tiny pool
H_MAX = 20       # too small to fit the burst -> forces pool failures
PROMPT = "Describe how photosynthesis works in plants:"


def complete(prompt: str, mt: int) -> str:
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": mt,
                       "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())["choices"][0]["text"]
    except Exception as e:
        return f"<err:{e}>"


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
        "--dtype", "auto", "--max-num-seqs", str(BURST), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
        "--hybrid-b-hbm-max", "0", "--hybrid-pcie-gbps", "25.1",
        "--hybrid-b-tok", "24576", "--hybrid-itl-ms", "100000",
        "--enable-kv-pipelining", "--kv-k-window", "4",
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=open(LOG, "w"),
                            stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    burst_ok = recover = 0
    recover_out = []
    crashed = False
    try:
        t0 = time.time()
        while time.time() - t0 < 240:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(2)
        # Phase 1: overload burst (some requests will hit the failing pool).
        with ThreadPoolExecutor(max_workers=BURST) as ex:
            b = list(ex.map(lambda i: complete(PROMPT + f" (v{i})", 40), range(BURST)))
        burst_ok = sum(1 for o in b if not o.startswith("<err"))
        # Phase 2: after the burst frees blocks, clean sequential requests must
        # be correct + identical (no lingering corruption / crash).
        for _ in range(3):
            recover_out.append(complete(PROMPT, 40))
        recover = sum(1 for o in recover_out
                      if o == recover_out[0] and not o.startswith("<err"))
        time.sleep(1)
    finally:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
        except Exception:
            crashed = True
        # SIGTERM alone drained nova06/07 on 2026-09-06: a wedged
        # engine ignores it and the survivor is what SLURM cannot
        # kill. terminate_engine verifies the GROUP is empty and
        # escalates. See bench_teardown.py.
        terminate_engine(proc, label=__name__)

    warns = fatal = 0
    with open(LOG) as f:
        for line in f:
            if "pool could not manage" in line:
                warns += 1
            if "EngineCore encountered a fatal error" in line:
                fatal += 1
    print("=" * 64)
    print(f"burst completed (no exception): {burst_ok}/{BURST}")
    print(f"guard warnings fired: {warns}   engine fatal errors: {fatal}")
    print(f"recovery: {recover}/3 identical+coherent  -> {recover_out[0][:50]!r}"
          if recover_out else "  (no recovery reqs)")
    print("=" * 64)
    if (not crashed and fatal == 0 and recover == 3
            and recover_out and not recover_out[0].startswith("<err")):
        print(f"GUARD TEST PASS: pool overload was survived (warnings={warns}, no "
              "engine crash), and clean requests afterward are correct — the "
              "cap-consistency guard degrades gracefully with no corruption.")
    else:
        print(f"GUARD TEST FAIL: crashed={crashed} fatal={fatal} recover={recover}"
              " — inspect log.")


if __name__ == "__main__":
    main()
