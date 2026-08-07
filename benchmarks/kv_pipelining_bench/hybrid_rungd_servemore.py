#!/usr/bin/env python3
"""Serve-more measurement for rung-d (A100): served concurrency vs HBM-only.

With a small KV pool, admit a burst of long-generation requests. Without rung-d,
the CPU-tier cohort consumes full main-pool blocks (prefill+decode), so peak
concurrency is limited by the pool. With rung-d (RUNGD_FREE=1), the CPU cohort's
decode blocks are capped/reclaimed (its KV lives in the host-block-id pool), so
MORE requests run concurrently. Uses REAL KV (no fake-filling bench connector)
so the served requests are genuinely correct. Peak concurrency is read from the
scheduler's TIER_OCCUPANCY log.
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
N_REQS = 32
MAX_TOKENS = 256          # long generation so decode blocks dominate
GPU_MEM = "0.06"          # small KV pool -> capacity-bound
B_HBM_MAX = 6


def launch(port: int, rungd: bool) -> subprocess.Popen:
    env = dict(os.environ)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    env["PYTHONUNBUFFERED"] = "1"
    env["VLLM_KVPIPE_TIER_OCCUPANCY"] = "1"
    if rungd:
        env["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
        env["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
        env["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
        "--port", str(port), "--max-model-len", "1024",
        "--gpu-memory-utilization", GPU_MEM, "--disable-log-requests",
        "--dtype", "auto", "--max-num-seqs", str(N_REQS), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
        "--hybrid-b-hbm-max", str(B_HBM_MAX), "--hybrid-pcie-gbps", "25.1",
        "--hybrid-b-tok", "131072", "--enable-kv-pipelining", "--kv-k-window", "4",
    ]
    logf = open(f"{LOGDIR}/servemore_{port}.log", "w")
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


def complete(port: int, i: int) -> int:
    prompt = f"Write a long detailed essay (topic #{i}) about the history of science:"
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": MAX_TOKENS,
                       "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
        return len(d.get("choices", [{}])[0].get("text", ""))
    except Exception:
        return -1


def run(port: int, rungd: bool) -> dict:
    proc = launch(port, rungd)
    t0 = time.time()
    try:
        if not wait_ready(port):
            return {"ok": 0, "peak_running": 0, "wall": 0.0}
        t_start = time.time()
        with ThreadPoolExecutor(max_workers=N_REQS) as ex:
            res = list(ex.map(lambda i: complete(port, i), range(N_REQS)))
        wall = time.time() - t_start
        ok = sum(1 for r in res if r > 0)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(3)
    # Parse peak concurrency + peak main-pool usage from the occupancy log.
    pat = re.compile(r"TIER_OCCUPANCY: n_hbm=(\d+) n_cpu=(\d+) blk_hbm=(\d+) "
                     r"blk_cpu=(\d+) free=(-?\d+) total=(-?\d+)")
    peak_run = 0
    peak_used = 0
    total_blocks = 0
    with open(f"{LOGDIR}/servemore_{port}.log") as f:
        for line in f:
            m = pat.search(line)
            if m:
                nh, nc, bh, bc, fr, tot = (int(x) for x in m.groups())
                peak_run = max(peak_run, nh + nc)
                peak_used = max(peak_used, bh + bc)
                total_blocks = tot
    return {"ok": ok, "peak_running": peak_run, "peak_blocks_used": peak_used,
            "total_blocks": total_blocks, "wall": wall}


def main() -> None:
    print(f"=== BASELINE (hybrid tiers, NO rung-d) — small pool, {N_REQS} reqs ===",
          flush=True)
    base = run(9941, rungd=False)
    print(f"  {base}", flush=True)
    print(f"=== RUNG-D (CPU cohort reclaimed) — same pool, {N_REQS} reqs ===",
          flush=True)
    rd = run(9942, rungd=True)
    print(f"  {rd}", flush=True)
    print("=" * 64)
    print(f"peak concurrent running: baseline={base['peak_running']}  "
          f"rung-d={rd['peak_running']}")
    print(f"completed: baseline={base['ok']}/{N_REQS}  rung-d={rd['ok']}/{N_REQS}")
    print(f"wall time: baseline={base['wall']:.1f}s  rung-d={rd['wall']:.1f}s")
    print(f"pool total blocks={base.get('total_blocks')}")
    if rd["peak_running"] > base["peak_running"] and rd["ok"] >= base["ok"]:
        print(f"SERVE-MORE PASS: rung-d ran {rd['peak_running']} concurrent vs "
              f"{base['peak_running']} baseline at the same pool "
              f"({rd['peak_running']/max(1,base['peak_running']):.2f}x) — the CPU "
              "cohort's reclaimed HBM lets more requests run.")
    else:
        print("SERVE-MORE: no concurrency gain measured (check pool sizing / load).")


if __name__ == "__main__":
    main()
