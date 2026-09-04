#!/usr/bin/env python3
"""Measured A100 capacity opportunity: main-pool block occupancy by tier.

The rung-c (compact read) and write-redirect mechanisms are both bitwise-
validated, so the CPU tier's KV can be served from the rolling region + host
instead of the main pool. This harness MEASURES how many main-pool blocks the
CPU tier actually holds under a sustained mixed load -> exactly the HBM that
mechanism frees, and hence the served-concurrency headroom it buys, without yet
building the allocator routing.

Drives a sustained mixed workload at a hybrid server (VLLM_KVPIPE_TIER_OCCUPANCY
=1) and parses the per-step TIER_OCCUPANCY log for steady/peak occupancy.

Run on the GPU node:
  srun --jobid=<J> --overlap --mem=0 --gres=gpu:1 bash -lc \
    'cd /mnt/beegfs/benishay/vllm-kvpipe && HF_HUB_CACHE=/mnt/beegfs/benishay/huggingface \
     .venv/bin/python benchmarks/kv_pipelining_bench/hybrid_tier_occupancy.py'
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
PORT = 9911
LOG = f"{LOGDIR}/tier_occupancy_server_{PORT}.log"
CONC = 24          # concurrent requests
B_HBM_MAX = 8      # -> ~8 hbm, rest cpu
MAX_TOKENS = 200   # long enough to accumulate several blocks/req


def main() -> None:
    env = dict(os.environ)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    env["VLLM_KVPIPE_TIER_OCCUPANCY"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    kv_cfg = {"kv_connector": "PipelinedDecodeBenchConnector", "kv_role": "kv_both",
              "kv_connector_extra_config": {"fill_mean": 0.015, "fill_std": 0.0}}
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
        "--port", str(PORT), "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.30", "--disable-log-requests",
        "--dtype", "auto", "--max-num-seqs", str(CONC), "--enforce-eager",
        "--no-enable-prefix-caching", "--enable-kv-pipelining", "--kv-k-window", "4",
        "--kv-transfer-config", json.dumps(kv_cfg),
        "--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
        "--hybrid-b-hbm-max", str(B_HBM_MAX), "--hybrid-pcie-gbps", "25.1",
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

        base = "Explain in detail, step by step, with examples, the following topic: "
        topics = [
            "photosynthesis", "the French revolution", "how CPUs pipeline",
            "quicksort", "black holes", "the immune system", "TCP congestion",
            "monetary policy", "protein folding", "the Krebs cycle",
            "plate tectonics", "neural network backprop", "the water cycle",
            "RSA encryption", "general relativity", "the stock market",
            "photolithography", "the Roman senate", "garbage collection",
            "ocean currents", "DNA replication", "compiler optimization",
            "supply and demand", "quantum entanglement",
        ]

        def one(i: int) -> None:
            body = json.dumps({"model": MODEL, "prompt": base + topics[i % len(topics)],
                               "max_tokens": MAX_TOKENS, "temperature": 0.0,
                               "stream": False}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=300).read()

        with ThreadPoolExecutor(max_workers=CONC) as ex:
            list(ex.map(one, range(CONC)))
        time.sleep(2)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(3)

    # Parse occupancy; report the step of peak combined main-pool usage.
    pat = re.compile(
        r"TIER_OCCUPANCY: n_hbm=(\d+) n_cpu=(\d+) blk_hbm=(\d+) blk_cpu=(\d+) "
        r"free=(-?\d+) total=(-?\d+)")
    rows = []
    with open(LOG) as f:
        for line in f:
            m = pat.search(line)
            if m:
                rows.append(tuple(int(x) for x in m.groups()))
    if not rows:
        print("INCONCLUSIVE: no TIER_OCCUPANCY lines (no mixed running steps).")
        return
    # Peak by main-pool blocks used (blk_hbm + blk_cpu).
    peak = max(rows, key=lambda r: r[2] + r[3])
    n_hbm, n_cpu, blk_hbm, blk_cpu, free, total = peak
    used = blk_hbm + blk_cpu
    freeable_frac = blk_cpu / used if used else 0.0
    avg_per_req = used / (n_hbm + n_cpu) if (n_hbm + n_cpu) else 0.0
    extra_hbm_reqs = (blk_cpu / avg_per_req) if avg_per_req else 0.0
    cap_mult = (1.0 / (1.0 - freeable_frac)) if freeable_frac < 1.0 else float("inf")
    print("=" * 64)
    print(f"steps measured: {len(rows)}")
    print(f"PEAK main-pool step: n_hbm={n_hbm} n_cpu={n_cpu} "
          f"blk_hbm={blk_hbm} blk_cpu={blk_cpu} free={free} total={total}")
    print(f"main-pool blocks the CPU tier holds (freeable by the rolling "
          f"mechanism): {blk_cpu} / {used} used = {freeable_frac*100:.1f}%")
    print(f"avg blocks/req = {avg_per_req:.1f}  ->  freeing enables "
          f"~{extra_hbm_reqs:.1f} more HBM-cohort requests")
    print(f"implied HBM-cohort capacity multiplier ~= {cap_mult:.2f}x "
          f"(1 / (1 - freeable_frac))")
    print("=" * 64)
    print("MEASURED CAPACITY OPPORTUNITY (A100): the validated rung-c + write-"
          f"redirect mechanism frees {freeable_frac*100:.1f}% of used main-pool "
          "KV blocks. Realizing served-concurrency needs the allocator routing "
          "(GH200-phase); this is the measured headroom it unlocks.")


if __name__ == "__main__":
    main()
