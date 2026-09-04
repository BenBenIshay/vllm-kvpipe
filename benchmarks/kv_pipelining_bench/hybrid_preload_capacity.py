#!/usr/bin/env python3
"""THE capacity question: can the mixture hold more requests than HBM alone?

Everything measured before this said no, and the b_hbm_max sweep showed why it
could never have said yes:

    b_hbm_max   riders    total   (all-HBM = 6.0)
        3        ~2        5.3
        4        ~1.5      5.7
        5        ~0.3      5.8
        6         0        6.0
        8         0        6.0

Riders shrank exactly as b_hbm_max grew, so the sum crept toward capacity from
below and never passed it. The cause was structural: a rider had to prefill into
HBM before it could be seeded to host and reclaimed, so becoming a rider cost
the very blocks the resident cohort wanted. Riders were bought WITH resident
slots, never added to them.

A preloaded rider arrives with its KV already on the host and is admitted with
no main-pool blocks at any instant, so it should ADD. This measures whether it
does, at a fixed KV pool:

    all_hbm   tiers off                      -- the ceiling to beat
    mixture   b_hbm_max = the measured ceiling, riders on top

HONESTY: this runs with VLLM_KVPIPE_RUNGD_SYNTHETIC_PRELOAD=1, which fills a
rider's KV synthetically instead of receiving it from a prefill node, because
that transport does not exist yet. That makes generated TOKENS meaningless --
exactly like the bench connectors' fake fills -- so this measures CAPACITY only.
Token correctness of the pool path is covered separately and bitwise by
pool_preload_test.py, rungd_host_pool_test.py and hybrid_rungd_real.py.

Run:
  srun ... .venv/bin/python benchmarks/kv_pipelining_bench/hybrid_preload_capacity.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = "/mnt/beegfs/benishay/vllm-kvpipe"
PY = f"{ROOT}/.venv/bin/python"
LOGDIR = "/mnt/beegfs/benishay/paper_runs/20260830"


def launch(port: int, arm: str, args) -> subprocess.Popen:
    env = dict(os.environ)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    env["PYTHONUNBUFFERED"] = "1"
    for k in ("VLLM_KVPIPE_HYBRID_RUNGD", "VLLM_KVPIPE_HYBRID_RUNGD_FREE",
              "VLLM_KVPIPE_HYBRID_TWOPASS", "VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM",
              "VLLM_KVPIPE_RUNGD_ASYNC", "VLLM_KVPIPE_RUNGD_SYNTHETIC_PRELOAD"):
        env.pop(k, None)
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server", "--model", args.model,
        "--port", str(port), "--max-model-len", str(args.ctx + args.out_len + 64),
        "--gpu-memory-utilization", str(args.gpu_mem_util),
        "--dtype", "float16", "--max-num-seqs", str(args.offered),
        "--enforce-eager", "--no-enable-prefix-caching", "--disable-log-requests",
    ]
    if arm == "mixture":
        env["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
        env["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
        env["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
        env["VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM"] = "1"
        env["VLLM_KVPIPE_RUNGD_ASYNC"] = "1"
        env["VLLM_KVPIPE_RUNGD_SYNTHETIC_PRELOAD"] = "1"
        cmd += [
            "--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
            "--hybrid-b-hbm-max", str(args.b_hbm_max),
            "--hybrid-pcie-gbps", "25.1",
            "--hybrid-itl-ms", str(args.hybrid_itl_ms),
            "--enable-kv-pipelining", "--kv-k-window", "4",
        ]
    os.makedirs(LOGDIR, exist_ok=True)
    logf = open(f"{LOGDIR}/preload_cap_{arm}.log", "w")
    return subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def wait_ready(port: int, timeout: int = 600) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def scrape(arm: str) -> dict:
    """Peak concurrency from the engine's own `Running: N reqs` line, plus the
    evidence needed to decide whether that number means anything.

    A first version of this reported `completed: 24, errors: 0` and a peak of 24
    for a run whose engine had DIED mid-execution (a NameError in the delivery
    path). The client-side counters cannot see that, so the engine log has to be
    checked for death directly -- a harness that reports success through a crash
    is worse than one that reports nothing.

    `delivered` is the other required signal: with preloaded riders there is no
    reclaim to count, because they never hold a block. If nothing was delivered,
    no request was ever pool-served and the concurrency is not a mixture result.
    """
    peak, reclaims, preload_warn, delivered = 0, 0, 0, 0
    dead: str | None = None
    try:
        with open(f"{LOGDIR}/preload_cap_{arm}.log") as f:
            for line in f:
                m = re.search(r"Running:\s*(\d+)\s*reqs", line)
                if m:
                    peak = max(peak, int(m.group(1)))
                if "RECLAIMED" in line:
                    reclaims += 1
                if "PRELOAD DELIVERED" in line:
                    delivered += 1
                if "marked pool-preloaded but no KV was delivered" in line:
                    preload_warn += 1
                if dead is None:
                    for sig in ("EngineDeadError", "EngineCore encountered",
                                "Dumping input data"):
                        if sig in line:
                            dead = sig
                            break
    except OSError:
        pass
    return {"peak_running": peak, "reclaims": reclaims, "delivered": delivered,
            "undelivered_warnings": preload_warn, "engine_died": dead}


def run_arm(port: int, arm: str, args) -> dict:
    proc = launch(port, arm, args)
    done, errs = [], []
    lock = threading.Lock()
    try:
        if not wait_ready(port):
            return {"arm": arm, "error": "server did not start", **scrape(arm)}
        base = [1000 + i for i in range(args.ctx)]

        def one(i: int) -> None:
            body = json.dumps({
                "model": args.model, "prompt": [t + i for t in base],
                "max_tokens": args.out_len, "temperature": 0.0,
                "ignore_eos": True, "min_tokens": args.out_len, "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/completions", data=body,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=args.req_timeout) as r:
                    r.read()
                with lock:
                    done.append(i)
            except urllib.error.HTTPError as e:
                try:
                    d = e.read().decode()[:160]
                except Exception:                            # noqa: BLE001
                    d = ""
                with lock:
                    errs.append(f"{i}: HTTP {e.code} {d}")
            except Exception as e:                           # noqa: BLE001
                with lock:
                    errs.append(f"{i}: {type(e).__name__}: {e}")

        threads = [threading.Thread(target=one, args=(i,), daemon=True)
                   for i in range(args.offered)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=args.req_timeout)
        wall = time.time() - t0
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(4)
    return {"arm": arm, "completed": len(done), "errors": len(errs),
            "err_sample": errs[:2], "wall_s": round(wall, 1), **scrape(arm)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--out-len", type=int, default=64)
    ap.add_argument("--offered", type=int, default=24)
    ap.add_argument("--b-hbm-max", type=int, default=6,
                    help="set to the MEASURED all-HBM ceiling: riders now add "
                         "on top of a full resident cohort instead of "
                         "displacing it, so this no longer has to be starved")
    ap.add_argument("--hybrid-itl-ms", type=float, default=100000,
                    help="capacity question, not an SLO question")
    ap.add_argument("--gpu-mem-util", type=float, default=0.83)
    ap.add_argument("--req-timeout", type=int, default=1800)
    ap.add_argument("--arms", default="all_hbm,mixture")
    ap.add_argument("--out", default=f"{LOGDIR}/preload_capacity.json")
    args = ap.parse_args()

    print(f"model={args.model} ctx={args.ctx} offered={args.offered} "
          f"b_hbm_max={args.b_hbm_max}  [SYNTHETIC KV: capacity only]",
          flush=True)
    rows = []
    for i, arm in enumerate(args.arms.split(",")):
        print(f"=== {arm} ===", flush=True)
        r = run_arm(9980 + i, arm, args)
        rows.append(r)
        print(f"    {r}", flush=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2)

    print("=" * 74)
    print(f"{'arm':<10} {'peak running':>13} {'completed':>10} {'wall':>8}  notes")
    for r in rows:
        print(f"{r['arm']:<10} {r.get('peak_running', 0):>13} "
              f"{r.get('completed', 0):>10} {r.get('wall_s', 0):>7}s  "
              f"{r.get('reclaims', 0)} reclaims"
              + (f", {r['undelivered_warnings']} UNDELIVERED"
                 if r.get("undelivered_warnings") else ""))
    print("=" * 74)
    by = {r["arm"]: r for r in rows}
    a = by.get("all_hbm", {}).get("peak_running", 0)
    m = by.get("mixture", {}).get("peak_running", 0)
    if by.get("mixture", {}).get("errors"):
        print(f"mixture had {by['mixture']['errors']} failed requests: "
              f"{by['mixture'].get('err_sample')}")
    # Validity gates BEFORE any comparison, because each of these makes the
    # number meaningless in a way the client-side counters cannot see.
    for r in rows:
        if r.get("engine_died"):
            print(f"INVALID: the {r['arm']} engine died mid-run "
                  f"({r['engine_died']}). Any concurrency scraped before the "
                  "crash is not a measurement. Fix the crash and re-run.")
            return 2
    mix = by.get("mixture", {})
    if mix and not mix.get("delivered"):
        print("INVALID: no rider ever took delivery of pool KV, so nothing was "
              "actually served from the host tier. The mixture was all-HBM "
              "under another name.")
        return 2
    if mix.get("undelivered_warnings"):
        print(f"INVALID: {mix['undelivered_warnings']} requests were admitted "
              "block-free with no KV delivered -- they were served from an "
              "empty pool.")
        return 2
    if not a or not m:
        print("INCONCLUSIVE: an arm produced no concurrency samples.")
        return 2
    if m >= args.offered:
        print(f"NOTE: the mixture hit the offered load ({args.offered}), so "
              "this is a LOWER BOUND on its capacity, not the ceiling. "
              "Re-run with more --offered to find the real limit.")
    print(f"peak concurrency  all_hbm {a}  ->  mixture {m}   "
          f"({(m / a - 1) * 100:+.0f}%)")
    if m > a:
        print("MIXTURE HOLDS MORE THAN HBM ALONE -- riders are additive: the "
              "running set exceeded what the KV pool can hold, which no "
              "b_hbm_max setting ever achieved.")
        return 0
    print("mixture does NOT exceed the all-HBM ceiling; the preloaded path did "
          "not convert to capacity. Check reclaims>0 and undelivered=0 first.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
