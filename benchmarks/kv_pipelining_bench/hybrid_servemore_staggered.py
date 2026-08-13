#!/usr/bin/env python3
"""Can the mixture hold MORE concurrent requests than HBM alone? Staggered.

`endpoint_comparison.py` says no, and at 48-requests-at-once it is right, but
for a reason that is an artefact of the arrival pattern rather than the design:

  A host-tier request must materialise its full prefill in HBM before the worker
  can seed it to host and the scheduler can reclaim it. Fire 48 requests at once
  and the scheduler admits whatever fits in HBM -- there is never room for a
  49th to prefill, so no request ever becomes a rider. Measured tier samples in
  that setup: {'hbm': 1584}, i.e. zero host-tier requests, at every b_hbm_max
  at or above the all-HBM capacity.

But that residency is TRANSIENT. Riders do not need simultaneous HBM room, only
sequential room: each one prefills, is seeded to host, is reclaimed, and hands
its blocks to the next. They coexist afterwards, during decode, holding no HBM.

So this harness admits requests in WAVES instead of a burst, with long
generation so early arrivals are still decoding when late ones arrive, and asks
one question:

    what is the highest concurrency actually sustained, tiers on vs off,
    at the SAME fixed KV pool?

If the mixture exceeds the all-HBM ceiling, the reclaim converts to capacity in
a way the burst experiment structurally cannot show. If it does not, the design
needs KV to arrive already in host memory (a real P/D decode node), and that
should be stated plainly rather than tuned around.

Run:
  srun ... .venv/bin/python benchmarks/kv_pipelining_bench/hybrid_servemore_staggered.py
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
LOGDIR = "/mnt/beegfs/benishay/paper_runs/20260813"


def launch(port: int, arm: str, args) -> subprocess.Popen:
    env = dict(os.environ)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    env["PYTHONUNBUFFERED"] = "1"
    for k in ("VLLM_KVPIPE_HYBRID_RUNGD", "VLLM_KVPIPE_HYBRID_RUNGD_FREE",
              "VLLM_KVPIPE_HYBRID_TWOPASS", "VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM",
              "VLLM_KVPIPE_RUNGD_ASYNC"):
        env.pop(k, None)
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server", "--model", args.model,
        "--port", str(port), "--max-model-len", str(args.ctx + args.out_len + 64),
        "--gpu-memory-utilization", str(args.gpu_mem_util),
        "--dtype", "float16", "--max-num-seqs", str(args.offered),
        "--enforce-eager", "--no-enable-prefix-caching", "--disable-log-requests",
    ]
    if args.num_blocks:
        cmd += ["--num-gpu-blocks-override", str(args.num_blocks)]
    if arm == "mixture":
        env["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
        env["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
        env["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
        env["VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM"] = "1"
        env["VLLM_KVPIPE_RUNGD_ASYNC"] = "1"      # pays now the gather is gone
        cmd += [
            "--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
            "--hybrid-b-hbm-max", str(args.b_hbm_max),
            "--hybrid-pcie-gbps", "25.1", "--hybrid-b-tok", "131072",
            # Capacity question, not an SLO question: let admission accept
            # riders and measure what the machine actually sustains.
            "--hybrid-itl-ms", "100000",
            "--enable-kv-pipelining", "--kv-k-window", "4",
        ]
    os.makedirs(LOGDIR, exist_ok=True)
    logf = open(f"{LOGDIR}/servemore_{arm}.log", "w")
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


def run_arm(port: int, arm: str, args) -> dict:
    proc = launch(port, arm, args)
    done: list[int] = []
    errs: list[str] = []
    lock = threading.Lock()
    try:
        if not wait_ready(port):
            return {"arm": arm, "error": "server did not start"}
        prompt_tokens = [1000 + i for i in range(args.ctx)]

        def one(i: int) -> None:
            # vLLM's /v1/completions takes token ids in `prompt` itself; a
            # separate prompt_token_ids field is a 400. Offset the ids per
            # request so no two share a prefix.
            body = json.dumps({
                "model": args.model,
                "prompt": [t + i for t in prompt_tokens],
                "max_tokens": args.out_len, "temperature": 0.0,
                "ignore_eos": True, "min_tokens": args.out_len,
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/completions", data=body,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=args.req_timeout) as r:
                    r.read()
                with lock:
                    done.append(i)
            except urllib.error.HTTPError as e:              # noqa: BLE001
                # Read the body. Recording just "HTTPError" hid a malformed
                # request for a whole 96 s arm.
                try:
                    detail = e.read().decode()[:200]
                except Exception:                            # noqa: BLE001
                    detail = ""
                with lock:
                    errs.append(f"{i}: HTTP {e.code} {detail}")
            except Exception as e:                           # noqa: BLE001
                with lock:
                    errs.append(f"{i}: {type(e).__name__}: {e}")

        # STAGGERED arrival: this is the whole point. Each rider needs HBM only
        # long enough to prefill and be seeded; spacing arrivals lets them take
        # turns at that transient cost instead of competing for it at once.
        threads = []
        t0 = time.time()
        for i in range(args.offered):
            t = threading.Thread(target=one, args=(i,), daemon=True)
            t.start()
            threads.append(t)
            time.sleep(args.stagger_s)
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
            "wall_s": round(wall, 1), "err_sample": errs[:3],
            **scrape(arm, args)}


def scrape(arm: str, args) -> dict:
    """Peak concurrency the engine actually ran, from its own logs.

    vLLM logs "Running: N reqs" periodically; that is the engine's own count of
    requests in the running set, which is exactly the quantity under test.
    """
    path = f"{LOGDIR}/servemore_{arm}.log"
    peak, reclaims, blocks = 0, 0, 0
    try:
        with open(path) as f:
            for line in f:
                m = re.search(r"Running:\s*(\d+)\s*reqs", line)
                if m:
                    peak = max(peak, int(m.group(1)))
                m2 = re.search(r"RECLAIMED (\d+) main-pool blocks", line)
                if m2:
                    reclaims += 1
                    blocks += int(m2.group(1))
    except OSError:
        pass
    return {"peak_running": peak, "reclaim_events": reclaims,
            "blocks_reclaimed": blocks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--out-len", type=int, default=512,
                    help="long, so early arrivals still decode when late ones land")
    ap.add_argument("--offered", type=int, default=16)
    ap.add_argument("--stagger-s", type=float, default=6.0)
    ap.add_argument("--b-hbm-max", type=int, default=6)
    ap.add_argument("--gpu-mem-util", type=float, default=0.83)
    ap.add_argument("--num-blocks", type=int, default=0)
    ap.add_argument("--req-timeout", type=int, default=1800)
    ap.add_argument("--arms", default="all_hbm,mixture")
    ap.add_argument("--out", default=f"{LOGDIR}/servemore_staggered.json")
    args = ap.parse_args()

    print(f"model={args.model} ctx={args.ctx} out={args.out_len} "
          f"offered={args.offered} stagger={args.stagger_s}s "
          f"b_hbm_max={args.b_hbm_max}", flush=True)
    rows = []
    for i, arm in enumerate(args.arms.split(",")):
        print(f"=== {arm} ===", flush=True)
        r = run_arm(9970 + i, arm, args)
        rows.append(r)
        print(f"    {r}", flush=True)

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2)

    print("=" * 72)
    print(f"{'arm':<10} {'peak running':>13} {'completed':>10} {'wall':>8}  reclaims")
    for r in rows:
        print(f"{r['arm']:<10} {r.get('peak_running', 0):>13} "
              f"{r.get('completed', 0):>10} {r.get('wall_s', 0):>7}s  "
              f"{r.get('reclaim_events', 0)} ev / "
              f"{r.get('blocks_reclaimed', 0)} blocks")
    print("=" * 72)
    by = {r["arm"]: r for r in rows}
    if "all_hbm" in by and "mixture" in by:
        a = by["all_hbm"].get("peak_running", 0)
        m = by["mixture"].get("peak_running", 0)
        if by["mixture"].get("reclaim_events", 0) == 0:
            print("INCONCLUSIVE: no reclaim fired, so no request ever became a "
                  "rider. The mixture was all-HBM under another name.")
        elif a:
            print(f"peak concurrency {a} -> {m}  ({(m/a - 1)*100:+.0f}%)")
            print("MIXTURE HOLDS MORE" if m > a else
                  "mixture does NOT exceed the all-HBM ceiling")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
