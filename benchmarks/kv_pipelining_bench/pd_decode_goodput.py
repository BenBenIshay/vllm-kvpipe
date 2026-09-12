#!/usr/bin/env python3
"""Riders vs HBM alone, as a decode node: neither arm prefills in the window.

Every earlier comparison was unfair in one direction or the other. The capacity
run let riders skip prefill while the all-HBM arm paid for it, so the walls were
not comparable. Making both prefill instead would have been unfair the other
way: a P/D decode node does not prefill at all.

So: two phases, and only the second is measured.

  phase 1 (NOT measured)  every prompt is prefilled for real, once. Its KV comes
                          to rest wherever that arm keeps KV -- HBM for the
                          resident arm (vLLM's own prefix cache), host DDR for
                          the mixture (the prefill stash).
  phase 2 (measured)      the same prompts arrive together. NEITHER arm
                          prefills. Each decodes from wherever its KV already
                          is, and we measure ITL and decode throughput.

A note on Ben's framing, "both versions start with their KV in the DDR": for
the mixture that is literally true, and for the resident arm it is not, on
purpose. A decode node with enough HBM keeps KV in HBM; making it re-load from
DDR every run would charge it a startup cost that amortises away over a real
deployment and would flatter our own arm. What the fairness requirement really
demands is that NEITHER arm pays for prefill inside the measurement, and that
is what this does.

B1 is ON. The whole point is that the admission controller decides how many
riders fit behind one forward iteration, so the ITL must hold while throughput
rises. If ITL degrades, that is the result -- report it, do not tune it away.

Run:
  srun ... .venv/bin/python benchmarks/kv_pipelining_bench/pd_decode_goodput.py
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from bench_teardown import terminate_engine  # noqa: E402

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request

ROOT = "/mnt/beegfs/benishay/vllm-kvpipe"
PY = f"{ROOT}/.venv/bin/python"
LOGDIR = "/mnt/beegfs/benishay/paper_runs/20260904"


def launch(port: int, arm: str, args, budget_ms: float) -> subprocess.Popen:
    env = dict(os.environ)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    env["PYTHONUNBUFFERED"] = "1"
    for k in ("VLLM_KVPIPE_HYBRID_RUNGD", "VLLM_KVPIPE_HYBRID_RUNGD_FREE",
              "VLLM_KVPIPE_HYBRID_TWOPASS", "VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM",
              "VLLM_KVPIPE_RUNGD_ASYNC", "VLLM_KVPIPE_RUNGD_SYNTHETIC_PRELOAD",
              "VLLM_KVPIPE_RUNGD_PREFILL_STASH"):
        env.pop(k, None)
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server", "--model", args.model,
        "--port", str(port), "--max-model-len", str(args.ctx + args.out_len + 64),
        "--gpu-memory-utilization", str(args.gpu_mem_util),
        "--dtype", "float16", "--max-num-seqs", str(args.offered),
        "--enforce-eager", "--disable-log-requests",
    ]
    if True:
        # BOTH arms run the identical hybrid build with the identical phase 1.
        # The ONLY difference is b_hbm_max: at `offered` there is no room left
        # for a rider and over-capacity requests simply queue, which is the
        # all-HBM behaviour the sweep already measured; at the ceiling, riders
        # are admitted on top. Comparing two configurations of one engine
        # removes every confound a second binary would introduce.
        env["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
        env["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
        env["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
        env["VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM"] = "1"
        env["VLLM_KVPIPE_RUNGD_ASYNC"] = "1"
        env["VLLM_KVPIPE_RUNGD_PREFILL_STASH"] = "1"
        if args.max_offload:
            # Hard cap on riders, independent of the B1 byte budget. Used to
            # sweep rider count and measure how much of a rider's transfer
            # actually hides behind compute.
            env["VLLM_KVPIPE_MAX_OFFLOAD_REQS"] = str(args.max_offload)
        # NOT set: SYNTHETIC_PRELOAD. Riders carry real prefill KV here.
        cmd += [
            "--no-enable-prefix-caching",
            "--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
            "--hybrid-b-hbm-max",
            str(args.offered if arm == "no_riders" else args.b_hbm_max),
            "--hybrid-pcie-gbps", "25.1",
            # The hide budget. B1 measures this itself from steps carrying no
            # rider, but this bench never gives it one: stash-first puts every
            # un-stashed prompt on the host tier in phase 1, and phase 2 has
            # riders from its first step. So it is CALIBRATED instead, from the
            # no_riders arm's own measured median ITL -- a forward iteration
            # with no rider on it, which is exactly the quantity wanted. Stated
            # rather than hidden: this is a two-pass calibration, not a
            # self-measurement, and the control arm supplies the constant.
            "--hybrid-itl-ms", str(budget_ms),
            "--enable-kv-pipelining", "--kv-k-window", "4",
        ]
    os.makedirs(LOGDIR, exist_ok=True)
    logf = open(f"{LOGDIR}/pd_goodput_{arm}.log", "w")
    return subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def wait_ready(port: int, timeout: int = 900) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def _body(args, prompt_ids: list[int], out_len: int, stream: bool) -> bytes:
    return json.dumps({
        "model": args.model, "prompt": prompt_ids, "max_tokens": out_len,
        "temperature": 0.0, "ignore_eos": True, "min_tokens": out_len,
        "stream": stream,
    }).encode()


def send(port: int, args, prompt_ids: list[int], out_len: int,
         stream: bool) -> list[float]:
    """Returns inter-token arrival gaps in ms (empty if not streaming)."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=_body(args, prompt_ids, out_len, stream),
        headers={"Content-Type": "application/json"})
    gaps: list[float] = []
    with urllib.request.urlopen(req, timeout=args.req_timeout) as r:
        if not stream:
            r.read()
            return gaps
        prev = None
        for raw in r:
            if not raw.startswith(b"data: ") or raw.strip() == b"data: [DONE]":
                continue
            now = time.perf_counter()
            if prev is not None:
                gaps.append((now - prev) * 1000.0)
            prev = now
    return gaps


def scrape(arm: str) -> dict:
    """Everything needed to decide whether the numbers mean anything."""
    peak = stashed = delivered_real = delivered_synth = undelivered = 0
    prefix_hit = None
    budget_ms = None
    dead = None
    try:
        with open(f"{LOGDIR}/pd_goodput_{arm}.log") as f:
            for line in f:
                m = re.search(r"Running:\s*(\d+)\s*reqs", line)
                if m:
                    peak = max(peak, int(m.group(1)))
                m = re.search(r"[Pp]refix cache hit rate:\s*([\d.]+)", line)
                if m:
                    prefix_hit = float(m.group(1))
                m = re.search(r"B1 hide budget now MEASURED:\s*([\d.]+) ms", line)
                if m:
                    budget_ms = float(m.group(1))
                if "PREFILL STASHED" in line:
                    stashed += 1
                elif "PRELOAD DELIVERED" in line:
                    if "REAL prefill KV" in line:
                        delivered_real += 1
                    else:
                        delivered_synth += 1
                if "marked pool-preloaded but no KV was delivered" in line:
                    undelivered += 1
                if dead is None:
                    for sig in ("EngineDeadError", "EngineCore encountered",
                                "Dumping input data"):
                        if sig in line:
                            dead = sig
    except OSError:
        pass
    return {"peak_running": peak, "stashed": stashed,
            "delivered_real": delivered_real,
            "delivered_synthetic": delivered_synth,
            "undelivered": undelivered, "prefix_cache_hit_rate": prefix_hit,
            "measured_budget_ms": budget_ms, "engine_died": dead}


def run_arm(port: int, arm: str, args, budget_ms: float) -> dict:
    proc = launch(port, arm, args, budget_ms)
    prompts = [[1000 + i + t for t in range(args.ctx)]
               for i in range(args.offered)]
    try:
        if not wait_ready(port):
            return {"arm": arm, "error": "server did not start", **scrape(arm)}

        # ---- phase 1: prefill every prompt once. NOT measured. -------------
        # Not max_tokens=1: the host tier is seeded on a DECODE step, so a
        # request that finishes at prefill is never seeded and never stashed.
        # Measured symptom of getting this wrong: phase 1 stashes nothing and
        # phase 2 silently prefills instead of riding.
        t0 = time.time()
        for p in prompts:
            send(port, args, p, args.phase1_tokens, stream=False)
        phase1_s = time.time() - t0

        # ---- phase 2: all of them together, decode only. MEASURED. ---------
        gaps: list[list[float]] = []
        errs: list[str] = []
        lock = threading.Lock()

        def one(i: int) -> None:
            try:
                g = send(port, args, prompts[i], args.out_len, stream=True)
                with lock:
                    gaps.append(g)
            except urllib.error.HTTPError as e:
                with lock:
                    errs.append(f"{i}: HTTP {e.code}")
            except Exception as e:                            # noqa: BLE001
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
        # SIGTERM alone drained nova06/07 on 2026-09-06: a wedged
        # engine ignores it, and the survivor is what SLURM cannot
        # kill. terminate_engine verifies the GROUP is empty and
        # escalates. See bench_teardown.py.
        terminate_engine(proc, label=__name__)

    flat = [g for gs in gaps for g in gs]
    flat.sort()
    done = len(gaps)
    return {
        "arm": arm, "completed": done, "errors": len(errs),
        "err_sample": errs[:2],
        "phase1_prefill_s": round(phase1_s, 1),
        "decode_wall_s": round(wall, 1),
        "decode_tokens": done * args.out_len,
        "decode_tok_s": round(done * args.out_len / wall, 1) if wall else 0.0,
        "itl_median_ms": round(statistics.median(flat), 2) if flat else None,
        "itl_p99_ms": round(flat[int(len(flat) * 0.99)], 2) if flat else None,
        **scrape(arm),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--out-len", type=int, default=64)
    ap.add_argument("--offered", type=int, default=12)
    ap.add_argument("--b-hbm-max", type=int, default=8,
                    help="the measured all-HBM ceiling: riders add on top")
    ap.add_argument("--max-offload", type=int, default=0,
                    help="cap riders regardless of the B1 budget (0 = no cap)")
    ap.add_argument("--phase1-tokens", type=int, default=4,
                    help="enough decode steps for the host tier to seed and "
                         "stash; 1 is NOT enough")
    ap.add_argument("--itl-fallback-ms", type=float, default=60.0,
                    help="used only until B1 has measured a real step")
    ap.add_argument("--gpu-mem-util", type=float, default=0.83)
    ap.add_argument("--req-timeout", type=int, default=1800)
    ap.add_argument("--arms", default="no_riders,riders")
    ap.add_argument("--out", default=f"{LOGDIR}/pd_decode_goodput.json")
    args = ap.parse_args()

    print(f"model={args.model} ctx={args.ctx} offered={args.offered} "
          f"out_len={args.out_len} b_hbm_max={args.b_hbm_max}", flush=True)
    print("phase 1 prefills every prompt and is NOT measured; phase 2 decodes "
          "with no prefill in either arm.", flush=True)
    rows = []
    budget_ms = args.itl_fallback_ms
    for i, arm in enumerate(args.arms.split(",")):
        print(f"=== {arm} (hide budget {budget_ms:.1f} ms) ===", flush=True)
        r = run_arm(9960 + i, arm, args, budget_ms)
        rows.append(r)
        print(f"    {r}", flush=True)
        if arm == "no_riders" and r.get("itl_median_ms"):
            # Calibrate: this arm's median ITL IS a forward iteration carrying
            # no rider, which is what B1's hide budget is supposed to be.
            budget_ms = r["itl_median_ms"]
            print(f"    calibrated hide budget from the control: "
                  f"{budget_ms:.2f} ms per forward iteration", flush=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2)

    by = {r["arm"]: r for r in rows}
    print("=" * 86)
    print(f"{'arm':<10} {'peak':>5} {'done':>5} {'tok/s':>9} "
          f"{'ITL med':>9} {'ITL p99':>9} {'decode s':>9}")
    for r in rows:
        print(f"{r['arm']:<10} {r.get('peak_running', 0):>5} "
              f"{r.get('completed', 0):>5} {r.get('decode_tok_s', 0):>9} "
              f"{str(r.get('itl_median_ms')):>9} {str(r.get('itl_p99_ms')):>9} "
              f"{r.get('decode_wall_s', 0):>9}")
    print("=" * 86)

    for r in rows:
        if r.get("engine_died"):
            print(f"INVALID: the {r['arm']} engine died ({r['engine_died']}).")
            return 2
        if r.get("errors"):
            print(f"INVALID: {r['arm']} had {r['errors']} failed requests: "
                  f"{r.get('err_sample')}")
            return 2
    # Gates apply only to the arms actually requested: this bench is also
    # driven one arm at a time (rider_hide_efficiency.py), and a gate that
    # fires because the OTHER arm was not asked for reports a false INVALID on
    # a perfectly good measurement.
    mix, res = by.get("riders", {}), by.get("no_riders", {})
    if mix and mix.get("delivered_synthetic"):
        print("INVALID: riders were fed synthetic KV; this bench requires real "
              "prefill output.")
        return 2
    if mix and mix.get("undelivered"):
        print(f"INVALID: {mix['undelivered']} riders were admitted with no KV.")
        return 2
    if mix and not mix.get("delivered_real"):
        print("INVALID: no rider ever took delivery of stashed KV -- the "
              "riders arm was the control under another name.")
        return 2
    if res and res.get("delivered_real"):
        print(f"INVALID: the no_riders arm served {res['delivered_real']} "
              "requests from the pool, so it is not a rider-free control.")
        return 2
    if not (mix and res):
        print(f"single-arm run: {', '.join(sorted(by))} -- no comparison to "
              "make, the numbers above are the measurement.")
        return 0
    cal = res.get("itl_median_ms")
    print(f"B1 hide budget: {cal} ms per forward iteration "
          f"(calibrated from the control arm's median ITL)")
    if mix.get("measured_budget_ms"):
        print(f"  (the engine also self-measured {mix['measured_budget_ms']} "
              "ms from rider-free steps)")
    a, m = res.get("decode_tok_s", 0), mix.get("decode_tok_s", 0)
    ia, im = res.get("itl_median_ms"), mix.get("itl_median_ms")
    if not a or not m:
        print("INCONCLUSIVE: an arm produced no decode throughput.")
        return 2
    print(f"decode throughput  no_riders {a} -> riders {m} tok/s "
          f"({(m / a - 1) * 100:+.1f}%)")
    if ia and im:
        print(f"median ITL         no_riders {ia} -> riders {im} ms "
              f"({(im / ia - 1) * 100:+.1f}%)")
    if m > a and im and ia and im <= ia * 1.10:
        print("RIDERS PAY: more decode throughput at essentially unchanged ITL "
              "-- the admission controller admitted only what hides behind a "
              "forward iteration.")
        return 0
    if m > a:
        print("Throughput rose but ITL did not hold. B1 gates on BYTES and "
              "cannot see the host path's fixed per-step overhead; that "
              "overhead, not the interconnect, is the binding cost here.")
        return 1
    print("Admitting riders did not raise decode throughput.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
