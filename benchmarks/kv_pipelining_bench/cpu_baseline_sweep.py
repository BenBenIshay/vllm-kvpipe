#!/usr/bin/env python3
"""Production baseline: CPU cores vs concurrency at peak A100 (long-context decode).

Drives an ALREADY-RUNNING vanilla vLLM server (no KV connector, CUDA graphs on),
sweeps concurrency, and during each concurrency's STEADY-decode window samples:
  - engine process-tree CPU (cores busy)
  - busiest single thread (cores) -> GIL-bound main-loop saturation proxy
  - GPU SM-utilization + KV memory
Reports the tok/s saturation curve and the CPU load to keep the A100 fed.

NOTE: agent sandbox caps launched procs to 8 cores; numbers labeled accordingly.

Usage:
  cpu_baseline_sweep.py --port P --root-pid PID --model M --input-len 8192 \
      --output-len 200 --concurrencies 1 4 8 16 32 48 --out results.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psutil  # noqa: E402

from decode_only_bench import _generate_prompt, _send_streaming_request  # noqa: E402


def gpu_sample() -> tuple[float, float]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip().splitlines()[0]
        sm, mem = out.split(",")
        return float(sm), float(mem)
    except Exception:
        return float("nan"), float("nan")


class Sampler(threading.Thread):
    def __init__(self, root_pid: int, interval: float = 0.5):
        super().__init__(daemon=True)
        self.root_pid = root_pid
        self.interval = interval
        self._stop = threading.Event()
        self.samples: list[dict] = []

    def _tree(self):
        try:
            p = psutil.Process(self.root_pid)
            return [p] + p.children(recursive=True)
        except Exception:
            return []

    def run(self):
        while not self._stop.is_set():
            wall = time.time()
            cpu = 0.0
            tcpu: dict[tuple[int, int], float] = {}
            for p in self._tree():
                try:
                    t = p.cpu_times()
                    cpu += t.user + t.system
                except Exception:
                    pass
                try:
                    for th in p.threads():
                        tcpu[(p.pid, th.id)] = th.user_time + th.system_time
                except Exception:
                    pass
            sm, mem = gpu_sample()
            self.samples.append({"wall": wall, "cpu": cpu, "threads": tcpu,
                                 "sm": sm, "mem": mem})
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


def window_stats(samples: list[dict], lo: float, hi: float):
    win = [s for s in samples if lo <= s["wall"] <= hi]
    if len(win) < 2:
        return None
    dt = win[-1]["wall"] - win[0]["wall"]
    if dt <= 0:
        return None
    cores = (win[-1]["cpu"] - win[0]["cpu"]) / dt
    keys = set(win[0]["threads"]) & set(win[-1]["threads"])
    busiest = 0.0
    for k in keys:
        busiest = max(busiest, (win[-1]["threads"][k] - win[0]["threads"][k]) / dt)
    sm_vals = [s["sm"] for s in win if s["sm"] == s["sm"]]
    sm = sum(sm_vals) / len(sm_vals) if sm_vals else float("nan")
    mem = max((s["mem"] for s in win if s["mem"] == s["mem"]), default=float("nan"))
    return {"cores": cores, "busiest_thread": busiest, "gpu_sm": sm,
            "mem_mib": mem, "n_samples": len(win)}


def run_conc(port, model, prompt, output_len, C, sampler, timeout):
    def one():
        t0 = time.time()
        r = _send_streaming_request(port, model, prompt, output_len, timeout,
                                    itl_steady_skip=1, ignore_eos=True)
        r["_t0"] = t0
        r["_end"] = time.time()
        return r

    with ThreadPoolExecutor(max_workers=C) as pool:
        recs = [f.result() for f in [pool.submit(one) for _ in range(C)]]

    ok = [r for r in recs if r.get("success")]
    if not ok:
        return {"C": C, "ok": 0, "error": (recs[0].get("error") if recs else "?")}
    starts = [r["_t0"] + r["ttft_s"] for r in ok if r.get("ttft_s")]
    ends = [r["_end"] for r in ok]
    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s")]
    lo = max(starts) + 0.5
    hi = min(ends) - 0.3
    itls = [r["mean_itl_steady_s"] for r in ok if r.get("mean_itl_steady_s")]
    mitl = sum(itls) / len(itls) if itls else float("nan")
    ws = window_stats(sampler.samples, lo, hi) or {}
    return {
        "C": C, "ok": len(ok),
        "mean_itl_ms": round(mitl * 1000, 2),
        "decode_tok_s": round(C / mitl, 1) if mitl == mitl and mitl > 0 else 0,
        "ttft_s": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
        "steady_window_s": round(hi - lo, 2),
        "cores": round(ws.get("cores", float("nan")), 2),
        "busiest_thread_cores": round(ws.get("busiest_thread", float("nan")), 2),
        "gpu_sm_pct": round(ws.get("gpu_sm", float("nan")), 1),
        "kv_mem_mib": ws.get("mem_mib"),
        "win_samples": ws.get("n_samples"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--root-pid", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-len", type=int, default=8192)
    ap.add_argument("--output-len", type=int, default=200)
    ap.add_argument("--concurrencies", type=int, nargs="+",
                    default=[1, 4, 8, 16, 32, 48])
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"affinity cores available: {len(os.sched_getaffinity(0))}")
    prompt = _generate_prompt(args.input_len)
    sampler = Sampler(args.root_pid, interval=0.5)
    sampler.start()

    rows = []
    print(f"{'C':>4} {'ok':>4} {'ITL ms':>8} {'tok/s':>9} {'TTFT s':>7} "
          f"{'cores':>7} {'busy_thr':>9} {'GPU%':>6} {'KV MiB':>9} {'win_s':>6}")
    for C in args.concurrencies:
        # warmup wave (untimed) so graphs/caches for this batch are hot
        try:
            run_conc(args.port, args.model, prompt, min(32, args.output_len), C,
                     sampler, args.timeout)
        except Exception as e:
            print(f"  warmup C={C} err: {str(e)[:80]}")
        time.sleep(0.5)
        r = run_conc(args.port, args.model, prompt, args.output_len, C,
                     sampler, args.timeout)
        rows.append(r)
        if r.get("ok"):
            print(f"{r['C']:>4} {r['ok']:>4} {r['mean_itl_ms']:>8} "
                  f"{r['decode_tok_s']:>9} {str(r['ttft_s']):>7} "
                  f"{r['cores']:>7} {r['busiest_thread_cores']:>9} "
                  f"{r['gpu_sm_pct']:>6} {str(r['kv_mem_mib']):>9} "
                  f"{r['steady_window_s']:>6}")
        else:
            print(f"{C:>4} FAILED: {r.get('error')}")
            break

    sampler.stop()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
