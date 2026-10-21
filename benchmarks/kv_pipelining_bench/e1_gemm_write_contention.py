#!/usr/bin/env python3
"""E1: do decode GEMMs slow down when something else writes into HBM?

M23 measured that a rider costs ~8-10% of its link time more with real bytes
than with the bytes removed, and part of that is the decode GEMMs running ~2%
slower while the copy stream is busy. On A100 the link (PCIe, ~25 GB/s) writes
into HBM at 1.3% of HBM bandwidth; on GH200 C2C writes at 419 of ~4000 GB/s,
about 10.5%. If the slowdown is HBM contention it grows with that share, and
the GH200 projection must carry it; if it is something PCIe/copy-engine
specific, the A100 fraction is the better estimate. This separates the two.

Method, off-engine (no vLLM):
  * Workload: one decode step's weight GEMMs for Qwen2.5-32B shapes, M rows
    (residents + riders), fp16, cuBLAS, over N_LAYERS distinct layers so every
    weight read comes from HBM (each pass reads ~N_LAYERS x 0.975 GB).
  * Interference on a side stream, at a controlled average rate:
      d2d  -- device-to-device copies (copy engine; each byte is one HBM read
              plus one HBM write), spaced by torch.cuda._sleep;
      h2d  -- pinned host-to-device copies, the A100 link itself;
      sleep-- the spacing kernels alone, as the control for the spacing.
  * Rate is reported as ACHIEVED (bytes / side-stream window), and as injected
    HBM traffic (d2d counts 2x).
  * Points are interleaved and repeated; the order is shuffled per repeat.

Run: .venv/bin/python benchmarks/kv_pipelining_bench/e1_gemm_write_contention.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import time

import torch

HIDDEN, INTER, NQ, NKV, HD = 5120, 27648, 40, 8, 128   # Qwen2.5-32B
SHAPES = [(HIDDEN, (NQ + 2 * NKV) * HD),   # qkv
          (NQ * HD, HIDDEN),               # o
          (HIDDEN, 2 * INTER),             # gate_up
          (INTER, HIDDEN)]                 # down


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


class Bench:
    def __init__(self, a):
        self.a = a
        self.dev = torch.device("cuda")
        self.W = [[torch.randn(k, n, device=self.dev, dtype=torch.float16) * 0.01
                   for (k, n) in SHAPES] for _ in range(a.layers)]
        self.wbytes = sum(w.numel() * 2 for lw in self.W for w in lw)
        self.x = {m: torch.randn(m, HIDDEN, device=self.dev, dtype=torch.float16)
                  for m in a.rows}
        self.xi = {m: torch.randn(m, INTER, device=self.dev, dtype=torch.float16)
                   for m in a.rows}
        cb = a.chunk_mb << 20
        self.src = torch.empty(cb, dtype=torch.uint8, device=self.dev)
        self.dst = torch.empty(cb, dtype=torch.uint8, device=self.dev)
        self.host = torch.empty(cb, dtype=torch.uint8, pin_memory=True)
        self.side = torch.cuda.Stream()
        self.chunk = cb
        self.cyc_per_us = self._calibrate_sleep()
        self.copy_us = {k: self._time_copy(k) for k in ("d2d", "h2d")}

    # ------------------------------------------------------------ primitives
    def _calibrate_sleep(self) -> float:
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda._sleep(1000)
        s.record(); torch.cuda._sleep(1_000_000); e.record()
        torch.cuda.synchronize()
        return 1_000_000 / (s.elapsed_time(e) * 1000.0)

    def _copy(self, kind):
        if kind == "d2d":
            self.dst.copy_(self.src, non_blocking=True)
        else:
            self.dst.copy_(self.host, non_blocking=True)

    def _time_copy(self, kind) -> float:
        for _ in range(5):
            self._copy(kind)
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(20):
            self._copy(kind)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000.0 / 20

    def _gemm_pass(self, m):
        x, xi = self.x[m], self.xi[m]
        for lw in self.W:
            torch.matmul(x, lw[0]); torch.matmul(x, lw[1])
            torch.matmul(x, lw[2]); torch.matmul(xi, lw[3])

    # ------------------------------------------------------------ one point
    def point(self, m, kind, rate_gbs, pass_us):
        """GEMM pass time with `kind` traffic at ~rate_gbs on the side stream."""
        a = self.a
        n_side = 0
        if kind != "none":
            if kind == "sleep":
                period_us = self.chunk / (rate_gbs * 1e3)   # same cadence as d2d
                n_per_pass = max(1, int(pass_us / period_us))
                sleep_us = period_us
            else:
                period_us = self.chunk / (rate_gbs * 1e3) if rate_gbs > 0 else 0
                n_per_pass = max(1, int(pass_us / period_us)) if period_us else 0
                sleep_us = max(0.0, period_us - self.copy_us[kind])
        g_s = [torch.cuda.Event(True) for _ in range(a.passes)]
        g_e = [torch.cuda.Event(True) for _ in range(a.passes)]
        side_s, side_e = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        if kind != "none":
            with torch.cuda.stream(self.side):
                side_s.record()
        for i in range(a.passes + a.warm):
            if kind != "none":
                # Enqueue this pass's share of side traffic just before it, so
                # the host never runs a full launch queue ahead of the GEMMs.
                with torch.cuda.stream(self.side):
                    for _ in range(n_per_pass):
                        if kind != "sleep":
                            self._copy(kind); n_side += 1
                        if sleep_us > 0:
                            torch.cuda._sleep(int(sleep_us * self.cyc_per_us))
            if i >= a.warm:
                g_s[i - a.warm].record()
            self._gemm_pass(m)
            if i >= a.warm:
                g_e[i - a.warm].record()
        if kind != "none":
            with torch.cuda.stream(self.side):
                side_e.record()
        torch.cuda.synchronize()
        t = [s.elapsed_time(e) * 1000.0 for s, e in zip(g_s, g_e)]
        out = {"rows": m, "kind": kind, "target_gbs": rate_gbs,
               "pass_us_median": statistics.median(t),
               "pass_us_p10": sorted(t)[len(t) // 10],
               "pass_us_p90": sorted(t)[9 * len(t) // 10]}
        if kind in ("d2d", "h2d"):
            win = side_s.elapsed_time(side_e) / 1e3
            ach = n_side * self.chunk / win / 1e9
            # Did the side traffic cover the GEMM window? Compare spans.
            gwin = g_s[0].elapsed_time(g_e[-1]) / 1e3
            out.update(achieved_gbs=ach,
                       hbm_traffic_gbs=ach * (2 if kind == "d2d" else 1),
                       hbm_write_gbs=ach,
                       side_window_s=win, gemm_window_s=gwin)
        return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="4,16")
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--passes", type=int, default=60)
    p.add_argument("--warm", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--chunk-mb", type=int, default=8)
    p.add_argument("--d2d-rates", default="12.5,25,50,100,150,200,300")
    p.add_argument("--h2d-rates", default="12.5,25,40")
    p.add_argument("--out", default="e1_gemm_write_contention.json")
    a = p.parse_args()
    a.rows = [int(x) for x in a.rows.split(",")]
    d2d = [float(x) for x in a.d2d_rates.split(",")]
    h2d = [float(x) for x in a.h2d_rates.split(",")]

    torch.backends.cuda.matmul.allow_tf32 = False
    b = Bench(a)
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}  weights/pass {b.wbytes / 1e9:.2f} GB  "
          f"sleep {b.cyc_per_us:.0f} cyc/us  copy {a.chunk_mb} MB: "
          f"d2d {b.copy_us['d2d']:.1f} us ({b.chunk / b.copy_us['d2d'] / 1e3:.0f} GB/s), "
          f"h2d {b.copy_us['h2d']:.1f} us ({b.chunk / b.copy_us['h2d'] / 1e3:.1f} GB/s)")

    base_us = {m: b.point(m, "none", 0, 0)["pass_us_median"] for m in a.rows}
    for m in a.rows:
        print(f"  M={m}: pass {base_us[m] / 1e3:.2f} ms  "
              f"(weights at {b.wbytes / base_us[m] / 1e3:.0f} GB/s)")

    plan = []
    for m in a.rows:
        plan.append((m, "none", 0.0))
        plan += [(m, "d2d", r) for r in d2d]
        plan += [(m, "h2d", r) for r in h2d]
        plan += [(m, "sleep", r) for r in (25.0, 100.0, 300.0)]
    res = []
    for rep in range(a.repeats):
        order = plan[:]
        random.Random(rep).shuffle(order)
        for m, kind, rate in order:
            r = b.point(m, kind, rate, base_us[m])
            r["rep"] = rep
            res.append(r)

    # Summary: slowdown vs the same-repeat-agnostic median of the 'none' points.
    print("\n rows kind   target  achieved  HBM traffic  pass ms   slowdown (median of reps)")
    summ = []
    for m in a.rows:
        none = statistics.median([r["pass_us_median"] for r in res
                                  if r["rows"] == m and r["kind"] == "none"])
        keys = sorted({(r["kind"], r["target_gbs"]) for r in res if r["rows"] == m},
                      key=lambda k: (k[0], k[1]))
        for kind, rate in keys:
            rr = [r for r in res if r["rows"] == m and r["kind"] == kind
                  and r["target_gbs"] == rate]
            t = statistics.median([r["pass_us_median"] for r in rr])
            sd = [r["pass_us_median"] / none - 1 for r in rr]
            ach = statistics.median([r.get("achieved_gbs", 0.0) for r in rr])
            tr = statistics.median([r.get("hbm_traffic_gbs", 0.0) for r in rr])
            summ.append(dict(rows=m, kind=kind, target=rate, achieved_gbs=ach,
                             hbm_traffic_gbs=tr, pass_ms=t / 1e3,
                             slowdown=t / none - 1, slowdown_reps=sd))
            print(f" {m:4d} {kind:5s} {rate:7.1f} {ach:9.1f} {tr:11.1f}  {t / 1e3:7.3f}  "
                  f"{(t / none - 1) * 100:+6.2f}%  reps {['%+.2f' % (x * 100) for x in sd]}")

    out = {"device": props.name, "git_head": git_head(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
           "hostname": os.uname().nodename, "args": vars(a),
           "weights_per_pass_gb": b.wbytes / 1e9, "copy_us": b.copy_us,
           "sleep_cyc_per_us": b.cyc_per_us, "summary": summ, "raw": res,
           "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
