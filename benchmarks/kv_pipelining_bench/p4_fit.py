#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fit the P4 measurements against the decode-step ITL model.

  model:  ITL(B,S) = (W + B*S*k) / (BW*eff)
          => at fixed S, ITL is linear in B with
             intercept a = W/(BW*eff_w)      slope m = S*k/(BW*eff_kv)

Reports, per context:
  - least-squares fit of ITL vs B, R^2 (prediction A: is it linear?)
  - achieved bandwidth implied by the intercept (weights) and by the
    slope (KV) -- these are the two numbers the whole plan rests on
  - the predicted full-pool ITL = pool_bytes_total / (BW*eff)   (prediction C)
"""

from __future__ import annotations

import json
import sys


def lstsq(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    m = sxy / sxx if sxx else float("nan")
    a = my - m * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + m * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return a, m, r2


def main(paths):
    for path in paths:
        d = json.load(open(path))
        meta, res = d["meta"], d["results"]
        W = meta["weight_bytes"]
        k = meta["kv_bytes_per_token"]
        BW = meta["peak_bw_gbs"] * 1e9
        print(f"\n{'='*78}\n{meta['model']}   "
              f"W={W/1e9:.1f} GB   kv={k/1024:.0f} KiB/tok   "
              f"peak BW={BW/1e9:.0f} GB/s\n{'='*78}")

        by_ctx = {}
        for r in res:
            by_ctx.setdefault(r["ctx"], []).append(r)

        for ctx, rows in sorted(by_ctx.items()):
            rows.sort(key=lambda r: r["batch"])
            if len(rows) < 3:
                print(f"\n S={ctx}: only {len(rows)} points, skipping fit")
                continue
            xs = [r["batch"] for r in rows]
            ys = [r["itl_ms"] for r in rows]
            a, m, r2 = lstsq(xs, ys)

            # intercept -> weight-read bandwidth; slope -> KV-read bandwidth
            eff_w = (W / (a / 1e3)) / BW if a > 0 else float("nan")
            eff_kv = ((ctx * k) / (m / 1e3)) / BW if m > 0 else float("nan")
            pred_a = W / (BW * 0.75) * 1e3
            pred_m = ctx * k / (BW * 0.75) * 1e3

            print(f"\n S={ctx}   ITL = {a:.2f} ms + {m:.3f} ms x B      "
                  f"R^2 = {r2:.4f}")
            print(f"   intercept {a:7.2f} ms   (model @eff=.75: {pred_a:6.2f})"
                  f"   -> weight-read achieved BW = {eff_w*BW/1e9:6.0f} GB/s"
                  f"  (eff {eff_w:.2f})")
            print(f"   slope     {m:7.3f} ms/req (model @eff=.75: {pred_m:6.3f})"
                  f"   -> KV-read achieved BW     = {eff_kv*BW/1e9:6.0f} GB/s"
                  f"  (eff {eff_kv:.2f})")
            print(f"   {'B':>4} {'meas ITL':>10} {'fit':>9} {'resid':>8}"
                  f" {'KV share':>9}")
            for r in rows:
                fit = a + m * r["batch"]
                print(f"   {r['batch']:>4} {r['itl_ms']:>9.2f}m {fit:>8.2f}m"
                      f" {r['itl_ms']-fit:>+7.2f}m {100*r['kv_fraction']:>8.1f}%")

            if meta.get("pool_bytes"):
                full = (W + meta["pool_bytes"]) / (BW * eff_kv) * 1e3 \
                    if eff_kv == eff_kv else float("nan")
                print(f"   PREDICTION C: full-pool ITL "
                      f"= (W + {meta['pool_bytes']/1e9:.0f} GB) / BW"
                      f" ~= {full:.1f} ms  (model-independent claim)")


if __name__ == "__main__":
    main(sys.argv[1:] or ["p4_results.json"])
