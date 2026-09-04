#!/usr/bin/env python3
"""How much of a rider's transfer actually hides behind compute?

The whole design rests on one premise: a host-tier request's per-step KV
transfer overlaps the forward pass, so it is nearly free until the interconnect
saturates. B1 admits riders on exactly that assumption -- it compares transfer
BYTES against a hide budget of one forward iteration.

The first honest end-to-end measurement (2026-09-04, Qwen-32B, ctx 2048,
offered 12) says the premise is wrong on PCIe:

    no_riders   peak  8   ITL 50.89 ms   57.4 tok/s
    riders      peak 10   ITL 88.91 ms   50.9 tok/s   (-11.3%)

B1 admitted exactly the 2 riders its roofline allows. Two riders at ctx 2048 is
1.074 GB per step = 42.8 ms at 25.1 GB/s, and the step got 38.0 ms slower. So
~89% of the transfer did NOT hide. The controller did its job; the model it
optimises against is optimistic.

That single point cannot separate a fixed per-step cost of using the host path
at all from a per-byte cost that fails to overlap, and the two extrapolate to
completely different answers on C2C. So sweep the rider count and fit

    ITL(r) = ITL_0 + fixed + r * (bytes_per_rider / B_effective)

  * the INTERCEPT above the rider-free baseline is the fixed cost of the host
    path -- paid once, no matter how many riders;
  * the SLOPE gives B_effective, and B_effective / B_link is the fraction of
    the transfer that does not hide.

Both are needed to project GH200 honestly, and neither needs real KV, so this
is the measurement that decides whether the trip is worth booking.

Run:
  srun ... .venv/bin/python benchmarks/kv_pipelining_bench/rider_hide_efficiency.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = "/mnt/beegfs/benishay/vllm-kvpipe"
PY = f"{ROOT}/.venv/bin/python"
BENCH = f"{ROOT}/benchmarks/kv_pipelining_bench/pd_decode_goodput.py"
LOGDIR = "/mnt/beegfs/benishay/paper_runs/20260904"


def run(args, arm: str, cap: int) -> dict:
    out = f"{LOGDIR}/hide_eff_{arm}_{cap}.json"
    cmd = [PY, BENCH, "--model", args.model, "--ctx", str(args.ctx),
           "--out-len", str(args.out_len), "--offered", str(args.offered),
           "--b-hbm-max", str(args.b_hbm_max), "--arms", arm,
           "--itl-fallback-ms", str(args.itl_fallback_ms),
           "--gpu-mem-util", str(args.gpu_mem_util), "--out", out]
    if cap:
        cmd += ["--max-offload", str(cap)]
    print(f"--- {arm} cap={cap or '-'} ---", flush=True)
    subprocess.run(cmd, check=False)
    try:
        with open(out) as f:
            return json.load(f)["rows"][0]
    except Exception as e:                                    # noqa: BLE001
        return {"error": str(e)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--out-len", type=int, default=64)
    ap.add_argument("--offered", type=int, default=12)
    ap.add_argument("--b-hbm-max", type=int, default=8)
    ap.add_argument("--caps", default="1,2,3")
    ap.add_argument("--itl-fallback-ms", type=float, default=1000.0,
                    help="deliberately huge: the CAP decides the rider count "
                         "here, not the byte budget")
    ap.add_argument("--gpu-mem-util", type=float, default=0.83)
    ap.add_argument("--b-tok", type=int, default=262144)
    ap.add_argument("--link-gbps", type=float, default=25.1)
    args = ap.parse_args()

    rows = [("no_riders", 0, run(args, "no_riders", 0))]
    for cap in [int(c) for c in args.caps.split(",")]:
        rows.append(("riders", cap, run(args, "riders", cap)))

    with open(f"{LOGDIR}/hide_efficiency.json", "w") as f:
        json.dump([{"arm": a, "cap": c, "row": r} for a, c, r in rows],
                  f, indent=2)

    print("=" * 78)
    print(f"{'arm':<10} {'cap':>4} {'riders':>7} {'peak':>5} {'ITL med':>9} "
          f"{'tok/s':>8}")
    pts: list[tuple[int, float]] = []
    base = None
    for arm, cap, r in rows:
        if r.get("error") or r.get("engine_died") or r.get("errors"):
            print(f"{arm:<10} {cap:>4}   INVALID ({r.get('error') or r.get('engine_died') or str(r.get('errors')) + ' failed reqs'})")
            continue
        riders = r.get("delivered_real", 0)
        itl = r.get("itl_median_ms")
        print(f"{arm:<10} {cap:>4} {riders:>7} {r.get('peak_running', 0):>5} "
              f"{itl:>9} {r.get('decode_tok_s', 0):>8}")
        if itl is None:
            continue
        if arm == "no_riders":
            base = itl
        else:
            pts.append((riders, itl))
    print("=" * 78)

    if base is None or len(pts) < 2:
        print("INCONCLUSIVE: need a rider-free baseline and at least two rider "
              "counts to separate the fixed cost from the per-byte cost.")
        return 2

    # Least squares on ITL = a + b*r, over the rider points only.
    n = len(pts)
    sx = sum(r for r, _ in pts)
    sy = sum(t for _, t in pts)
    sxx = sum(r * r for r, _ in pts)
    sxy = sum(r * t for r, t in pts)
    den = n * sxx - sx * sx
    if den == 0:
        print("INCONCLUSIVE: every arm admitted the same number of riders, so "
              "the slope is undefined. Widen --caps.")
        return 2
    slope = (n * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / n

    bytes_per_rider = args.ctx * args.b_tok
    b_eff = bytes_per_rider / (slope / 1000.0) / 1e9 if slope > 0 else float("inf")
    hidden = max(0.0, 1.0 - (bytes_per_rider / 1e9) / (args.link_gbps * slope / 1000.0)) if slope > 0 else 1.0
    fixed = intercept - base

    print(f"rider-free baseline ITL          {base:.2f} ms")
    print(f"fixed cost of the host path      {fixed:+.2f} ms  (intercept - baseline)")
    print(f"per-rider cost                   {slope:.2f} ms")
    print(f"  a rider is {bytes_per_rider / 1e9:.3f} GB/step; at the "
          f"{args.link_gbps:.1f} GB/s link that is "
          f"{bytes_per_rider / 1e9 / args.link_gbps * 1000:.2f} ms unoverlapped")
    print(f"effective bandwidth              {b_eff:.2f} GB/s "
          f"({b_eff / args.link_gbps * 100:.0f}% of link)")
    print(f"fraction of the transfer HIDDEN  {hidden * 100:.0f}%")
    print("=" * 78)
    if hidden < 0.25:
        print("The transfer essentially does not overlap compute. B1's roofline "
              "assumes it does, so it over-admits by construction -- the "
              "budget should be the per-rider cost measured here, not the "
              "link rate. Note this does NOT by itself sink C2C: the same "
              "bytes cost 36x less there, so even zero overlap leaves room.")
    else:
        print("A meaningful fraction of the transfer hides; the roofline is a "
              "usable model once discounted by the measured efficiency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
