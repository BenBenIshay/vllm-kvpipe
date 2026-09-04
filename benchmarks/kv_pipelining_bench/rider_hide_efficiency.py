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
    # Time the transfer would take with NO overlap at all.
    unoverlapped_ms = bytes_per_rider / 1e9 / args.link_gbps * 1000.0
    # The APPARENT rate implied by the marginal cost. Exceeding the link rate
    # is not an error -- it is the signature of overlap, because part of the
    # transfer was paid for out of time the GPU was computing anyway.
    b_eff = bytes_per_rider / (slope / 1000.0) / 1e9 if slope > 0 else float("inf")
    hidden = max(0.0, 1.0 - slope / unoverlapped_ms) if slope > 0 else 1.0
    fixed = intercept - base

    print(f"rider-free baseline ITL          {base:.2f} ms")
    print(f"fixed cost of the host path      {fixed:+.2f} ms  (intercept - baseline)")
    print(f"per-rider cost                   {slope:.2f} ms")
    print(f"  a rider is {bytes_per_rider / 1e9:.3f} GB/step; at the "
          f"{args.link_gbps:.1f} GB/s link that is "
          f"{unoverlapped_ms:.2f} ms unoverlapped")
    print(f"apparent bandwidth               {b_eff:.2f} GB/s "
          f"({b_eff / args.link_gbps:.2f}x the link)")
    print(f"fraction of the transfer HIDDEN  {hidden * 100:.0f}%")
    print("=" * 78)
    if hidden < 0.25:
        print("The transfer barely overlaps compute. B1's roofline assumes it "
              "does, so it over-admits by construction: the budget should be "
              "the per-rider cost measured here, not the link rate.")
    else:
        print(f"{hidden * 100:.0f}% of the transfer hides -- the roofline's "
              "premise holds and the interconnect is NOT the binding cost "
              "here. The binding cost is the FIXED term: paid once per step "
              "however few riders ride, and unchanged by a faster link. A "
              "faster interconnect shrinks the per-rider term and leaves the "
              "fixed one exactly where it is, so this number, not the "
              "bandwidth, decides whether C2C is worth buying.")
    print(f"break-even rider count on this link: solving "
          f"(R+r)/(ITL0+fixed+slope*r) = R/ITL0 for r, with R={args.b_hbm_max} "
          f"residents, needs r ~ "
          f"{_breakeven(args.b_hbm_max, base, fixed, slope):.0f}")
    return 0


def _breakeven(residents: int, base_ms: float, fixed_ms: float,
               slope_ms: float) -> float:
    """Riders needed before concurrency/ITL beats the rider-free arm.

    Throughput goes as concurrency / ITL, so a rider pays only once the extra
    request it adds outweighs the ITL it costs -- including its share of the
    fixed term, which is what makes the first rider such a bad deal.
    """
    if slope_ms <= 0:
        return float("inf")
    rate0 = residents / base_ms
    # (residents + r) = rate0 * (base + fixed + slope*r)
    denom = 1.0 - rate0 * slope_ms
    if denom <= 0:
        return float("inf")
    return (rate0 * (base_ms + fixed_ms) - residents) / denom


if __name__ == "__main__":
    raise SystemExit(main())
