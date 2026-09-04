#!/usr/bin/env python3
"""Is the host path's fixed cost paid by the RIDERS or by the whole BATCH?

Measured (rider_hide_efficiency.py, Qwen-32B, ctx 2048):

    ITL = 49.91 ms baseline + 29.59 ms fixed + 5.42 ms per rider

~75% of a rider's transfer hides, so the interconnect is not the binding cost.
The ~30 ms fixed term is, and it is the one thing a faster link cannot help:
it moves break-even from the first rider to ~17 on PCIe and ~4-8 on C2C.

Its shape says a lot. It switches on with rider #1 and barely grows after, so
it is not rider-shaped. `_maybe_hybrid_two_pass` (flash_attn.py) suggests why:
once ANY rider exists a per-step split is published, and from then on EVERY
layer index_selects the query rows, seqused and block table for BOTH tiers,
allocates a per-tier output, runs a flash call each, and scatters back. The
resident cohort stops being one fused flash call and becomes a gathered
sub-batch -- so the residents pay for the riders' existence.

This separates the two explanations by holding riders at ONE and varying the
number of residents:

  * fixed cost GROWS with resident count  -> the batch-wide gather/scatter is
    the cost, and the fix is to leave the resident rows on their fused path;
  * fixed cost is FLAT                    -> it is per-layer overhead
    independent of batch composition (stream ping-pong, build_step), and the
    fix is elsewhere.

Run:
  srun ... .venv/bin/python benchmarks/kv_pipelining_bench/fixed_cost_locator.py
"""

from __future__ import annotations

import argparse
import json
import subprocess

ROOT = "/mnt/beegfs/benishay/vllm-kvpipe"
PY = f"{ROOT}/.venv/bin/python"
BENCH = f"{ROOT}/benchmarks/kv_pipelining_bench/pd_decode_goodput.py"
LOGDIR = "/mnt/beegfs/benishay/paper_runs/20260904"


def run(args, residents: int, riders: int) -> dict:
    """`residents` requests on HBM plus `riders` on the host tier."""
    arm = "no_riders" if riders == 0 else "riders"
    offered = residents + riders
    out = f"{LOGDIR}/fixedcost_r{residents}_k{riders}.json"
    cmd = [PY, BENCH, "--model", args.model, "--ctx", str(args.ctx),
           "--out-len", str(args.out_len), "--offered", str(offered),
           "--b-hbm-max", str(residents), "--arms", arm,
           "--itl-fallback-ms", "1000",
           "--gpu-mem-util", str(args.gpu_mem_util), "--out", out]
    if riders:
        cmd += ["--max-offload", str(riders)]
    print(f"--- residents={residents} riders={riders} ---", flush=True)
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
    ap.add_argument("--residents", default="2,4,8")
    ap.add_argument("--gpu-mem-util", type=float, default=0.83)
    args = ap.parse_args()

    rows = []
    for R in [int(x) for x in args.residents.split(",")]:
        base = run(args, R, 0)
        trt = run(args, R, 1)
        rows.append({"residents": R, "base": base, "one_rider": trt})

    with open(f"{LOGDIR}/fixed_cost_locator.json", "w") as f:
        json.dump(rows, f, indent=2)

    print("=" * 74)
    print(f"{'residents':>9} {'ITL base':>9} {'ITL +1 rider':>13} "
          f"{'fixed cost':>11} {'riders seen':>12}")
    pts: list[tuple[int, float]] = []
    for r in rows:
        b, t = r["base"], r["one_rider"]
        for row, what in ((b, "base"), (t, "one_rider")):
            if row.get("error") or row.get("engine_died") or row.get("errors"):
                print(f"{r['residents']:>9}   INVALID ({what}: "
                      f"{row.get('error') or row.get('engine_died') or str(row.get('errors')) + ' failed'})")
                break
        else:
            if t.get("delivered_real", 0) < 1:
                print(f"{r['residents']:>9}   INVALID (no rider was actually "
                      "pool-served, so this is two copies of the control)")
                continue
            ib, it = b.get("itl_median_ms"), t.get("itl_median_ms")
            if ib is None or it is None:
                continue
            print(f"{r['residents']:>9} {ib:>9} {it:>13} {it - ib:>11.2f} "
                  f"{t.get('delivered_real', 0):>12}")
            pts.append((r["residents"], it - ib))
    print("=" * 74)

    if len(pts) < 2:
        print("INCONCLUSIVE: need at least two resident counts.")
        return 2
    lo, hi = pts[0], pts[-1]
    growth = hi[1] - lo[1]
    per_resident = growth / (hi[0] - lo[0]) if hi[0] != lo[0] else 0.0
    print(f"fixed cost went {lo[1]:.2f} ms at {lo[0]} residents -> "
          f"{hi[1]:.2f} ms at {hi[0]} ({per_resident:+.2f} ms per resident)")
    if growth > 0.4 * lo[1]:
        print("IT SCALES WITH THE RESIDENTS. The riders are not paying this -- "
              "the resident cohort is, because publishing a split re-routes "
              "the WHOLE batch through per-tier gather/scatter instead of one "
              "fused flash call. Fix: keep the fused pass over all rows and "
              "overwrite only the rider rows from a second small pass.")
    else:
        print("It is FLAT in the resident count, so it is per-step/per-layer "
              "overhead, not the batch-wide gather. Look at the stream "
              "ping-pong in _launch_prefetch (one layer of lookahead despite "
              "k_window slots) and at build_step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
