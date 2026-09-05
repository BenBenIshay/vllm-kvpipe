#!/usr/bin/env python3
"""F5 -- riders are additive: served concurrency against a fixed KV pool.

The claim is that a host-tier request is served IN ADDITION TO a full resident
cohort rather than in place of part of it. That is the whole point of zero-block
admission, and the b_hbm_max sweep that preceded it is the falsification: with
riders drawing on the same headroom as residents, the served total crept toward
the all-HBM capacity from below and never passed it.

TWO CAVEATS THAT MUST TRAVEL WITH THIS FIGURE, in the caption, every time:

  1. Riders carry SYNTHETICALLY filled KV, because the prefill-node transport
     is emulated. Generated tokens are therefore meaningless. This is a
     CAPACITY result and must never be quoted in the same breath as a tok/s
     number -- goodput with real KV is a separate experiment.
  2. The mixture figures are LOWER BOUNDS. The mixture held the entire offered
     load, so its own ceiling was never reached; the bar is clipped by the
     workload, not by the system.

  python paper/figs/f5_capacity.py \
      --data paper/data/preload_capacity.json --out paper/figs/f5.pdf
"""

from __future__ import annotations

import argparse
import json
import sys


def load(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    rows = {}
    for r in d["rows"]:
        if r.get("error") or r.get("engine_died"):
            sys.exit(f"FATAL: arm {r.get('arm')} failed: "
                     f"{r.get('error') or 'engine died'}. Re-run the freeze.")
        rows[r["arm"]] = r
    missing = {"all_hbm", "mixture"} - set(rows)
    if missing:
        sys.exit(f"FATAL: missing arm(s) {missing}; the comparison is the point.")
    return {"config": d["config"], "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="paper/figs/f5.pdf")
    args = ap.parse_args()

    d = load(args.data)
    cfg, rows = d["config"], d["rows"]
    offered = cfg["offered"]

    hbm = rows["all_hbm"]["peak_running"]
    mix = rows["mixture"]["peak_running"]

    print(f"ctx {cfg['ctx']}  offered {offered}  b_hbm_max {cfg['b_hbm_max']}")
    print(f"  all_hbm peak running   {hbm}")
    print(f"  mixture peak running   {mix}")
    print(f"  ratio                  {mix/hbm:.2f}x" if hbm else "  ratio  n/a")

    # A guard against the trap that produced plausible EQUAL numbers twice: if
    # b_hbm_max is at or above the all-HBM ceiling, admission tags every
    # request resident, no rider is ever created, and the arms are identical BY
    # CONSTRUCTION rather than by measurement.
    if mix == hbm:
        print("\nWARNING: arms are identical. Check that riders were actually "
              "created (b_hbm_max below the load's natural concurrency), "
              "otherwise this is the inert-mixture trap, not a result.")
    if mix >= offered:
        print(f"\nNOTE: mixture hit the offered load ({offered}); this is a "
              f"LOWER BOUND, not the mixture's ceiling. Say so in the caption.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib unavailable; numbers above are the deliverable.")
        return 0

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    bars = ax.bar(["all-HBM", "mixture"], [hbm, mix],
                  color=["#a0aec0", "#2b6cb0"], width=0.55)
    ax.axhline(offered, color="#e53e3e", lw=1, ls=":",
               label=f"offered load ({offered})")
    for b, v in zip(bars, [hbm, mix]):
        ax.text(b.get_x() + b.get_width() / 2, v + offered * 0.02, str(v),
                ha="center", fontsize=9)
    ax.set_ylabel("peak concurrent requests")
    ax.set_ylim(0, offered * 1.18)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
