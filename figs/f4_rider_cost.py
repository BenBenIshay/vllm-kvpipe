#!/usr/bin/env python3
"""F4 -- what a rider costs, decomposed.

The load-bearing evaluation figure. It separates the FIXED per-step cost of
using the host path at all from the PER-RIDER cost that fails to overlap,
because those two extrapolate to completely different answers on a faster
interconnect and a single rider count cannot tell them apart.

F8 (the crossover) is this figure plus one C2C point, so the axes and the fit
are deliberately written to be reused there.

Reads ONLY the frozen dataset. Never scrape a log: numbers in this repo span
commits from before the b_tok fix (which charged every rider half its real
transfer) and before the hide-budget became measured rather than declared, and
a figure that silently mixes them is unfixable after the fact.

  python paper/figs/f4_rider_cost.py \
      --data paper/data/hide_efficiency.json --out paper/figs/f4.pdf
"""

from __future__ import annotations

import argparse
import json
import sys


def load(path: str) -> tuple[list[int], list[float], list[float]]:
    """Return (riders, itl_ms, tok_s) sorted by rider count.

    Refuses a dataset with a failed arm. A failed arm still writes its JSON --
    with an "error" key in the row -- and a freeze that skipped over one would
    otherwise be plotted as if it were real.
    """
    with open(path) as f:
        raw = json.load(f)

    riders, itl, toks = [], [], []
    for entry in raw:
        row = entry["row"]
        if row.get("error") or row.get("engine_died"):
            sys.exit(f"FATAL: arm {entry['arm']} cap={entry['cap']} failed: "
                     f"{row.get('error') or 'engine died'}. "
                     f"Re-run the freeze; do not plot this.")
        riders.append(0 if entry["arm"] == "no_riders" else int(entry["cap"]))
        itl.append(float(row["itl_median_ms"]))
        toks.append(float(row["decode_tok_s"]))

    order = sorted(range(len(riders)), key=lambda i: riders[i])
    return ([riders[i] for i in order],
            [itl[i] for i in order],
            [toks[i] for i in order])


def fit(riders: list[int], itl: list[float]) -> tuple[float, float, float]:
    """ITL(r) = baseline + fixed + slope*r, fitted over r >= 1 only.

    The fixed term is the STEP between r=0 and the fit's intercept, so it must
    not be fitted through the r=0 point -- doing so folds the fixed cost into
    the slope and makes a faster interconnect look more valuable than it is.
    """
    baseline = itl[riders.index(0)]
    xs = [r for r in riders if r >= 1]
    ys = [itl[riders.index(r)] for r in xs]
    if len(xs) < 2:
        sys.exit("FATAL: need at least two rider points to separate "
                 "fixed cost from slope.")
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope * mx
    return baseline, intercept - baseline, slope


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="paper/figs/f4.pdf")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--link-gbps", type=float, default=25.1,
                    help="MEASURED, not nominal. On C2C use the per-direction "
                         "figure (450), never the 900 GB/s bidirectional one.")
    ap.add_argument("--rider-gb", type=float, default=0.537,
                    help="KV bytes streamed per rider per step at --ctx")
    args = ap.parse_args()

    riders, itl, toks = load(args.data)
    baseline, fixed, slope = fit(riders, itl)

    unoverlapped = args.rider_gb / args.link_gbps * 1000.0
    hidden = 1.0 - slope / unoverlapped

    print(f"baseline (0 riders)   {baseline:8.2f} ms")
    print(f"fixed cost            {fixed:+8.2f} ms   (paid once per step)")
    print(f"per-rider             {slope:8.2f} ms")
    print(f"  a rider is {args.rider_gb:.3f} GB/step; at {args.link_gbps} GB/s "
          f"that is {unoverlapped:.2f} ms unoverlapped")
    print(f"fraction HIDDEN       {hidden*100:7.0f}%")

    # The break-even that goes in the caption is the MEASURED crossing of the
    # tok/s column, not a concurrency/ITL model. That model ignores queueing --
    # the rider-free arm leaves requests waiting and riders shorten the queue --
    # and it predicted ~7 riders where the measurement crosses at 3-4.
    base_tok = toks[riders.index(0)]
    crossing = next((r for r, t in zip(riders, toks) if r >= 1 and t >= base_tok),
                    None)
    print(f"tok/s rider-free      {base_tok:8.1f}")
    print(f"MEASURED break-even   {crossing if crossing else 'not reached in this sweep'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib unavailable; numbers above are the deliverable.")
        return 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax1.plot(riders, itl, "o-", color="#2b6cb0", zorder=3)
    xs = [r for r in riders if r >= 1]
    ax1.plot(xs, [baseline + fixed + slope * r for r in xs], "--",
             color="#a0aec0", zorder=2,
             label=f"{fixed:+.1f} ms fixed $+$ {slope:.2f} ms/rider")
    ax1.axhline(baseline, color="#e53e3e", lw=1, ls=":", zorder=1,
                label=f"rider-free {baseline:.1f} ms")
    ax1.set_xlabel("riders"); ax1.set_ylabel("median ITL (ms)")
    ax1.legend(fontsize=7, frameon=False); ax1.set_title("cost", fontsize=9)

    ax2.plot(riders, toks, "o-", color="#2b6cb0", zorder=3)
    ax2.axhline(base_tok, color="#e53e3e", lw=1, ls=":", zorder=1,
                label=f"rider-free {base_tok:.0f} tok/s")
    ax2.set_xlabel("riders"); ax2.set_ylabel("decode tok/s")
    ax2.legend(fontsize=7, frameon=False); ax2.set_title("recovery", fontsize=9)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
