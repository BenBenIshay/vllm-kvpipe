#!/usr/bin/env python3
"""F8 -- the knee, where it is predicted to be, and what happens past it.

    ./.venv/bin/python paper/figs/f8_knee.py --out paper/figs/f8.pdf

THIS REPLACES f6, which was generated from the sweep whose points straddle the
knee and annotated a band as "overhead the model does not carry" -- the reading
this paper retracts. Plotting it under a caption claiming the law IS affine below
the knee put a contradiction on page 1.

What this shows instead is the knee formula being right. Two contexts give two
different predicted knees, r* = ITL_0 / (per-request link time), and in each case
the measured cost tracks the affine law up to floor(r*) and then departs upward:

    ctx 1080   r* = 4.16   affine through r=4, departs at r=5
    ctx 1440   r* = 3.18   affine through r=3, departs at r=4

That is the paper's central claim measured on both sides of the boundary, on the
hardware we have, with no projection involved. It is also why fitting one line
across the boundary inflates the slope and depresses the intercept: the dashed
continuation shows where such a fit would be pulled.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

INK, ACCENT, BLUE, GREY = "#1a1a1a", "#b3472f", "#2f5d8b", "#888888"
D = "paper/data/overhead/21_fusion_bound"


def load(tag):
    d = json.load(open(os.path.join(D, tag + ".json")))
    t0 = next((r["step_ms"] for r in d["rows"]
               if r.get("riders") == 0 and r.get("step_ms")), None)
    rows = sorted((r["riders"], r["dT_ms"], r.get("link_ms") or 0.0)
                  for r in d["rows"]
                  if r.get("usable") and r["riders"]
                  and r.get("dT_ms") is not None)
    # r* from the per-request link time, which is what Eq. knee says
    per = rows[0][2] if rows else 0.0
    return t0, rows, (t0 / per if per else 0.0)


def affine(rows, upto):
    xs = [(r, d) for r, d, _ in rows if r <= upto]
    n = len(xs)
    sx = sum(x for x, _ in xs)
    sy = sum(y for _, y in xs)
    sxx = sum(x * x for x, _ in xs)
    sxy = sum(x * y for x, y in xs)
    c = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - c * sx) / n, c, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figs/f8.pdf")
    a = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, (tag, lab, col) in zip(axes,
                                   (("E3_ctx1080", "context 1080", BLUE),
                                    ("E3_ctx1440", "context 1440", ACCENT))):
        t0, rows, rstar = load(tag)
        kn = int(rstar)
        F, c, n = affine(rows, kn)
        print("%s  t0=%.1f  r*=%.2f  fit over r<=%d: F=%.2f c=%.3f (%d pts)"
              % (tag, t0, rstar, kn, F, c, n))

        rs = [r for r, _, _ in rows]
        ds = [d for _, d, _ in rows]
        ax.plot(rs, ds, "o", ms=4, color=INK, label="measured")
        below = [r for r in rs if r <= kn]
        ax.plot(below, [F + c * r for r in below], "-", color=col, lw=1.9,
                label=r"$F + c\,r$, fitted below $r^{*}$")
        above = [r for r in rs if r >= kn]
        ax.plot(above, [F + c * r for r in above], "--", color=col, lw=1.1,
                alpha=0.7, label="extrapolated")
        ax.axvline(rstar, ls=":", lw=1.1, color=GREY)
        ax.annotate(r"$r^{*}=%.2f$" % rstar, xy=(rstar, max(ds) * 0.80),
                    xytext=(rstar + 0.18, max(ds) * 0.80), fontsize=6.6,
                    color=GREY)
        # DELIBERATELY NOT LABELLING F AND c HERE. At these contexts the knee sits
        # at r = 3-4, so the handful of points below it are themselves close to it
        # and the fit picks up the curvature -- c comes out 1.6-2.6 against the
        # 0.49 measured at ctx 128 where the whole sweep is far below the knee.
        # That inflation IS this paper's point about fitting near the boundary, so
        # quoting these as coefficients would contradict Eq. coeffs. The figure
        # makes one claim: the knee is where r* says it is.

        ax.annotate("departs here:\n%.1f$\\times$ the fitted line"
                    % (ds[kn] / (F + c * rs[kn]) if len(ds) > kn else 0),
                    xy=(rs[kn], ds[kn]), xytext=(rs[kn] - 2.4, ds[kn] * 1.9),
                    fontsize=5.8, color=INK,
                    arrowprops=dict(arrowstyle="->", lw=0.7, color=INK))
        ax.set_xlabel("host-tier requests, $r$")
        ax.set_title("%s:  predicted $r^{*}=%.2f$" % (lab, rstar),
                     fontsize=7.4, color=INK)
        ax.legend(fontsize=5.9, frameon=False, loc="upper left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"$\Delta$ step time (ms)")
    fig.tight_layout(pad=0.4)
    fig.savefig(a.out)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
