#!/usr/bin/env python3
"""F7 -- the two coefficients against layer depth, with the measured drift.

    ./.venv/bin/python paper/figs/f7_coefficients.py \
        --data paper/data/overhead --out paper/figs/f7.pdf

WHY THIS FIGURE EXISTS. Section 6 had no plots at all, which a reviewer reads as
an evaluation with no results. The claim it has to support is Eq. coeffs: that both
coefficients are affine in layer depth, that `c` is small and context-independent,
and that `F` is the term that decides the margin. The depth axis is the one we can
vary INSIDE one model at fixed head geometry (the layer-limit measurement knob), so
it is the only clean slope in the campaign.

Left panel: c(L) and F(L) from the below-the-knee fits, with F's error bar set by
its MEASURED between-job drift (+-0.5 ms), not by the fit residual -- that drift is
bimodal (0.04/0.43/0.04/0.51 across four jobs) and is the honest uncertainty.
Right panel: the same two coefficients as a share of the step, which is what
decides whether the mechanism pays.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

INK = "#1a1a1a"
ACCENT = "#b3472f"
BLUE = "#2f5d８b".replace("８", "8")


def fit_below_knee(path):
    """dT = F + c*r over rows strictly below the knee. Returns (F, c, n)."""
    d = json.load(open(path))
    t0 = next((r["step_ms"] for r in d["rows"]
               if r.get("riders") == 0 and r.get("step_ms")), None)
    xs = [(r["riders"], r["dT_ms"]) for r in d["rows"]
          if r.get("usable") and r["riders"] and r.get("dT_ms") is not None
          and (not r.get("link_ms") or not t0 or r["link_ms"] / t0 <= 1 / 3.0)]
    n = len(xs)
    if n < 2:
        return None
    sx = sum(x for x, _ in xs)
    sy = sum(y for _, y in xs)
    sxx = sum(x * x for x, _ in xs)
    sxy = sum(x * y for x, y in xs)
    c = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return ((sy - c * sx) / n, c, n, t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="paper/data/overhead")
    ap.add_argument("--out", default="paper/figs/f7.pdf")
    a = ap.parse_args()

    # The depth sweep: CF_L16 / CF_L32 are the layer-limited arms, CF_c128_* the
    # full 64-layer ones. All from 15_clean_cf, all one job, all ctx 128.
    want = [(16, "15_clean_cf/CF_L16.json"),
            (32, "15_clean_cf/CF_L32.json"),
            (64, "15_clean_cf/CF_c128_a.json")]
    pts = []
    for L, rel in want:
        p = os.path.join(a.data, rel)
        if not os.path.exists(p):
            print("  missing %s" % p)
            continue
        r = fit_below_knee(p)
        if r:
            pts.append((L, r[0], r[1], r[3]))
            print("  L=%-3d F=%6.2f  c=%6.3f  (t0 %.1f)" % (L, r[0], r[1], r[3]))
    if len(pts) < 2:
        raise SystemExit("not enough depth points to plot")

    # The 64-layer F is also bracketed by three arms in 21_fusion_bound; use their
    # spread as the plotted uncertainty where we have it.
    base = sorted(glob.glob(os.path.join(a.data, "21_fusion_bound/FB_base_*.json")))
    f64 = [fit_below_knee(p)[0] for p in base if fit_below_knee(p)]
    drift64 = (max(f64) - min(f64)) if len(f64) > 1 else None
    if drift64 is not None:
        print("  F at L=64 bracketed by %d arms, spread %.2f ms" % (len(f64), drift64))

    Ls = [p[0] for p in pts]
    Fs = [p[1] for p in pts]
    cs = [p[2] for p in pts]
    t0s = [p[3] for p in pts]

    # THE DEPTH SWEEP PREDATES T7a AND T8. Those two changes cut F at L=64 from
    # 6.05 to 3.36 and did not touch c. Plotting the sweep's F beside the final F
    # without saying so would put a third value of F into circulation, which is
    # exactly the confusion a reviewer flagged. So the figure shows BOTH lines and
    # labels which is which; the post-optimisation line is anchored on the one
    # depth where we measured it, scaled per layer.
    F64_FINAL = None
    if f64:
        F64_FINAL = sum(f64) / len(f64)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.5))

    # ---- left: absolute coefficients vs depth -----------------------------
    ax.errorbar(Ls, Fs, yerr=0.5, fmt="o-", color="#999", lw=1.2, ms=4,
                capsize=3, label=r"$F$, before the host-path fixes")
    if F64_FINAL:
        per = F64_FINAL / 64.0
        ax.errorbar(Ls, [per * L for L in Ls],
                    yerr=drift64 if drift64 else 0.1,
                    fmt="o-", color=ACCENT, lw=1.6, ms=4, capsize=3,
                    label=r"$F$, after (%.3f ms/layer)" % per)
    ax.plot(Ls, cs, "s-", color=BLUE, lw=1.6, ms=4,
            label=r"$c$, per host-tier request")
    ax.set_xlabel("layers streamed, $L$")
    ax.set_ylabel("ms")
    ax.set_xticks(Ls)
    ax.set_ylim(0, max(Fs) * 1.35)
    # c IS affine in depth and F IS NOT, and the earlier title claimed both were.
    # Measured on these arms F/L runs 0.058, 0.071, 0.095 at L = 16/32/64 -- rising
    # 64% -- so the flat 0.053 ms/layer of Eq. coeffs, obtained by dividing one
    # depth's measurement by its depth, is an assumption with this evidence against
    # it. Section 6's sensitivity table carries the consequence.
    ax.plot(Ls, [0.053 * L for L in Ls], ":", color="#555", lw=1.1,
            label=r"$F$, assumed $0.053L$")
    ax.legend(fontsize=5.9, frameon=False, loc="upper left")
    ax.set_title(r"$c$ is affine in depth; $F$ rises faster", fontsize=7.5,
                 color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- right: as a share of the step ------------------------------------
    if F64_FINAL:
        per = F64_FINAL / 64.0
        bx.plot(Ls, [per * L / t * 100 for L, t in zip(Ls, t0s)], "o-",
                color=ACCENT, lw=1.6, ms=4, label=r"$F/\mathrm{ITL}_0$")
    bx.plot(Ls, [c / t * 100 for c, t in zip(cs, t0s)], "s-", color=BLUE,
            lw=1.6, ms=4, label=r"$c/\mathrm{ITL}_0$, one request")
    bx.axhline(100.0 / 16, ls=":", lw=0.9, color="#888")
    bx.annotate("one resident's share of the step ($1/B$)",
                xy=(Ls[0], 100.0 / 16), xytext=(Ls[0] + 1, 100.0 / 16 + 0.6),
                fontsize=5.6, color="#666")
    bx.set_xlabel("layers streamed, $L$")
    bx.set_ylabel("\\% of the rider-free step")
    bx.set_xticks(Ls)
    bx.legend(fontsize=6.2, frameon=False, loc="upper left")
    bx.set_title(r"$c$ stays far under what a resident is worth",
                 fontsize=7.5, color=INK)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)

    fig.tight_layout(pad=0.4)
    fig.savefig(a.out)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
