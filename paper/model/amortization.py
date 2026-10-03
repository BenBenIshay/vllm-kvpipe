#!/usr/bin/env python3
"""Does c amortize across riders?  (Ben's question, 2026-10-03.)

THE ANSWER, after three rounds of getting it wrong in both directions:

  MEASURABLY, NO. Within the resolution of this measurement the per-rider cost
  is constant and the affine law dT = F + c*r is the right description. Any
  rider-sharing is below what these sweeps can resolve.

  STRUCTURALLY, THREE OF FOUR COMPONENTS COULD BE. Only the bytes on the wire
  is per-rider by physics. The 2*L*r copy launches and the r pageable D2H calls
  are batchable, which would move them OUT of c and INTO F -- a charge paid r
  times becoming one paid once. That is an engineering lever even though no
  sharing is currently measurable, and it helps in proportion to r, which the
  knee caps.

HOW I GOT IT WRONG TWICE, because the second mistake is the instructive one.

  Round 1. Claimed "c is constant in r" citing a fitted r^2 coefficient of
  -0.00027. Wrong statistic: an r^2 term is not how you test for a power law.

  Round 2. Fitted dT = F + c*r^alpha, got alpha = 0.69 at ctx 512, and reported
  that the affine law was REJECTED -- F(1,5) = 10.28, bootstrap P(alpha<1) =
  98%, residual sign pattern consistent with concavity, and the estimate
  surviving leave-one-out at 0.52-0.74. Every one of those checks passed and
  the conclusion was still wrong.

  What they all missed is that the fit is nearly UNIDENTIFIED. At the first
  rider point F is about 95% of dT, so F and c*r^alpha trade off against each
  other with almost no leverage on the exponent. The F-test, the bootstrap and
  leave-one-out all condition on the fit window and the points inside it, so
  none of them can see that the estimate moves wholesale when the window
  changes. It does:

      arm            r<=4  r<=6  r<=8  r<=12 r<=16
      ctx 512 (M2)   0.900 0.855 0.690 0.690 0.690
      ctx 256 (M2)     -   0.890 0.960 0.750 0.750
      ctx 128 (A1)   0.400 0.400 0.590 0.940 1.085
      ctx 128 (A2)   0.400 0.615 0.980 0.795 0.980

  Read the bottom two rows: same context, same hardware, two runs of the
  IDENTICAL configuration. alpha runs 0.40 to 1.085 as the window widens on one
  of them, and the two disagree by 0.39 at the same window (r<=8). The exponent
  is set by the fit window, not by the system.

  The window sweep is therefore the PRIMARY diagnostic here, ahead of any
  goodness-of-fit statistic, and `_selftest` asserts that alpha is unstable so
  that a future change cannot quietly re-report it as a finding.

WHAT THIS MEANS FOR THE PROJECTION. The affine law stands, so `c < T0/B` is the
gate again rather than a screen, and "never pays at any rider count" is a
statement about the law rather than an artefact of one. The structural half of
the answer is unaffected and is where the remaining leverage is.

    ./.venv/bin/python paper/model/amortization.py [--selftest|--m7|--window]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

DATA = os.environ.get(
    "OVERHEAD_DATA",
    "/mnt/beegfs/benishay/paper_runs/20260930/overhead/02_law")

# Components of c, and whether each is per-rider by physics or by implementation.
# This table is the part the original report was missing.
COMPONENTS = [
    ("link / bytes on the wire",
     "each rider's KV is distinct data that must cross the bus",
     "FLOOR", "nothing to batch -- stays proportional to r"),
    ("per-rider per-layer launch",
     "2*L*R separate cudaMemcpyAsync, one pair per rider per layer",
     "BATCHABLE", "one batched copy per layer -> 2*L launches, moves into F"),
    ("D2H write-back",
     "r separate pageable .to('cpu') calls, one token each",
     "BATCHABLE", "one batched copy for all riders, moves into F"),
    ("per-block host work",
     "one Python __setitem__ per rider block per step",
     "COUNT IRREDUCIBLE", "vectorising collapses cost per item, not the "
     "count -- stays proportional to r"),
]


M7DATA = os.environ.get(
    "OVERHEAD_M7DATA",
    "/mnt/beegfs/benishay/paper_runs/20260930/overhead/07_amortization")

# M7: the high-concurrency sweep. The knee goes as 1/S, so a shorter context
# widens the rider axis on the same hardware. ctx 128 is run TWICE, adjacent in
# time, because alpha is a shape parameter and its run-to-run spread is what
# decides whether a context trend in alpha is real.
M7 = [("A1_c128", 128), ("A2_c128", 128), ("B_c192", 192), ("C_c256", 256)]


def load(S, datadir=DATA, fname=None):
    """Below-knee points of one context sweep, direct estimator."""
    d = json.load(open(os.path.join(datadir, fname or ("A_c%d.json" % S))))
    rows = [r for r in d["rows"] if r.get("usable")]
    base = [r for r in rows if r["riders"] == 0]
    if not base:
        return None, None, []
    t0 = base[0]["step_ms_direct"] or base[0]["step_ms"]
    pts, lk = [], None
    for r in sorted(rows, key=lambda r: r["riders"]):
        n = r["riders"]
        if not n:
            continue
        lk = r["link_ms"] / n
        t = r["step_ms_direct"] or r["step_ms"]
        if t is not None and n < t0 / lk:
            pts.append((n, t - t0))
    return t0, lk, pts


def lsq(pts, alpha):
    """Linear least squares of dT = F + c*r^alpha at fixed alpha."""
    xs = [n ** alpha for n, _ in pts]
    ys = [d for _, d in pts]
    k = len(xs)
    mx, my = sum(xs) / k, sum(ys) / k
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    F = my - c * mx
    rss = sum((F + c * x - y) ** 2 for x, y in zip(xs, ys))
    return F, c, rss


def best_alpha(pts, lo=0.40, hi=1.60, step=0.005):
    best = None
    a = lo
    while a <= hi + 1e-9:
        r = lsq(pts, a)
        if r and (best is None or r[2] < best[3]):
            best = (a, r[0], r[1], r[2])
        a += step
    return best


def analyse(S, datadir=DATA, nboot=2000, seed=7, fname=None):
    t0, lk, pts = load(S, datadir, fname)
    n = len(pts)
    if n < 5:
        return {"ctx": S, "n": n, "enough": False}
    F1, c1, rss1 = lsq(pts, 1.0)
    a, Fa, ca, rssa = best_alpha(pts)
    dof = n - 3
    fstat = ((rss1 - rssa) / 1.0) / (rssa / dof) if dof > 0 and rssa > 0 else 0.0

    # residual sign pattern: concavity is negative at BOTH ends, positive mid
    res = [(k, d - (F1 + c1 * k)) for k, d in pts]
    ends = res[0][1] + res[-1][1]
    mid = [e for k, e in res[1:-1]]
    concave_sig = ends < 0 and (sum(mid) / len(mid)) > 0

    # is the bend just the r=1 point?
    sub = [p for p in pts if p[0] >= 2]
    F2, c2, _ = lsq(sub, 1.0)
    drop_r1_pct = 100 * (c2 / c1 - 1)

    # LEAVE-ONE-OUT ON ALPHA. The bootstrap says how much the data constrain
    # alpha; LOO says whether any SINGLE point is carrying it. These are
    # different questions and the second one changed the reading: at ctx 128 the
    # LOO range straddles 1, so the sign of the effect there is not robust,
    # while at ctx 256 and 512 it does not come near 1.
    loo = []
    for i in range(len(pts)):
        sub = pts[:i] + pts[i + 1:]
        if len(sub) < 4:
            continue
        b = best_alpha(sub)
        if b:
            loo.append((pts[i][0], b[0]))
    loo_alphas = [x for _, x in loo]
    loo_lo = min(loo_alphas) if loo_alphas else float("nan")
    loo_hi = max(loo_alphas) if loo_alphas else float("nan")

    random.seed(seed)
    boots = []
    for _ in range(nboot):
        samp = [random.choice(pts) for _ in pts]
        if len(set(x for x, _ in samp)) < 3:
            continue
        b = best_alpha(samp, 0.40, 1.60, 0.02)
        if b:
            boots.append(b[0])
    boots.sort()
    q = (lambda p: boots[int(p * (len(boots) - 1))]) if boots else (lambda p: float("nan"))
    return {
        "ctx": S, "n": n, "enough": True, "knee": t0 / lk, "pts": pts,
        "affine": (F1, c1, rss1), "power": (a, Fa, ca, rssa),
        "fstat": fstat, "rejects_affine": fstat > 5.0,
        "residuals": res, "concavity_signature": concave_sig,
        "c_change_dropping_r1_pct": drop_r1_pct,
        "loo": loo, "loo_lo": loo_lo, "loo_hi": loo_hi,
        "loo_straddles_1": bool(loo_alphas) and (loo_lo < 1.0 < loo_hi),
        "alpha_median": q(.5), "alpha_ci": (q(.05), q(.95)),
        "p_alpha_lt_1": (sum(1 for x in boots if x < 1.0) / len(boots)) if boots else float("nan"),
    }


def report(datadir=DATA):
    print("=" * 78)
    print(" (A) EMPIRICAL -- is dT concave in r?  dT = F + c*r^alpha")
    print("=" * 78)
    print()
    for S in (256, 512, 1024):
        a = analyse(S, datadir)
        if not a["enough"]:
            print(" ctx %-5d only %d below-knee points, cannot fit alpha\n"
                  % (S, a["n"]))
            continue
        F1, c1, rss1 = a["affine"]
        al, Fa, ca, rssa = a["power"]
        print(" ctx %-5d  n=%d below-knee, knee %.1f" % (S, a["n"], a["knee"]))
        print("   affine   : F=%6.2f c=%6.3f  RSS=%7.4f" % (F1, c1, rss1))
        print("   power    : alpha=%.3f F=%6.2f c=%6.3f  RSS=%7.4f"
              % (al, Fa, ca, rssa))
        print("   F(1,%d)=%5.2f -> affine %s"
              % (a["n"] - 3, a["fstat"],
                 "REJECTED" if a["rejects_affine"] else "not rejected"))
        print("   bootstrap alpha %.2f, 90%% CI [%.2f, %.2f], P(alpha<1)=%.0f%%"
              % (a["alpha_median"], a["alpha_ci"][0], a["alpha_ci"][1],
                 100 * a["p_alpha_lt_1"]))
        print("   affine residuals: %s"
              % "  ".join("r%d:%+.2f" % (k, e) for k, e in a["residuals"]))
        print("   concavity signature (neg both ends, pos middle): %s"
              % ("YES" if a["concavity_signature"] else "no"))
        print("   leave-one-out alpha %.2f to %.2f -> %s"
              % (a["loo_lo"], a["loo_hi"],
                 "STRADDLES 1, sign not robust" if a["loo_straddles_1"]
                 else "stays on one side of 1"))
        print("   dropping r=1 moves c by %+.1f%% -> %s"
              % (a["c_change_dropping_r1_pct"],
                 "not an endpoint artifact"
                 if abs(a["c_change_dropping_r1_pct"]) < 10 else "ENDPOINT EFFECT"))
        print()

    a512 = analyse(512, datadir)
    if a512["enough"]:
        F1, c1, _ = a512["affine"]
        al, Fa, ca, _ = a512["power"]
        print(" WHERE THE TWO MODELS DIVERGE (ctx 512, knee %.0f)" % a512["knee"])
        print("   %5s %10s %10s %9s  %s"
              % ("r", "affine", "power", "gap", "status"))
        for r in (4, 8, 12, 16, 32, 64):
            av, pv = F1 + c1 * r, Fa + ca * r ** al
            st = "measured" if r <= 8 else "EXTRAPOLATION (past the knee)"
            print("   %5d %10.2f %10.2f %8.1f%%  %s"
                  % (r, av, pv, 100 * (pv / av - 1), st))
        print()
        print(" They agree to 1.3% at r=8 and are 35% apart at r=64. alpha is")
        print(" therefore NOT pinned in the regime where it would matter, and")
        print(" ctx 512 on PCIe cannot reach it -- the knee is 8.")
        print()

    print("=" * 78)
    print(" (B) STRUCTURAL -- per-rider by physics, or only as written?")
    print("=" * 78)
    print()
    for name, why, kind, effect in COMPONENTS:
        print(" %-28s %s" % (name, kind))
        print("   %s" % why)
        print("   -> %s" % effect)
        print()
    print(" Only the bytes on the wire is a floor. At the target that is 9.2%")
    print(" of c, so most of c is per-rider by implementation, not by physics.")
    print()


WINDOWS = (4, 6, 8, 12, 16, 24)


def window_sweep(arms=None):
    """alpha as a function of the FIT WINDOW, per arm.

    This is the diagnostic that settled the question. If alpha moves when the
    window moves, with everything physical held fixed, then alpha is a property
    of the fit and not of the system -- and no goodness-of-fit statistic
    computed INSIDE one window can detect that.
    """
    if arms is None:
        arms = [("ctx 512 (M2)", 512, DATA, None),
                ("ctx 256 (M2)", 256, DATA, None),
                ("ctx 1024 (M2)", 1024, DATA, None),
                ("ctx 128 (M7 A1)", 128, M7DATA, "A1_c128.json"),
                ("ctx 128 (M7 A2)", 128, M7DATA, "A2_c128.json"),
                ("ctx 192 (M7 B)", 192, M7DATA, "B_c192.json"),
                ("ctx 256 (M7 C)", 256, M7DATA, "C_c256.json")]
    out = []
    for name, S, d, fn in arms:
        try:
            t0, lk, pts = load(S, d, fn)
        except Exception:
            continue
        if len(pts) < 4:
            continue
        cells = []
        for cap in WINDOWS:
            sub = [p for p in pts if p[0] <= cap]
            cells.append(best_alpha(sub)[0] if len(sub) >= 4 else None)
        vals = [c for c in cells if c is not None]
        out.append({"name": name, "ctx": S, "cells": cells,
                    "swing": (max(vals) - min(vals)) if vals else float("nan"),
                    "r_max": max(r for r, _ in pts), "n": len(pts)})
    return out


def window_report():
    rows = window_sweep()
    print("=" * 78)
    print(" alpha AS A FUNCTION OF THE FIT WINDOW")
    print("=" * 78)
    print()
    print(" Everything physical is fixed within a row. Only the fit window")
    print(" changes. The grid floor is 0.40, so a cell reading exactly 0.400")
    print(" means the fit wanted a smaller exponent still: that window is")
    print(" degenerate.")
    print()
    print(" %-17s %s %8s" % ("arm",
                             " ".join("r<=%-3d" % c for c in WINDOWS),
                             "swing"))
    for r in rows:
        cells = " ".join("%-6s" % ("%.3f" % c if c is not None else "-")
                         for c in r["cells"])
        print(" %-17s %s %8.3f" % (r["name"], cells, r["swing"]))
    print()
    if len(rows) >= 2:
        worst = max(rows, key=lambda r: r["swing"])
        print(" Largest within-arm swing: %s, %.3f -- with the context, the"
              % (worst["name"], worst["swing"]))
        print(" hardware and the data all held fixed.")
    pair = [r for r in rows if "A1" in r["name"] or "A2" in r["name"]]
    if len(pair) == 2:
        i = WINDOWS.index(8)
        a, b = pair[0]["cells"][i], pair[1]["cells"][i]
        if a is not None and b is not None:
            print()
            print(" Two runs of the IDENTICAL configuration, same window r<=8:")
            print("   %.3f and %.3f, differing by %.3f." % (a, b, abs(a - b)))
            print(" alpha is not identified. The affine law is the right")
            print(" description, and any sharing is below resolution.")
    print()


def m7_report(datadir=M7DATA):
    """The high-concurrency sweep, against the two predictions registered in
    m7_amortization.sbatch BEFORE the run.

      P1  If the sharing is launch pipelining, alpha should FALL with shorter
          context, because the amortizable share of c (launch + mirror write)
          rises as context shrinks: ~60% at ctx 512, ~80% at ctx 128. So
          predict alpha(128) < alpha(192) < alpha(256) < alpha(512).

      P2  The already-measured alpha(256)=0.75 > alpha(512)=0.69 is the WRONG
          order for P1. A1 vs A2 -- the same context twice, adjacent in time --
          gives the run-to-run spread of alpha itself, which decides whether
          that gap is a real trend or job scatter.
    """
    print("=" * 78)
    print(" M7: alpha AT HIGH CONCURRENCY")
    print("=" * 78)
    print()
    rows = []
    for fn, S in M7:
        path = os.path.join(datadir, fn + ".json")
        if not os.path.exists(path):
            print(" %-10s ctx %-5d NOT PRESENT YET (%s)" % (fn, S, path))
            continue
        a = analyse(S, datadir, fname=fn + ".json")
        if not a["enough"]:
            print(" %-10s ctx %-5d only %d below-knee points"
                  % (fn, S, a["n"]))
            continue
        a["tag"], a["S"] = fn, S
        rows.append(a)
        F1, c1, rss1 = a["affine"]
        al, Fa, ca, rssa = a["power"]
        rmax = max(r for r, _ in a["pts"])
        print(" %-10s ctx %-5d n=%-3d knee %-5.1f r_max %d"
              % (fn, S, a["n"], a["knee"], rmax))
        print("    affine  F=%6.2f c=%6.3f RSS=%8.4f" % (F1, c1, rss1))
        print("    power   alpha=%.3f F=%6.2f c=%6.3f RSS=%8.4f" % (al, Fa, ca, rssa))
        print("    F(1,%d)=%6.2f -> affine %s   boot alpha %.2f CI [%.2f, %.2f]"
              % (a["n"] - 3, a["fstat"],
                 "REJECTED" if a["rejects_affine"] else "kept",
                 a["alpha_median"], a["alpha_ci"][0], a["alpha_ci"][1]))
        print("    concavity signature: %-3s   dropping r=1 moves c %+.1f%%"
              % ("YES" if a["concavity_signature"] else "no",
                 a["c_change_dropping_r1_pct"]))
        print("    LOO alpha %.2f to %.2f -> %s"
              % (a["loo_lo"], a["loo_hi"],
                 "STRADDLES 1, sign not robust" if a["loo_straddles_1"]
                 else "stays on one side of 1"))
        print()

    if not rows:
        print(" nothing to analyse yet.")
        return

    # P2 FIRST: without the repeatability number, P1 cannot be judged.
    a1 = next((r for r in rows if r["tag"] == "A1_c128"), None)
    a2 = next((r for r in rows if r["tag"] == "A2_c128"), None)
    print("-" * 78)
    print(" P2  run-to-run spread of alpha (same ctx, adjacent in time)")
    print("-" * 78)
    if a1 and a2:
        d = abs(a1["power"][0] - a2["power"][0])
        print("   alpha  A1 = %.3f   A2 = %.3f   |difference| = %.3f"
              % (a1["power"][0], a2["power"][0], d))
        print("   c      A1 = %.3f   A2 = %.3f   (%+.1f%%)"
              % (a1["affine"][1], a2["affine"][1],
                 100 * (a2["affine"][1] / a1["affine"][1] - 1)))
        print("   F      A1 = %.2f    A2 = %.2f    (%+.1f%%)"
              % (a1["affine"][0], a2["affine"][0],
                 100 * (a2["affine"][0] / a1["affine"][0] - 1)))
        print()
        print("   => any context trend in alpha smaller than %.3f is NOT"
              % d)
        print("      resolvable by this campaign.")
    else:
        print("   needs both A1_c128 and A2_c128; not both present.")
    print()

    print("-" * 78)
    print(" P1  does alpha fall with shorter context?")
    print("-" * 78)
    known = {512: 0.69, 256: 0.75}   # from the 02_law sweep
    seen = {}
    for r in rows:
        seen.setdefault(r["S"], []).append(r["power"][0])
    print("   %6s %10s %10s" % ("ctx", "alpha (M7)", "alpha (M2)"))
    for S in sorted(set(list(seen) + list(known))):
        m7v = ("%.3f" % (sum(seen[S]) / len(seen[S]))) if S in seen else "-"
        m2v = ("%.2f" % known[S]) if S in known else "-"
        print("   %6d %10s %10s" % (S, m7v, m2v))
    order = [sum(seen[S]) / len(seen[S]) for S in sorted(seen)]
    if len(order) >= 2:
        rising = all(order[i] <= order[i + 1] for i in range(len(order) - 1))
        print()
        print("   alpha %s with context across the M7 contexts."
              % ("RISES" if rising else "does not rise monotonically"))
        print("   P1 predicted rising (alpha lower at shorter context): %s"
              % ("CONSISTENT" if rising else "NOT CONSISTENT"))
        if a1 and a2:
            spread = abs(a1["power"][0] - a2["power"][0])
            rng = max(order) - min(order)
            print("   range across contexts %.3f vs run-to-run spread %.3f -> %s"
                  % (rng, spread,
                     "trend exceeds noise" if rng > spread
                     else "INSIDE NOISE, no trend claimable"))
    print()

    # what the wider span buys: affine vs power at the largest cohort reached
    print("-" * 78)
    print(" WHAT THE WIDER RIDER AXIS BUYS")
    print("-" * 78)
    for r in rows:
        F1, c1, _ = r["affine"]
        al, Fa, ca, _ = r["power"]
        rmax = max(x for x, _ in r["pts"])
        av, pv = F1 + c1 * rmax, Fa + ca * rmax ** al
        print("   %-10s r_max=%-3d affine %7.2f  power %7.2f  gap %+6.1f%%"
              % (r["tag"], rmax, av, pv, 100 * (pv / av - 1)))
    print()
    print("   At ctx 512 the two models were 1.3% apart at r=8, which is why")
    print("   alpha was unconstrained. A gap well outside the run-to-run")
    print("   spread here is what pins it.")
    print()


def _selftest(datadir=DATA):
    ok = True
    a = analyse(512, datadir, nboot=400)
    good = a["enough"] and a["n"] >= 7
    ok &= good
    print("  [%s] ctx 512 has enough below-knee points (%d)"
          % ("ok " if good else "BAD", a["n"]))

    # THE HEADLINE, asserted so it cannot be quietly re-reported: alpha is NOT
    # identified. It must move substantially when only the fit window moves.
    rows = window_sweep()
    good = bool(rows) and max(r["swing"] for r in rows) > 0.25
    ok &= good
    print("  [%s] alpha is UNSTABLE in the fit window (largest within-arm "
          "swing %.2f) -- it is not a property of the system"
          % ("ok " if good else "BAD",
             max(r["swing"] for r in rows) if rows else float("nan")))

    # and two runs of the same configuration must disagree about it, which is
    # the other half of why no alpha should be quoted
    pair = [r for r in rows if "A1" in r["name"] or "A2" in r["name"]]
    if len(pair) == 2:
        i = WINDOWS.index(8)
        va, vb = pair[0]["cells"][i], pair[1]["cells"][i]
        good = va is not None and vb is not None and abs(va - vb) > 0.15
        ok &= good
        print("  [%s] identical configs disagree on alpha at a fixed window "
              "(%.2f vs %.2f)" % ("ok " if good else "BAD", va or -1, vb or -1))
    else:
        print("  [--] A1/A2 pair not both present, skipping the repeat check")

    # the goodness-of-fit statistics must NOT be trusted here, and the way to
    # show that is that they endorse a value the window sweep destroys
    good = a["rejects_affine"]
    ok &= good
    print("  [%s] the F-test still 'rejects' affine at ctx 512 (F=%.2f) while "
          "alpha is unidentified -- why the window sweep leads"
          % ("ok " if good else "BAD", a["fstat"]))

    # but the models must agree inside the measured range -- that is what
    # stops alpha from being usable for extrapolation
    F1, c1, _ = a["affine"]
    al, Fa, ca, _ = a["power"]
    gap8 = abs((Fa + ca * 8 ** al) / (F1 + c1 * 8) - 1)
    good = gap8 < 0.03
    ok &= good
    print("  [%s] the two models agree to %.1f%% at r=8, so alpha is not "
          "pinned for extrapolation" % ("ok " if good else "BAD", 100 * gap8))

    # structural: exactly one component is a physical floor
    floors = [c for c in COMPONENTS if c[2] == "FLOOR"]
    good = len(floors) == 1 and "link" in floors[0][0]
    ok &= good
    print("  [%s] exactly one component is a physical floor (%s)"
          % ("ok " if good else "BAD", floors[0][0] if floors else "none"))

    print("\nAMORTIZATION SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--m7", action="store_true",
                   help="the high-concurrency sweep and its two predictions")
    p.add_argument("--window", action="store_true",
                   help="alpha vs the fit window -- the primary diagnostic")
    args = p.parse_args()
    if args.selftest:
        return _selftest()
    if args.window:
        window_report()
        return 0
    if args.m7:
        m7_report()
        window_report()
        return 0
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
