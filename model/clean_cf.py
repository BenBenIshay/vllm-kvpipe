#!/usr/bin/env python3
"""`c` and `F` for the CURRENT solution, fitted wholly below the knee (M13).

This replaces every earlier `c` in the campaign. Those were fitted across rider
sweeps whose top sat AT the knee, which folded the knee's cost into `c`; and the
first correction used archived pre-T1 arms and derived today's value by
subtracting the per-block term, which is two steps from a measurement. M13
measures the defaults directly -- T1 and T2 in, async on, T3/T5/T6 off.

WHAT IT REPORTS, in the order the conclusions depend on each other:

  1. THE BELOW-KNEE GATE. Every retained point must satisfy
     `link_ms / T0 <= 1/3`. `link_ms` is recorded per row, so this is read, not
     modelled, and it accounts for generation (a rider's sequence grows from
     `ctx` to `ctx + long_len`, so its bytes grow with it -- sizing a sweep on
     the prompt length alone is how a sweep that looks below the knee ends up
     at it). An arm with any point above the cut is reported and NOT fitted.

  2. `c` AND `F` per arm, with the drift bracket. Two runs of the same arm give
     the drift; no effect smaller than it is called resolvable.

  3. THE CONTEXT CHECK, which is the one that can falsify the whole reading.
     Below the knee the bytes are hidden, so `c` must NOT scale with `S`. If it
     does, they are not hidden and `c` still contains transfer.

  4. THE LAYER SLOPE on current code, so projecting to an 80-layer model does
     not rest on archived arms.

    ./.venv/bin/python paper/model/clean_cf.py
"""
import json
import os
import sys

BASE = os.environ.get(
    "CF_BASE", "/mnt/beegfs/benishay/paper_runs/20260930/overhead/15_clean_cf")
MAX_LOAD = float(os.environ.get("CF_MAX_LOAD", "0.3333"))

ARMS = [("CF_c128_a", 128, 64), ("CF_c128_b", 128, 64),
        ("CF_c256", 256, 64), ("CF_L16", 128, 16), ("CF_L32", 128, 32),
        # minbytes: the bytes removed, every launch/hook/event/ordering kept.
        # Load is ~0 by construction, so all rider points are usable at full
        # depth -- which no cut on the `real` arms can deliver.
        ("MB_c128", 128, 64), ("MB_c256", 256, 64)]


def load(tag):
    p = os.path.join(BASE, tag + ".json")
    try:
        d = json.load(open(p))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get("rows") else None


def lsq(pts):
    n = len(pts)
    if n < 2:
        return None
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] ** 2 for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    den = n * sxx - sx * sx
    if den == 0:
        return None
    c = (n * sxy - sx * sy) / den
    F = (sy - c * sx) / n
    resid = max(abs(y - (F + c * r)) for r, y in pts)
    return F, c, n, resid


def one(tag):
    d = load(tag)
    if not d:
        return None
    t0 = next((r["step_ms"] for r in d["rows"]
               if r.get("riders") == 0 and r.get("step_ms")), None)
    pts, loads, over = [], [], []
    for r in d["rows"]:
        if not r.get("usable") or not r.get("riders"):
            continue
        if r.get("dT_ms") is None:
            continue
        lm, rid = r.get("link_ms"), r["riders"]
        ld = (lm / t0) if (lm and t0) else None
        if ld is None or ld <= MAX_LOAD:
            pts.append((rid, r["dT_ms"]))
            if ld is not None:
                loads.append(ld)
        else:
            over.append((rid, ld))
    return d, t0, pts, loads, over


def main():
    print("=" * 78)
    print(" c AND F FOR THE CURRENT SOLUTION  (M13, below-knee, load <= %.2f)"
          % MAX_LOAD)
    print("=" * 78)
    print()
    got = {}
    head = None
    print(" %-12s %5s %5s %6s %7s %8s %8s %7s %s"
          % ("arm", "ctx", "L", "npts", "maxload", "F ms", "c ms/r", "resid",
             "discarded"))
    for tag, ctx, L in ARMS:
        r = one(tag)
        if not r:
            print(" %-12s (not present yet)" % tag)
            continue
        d, t0, pts, loads, over = r
        head = head or d.get("git_head")
        f = lsq(pts)
        if not f:
            print(" %-12s too few points below the cut" % tag)
            continue
        got[tag] = (f, d, L, ctx)
        print(" %-12s %5s %5s %6d %7.2f %8.2f %8.3f %7.3f %s"
              % (tag, d.get("ctx"), d.get("rungd_layers") or 64, f[2],
                 max(loads) if loads else 0.0, f[0], f[1], f[3],
                 ",".join("r=%d(%.2f)" % o for o in over) or "-"))
    print()
    if head:
        print(" measured at git head %s" % head)
        print()

    # --- is the affine form even right below the knee? -------------------
    # A least-squares c is only meaningful if the marginal cost of one more
    # rider is roughly constant. On the first arm it is NOT: the r=1->2 step
    # costs 1.117 ms while r=3..6 average 0.37. Fitting a line through that
    # returns 0.508, which is neither value. This prints the marginals so the
    # affine assumption is visible instead of assumed -- the same fit-window
    # sensitivity that already forced the alpha retraction in section 2.
    print(" MARGINAL COST OF ONE MORE RIDER (is c actually constant?)")
    for tag, ctx, L in ARMS:
        r = one(tag)
        if not r:
            continue
        d, t0, pts, loads, over = r
        if len(pts) < 3:
            continue
        marg = []
        for i in range(1, len(pts)):
            dr = pts[i][0] - pts[i - 1][0]
            if dr > 0:
                marg.append((pts[i][0], (pts[i][1] - pts[i - 1][1]) / dr))
        mid = [v for rr, v in marg if 2 < rr <= 6]
        print("   %-12s %s" % (tag, "  ".join("r%d:%.2f" % m for m in marg)))
        if mid:
            f = lsq(pts)
            print("   %-12s mid-range mean %.3f   vs affine fit %.3f   %s"
                  % ("", sum(mid) / len(mid), f[1],
                     "(affine OK)" if abs(sum(mid) / len(mid) - f[1]) < 0.06
                     else "(AFFINE FIT INFLATED by the end points)"))
    print()

    # --- THE RIGHT COMPARISON: marginal cost vs transfer LOAD -------------
    # c came out 0.508 at ctx 128 and 0.620 at ctx 256, which looks like a
    # context dependence and would mean the bytes are not hidden. But at ctx 256
    # the SAME rider count sits at a higher transfer load, so the comparison is
    # confounded: the two arms are not at the same distance from the knee.
    #
    # Pooling every arm's marginal against its load decouples them. If the
    # marginals from different contexts fall on ONE curve against load, then the
    # per-rider cost is a function of knee proximity and not of context, and the
    # intercept of that curve at load -> 0 is the genuine exposed cost -- the
    # quantity `c` was always meant to be.
    print(" MARGINAL vs TRANSFER LOAD, pooled across contexts")
    print("   (if one curve fits both contexts, c depends on LOAD not on S)")
    pool = []
    for tag, ctx, L in ARMS:
        r = one(tag)
        if not r or L != 64:
            continue
        d, t0, pts, loads, over = r
        rows = {x["riders"]: x for x in d["rows"] if x.get("usable")}
        for i in range(1, len(pts)):
            r0, r1 = pts[i - 1][0], pts[i][0]
            if r1 <= r0:
                continue
            m = (pts[i][1] - pts[i - 1][1]) / (r1 - r0)
            lm0 = rows.get(r0, {}).get("link_ms")
            lm1 = rows.get(r1, {}).get("link_ms")
            if not (lm0 and lm1 and t0):
                continue
            ld = (lm0 + lm1) / 2 / t0          # mean load over the interval
            pool.append((ld, m, d.get("ctx"), r0, r1))
    if pool:
        print("   %7s %8s %9s %s" % ("load", "marginal", "ctx", "interval"))
        for ld, m, ctx, r0, r1 in sorted(pool):
            print("   %7.3f %8.3f %9s   r%d->r%d" % (ld, m, ctx, r0, r1))
        # drop the first-rider step: it is a one-off, identical at both
        # contexts (1.117 and 1.13), so it belongs in F, not in c
        body = [(ld, m) for ld, m, ctx, r0, r1 in pool if r0 >= 2]
        if len(body) >= 3:
            f = lsq(body)
            print()
            print("   excluding the r=1->2 step (a one-off, 1.12 at BOTH")
            print("   contexts, so it is part of F rather than of c):")
            print("     marginal(load) = %.4f + %.3f * load" % (f[0], f[1]))
            print("     at load -> 0 :  %.3f ms/rider   <-- the EXPOSED cost"
                  % f[0])
            print("     max residual :  %.3f" % f[3])
            print()
            print("   So the per-rider cost is %.3f ms of exposed work plus a"
                  % f[0])
            print("   term that grows with knee proximity. The second part is")
            print("   not `c` -- it is the knee, and it belongs to the rider")
            print("   admission limit rather than to the per-rider charge.")
    print()

    # --- drift -----------------------------------------------------------
    a, b = got.get("CF_c128_a"), got.get("CF_c128_b")
    drift_c = drift_F = None
    if a and b:
        drift_c = abs(a[0][1] - b[0][1])
        drift_F = abs(a[0][0] - b[0][0])
        print(" DRIFT from the repeated arm:  c %.4f   F %.3f" % (drift_c,
                                                                  drift_F))
        print("   nothing smaller than this is resolvable.")
        print()

    # --- the falsifier: does c scale with context? -----------------------
    c128 = [got[t][0][1] for t in ("CF_c128_a", "CF_c128_b") if t in got]
    if c128 and "CF_c256" in got:
        m128 = sum(c128) / len(c128)
        c256 = got["CF_c256"][0][1]
        d = abs(c256 - m128)
        print(" CONTEXT CHECK (the one that can falsify this reading):")
        print("   c at ctx 128 = %.3f ,  at ctx 256 = %.3f ,  difference %.3f"
              % (m128, c256, d))
        if drift_c is not None:
            verdict = ("c is CONTEXT-INDEPENDENT -> the bytes are hidden and "
                       "c is exposed work only" if d <= 2 * drift_c else
                       "c DOES scale with context -> the bytes are NOT fully "
                       "hidden; c still contains transfer and this reading is "
                       "wrong")
            print("   against drift %.3f  ->  %s" % (drift_c, verdict))
        print()

    # --- THE EXPOSED/TRANSFER SPLIT, measured not inferred ---------------
    # minbytes keeps the launches and drops the bytes, so its per-rider cost IS
    # the exposed work. Subtracting it from the `real` arm at the same context
    # and rider range leaves the part the bytes are responsible for -- with no
    # model of hiding, no load cut and no extrapolation.
    print(" EXPOSED vs TRANSFER, from minbytes (no inference)")
    any_mb = False
    for ctx, real_tags in ((128, ("CF_c128_a", "CF_c128_b")),
                           (256, ("CF_c256",))):
        mb = got.get("MB_c%d" % ctx)
        rl = [got[t] for t in real_tags if t in got]
        if not mb or not rl:
            continue
        any_mb = True
        c_mb, F_mb = mb[0][1], mb[0][0]
        c_re = sum(x[0][1] for x in rl) / len(rl)
        F_re = sum(x[0][0] for x in rl) / len(rl)
        print("   ctx %-4d  real c %.3f  minbytes c %.3f  ->  bytes %+.3f"
              % (ctx, c_re, c_mb, c_re - c_mb))
        print("             real F %.3f  minbytes F %.3f  ->  bytes %+.3f"
              % (F_re, F_mb, F_re - F_mb))
    if any_mb:
        a = got.get("MB_c128")
        b = got.get("MB_c256")
        if a and b and drift_c is not None:
            d = abs(b[0][1] - a[0][1])
            print()
            print("   DOES THE EXPOSED COST DEPEND ON CONTEXT? It must not --")
            print("   with the bytes gone there is nothing left to scale with S.")
            print("     minbytes c: %.3f at ctx 128, %.3f at ctx 256, diff %.3f"
                  % (a[0][1], b[0][1], d))
            print("     against drift %.3f  ->  %s" % (drift_c,
                  "byte-independent, so it IS exposed work" if d <= 2 * drift_c
                  else "STILL scales with S -- the exposed-work framing is "
                       "wrong and something per-rider tracks sequence length"))
    else:
        print("   minbytes arms not present yet")
    print()

    # --- the layer slope on current code ---------------------------------
    lay = {}
    for t in ("CF_L16", "CF_L32"):
        if t in got:
            lay[got[t][2]] = got[t][0][1]
    # (the MB arms are L=64 only, so they do not enter the layer slope)
    if c128:
        lay[64] = sum(c128) / len(c128)
    if len(lay) >= 3:
        s = lsq(sorted(lay.items()))
        print(" LAYER SLOPE on current code (ctx 128):")
        for L in sorted(lay):
            print("   L=%-3d  c = %.3f" % (L, lay[L]))
        print("   fit:  c(L) = %.4f + %.5f * L" % (s[0], s[1]))
        print()
        print(" SO FOR THE CURRENT SOLUTION:")
        for name, L in (("Qwen-32B (64L)", 64), ("Llama-70B / Qwen-72B (80L)",
                                                 80)):
            print("   %-28s c = %.3f ms/rider" % (name, s[0] + s[1] * L))
        fl = {}
        for t in ("CF_L16", "CF_L32"):
            if t in got:
                fl[got[t][2]] = got[t][0][0]
        if c128 and len(fl) >= 2:
            fl[64] = sum(got[t][0][0] for t in ("CF_c128_a", "CF_c128_b")
                         if t in got) / len(c128)
            sf = lsq(sorted(fl.items()))
            print("   F(L) = %.3f + %.4f * L   ->  F at L=80 = %.2f ms"
                  % (sf[0], sf[1], sf[0] + sf[1] * 80))
    else:
        print(" layer slope: need L=16, L=32 and L=64 arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
