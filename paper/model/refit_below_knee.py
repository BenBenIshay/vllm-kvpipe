#!/usr/bin/env python3
"""Re-fit `c` using only rider counts WELL BELOW the knee.

WHY THIS EXISTS. `c` was fitted as a single straight line through every rider
point in a sweep, and at the short contexts the campaign used, the top of that
sweep sits AT the knee. Below the knee the rider transfer hides behind compute
and contributes nothing to step time; at the knee it stops hiding and the whole
transfer lands in the step. A straight line through both regimes smears the
knee-proximity cost backwards and reports it as a constant per-rider charge --
which is how a transfer cost ended up inside a coefficient named for memcpy
launches, and why three optimisations aimed at launches all failed.

The measurement that exposed it (`01_perturbation`, differencing the xfer_mode
arms at ctx 512, so no fitting at all):

      riders   bytes cost   transfer needed   un-hidden
           1       -0.26              5.3        ~0%
           2       -0.25             10.7        ~0%
           4        1.44             21.4        6.7%
           8        3.50             42.8        8.2%     <- knee is 9.2

At 1-2 riders the bytes cost NOTHING. The cost appears only on approach to the
knee. So the honest `c` -- the extra work a rider adds that cannot hide -- must
be fitted where the transfer is still hidden.

THE CUT. A point is kept when `link_ms / T0 <= MAX_LOAD`, i.e. the rider
transfer needs no more than that fraction of the step. `link_ms` is recorded per
row by the harness, so this is read rather than modelled. 1/3 is the default:
the differenced table above shows hiding still complete at 0.22 and breaking by
0.33.

TWO ESTIMATORS, reported side by side, because neither alone is convincing:

  * BELOW-KNEE fit of the `real` arm -- the direct answer, but on few points.
  * The `minbytes` arm at ALL rider counts -- bytes removed, so there is no knee
    to approach and every point is usable. It measures launches + host work,
    which below the knee is what `c` should be.

Agreement between them is the evidence that the cut is in the right place.

    ./.venv/bin/python paper/model/refit_below_knee.py
"""
import glob
import json
import os
import sys

BASE = os.environ.get(
    "REFIT_BASE", "/mnt/beegfs/benishay/paper_runs/20260930/overhead")
MAX_LOAD = float(os.environ.get("REFIT_MAX_LOAD", "0.3333"))


def load(path):
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get("rows") else None


def lsq(pts):
    """(F, c, n) for dT = F + c*r."""
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
    return (sy - c * sx) / n, c, n


def arm(d):
    """(T0, [(r, dT, load)]) with load = link_ms / T0."""
    t0 = None
    for r in d["rows"]:
        if r.get("riders") == 0 and r.get("step_ms"):
            t0 = r["step_ms"]
    pts = []
    for r in d["rows"]:
        if not r.get("usable") or not r.get("riders"):
            continue
        if r.get("dT_ms") is None:
            continue
        lm = r.get("link_ms")
        load = (lm / t0) if (lm and t0) else None
        pts.append((r["riders"], r["dT_ms"], load))
    return t0, pts


def report(label, d):
    t0, pts = arm(d)
    if not pts:
        return None
    allf = lsq([(r, y) for r, y, _ in pts])
    kept = [(r, y) for r, y, L in pts if L is not None and L <= MAX_LOAD]
    sub = lsq(kept)
    knee = None
    lm1 = next((L for r, _, L in pts if r == 1 and L), None)
    if lm1:
        knee = 1.0 / lm1
    print(" %-16s L=%-3s S=%-5s  knee~%-5s  rows %d, kept %d (load<=%.2f)"
          % (label, d.get("rungd_layers") or 64, d.get("ctx"),
             "%.1f" % knee if knee else "?", len(pts), len(kept), MAX_LOAD))
    if allf:
        print("      all points : F %7.2f  c %7.3f" % (allf[0], allf[1]))
    if sub:
        print("      below knee : F %7.2f  c %7.3f   <-- %s"
              % (sub[0], sub[1],
                 "%.0f%% lower" % ((1 - sub[1] / allf[1]) * 100)
                 if allf and allf[1] else ""))
    else:
        print("      below knee : too few points at this load cut")
    return (d, allf, sub, knee)


def main():
    print("=" * 78)
    print(" RE-FIT BELOW THE KNEE   (load cut: link_ms / T0 <= %.3f)" % MAX_LOAD)
    print("=" * 78)
    print()

    print(" --- the context sweep (arm A of the law), xfer_mode=real ---")
    ctxs = {}
    for f in sorted(glob.glob(os.path.join(BASE, "02_law", "A_c*.json"))):
        d = load(f)
        if not d:
            continue
        r = report(os.path.basename(f).replace(".json", ""), d)
        if r:
            ctxs[d.get("ctx")] = r
    print()

    print(" --- the LAYER sweep, which is how a per-layer term gets its slope ---")
    lay = {}
    for tag, L in (("C_L8_c512", 8), ("C_L16_c512", 16), ("C_L32_c512", 32),
                   ("C_L48_c512", 48), ("A_c512", 64)):
        d = load(os.path.join(BASE, "02_law", tag + ".json"))
        if not d:
            continue
        r = report(tag, d)
        if r:
            lay[L] = r
    print()

    # THE CUT IS NOT ENOUGH ON ITS OWN. Truncating a sweep to its lowest 2-3
    # points leaves the fit ill-conditioned: with F ~ 11 ms and only r = 1,2
    # surviving, the "slope" is mostly the r=1 point's share of F, and the
    # answer comes out HIGHER than the contaminated fit rather than lower. So
    # the arms to trust are the ones that are ENTIRELY below the knee and need
    # no truncation at all -- they keep every point and stay well conditioned.
    print(" --- arms that are WHOLLY below the knee (no truncation needed) ---")
    clean = {}
    for L, (d, allf, sub, knee) in sorted(lay.items()):
        t0, pts = arm(d)
        loads = [x[2] for x in pts if x[2] is not None]
        if loads and max(loads) <= MAX_LOAD and allf:
            clean[L] = allf[1]
            print("  L=%-3d  c %7.3f   all %d points kept, max load %.2f"
                  % (L, allf[1], allf[2], max(loads)))
        else:
            print("  L=%-3d  SKIPPED -- reaches load %.2f, fit contaminated"
                  % (L, max(loads) if loads else float("nan")))
    # the minbytes arm supplies a clean L=64 point: bytes gone, so no knee
    mb = load(os.path.join(BASE, "01_perturbation", "xfer_min_c512.json"))
    if mb:
        t0, pts = arm(mb)
        f = lsq([(r, y) for r, y, _ in pts])
        if f:
            clean[64] = f[1]
            print("  L=64   c %7.3f   from minbytes (bytes removed, %d points)"
                  % (f[1], f[2]))
    print()
    if len(clean) >= 3:
        s2 = lsq(sorted(clean.items()))
        print(" CLEAN FIT  c(L) = %.4f + %.5f * L" % (s2[0], s2[1]))
        print("   at L=64 that predicts c = %.3f" % (s2[0] + s2[1] * 64))
        print("   the law's b_per_rider_layer is 0.00838 -> overstated %.1fx"
              % (0.00838 / s2[1]))
        print("   the contaminated L-slopes were 0.0131 (all points) and"
              " 0.0185 (truncated)")
    print()

    print(" --- the INDEPENDENT estimator: bytes removed, all rider points ---")
    for tag in ("xfer_min_c512", "xfer_min_c256"):
        d = load(os.path.join(BASE, "01_perturbation", tag + ".json"))
        if not d:
            continue
        t0, pts = arm(d)
        f = lsq([(r, y) for r, y, _ in pts])
        if f:
            print(" %-16s S=%-5s  F %7.2f  c %7.3f   (%d points, no knee to"
                  " approach)" % (tag, d.get("ctx"), f[0], f[1], f[2]))
    print()

    print("=" * 78)
    print(" WHAT TO CONCLUDE")
    print("=" * 78)
    print()
    print(" If the below-knee `c` of the real arm agrees with the minbytes `c`,")
    print(" then the cut is in the right place and `c` is launches + host work")
    print(" only -- the bytes are hidden, as the design intends, and the knee")
    print(" (not `c`) is what the link bandwidth moves.")
    print()
    print(" If `c` below the knee is also INDEPENDENT OF CONTEXT, that is the")
    print(" strongest version of the claim: a per-rider cost that does not")
    print(" scale with the bytes cannot be the transfer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
