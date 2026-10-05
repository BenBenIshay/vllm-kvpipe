#!/usr/bin/env python3
"""What is `F` made of, and how low can it go? (M14)

`F` is the gap worth closing. M13 measured `c` = 0.61 ms/rider at L=80, which
clears the budget by 5.5x, against `F` = 7.68 ms -- 31.7% of the step, three
times the whole per-rider cost at the knee, and paid the moment one rider is
admitted. Halving `F` moves the C2C gain from 1.081x to 1.217x.

`F = L * (per-layer host work the feature adds)`, and two mechanisms do that
work on every layer of every step:

    D2H   the mirror write: the new token's KV copied device->host so the host
          mirror stays current.
    H2D   the streaming: the rider window copied back into the rolling slot.

M14 removes each one, and both, leaving every hook, event and stream ordering in
place. At fixed depth:

    F(base) - F(no_d2h)   what the mirror write costs
    F(base) - F(no_h2d)   what the streaming costs
    F(both)               THE IRREDUCIBLE REMAINDER -- work neither mechanism
                          needs, and therefore the part that could be
                          engineered away

ADDITIVITY IS CHECKED, NOT ASSUMED. If the two mechanisms interact (one hiding
behind the other), the single-ablation costs will not sum to the difference
between base and both, and the interaction term says by how much. Treating them
as additive without checking is the shape of error this campaign has already
made several times.

THE FLOOR this reports is `F` minus the irreducible remainder, i.e. what `F`
would be if the bookkeeping were removed and both mechanisms stayed as they are.
It is an upper bound on what is achievable, not a promise: T4 already measured
the D2H mechanism at its own floor, but nothing here shows the bookkeeping can
actually be deleted.

    ./.venv/bin/python paper/model/f_floor.py
"""
import json
import os
import sys

BASE = os.environ.get(
    "FF_BASE", "/mnt/beegfs/benishay/paper_runs/20260930/overhead/16_f_floor")
MAX_LOAD = float(os.environ.get("FF_MAX_LOAD", "0.3333"))

CORNERS = ("base", "nod2h", "nobus", "both")
LABEL = {"base": "as shipped", "nod2h": "mirror write OFF",
         "nobus": "streaming OFF", "both": "both OFF"}


def fit(tag):
    p = os.path.join(BASE, tag + ".json")
    try:
        d = json.load(open(p))
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or not d.get("rows"):
        return None
    t0 = next((r["step_ms"] for r in d["rows"]
               if r.get("riders") == 0 and r.get("step_ms")), None)
    pts = []
    for r in d["rows"]:
        if not r.get("usable") or not r.get("riders"):
            continue
        if r.get("dT_ms") is None:
            continue
        lm = r.get("link_ms")
        ld = (lm / t0) if (lm and t0) else 0.0
        if ld <= MAX_LOAD:
            pts.append((r["riders"], r["dT_ms"]))
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
    return F, c, n, d


def main():
    print("=" * 78)
    print(" WHAT F IS MADE OF, AND ITS FLOOR   (M14, below-knee, ctx 128)")
    print("=" * 78)
    print()

    got = {}
    for L in (64, 16):
        for k in CORNERS:
            r = fit("FF_%s_L%d" % (k, L))
            if r:
                got[(L, k)] = r
    extra = fit("FF_syncd2h_L64")

    print(" %-20s %6s %8s %8s %6s" % ("arm", "L", "F ms", "c ms/r", "pts"))
    for L in (64, 16):
        for k in CORNERS:
            v = got.get((L, k))
            if not v:
                print(" %-20s %6d   (not present yet)" % (LABEL[k], L))
                continue
            print(" %-20s %6d %8.2f %8.3f %6d" % (LABEL[k], L, v[0], v[1],
                                                  v[2]))
    if extra:
        print(" %-20s %6d %8.2f %8.3f %6d"
              % ("pre-T2 blocking D2H", 64, extra[0], extra[1], extra[2]))
    print()

    for L in (64, 16):
        have = all((L, k) in got for k in CORNERS)
        if not have:
            continue
        Fb = got[(L, "base")][0]
        Fd = got[(L, "nod2h")][0]
        Fh = got[(L, "nobus")][0]
        Fo = got[(L, "both")][0]
        d2h = Fb - Fd
        h2d = Fb - Fh
        inter = Fb - Fd - Fh + Fo
        print(" --- L = %d -------------------------------------------------" % L)
        print("   F as shipped                      %7.2f ms   (%.4f /layer)"
              % (Fb, Fb / L))
        print("   cost of the mirror write (D2H)    %7.2f ms   (%4.0f%% of F)"
              % (d2h, d2h / Fb * 100))
        print("   cost of the streaming (H2D)       %7.2f ms   (%4.0f%%)"
              % (h2d, h2d / Fb * 100))
        print("   IRREDUCIBLE REMAINDER             %7.2f ms   (%4.0f%%)"
              % (Fo, Fo / Fb * 100))
        print("   interaction (base-d2h-h2d+both)   %7.2f ms   %s"
              % (inter,
                 "-> additive" if abs(inter) < 0.15 * Fb else
                 "-> NOT additive; the two mechanisms overlap, so the single"
                 " ablations cannot simply be summed"))
        print()
        print("   THE FLOOR, if the remainder were engineered away and both")
        print("   mechanisms stayed as they are:")
        floor = Fb - Fo
        print("     F_floor = %.2f - %.2f = %7.2f ms  (%.4f /layer, %.0f%% of"
              " today)" % (Fb, Fo, floor, floor / L, floor / Fb * 100))
        print()

    # per-layer scaling of each component, from the two depths
    if all((L, k) in got for L in (16, 64) for k in CORNERS):
        print(" --- PER-LAYER COST OF EACH COMPONENT (from L=16 and L=64) ---")
        print("   %-28s %10s %10s %12s"
              % ("component", "L=16 /lay", "L=64 /lay", "linear in L?"))
        for nm, f in (("F as shipped", lambda L: got[(L, "base")][0]),
                      ("mirror write (D2H)",
                       lambda L: got[(L, "base")][0] - got[(L, "nod2h")][0]),
                      ("streaming (H2D)",
                       lambda L: got[(L, "base")][0] - got[(L, "nobus")][0]),
                      ("irreducible remainder",
                       lambda L: got[(L, "both")][0])):
            a, b = f(16) / 16, f(64) / 64
            same = abs(a - b) <= 0.3 * max(abs(a), abs(b), 1e-9)
            print("   %-28s %10.4f %10.4f %12s"
                  % (nm, a, b, "yes" if same else "NO"))
        print()
        print("   A component that is linear in L scales with depth, so an")
        print("   80-layer model pays it pro rata. One that is NOT linear has")
        print("   a fixed part, and extrapolating it by L would overstate it.")
    print()

    if extra and (64, "base") in got:
        Fb = got[(64, "base")][0]
        print(" --- WHAT T2 ALREADY WON, on the same footing ----------------")
        print("   pre-T2 blocking write  F = %.2f" % extra[0])
        print("   as shipped (async)     F = %.2f   -> T2 saved %.2f ms (%.0f%%)"
              % (Fb, extra[0] - Fb, (extra[0] - Fb) / extra[0] * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
