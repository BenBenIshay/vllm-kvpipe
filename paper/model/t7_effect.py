#!/usr/bin/env python3
"""Did T7 fire in the engine, and what did it take off `F`? (M15)

T7 attacks the two parts of `F` that need no architectural change:

  T7a  the L per-layer host mirrors become ONE pinned allocation, so T2's
       deferred scatter goes from 2*L `index_copy_` calls to one per tensor
       with a vectorised index.                       target -0.018 ms/layer
  T7b  the four per-layer allocations in the rider override become grown-only
       buffers cached per layer.                       target -0.015 ms/layer

THE MECHANISM IS CHECKED BEFORE THE TIMING. T7a's fast path is guarded on three
conditions, and if any fails in the real engine the per-layer loop runs instead
and T7a is a no-op however well it tests off-GPU. `wt_flush_vectorised` and
`wt_flush_per_layer` are counted per step, so which path ran is read rather than
assumed. T1 nearly failed this way and T5 did.

T7b HAS NO KILL SWITCH -- it is a pure allocation change with no behavioural
flag -- so the off arms isolate T7a, and T7b shows up only as the shift between
these base arms and M14's. Between-job drift on `F` is ~0.25 ms (M13's 6.03 vs
M14's 5.80 on the same configuration), so that cross-job comparison is weaker
than the within-job bracket and is labelled as such.

    ./.venv/bin/python paper/model/t7_effect.py
"""
import json
import os
import sys

BASE = os.environ.get(
    "T7_BASE", "/mnt/beegfs/benishay/paper_runs/20260930/overhead/17_t7")
MAX_LOAD = float(os.environ.get("T7_MAX_LOAD", "0.3333"))

# M14, same configuration (ctx 128, riders 0..8), for the cross-job reference.
M14 = {("base", 64): 5.80, ("nod2h", 64): 1.85, ("base", 16): 1.09}

ARMS = [("T7_on_a", 64, "T7a ON"), ("T7_off", 64, "T7a OFF (control)"),
        ("T7_on_b", 64, "T7a ON"), ("T7_on_L16", 16, "T7a ON"),
        ("T7_off_L16", 16, "T7a OFF (control)"),
        ("T7_nod2h_L64", 64, "T7a ON, mirror write OFF")]


def fit(tag):
    try:
        d = json.load(open(os.path.join(BASE, tag + ".json")))
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
        if not lm or not t0 or lm / t0 <= MAX_LOAD:
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
    return (sy - c * sx) / n, c, n


def paths(tag):
    """(vectorised steps, per-layer steps) from the phase JSONL."""
    p = os.path.join(BASE, tag + "_phases.jsonl")
    vec = per = 0
    try:
        for line in open(p):
            if '"wt_flush_vectorised"' in line:
                vec += 1
            if '"wt_flush_per_layer"' in line:
                per += 1
    except OSError:
        return None
    return vec, per


def main():
    print("=" * 78)
    print(" 1. DID T7a's FAST PATH FIRE IN THE ENGINE?")
    print("=" * 78)
    print()
    print(" %-16s %12s %12s %s" % ("arm", "vectorised", "per-layer", "verdict"))
    for tag, L, lab in ARMS:
        pr = paths(tag)
        if pr is None:
            print(" %-16s   (no phase file yet)" % tag)
            continue
        vec, per = pr
        if "OFF" in lab and "mirror write" not in lab:
            ok = vec == 0 and per > 0
            v = "control: per-layer, as intended" if ok else "CONTROL WRONG"
        else:
            ok = vec > 0 and per == 0
            v = ("vectorised on every flush" if ok else
                 "FELL BACK -- the guard did not pass, so T7a is a no-op here")
        print(" %-16s %12d %12d %s" % (tag, vec, per, v))
    print()

    print("=" * 78)
    print(" 2. WHAT IT BOUGHT")
    print("=" * 78)
    print()
    print(" %-16s %5s %-26s %8s %8s %6s"
          % ("arm", "L", "config", "F ms", "c ms/r", "pts"))
    got = {}
    for tag, L, lab in ARMS:
        f = fit(tag)
        if not f:
            print(" %-16s %5d   (not present yet)" % (tag, L))
            continue
        got[tag] = (f, L)
        print(" %-16s %5d %-26s %8.2f %8.3f %6d"
              % (tag, L, lab, f[0], f[1], f[2]))
    print()

    # within-job bracket: T7a on vs off, which is the trustworthy comparison
    for L, ons, off in ((64, ("T7_on_a", "T7_on_b"), "T7_off"),
                        (16, ("T7_on_L16",), "T7_off_L16")):
        have_on = [got[t][0][0] for t in ons if t in got]
        if not have_on or off not in got:
            continue
        f_on = sum(have_on) / len(have_on)
        f_off = got[off][0][0]
        drift = (max(have_on) - min(have_on)) if len(have_on) > 1 else None
        eff = f_on - f_off
        print(" L=%-3d  T7a OFF %.2f -> ON %.2f   effect %+.2f ms (%+.4f/layer)"
              % (L, f_off, f_on, eff, eff / L))
        if drift is not None:
            print("        bracket drift on the ON arms: %.2f  -> %s"
                  % (drift, "RESOLVABLE" if abs(eff) > drift
                     else "not resolvable"))
        print()

    # cross-job, weaker: T7b has no flag so it only shows against M14
    if "T7_on_a" in got and ("base", 64) in M14:
        f_on = sum(got[t][0][0] for t in ("T7_on_a", "T7_on_b") if t in got)
        n_on = len([t for t in ("T7_on_a", "T7_on_b") if t in got])
        f_on /= max(n_on, 1)
        print(" CROSS-JOB, T7a+T7b together against M14's base (weaker: the")
        print(" between-job drift on F is ~0.25 ms, M13 6.03 vs M14 5.80):")
        print("   L=64   M14 %.2f -> M15 %.2f   = %+.2f ms (%+.4f/layer)"
              % (M14[("base", 64)], f_on, f_on - M14[("base", 64)],
                 (f_on - M14[("base", 64)]) / 64))
        if "T7_on_L16" in got and ("base", 16) in M14:
            f16 = got["T7_on_L16"][0][0]
            print("   L=16   M14 %.2f -> M15 %.2f   = %+.2f ms (%+.4f/layer)"
                  % (M14[("base", 16)], f16, f16 - M14[("base", 16)],
                     (f16 - M14[("base", 16)]) / 16))
        print()

    # where F stands now, and what is left
    if "T7_on_a" in got and "T7_nod2h_L64" in got:
        f_on = got["T7_on_a"][0][0]
        f_nod = got["T7_nod2h_L64"][0][0]
        print(" F's SPLIT AFTER T7, at L=64:")
        print("   F                        %.2f ms" % f_on)
        print("   mirror write             %.2f ms  (%.0f%%)"
              % (f_on - f_nod, (f_on - f_nod) / f_on * 100))
        print("   everything else          %.2f ms  (%.0f%%)"
              % (f_nod, f_nod / f_on * 100))
        print()
        print("   M14 had mirror 3.95 / else 1.85 of 5.80. If the mirror share")
        print("   fell, T7a worked on the scatter; if 'everything else' fell,")
        print("   that is T7b on the override allocations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
