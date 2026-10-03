#!/usr/bin/env python3
"""Reproduce sections 3.1 and 4 of OVERHEAD_CHARACTERISATION.md from the
archived JSONs -- how `c` scales with context, and whether the verdict survives
the two routes to `c`'s invariant/proportional split.

This exists because those two tables decide the campaign's answer, and a table
typed into a markdown file is not a result. Run it and the numbers regenerate:

    ./.venv/bin/python paper/model/c_scaling.py
    ./.venv/bin/python paper/model/c_scaling.py --selftest

The knee filter is the load-bearing part. `dT = F + c*r` describes only the
regime where a rider's transfer still hides inside compute; above the knee the
step IS the transfer and a fit over both regimes absorbs the kink as curvature.
That failure already happened once in this campaign and passed its held-out
check at 4.2% while being structurally wrong.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

DATA = os.environ.get(
    "OVERHEAD_DATA",
    "/mnt/beegfs/benishay/paper_runs/20260930/overhead/02_law")

# Qwen-32B fp16: 64 layers, 8 KV heads, head_dim 128, 2 bytes
Q32_BTOK = 64 * 2 * 8 * 128 * 2
A100_LINK = 25.1e9

# route 2, from overhead_model.COEFF -- duplicated deliberately so this script
# reproduces the report standalone even if the model module is refactored.
B_LAYER, D_BLOCK, C_LINK, D2H_R, BS = 0.00838, 0.00765, 0.0744, 0.109, 16


def _ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    f = my - c * mx
    return f, c, max(abs(f + c * x - y) for x, y in zip(xs, ys))


def per_context_fits(datadir=DATA, contexts=(256, 512, 1024, 2048)):
    """Affine fit per context, below the knee only, DIRECT estimator.

    The direct estimator (median over steady-window step intervals) is used
    rather than differencing: at r=16 the two disagree by -2.7% because the
    ramp to a full cohort outlasts a short run, and differencing inflates while
    the direct one is immune.

    EFFECTIVE CONTEXT. `ctx` is the PROMPT length, but what a rider streams
    each step is its CURRENT sequence length, which grows as it generates. The
    context that c actually scales with is therefore measured, not nominal, and
    we take it from the bytes the harness reports at r=1:

        S_eff = stream_bytes_per_step(r=1) / b_tok

    Across the four sweeps this comes out a constant ~79 tokens above the
    prompt (335/591/1103/2127 for 256/512/1024/2048), which is the average
    generated length over the steady window. The correction leaves the fitted
    SLOPE untouched -- a constant x-offset cannot change it -- and moves the
    INTERCEPT by slope*79, from 0.205 to 0.060 ms/rider. That matters for the
    one claim the intercept carries, namely the size of c's context-independent
    floor, and it is why the floor is better estimated from the layer axis than
    by extrapolating this fit to S=0.
    """
    out = {}
    for S in contexts:
        path = os.path.join(datadir, "A_c%d.json" % S)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        rows = [r for r in d["rows"] if r.get("usable")]
        base = [r for r in rows if r["riders"] == 0]
        if not base:
            continue
        t0 = base[0]["step_ms_direct"] or base[0]["step_ms"]
        xs, ys, link_per_rider = [], [], None
        for r in rows:
            n = r["riders"]
            if not n:
                continue
            lk = r["link_ms"] / n          # nominal link ms for ONE rider
            link_per_rider = lk
            if n >= t0 / lk:               # KNEE FILTER
                continue
            t = r["step_ms_direct"] or r["step_ms"]
            if t is None:
                continue
            xs.append(n)
            ys.append(t - t0)
        # effective context from the measured bytes, not the nominal prompt
        r1 = next((r for r in rows if r["riders"] == 1), None)
        s_eff = None
        if r1 and r1.get("stream_gb_per_step"):
            s_eff = r1["stream_gb_per_step"] * 1e9 / Q32_BTOK
        rec = {"n": len(xs), "t0": t0, "riders": xs,
               "knee": (t0 / link_per_rider) if link_per_rider else None,
               "link_per_rider": link_per_rider,
               "s_nominal": S, "s_eff": s_eff}
        if len(xs) >= 3:
            F, c, res = _ols(xs, ys)
            rec.update(F=F, c=c, resid=res,
                       c_over_S=c / (s_eff or S),
                       c_over_S_nominal=c / S,
                       hidden=1.0 - c / link_per_rider)
        out[S] = rec
    return out


def route1(fits):
    """c = c0 + k*S fitted across contexts. Needs >=3 fitted contexts."""
    pts = [(r.get("s_eff") or S, r["c"])
           for S, r in sorted(fits.items()) if "c" in r]
    if len(pts) < 3:
        raise SystemExit("route 1 needs >=3 fitted contexts, have %d" % len(pts))
    c0, k, _ = _ols([p[0] for p in pts], [p[1] for p in pts])
    return c0, k, pts


def route2():
    """c0 and k implied by the component fit, for Qwen-32B on A100."""
    c0 = B_LAYER * 64 + D2H_R
    k = D_BLOCK / BS + C_LINK * Q32_BTOK / A100_LINK * 1e3
    return c0, k


def verdict_sensitivity(S=8192, L=80, w=72.7, hbm_bw=4000.0, hbm=96.0,
                        ovh=4.0, eff=0.75, link_gbs=419.0):
    """Run the GH200 verdict under BOTH routes.

    The S-slope must be split into the link-invariant per-block piece and the
    link-bound piece, and only route 2 can do that (slice B swept block_size).
    So route 1 inherits route 2's link coefficient and attributes the REMAINDER
    of its larger slope to per-block, which is the pessimistic reading.
    """
    btok = L * 2 * 8 * 128 * 2
    t0 = w / (hbm_bw * eff) * 1e3
    cap = hbm - w - ovh
    b_res = max(1.0, cap / (btok * S / 1e9))
    t0_over_b = t0 / b_res
    link_gh = C_LINK * btok / (link_gbs * 1e9) * 1e3
    link_a100 = C_LINK * Q32_BTOK / A100_LINK * 1e3
    knee = t0 / (btok * S / (link_gbs * 1e9) * 1e3)

    fits = per_context_fits()
    r1_c0, r1_k, _ = route1(fits)
    r2_c0, r2_k = route2()

    rows = []
    for name, c0, k in (("route 1 (3-context)", r1_c0, r1_k),
                        ("route 2 (component)", r2_c0, r2_k)):
        pb = max(0.0, k - link_a100)           # per-block part of the slope
        for fixed, lab in ((False, "as built"), (True, "block+D2H fixed")):
            c = (c0 * L / 64 - (D2H_R if fixed else 0.0)
                 + (0.0 if fixed else pb * S) + link_gh * S)
            F = L * (0.026 if fixed else 0.142 + 0.026)
            if c >= t0_over_b:
                v, r_be = "never pays", None
            else:
                r_be = F / (t0_over_b - c)
                v = ("PAYS from r=%.1f" % r_be if r_be <= knee
                     else "r_be %.1f > knee %.1f" % (r_be, knee))
            rows.append({"route": name, "config": lab, "c": c, "F": F,
                         "t0_over_b": t0_over_b, "knee": knee, "r_be": r_be,
                         "verdict": v, "pays": r_be is not None and r_be <= knee})
    return rows, t0, b_res, t0_over_b


def report():
    fits = per_context_fits()
    print("=" * 76)
    print(" 3.1  HOW c SCALES WITH CONTEXT   (A100, Qwen-32B, L=64, bs=16)")
    print("=" * 76)
    print(" below-knee points only, direct estimator, data: %s\n" % DATA)
    print(" %-5s %6s %4s %8s %9s %11s %7s %8s %7s"
          % ("ctx", "S_eff", "pts", "F", "c", "c/S_eff", "knee", "maxres",
             "hidden"))
    for S, r in sorted(fits.items()):
        if "c" not in r:
            print(" %-5d %6s %4d   excluded: below-knee points are only %s"
                  % (S, "%.0f" % r["s_eff"] if r.get("s_eff") else "-",
                     r["n"], r["riders"]))
            continue
        print(" %-5d %6.0f %4d %8.2f %9.3f %11.6f %7.1f %8.2f %6.0f%%"
              % (S, r["s_eff"], r["n"], r["F"], r["c"], r["c_over_S"],
                 r["knee"], r["resid"], 100 * r["hidden"]))
    print()
    print(" S_eff is the MEASURED streamed context (prompt plus the average")
    print(" generated length over the steady window), which is what c scales")
    print(" with. It sits a constant ~79 tokens above the prompt here.")
    c0, k, pts = route1(fits)
    print("\n c/S is NOT constant, which is the signature of an S-independent")
    print(" floor inside c. Fitting c = c0 + k*S over the fitted contexts:")
    print("     c = %.3f + %.3e * S   ms/rider" % (c0, k))
    for S, c in pts:
        print("       ctx %4d  measured %.3f   fit %.3f" % (S, c, c0 + k * S))

    print()
    print("=" * 76)
    print(" 4.  DOES THE VERDICT SURVIVE HOW c IS APPORTIONED?")
    print("=" * 76)
    r2c0, r2k = route2()
    print("\n %-34s %10s %12s %9s" % ("route", "c0", "k", "c@512"))
    print(" %-34s %10.3f %12.3e %9.3f"
          % ("three contexts, 3 jobs, direct", c0, k, c0 + k * 512))
    print(" %-34s %10.3f %12.3e %9.3f"
          % ("105-pt within-job component fit", r2c0, r2k, r2c0 + r2k * 512))
    m512 = fits.get(512, {}).get("c")
    if m512:
        print(" %-34s %10s %12s %9.3f" % ("measured at ctx 512", "-", "-", m512))
    print("\n Route 1 puts %.0f%% less in the floor and %.0f%% more in the slope."
          % (100 * (1 - c0 / r2c0), 100 * (k / r2k - 1)))
    print(" They diverge only on EXTRAPOLATION, which is where the projection")
    print(" lives -- so the verdict is run under both.\n")

    rows, t0, b_res, tb = verdict_sensitivity()
    print(" GH200 / Llama-70B fp8 / ctx 8192:  T0=%.1f  B_cap=%.1f  T0/B=%.2f\n"
          % (t0, b_res, tb))
    print(" %-22s %-18s %8s %8s  %s"
          % ("route", "configuration", "c", "T0/B", "verdict"))
    for r in rows:
        print(" %-22s %-18s %8.3f %8.3f  %s"
              % (r["route"], r["config"], r["c"], r["t0_over_b"], r["verdict"]))
    built = [r for r in rows if r["config"] == "as built"]
    fixed = [r for r in rows if r["config"] != "as built"]
    print("\n as built:          %s"
          % ("NEITHER route pays" if not any(r["pays"] for r in built)
             else "routes DISAGREE -- the report's headline is unsafe"))
    print(" block+D2H fixed:   %s"
          % ("BOTH routes pay" if all(r["pays"] for r in fixed)
             else "routes DISAGREE -- the report's headline is unsafe"))
    print()


def selftest():
    ok = True
    fits = per_context_fits()

    good = len([1 for r in fits.values() if "c" in r]) >= 3
    ok &= good
    print("  [%s] at least 3 contexts have enough below-knee points (%d)"
          % ("ok " if good else "BAD",
             len([1 for r in fits.values() if "c" in r])))

    # c must RISE with context -- it is a transfer term
    cs = [r["c"] for _, r in sorted(fits.items()) if "c" in r]
    good = all(cs[i] < cs[i + 1] for i in range(len(cs) - 1))
    ok &= good
    print("  [%s] c rises monotonically with context: %s"
          % ("ok " if good else "BAD", " < ".join("%.3f" % c for c in cs)))

    # F must NOT follow the context -- it is launch-bound
    Fs = [r["F"] for _, r in sorted(fits.items()) if "c" in r]
    spread = (max(Fs) - min(Fs)) / (sum(Fs) / len(Fs))
    good = spread < 0.15
    ok &= good
    print("  [%s] F is context-independent to %.1f%% (%s)"
          % ("ok " if good else "BAD", 100 * spread,
             " ".join("%.2f" % f for f in Fs)))

    # the hidden fraction must be high and stable
    hs = [r["hidden"] for _, r in sorted(fits.items()) if "c" in r]
    good = all(0.70 < h < 0.90 for h in hs)
    ok &= good
    print("  [%s] hidden fraction stable in 70-90%%: %s"
          % ("ok " if good else "BAD", " ".join("%.0f%%" % (100 * h) for h in hs)))

    # c/S_eff is NEARLY CONSTANT, and that is the corrected finding.
    #
    # An earlier version of this selftest asserted the opposite -- that c/S
    # FALLS from the shortest context, which it read as the signature of a
    # context-independent floor inside c. That was an artifact of dividing by
    # the NOMINAL prompt length. Against the measured streamed context the
    # spread collapses from 37% to 13% and stops being monotonic, so these
    # three points carry almost no evidence for a floor: the fitted intercept
    # is 0.06 ms/rider, about 8% of c at the shortest context.
    #
    # Consequence, and it is why this is asserted rather than just noted: the
    # floor must be estimated from the LAYER axis, which measures the
    # per-rider-per-layer term directly at fixed head geometry, and NOT by
    # extrapolating this fit to S=0 across a 3.3x context range.
    rs = [r["c_over_S"] for _, r in sorted(fits.items()) if "c" in r]
    spread = max(rs) / min(rs) - 1
    good = spread < 0.25
    ok &= good
    print("  [%s] c/S_eff spread is only %.0f%% -- c is nearly proportional to "
          "streamed bytes" % ("ok " if good else "BAD", 100 * spread))

    c0, k, _ = route1(fits)
    shortest = min((r["c"] for _, r in fits.items() if "c" in r))
    frac = c0 / shortest
    good = frac < 0.20
    ok &= good
    print("  [%s] route-1 intercept is %.0f%% of the smallest measured c, so it"
          " cannot estimate the floor" % ("ok " if good else "BAD", 100 * frac))

    # THE HEADLINE: both routes must agree on both verdicts
    rows, _, _, _ = verdict_sensitivity()
    built = [r for r in rows if r["config"] == "as built"]
    fixed = [r for r in rows if r["config"] != "as built"]
    good = not any(r["pays"] for r in built)
    ok &= good
    print("  [%s] as built: neither route pays (c = %s vs T0/B = %.2f)"
          % ("ok " if good else "BAD",
             " / ".join("%.2f" % r["c"] for r in built), built[0]["t0_over_b"]))
    good = all(r["pays"] for r in fixed)
    ok &= good
    print("  [%s] block+D2H fixed: both routes pay (r_be = %s, knee %.1f)"
          % ("ok " if good else "BAD",
             " / ".join("%.1f" % r["r_be"] for r in fixed), fixed[0]["knee"]))

    print("\nC-SCALING SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
