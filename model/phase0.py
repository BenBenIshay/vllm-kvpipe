#!/usr/bin/env python3
"""Phase 0: price the mirror write's synchronisation, and decide T2.

Three arms plus a drift bracket, all at one configuration:

    real_a - probe  = the SYNCHRONISATION cost = the prize a correct fix wins
    probe  - skip   = the async copy + scatter a correct fix still pays
    real_a - real_b = drift, which must be small against the prize

`probe` keeps both mirror copies and only makes them non-blocking into pinned
staging, skipping the scatter. `skip` removes the write entirely, so it bounds
the prize only from ABOVE -- which is why the probe exists.

WHY THIS NEEDED MEASURING. Two numbers disagreed. SKIP_D2H differencing on this
code says 9.10 ms, 85% of F. Commit c609d93c7 made exactly this copy async,
measured 1.6 ms and reverted it. They measure different code: the async prefetch
chain now puts more behind the same drain, and the per-row Python loop that
commit named as its own cost driver has since been vectorised away.

STANDING RECOMMENDATION from the proposal: if the prize is <= 2 ms, do not
implement T2. The remaining effort is better spent elsewhere and T2 carries
ordering risk that T1 does not.

    ./.venv/bin/python paper/model/phase0.py
"""
import json
import os
import statistics
import sys

DATA = os.environ.get(
    "OPT_DATA", "/mnt/beegfs/benishay/paper_runs/20260930/overhead/08_opt")

ARMS = [("P0_real_a", "real, before"),
        ("P0_probe", "async into pinned, no scatter"),
        ("P0_skip", "no mirror write at all"),
        ("P0_real_b", "real, after")]

DECIDE_MS = 2.0          # the proposal's abandon threshold


def _ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    f = my - c * mx
    return f, c, max(abs(f + c * x - y) for x, y in zip(xs, ys))


def fit_arm(tag):
    """(F, c, resid, n, t0) below the knee, direct estimator."""
    path = os.path.join(DATA, tag + ".json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    rows = [r for r in d["rows"] if r.get("usable")]
    base = [r for r in rows if r["riders"] == 0]
    if not base:
        return None
    t0 = base[0]["step_ms_direct"] or base[0]["step_ms"]
    xs, ys, lk = [], [], None
    for r in sorted(rows, key=lambda r: r["riders"]):
        n = r["riders"]
        if not n:
            continue
        lk = r["link_ms"] / n
        t = r["step_ms_direct"] or r["step_ms"]
        if t is not None and n < t0 / lk:
            xs.append(n)
            ys.append(t - t0)
    if len(xs) < 3:
        return None
    F, c, res = _ols(xs, ys)
    return {"F": F, "c": c, "resid": res, "n": len(xs), "t0": t0,
            "knee": t0 / lk if lk else None,
            "valid": d.get("numerically_valid"),
            "probe": d.get("d2h_probe"), "skip": d.get("skip_d2h")}


def main():
    got = {}
    print("=" * 78)
    print(" PHASE 0: what is the mirror write's synchronisation worth?")
    print("=" * 78)
    print()
    print(" %-12s %-32s %8s %8s %8s %6s" % ("arm", "what it does", "F", "c",
                                            "resid", "pts"))
    for tag, desc in ARMS:
        a = fit_arm(tag)
        if a is None:
            print(" %-12s %-32s %8s" % (tag, desc, "pending"))
            continue
        got[tag] = a
        print(" %-12s %-32s %8.2f %8.3f %8.3f %6d"
              % (tag, desc, a["F"], a["c"], a["resid"], a["n"]))
    print()

    if "P0_real_a" not in got or "P0_probe" not in got:
        print(" Need at least real_a and probe to decide. Still running.")
        return 0

    ra, pr = got["P0_real_a"], got["P0_probe"]
    prize = ra["F"] - pr["F"]

    drift = None
    if "P0_real_b" in got:
        drift = abs(ra["F"] - got["P0_real_b"]["F"])

    residual = None
    if "P0_skip" in got:
        residual = pr["F"] - got["P0_skip"]["F"]
        upper = ra["F"] - got["P0_skip"]["F"]
    else:
        upper = None

    print("-" * 78)
    print(" THE DECOMPOSITION, on F (the mirror write is per-LAYER, so it")
    print(" lands in F and not in c)")
    print("-" * 78)
    print()
    print("   real_a - probe   = %7.2f ms   the SYNCHRONISATION cost" % prize)
    if residual is not None:
        print("   probe  - skip    = %7.2f ms   async copy + scatter still paid"
              % residual)
        print("   real_a - skip    = %7.2f ms   the SKIP_D2H upper bound" % upper)
    if drift is not None:
        print("   real_a - real_b  = %7.2f ms   DRIFT between the brackets"
              % drift)
    print()

    if drift is not None and prize != 0:
        ratio = abs(prize) / drift if drift else float("inf")
        print(" Drift is %.2f ms against a %.2f ms effect, a ratio of %.1fx."
              % (drift, prize, ratio))
        if ratio < 3:
            print(" THAT IS NOT A CLEAN SEPARATION. The arms are adjacent in")
            print(" time and the brackets straddle the test arms, so if drift")
            print(" is this large relative to the effect the number should not")
            print(" be acted on without repeating the bracket.")
        print()

    print("=" * 78)
    print(" THE T2 DECISION")
    print("=" * 78)
    print()
    print(" Proposal's standing recommendation: abandon T2 if the prize is")
    print(" <= %.1f ms." % DECIDE_MS)
    print()
    print("   measured prize: %.2f ms" % prize)
    print("   c609d93c7 measured 1.6 ms for an async fix and reverted on cost")
    print("   SKIP_D2H differencing bounds it at %s"
          % ("%.2f ms" % upper if upper is not None else "9.10 ms (M3)"))
    print()
    if prize <= DECIDE_MS:
        print(" => DO NOT IMPLEMENT T2. The prize does not clear the threshold,")
        print("    which is the same conclusion c609d93c7 reached by building")
        print("    it and measuring. T1 is already delivered and is worth")
        print("    3.9 ms per rider at the target with no ordering risk.")
    else:
        print(" => IMPLEMENT T2. The prize clears the threshold. Note the")
        print("    ordering obligation: _issue_runs reads the mirror H2D on the")
        print("    hot path, so the deferred scatter must precede it, which")
        print("    build_step does by construction.")
    print()

    # F should be the only thing that moves: the mirror write is per-layer
    print(" SANITY -- c should NOT move, since the write is per-layer:")
    for tag, _ in ARMS:
        if tag in got:
            print("   %-12s c = %.3f" % (tag, got[tag]["c"]))
    cs = [got[t]["c"] for t, _ in ARMS if t in got]
    if len(cs) >= 2:
        spread = (max(cs) - min(cs)) / (sum(cs) / len(cs))
        print("   spread %.0f%% -- %s" % (
            100 * spread,
            "consistent with a per-layer cost" if spread < 0.25
            else "LARGER than expected; the arms may differ in more than the "
                 "mirror write"))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
