#!/usr/bin/env python3
"""M21: is c's HOST component context-independent, tested at the target context?

    ./.venv/bin/python paper/model/c_host_ladder.py

THE INSTRUMENT. `c` is defined below the knee, and on a PCIe part the knee forbids
measuring it at long context: at ctx 512 only two rider points clear l/T0 <= 1/3, at
4096 one, at 8192 none. Removing the bytes removes the knee -- `minbytes` issues
every launch, gather and block-table entry and sends ~none of the bytes, so
l/T0 ~ 0.007 at ANY context and every rider point is below the knee by construction.

WHAT IT MEASURES. c_host: the launch and bookkeeping part of the per-request cost.
That is exactly the quantity the paper's claim concerns ("below the knee the
transfer is hidden, so c is host work"), and because at ctx 8192 on C2C the knee is
6.4 riders -- so the transfer IS hidden there -- c_host is the right proxy for the
coefficient Table 2 needs on hardware we do not have.

Every arm is fitted over the SAME rider range, and every point is below the knee in
the bytes-removed arms, so neither of the two confounds that spoiled M20 applies:
unequal point counts, and transfer-bound points entering the fit.
"""
import glob
import json
import os
import statistics as st
import sys

D = os.environ.get(
    "M21_DIR",
    "/mnt/beegfs/benishay/paper_runs/20260930/overhead/23_c_nobus_ladder")
RS = (1, 2, 3, 4, 5, 6)          # identical at every context


def fit(path):
    d = json.load(open(path))
    t0 = next((r["step_ms"] for r in d["rows"]
               if r.get("riders") == 0 and r.get("step_ms")), None)
    by = {r["riders"]: (r["dT_ms"], (r.get("link_ms") or 0.0) / t0)
          for r in d["rows"]
          if r.get("usable") and r.get("riders") and r.get("dT_ms") is not None}
    xs = [(r, by[r][0]) for r in RS if r in by]
    if len(xs) < 3:
        return None
    n = len(xs)
    sx = sum(x for x, _ in xs); sy = sum(y for _, y in xs)
    sxx = sum(x * x for x, _ in xs); sxy = sum(x * y for x, y in xs)
    c = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    worst = max(by[r][1] for r in RS if r in by)
    return dict(ctx=d["ctx"], c=c, F=(sy - c * sx) / n, n=n, worst=worst,
                xfer=d.get("xfer_mode", "real"), bs=d.get("batch"),
                job=d.get("slurm_job_id", "?"))


def group(prefix):
    g = {}
    for p in sorted(glob.glob(os.path.join(D, prefix + "_c*.json"))):
        r = fit(p)
        if r:
            g.setdefault(r["ctx"], []).append(r)
    return g


def main():
    nb, rb = group("NB"), group("RB")
    if not nb:
        print("no arms in %s" % D)
        return 1
    jobs = {r["job"] for g in (nb, rb) for v in g.values() for r in v}
    print("all arms from slurm job(s): %s   rider range r in %s"
          % (", ".join(sorted(jobs)), list(RS)))
    if len(jobs) > 1:
        print("REFUSING: arms span more than one job.")
        return 1
    print()

    print("BYTES REMOVED -- the knee is gone, so every point is below it")
    print("%7s %4s %9s %9s %9s %9s"
          % ("ctx", "n", "c_host", "sd", "worst l/T0", "F"))
    print("-" * 56)
    means = {}
    for ctx in sorted(nb):
        v = [r["c"] for r in nb[ctx]]
        means[ctx] = (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v))
        print("%7d %4d %9.3f %9.3f %9.3f %9.2f"
              % (ctx, len(v), st.mean(v),
                 st.stdev(v) if len(v) > 1 else 0.0,
                 max(r["worst"] for r in nb[ctx]),
                 st.mean(r["F"] for r in nb[ctx])))
    print()

    if rb:
        print("BYTES SENT, at the two contexts where they are still fittable")
        print("%7s %4s %9s %9s %9s" % ("ctx", "n", "c", "sd", "worst l/T0"))
        print("-" * 46)
        for ctx in sorted(rb):
            v = [r["c"] for r in rb[ctx]]
            print("%7d %4d %9.3f %9.3f %9.3f"
                  % (ctx, len(v), st.mean(v),
                     st.stdev(v) if len(v) > 1 else 0.0,
                     max(r["worst"] for r in rb[ctx])))
        print()

    lo, hi = min(means), max(means)
    m1, s1, n1 = means[lo]
    m2, s2, n2 = means[hi]
    se = (s1 ** 2 / max(n1, 1) + s2 ** 2 / max(n2, 1)) ** 0.5
    print("=" * 64)
    print(" c_host across %dx of context: %d -> %d tokens" % (hi // lo, lo, hi))
    print("   %.3f -> %.3f ms   = %+.1f%%   (+-%.3f on the difference)"
          % (m1, m2, (m2 / m1 - 1) * 100, se))
    print()
    print(" the withdrawn context law would need %+.0f%% over this range"
          % (((0.205 + 1.84e-3 * hi) / (0.205 + 1.84e-3 * lo) - 1) * 100))
    print()
    if abs(m2 - m1) < 2 * se:
        print(" => NOT RESOLVABLE over %dx of context. c's host component is FLAT"
              % (hi // lo))
        print("    to this machine's precision, tested at %d tokens rather than" % hi)
        print("    assumed there. Table 2's use of a low-context c is supported")
        print("    for the part of c that survives below the knee.")
    else:
        print(" => RESOLVABLE: c_host carries a context term of %+.0f%% over %dx."
              % ((m2 / m1 - 1) * 100, hi // lo))
        print("    Per doubling that is %+.0f%%. Table 2 must charge c at the"
              % (((m2 / m1) ** (1.0 / (hi / lo).bit_length()) - 1) * 100))
        print("    context it evaluates, not at the context it was measured.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
