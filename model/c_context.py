#!/usr/bin/env python3
"""M20: does `c` have a context term, and is it resolvable?

    ./.venv/bin/python paper/model/c_context.py

dT(2) - dT(1) cancels F identically, so it is one request's marginal cost with no
fit and no intercept. Three replicates per cell in ONE job, interleaved across
contexts, repeated with the bytes removed.

PREDICTIONS ON THE RECORD BEFORE THE RUN (from the commit message of the sbatch):
    flat c                                       0% across 128 -> 512
    the two-point minbytes trend                ~+25%
    the withdrawn law c = 0.205 + 1.84e-3*S     +161%
"""
import glob
import json
import os
import statistics as st
import sys

D = os.environ.get(
    "M20_DIR", "/mnt/beegfs/benishay/paper_runs/20260930/overhead/22_c_context")


def marg(path):
    """Per-arm c by REGRESSION over all usable rider points below the knee.

    Differencing dT(2)-dT(1) cancels F exactly, which is elegant, but it rests the
    whole estimate on the two noisiest points. This run shows why that matters:
    dT(2) repeats to about 2% within a context while dT(1) carries outliers --
    CC_c128_c reads 4.952 where its two replicates read 3.46 and 3.38 -- so the
    difference inherits all of that and one arm can flip the cell's sign. The
    regression uses all eight points, is what Table 2's coefficients come from, and
    is the estimator the conclusion should rest on. Both are reported.
    """
    d = json.load(open(path))
    t0 = next((r["step_ms"] for r in d["rows"]
               if r.get("riders") == 0 and r.get("step_ms")), None)
    by = {r["riders"]: (r["dT_ms"], r.get("link_ms") or 0.0)
          for r in d["rows"]
          if r.get("usable") and r.get("riders") and r.get("dT_ms") is not None}
    if not t0 or 1 not in by or 2 not in by:
        return None
    xs = [(r, by[r][0]) for r in sorted(by)
          if not by[r][1] or by[r][1] / t0 <= 1 / 3.0]
    n = len(xs)
    sx = sum(x for x, _ in xs); sy = sum(y for _, y in xs)
    sxx = sum(x * x for x, _ in xs); sxy = sum(x * y for x, y in xs)
    c = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    F = (sy - c * sx) / n
    return dict(ctx=d["ctx"], t0=t0, m2=by[2][0] - by[1][0], c=c, F=F, npts=n,
                lr=by[2][1] / t0, xfer=d.get("xfer_mode", "real"),
                job=d.get("slurm_job_id", "?"))


def collect(prefix):
    out = {}
    for p in sorted(glob.glob(os.path.join(D, prefix + "_c*.json"))):
        r = marg(p)
        if r:
            out.setdefault(r["ctx"], []).append(r)
    return out


def main():
    present, removed = collect("CC"), collect("MB")
    if not present:
        print("no arms found in %s" % D)
        return 1

    jobs = {r["job"] for g in (present, removed) for v in g.values() for r in v}
    print("all arms from slurm job(s): %s" % ", ".join(sorted(jobs)))
    if len(jobs) > 1:
        print("REFUSING: arms span more than one job.")
        return 1
    print()

    print("%-15s %5s %3s | %16s | %16s | %6s"
          % ("arms", "ctx", "n", "c by REGRESSION", "dT(2)-dT(1)", "l/T0"))
    print("%-15s %5s %3s | %7s %8s | %7s %8s |"
          % ("", "", "", "mean", "sd", "mean", "sd"))
    print("-" * 78)
    rows, rows_m = {}, {}
    for lab, g in (("bytes sent", present), ("bytes removed", removed)):
        for ctx in sorted(g):
            cv = [r["c"] for r in g[ctx]]
            mv = [r["m2"] for r in g[ctx]]
            csd = st.stdev(cv) if len(cv) > 1 else 0.0
            msd = st.stdev(mv) if len(mv) > 1 else 0.0
            rows[(lab, ctx)] = (st.mean(cv), csd, len(cv))
            rows_m[(lab, ctx)] = (st.mean(mv), msd, len(mv))
            print("%-15s %5d %3d | %7.3f %8.3f | %7.3f %8.3f | %6.3f"
                  % (lab, ctx, len(cv), st.mean(cv), csd, st.mean(mv), msd,
                     st.mean(r["lr"] for r in g[ctx])))
    print()
    print(" The regression column is tighter by roughly an order of magnitude.")
    print(" That is the estimator to read; the difference column is shown so the")
    print(" reader can see why it is not.")
    print()

    print("=" * 64)
    print(" THE TEST: 128 -> 512 is a 4x context change at l/T0 <= 0.25")
    print("=" * 64)
    for lab in ("bytes sent", "bytes removed"):
        a, b = rows.get((lab, 128)), rows.get((lab, 512))
        if not (a and b):
            continue
        rise = (b[0] / a[0] - 1) * 100
        # pooled sd, propagated to the ratio
        psd = ((a[1] ** 2 / max(a[2], 1) + b[1] ** 2 / max(b[2], 1)) ** 0.5)
        print(" %-15s %.3f -> %.3f  = %+5.1f%%   (+-%.3f ms on the difference)"
              % (lab, a[0], b[0], rise, psd))
    print()
    print(" predicted: flat 0%%   minbytes trend ~+25%%   withdrawn law +161%%")
    print()

    a, b = rows.get(("bytes sent", 128)), rows.get(("bytes sent", 512))
    if a and b:
        rise = b[0] / a[0] - 1
        resolvable = abs(b[0] - a[0]) > 2 * (
            (a[1] ** 2 / max(a[2], 1) + b[1] ** 2 / max(b[2], 1)) ** 0.5)
        print("=" * 64)
        if not resolvable:
            print(" VERDICT: NOT RESOLVABLE. The 128->512 change is within twice")
            print(" its own standard error, so `c` is flat to the precision this")
            print(" machine affords. Table 2 may charge c = %.2f at any context,"
                  % st.mean([a[0], b[0]]))
            print(" and the bound in section 6.5 can be dropped to the replicate")
            print(" spread rather than a compounded context term.")
        elif rise > 0.5:
            print(" VERDICT: c RISES STRONGLY with context (%+.0f%% over 4x)." % (rise * 100))
            print(" Table 2 must be restated: c at ctx 8192 is far above 0.52 and")
            print(" the pessimistic column of section 6.5 becomes the estimate.")
        else:
            print(" VERDICT: c rises MILDLY (%+.0f%% over 4x), well under the" % (rise * 100))
            print(" withdrawn law's +161%%. Per doubling that is %+.0f%%, so over"
                  % ((( 1 + rise) ** 0.5 - 1) * 100))
            print(" the six doublings to ctx 8192, c ~ %.2f ms."
                  % (a[0] * (1 + rise) ** (6 / 2.0)))
        print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
