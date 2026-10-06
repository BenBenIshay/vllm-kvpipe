#!/usr/bin/env python3
"""M19: how much of `F` could fusing the two attention calls actually remove?

`F` is the binding constraint and ~56% of it at L=64 is the per-layer SECOND
attention pass over the rider cohort. Removing it means Solution 1 step 3 -- fuse
it into the resident call -- which is days of work in the attention backend. The
+10 points of projected gain assumes fusing REMOVES that cost rather than
relocating it, and nothing measured said so.

So this prices it by ablation instead, from two arms that bound it from opposite
sides:

    noattend   gather + kernel + scatter all skipped.  UPPER BOUND: a fused call
               cannot beat not doing the work at all.
    nokernel   only the flash call skipped, gather and scatter kept.

    base - noattend        the whole second pass
    noattend - nokernel    the data movement a fused call removes
    base - nokernel        kernel + launch, of which only the launch is
                           recoverable -- the arithmetic still has to happen

THREE base arms, interleaved with the ablations, because F's within-job drift has
been 0.04 / 0.43 / 0.04 / 0.51 ms across four jobs on identical configuration. The
effect is compared against drift measured IN THIS JOB, never borrowed.

    ./.venv/bin/python paper/model/fusion_bound.py
"""
import json
import os
import sys

D = os.environ.get(
    "FB_DIR",
    "/mnt/beegfs/benishay/paper_runs/20260930/overhead/21_fusion_bound")

# The measured split of F at L=64 that this experiment is testing (M18).
REM_MEASURED = 1.94          # the second attention call, ~56% of F
# Build only if at least this much of it is actually recoverable.
GATE_MS = 1.2


def fit(tag):
    """Fit dT = F + c*r over the rows strictly BELOW the knee."""
    p = os.path.join(D, tag + ".json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
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
    F = (sy - c * sx) / n
    return dict(F=F, c=c, n=n, t0=t0,
                mode=d.get("attend_mode", "?"),
                valid=d.get("numerically_valid"),
                head=d.get("git_head", "?"), job=d.get("slurm_job_id", "?"))


def main():
    print("=" * 78)
    print(" M19  THE FUSION BOUND   (ctx 128, L=64, below the knee)")
    print("=" * 78)
    print()

    arms = {}
    print(" %-12s %-10s %7s %8s %5s %6s  %s"
          % ("arm", "mode", "F ms", "c ms/r", "pts", "valid", "note"))
    for tag, want in (("FB_base_a", "full"), ("FB_noattend", "noattend"),
                      ("FB_base_b", "full"), ("FB_nokernel", "nokernel"),
                      ("FB_base_c", "full")):
        r = fit(tag)
        if not r:
            print(" %-12s %-10s %7s" % (tag, want, "(pending)"))
            continue
        bad = "" if r["mode"] == want else "MODE MISMATCH, discard"
        arms[tag] = r
        print(" %-12s %-10s %7.2f %8.3f %5d %6s  %s"
              % (tag, r["mode"], r["F"], r["c"], r["n"],
                 str(r["valid"]), bad))
    print()

    # ALL ARMS MUST COME FROM ONE SLURM JOB. The arms share an output directory,
    # so a re-run overwrites them one at a time and the directory is a MIXTURE
    # until it finishes. Comparing across that mixture is the exact error the
    # campaign's own rule forbids -- F's between-job offset (1.78 ms) is larger
    # than its within-job drift (0.58) -- and it would be invisible in the
    # numbers, which is why it is checked rather than remembered.
    jobs = {t: r["job"] for t, r in arms.items()}
    if len(set(jobs.values())) > 1:
        print(" REFUSING TO ANALYSE: arms come from %d different SLURM jobs."
              % len(set(jobs.values())))
        for t, j in sorted(jobs.items()):
            print("     %-12s job %s" % (t, j))
        print()
        print(" The output directory is shared, so a re-run overwrites arms one")
        print(" at a time. Wait for the job to finish, then re-run this.")
        return 1

    bases = [arms[t]["F"] for t in ("FB_base_a", "FB_base_b", "FB_base_c")
             if t in arms]
    if len(bases) < 2:
        print(" need at least two base arms before anything can be concluded")
        return 0
    Fb = sum(bases) / len(bases)
    drift = max(bases) - min(bases)
    print(" base F = %.2f ms   (%d arms, drift %.2f)"
          % (Fb, len(bases), drift))
    print()

    if "FB_noattend" not in arms or "FB_nokernel" not in arms:
        print(" ablation arms still pending")
        return 0

    noatt, nokern = arms["FB_noattend"]["F"], arms["FB_nokernel"]["F"]
    cb = sum(arms[t]["c"] for t in ("FB_base_a", "FB_base_b", "FB_base_c")
             if t in arms) / len(bases)
    c_noatt, c_nokern = arms["FB_noattend"]["c"], arms["FB_nokernel"]["c"]

    whole = Fb - noatt            # the entire second pass, as it lands in F
    in_F_kernel = Fb - nokern     # kernel + its per-layer launch
    in_c_gather = cb - c_noatt    # gather + scatter -- PER RIDER, so it is in c

    # THE SPLIT IS ACROSS BOTH COEFFICIENTS, and that is the finding. The gather
    # and the scatter size with the rider count, so they land in `c`, not in `F`.
    # Subtracting them inside F therefore reads as zero -- which it does below --
    # and an F-only decomposition cannot see them at all.
    print(" WHERE THE SECOND ATTENTION PASS ACTUALLY LIVES")
    print()
    print(" %-46s %9s" % ("in F (fixed, per step)", "ms"))
    print(" %-46s %9.2f" % ("  whole pass        base - noattend", whole))
    print(" %-46s %9.2f" % ("  kernel + launch   base - nokernel", in_F_kernel))
    print(" %-46s %9.2f" % ("  gather + scatter  (residual in F)",
                            noatt - nokern))
    print()
    print(" %-46s %9s" % ("in c (per rider)", "ms/rider"))
    print(" %-46s %9.3f" % ("  gather + scatter  base - noattend", in_c_gather))
    print(" %-46s %9.3f" % ("  kernel only       base - nokernel",
                            cb - c_nokern))
    print()
    print(" So the gather and the scatter are PER-RIDER work and sit in c, where")
    print(" removing them is worth %.3f ms/rider. The F term is the kernel and" % in_c_gather)
    print(" its per-layer launch. Base drift this job: F %.2f, c %.3f."
          % (drift, max(arms[t]["c"] for t in ("FB_base_a", "FB_base_b",
                                               "FB_base_c") if t in arms)
             - min(arms[t]["c"] for t in ("FB_base_a", "FB_base_b", "FB_base_c")
                   if t in arms)))
    print()
    print(" M18 inferred the remainder at %.2f ms. Measured directly it is %.2f,"
          % (REM_MEASURED, whole))
    print(" so that inference was high by %.0f%% -- one more reason to ablate"
          % ((REM_MEASURED / whole - 1) * 100 if whole else 0))
    print(" rather than subtract.")
    print()

    # ---- what fusion would actually be worth ----
    T0, B, KNEE, L = 24.2, 7.2, 3.78, 80
    C_NOW = 0.609

    def gain(F64, c):
        return (1 + KNEE / B) / (1 + (F64 / 64 * L + c * KNEE) / T0)

    now = gain(Fb, C_NOW)
    # FLOOR: only the per-rider gather/scatter goes. The kernel's arithmetic
    # survives fusion, and we have not separated its launch from it.
    floor = gain(Fb, C_NOW - in_c_gather)
    # CEILING: the whole F term goes too, i.e. the fused call's marginal cost is
    # zero. It is not, so this is unreachable by construction.
    ceil = gain(noatt, C_NOW - in_c_gather)

    print("=" * 78)
    print(" WHAT FUSING WOULD BE WORTH, at the GH200 target (L=80, ctx 8192)")
    print("=" * 78)
    print("   today                                    %.3fx" % now)
    print("   + per-rider gather/scatter removed       %.3fx   (+%.1f pts)"
          % (floor, (floor - now) * 100))
    print("   + the whole F term as well (CEILING)     %.3fx   (+%.1f pts)"
          % (ceil, (ceil - now) * 100))
    print()
    print(" The middle line is what fusing is WORTH ON THE EVIDENCE. The bottom")
    print(" line requires the fused call to cost nothing, which it cannot.")
    print()

    resolvable = whole > 2 * drift
    if not resolvable:
        print(" INCONCLUSIVE: %.2f ms is within 2x the drift (%.2f). Do not act."
              % (whole, drift))
    elif (floor - now) * 100 >= 5.0:
        print(" BUILD IT: the defensible saving alone is +%.1f points."
              % ((floor - now) * 100))
    else:
        print(" DO NOT BUILD IT YET, and the reason is not the size of the")
        print(" ceiling but its composition. The part fusion certainly removes --")
        print(" the per-rider gather and scatter -- is worth +%.1f points. The"
              % ((floor - now) * 100))
        print(" rest sits in a %.2f ms term that is kernel PLUS launch, and a"
              % in_F_kernel)
        print(" fused call still performs the attention arithmetic. Nothing here")
        print(" bounds the launch share, so the true prize is somewhere in")
        print(" +%.1f to +%.1f points for days of attention-backend work."
              % ((floor - now) * 100, (ceil - now) * 100))
        print()
        print(" ONE CHEAP MEASUREMENT CLOSES THIS: price one varlen call against")
        print(" two at equal total attention work, off-engine, in")
        print(" launch_pattern_bench. That separates launch from kernel and turns")
        print(" the range above into a number. Hours, no engine, no allocation.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
