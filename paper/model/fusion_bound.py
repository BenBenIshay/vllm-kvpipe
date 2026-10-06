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
    whole = Fb - noatt            # the entire second pass
    movement = noatt - nokern     # gather + scatter (negative if nokernel < )
    kernel = Fb - nokern          # kernel + its launch

    print(" %-44s %8s" % ("component", "ms"))
    print(" %-44s %8.2f" % ("the whole second pass (base - noattend)", whole))
    print(" %-44s %8.2f" % ("  of it: gather + scatter", movement))
    print(" %-44s %8.2f" % ("  of it: kernel + launch", kernel))
    print()
    print(" against this job's own base drift of %.2f ms" % drift)
    print(" and M18's independent estimate of the remainder, %.2f ms"
          % REM_MEASURED)
    print()

    # ---- the decision ----
    resolvable = whole > 2 * drift
    passes = whole >= GATE_MS and resolvable
    print("=" * 78)
    if passes:
        print(" GATE PASSED: %.2f ms recoverable, %.1fx this job's drift."
              % (whole, whole / drift if drift else float("inf")))
        print(" Fusing is worth building -- but note the UPPER BOUND is what was")
        print(" measured. A fused call still performs the attention arithmetic,")
        print(" so the realistic prize is nearer the %.2f ms of gather/scatter"
              % movement)
        print(" plus the launch share of the %.2f ms kernel term." % kernel)
    elif not resolvable:
        print(" INCONCLUSIVE: the effect (%.2f ms) is within 2x the drift (%.2f)."
              % (whole, drift))
        print(" Do NOT build on this. Re-run with more base arms first.")
    else:
        print(" GATE FAILED: only %.2f ms is recoverable, under the %.1f ms bar."
              % (whole, GATE_MS))
        print(" Solution 1 step 3 is NOT worth days of attention-backend work,")
        print(" and the +10 point projection should be retracted rather than")
        print(" shipped.")
    print("=" * 78)

    # What it would be worth at the target, if the whole bound were realised.
    T0, B, C, KNEE, L = 24.2, 7.2, 0.609, 3.78, 80

    def gain(F64):
        return (1 + KNEE / B) / (1 + (F64 / 64 * L + C * KNEE) / T0)

    print()
    print(" at the GH200 target (L=80, ctx 8192, C2C), if fully realised:")
    print("   today            F@64 %.2f -> %.3fx" % (Fb, gain(Fb)))
    print("   fusion ceiling   F@64 %.2f -> %.3fx" % (noatt, gain(noatt)))
    print(" The second line is a CEILING, not a projection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
