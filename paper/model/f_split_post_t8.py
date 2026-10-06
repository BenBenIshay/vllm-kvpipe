#!/usr/bin/env python3
"""`F`'s split after T8, and what is left to go after (M18).

M14 split `F` at L=64 into mirror write 3.95 (68%) and a remainder of 1.85 that
survives with both mechanisms off. That was measured BEFORE T8, so the 3.95
included the waste T8 removed -- a disabled feature flushing the deferred mirror
writes on every layer, 64 times per step, each with one pending entry and each
synchronising the D2H event.

With `F` now 3.27 the old split cannot hold, and the ranking of what to do next
depends on the new one. Inferring it (3.27 - 1.85 = 1.42, assuming the remainder
is untouched) is plausible but is the kind of step this campaign has been
punished for, so it is measured.

    FS_base    as shipped                     -> F
    FS_nod2h   SKIP_D2H=1                     -> F(base) - F(nod2h) = mirror
    FS_both    SKIP_D2H=1 + nobus             -> the remainder

    ./.venv/bin/python paper/model/f_split_post_t8.py
"""
import json
import os
import sys

D = os.environ.get(
    "FS_BASE",
    "/mnt/beegfs/benishay/paper_runs/20260930/overhead/20_f_split_post_t8")
# M14, same configuration, before T8
M14 = {"F": 5.80, "mirror": 3.95, "remainder": 1.85}
# launch-only cost of the 2L D2H copies, launch_pattern_bench on A100
LAUNCH_FLOOR_64 = 0.834


def fit(tag):
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
    return (sy - c * sx) / n, c, n


def main():
    print("=" * 76)
    print(" F's SPLIT AFTER T8   (M18, ctx 128, L=64, below the knee)")
    print("=" * 76)
    print()
    got = {}
    print(" %-12s %-32s %8s %9s" % ("arm", "live", "F ms", "c ms/r"))
    for tag, lab in (("FS_base", "as shipped"),
                     ("FS_base_b", "as shipped (repeat)"),
                     ("FS_nod2h", "mirror write OFF"),
                     ("FS_both", "both mechanisms OFF")):
        r = fit(tag)
        if r:
            got[tag] = r
            print(" %-12s %-32s %8.2f %9.3f" % (tag, lab, r[0], r[1]))
        else:
            print(" %-12s %-32s  (pending)" % (tag, lab))
    print()

    bases = [got[t][0] for t in ("FS_base", "FS_base_b") if t in got]
    if not bases:
        return 0
    Fb = sum(bases) / len(bases)
    drift = (max(bases) - min(bases)) if len(bases) > 1 else None
    print(" F as shipped: %.2f ms%s" % (Fb, "" if drift is None
                                        else "   (drift %.2f)" % drift))
    if "FS_nod2h" not in got or "FS_both" not in got:
        print(" ablation arms still pending")
        return 0
    Fd, Fo = got["FS_nod2h"][0], got["FS_both"][0]
    mirror, rem = Fb - Fd, Fo
    print()
    print(" %-26s %8s %8s %10s" % ("component", "M14", "now", "change"))
    print(" %-26s %8.2f %8.2f %+9.2f"
          % ("F", M14["F"], Fb, Fb - M14["F"]))
    print(" %-26s %8.2f %8.2f %+9.2f"
          % ("mirror write", M14["mirror"], mirror, mirror - M14["mirror"]))
    print(" %-26s %8.2f %8.2f %+9.2f"
          % ("remainder (attention)", M14["remainder"], rem,
             rem - M14["remainder"]))
    print()
    print(" so T8+T7a took %.2f ms off the MIRROR WRITE and %+.2f off the"
          % (M14["mirror"] - mirror, rem - M14["remainder"]))
    print(" remainder -- which is what should happen, since both changes act on")
    print(" the mirror write and neither touches the attention path.")
    print()
    print(" SHARE OF F NOW: mirror %.0f%%, remainder %.0f%%"
          % (mirror / Fb * 100, rem / Fb * 100))
    print()

    # what is left, and what each would be worth at the target
    T0, B, C, KNEE, L = 24.2, 7.2, 0.609, 3.78, 80

    def gain(F64):
        return (1 + KNEE / B) / (1 + (F64 / 64 * L + C * KNEE) / T0)

    floor_mirror = LAUNCH_FLOOR_64
    print(" WHAT IS LEFT, at the GH200 target (L=80, ctx 8192, C2C):")
    print(" %-40s %8s %9s" % ("state", "F@L=80", "gain"))
    for lab, F64 in (
            ("today", Fb),
            ("fuse the attention calls (remainder -> 0)", Fb - rem),
            ("+ mirror write at its launch floor",
             max(floor_mirror, Fb - rem - max(0.0, mirror - floor_mirror)))):
        print(" %-40s %8.2f %8.3fx" % (lab, F64 / 64 * L, gain(F64)))
    print()
    print(" The remainder is the per-layer attention override -- the second")
    print(" attention call over the rider window. Removing it is Solution 1")
    print(" step 3 (fuse to one unified attention call), which needs the rider")
    print(" cohort's block table rebuilt per layer and probably its own")
    print(" KV-cache group: days of work in the attention backend, not a")
    print(" contained change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
