#!/usr/bin/env python3
"""Where to run the feature: the operating point, under the FINAL coefficients.

THE FINDING THIS FILE EXISTS FOR. Riders are integers, so throughput gain is a
SAWTOOTH in context, not a smooth curve:

    knee  r*(S) = T0 / link_ms(S)        falls as 1/S
    riders admitted = floor(r*)          <-- the step
    gain = (1 + r/B) / (1 + (F + c*r)/T0)

`gain` therefore peaks JUST BELOW every integer crossing of the knee and drops
by a whole rider's worth just above it. `best_model.py` originally credited the
fractional knee (`r = min(knee, 64)`), which smooths the sawtooth away and
systematically overstates any context sitting just above a crossing. That defect
is what selected ctx 16384 as the target: its knee is 1.98, so it admits ONE
rider and yields 1.12x, while ctx ~16200 has a knee of 2.00, admits TWO, and
yields 1.37x. A 1.2% reduction in context is worth +22% in gain.

    ./.venv/bin/python paper/model/operating_point.py
    ./.venv/bin/python paper/model/operating_point.py --selftest

THE CAVEAT THAT MATTERS MORE THAN THE FINDING. A crossing's location is computed
from `T0`, which is MODELLED (weight-read-bound) and has never been measured on a
GH200. A 2% overestimate of T0 moves the crossing below your chosen context and
you lose the rider -- 1.37x collapses to 1.11x. So the recommendation is NOT to
sit 1% below a cliff. It is to leave margin (target a knee of ~2.2, not 2.00) and
to MEASURE T0 before trusting any crossing. `margin_point()` does the former.
"""
import argparse
import math
import sys

from overhead_model import (FINAL, GH200, L70, Q32, Q72, final_F,  # noqa: E402
                            final_c)

MODELS = [("Qwen-72B fp8", Q72), ("Llama-70B fp8", L70), ("Qwen-32B fp16", Q32)]

# `F` variants, as ms at L=64, from the measured decomposition
F64 = {
    "today": final_F(64),                       # 3.39
    "fused": FINAL["f_mirror_write"][0],        # attention remainder removed
    "both": FINAL["f_mirror_launch_floor"][0],  # and the mirror write at floor
}
# A configuration with fewer than this many resident slots is degenerate: b_cap()
# floors at 1.0, which silently turns "does not fit with any KV" into "one fits"
# and inflates (1 + r/B) without bound.
MIN_RESIDENTS = 2.0


def evaluate(m, hw, S, variant="today", t0_scale=1.0):
    """The operating point at one context. `t0_scale` probes T0 error."""
    t0 = m.t0_ms(hw) * t0_scale
    cap = (hw.hbm_gb - m.w - 4.0) / (m.b_tok * S / 1e9)
    link_ms = m.b_tok * S / (hw.link * 1e9) * 1e3
    knee = t0 / link_ms
    r = int(math.floor(knee))
    c = final_c(m.L)
    F = F64[variant] / 64.0 * m.L
    budget = t0 / cap
    r_be = F / (budget - c) if budget > c else None
    valid = r >= 1 and cap >= MIN_RESIDENTS
    gain = ((1.0 + r / cap) / (1.0 + (F + c * r) / t0)) if r >= 1 \
        else float("nan")
    return dict(S=S, t0=t0, cap=cap, knee=knee, r=r, c=c, F=F, budget=budget,
                r_be=r_be, gain=gain, valid=valid,
                itl=t0 + F + c * r)


def crossing(m, hw, riders):
    """Largest context whose knee still admits `riders`. The sawtooth peak."""
    t0 = m.t0_ms(hw)
    S = int(t0 * (hw.link * 1e9) / (riders * m.b_tok * 1e3))
    while evaluate(m, hw, S)["r"] < riders and S > 1:
        S -= 1
    return S


def margin_point(m, hw, riders, knee_target=None):
    """The peak, pulled back so a T0 error does not cost the rider.

    `knee_target` defaults to riders * 1.1, i.e. 10% of headroom, which absorbs
    the ~2% that was enough to lose a rider at the bare crossing.
    """
    kt = knee_target if knee_target is not None else riders * 1.1
    t0 = m.t0_ms(hw)
    S = int(t0 * (hw.link * 1e9) / (kt * m.b_tok * 1e3))
    return S


def peaks(m, hw, variant="today", upto=8):
    out = []
    for riders in range(1, upto + 1):
        S = crossing(m, hw, riders)
        d = evaluate(m, hw, S, variant)
        if d["r"] == riders:
            out.append(d)
    return out


def report():
    print("=" * 78)
    print(" OPERATING POINT under the FINAL coefficients   (GH200 / NVLink-C2C)")
    print("=" * 78)
    print()
    print(" c(L) = %.3f + %.5f*L     F(L) = %.3f*L     F carries +-0.5 at L=64"
          % (FINAL["c_intercept"][0], FINAL["c_per_layer"][0],
             FINAL["f_per_layer_post_t8"][0]))
    print()

    for name, m in MODELS:
        print(" %s -- the sawtooth peaks (gain just below each knee crossing)"
              % name)
        print(" %7s %8s %6s %6s %8s %8s  %s"
              % ("riders", "ctx", "knee", "B", "gain", "ITL ms", "note"))
        for d in peaks(m, GH200):
            note = "" if d["valid"] else "DEGENERATE: B < %.0f" % MIN_RESIDENTS
            print(" %7d %8d %6.2f %6.2f %7.3fx %8.1f  %s"
                  % (d["r"], d["S"], d["knee"], d["cap"], d["gain"], d["itl"],
                     note))
        print()

    print("=" * 78)
    print(" THE COMMITTED TARGET SITS IN THE WORST PLACE (Qwen-72B)")
    print("=" * 78)
    print()
    print(" %8s %6s %4s %6s %8s  %s"
          % ("ctx", "knee", "r", "B", "gain", "what this is"))
    for S, what in ((16384, "the committed TARGET -- knee 1.98, just UNDER 2"),
                    (16196, "the r=2 crossing -- 1.2% less context"),
                    (14724, "r=2 with 10% knee margin (recommended)"),
                    (8192, "the previous anchor"),
                    (8098, "the r=4 crossing")):
        d = evaluate(Q72, GH200, S)
        print(" %8d %6.2f %4d %6.2f %7.3fx  %s"
              % (S, d["knee"], d["r"], d["cap"], d["gain"], what))
    print()

    print("=" * 78)
    print(" FRAGILITY: T0 is MODELLED, and a 2% error costs a whole rider")
    print("=" * 78)
    print()
    print(" %-28s %7s %4s %8s" % ("", "knee", "r", "gain"))
    for S, lab in ((16196, "at the bare crossing"),
                   (14724, "with 10%% margin")):
        for err in (0.0, -0.02, -0.05, -0.10):
            d = evaluate(Q72, GH200, S, t0_scale=1 + err)
            tag = "%s, T0 %+.0f%%" % (lab, err * 100)
            print(" %-28s %7.2f %4d %7.3fx" % (tag, d["knee"], d["r"],
                                               d["gain"]))
        print()

    print("=" * 78)
    print(" WHAT THE REMAINING OPTIMISATIONS ADD, at the recommended point")
    print("=" * 78)
    print()
    print(" %-15s %8s %8s %8s %8s"
          % ("model", "ctx", "today", "fused", "both"))
    for name, m in MODELS:
        S = margin_point(m, GH200, 2)
        g = [evaluate(m, GH200, S, v)["gain"] for v in ("today", "fused", "both")]
        print(" %-15s %8d %7.3fx %7.3fx %7.3fx" % (name, S, *g))
    print()
    print(" 'fused' removes the per-layer second attention call (Solution 1 step")
    print(" 3); 'both' also puts the mirror write at its measured launch floor.")
    print(" Neither is built. The fused figure assumes the cost is REMOVED rather")
    print(" than relocated, which is exactly what experiment E2 must test first.")


def selftest():
    ok = True

    # The defect: crediting the fractional knee overstates ctx 16384 badly.
    d = evaluate(Q72, GH200, 16384)
    frac = ((1 + d["knee"] / d["cap"])
            / (1 + (d["F"] + d["c"] * d["knee"]) / d["t0"]))
    good = d["r"] == 1 and frac > d["gain"] * 1.15
    ok &= good
    print("  [%s] ctx 16384 admits 1 rider for %.3fx; the fractional knee "
          "(%.2f) would claim %.3fx" % ("ok " if good else "BAD", d["gain"],
                                        d["knee"], frac))

    # The finding: a slightly SMALLER context is worth a lot more.
    better = evaluate(Q72, GH200, crossing(Q72, GH200, 2))
    good = better["r"] == 2 and better["gain"] > d["gain"] * 1.15
    ok &= good
    print("  [%s] ctx %d (1.2%% less) admits %d riders for %.3fx, +%.0f%%"
          % ("ok " if good else "BAD", better["S"], better["r"], better["gain"],
             (better["gain"] / d["gain"] - 1) * 100))

    # The caveat: the bare crossing is fragile to a small T0 error.
    frag = evaluate(Q72, GH200, better["S"], t0_scale=0.98)
    good = frag["r"] == 1
    ok &= good
    print("  [%s] and a 2%% T0 overestimate drops it back to %d rider(s), "
          "%.3fx -- so the bare crossing is NOT a safe operating point"
          % ("ok " if good else "BAD", frag["r"], frag["gain"]))

    # The margin point holds the rider through a 9% T0 error.
    S = margin_point(Q72, GH200, 2)
    held = all(evaluate(Q72, GH200, S, t0_scale=1 + e)["r"] == 2
               for e in (0.0, -0.02, -0.05, -0.09))
    good = held and evaluate(Q72, GH200, S)["gain"] > d["gain"] * 1.1
    ok &= good
    print("  [%s] the 10%%-margin point ctx %d keeps 2 riders through a -9%% T0 "
          "error, at %.3fx" % ("ok " if good else "BAD", S,
                               evaluate(Q72, GH200, S)["gain"]))

    # Short context genuinely loses TODAY, and by reachability, not by the gate.
    sc = evaluate(L70, GH200, 2048)
    good = (sc["gain"] < 1.0 and sc["c"] < sc["budget"]
            and sc["r_be"] > sc["knee"])
    ok &= good
    print("  [%s] ctx 2048 loses (%.3fx) by REACHABILITY: c %.2f passes the "
          "%.2f budget, but r_be %.1f > knee %.1f"
          % ("ok " if good else "BAD", sc["gain"], sc["c"], sc["budget"],
             sc["r_be"], sc["knee"]))

    # ...and fusion flips its sign, which is why short context is not "dead".
    sf = evaluate(L70, GH200, 2048, "fused")
    good = sf["gain"] > 1.0 and sf["gain"] < evaluate(Q72, GH200, S)["gain"]
    ok &= good
    print("  [%s] fusion flips ctx 2048 to %.3fx -- positive, but still far "
          "below the long-context point" % ("ok " if good else "BAD",
                                            sf["gain"]))

    # THE T0 ERROR BAR CUTS ONE WAY ONLY, and section 3 now says so, so pin it.
    # T0 = W / (HBM_bw * eff) with eff = 0.75 taken from A100 weight reads. If the
    # real GH200 efficiency is lower, T0 RISES, which both raises the knee (more
    # riders admissible) and shrinks F/T0. The lowest GH200 HBM throughput anyone
    # reports is DirectKV's 1.9 of 4 TB/s for a zero-copy attention kernel, i.e.
    # 0.475 -- a pessimistic bound for a dense weight read, not an estimate.
    gains = [evaluate(L70, GH200, 8192, t0_scale=0.75 / e)["gain"]
             for e in (0.75, 0.65, 0.55, 0.475)]
    good = all(b > a for a, b in zip(gains, gains[1:]))
    ok &= good
    print("  [%s] a LOWER HBM efficiency only helps: gain %s as eff goes "
          "0.75 -> 0.475, so 0.75 is the conservative end"
          % ("ok " if good else "BAD",
             " -> ".join("%.3f" % g for g in gains)))

    # And the knee here is computed on the WEIGHT-READ-bound step, which omits the
    # resident cohort's KV read, so it is a lower bound on the hideable window.
    # killer_v2 charges the full step and therefore admits more riders; section 3
    # states that these counts are the conservative ones. Guard the direction.
    import killer_v2 as K  # noqa: E402
    m70 = [x for x in K.MODELS if x.n.strip() == "Llama-70B fp8"][0]
    full_t0 = K.analyze(m70, 8192, K.GH200M, slo_ms=80.0, stage=2.0)["T0"]
    good = full_t0 > L70.t0_ms(GH200)
    ok &= good
    print("  [%s] our T0 (%.1f ms) is BELOW the full-step T0 (%.1f ms), so the "
          "rider counts are conservative"
          % ("ok " if good else "BAD", L70.t0_ms(GH200), full_t0))

    # Qwen-72B remains the best of the three, which the retarget got right.
    at = {n: evaluate(m, GH200, margin_point(m, GH200, 2))["gain"]
          for n, m in MODELS}
    good = (at["Qwen-72B fp8"] > at["Llama-70B fp8"]
            and at["Qwen-72B fp8"] > at["Qwen-32B fp16"])
    ok &= good
    print("  [%s] Qwen-72B still wins: %.3fx vs %.3fx (L70) and %.3fx (Q32)"
          % ("ok " if good else "BAD", at["Qwen-72B fp8"],
             at["Llama-70B fp8"], at["Qwen-32B fp16"]))

    print("\nOPERATING POINT SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
