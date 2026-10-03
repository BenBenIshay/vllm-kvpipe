#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The measured overhead law, and what it projects onto NVLink-C2C.

Replaces `killer_v2.f_fixed_ms`, which was a single constant scaled by layer
count:

    F_REF_MS, F_REF_LAYERS = 12.3, 64
    def f_fixed_ms(m): return F_REF_MS * m.L / F_REF_LAYERS

That shape is right and its rate is not. This module carries the decomposition
measured in the 2026-09-30/10-01 campaign, every coefficient with its
provenance, and -- the point of the exercise -- the projection as an INTERVAL
whose dominating uncertainty is named rather than averaged away.

THE LAW (Qwen-32B, A100-80 PCIe, below the knee; 105 points, R2 0.990,
held-out max error 9.3%, fitted within-job):

    dT(r) = F + c * r

    F = L * (d2h_per_layer + async_exposed_per_layer)
          d2h_per_layer      blocking pageable D2H, set by KV GEOMETRY, and it
                             CANNOT hide because it blocks: 0.142 ms/layer at
                             8 kv-heads x 128 head-dim, 0.086 at 2x64.
          async_exposed      host work per layer that DOES hide, in proportion
                             to compute per layer: measured 0.326 / 0.143 /
                             0.026 ms/layer at compute/layer 0.369 / 0.426 /
                             0.772.

    c = b_per_rider_layer * L          invariant   0.00838 ms/rider/layer
      + d_per_block * blocks_per_rider invariant   0.00765 ms/block
      + c_link * nominal_link_ms       LINK-BOUND  0.0744 of the nominal
      + d2h_per_rider                  invariant   0.109 ms/rider

Above the knee the exposed transfer is already inside max(T0, link), so the
residual is F plus only the INVARIANT part of c -- validated at 0.706 (ctx 512)
and 0.853 (ctx 1024) against an invariant share of 0.43-0.60.

WHAT THE PROJECTION TURNS ON, and it is not the link. On GH200 the 70B's
compute per layer is ~0.303 ms -- BELOW every A100 point measured -- because
4000 GB/s HBM reads the weights faster. Hiding requires the host to run AHEAD of
the GPU, so a faster GPU hides LESS. F(70B) then spans 2.1 to 37.4 ms depending
on the hiding fraction and whether the D2H is fixed, against the paper's assumed
15.4. That is an 18x range on F, where the link changes c by 44%.

Two unmeasurables pull opposite ways and neither is knowable from an A100: the
H100-class GPU has less slack per layer (hurts), a Grace CPU may do the host work
faster (helps). So this module reports a band, and names the one measurement that
would close it.

Usage:
    python overhead_model.py              # the projection table
    python overhead_model.py --selftest
"""

from __future__ import annotations

import argparse
import sys

# ----------------------------------------------------------------- provenance
# Every number here is measured, with where it came from. Nothing is assumed.
COEFF = {
    "b_per_rider_layer": (0.00838, "ms/rider/layer", "105-pt within-job fit, "
                          "CI [0.0055, 0.0119]"),
    "d_per_block": (0.00765, "ms/block", "105-pt within-job fit; the layer axis "
                    "independently gives 0.252 ms/rider at ctx 512 vs this "
                    "0.245, agreeing to 3%"),
    "c_link": (0.0744, "fraction of nominal link time", "105-pt within-job "
               "fit; the minbytes arm gives 47% of c link-bound vs 37% here, "
               "so the interval is 37-57%"),
    "d2h_per_rider": (0.109, "ms/rider", "M3 real-minus-skip_d2h on c"),
    "d2h_per_layer_8x128": (0.142, "ms/layer", "M3, Qwen-32B 0.142 and "
                            "Llama-8B 0.138 -- identical despite 1.8x "
                            "different compute per layer, so it tracks KV "
                            "geometry not compute"),
    "d2h_per_layer_2x64": (0.086, "ms/layer", "M3, Qwen-0.5B"),
}

# exposed async host work per layer, against compute per layer. THREE POINTS.
# Deliberately not fitted to a curve: three points cannot distinguish the
# functional form, and the GH200 case sits OUTSIDE this range, so any fit would
# be extrapolating a shape the data does not determine.
HIDING = [
    (0.369, 0.326, "Qwen-0.5B  A100"),
    (0.426, 0.143, "Llama-8B   A100"),
    (0.772, 0.026, "Qwen-32B   A100"),
]

JOB_OFFSET_PCT = 14.0   # between-job spread of F; within-job is 5%


class HW:
    def __init__(s, name, hbm_gbs, link_gbs, hbm_gb, eff=0.75):
        s.name, s.hbm, s.link, s.eff = name, hbm_gbs, link_gbs, eff
        s.hbm_gb = hbm_gb          # capacity, as distinct from bandwidth


A100 = HW("A100-80 PCIe", 1935, 25.1, 80)
# 419 GB/s is the MEASURED H2D, not the 450 spec figure.
GH200 = HW("GH200-96 C2C", 4000, 419, 96)


class Model:
    def __init__(s, name, layers, weight_gb, kv_heads, head_dim, kv_bytes=2):
        s.name, s.L, s.w = name, layers, weight_gb
        s.kv_heads, s.head_dim, s.kv_bytes = kv_heads, head_dim, kv_bytes
        s.b_tok = layers * 2 * kv_heads * head_dim * kv_bytes

    def t0_ms(s, hw):
        """Weight-read-bound step time. The KV term is small at the contexts
        this law was measured at and is omitted deliberately -- adding it would
        import killer_v2's eff_kv curve and couple two models together."""
        return s.w / (hw.hbm * hw.eff) * 1e3

    def compute_per_layer(s, hw):
        return s.t0_ms(hw) / s.L

    def b_cap(s, hw, S, overhead_gb=4.0):
        """Residents that fit, from capacity -- not a chosen constant.

        Using a fixed 16 was wrong and it mattered: on GH200 at ctx 8192 the
        70B's weights leave 19.3 GB, and one request's KV is 2.68 GB, so only
        7.2 residents fit. A rider's worth is T0/B, so understating B overstates
        what each rider has to beat.
        """
        C = hw.hbm_gb - s.w - overhead_gb
        u = s.b_tok * S / 1e9
        return max(1.0, C / u)

    def d2h_per_layer(s):
        key = ("d2h_per_layer_8x128" if (s.kv_heads, s.head_dim) == (8, 128)
               else "d2h_per_layer_2x64")
        return COEFF[key][0]


Q32 = Model("Qwen-32B fp16", 64, 65.53, 8, 128)
L70 = Model("Llama-70B fp8", 80, 72.7, 8, 128)


def F_ms(m: Model, hw: HW, exposed_per_layer: float, d2h_fixed: bool) -> float:
    d2h = 0.0 if d2h_fixed else m.d2h_per_layer()
    return m.L * (d2h + exposed_per_layer)


def c_ms(m: Model, hw: HW, S: int, bs: int, d2h_fixed: bool,
         link_frac: float | None = None) -> dict:
    nominal = m.b_tok * S / (hw.link * 1e9) * 1e3
    frac = COEFF["c_link"][0] if link_frac is None else link_frac
    parts = {
        "per_rider_layer": COEFF["b_per_rider_layer"][0] * m.L,
        "per_block": COEFF["d_per_block"][0] * (S // bs),
        "link": frac * nominal,
        "d2h": 0.0 if d2h_fixed else COEFF["d2h_per_rider"][0],
    }
    parts["total"] = sum(parts.values())
    parts["nominal_link_ms"] = nominal
    return parts


def break_even(m: Model, hw: HW, S: int, b_res: float | None, exposed: float,
               d2h_fixed: bool, bs: int = 16, drop_per_block: bool = False)\
        -> dict:
    if b_res is None:
        b_res = m.b_cap(hw, S)
    t0 = m.t0_ms(hw)
    F = F_ms(m, hw, exposed, d2h_fixed)
    parts = c_ms(m, hw, S, bs, d2h_fixed)
    c = parts["total"] - (parts["per_block"] if drop_per_block else 0.0)
    per_resident = t0 / b_res
    knee = t0 / (m.b_tok * S / (hw.link * 1e9) * 1e3)
    out = {"F": F, "c": c, "t0": t0, "knee": knee, "b_res": b_res,
           "parts": parts}
    if c >= per_resident:
        out.update(r_be=None,
                   verdict="never pays: c=%.2f >= T0/B=%.2f" % (c, per_resident))
        return out
    r_be = F / (per_resident - c)
    out.update(r_be=r_be,
               verdict=("PAYS from r=%.1f" % r_be if r_be <= knee
                        else "break-even %.1f beyond the knee %.1f"
                             % (r_be, knee)))
    return out


def gain(m: Model, hw: HW, S: int, r: float, exposed: float,
         d2h_fixed: bool, b_res: float | None = None, bs: int = 16,
         drop_per_block: bool = False) -> float:
    """Throughput gain from admitting r riders on top of B residents.

        gain = (1 + r/B) / (1 + (F + c*r)/T0)

    Numerator: the extra requests served. Denominator: the longer step they
    cost. Valid only BELOW the knee -- above it the step is the transfer and
    this formula no longer describes the step time.
    """
    if b_res is None:
        b_res = m.b_cap(hw, S)
    t0 = m.t0_ms(hw)
    F = F_ms(m, hw, exposed, d2h_fixed)
    parts = c_ms(m, hw, S, bs, d2h_fixed)
    c = parts["total"] - (parts["per_block"] if drop_per_block else 0.0)
    return (1.0 + r / b_res) / (1.0 + (F + c * r) / t0)


def link_sweep(m: Model, S: int, exposed: float, d2h_fixed: bool,
               drop_per_block: bool, bs: int = 16, hbm_gbs: float = 4000,
               hbm_gb: float = 96) -> list:
    """THE CORE QUESTION, isolated: hold the model, the context and the host
    fixes constant, and change ONLY the link bandwidth.

    Each row is evaluated at r = the knee, the most riders that regime admits.
    `INFINITE` is the ceiling: what no interconnect can beat.
    """
    rows = []
    for x, nm in ((25.1, "PCIe4 x16"), (64.0, "PCIe5 x16"),
                  (128.0, "PCIe6 x16"), (419.0, "NVLink-C2C"),
                  (900.0, "2x C2C"), (1e5, "INFINITE")):
        hw = HW(nm, hbm_gbs, x, hbm_gb)
        be = break_even(m, hw, S, None, exposed, d2h_fixed, bs,
                        drop_per_block)
        r = min(be["knee"], 64.0)
        rows.append({
            "link_gbs": x, "name": nm, "c": be["c"],
            "c_link": be["parts"]["link"],
            "link_share": be["parts"]["link"] / be["c"],
            "knee": be["knee"], "r_be": be["r_be"],
            "t0_over_b": be["t0"] / be["b_res"],
            "gain": gain(m, hw, S, r, exposed, d2h_fixed, None, bs,
                         drop_per_block),
            "wins": be["r_be"] is not None and be["r_be"] <= be["knee"],
        })
    return rows


def core_question(S: int = 8192) -> None:
    """Printed first, because it is the question the campaign existed for."""
    print("=" * 78)
    print(" DOES A FASTER LINK, ADMITTING MORE RIDERS, GIVE MORE THROUGHPUT?")
    print("=" * 78)
    m = L70
    print("\n %s, ctx %d, GH200-class HBM. Rows differ ONLY in link speed."
          % (m.name, S))
    print(" B_cap = %.1f residents, T0 = %.1f ms, T0/B = %.2f ms -- what one"
          % (m.b_cap(GH200, S), m.t0_ms(GH200),
             m.t0_ms(GH200) / m.b_cap(GH200, S)))
    print(" rider must cost LESS than, or no rider count ever helps.\n")
    for drop, dfix, exp, lab in (
            (False, False, HIDING[2][1], "as built"),
            (True, True, HIDING[2][1],
             "per-block loop removed + D2H fixed")):
        print(" %s" % lab)
        print("   %-12s %8s %8s %7s %7s %7s %8s"
              % ("link", "c", "c_link", "share", "knee", "r_be", "gain"))
        for row in link_sweep(m, S, exp, dfix, drop):
            print("   %-12s %8.3f %8.3f %6.1f%% %7.1f %7s %7.2fx%s"
                  % (row["name"], row["c"], row["c_link"],
                     100 * row["link_share"], row["knee"],
                     "-" if row["r_be"] is None else "%.1f" % row["r_be"],
                     row["gain"], "" if row["wins"] else "   no win"))
        print()
    inf = link_sweep(m, S, HIDING[2][1], False, False)[-1]
    print(" The ceiling, as built: an INFINITE link still leaves c = %.2f"
          % inf["c"])
    print(" against T0/B = %.2f. Bandwidth alone cannot satisfy c < T0/B,"
          % inf["t0_over_b"])
    print(" so the verdict as built does not depend on the interconnect.\n")


def projection_table() -> None:
    print("=" * 78)
    print(" MEASURED LAW -> C2C PROJECTION, as an interval")
    print("=" * 78)
    print()
    print("The A100 anchor, which this law was fitted on:")
    for S in (512, 2048):
        r = break_even(Q32, A100, S, 16, 0.026, False)
        print("  Qwen-32B ctx %-5d F=%5.2f c=%5.3f T0/B=%5.3f knee=%5.1f  %s"
              % (S, r["F"], r["c"], r["t0"] / 16, r["knee"], r["verdict"]))
    print()
    print("GH200, Llama-70B fp8, 80 layers, ctx 8192, residents from capacity.")
    print("compute/layer = %.3f ms, BELOW every A100 point measured (min %.3f)"
          % (L70.compute_per_layer(GH200), min(h[0] for h in HIDING)))
    print("so the hiding fraction is EXTRAPOLATED, and the band is the finding.")
    print()
    ref = break_even(L70, GH200, 8192, None, 0.026, False)
    print("residents from CAPACITY, not a constant: B_cap = %.1f at ctx 8192"
          % L70.b_cap(GH200, 8192))
    print("  (C = %.1f - %.1f - 4 = %.1f GB, u = %.2f GB/request)"
          % (GH200.hbm_gb, L70.w, GH200.hbm_gb - L70.w - 4,
             L70.b_tok * 8192 / 1e9))
    print()
    print("Where c goes at ctx 8192 on C2C -- the link is NINE PERCENT of it:")
    for k, v in sorted(ref["parts"].items(), key=lambda kv: -kv[1]
                       if isinstance(kv[1], float) else 0):
        if k in ("total", "nominal_link_ms"):
            continue
        print("    %-18s %7.3f ms/rider  (%4.1f%%)"
              % (k, v, v / ref["parts"]["total"] * 100))
    print("    %-18s %7.3f" % ("total", ref["parts"]["total"]))
    print()
    print("%-26s %-11s %7s %7s %7s %s"
          % ("hiding assumption", "D2H", "F", "c", "knee", "verdict"))
    for cl, exposed, src in HIDING:
        for fixed, lab in ((True, "fixed"), (False, "as built")):
            r = break_even(L70, GH200, 8192, None, exposed, fixed)
            print("%-26s %-11s %7.1f %7.3f %7.1f %s"
                  % ("as %s (%.3f)" % (src.split()[0], exposed), lab,
                     r["F"], r["c"], r["knee"], r["verdict"]))
    print()
    print("EVERY ROW FAILS, and not because of the link. Now remove the")
    print("per-block Python loop in build_step as well:")
    print("%-26s %-11s %7s %7s %7s %s"
          % ("hiding assumption", "D2H", "F", "c", "knee", "verdict"))
    for cl, exposed, src in HIDING:
        r = break_even(L70, GH200, 8192, None, exposed, True, drop_per_block=True)
        print("%-26s %-11s %7.1f %7.3f %7.1f %s"
              % ("as %s (%.3f)" % (src.split()[0], exposed), "fixed",
                 r["F"], r["c"], r["knee"], r["verdict"]))
    print()
    print("  the paper assumes F(70B) = 12.3 * 80/64 = %.1f ms" % (12.3 * 80 / 64))
    print()
    print("DOMINATING UNCERTAINTY: the hiding fraction, spanning %.1fx on F."
          % (F_ms(L70, GH200, max(h[1] for h in HIDING), False)
             / F_ms(L70, GH200, min(h[1] for h in HIDING), True)))
    print("  By comparison the link changes c by about 44%. The projection is")
    print("  limited by a host-side property, not by the interconnect.")
    print()
    print("  Carry also: F has a %.0f%% BETWEEN-JOB offset on identical "
          "hardware" % JOB_OFFSET_PCT)
    print("  (within-job 5%), and the link-bound share of c is 37-57% "
          "depending on method.")
    print()
    print("THE ONE MEASUREMENT THAT WOULD CLOSE IT")
    print("  Run the three-arm D2H experiment on any GPU faster than an A100.")
    print("  It needs no C2C link and no 70B checkpoint: the quantity wanted is")
    print("  the EXPOSED async host work per layer at a compute/layer near 0.3")
    print("  ms, which an H100 with Qwen-32B reaches directly. Engine-free and")
    print("  under an hour. That single number collapses an 18x band to a point.")


def _selftest() -> int:
    ok = True
    # the law must reproduce the A100 measurements it was fitted on
    c = c_ms(Q32, A100, 512, 16, False)
    meas = 1.25
    good = abs(c["total"] - meas) / meas < 0.20
    ok &= good
    print("  [%s] c at ctx 512 reproduces the measurement: %.3f vs %.2f"
          % ("ok " if good else "BAD", c["total"], meas))

    F = F_ms(Q32, A100, 0.026, False)
    good = abs(F - 10.76) / 10.76 < 0.15
    ok &= good
    print("  [%s] F on Qwen-32B reproduces M3: %.2f vs 10.76"
          % ("ok " if good else "BAD", F))

    # removing the D2H must drop F by ~85% on Qwen-32B
    drop = 1 - F_ms(Q32, A100, 0.026, True) / F
    good = 0.75 < drop < 0.95
    ok &= good
    print("  [%s] D2H fix removes %.0f%% of F on Qwen-32B (M3 measured 85%%)"
          % ("ok " if good else "BAD", drop * 100))

    # ctx 2048 on PCIe must come out as never paying
    r = break_even(Q32, A100, 2048, 16, 0.026, False)
    good = r["r_be"] is None
    ok &= good
    print("  [%s] ctx 2048 on PCIe never pays (c=%.2f vs T0/B=%.2f)"
          % ("ok " if good else "BAD", r["c"], r["t0"] / 16))

    # the GH200 band must be wide -- if it is not, the model lost the finding
    lo = F_ms(L70, GH200, min(h[1] for h in HIDING), True)
    hi = F_ms(L70, GH200, max(h[1] for h in HIDING), False)
    good = hi / lo > 10
    ok &= good
    print("  [%s] GH200 F band spans %.1fx (%.1f to %.1f ms)"
          % ("ok " if good else "BAD", hi / lo, lo, hi))

    # THE CORE CLAIM 1: as built, no link speed satisfies c < T0/B, so the
    # verdict cannot be changed by the interconnect. If this ever flips, the
    # report's headline is wrong.
    rows = link_sweep(L70, 8192, HIDING[2][1], False, False)
    good = not any(r["wins"] for r in rows)
    ok &= good
    print("  [%s] as built, NO link wins -- infinite link leaves c=%.2f vs "
          "T0/B=%.2f" % ("ok " if good else "BAD", rows[-1]["c"],
                         rows[-1]["t0_over_b"]))

    # THE CORE CLAIM 2: with the two host fixes the gain is monotone in
    # bandwidth, and C2C is the FIRST point that wins -- PCIe6 is not enough.
    fx = link_sweep(L70, 8192, HIDING[2][1], True, True)
    mono = all(fx[i]["gain"] <= fx[i + 1]["gain"] for i in range(len(fx) - 1))
    first = next((r["name"] for r in fx if r["wins"]), None)
    good = mono and first == "NVLink-C2C"
    ok &= good
    print("  [%s] fixed: gain monotone in bandwidth (%.2fx -> %.2fx), first "
          "winner %s" % ("ok " if good else "BAD", fx[0]["gain"],
                         fx[-1]["gain"], first))

    # and PCIe6 must fail on the KNEE, not on c < T0/B -- that is what shows
    # the link's real contribution is raising the knee, not shrinking c.
    p6 = next(r for r in fx if r["name"] == "PCIe6 x16")
    good = (p6["c"] < p6["t0_over_b"]) and (p6["r_be"] > p6["knee"])
    ok &= good
    print("  [%s] PCIe6 fails on the KNEE not on c<T0/B (c=%.2f<%.2f, "
          "r_be=%.1f>knee=%.1f)" % ("ok " if good else "BAD", p6["c"],
                                    p6["t0_over_b"], p6["r_be"], p6["knee"]))

    print("\nOVERHEAD MODEL SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return _selftest()
    core_question()
    projection_table()
    return 0


if __name__ == "__main__":
    sys.exit(main())
