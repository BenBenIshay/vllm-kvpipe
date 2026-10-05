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
# PROVENANCE DEFECT, found 2026-10-05. Every coefficient below was fitted from
# runs taken with VLLM_KVPIPE_RUNGD_ASYNC enabled, and that path produces
# INCORRECT OUTPUT: hybrid_rungd_real.py passes 3/3 with it unset and fails 2/3
# with it set. run_suite.sh never sets the variable, so the suite validated the
# non-async path, while rider_cost_curve.py defaults --rungd-async to True and
# kv_k_window defaults to 4 -- so every archived artifact reads
# async=True, k_window=4, M0 through M10.
#
# A missing wait makes a step FASTER, so these are plausibly OPTIMISTIC and are
# PROVISIONAL pending the no-async re-measurement. The SHAPE of the law is not
# in doubt -- the four-way identification and the gate c < T0/B rest on ratios
# and on which axis each term scales with, and the defect is common to every arm
# of every sweep -- but the absolute values may move.
#
# `--selftest` asserts relationships, not absolute values, for exactly this
# reason; it is the magnitudes quoted in projections that need re-checking.
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
    # POST-T1, MEASURED. The block-table rebuild was 0.00765 ms per rider BLOCK
    # (d_per_block above), so it scaled with context: 3.917 ms/rider at ctx
    # 8192, 76% of c at the target. T1 made it one slice assignment per REQUEST
    # and the measured cost is now per-RIDER and block-independent:
    #
    #   ctx 512 bs 16, build_step_bt_python, Class A host timer
    #     riders   1      2      4      8     12
    #     blocks  36     72    144    288    432
    #     bt ms   .0243  .0283  .0341  .0475  .0593
    #
    #   fit: bt_ms = 0.0216 + 0.00317 * riders, max residual 0.00055 ms
    #
    # Blocks rose 12x while the time rose 2.4x, and ms/BLOCK fell 5x -- a
    # per-block cost would have held ms/block constant. 87x less per rider at
    # ctx 512, and because the coefficient no longer carries S, 1234x at the
    # ctx-8192 target.
    # CONFIRMED on a second, independent axis. bs 16 -> 64 cuts the block
    # count 4x at fixed context and cohort, and the timer moves 2-6%:
    #
    #   riders   bs16 ms  bs64 ms   bs16 blk  bs64 blk  ratio
    #        1    0.0243   0.0234         36         9   0.96x
    #        4    0.0341   0.0333        144        36   0.98x
    #       12    0.0593   0.0574        432       108   0.97x
    #
    # Pre-T1 that column would read ~0.25x. The two arms' per-rider slopes
    # agree to 3% (0.00317 vs 0.00308) while differing 4x in block count
    # throughout, which is what per-RIDER means.
    # POST-T2, MEASURED. The blocking pageable mirror write was 0.142 ms per
    # layer (d2h_per_layer_8x128) and 85% of F. T2 made the copy async into
    # pinned staging and deferred the host-side scatter to the next step behind
    # one event wait. Bracketed before/after, both arms numerically valid:
    #
    #   ctx 512    F 11.48 -> 4.74 ms   saved 6.74, drift 0.19 (35x ratio)
    #   ctx 2048   F  9.70 -> 4.69 ms   saved 5.01
    #
    # per layer on L=64 that is 0.1655 -> 0.0737 mean. Subtracting the 0.026
    # exposed async host work leaves the D2H part at 0.0477 against 0.142, so
    # T2 removed 66% of it -- NOT all of it, which matters: Phase 0's 8.47 ms
    # was the ceiling (that arm skipped the scatter entirely) and T2 realised
    # 80% of it, the shortfall being the 2L CPU index_copy_ calls it still does.
    "f_per_layer_post_t2": (0.0737, "ms/layer",
                            "M9, mean of ctx 512 (0.0741) and ctx 2048 "
                            "(0.0733); REPLACES d2h_per_layer + exposed, and "
                            "is 66% off the D2H part rather than all of it"),
    "t1_per_rider": (0.00313, "ms/rider",
                     "M8, mean of T1_c512_bs16 (0.00317, resid 0.00055) and "
                     "T1_c512_bs64 (0.00308, resid 0.00045), 5 rider points "
                     "each; replaces d_per_block and is INDEPENDENT of block "
                     "count -- a 4x block change moves it 2-6%"),
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
Q72 = Model("Qwen-72B fp8", 80, 76.0, 8, 128)

# THE TARGET, retargeted 2026-10-05 from (L70, 8192) on Ben's instruction after
# the model search (best_model.py) put Qwen-72B fp8 at ctx 16384 ahead of every
# other (model, context) pair. It wins on nothing but 76.0 GB of weights against
# 72.7: heavier weights raise T0 and shrink the capacity left for KV, so fewer
# residents fit and each rider is worth more. L70 is KEPT, not deleted -- the
# whole projection section and several selftests are stated against it, and it
# remains the anchor the paper's own tables use.
TARGET, TARGET_S = Q72, 16384
LEGACY, LEGACY_S = L70, 8192


def F_ms(m: Model, hw: HW, exposed_per_layer: float, d2h_fixed: bool,
         t2: bool = False) -> float:
    """Fixed cost, paid once any rider exists.

    `t2=True` uses the MEASURED post-T2 per-layer cost rather than modelling the
    mirror write as removed. That distinction matters: `d2h_fixed=True` assumes
    the whole blocking copy goes away, which no implementation achieves, while
    T2 measurably removes 66% of it.
    """
    if t2:
        return m.L * COEFF["f_per_layer_post_t2"][0]
    d2h = 0.0 if d2h_fixed else m.d2h_per_layer()
    return m.L * (d2h + exposed_per_layer)


def c_ms(m: Model, hw: HW, S: int, bs: int, d2h_fixed: bool,
         link_frac: float | None = None, t1: bool = False) -> dict:
    """Per-rider cost, decomposed.

    `t1=True` uses the MEASURED post-T1 block-table coefficient instead of the
    per-block one. That is not an estimate of a proposed change: T1 is
    implemented, the suite is 9/9 with it, and the coefficient is a fit over
    five rider points (see COEFF["t1_per_rider"]).
    """
    nominal = m.b_tok * S / (hw.link * 1e9) * 1e3
    frac = COEFF["c_link"][0] if link_frac is None else link_frac
    parts = {
        "per_rider_layer": COEFF["b_per_rider_layer"][0] * m.L,
        "per_block": (COEFF["t1_per_rider"][0] if t1
                      else COEFF["d_per_block"][0] * (S // bs)),
        "link": frac * nominal,
        "d2h": 0.0 if d2h_fixed else COEFF["d2h_per_rider"][0],
    }
    parts["total"] = sum(parts.values())
    parts["nominal_link_ms"] = nominal
    return parts


def break_even(m: Model, hw: HW, S: int, b_res: float | None, exposed: float,
               d2h_fixed: bool, bs: int = 16, drop_per_block: bool = False,
               t1: bool = False, t2: bool = False) -> dict:
    if b_res is None:
        b_res = m.b_cap(hw, S)
    t0 = m.t0_ms(hw)
    F = F_ms(m, hw, exposed, d2h_fixed, t2=t2)
    parts = c_ms(m, hw, S, bs, d2h_fixed, t1=t1)
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
         drop_per_block: bool = False, t1: bool = False,
         t2: bool = False) -> float:
    """Throughput gain from admitting r riders on top of B residents.

        gain = (1 + r/B) / (1 + (F + c*r)/T0)

    Numerator: the extra requests served. Denominator: the longer step they
    cost. Valid only BELOW the knee -- above it the step is the transfer and
    this formula no longer describes the step time.
    """
    if b_res is None:
        b_res = m.b_cap(hw, S)
    t0 = m.t0_ms(hw)
    F = F_ms(m, hw, exposed, d2h_fixed, t2=t2)
    parts = c_ms(m, hw, S, bs, d2h_fixed, t1=t1)
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


def delivered(S: int | None = None, m: "Model | None" = None) -> dict:
    """The projection with T1 and T2 as BUILT AND MEASURED, not as proposed.

    Both coefficients here are measurements on this code: T1's 0.00313 ms/rider
    from a 5-point rider sweep confirmed by a 4x block-size change, and T2's
    0.0737 ms/layer from a bracketed before/after. Neither is an estimate of a
    change that might be made.

    Defaults to the TARGET (Qwen-72B fp8, ctx 16384). Pass (LEGACY_S, LEGACY) for
    the Llama-70B / ctx 8192 point the earlier sections are written against.
    """
    S = TARGET_S if S is None else S
    m = TARGET if m is None else m
    hw = GH200
    out = {}
    for lab, kw in (("as built", {}),
                    ("T1 only", {"t1": True}),
                    ("T1 + T2", {"t1": True, "t2": True})):
        be = break_even(m, hw, S, None, HIDING[2][1], False, **kw)
        r = min(be["knee"], 64.0)
        be["gain"] = gain(m, hw, S, r, HIDING[2][1], False, **kw)
        be["label"] = lab
        out[lab] = be
    return out


def _delivered_block(m, S) -> None:
    d = delivered(S, m)
    any_v = next(iter(d.values()))
    print(" %s, ctx %d, GH200 C2C 419 GB/s" % (m.name, S))
    print(" B_cap %.1f residents, T0 %.1f ms, T0/B %.2f ms, knee %.1f riders"
          % (any_v["b_res"], any_v["t0"], any_v["t0"] / any_v["b_res"],
             any_v["knee"]))
    print()
    print(" %-12s %8s %8s %9s %9s %s"
          % ("state", "c", "F", "r_be", "gain", "verdict"))
    for lab in ("as built", "T1 only", "T1 + T2"):
        v = d[lab]
        print(" %-12s %8.3f %8.2f %9s %8.2fx %s"
              % (lab, v["c"], v["F"],
                 "-" if v["r_be"] is None else "%.1f" % v["r_be"],
                 v["gain"], v["verdict"]))
    print()


def delivered_table() -> None:
    print("=" * 78)
    print(" WHAT WAS BUILT: T1 AND T2 AS MEASURED")
    print("=" * 78)
    print()
    print(" THE TARGET (retargeted after the model search):")
    _delivered_block(TARGET, TARGET_S)
    print(" the earlier anchor, kept for comparison:")
    _delivered_block(LEGACY, LEGACY_S)
    t = delivered()["T1 + T2"]
    l = delivered(LEGACY_S, LEGACY)["T1 + T2"]
    print(" retargeting is worth %.2fx -> %.2fx, and buys it by fitting FEWER"
          % (l["gain"], t["gain"]))
    print(" residents (%.1f against %.1f), which raises what each rider is"
          % (t["b_res"], l["b_res"]))
    print(" worth (T0/B %.2f against %.2f ms)."
          % (t["t0"] / t["b_res"], l["t0"] / l["b_res"]))
    print()
    hi = delivered(TARGET_S * 2, TARGET)["T1 + T2"]
    print(" READ THAT MECHANISM BEFORE BANKING THE NUMBER. The gain is a RATIO")
    print(" against a baseline that holds only %.1f residents, and it rises as"
          % t["b_res"])
    print(" that count falls -- %.2fx at ctx %d where just %.1f fit. A bigger"
          % (hi["gain"], TARGET_S * 2, hi["b_res"]))
    print(" ratio against a weaker baseline is not more tokens per second, so")
    print(" the best RATIO is not automatically the best deployment. Absolute")
    print(" throughput at the target is %.0f tok/s against %.0f at the anchor."
          % (t["b_res"] * (1 + min(t["knee"], 64.0) / t["b_res"])
             / (t["t0"] + t["F"] + t["c"] * min(t["knee"], 64.0)) * 1e3,
             l["b_res"] * (1 + min(l["knee"], 64.0) / l["b_res"])
             / (l["t0"] + l["F"] + l["c"] * min(l["knee"], 64.0)) * 1e3))
    print()


def kv_precision_table() -> None:
    """Does fp8 KV help, hurt, or neither? Neither -- it is a context rescale.

    This was an open question flagged as possibly flipping the verdict, and the
    answer is that it cannot: the law sees the per-token KV footprint and the
    context ONLY through their product. Halving the bytes per token and doubling
    the context are the same point to within floating point.
    """
    print("=" * 78)
    print(" fp16 KV vs fp8 KV: the law sees only the PRODUCT b_tok * S")
    print("=" * 78)
    print()
    print(" %-9s %7s %6s %7s %7s %6s %8s"
          % ("KV", "ctx", "B", "T0/B", "c", "knee", "gain"))
    rows = {}
    for lab, kvb in (("fp16", 2), ("fp8", 1)):
        m = Model(TARGET.name, TARGET.L, TARGET.w, TARGET.kv_heads,
                  TARGET.head_dim, kvb)
        for S in (8192, 16384, 32768):
            d = delivered(S, m)["T1 + T2"]
            rows[(lab, S)] = d
            print(" %-9s %7d %6.1f %7.2f %7.3f %6.1f %7.3fx"
                  % (lab, S, d["b_res"], d["t0"] / d["b_res"], d["c"],
                     d["knee"], d["gain"]))
    print()
    a, b = rows[("fp16", 16384)], rows[("fp8", 32768)]
    print(" fp8 at 32768 IS fp16 at 16384: gain %.6f vs %.6f, c %.6f vs %.6f."
          % (a["gain"], b["gain"], a["c"], b["c"]))
    print(" So KV quantisation neither helps nor hurts this feature at a matched")
    print(" b_tok*S. What it does is move the best context: the same gain now")
    print(" sits at twice the context, which is a REAL operational win even")
    print(" though the law cannot see it. (An earlier note in this campaign said")
    print(" fp8 KV loses. That compared the two at a FIXED context, where fp8")
    print(" doubles the resident count and so halves T0/B -- the right")
    print(" comparison holds b_tok*S fixed.)")
    print()


def absolute_table() -> None:
    """The gain RATIO and the absolute tokens/s rank in OPPOSITE directions.

    This is the one result that changes what should be deployed, as opposed to
    what should be built. The ratio measures the feature against the same system
    without it, which is the right test of the mechanism. But maximising the
    ratio over (model, context) selects for a WEAK baseline: fewer residents fit,
    each is worth more, so each rider clears the bar more easily. In absolute
    tokens per second the ranking inverts, and the heavier model loses to the
    lighter one at every context.
    """
    print("=" * 78)
    print(" THE RATIO AND THE ABSOLUTE RATE RANK OPPOSITELY")
    print("=" * 78)
    print()
    print(" %-16s %6s %5s %6s %6s %7s %10s %10s"
          % ("model", "ctx", "B", "knee", "gain", "ITL", "base tok/s",
             "+ riders"))
    best_ratio = best_abs = None
    for m in (LEGACY, TARGET):
        for S in (4096, 8192, 16384, 32768):
            d = delivered(S, m)["T1 + T2"]
            r = min(d["knee"], 64.0)
            itl = d["t0"] + d["F"] + d["c"] * r
            base = d["b_res"] / d["t0"] * 1e3
            wr = (d["b_res"] + r) / itl * 1e3
            print(" %-16s %6d %5.1f %6.1f %5.2fx %7.1f %10.0f %10.0f"
                  % (m.name, S, d["b_res"], d["knee"], d["gain"], itl, base, wr))
            if best_ratio is None or d["gain"] > best_ratio[0]:
                best_ratio = (d["gain"], wr, m.name, S)
            if best_abs is None or wr > best_abs[1]:
                best_abs = (d["gain"], wr, m.name, S)
    print()
    print(" best RATIO    %s ctx %d: %.2fx, %.0f tok/s"
          % (best_ratio[2], best_ratio[3], best_ratio[0], best_ratio[1]))
    print(" best ABSOLUTE %s ctx %d: %.2fx, %.0f tok/s  (%.1fx the rate)"
          % (best_abs[2], best_abs[3], best_abs[0], best_abs[1],
             best_abs[1] / best_ratio[1]))
    print()
    print(" So TARGETING Qwen-72B fp8 at ctx 16384 is correct for the question")
    print(" 'does the feature pay', and WRONG read as 'this is the deployment")
    print(" to ship'. The two are not in conflict: the feature's value is")
    print(" largest exactly where the baseline is weakest -- heavy weights, long")
    print(" context, few residents -- which is the regime the project exists to")
    print(" serve. It just must not be reported as a throughput record.")
    print()


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

    # T1 IS IMPLEMENTED AND MEASURED, so its effect on the gate is asserted,
    # not projected. The whole point of the block-table rewrite was to get c
    # under T0/B at the target, which it failed by 53% before.
    S = 8192
    before = break_even(L70, GH200, S, None, HIDING[2][1], False)
    after = break_even(L70, GH200, S, None, HIDING[2][1], False, t1=True)
    tb = before["t0"] / before["b_res"]
    good = before["c"] > tb and after["c"] < tb
    ok &= good
    print("  [%s] T1 brings c under T0/B at the target: %.2f -> %.2f against "
          "%.2f" % ("ok " if good else "BAD", before["c"], after["c"], tb))

    good = after["parts"]["per_block"] < 0.01
    ok &= good
    print("  [%s] the block-table term is now %.5f ms/rider and carries no S"
          % ("ok " if good else "BAD", after["parts"]["per_block"]))

    # T2 IS BUILT AND MEASURED. It must cut F substantially but NOT to the
    # idealised "d2h_fixed" floor, because it removes 66% of the blocking copy
    # rather than all of it -- conflating the two would overstate the result.
    d = delivered(LEGACY_S, LEGACY)
    f_built, f_t2 = d["as built"]["F"], d["T1 + T2"]["F"]
    f_ideal = F_ms(L70, GH200, HIDING[2][1], True)
    good = f_t2 < 0.6 * f_built and f_t2 > f_ideal
    ok &= good
    print("  [%s] T2 cuts F %.1f -> %.1f, above the idealised floor %.1f"
          % ("ok " if good else "BAD", f_built, f_t2, f_ideal))

    # and the pair must flip the target from losing to paying
    good = (d["as built"]["r_be"] is None
            and d["T1 + T2"]["r_be"] is not None
            and d["T1 + T2"]["r_be"] <= d["T1 + T2"]["knee"])
    ok &= good
    print("  [%s] T1+T2 turns the target from 'never pays' into paying from "
          "r=%.1f (knee %.1f)"
          % ("ok " if good else "BAD",
             d["T1 + T2"]["r_be"] or -1, d["T1 + T2"]["knee"]))

    # THE RETARGET. Qwen-72B fp8 at ctx 16384 must beat the Llama-70B anchor,
    # and it must do so through the stated mechanism -- fewer residents, each
    # worth more -- rather than through anything else moving.
    t = delivered()["T1 + T2"]
    l = delivered(LEGACY_S, LEGACY)["T1 + T2"]
    good = (t["gain"] > l["gain"] and t["b_res"] < l["b_res"]
            and t["t0"] / t["b_res"] > l["t0"] / l["b_res"])
    ok &= good
    print("  [%s] target beats the anchor %.3fx > %.3fx via B %.1f < %.1f and "
          "T0/B %.2f > %.2f" % ("ok " if good else "BAD", t["gain"], l["gain"],
                                t["b_res"], l["b_res"], t["t0"] / t["b_res"],
                                l["t0"] / l["b_res"]))

    good = t["r_be"] is not None and t["r_be"] <= t["knee"]
    ok &= good
    print("  [%s] the target pays, from r=%.1f against a knee of %.1f"
          % ("ok " if good else "BAD", t["r_be"] or -1, t["knee"]))

    # fp8 KV is a CONTEXT RESCALE, not a win or a loss. This is the open
    # question that was flagged as possibly flipping the verdict; asserting the
    # identity is what closes it.
    fp8 = Model(TARGET.name, TARGET.L, TARGET.w, TARGET.kv_heads,
                TARGET.head_dim, 1)
    a = delivered(TARGET_S, TARGET)["T1 + T2"]
    b = delivered(TARGET_S * 2, fp8)["T1 + T2"]
    good = all(abs(a[k] - b[k]) < 1e-9
               for k in ("gain", "c", "b_res", "knee"))
    ok &= good
    print("  [%s] fp8 KV at 2S == fp16 KV at S exactly (gain %.6f vs %.6f)"
          % ("ok " if good else "BAD", a["gain"], b["gain"]))

    # AND THE GAIN IS BOUGHT WITH LATENCY. The design's own precondition is an
    # ITL budget above HBM/BW; the post-admission ITL must be checked against
    # it, because it is ABOVE it and that constrains where the feature may run.
    r = min(t["knee"], 64.0)
    itl = t["t0"] + t["F"] + t["c"] * r
    floor = GH200.hbm_gb / GH200.hbm * 1e3
    good = itl > floor and itl / t["t0"] > 1.2
    ok &= good
    print("  [%s] ITL %.1f -> %.1f ms (%.2fx), ABOVE the %.1f ms HBM/BW "
          "precondition" % ("ok " if good else "BAD", t["t0"], itl,
                            itl / t["t0"], floor))

    # THE RATIO AND THE ABSOLUTE RATE RANK OPPOSITELY. The target has the best
    # gain and a LOWER tokens/s than the anchor; if that ever stops being true
    # the "best ratio is not the best deployment" caveat must be rewritten.
    def _rate(m, S):
        d = delivered(S, m)["T1 + T2"]
        r = min(d["knee"], 64.0)
        return (d["b_res"] + r) / (d["t0"] + d["F"] + d["c"] * r) * 1e3
    rt, rl = _rate(TARGET, TARGET_S), _rate(LEGACY, LEGACY_S)
    good = t["gain"] > l["gain"] and rt < rl
    ok &= good
    print("  [%s] best ratio is NOT best rate: %.2fx at %.0f tok/s beats "
          "%.2fx at %.0f tok/s on ratio and loses on rate"
          % ("ok " if good else "BAD", t["gain"], rt, l["gain"], rl))

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
    delivered_table()
    kv_precision_table()
    absolute_table()
    projection_table()
    return 0


if __name__ == "__main__":
    sys.exit(main())
