"""Projected gain of each design class at the operating point (Ben, 2026-10-21:
"projected graph comparison with DirectKV/Pie/DAK"). Every value is our capacity
model applied to the MECHANISM each paper describes, not a model of the system.

  HBM-only        1
  aggregation     1 + X/BW: the same batch read over HBM and link together (DAK
                  as published; bound with copies free)
  zero-copy, all  every request's KV read over the link (DirectKV keeps every
                  request's KV in CPU memory): at most X/(S b_tok) requests per
                  second per request-step, against B/T0 from HBM
  layer swap      Pie: m of n layers in host memory, swapped in ahead of use and
                  the finished layer swapped out, so each moved byte is written
                  into HBM and read out of it: 2*kappa. No exposed first layer
                  (Pie can keep it on the GPU). Same KV-per-step ceiling T0*X.
  HeteroKV        eq. 2 with kappa, first layer
2026-10-22 (Ben: "apply the 1.37x rule across the paper"): every design that
adds KV over the link gets the same past-knee rule, the r that maximises
(B+r)/T(r) with T(r) = max(T0+F+c r, r*link + DELTA) (gain_curve.py). Layer
swap's link time is set by its swap-out, 371 GB/s (pie s2.4); the ceiling is
the same rule with no costs and no tail.
'mechanism' drops c_m and F (implementation overheads); 'with overheads' adds
HeteroKV's measured c_m and F to both, as no overheads are published for Pie.
"""
from gain_curve import C_MECH64, DELTA, EFF_KV, F64, KAPPA
from overhead_model import GH200, Q72

m, hw, S, B = Q72, GH200, 14724, 3
kv = m.b_tok * S / 1e9
att = kv / (hw.hbm * EFF_KV) * 1e3
cont = KAPPA * kv / hw.hbm * 1e3
first = kv / m.L / hw.link * 1e3
cm, F = C_MECH64 * m.L / 64, F64 * m.L / 64
t0 = m.t0_ms(hw) + B * att
x = hw.link / hw.hbm


def g(c, f, x_gbs=hw.link, d=DELTA):
    link = kv / x_gbs * 1e3

    def gain(r):
        t = t0 if r == 0 else max(t0 + f + c * r, r * link + d)
        return ((B + r) / t) / (B / t0)
    return max(gain(r) for r in range(13))


rows = [
    ("HBM only", 1.0),
    ("aggregation (DAK)", 1 + x),
    ("zero-copy, all KV (DirectKV)", (hw.link / kv) / (B / t0 * 1e3)),
    ("layer swap (Pie), mechanism", g(att + 2 * cont, 0, 371)),
    ("layer swap (Pie), with overheads", g(cm + att + 2 * cont, F, 371)),
    ("HeteroKV, mechanism", g(att + cont + first, 0)),
    ("HeteroKV, with overheads", g(cm + att + cont + first, F)),
    ("ceiling, no costs", g(0, 0, d=0.0)),
]
if __name__ == "__main__":
    for name, v in rows:
        print("%-34s G %.3f  eta %.2f" % (name, v, (v - 1) / x))
