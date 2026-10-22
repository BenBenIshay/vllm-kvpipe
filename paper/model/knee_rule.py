"""Admission past the knee (Ben, 2026-10-22: "update the knee ... claim 1.45x").

The paper's rule admits floor(T0*X / (S*b)) riders: every rider's transfer
hides. M22c (job 2190742, A100, ctx 1024, 1 resident, real bytes) measured what
happens past that point: the step becomes the link time plus a tail,
    T(r) = max(T0 + F + c*r,  r*link + delta),   delta = 6.8 ms (r = 5), 7.5 (r = 6)
and throughput at 5 riders was 8% above the rule's 4. This script
  1. checks that law against M22c's measured decode throughput, and
  2. projects the target with the rule "admit the r that maximises (B+r)/T(r)".
The 1.45x that motivated the request came from the pre-M22 cost (c = 2.51 ms)
with the full-step knee; M22c rejected that knee (M22_ANALYSIS s5).
"""
from gain_curve import point

# M22c, M22_ANALYSIS s5 and paper s6.3 src comment
A100 = dict(B=1, T0=45.2, F=5.16, c=1.55, link=11.05)
MEASURED = {4: (88.4, 88.8), 5: (96.6, 96.7), 6: (94.7, 94.8)}   # decode tok/s


def step(p, r, delta):
    if r == 0:
        return p["T0"]
    return max(p["T0"] + p["F"] + p["c"] * r, r * p["link"] + delta)


def best_r(p, delta, rmax=12):
    return max(range(rmax + 1), key=lambda r: (p["B"] + r) / step(p, r, delta))


if __name__ == "__main__":
    print("M22c check (delta 6.8 ms; 7.5 ms at r = 6 as measured):")
    for r in range(0, 7):
        d = 7.5 if r == 6 else 6.8
        tps = (A100["B"] + r) / step(A100, r, d) * 1e3
        print("  r=%d  predicted %.1f tok/s  measured %s" % (r, tps, MEASURED.get(r, "-")))
    print("  rule T0*X admits %d; throughput-max rule admits %d"
          % (int(A100["T0"] / A100["link"]), best_r(A100, 6.8)))

    t = point(14724)
    F = 3.39 * 80 / 64
    tgt = dict(B=t["B"], T0=t["t0"], F=F, c=t["c"], link=4.825e9 / 419e9 * 1e3)
    tgt["link"] = 327680 * 14724 / 1e9 / 419 * 1e3
    print("\ntarget: B=%d T0=%.2f F=%.2f c=%.2f link=%.2f ms/rider"
          % (tgt["B"], tgt["T0"], F, tgt["c"], tgt["link"]))
    for name, d in (("delta 6.8 ms (A100 value, unscaled)", 6.8),
                    ("delta scaled with T0", 6.8 * tgt["T0"] / A100["T0"]),
                    ("delta = 0 (full-step knee)", 0.0)):
        rows = []
        for r in range(0, 7):
            g = ((tgt["B"] + r) / step(tgt, r, d)) / (tgt["B"] / tgt["T0"])
            rows.append("r%d %.3f" % (r, g))
        rb = best_r(tgt, d)
        gb = ((tgt["B"] + rb) / step(tgt, rb, d)) / (tgt["B"] / tgt["T0"])
        print("  %-38s best r=%d G=%.3f eta=%.1f | %s"
              % (name, rb, gb, (gb - 1) / (419 / 4000), "  ".join(rows)))
