"""Gain against context at the target, under the M22-corrected model.

Ben, 2026-10-20: "show gain vs context as a curve in the paper". Two corrections
to operating_point.evaluate(), both from M22 (paper_versions/m22_analysis):

    c(S)  = c_mech + S*b_tok / BW_kv + KAPPA * S*b_tok / BW_hbm + S*b_tok / (L*X)
                                           c_mech = 0.50 ms at L=64, scaled by L/64
    T0(S) = W/(BW*0.75) + B*S*b_tok/BW_kv  residents read their KV too
    BW_kv = 0.64 * HBM peak                A100: 0.212 ms per 1k tok of 256 KiB

Third correction, E1 (2026-10-21, Ben approved): the rider's bytes land in HBM
while the step runs, and decode GEMMs lose bandwidth to them. E1 measured the
slowdown as KAPPA * traffic / HBM peak with KAPPA = 1.06 for bulk copies, so a
rider whose S*b_tok bytes are written into HBM costs KAPPA * S*b_tok / BW_hbm of
step time, independent of the link speed. Assumes C2C writes behave like bulk
copies; PCIe DMA measured ~4.5x worse per byte.

Fourth (2026-10-21, s6 validation): the first layer has no previous layer to
hide behind, so 1/L of each rider's transfer is exposed, S*b_tok/(L*X). Seen in
the M23 trace (H2D on the compute stream) and needed to predict the A100
real-bytes runs within 1.7%.

The knee stays at T0*X (the paper's definition); the full-step knee is not agreed.
Prints the pgfplots table embedded in figs/fig-gain-ctx.tex: the integer gain
(whole residents, whole riders) and the smooth envelope (fractional B and r).
"""
import math

from overhead_model import GH200, Q72

F64 = 3.39            # ms at L=64 (operating_point.F64["today"])
C_MECH64 = 0.50       # ms at L=64, M22 intercept of c(S)
EFF_KV = 0.64         # M22: KV read at 0.64 of A100 peak HBM bandwidth
MIN_RESIDENTS = 2
KAPPA = 1.06          # E1: GEMM slowdown = KAPPA * (HBM write traffic / HBM peak)


def point(S, m=Q72, hw=GH200, kappa=KAPPA, first_layer=True, t0_scale=1.0,
          eff_kv=None, f_ms=None):
    kv = m.b_tok * S / 1e9                      # GB per request
    att = kv / (hw.hbm * (eff_kv or EFF_KV)) * 1e3  # ms to read one request's KV
    cont = kappa * kv / hw.hbm * 1e3            # ms of step lost to its HBM writes
    first = kv / m.L / hw.link * 1e3 if first_layer else 0.0
    c = C_MECH64 * m.L / 64 + att + cont + first
    F = F64 * m.L / 64 if f_ms is None else f_ms
    cap = (hw.hbm_gb - m.w - 4.0) / kv
    link = kv / hw.link * 1e3
    tw = m.t0_ms(hw) * t0_scale

    def gain(B, r):
        t0 = tw + B * att
        return (1 + r / B) / (1 + (F + c * r) / t0) if r > 0 else 1.0

    t0s = tw + cap * att
    smooth = gain(cap, t0s / link)
    B = math.floor(cap)
    if B < MIN_RESIDENTS:
        return None
    t0i = tw + B * att
    r = math.floor(t0i / link)
    g = gain(B, r)
    eta = (g - 1) / (hw.link / hw.hbm)          # gain per unit of added bandwidth
    return dict(S=S, B=B, r=r, c=c, att=att, cont=cont, first=first, t0=t0i, integer=g,
                smooth=smooth, eta=eta, eta_smooth=(smooth - 1) / (hw.link / hw.hbm))


if __name__ == "__main__":
    print("S  integer  smooth")
    for S in range(2000, 24001, 100):
        p = point(S)
        if p:
            print("%5.1f %.4f %.4f" % (S / 1000, p["integer"], p["smooth"]))
    for S in (8192, 14724, 16300, 24000):
        print("%%", point(S))
    print("%% without contention:", point(14724, kappa=0.0, first_layer=False))
