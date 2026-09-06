#!/usr/bin/env python3
"""v2: SLO-gated. Is the batch actually capacity-limited, and what do riders buy?

Baseline batch B0 = largest batch that fits HBM *and* meets the ITL SLO.
If HBM is the binding one, there is ITL headroom -> ride-along requests can
fill it, bounded by (a) C2C hide budget and (b) the SLO itself.
"""

EFF_BW, EFF_FLOPS, OVERHEAD = 0.75, 0.70, 4e9


class HW:
    def __init__(s, n, hbm, bw, x, fl):
        s.n, s.hbm, s.bw, s.x, s.fl = n, hbm*1e9, bw*1e9, x*1e9, fl


GH200 = HW("GH200-96",  96, 4000, 450, 495e12)
GH200H = HW("GH200-144", 141, 4800, 450, 495e12)
A100 = HW("A100-80",    80, 2039,  25, 312e12)
H100 = HW("H100-80(PCIe5)", 80, 3350, 55, 495e12)


class M:
    """A model's memory and compute footprint.

    `wgb` overrides the weight bytes with a MEASURED checkpoint size. It exists
    because parameter count and checkpoint size are not the same number: a
    quantized checkpoint also carries scales and zero-points, so deriving the
    weight footprint from `pb * wb` understates it. Compute still uses `pb`,
    since the FLOP count follows the parameter count, not the file size.
    """

    def __init__(s, n, L, kvh, hd, pb, wb=2, tp=1, kvb=2, wgb=None):
        s.n, s.L, s.tp = n, L, tp
        s.k = L*2*(kvh/tp)*hd*kvb
        s.w = (wgb*1e9/tp) if wgb is not None else (pb*1e9*wb/tp)
        s.p = pb*1e9/tp


MODELS = [
    # MEASURED weight footprints carry wgb=; the rest are nominal (params x
    # bytes) and should be measured before anything depends on them, because
    # the weight term sits in the capacity denominator: a heavier checkpoint
    # leaves LESS room for KV and therefore raises the projected gain.
    M("Llama-8B  fp16",      32, 8, 128,  8),
    M("Qwen-32B  fp16",      64, 8, 128, 32),
    # RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8: 72.7 GB across 15 shards,
    # measured from the checkpoint 2026-08-12 (paper/GH200_EXPERIMENT_PLAN.md
    # section 2). Nominal 70x1 byte = 70 GB understates it by 2.7 GB.
    M("Llama-70B fp8",       80, 8, 128, 70, wb=1, wgb=72.7),
    M("Llama-70B fp16 TP2",  80, 8, 128, 70, tp=2),
    M("Llama-405B fp8 TP4", 126, 8, 128, 405, wb=1, tp=4),
]


# Measured paged-KV read efficiency against KV working set, A100-80 / FA-2 era
# (paper/model/p4_*.json, n=3 repeats). It is NOT a constant: it degrades ~1.6x
# across the range, and it degrades WORST exactly at the capacity-bound
# operating point this model is about.
#
# Charging weights and KV at one efficiency -- as this script did until
# 2026-09-06 -- makes a full pool take HBM/BW seconds regardless of model, i.e.
# it predicts a model-INDEPENDENT full-pool ITL. Measurement says otherwise
# (79.3 ms for Llama-8B vs 55.7 ms for Qwen-32B on A100), and the paper reports
# that as a finding. The script has to model what the paper claims.
EFF_KV_CURVE = [(4.3, 0.77), (8.6, 0.72), (17.2, 0.55),
                (34.4, 0.55), (43.0, 0.49), (51.5, 0.46)]


def eff_kv(ws_bytes):
    """Interpolate the measured curve in log(working set); clamp at the ends."""
    import math
    gb = max(ws_bytes, 1.0)/1e9
    pts = EFF_KV_CURVE
    if gb <= pts[0][0]:
        return pts[0][1]
    if gb >= pts[-1][0]:
        return pts[-1][1]
    for (g0, e0), (g1, e1) in zip(pts, pts[1:]):
        if g0 <= gb <= g1:
            t = (math.log(gb) - math.log(g0))/(math.log(g1) - math.log(g0))
            return e0 + t*(e1 - e0)
    return pts[-1][1]


def step_ms(m, hw, S, B_res, B_off=0.0, stage=1.0, f=0.0):
    """max(memory, compute) time for one decode step, ms.

    Weights and paged KV are charged at DIFFERENT efficiencies; see EFF_KV_CURVE.
    """
    kv_bytes = (B_res + stage*B_off)*m.k*S
    t_mem = m.w/(hw.bw*EFF_BW) + kv_bytes/(hw.bw*eff_kv(kv_bytes))
    t_cmp = 2*m.p*(B_res+B_off)/(hw.fl*EFF_FLOPS)
    return max(t_mem, t_cmp)*1e3


def analyze(m, S, hw, slo_ms, kwin=8, stage=1.0):
    u = m.k*S
    C = hw.hbm - m.w - OVERHEAD
    if C <= 0 or u <= 0:
        return None
    f = kwin/m.L
    B_cap = C/u
    # largest B meeting SLO (bisect on the max() model)
    lo, hi = 0.0, max(B_cap, 1.0)*4
    for _ in range(60):
        mid = (lo+hi)/2
        if step_ms(m, hw, S, mid) <= slo_ms:
            lo = mid
        else:
            hi = mid
    B_slo = lo
    B0 = min(B_cap, B_slo)
    T0 = step_ms(m, hw, S, B0)
    cap_bound = B_cap < B_slo - 1e-9

    # riders: displace f of a resident each; must hide in T and keep T <= SLO
    best = 0.0
    for _ in range(60):
        n = best + max(B_cap, 1.0)/64
        B_res = max(0.0, C/u - n*f)
        T = step_ms(m, hw, S, B_res, n, stage=stage, f=f)
        if T <= slo_ms and n*u <= T*hw.x/1e3*1e3*(T/T) and n*u <= (T/1e3)*hw.x:
            best = n
        else:
            break
    n = best if cap_bound else 0.0      # latency-bound => no headroom to fill
    B_res = min(B0, max(0.0, C/u - n*f))
    T = step_ms(m, hw, S, B_res, n, stage=stage, f=f) if n > 0 else T0
    B_tot = B_res + n
    return dict(B_cap=B_cap, B_slo=B_slo, B0=B0, T0=T0, capb=cap_bound,
                n=n, B_tot=B_tot, T=T,
                gain=(B_tot/T)/(B0/T0) if B0 > 0 else 0.0)


def report(hw, slo, stage=1.0):
    print(f"\n{'='*100}")
    print(f"{hw.n}  |  ITL SLO {slo} ms  |  HBM-read time {hw.hbm/(hw.bw*EFF_BW)*1e3:.0f} ms"
          f"  |  link {hw.x/1e9:.0f} GB/s  |  stage x{stage}")
    print('='*100)
    print(f"{'model':<22}{'ctx':>7}{'B_cap':>7}{'B_slo':>7}  {'bound':<9}"
          f"{'B0':>5}{'ITL0':>7}{'riders':>8}{'B_tot':>7}{'ITL':>7}{'tput':>7}")
    for m in MODELS:
        for S in (4096, 8192, 16384, 32768):
            r = analyze(m, S, hw, slo, stage=stage)
            if r is None or r['B0'] < 1:
                continue
            print(f"{m.n:<22}{S:>7}{r['B_cap']:>7.0f}{r['B_slo']:>7.0f}  "
                  f"{('CAPACITY' if r['capb'] else 'latency'):<9}"
                  f"{r['B0']:>5.0f}{r['T0']:>6.0f}m{r['n']:>8.1f}"
                  f"{r['B_tot']:>7.0f}{r['T']:>6.0f}m{r['gain']:>6.2f}x")


if __name__ == "__main__":
    for slo in (25, 50, 100, 200):
        report(GH200, slo)
    report(GH200H, 100)
    report(A100, 100)
    report(H100, 100)
    print("\n--- staged-into-HBM (write+read) instead of zero-copy, SLO=100 ---")
    report(GH200, 100, stage=2.0)
