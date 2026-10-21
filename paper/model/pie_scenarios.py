"""HeteroKV against Pie's mechanism in the two cases Ben asked about (2026-10-21):
(A) a PCIe link, where E1 measured copy contention 4.3x worse per byte, and
(B) mixed contexts on GH200, where HeteroKV packs two bins (HBM, link) with whole
requests and Pie packs one (HBM + link) with a layer split of every request.

Pie, as its paper describes it (pie s3.1, s3.2): m of n layers of every request
in host memory, swapped in ahead of use and swapped out after, with m set online
so the swap hides behind compute. Modelled: every moved byte is written into HBM
and later read out of it, 2*kappa per byte (the D2H read's contention is NOT
measured; E1 measured H2D writes only); no exposed first layer; whole requests
admitted while total KV <= HBM + T0*X. 'mech' omits implementation overheads;
'ovh' gives Pie HeteroKV's F and c_m per added request-equivalent.
"""
import random

from gain_curve import C_MECH64, EFF_KV, F64, KAPPA
from overhead_model import GH200, Q72


def step(t_w, att_ms, kv_gb, hbm_gbs, link_gbs, L, kappa, cm, F, design, extra):
    """Step time for a batch; `extra` is the KV (GB per request) that crosses the
    link: the riders' whole KV for HeteroKV, the offloaded share for Pie."""
    t = t_w + sum(att_ms)
    if design == "hetero":
        if extra:
            t += F + sum(cm + kappa * e / hbm_gbs * 1e3 + e / (L * link_gbs) * 1e3
                         for e in extra)
    elif design.startswith("pie"):
        moved = sum(extra)
        if moved:
            t += 2 * kappa * moved / hbm_gbs * 1e3
            if design == "pie_ovh":
                t += F + cm * moved / (sum(kv_gb) / len(kv_gb))
    return t


# ---------------------------------------------------------------- (A) PCIe, A100
def pcie():
    BT, L, X, BW, W = 262144, 64, 25.1, 1935, 65.53   # Qwen2.5-32B fp16, A100-80
    a = 80 - W - 4.0
    t_w = W / (BW * 0.75) * 1e3
    cm, slope, F, k_pcie = 0.497, 0.2121, 5.0, 4.55
    print("(A) A100 PCIe, Qwen2.5-32B fp16, HBM for KV %.1f GB, kappa %.2f" % (a, k_pcie))
    print("  S    B  r   hetero   pie_mech  pie_ovh   hetero_mech")
    for S in (1024, 2048, 3072, 4096):
        kv = S * BT / 1e9
        att = slope * S / 1000
        B = int(a // kv)
        t0 = t_w + B * att
        r = int(t0 * X / 1e3 // kv)
        if r == 0:
            print("%5d %3d  0   no rider fits" % (S, B)); continue
        base = B / t0
        g = {}
        for d, c_, F_ in (("hetero", cm, F), ("hetero_mech", 0, 0)):
            g[d] = (B + r) / step(t_w, [att] * (B + r), [kv] * (B + r), BW, X, L,
                                  k_pcie, c_, F_, "hetero", [kv] * r) / base
        for d in ("pie_mech", "pie_ovh"):
            g[d] = (B + r) / step(t_w, [att] * (B + r), [kv] * (B + r), BW, X, L,
                                  k_pcie, cm, F, d, [kv] * r) / base
        print("%5d %3d %2d   %.3f    %.3f     %.3f    %.3f" % (
            S, B, r, g["hetero"], g["pie_mech"], g["pie_ovh"], g["hetero_mech"]))


# ------------------------------------------------------- (B) mixed contexts, GH200
def mixed(draws=2000, lo=2000, hi=32000, seed=1):
    m, hw = Q72, GH200
    a = hw.hbm_gb - m.w - 4.0
    t_w = m.t0_ms(hw)
    cm, F = C_MECH64 * m.L / 64, F64 * m.L / 64
    rnd = random.Random(seed)
    acc = {k: 0.0 for k in ("hetero", "hetero_mech", "pie_mech", "pie_ovh")}
    n = 0
    for _ in range(draws):
        q = [int(lo * (hi / lo) ** rnd.random()) for _ in range(64)]   # log-uniform
        kvs = [m.b_tok * S / 1e9 for S in q]
        atts = [kv / (hw.hbm * EFF_KV) * 1e3 for kv in kvs]
        # HBM only: FCFS until the next request does not fit
        i, used = 0, 0.0
        while used + kvs[i] <= a:
            used += kvs[i]; i += 1
        B = i
        if B == 0:
            continue
        t0 = step(t_w, atts[:B], kvs[:B], hw.hbm, hw.link, m.L, KAPPA, 0, 0, "none", [])
        base = B / t0
        budget = t0 * hw.link / 1e3                                    # GB per step
        # HeteroKV: the next requests ride while their KV fits the link budget
        j, lk = B, 0.0
        while lk + kvs[j] <= budget:
            lk += kvs[j]; j += 1
        riders = kvs[B:j]
        for d, c_, F_ in (("hetero", cm, F), ("hetero_mech", 0, 0)):
            acc[d] += j / step(t_w, atts[:j], kvs[:j], hw.hbm, hw.link, m.L, KAPPA,
                               c_, F_, "hetero", riders) / base
        # Pie: one bin of HBM + link budget, every request split by layer
        p, tot = 0, 0.0
        while tot + kvs[p] <= a + budget:
            tot += kvs[p]; p += 1
        moved = [max(0.0, tot - a) / p] * p
        for d in ("pie_mech", "pie_ovh"):
            acc[d] += p / step(t_w, atts[:p], kvs[:p], hw.hbm, hw.link, m.L, KAPPA,
                               cm, F, d, moved) / base
        n += 1
    print("(B) GH200, Qwen2.5-72B, contexts log-uniform %d-%d, %d draws, FCFS" % (lo, hi, n))
    for k, v in acc.items():
        print("  %-12s mean gain %.3f" % (k, v / n))


if __name__ == "__main__":
    pcie()
    mixed()
    mixed(lo=4000, hi=16000)
