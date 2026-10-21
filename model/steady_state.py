"""Steady-state decode simulation: HBM-only vs HeteroKV vs Pie's mechanism
(Ben, 2026-10-21: "run the simulation and look for a solid decode-only scenario").

A decode node with a backlog: requests arrive already prefilled (context S0),
decode O tokens, and leave. Every design sees the same queue, admits first come
first served with a look-ahead of W skipped requests, and reserves each
request's final KV (S0 + O) on admission. Step time per design:

  HBM only   T = T_w + sum(att)
  HeteroKV   T = T_w + sum(att) + F + sum_riders(c_m + kappa*kv/BW + kv/(L*X)),
             riders admitted while their KV <= T0*X (T0: residents only), and
             T >= rider bytes / X (a transfer that no longer hides is exposed)
  Pie        m of n layers of every request offloaded, chosen every step by an
             oracle (Pie reports its online controller within 2% of the best,
             pie s5.5). Each offloaded layer is swapped in before use and the
             coldest layer swapped back out (pie s3.3: "another two layer
             computation times to swap out"), so the swap-out runs in the slower
             direction and binds: K*m/n <= T*X_d2h, with pie s2.4's measured 371
             GB/s GPU to CPU (419 CPU to GPU). HBM holds K*(n-m)/n plus `stage`
             layers of the batch as swap buffer (pie s3.3: "both CPU and GPU use
             the memory space of two layers as a swapping buffer"), and each moved
             byte is written into HBM and read back out: 2*kappa*moved/BW.
'with overheads' charges both designs HeteroKV's measured F and c_m (per rider,
or per request-equivalent of offloaded KV for Pie); 'mechanism' charges neither.
"""
import math
import random
import sys

from gain_curve import C_MECH64, EFF_KV, F64, KAPPA


class HW:
    def __init__(s, name, bw, x, cap, x_d2h=None):
        s.name, s.bw, s.x, s.cap = name, bw, x, cap
        s.x_d2h = x_d2h or x


class M:
    def __init__(s, name, L, w, b_tok):
        s.name, s.L, s.w, s.b_tok = name, L, w, b_tok


GH200 = HW("GH200-96", 4000, 419, 96, x_d2h=371)
GH200_144 = HW("GH200-144", 4900, 419, 144, x_d2h=371)
Q72_16 = M("Qwen2.5-72B fp8w fp16kv", 80, 76.0, 327680)
Q72_8 = M("Qwen2.5-72B fp8w fp8kv", 80, 76.0, 163840)
Q72_BF16_144 = M("Qwen2.5-72B bf16w fp16kv", 80, 145.0, 327680)


def run(hw, m, sample, n_req=400, W=8, stage=2, overheads=True,
        seed=0, kappa=KAPPA, designs=("hbm", "hetero", "pie")):
    rnd = random.Random(seed)
    a = hw.cap - m.w - 4.0
    t_w = m.w / (hw.bw * 0.75) * 1e3
    cm = C_MECH64 * m.L / 64 if overheads else 0.0
    F = F64 * m.L / 64 if overheads else 0.0
    kvg = lambda S: m.b_tok * S / 1e9
    att = lambda S: kvg(S) / (hw.bw * EFF_KV) * 1e3
    out = {}
    for design in designs:
        rnd.seed(seed)
        queue = [sample(rnd) for _ in range(n_req)]
        qi = 0
        run_ = []          # [S_now, left, reserve_gb, tier, host_fraction]
        tok = tim = 0.0
        while queue or run_:
            # admission
            res_gb = sum(r[2] * (1 - r[4]) for r in run_)
            rid_gb = sum(r[2] * r[4] for r in run_)
            t0 = t_w + sum(att(r[0]) for r in run_ if r[3] == "res")
            has_split = any(r[3] == "split" for r in run_)
            skipped, j, taken = 0, qi, []
            while skipped <= W and j < len(queue):
                S0, O = queue[j]
                g = kvg(S0 + O)
                tier, fh = None, 0.0
                if design in ("hbm", "hetero", "split"):
                    budget = t0 * hw.x / 1e3
                    if res_gb + g <= a:
                        tier, res_gb = "res", res_gb + g
                        t0 += att(S0)
                    elif design != "hbm" and rid_gb + g <= budget:
                        tier, fh, rid_gb = "rider", 1.0, rid_gb + g
                    elif (design == "split" and not has_split and a - res_gb > 0
                          and rid_gb + g - (a - res_gb) <= budget):
                        # one request fills the HBM left over with its first
                        # layers and streams the rest as a rider
                        fh = (g - (a - res_gb)) / g
                        tier, has_split = "split", True
                        rid_gb, res_gb = rid_gb + g * fh, a
                else:   # pie: one pool, checked below with the oracle split
                    K = res_gb + rid_gb + g
                    tc = t_w + sum(att(r[0]) for r in run_) + att(S0)
                    if pie_fits(K, a, tc * hw.x_d2h / 1e3, m.L, stage):
                        tier, res_gb = "pie", res_gb + g
                if tier:
                    taken.append(j)
                    run_.append([S0, O, g, tier, fh])
                else:
                    skipped += 1
                j += 1
            if taken:
                for k in reversed(taken):
                    queue.pop(k)
                # qi stays: the queue head moves as admitted entries are removed
            # step time
            T = t_w + sum(att(r[0]) for r in run_)
            if design in ("hetero", "split"):
                riders = [r for r in run_ if r[4] > 0]
                if riders:
                    T += F + sum(cm + kappa * kvg(r[0]) * r[4] / hw.bw * 1e3
                                 + (kvg(r[0]) / (m.L * hw.x) * 1e3 if r[3] == "rider"
                                    else 0.0) for r in riders)
                    T = max(T, sum(kvg(r[0]) * r[4] for r in riders) / hw.x * 1e3)
            elif design == "pie":
                K = sum(kvg(r[0]) for r in run_)
                moved = pie_moved(K, a, T * hw.x_d2h / 1e3, m.L, stage)
                if moved > 0:
                    T += 2 * kappa * moved / hw.bw * 1e3
                    if overheads:
                        T += F + cm * moved / (K / len(run_))
                    T = max(T, moved / hw.x_d2h * 1e3)
            tok += len(run_)
            tim += T
            for r in run_:
                r[0] += 1
                r[1] -= 1
            run_ = [r for r in run_ if r[1] > 0]
        assert not queue, design
        out[design] = tok / tim * 1e3
    return out


def pie_split(K, a, L, stage):
    """Smallest m (layers offloaded) such that the HBM part fits."""
    for mm in range(0, L):
        if K * (L - mm) / L + (K * stage / L if mm else 0.0) <= a:
            return mm
    return None


def pie_fits(K, a, budget, L, stage):
    mm = pie_split(K, a, L, stage)
    return mm is not None and K * mm / L <= budget


def pie_moved(K, a, budget, L, stage):
    mm = pie_split(K, a, L, stage)
    return 0.0 if not mm else K * mm / L


def fixed(S, O):
    return lambda r: (S, O)


def lognorm(med, sigma, O, lo=1000, hi=46000):
    return lambda r: (int(min(hi, max(lo, med * math.exp(sigma * r.gauss(0, 1))))), O)


def table(cases, **kw):
    print("%-28s %-26s %8s %8s %8s %7s %7s %7s" % (
        "hardware/model", "workload", "hbm", "hetero", "pie", "G_het", "G_pie", "margin"))
    for hw, m, name, sm in cases:
        o = run(hw, m, sm, **kw)
        gh, gp = o["hetero"] / o["hbm"], o["pie"] / o["hbm"]
        print("%-28s %-26s %8.1f %8.1f %8.1f %7.3f %7.3f %+6.1f%%" % (
            hw.name + " " + m.name.split()[1], name, o["hbm"], o["hetero"], o["pie"],
            gh, gp, (o["hetero"] / o["pie"] - 1) * 100))


if __name__ == "__main__":
    cases = [
        (GH200, Q72_16, "fixed 14.7k, O 512", fixed(14724, 512)),
        (GH200, Q72_16, "fixed 8k, O 512", fixed(8192, 512)),
        (GH200, Q72_16, "fixed 24k, O 512", fixed(24000, 512)),
        (GH200, Q72_16, "lognorm med 8k s.8, O 512", lognorm(8000, 0.8, 512)),
        (GH200, Q72_16, "lognorm med 12k s1, O 256", lognorm(12000, 1.0, 256)),
        (GH200, Q72_8, "fixed 14.7k, O 512", fixed(14724, 512)),
        (GH200, Q72_8, "lognorm med 12k s1, O 256", lognorm(12000, 1.0, 256)),
    ]
    for ov in (True, False):
        for st in (4, 0):
            print("\n== overheads=%s, Pie staging layers=%d" % (ov, st))
            table(cases, overheads=ov, stage=st)
