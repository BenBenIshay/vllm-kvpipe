"""Workload shift (Ben, 2026-10-22): HeteroKV's per-step admission against Pie's
online controller when the context mix changes during the run.

Same node, costs and overheads as steady_state.run(). Designs:
  hbm, hetero  as in steady_state.run()
  pie          oracle split, re-chosen at every admission (steady_state.run())
  pie_ctl      Pie's controller (pie s3.4.2). State: m layers offloaded.
               Admission is by capacity at the current m only (the block
               manager, pie s4.1); the link is the controller's job:
               - more swapping: admission is blocked on capacity and the swap
                 leaves the link idle ("If no on-CPU layers are required, it
                 indicates the current expansion is too small"), m += 1;
               - less swapping: the swap no longer hides ("If the model has to
                 wait for swapping to finish ... Pie reduces the expansion"):
                 admission is held to the capacity at m - 1 (pie s4.1: "Pie
                 avoids scheduling any requests that use blocks beyond the new
                 decreased size") and m drops once the running requests fit.
                 Until then the exposed swap is paid every step.
               At most one change every `adapt` steps. Pie gives no interval;
               it covers detection plus the reallocation of every layer's
               cache (pie s4.1: "Once every layer has undergone at least one
               swap"), so it is swept.
Workload: phases of `phase` requests alternating between two context mixes.
"""
import math
import random

from gain_curve import C_MECH64, EFF_KV, F64, KAPPA
from steady_state import GH200, pie_fits, pie_moved, pie_split


def queue_of(mixes, phase, n_req, seed):
    rnd = random.Random(seed)
    q = []
    for i in range(n_req):
        med, sig, O, lo, hi = mixes[(i // phase) % len(mixes)]
        S = int(min(hi, max(lo, med * math.exp(sig * rnd.gauss(0, 1)))))
        q.append((S, O))
    return q


def cap_at(a, L, stage, mm):
    """KV the HBM holds with mm of L layers offloaded (pie s3.1: n*a/(n-m))."""
    if mm == 0:
        return a
    return a / ((L - mm) / L + stage / L)


def run_dyn(hw, m, queue0, W=8, stage=2, overheads=True, kappa=KAPPA, adapt=8,
            designs=("hbm", "hetero", "pie", "pie_ctl")):
    a = hw.cap - m.w - 4.0
    t_w = m.w / (hw.bw * 0.75) * 1e3
    cm = C_MECH64 * m.L / 64 if overheads else 0.0
    F = F64 * m.L / 64 if overheads else 0.0
    kvg = lambda S: m.b_tok * S / 1e9
    att = lambda S: kvg(S) / (hw.bw * EFF_KV) * 1e3
    xp = min(hw.x, hw.x_d2h)
    out, trace = {}, {}
    for design in designs:
        queue = list(queue0)
        run_ = []          # [S_now, left, reserve_gb, tier]
        tok = tim = 0.0
        mm, last_change, step, stalls = 0, -10**9, 0, 0
        shrink_to = 0
        while queue or run_:
            res_gb = sum(r[2] for r in run_ if r[3] != "rider")
            rid_gb = sum(r[2] for r in run_ if r[3] == "rider")
            t0 = t_w + sum(att(r[0]) for r in run_ if r[3] == "res")
            skipped, j, taken, blocked = 0, 0, [], False
            while skipped <= W and j < len(queue):
                S0, O = queue[j]
                g = kvg(S0 + O)
                tier = None
                if design in ("hbm", "hetero"):
                    if res_gb + g <= a:
                        tier, res_gb = "res", res_gb + g
                        t0 += att(S0)
                    elif design == "hetero" and rid_gb + g <= t0 * hw.x / 1e3:
                        tier, rid_gb = "rider", rid_gb + g
                elif design == "pie":
                    tc = t_w + sum(att(r[0]) for r in run_) + att(S0)
                    if pie_fits(res_gb + g, a, tc * xp / 1e3, m.L, stage):
                        tier, res_gb = "pie", res_gb + g
                else:
                    if res_gb + g <= cap_at(a, m.L, stage, min(mm, shrink_to)):
                        tier, res_gb = "pie", res_gb + g
                    else:
                        blocked = True
                if tier:
                    taken.append(j)
                    run_.append([S0, O, g, tier])
                else:
                    skipped += 1
                j += 1
            for k in reversed(taken):
                queue.pop(k)
            T = t_w + sum(att(r[0]) for r in run_)
            K = sum(kvg(r[0]) for r in run_)
            if design == "hetero":
                riders = [r for r in run_ if r[3] == "rider"]
                if riders:
                    T += F + sum(cm + kappa * kvg(r[0]) / hw.bw * 1e3
                                 + kvg(r[0]) / (m.L * hw.x) * 1e3 for r in riders)
                    T = max(T, sum(kvg(r[0]) for r in riders) / hw.x * 1e3)
            elif design in ("pie", "pie_ctl"):
                exposed, hidden = False, T
                if design == "pie":
                    moved = pie_moved(K, a, T * xp / 1e3, m.L, stage)
                else:
                    moved = K * mm / m.L
                if moved > 0:
                    T += 2 * kappa * moved / hw.bw * 1e3
                    if overheads:
                        T += F + cm * moved / (K / len(run_))
                    hidden = T
                    T = max(T, moved / xp * 1e3)
                    exposed = T > hidden
                    stalls += exposed
                if design == "pie_ctl":
                    if shrink_to < mm and res_gb <= cap_at(a, m.L, stage, shrink_to):
                        mm, last_change = shrink_to, step
                    elif shrink_to == mm and step - last_change >= adapt:
                        # an increase is tried only if one more layer would
                        # still hide now (Pie tries it and backs off)
                        if exposed and mm > 0:
                            shrink_to, last_change = mm - 1, step
                        elif blocked and (K * (mm + 1) / m.L) / xp * 1e3 <= hidden \
                                and mm + 1 < m.L - stage:
                            mm = shrink_to = mm + 1
                            last_change = step
            tok += len(run_)
            tim += T
            step += 1
            for r in run_:
                r[0] += 1
                r[1] -= 1
            run_ = [r for r in run_ if r[1] > 0]
        out[design] = tok / tim * 1e3
        trace[design] = stalls
    return out, trace

