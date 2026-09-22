#!/usr/bin/env python3
"""Emit the paper's projection tables from killer_v2, so a reader can reproduce
them. Table 1 = the intro's endpoint comparison; Table 2 = the gain table.

    python3 tables.py            # both tables, with the inputs printed
"""
import killer_v2 as K

S = 8192
HW = K.GH200M           # measured link, not spec
ETA = K.ETA_O


def endpoints(m, hw=HW, S=S, eta_o=ETA):
    """The intro's three rows: all-HBM, all-host, and the interior point."""
    u = m.k * S                                   # bytes of KV per request
    C = hw.hbm - m.w - K.OVERHEAD                 # HBM left for KV
    B_cap = C / u
    T0 = K.step_ms(m, hw, S, B_cap) / 1e3         # full-pool step, seconds
    link = T0 * hw.x * eta_o                      # bytes the link delivers, hidden
    B_off = link / u
    return dict(C=C, u=u, B_cap=B_cap, T0=T0 * 1e3, link=link, B_off=B_off,
                both=B_cap + B_off)


if __name__ == "__main__":
    m = [x for x in K.MODELS if x.n.startswith("Qwen-32B")][0]
    e = endpoints(m)
    print(f"TABLE 1 -- {m.n}, S={S}, {HW.n}, X={HW.x/1e9:.0f} GB/s measured, "
          f"eta_o={ETA}")
    print(f"  inputs: W={m.w/1e9:.2f} GB (measured checkpoint), "
          f"reserve={K.OVERHEAD/1e9:.0f} GB, C={e['C']/1e9:.2f} GB, "
          f"KV/req={e['u']/1e9:.3f} GB, ITL_0={e['T0']:.1f} ms")
    print(f"  all-HBM   {e['C']/1e9:5.1f} GB HBM, 0 link      -> {e['B_cap']:.2f} -> {int(e['B_cap'])}")
    print(f"  all-host   0 HBM, {e['link']/1e9:5.2f} GB link   -> {e['B_off']:.2f} -> {int(e['B_off'])}")
    print(f"  both      {e['C']/1e9:5.1f} + {e['link']/1e9:.2f} GB       -> {e['both']:.2f} -> {int(e['B_cap'])+int(e['B_off'])}")

    print(f"\nTABLE 2 -- gain, S={S}, {HW.n}, eta_o={ETA}")
    print(f"  {'model':22} {'W':>7} {'C':>7} {'HBM/C':>6} {'ITL_0':>7} {'cap':>6} {'tput':>6}")
    for mm in K.MODELS:
        r = K.analyze(mm, S, HW, slo_ms=50.0, stage=0.0)
        if not r:
            continue
        C = HW.hbm - mm.w - K.OVERHEAD
        print(f"  {mm.n:22} {mm.w/1e9:6.1f}G {C/1e9:6.1f}G {HW.hbm/C:6.2f} "
              f"{r['T0']:6.1f}ms {r['B_tot']/r['B0'] if r['B0'] else 0:5.2f}x {r['gain']:5.2f}x")
