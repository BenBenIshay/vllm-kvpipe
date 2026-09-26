#!/usr/bin/env python3
"""Emit the paper's projection tables. This is the ONLY generator: the paper
cites this file, so it must reproduce what the paper prints.

    python3 tables.py          # both tables, with every input named
    python3 tables.py --json   # machine-readable, for check_tables.py
"""
import json
import sys

import killer_v2 as K

S = 8192
HW = K.GH200M                      # measured 419 GB/s link, not the 450 spec
ROWS = ["Llama-8B  fp16", "Qwen-32B  fp8", "Qwen-32B  fp16", "Llama-70B fp8"]


def model(name):
    return [m for m in K.MODELS if m.n == name][0]


def endpoints(m=None):
    """Table 1: the three single- and two-memory operating points.

    eta_o does NOT appear here. The budget is a link ceiling -- a step of length
    T carries at most T*X bytes, whatever fraction of them hides -- so eta_o is
    charged once, in the throughput form, and not twice.
    """
    m = m or model("Qwen-32B  fp16")
    u = m.k * S
    C = HW.hbm - m.w - K.OVERHEAD
    B_cap = C / u
    T0 = K.step_ms(m, HW, S, B_cap) / 1e3
    link = T0 * HW.x
    B_off = link / u
    return dict(C=C, u=u, B_cap=B_cap, T0=T0 * 1e3, link=link, B_off=B_off,
                both=int(B_cap) + int(B_off))


def gain_rows():
    """Table 2: capacity, the throughput form charging F and Delta, and F*."""
    out = []
    for name in ROWS:
        m = model(name)
        # stage=2: a rider's KV crosses the link into a rolling slot in HBM (a
        # write) and is then read from that slot by attention. Two HBM passes
        # per byte per step. Charging 0, as we did until 2026-09-26, credits the
        # rider with device bandwidth it actually consumes.
        r = K.analyze(m, S, HW, slo_ms=50.0, stage=2.0)
        C = HW.hbm - m.w - K.OVERHEAD
        out.append(dict(name=name, W=m.w / 1e9, C=C / 1e9, amp=HW.hbm / C,
                        T0=r["T0"], cap=r["cap_gain"], delta=r["delta_ms"],
                        tput=r["gain_tput"],
                        fstar=(r["cap_gain"] - 1) * r["T0"] - r["delta_ms"]))
    return out


if __name__ == "__main__":
    e, rows = endpoints(), gain_rows()
    if "--json" in sys.argv:
        print(json.dumps(dict(endpoints=e, gain=rows), indent=1))
        sys.exit(0)

    print(f"TABLE 1 -- Qwen2.5-32B fp16, S={S}, {HW.n}, X={HW.x/1e9:.0f} GB/s measured")
    print(f"  inputs: W={model('Qwen-32B  fp16').w/1e9:.2f} GB measured, "
          f"reserve={K.OVERHEAD/1e9:.0f} GB, C={e['C']/1e9:.2f} GB, "
          f"KV/req={e['u']/1e9:.3f} GB, ITL_0={e['T0']:.1f} ms")
    print(f"  all-HBM   {e['C']/1e9:5.1f} GB HBM, 0 link       -> {e['B_cap']:5.2f} -> {int(e['B_cap'])}")
    print(f"  all-host   0 HBM, {e['link']/1e9:5.1f} GB link    -> {e['B_off']:5.2f} -> {int(e['B_off'])}")
    print(f"  both      {e['C']/1e9:5.1f} + {e['link']/1e9:.1f} GB        -> {'':5} -> {e['both']}")

    print(f"\nTABLE 2 -- S={S}, {HW.n}, eta_o={K.ETA_O}, F={K.F_FIXED_MS} ms")
    print(f"  {'model':18}{'W':>7}{'C':>7}{'HBM/C':>7}{'ITL_0':>8}"
          f"{'cap':>7}{'delta':>8}{'tput':>7}{'F*':>8}")
    for r in rows:
        print(f"  {r['name']:18}{r['W']:6.1f}G{r['C']:6.1f}G{r['amp']:7.2f}"
              f"{r['T0']:7.1f}ms{r['cap']:7.2f}{r['delta']:7.1f}ms{r['tput']:7.2f}"
              f"{r['fstar']:+7.1f}ms")
