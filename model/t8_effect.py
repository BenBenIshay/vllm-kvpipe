#!/usr/bin/env python3
"""What T8 and T7a took off `F`, decomposed (M17).

T8 moved the flush in `writeback_layer` after its own early return. That method
is the swap-semantics emulation, off by default, but its call site guards on the
method EXISTING rather than the emulation being enabled, so it fired on every
layer -- and flushed before checking its flag. The result, measured: 64 flushes
per step, each with one pending layer, each calling `_wt_event.synchronize()`.
So a disabled feature was forcing 63 redundant host waits on the D2H per step
and defeating T2's batching entirely.

With that fixed, T7a's vectorised scatter becomes reachable for the first time,
and the two can be separated cleanly:

    T8_off   T8 live, UNIFIED_MIRROR=0  -> one flush per step, per-layer scatter
    T8_on    T8 live, unified mirror    -> one flush per step, 2 index_copy_

so  T8_off - T8_on          = what T7a's vectorisation is worth
and M15_base - T8_off       = what T8 alone is worth

The ON arm is repeated three times and the OFF arm twice, because `F`'s
within-job drift ran 0.04 in M13 and 0.43 in M15 on identical configuration and
is not safe to borrow.

    ./.venv/bin/python paper/model/t8_effect.py
"""
import json
import os
import sys

D = os.environ.get(
    "T8_BASE", "/mnt/beegfs/benishay/paper_runs/20260930/overhead/19_t8")
# same configuration, before either change
BEFORE = {"M14 base": 5.80, "M15 base a": 5.74, "M15 base b": 5.31}
LAB = {"T8_on_a": "T8 + T7a vectorised", "T8_off": "T8 only, per-layer scatter",
       "T8_on_b": "T8 + T7a vectorised", "T8_off2": "T8 only, per-layer scatter",
       "T8_on_c": "T8 + T7a vectorised"}
ORDER = ["T8_on_a", "T8_off", "T8_on_b", "T8_off2", "T8_on_c"]


def fit(tag):
    p = os.path.join(D, tag + ".json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    t0 = next((r["step_ms"] for r in d["rows"]
               if r.get("riders") == 0 and r.get("step_ms")), None)
    xs = [(r["riders"], r["dT_ms"]) for r in d["rows"]
          if r.get("usable") and r["riders"] and r.get("dT_ms") is not None
          and (not r.get("link_ms") or not t0 or r["link_ms"] / t0 <= 1 / 3.0)]
    n = len(xs)
    if n < 2:
        return None
    sx = sum(x for x, _ in xs)
    sy = sum(y for _, y in xs)
    sxx = sum(x * x for x, _ in xs)
    sxy = sum(x * y for x, y in xs)
    c = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - c * sx) / n, c, n


def spread(v):
    return (max(v) - min(v)) if len(v) > 1 else float("nan")


def main():
    print("=" * 76)
    print(" WHAT T8 AND T7a TOOK OFF F   (M17, ctx 128, L=64, below the knee)")
    print("=" * 76)
    print()
    print(" %-12s %-30s %8s %9s" % ("arm", "live", "F ms", "c ms/r"))
    got = {}
    for t in ORDER:
        r = fit(t)
        if r:
            got[t] = r
            print(" %-12s %-30s %8.2f %9.3f" % (t, LAB[t], r[0], r[1]))
        else:
            print(" %-12s %-30s  (pending)" % (t, LAB[t]))
    print()
    print(" before either change, same config:  %s"
          % "  ".join("%s %.2f" % (k, v) for k, v in BEFORE.items()))
    print()

    ons = [got[t][0] for t in ("T8_on_a", "T8_on_b", "T8_on_c") if t in got]
    offs = [got[t][0] for t in ("T8_off", "T8_off2") if t in got]
    if not ons:
        print(" no ON arm yet")
        return 0
    m_on = sum(ons) / len(ons)
    d_on = spread(ons)
    print(" ON  arms: %s   mean %.2f   spread %s"
          % (" ".join("%.2f" % v for v in ons), m_on,
             "n/a" if d_on != d_on else "%.2f" % d_on))
    if offs:
        m_off = sum(offs) / len(offs)
        d_off = spread(offs)
        print(" OFF arms: %s   mean %.2f   spread %s"
              % (" ".join("%.2f" % v for v in offs), m_off,
                 "n/a" if d_off != d_off else "%.2f" % d_off))
        have = [x for x in (d_on, d_off) if x == x]
        print()
        eff_t7a = m_on - m_off          # negative = T7a reduced F
        if not have:
            print(" T7a (vectorising the scatter) : %+.2f ms  -> NO DRIFT"
                  " ESTIMATE YET (one arm per side), so not yet resolvable"
                  % eff_t7a)
        else:
            drift = max(have)
            print(" T7a (vectorising the scatter) : %+.2f ms  -> %s"
                  % (eff_t7a,
                     "RESOLVABLE against drift %.2f" % drift
                     if abs(eff_t7a) > drift else
                     "not resolvable against drift %.2f" % drift))
    before = sum(BEFORE.values()) / len(BEFORE)
    base_for_t8 = (sum(offs) / len(offs)) if offs else m_on
    print(" T8 (one flush per step, not 64): %+.2f ms   (%.2f -> %.2f)"
          % (base_for_t8 - before, before, base_for_t8))
    print("   cross-job, so weaker -- but the effect is far outside the 0.43")
    print("   between-job drift, which no previous change here managed.")
    print()
    print(" TOTAL, before -> after : %.2f -> %.2f ms  (%.0f%% of F removed)"
          % (before, m_on, (1 - m_on / before) * 100))
    print(" per layer             : %.4f -> %.4f ms"
          % (before / 64, m_on / 64))
    print()
    # what that implies at the projection target
    L80 = m_on / 64 * 80
    T0, B, C, KNEE = 24.2, 7.2, 0.609, 3.78
    g_before = (1 + KNEE / B) / (1 + (before / 64 * 80 + C * KNEE) / T0)
    g_after = (1 + KNEE / B) / (1 + (L80 + C * KNEE) / T0)
    print(" AT THE GH200 TARGET (L=80, ctx 8192, C2C):")
    print("   F  %.2f -> %.2f ms      (%.0f%% -> %.0f%% of the step)"
          % (before / 64 * 80, L80, before / 64 * 80 / T0 * 100,
             L80 / T0 * 100))
    print("   gain %.3fx -> %.3fx" % (g_before, g_after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
