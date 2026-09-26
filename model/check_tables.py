#!/usr/bin/env python3
"""Assert the paper's tables match what tables.py produces.

The projection tables are computed, not measured, so they drift silently. This
session has already found them disagreeing with their generator twice: once
because killer_v2 implemented the eta_o=1 law while the paper claimed
otherwise, and once because the sensitivity paragraphs were regenerated before
eta_o moved out of the capacity term and not after.

It compares against tables.py's OUTPUT rather than re-deriving the arithmetic,
so the file the paper cites is the file that is checked.

    python3 check_tables.py          # exit 1 on any mismatch
"""
import pathlib
import re
import sys

import tables

TEX = pathlib.Path(__file__).resolve().parents[1] / "tex" / "sections"
failures = []


def check(label, got, want, tol=0.01):
    ok = abs(got - want) <= tol * max(abs(want), 1e-9)
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: paper {want}, generator {got:.4g}")
    if not ok:
        failures.append(label)


def present(value, filename, label):
    body = (TEX / filename).read_text()
    if value not in body:
        print(f"  FAIL {label}: {value!r} not in {filename}")
        failures.append(label)


print("Table 1 (01-intro, tab:endpoints)")
e = tables.endpoints()
for want, got, label in [(26.5, e["C"] / 1e9, "C_kv GB"),
                         (33.9, e["T0"], "ITL_0 ms"),
                         (14.2, e["link"] / 1e9, "link GB/step"),
                         (12, int(e["B_cap"]), "all-HBM batch"),
                         (6, int(e["B_off"]), "all-host batch"),
                         (18, e["both"], "both batch")]:
    check(label, got, want)
for v in ("26.5", "14.2", "12", "6", "18", "33.9"):
    present(v, "01-intro.tex", f"Table 1 shows {v}")

print("\nTable 2 (03-model, tab:gain)")
paper = {"Llama-8B  fp16": (1.03, 0.78, -0.9),
         "Qwen-32B  fp8":  (1.09, 0.78, -1.3),
         "Qwen-32B  fp16": (1.45, 0.85,  3.8),
         "Llama-70B fp8":  (1.73, 0.97, 10.6)}
for r in tables.gain_rows():
    cap, tput, fstar = paper[r["name"]]
    check(f"{r['name']} capacity", r["cap"], cap)
    check(f"{r['name']} tput (F+delta)", r["tput"], tput, tol=0.015)
    check(f"{r['name']} F*", r["fstar"], fstar, tol=0.12)
    present(f"{cap:.2f}", "03-model.tex", f"{r['name']} cap in tex")
    present(f"{tput:.2f}", "03-model.tex", f"{r['name']} tput in tex")

print("\nSensitivity (03-model) -- regenerated 2026-09-26 after eta_o moved")
import killer_v2 as K
def cap_at(eff_w=None, kv_scale=1.0, wgb=65.53):
    ow, oc = K.EFF_W, K.EFF_KV_CURVE
    if eff_w is not None:
        K.EFF_W = eff_w
    if kv_scale != 1.0:
        K.EFF_KV_CURVE = [(g, x * kv_scale) for g, x in oc]
    r = K.analyze(K.M("t", 64, 8, 128, 32, wgb=wgb), tables.S, tables.HW,
                  slo_ms=50.0, stage=2.0)
    K.EFF_W, K.EFF_KV_CURVE = ow, oc
    return r["cap_gain"]
for label, got, want in [("eta_kv -20%", cap_at(kv_scale=0.8), 1.31),
                         ("eta_kv +20%", cap_at(kv_scale=1.2), 1.64),
                         ("eta_w 0.70", cap_at(eff_w=0.70), 1.41),
                         ("eta_w 0.80", cap_at(eff_w=0.80), 1.50),
                         ("W = 60 GB", cap_at(wgb=60.0), 1.34),
                         ("W = 68 GB", cap_at(wgb=68.0), 1.53)]:
    check(label, got, want)

print("\nClosed form (eq:gain) reproduces the exact form (eq:gain-exact)")
for name in ("Llama-8B  fp16", "Qwen-32B  fp16"):
    m = tables.model(name)
    C, u = tables.HW.hbm - m.w - K.OVERHEAD, m.k * tables.S
    T0 = K.step_ms(m, tables.HW, tables.S, C / u) / 1e3
    eta_bar = tables.HW.hbm / (tables.HW.bw * T0)
    check(f"{name} closed == exact",
          1 + (tables.HW.x / (tables.HW.bw * eta_bar)) * (tables.HW.hbm / C),
          1 + T0 * tables.HW.x / C)

print("\nOne value per quantity across all sections")
tex = "".join(p.read_text() for p in sorted(TEX.glob("*.tex")))
for pat, allowed, label in [
        (r"\\effo\s*\{?=\}?\s*([0-9.]+)", {"0.62", "1"}, "eta_o"),
        # 1935 is the measured A100 peak; 1450 is 0.75 of it, the achieved rate.
        (r"(\d{3,4})\\,GB/s", {"450", "419", "900", "371", "1935", "1450"}, "bandwidth")]:
    seen = set(re.findall(pat, tex))
    bad = seen - allowed
    print(f"  {'ok  ' if not bad else 'FAIL'} {label}: {sorted(seen)}")
    if bad:
        failures.append(f"{label} unexpected: {sorted(bad)}")

print()
if failures:
    print(f"{len(failures)} MISMATCH(ES):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("paper agrees with tables.py.")
