#!/usr/bin/env python3
"""Assert the paper's tables still match what the generator produces.

Numbers in this paper have been measured, retracted and re-measured repeatedly,
and the projection tables are computed rather than measured -- so they can drift
out of agreement with model/killer_v2.py silently. This fails loudly instead.

    python3 check_tables.py          # exit 1 on any mismatch
"""
import pathlib
import re
import sys

import killer_v2 as K

TEX = pathlib.Path(__file__).resolve().parents[1] / "tex" / "sections"
S, HW = 8192, K.GH200M

failures = []


def check(label, got, want, tol=0.005):
    ok = abs(got - want) <= tol * max(abs(want), 1e-9)
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: paper {want}, generator {got:.4g}")
    if not ok:
        failures.append(label)


def model(name):
    return [m for m in K.MODELS if m.n == name][0]


print("Table 1 (01-intro, tab:endpoints)")
m = model("Qwen-32B  fp16")
u, C = m.k * S, HW.hbm - m.w - K.OVERHEAD
B_cap = C / u
T0 = K.step_ms(m, HW, S, B_cap) / 1e3
B_off = T0 * HW.x / u   # link ceiling: eta_o is a COST, not a discount
intro = (TEX / "01-intro.tex").read_text()
for want, got, label in [
    (26.5, C / 1e9, "C_kv GB"),
    (12, int(B_cap), "all-HBM batch"),
    (6, int(B_off), "all-host batch"),
    (18, int(B_cap) + int(B_off), "both batch"),
    (14.2, T0 * HW.x / 1e9, "link GB/step"),
    (33.9, T0 * 1e3, "ITL_0 ms"),
]:
    check(label, got, want, tol=0.01)
    if str(want) not in intro and f"{want}" not in intro:
        failures.append(f"{label} not present in 01-intro.tex")

print("\nTable 2 (03-model, tab:gain)")
model_tex = (TEX / "03-model.tex").read_text()
for name, cap, tput, fstar in [("Llama-8B  fp16", 1.22, 0.75, -7.0),
                               ("Qwen-32B  fp8", 1.28, 0.77, -4.1),
                               ("Qwen-32B  fp16", 1.48, 0.86, 4.0),
                               ("Llama-70B fp8", 1.67, 0.96, 9.7)]:
    r = K.analyze(model(name), S, HW, slo_ms=50.0, stage=0.0)
    check(f"{name} capacity", r["cap_gain"], cap, tol=0.01)
    check(f"{name} tput (F + delta)", r["gain_tput"], tput, tol=0.015)
    # Eq. fstar: the fixed cost this configuration could afford
    check(f"{name} F*", (r["cap_gain"] - 1) * r["T0"] - r["delta_ms"], fstar, tol=0.06)
    for v in (cap, tput):
        if f"{v:.2f}" not in model_tex:
            failures.append(f"{name} {v} missing from 03-model.tex")

print("\nClosed form (eq:gain) vs exact (eq:gain-exact) -- eta_bar must be carried")
for name in ["Llama-8B  fp16", "Qwen-32B  fp16"]:
    m = model(name)
    C, u = HW.hbm - m.w - K.OVERHEAD, m.k * S
    T0 = K.step_ms(m, HW, S, C / u) / 1e3
    eta_bar = HW.hbm / (HW.bw * T0)
    exact = 1 + T0 * HW.x / C   # eta_o is a cost, not a capacity discount
    approx = 1 + (HW.x / (HW.bw * eta_bar)) * (HW.hbm / C)
    check(f"{name} closed form == exact", approx, exact, tol=0.01)

print("\nOne value per quantity across all sections")
tex = "".join((TEX / f).read_text() for f in sorted(p.name for p in TEX.glob("*.tex")))
for pat, allowed, label in [
    (r"\\effo\s*\{?=\}?\s*([0-9.]+)", {"0.62", "1"}, "eta_o"),
    (r"(\d{3})\\,GB/s", {"450", "419", "900", "371"}, "link bandwidth"),
]:
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
print("tables agree with the generator.")
