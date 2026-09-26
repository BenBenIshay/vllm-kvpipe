#!/usr/bin/env python3
"""Derive EFF_KV_CURVE from the measured data rather than hardcoding it.

eta_kv is the projection's dominant sensitivity (a +-20% error spans x1.31 to
x1.47 on the headline), and until 2026-09-26 the curve lived as a literal in
killer_v2.py with no generator and no assertion. This reproduces it, so a
reviewer can check it and check_tables.py can pin it.

Method, which matters because a different one gives different numbers: subtract
the weight term W/(peak*eta_w) from each measured step, and attribute the
remainder to the KV working set. Points are Llama-3.1-8B only -- Qwen2.5-32B's
sweep reaches a 6.4 GB working set, below the range where the degradation lives.

    python3 eff_kv.py
"""
import json
import pathlib

EFF_W = 0.75
SRC = pathlib.Path(__file__).with_name("p4_llama8b_a100_rep3.json")


def curve():
    d = json.load(open(SRC))
    m = d["meta"]
    W, peak, k = m["weight_bytes"], m["peak_bw_gbs"] * 1e9, m["kv_bytes_per_token"]
    t_w = W / (peak * EFF_W)
    out = {}
    for r in d["results"]:
        B = r.get("batch") or r.get("B")
        S = r.get("ctx") or r.get("S")
        if not B or not S:
            continue
        kv = B * S * k
        t_kv = r["itl_ms"] / 1e3 - t_w
        if t_kv <= 0:
            continue
        out.setdefault(round(kv / 1e9, 1), []).append(kv / (peak * t_kv))
    return sorted((gb, vals) for gb, vals in out.items())


if __name__ == "__main__":
    print(f"eta_kv from {SRC.name}, peak from its own meta, eta_w = {EFF_W}")
    for gb, vals in curve():
        v = "/".join(f"{x:.3f}" for x in sorted(vals))
        print(f"  {gb:6.1f} GB  {v}")
    print("\nkiller_v2.EFF_KV_CURVE should track the >= 4.3 GB points.")
