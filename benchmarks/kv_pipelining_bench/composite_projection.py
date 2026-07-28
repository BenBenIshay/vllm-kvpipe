#!/usr/bin/env python3
"""M3 payoff: project hybrid composite goodput from the MEASURED R0 & R1 ITL
curves + the PCIe/NVLink hide roofline.

Composite (target) throughput for a mixed batch:
    tok/s ~= B_hbm / ITL_R0(B_hbm)  +  B_off / ITL_R1(B_hbm+B_off)
where B_off = roofline max_offload_batch(seq, ITL, X_eff, b_tok).

Shows A100/PCIe4 (X_eff~25 GB/s, small B_off) vs a fast-fabric node
(GH200/GB200, X_eff~480 GB/s, large B_off) — i.e. where the feature's value
lives. Reads the fresh hide-boundary summary_all.json (vanilla + pipelined).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, "/mnt/beegfs/benishay/vllm-kvpipe")
from vllm.v1.hybrid_kv_roofline import max_offload_batch, bytes_per_token

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/mnt/beegfs/benishay/paper_runs/20260728/hybrid_m1/hide_boundary_v2")
SEQ = 640            # 512 in + 128 out
# Llama-3.1-8B: 32 layers, 8 KV heads, head_dim 128, bf16
B_TOK = bytes_per_token(num_layers=32, num_kv_heads=8, head_dim=128, dtype_bytes=2)
FABRICS = {"A100 / PCIe4 (25 GB/s)": 25.1, "GH200 / NVLink-C2C (480 GB/s)": 480.0}
C_R0 = "#0072B2"; C_HY = "#009E73"; C_R1 = "#E69F00"


def load(mode):
    d = json.load(open(RD / "summary_all.json"))
    rows = {}
    for t in d.get(mode, {}).get("throughput", []):
        itl = t.get("concurrent_mean_itl_steady_s_mean")
        c = t["concurrency"]
        if not itl or itl <= 0:
            continue
        if t.get("failures", 0) > 0 or t.get("successes", 0) < c:
            continue
        rows[c] = itl * 1e3  # ms
    return rows


def interp(curve, x):
    ks = sorted(curve)
    if not ks:
        return None
    if x <= ks[0]:
        return curve[ks[0]]
    if x >= ks[-1]:
        return curve[ks[-1]]
    lo = max(k for k in ks if k <= x); hi = min(k for k in ks if k >= x)
    if lo == hi:
        return curve[lo]
    f = (x - lo) / (hi - lo)
    return curve[lo] * (1 - f) + curve[hi] * f


def main():
    r0 = load("vanilla"); r1 = load("pipelined")
    if not r0:
        print("no vanilla data yet"); return
    b_hbm_grid = [1, 4, 8, 16, 32, 48, 64]
    lines = ["# Composite hybrid goodput projection (measured curves + roofline)",
             "",
             f"S={SEQ} tok, b_tok={B_TOK} B/token.  R0/R1 ITL from measured sweep.",
             ""]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for fab, xeff in FABRICS.items():
        xs, gain = [], []
        lines.append(f"## {fab}")
        lines.append("| B_hbm | ITL_R0 ms | B_off (roofline) | R0-only tok/s | hybrid tok/s | gain |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for bh in b_hbm_grid:
            itl0 = interp(r0, bh)
            if itl0 is None:
                continue
            boff = max_offload_batch(SEQ, itl0, xeff, B_TOK)
            itl1 = interp(r1, bh + boff) if r1 else itl0
            r0_only = bh / (itl0 / 1e3)
            hybrid = bh / (itl0 / 1e3) + (boff / (itl1 / 1e3) if (boff and itl1) else 0.0)
            g = hybrid / r0_only if r0_only else 1.0
            xs.append(bh); gain.append(g)
            lines.append(f"| {bh} | {itl0:.2f} | {boff} | {r0_only:.0f} | {hybrid:.0f} | {g:.2f}x |")
        lines.append("")
        ax.plot(xs, gain, "o-", lw=2, label=fab)
    ax.axhline(1.0, color="#888", ls="--", alpha=0.6, label="R0-only (no offload)")
    ax.set_xlabel("B_hbm (HBM-resident batch)")
    ax.set_ylabel("hybrid goodput / R0-only goodput")
    ax.set_title("Where the hybrid win lives: composite goodput vs interconnect\n"
                 "Llama-3.1-8B, measured R0/R1 ITL + PCIe/NVLink roofline")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(RD / "fig_composite_goodput.png", dpi=150)
    fig.savefig(RD / "fig_composite_goodput.svg")
    (RD / "COMPOSITE_PROJECTION.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote COMPOSITE_PROJECTION.md + fig_composite_goodput.(png|svg) in {RD}")


if __name__ == "__main__":
    main()
