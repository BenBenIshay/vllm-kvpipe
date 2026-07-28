#!/usr/bin/env python3
"""Analyze the vanilla(R0) vs pipelined(R1) hide-boundary sweep and emit a
markdown table + two figures.

R0 = full-HBM residency (fair `vanilla` baseline).
R1 = all-CPU rolling K-window pipelining.
"Hidden" means the CPU-tier transfer is fully overlapped: R1 ITL <= R0 ITL + SLO.
Reads the bench's summary_all.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/mnt/beegfs/benishay/paper_runs/20260728/hybrid_m1/hide_boundary")
OUT = RESULT_DIR
SLO_MS = 2.0  # R1-R0 SLO used in prior THEORY_VS_MEASURED

# colorblind-safe
C_R0 = "#0072B2"   # blue
C_R1 = "#E69F00"   # orange
C_OK = "#009E73"   # green


def load():
    d = json.load(open(RESULT_DIR / "summary_all.json"))
    out = {}
    for mode, md in d.items():
        rows = {}
        for t in md.get("throughput", []):
            c = t["concurrency"]
            itl = t.get("concurrent_mean_itl_steady_s_mean")
            p99 = t.get("concurrent_mean_p99_itl_steady_s_mean")
            tok = t.get("throughput_tok_per_s")
            # drop failed/degenerate cells: require all requests to have completed
            # and a real positive ITL (a failed cell writes itl~0 / partial ok).
            if itl is None or not itl or itl <= 0:
                continue
            if t.get("failures", 0) > 0 or t.get("successes", 0) < c:
                continue
            rows[c] = dict(itl_ms=itl * 1e3,
                           p99_ms=(p99 * 1e3 if p99 else None),
                           tok_s=tok,
                           ok=t.get("successes", 0),
                           fail=t.get("failures", 0))
        out[mode] = rows
    return out


def main():
    data = load()
    r0 = data.get("vanilla", {})
    r1 = data.get("pipelined", {})
    concs = sorted(set(r0) | set(r1))

    # ---- markdown table ----
    lines = ["# Hide-boundary — vanilla(R0) vs pipelined(R1), Llama-3.1-8B / A100",
             "",
             f"SLO: R1-R0 <= {SLO_MS} ms.  ITL = steady-state mean inter-token latency.",
             "",
             "| conc | R0 ITL ms | R1 ITL ms | Δ (R1-R0) ms | hidden? | R0 tok/s | R1 tok/s |",
             "|---:|---:|---:|---:|:--:|---:|---:|"]
    max_hidden = 0
    for c in concs:
        a, b = r0.get(c), r1.get(c)
        a_itl = a["itl_ms"] if a else None
        b_itl = b["itl_ms"] if b else None
        d = (b_itl - a_itl) if (a_itl and b_itl) else None
        hid = (d is not None and d <= SLO_MS)
        if hid:
            max_hidden = max(max_hidden, c)
        lines.append("| {c} | {a} | {b} | {d} | {h} | {ta} | {tb} |".format(
            c=c,
            a=f"{a_itl:.2f}" if a_itl else "—",
            b=f"{b_itl:.2f}" if b_itl else "—",
            d=f"{d:+.2f}" if d is not None else "—",
            h="yes" if hid else ("no" if d is not None else "—"),
            ta=f"{a['tok_s']:.0f}" if a else "—",
            tb=f"{b['tok_s']:.0f}" if b else "—"))
    lines += ["", f"**Max hidden concurrency:** {max_hidden}", ""]
    (OUT / "HIDE_BOUNDARY.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # ---- figure 1: ITL vs concurrency ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs0 = [c for c in concs if c in r0]
    ys0 = [r0[c]["itl_ms"] for c in xs0]
    xs1 = [c for c in concs if c in r1]
    ys1 = [r1[c]["itl_ms"] for c in xs1]
    ax.plot(xs0, ys0, "o-", color=C_R0, label="R0 full-HBM (vanilla)", lw=2)
    ax.plot(xs1, ys1, "s-", color=C_R1, label="R1 CPU-tier rolling window", lw=2)
    ax.set_xlabel("decode concurrency (batch size)")
    ax.set_ylabel("steady-state mean ITL (ms)")
    ax.set_title("CPU-tier transfer is hidden: R1 ITL tracks R0 ITL\nLlama-3.1-8B, A100 80GB PCIe")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if max_hidden:
        ax.axvline(max_hidden, color=C_OK, ls="--", alpha=0.6)
        ax.text(max_hidden, ax.get_ylim()[1]*0.95, f" hidden ≤ {max_hidden}",
                color=C_OK, va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_itl_vs_conc.png", dpi=150)
    fig.savefig(OUT / "fig_itl_vs_conc.svg")

    # ---- figure 2: throughput vs concurrency ----
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    t0 = [r0[c]["tok_s"] for c in xs0]
    t1 = [r1[c]["tok_s"] for c in xs1]
    ax2.plot(xs0, t0, "o-", color=C_R0, label="R0 full-HBM (vanilla)", lw=2)
    ax2.plot(xs1, t1, "s-", color=C_R1, label="R1 CPU-tier rolling window", lw=2)
    ax2.set_xlabel("decode concurrency (batch size)")
    ax2.set_ylabel("decode throughput (tok/s)")
    ax2.set_title("Decode throughput vs concurrency\nLlama-3.1-8B, A100 80GB PCIe")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUT / "fig_tput_vs_conc.png", dpi=150)
    fig2.savefig(OUT / "fig_tput_vs_conc.svg")
    print(f"\nWrote: HIDE_BOUNDARY.md, fig_itl_vs_conc.(png|svg), fig_tput_vs_conc.(png|svg) in {OUT}")


if __name__ == "__main__":
    main()
