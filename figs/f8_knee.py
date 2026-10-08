#!/usr/bin/env python3
"""F8 -- the transfer-exposure ratio is the governing variable, and the knee is at 1.

    ./.venv/bin/python paper/figs/f8_knee.py --out paper/figs/f8.pdf

THIS REPLACES f6, which plotted the across-the-knee sweep this paper retracts under
a caption claiming the opposite.

AND IT REPLACES THE FIRST VERSION OF ITSELF, which drew one panel per context with
the predicted r* marked. That version was correct but weaker: it showed the knee
moving when context changed, as two anecdotes. Plotting instead against

    l/T0  =  (per-request link time) / (rider-free step time)

collapses both contexts onto ONE curve -- matched at equal l/T0 the two agree to
2-10% over the whole range -- with the departure at l/T0 = 1. That is the stronger
statement: the governing variable is neither the rider count nor the context but
their ratio to the compute window, and the boundary sits at unity by construction
rather than by fitting. r* = floor(T0 / l_per_request) is then a corollary.

Both series are purity-gated, 3 repeats, resident batch 8, on A100/PCIe.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

INK, ACCENT, BLUE, GREY = "#1a1a1a", "#b3472f", "#2f5d8b", "#888888"
D = "paper/data/overhead/21_fusion_bound"


def load(tag):
    d = json.load(open(os.path.join(D, tag + ".json")))
    t0 = next(r["step_ms"] for r in d["rows"]
              if r.get("riders") == 0 and r.get("step_ms"))
    pts = sorted((((r.get("link_ms") or 0.0) / t0), r["dT_ms"] / t0, r["riders"])
                 for r in d["rows"]
                 if r.get("usable") and r.get("riders")
                 and r.get("dT_ms") is not None)
    return t0, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figs/f8.pdf")
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for tag, lab, col, mk in (("E3_ctx1080", "context 1080", BLUE, "o"),
                              ("E3_ctx1440", "context 1440", ACCENT, "s")):
        t0, pts = load(tag)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, mk + "-", ms=4, lw=1.3, color=col, label=lab)
        print("%s t0=%.1f" % (tag, t0))
        for x, y, r in pts:
            print("   r=%-2d l/T0=%.2f dT/T0=%.3f" % (r, x, y))

    ax.axvline(1.0, ls=":", lw=1.2, color=GREY)
    ax.annotate("the transfer no longer\nfits inside the step",
                xy=(1.0, 0.72), xytext=(1.06, 0.62), fontsize=5.9, color=GREY)
    ax.set_xlabel(r"$\ell/\mathrm{ITL}_0$ : transfer demanded per step")
    ax.set_ylabel(r"$\Delta T/\mathrm{ITL}_0$ : step growth")
    ax.legend(fontsize=6.4, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.annotate("two contexts, one curve:\nmatched at equal $\\ell/\\mathrm{ITL}_0$ "
                "they agree to 2--10\\%",
                xy=(0.04, 0.80), xycoords="axes fraction", fontsize=5.7,
                color=INK)
    fig.tight_layout(pad=0.4)
    fig.savefig(a.out)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
