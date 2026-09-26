#!/usr/bin/env python3
"""The rider cost curve and the kink model it fits.

    python3 f6_cost_curve.py --data ../data/rider_cost_phase0.json --out f6.pdf

The paper's central empirical claim about cost is that it is NOT affine: dT per
rider rises, so a fixed-plus-slope fit is meaningless, and what the data fits is
step ~= max(T_compute, r*S*b_tok/X) + overhead. A table cannot show a kink; this
can.

Design constraints (this is a print figure, single column):
  * Identity is carried by LINE STYLE and MARKER, never by hue -- the figure has
    to survive a greyscale printer and a colourblind reader with no legend lookup.
  * Two series, so a legend is present; points are not individually labelled.
  * Grid and spines are recessive; ink is near-black, marks are black/grey.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1a1a1a"        # text
MEAS = "#1a1a1a"       # measured series
MODEL = "#8a8a8a"      # model series
GRID = "#d8d8d8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="f6.pdf")
    ap.add_argument("--link-gbps", type=float, default=25.1)
    a = ap.parse_args()

    d = json.load(open(a.data))
    rows = d["results"]
    r = [x["riders"] for x in rows]
    step = [x["step_ms"] for x in rows]

    t_compute = rows[0]["step_ms"]
    per_rider_gb = rows[1]["gb_per_step"] / rows[1]["riders"]
    link_ms = per_rider_gb / a.link_gbps * 1e3
    knee = t_compute / link_ms

    fine = [i / 10 for i in range(0, int(max(r) * 10) + 1)]
    model = [max(t_compute, x * link_ms) for x in fine]

    plt.rcParams.update({
        "font.size": 8, "font.family": "serif",
        "axes.edgecolor": GRID, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "out", "ytick.direction": "out",
    })
    fig, ax = plt.subplots(figsize=(3.3, 2.5))

    # the unexplained residual, as area rather than as a number per point
    ax.fill_between(fine, [max(t_compute, x * link_ms) for x in fine],
                    [_interp(fine_x, r, step) for fine_x in fine],
                    color="#000000", alpha=0.07, linewidth=0, zorder=1)

    ax.plot(fine, model, linestyle="--", color=MODEL, linewidth=1.3,
            zorder=2, label=r"$\max(T_{\rm compute},\ r\,S\,b_{\rm tok}/X)$")
    ax.plot(r, step, linestyle="-", color=MEAS, linewidth=1.4,
            marker="o", markersize=4.5, markerfacecolor=MEAS,
            markeredgecolor="white", markeredgewidth=0.7,
            zorder=3, label="measured step time")

    ax.axvline(knee, color=MODEL, linestyle=":", linewidth=1.0, zorder=1)
    ax.text(knee - 0.12, 155, f"knee $r={knee:.1f}$", fontsize=7, color=INK,
            ha="right", va="top")

    # one label for the mechanism, not a number on every point
    ax.annotate("overhead the model\ndoes not carry",
                xy=(5.0, (146.89 + 133.3) / 2), xytext=(2.75, 108),
                fontsize=7, color=INK,
                arrowprops=dict(arrowstyle="-", color=MODEL, linewidth=0.7))

    ax.set_xlabel("host-tier requests admitted ($r$)")
    ax.set_ylabel("decode step time (ms)")
    ax.set_xticks(r)
    ax.set_ylim(0, 165)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="lower right",
              handlelength=2.4, borderpad=0, bbox_to_anchor=(1.0, 0.03))

    fig.tight_layout(pad=0.3)
    fig.savefig(a.out)
    print(f"wrote {a.out}  (T_compute={t_compute:.2f} ms, "
          f"{link_ms:.1f} ms/rider, knee={knee:.2f})")


def _interp(x, xs, ys):
    """Piecewise-linear read of the measured curve, for the shaded residual."""
    if x <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), list(zip(xs, ys))[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return ys[-1]


if __name__ == "__main__":
    main()
