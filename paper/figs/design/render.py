#!/usr/bin/env python3
"""Render each TikZ figure to a cropped PNG, for embedding in DESIGN_AND_COST_FLOOR.md.

    .venv/bin/python paper/figs/design/render.py

WHY THIS EXISTS. The design document is markdown, read outside LaTeX, and a design
document without diagrams is not worth reading. The figures are authored once, as
TikZ in paper/tex/sections/fig-*.tex, so the paper and the document can never drift
apart; this script is the only bridge between them. Edit the .tex, re-run this, never
hand-edit a PNG.

HOW. One figure per page into a throwaway PDF, then each page is re-rendered clipped
to its own ink bounding box -- found by scanning for non-white pixels rather than by
asking PDF for a bbox, because a TikZ picture's declared bounds routinely exclude
labels that overhang it.

Needs: tectonic on PATH, and pymupdf + numpy (both already in .venv).
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, "..", ".."))
SECTIONS = os.path.join(PAPER, "tex", "sections")

# (figure file, output name). The NUMBER IS THE DOCUMENT'S FIGURE NUMBER, which is
# document order, not the paper's. If a figure moves in the document, renumber here
# and re-run; do not leave the two orderings to drift.
FIGS = [
    ("fig-arch",      "01-arch"),
    ("fig-addr",      "02-addr"),
    ("fig-admit",     "03-admit"),
    ("fig-step",      "04-step"),
    ("fig-exec",      "05-exec"),
    ("fig-pipeline",  "06-pipeline"),
    ("fig-basevsopt", "07-basevsopt"),
    ("fig-ladder",    "08-ladder"),
    ("fig-fq",        "09-fq"),
]

# A figure whose natural width exceeds one column gets a wider page, so \resizebox
# is not forced to shrink it into illegibility. This affects only the rendered PNG;
# in the paper the figure's own float class decides.
WIDE = {"fig-arch", "fig-basevsopt", "fig-step", "fig-admit", "fig-pipeline"}

PREAMBLE = r"""\documentclass[letterpaper]{article}
\usepackage[paperwidth=%s, paperheight=11in, margin=0.4in]{geometry}
\usepackage{booktabs}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,fit,calc,decorations.pathreplacing}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
\pagestyle{empty}
%% hyperref is deliberately NOT loaded (it is not needed to draw a picture and it
%% drags in link colouring), so \phantomsection has to be stubbed.
\providecommand{\phantomsection}{}
%% The captions are part of the figure for our purposes: the document shows the
%% picture and its caption together, exactly as the paper will.
\begin{document}
%% stubs so cross-references in captions resolve standalone
\phantomsection\label{sec:eval:cost}\phantomsection\label{sec:design:exec}
\phantomsection\label{app:ridercost}\phantomsection\label{fig:compact}
\phantomsection\label{tab:related}\phantomsection\label{tab:ledger}
\phantomsection\label{eq:knee}
"""


def build_one(fig, wide, workdir):
    """Compile a single figure to a one-page PDF. Returns the pdf path."""
    width = "11in" if wide else "7.0in"
    tex = PREAMBLE % width + "\\input{sections/%s}\n\\clearpage\n\\end{document}\n" % fig
    name = "only_%s" % fig.replace("-", "_")
    with open(os.path.join(workdir, name + ".tex"), "w") as fh:
        fh.write(tex)
    r = subprocess.run(["tectonic", "-X", "compile", name + ".tex"],
                       cwd=workdir, capture_output=True, text=True, timeout=420)
    if r.returncode != 0:
        sys.stderr.write("FAILED %s\n%s\n" % (fig, r.stderr[-1500:]))
        return None
    return os.path.join(workdir, name + ".pdf")


def crop_to_png(pdf, out_png, dpi=190, pad_pt=6):
    """Re-render page 0 clipped to its ink. Returns (w,h) in pixels."""
    import numpy as np
    import pymupdf

    doc = pymupdf.open(pdf)
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.width, pix.n)
    ink = arr[:, :, :3].min(axis=2) < 248          # anything not near-white
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        sys.stderr.write("no ink found in %s\n" % pdf)
        return None
    scale = dpi / 72.0
    clip = pymupdf.Rect(cols[0] / scale - pad_pt, rows[0] / scale - pad_pt,
                        (cols[-1] + 1) / scale + pad_pt,
                        (rows[-1] + 1) / scale + pad_pt) & page.rect
    out = page.get_pixmap(dpi=dpi, clip=clip)
    out.save(out_png)
    return out.width, out.height


def main():
    workdir = tempfile.mkdtemp(prefix="designfigs_")
    shutil.copytree(SECTIONS, os.path.join(workdir, "sections"))
    ok = True
    for fig, out in FIGS:
        src = os.path.join(SECTIONS, fig + ".tex")
        if not os.path.exists(src):
            sys.stderr.write("missing %s\n" % src)
            ok = False
            continue
        pdf = build_one(fig, fig in WIDE, workdir)
        if pdf is None:
            ok = False
            continue
        png = os.path.join(HERE, out + ".png")
        dim = crop_to_png(pdf, png)
        if dim is None:
            ok = False
            continue
        print("%-14s -> %-18s %4d x %4d px" % (fig, out + ".png", dim[0], dim[1]))
    shutil.rmtree(workdir, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
