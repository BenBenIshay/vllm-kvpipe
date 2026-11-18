#!/usr/bin/env bash
# Build ONE figure in the paper's exact preamble and layout, in a private temp
# dir, and rasterise the page so it can be looked at. Safe to run concurrently.
#
#   figs/preview.sh fig-rolling OUT.png      (figure file name without .tex)
#
# The figure is placed in a real two-column page with filler text, so a
# \begin{figure} is seen at column width and a figure* at text width -- the
# sizes it will print at.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIG="$1"; OUTPNG="$2"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
cp -r "$D/figs" "$D/refs.bib" "$T/"
sed -n '1,/^\\begin{document}/p' "$D/main.tex" > "$T/p.tex"
cat >> "$T/p.tex" <<EOF
\\section{Design}\\label{sec:design}\\subsection{x}\\label{sec:admission}\\label{sec:mm}
\\label{sec:rolling}\\label{sec:addressing}\\label{sec:fused}\\label{sec:residency}\\label{sec:ablation}
\\input{figs/$FIG}
Filler text so the figure floats on a real page. \\par
\\end{document}
EOF
cd "$T"
if ! tectonic -X compile --keep-logs p.tex > c.out 2>&1; then
  grep -n -A4 '^!' p.log | head -30; echo "PREVIEW BUILD FAILED"; exit 1
fi
grep -E "Overfull \\\\hbox" p.log | head -5 || true
"${PYTHON:-python3}" -I - "$T/p.pdf" "$OUTPNG" <<'PY'
import sys, pymupdf, numpy as np
d = pymupdf.open(sys.argv[1])
# the page with the most ink that is not filler: take every page, crop to ink
best = None
for p in d:
    pix = p.get_pixmap(dpi=220)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    ink = a.min(axis=2) < 245
    if best is None or ink.sum() > best[0]:
        best = (ink.sum(), p)
p = best[1]
pix = p.get_pixmap(dpi=220)
pix.save(sys.argv[2])
print("wrote", sys.argv[2], pix.width, pix.height)
PY
