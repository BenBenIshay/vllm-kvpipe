#!/usr/bin/env python3
"""Rasterise every page of the built PDF so the figures can be LOOKED at.

    rasterise.py main.pdf outdir [dpi]

Writes outdir/page-NN.png. Compiling is not checking: first-draft TikZ reliably
has label collisions that only show up in the rendered page.
"""
import sys

import pymupdf

pdf, out = sys.argv[1], sys.argv[2]
dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 150
doc = pymupdf.open(pdf)
for i, page in enumerate(doc):
    page.get_pixmap(dpi=dpi).save("%s/page-%02d.png" % (out, i + 1))
print("rasterised %d pages at %d dpi" % (len(doc), dpi))
