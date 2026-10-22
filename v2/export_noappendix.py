#!/usr/bin/env python3
r"""Export the paper without its appendices, as a folder and an Overleaf zip.

    export_noappendix.py OUTDIR [--final] [--prerender]

--final builds without line numbers (\draftfalse); the body has no draft
notes left, so it compiles either way.
--prerender renders every TikZ/pgfplots picture to a cropped PDF once (with
tectonic) and includes those PDFs instead, so the export compiles in a few
seconds; Overleaf's free plan times out on the 40 s the pictures take.
Captions, labels and references stay LaTeX. A \ref inside a picture is
resolved from the full compile's .aux before rendering.

Copies main.tex, refs.bib, the body sections and the figures they use into
OUTDIR/hetero-kv/, drops the \\appendix block, and rewrites the five body
sentences that point into an appendix so no reference is left dangling. The
source tree is not modified. Writes OUTDIR/hetero-kv.zip (files at the zip
root, so Overleaf's "Upload Project" opens it directly).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

SRC = os.path.dirname(os.path.abspath(__file__))
APPENDIX_FIGS = {"fig-residency.tex", "fig-fused.tex", "fig-abl-fusion.tex"}

# (file, old, new): sentences in the body that cite an appendix
EDITS = [
    ("sections/07-ablation.tex",
     r"""runtime switch of the same binary, not a separate build; the switches are
listed in \cref{sec:switches}.""",
     r"""runtime switch of the same binary, not a separate build."""),
    ("sections/04-design.tex",
     r"""and the kernel takes one KV pointer; \cref{sec:fusion-app} describes a fused
kernel that would remove it.""",
     r"""and the kernel takes one KV pointer; a fused kernel would remove it."""),
    ("sections/04-design.tex",
     r"""rider path (\S\ref{sec:fused}).""",
     r"""rider path."""),
    ("sections/08-related.tex",
     r"""behind \sysname{}'s tier interface (\S\ref{sec:fused}).""",
     r"""behind \sysname{}'s tier interface."""),
    ("sections/04-design.tex",
     r"""cheap. What each is worth is measured in \S\ref{sec:ablation}. The scheduler changes
that let a rider run without holding HBM blocks are described in
\cref{sec:residency}.""",
     r"""cheap. What each is worth is measured in \S\ref{sec:ablation}."""),
]


TIKZ = re.compile(r"^\\begin\{tikzpicture\}.*?^\\end\{tikzpicture\}[^\n]*\n", re.S | re.M)


def tectonic(tex_dir, name):
    r = subprocess.run(["tectonic", "-X", "compile", "--keep-intermediates", name],
                       cwd=tex_dir, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("tectonic failed in %s:\n%s" % (tex_dir, r.stderr[-3000:]))


def prerender(dst, main_tex):
    """Replace every tikzpicture in dst/figs with a cropped, pre-rendered PDF."""
    import pymupdf

    work = tempfile.mkdtemp(prefix="prerender-")
    shutil.copytree(dst, os.path.join(work, "full"))
    tectonic(os.path.join(work, "full"), "main.tex")
    labels = dict(re.findall(r"\\newlabel\{([^}@]+)\}\{\{([^}]*)\}",
                             open(os.path.join(work, "full", "main.aux")).read()))

    preamble = main_tex[:main_tex.index("\\begin{document}")]
    pics = {"col": [], "wide": []}
    srcs = {}
    for f in sorted(os.listdir(os.path.join(dst, "figs"))):
        if not (f.startswith("fig-") and f.endswith(".tex")):
            continue
        src = open(os.path.join(dst, "figs", f)).read()
        m = TIKZ.search(src)
        if not m:
            continue
        pic = re.sub(r"\\ref\{([^}]+)\}", lambda k: labels[k.group(1)], m.group(0))
        assert not re.search(r"\\(c?ref|Cref|cite)\{", pic), f
        kind = "wide" if "\\begin{figure*}" in src else "col"
        pics[kind].append((f, pic))
        srcs[f] = (src, m)

    os.makedirs(os.path.join(dst, "figs", "pdf"))
    for kind, items in pics.items():
        d = os.path.join(work, kind)
        shutil.copytree(dst, d)     # figure files read figs/data relative to here
        pre = preamble if kind == "col" else preamble.replace("twocolumn,", "onecolumn,", 1)
        # lrbox, not a macro argument: pgfplots inline tables need their line
        # ends; the white box marks the picture's own bounding box for the crop
        body = "".join("\\begin{lrbox}{\\picbox}%%\n%s\\end{lrbox}%%\n"
                       "\\noindent{\\setlength{\\fboxsep}{0pt}\\colorbox{white}{\\usebox{\\picbox}}}"
                       "\\clearpage\n" % pic for _, pic in items)
        open(os.path.join(d, "pics.tex"), "w").write(
            pre + "\\newsavebox{\\picbox}\n\\pagestyle{empty}\n\\begin{document}\n"
            + body + "\\end{document}\n")
        tectonic(d, "pics.tex")
        doc = pymupdf.open(os.path.join(d, "pics.pdf"))
        assert len(doc) == len(items), (kind, len(doc), len(items))
        for (f, _), page in zip(items, doc):
            log = page.get_bboxlog()
            # the white box is drawn first and is the picture's own bounding
            # box; ink may spill past it, as it does in the live document
            kind0, box = log[0]
            box = pymupdf.Rect(box)
            assert kind0 == "fill-path", (f, kind0)
            ink = pymupdf.Rect(box)
            for _, r in log[1:]:
                ink |= pymupdf.Rect(r)
            out = pymupdf.open()
            p = out.new_page(width=ink.width, height=ink.height)
            p.show_pdf_page(p.rect, doc, page.number, clip=ink)
            name = f[:-4] + ".pdf"
            out.save(os.path.join(dst, "figs", "pdf", name), garbage=3, deflate=True)
            # trim (no clip) gives LaTeX the box, so the layout is unchanged
            trim = (box.x0 - ink.x0, ink.y1 - box.y1, ink.x1 - box.x1, box.y0 - ink.y0)
            src, m = srcs[f]
            inc = "\\includegraphics[trim=%.2fbp %.2fbp %.2fbp %.2fbp]{figs/pdf/%s}\n" % (trim + (name,))
            open(os.path.join(dst, "figs", f), "w").write(src[:m.start()] + inc + src[m.end():])
    shutil.rmtree(os.path.join(dst, "figs", "data"))

    # with every plot pre-rendered, pgfplots is dead weight on each pass, and
    # an inlined bibliography saves BibTeX and a rerun (TikZ stays: \stepn)
    m = open(os.path.join(dst, "main.tex")).read()
    pgf = re.search(r"^\\usepackage\{pgfplots\}[^\n]*\n", m, re.M)
    assert pgf, "pgfplots line"
    m = m[:pgf.start()] + m[pgf.end():]
    bib = "\\bibliographystyle{plain}\n\\bibliography{refs}\n"
    assert m.count(bib) == 1, "bibliography lines"
    bbl = open(os.path.join(work, "full", "main.bbl")).read()
    m = m.replace(bib, bbl)
    open(os.path.join(dst, "main.tex"), "w").write(m)
    os.remove(os.path.join(dst, "refs.bib"))
    shutil.rmtree(work)


def main(outdir, final=False, prerender_figs=False):
    dst = os.path.join(outdir, "hetero-kv")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(os.path.join(dst, "sections"))
    os.makedirs(os.path.join(dst, "figs", "data"))
    shutil.copy(os.path.join(SRC, "refs.bib"), dst)

    main_tex = open(os.path.join(SRC, "main.tex")).read()
    main_tex, n = re.subn(r"\n\\appendix\n(\\input\{sections/[A-C]-[^}]*\}\n)+", "\n", main_tex)
    assert n == 1, "appendix block not found"
    if final:
        assert main_tex.count("\\newif\\ifdraft\\drafttrue") == 1
        main_tex = main_tex.replace("\\newif\\ifdraft\\drafttrue", "\\newif\\ifdraft\\draftfalse")
    open(os.path.join(dst, "main.tex"), "w").write(main_tex)

    # every section file except the appendices (A-, B-, C-): sections also
    # \input each other (04-design pulls in 04b-admission)
    for f in os.listdir(os.path.join(SRC, "sections")):
        if f.endswith(".tex") and not re.match(r"[A-Z]-", f):
            shutil.copy(os.path.join(SRC, "sections", f), os.path.join(dst, "sections"))
    for f in os.listdir(os.path.join(SRC, "figs")):
        if f.endswith(".tex") and f not in APPENDIX_FIGS:
            shutil.copy(os.path.join(SRC, "figs", f), os.path.join(dst, "figs"))
    for f in os.listdir(os.path.join(SRC, "figs", "data")):
        if f.endswith((".csv", ".txt")):
            shutil.copy(os.path.join(SRC, "figs", "data", f),
                        os.path.join(dst, "figs", "data"))

    for rel, old, new in EDITS:
        p = os.path.join(dst, rel)
        s = open(p).read()
        assert s.count(old) == 1, (rel, old[:60])
        open(p, "w").write(s.replace(old, new))

    # nothing left may point into an appendix
    labels = {"sec:switches", "tab:switches", "sec:wip", "fig:residency", "fig:fused",
              "sec:fusion-app", "sec:fused", "sec:residency", "fig:abl-fusion"}
    for root, _, files in os.walk(dst):
        for f in files:
            if f.endswith(".tex"):
                for line in open(os.path.join(root, f)):
                    if line.lstrip().startswith("%"):
                        continue
                    for lab in labels:
                        assert "ref{" + lab not in line and "ref{" + lab + "," not in line, (f, line)

    if prerender_figs:
        prerender(dst, main_tex)

    z = os.path.join(outdir, "hetero-kv.zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dst):
            for f in sorted(files):
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, dst))
    print(dst)
    print(z)


if __name__ == "__main__":
    main(sys.argv[1], final="--final" in sys.argv[2:],
         prerender_figs="--prerender" in sys.argv[2:])
