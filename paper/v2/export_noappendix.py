#!/usr/bin/env python3
r"""Export the paper without its appendices, as a folder and an Overleaf zip.

    export_noappendix.py OUTDIR [--final]

--final builds without line numbers (\draftfalse); the body has no draft
notes left, so it compiles either way.

Copies main.tex, refs.bib, the body sections and the figures they use into
OUTDIR/hetero-kv/, drops the \\appendix block, and rewrites the five body
sentences that point into an appendix so no reference is left dangling. The
source tree is not modified. Writes OUTDIR/hetero-kv.zip (files at the zip
root, so Overleaf's "Upload Project" opens it directly).
"""
import os
import re
import shutil
import sys
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


def main(outdir, final=False):
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

    z = os.path.join(outdir, "hetero-kv.zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dst):
            for f in sorted(files):
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, dst))
    print(dst)
    print(z)


if __name__ == "__main__":
    main(sys.argv[1], final="--final" in sys.argv[2:])
