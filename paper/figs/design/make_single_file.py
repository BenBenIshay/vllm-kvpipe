#!/usr/bin/env python3
"""Build a SINGLE self-contained file of the design document, figures included.

    .venv/bin/python paper/figs/design/make_single_file.py

WHY. The markdown references its figures by relative path, so reading it means
having the figs/ folder alongside and a viewer that resolves it. That is a folder to
carry around, not a document. This emits ONE .html with every PNG inlined as a
base64 data URI: open it in any browser, nothing else needed, nothing to chase.

The markdown stays the editable source -- do not hand-edit the generated html.
"""
import base64
import mimetypes
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(PAPER, "DESIGN_AND_COST_FLOOR.md")
OUT = os.path.join(PAPER, "DESIGN_AND_COST_FLOOR.html")

CSS = """
:root { color-scheme: light; }
body { max-width: 52rem; margin: 0 auto; padding: 2.5rem 1.25rem 6rem;
       font: 16px/1.65 "Charter","Georgia","Times New Roman",serif;
       color: #1a1a1a; background: #fff; }
h1 { font-size: 1.85rem; line-height: 1.25; margin: 0 0 .4rem;
     border-bottom: 2px solid #222; padding-bottom: .5rem; }
h2 { font-size: 1.3rem; margin: 2.6rem 0 .7rem; padding-top: .5rem;
     border-top: 1px solid #ddd; }
h3 { font-size: 1.08rem; margin: 1.8rem 0 .5rem; color: #333; }
p, li { }
code, kbd { font: 13.5px/1.45 "SF Mono","Menlo","Consolas",monospace;
            background: #f3f3f1; padding: .1em .32em; border-radius: 3px; }
pre code { display: block; padding: .8rem; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; margin: 1.1rem 0;
        font-size: 14.2px; }
th, td { border: 1px solid #d8d8d4; padding: .42rem .6rem;
         text-align: left; vertical-align: top; }
th { background: #f3f3f1; font-weight: 600; }
tr:nth-child(even) td { background: #fbfbfa; }
blockquote { margin: 1.2rem 0; padding: .7rem 1.1rem; background: #f7f7f5;
             border-left: 4px solid #999; }
blockquote p { margin: .35rem 0; }
img { display: block; max-width: 100%; height: auto; margin: 1.4rem auto .4rem;
      border: 1px solid #e4e4e0; border-radius: 3px; background: #fff; }
/* the italic line directly under each figure is its caption */
img + em, p > em:only-child { display: block; text-align: center;
      font-size: 13.5px; color: #555; margin: 0 auto 1.8rem; max-width: 46rem; }
hr { border: 0; border-top: 1px solid #ddd; margin: 2.2rem 0; }
a { color: #1a4f8a; }
strong { font-weight: 650; }
.meta { color: #666; font-size: 13.5px; margin-top: .2rem; }
@media print { body { max-width: none; padding: 0; } h2 { page-break-after: avoid; }
               img { page-break-inside: avoid; } }
"""


def inline_images(html, base):
    """Replace <img src="relative/path"> with a base64 data URI."""
    n = [0]

    def sub(m):
        src = m.group(1)
        if src.startswith("data:"):
            return m.group(0)
        path = os.path.join(base, src)
        if not os.path.exists(path):
            sys.stderr.write("MISSING IMAGE: %s\n" % path)
            return m.group(0)
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        n[0] += 1
        return m.group(0).replace(src, "data:%s;base64,%s" % (mime, b64))

    html = re.sub(r'<img[^>]*\ssrc="([^"]+)"', sub, html)
    return html, n[0]


def main():
    try:
        import markdown
    except ImportError:
        sys.stderr.write("needs python-markdown:  .venv/bin/pip install markdown\n")
        return 1

    text = open(SRC, encoding="utf-8").read()
    title = text.lstrip().split("\n", 1)[0].lstrip("# ").strip()

    body = markdown.markdown(
        text, extensions=["tables", "attr_list", "sane_lists", "toc"],
        output_format="html5")
    body, n = inline_images(body, PAPER)

    html = ("<!doctype html>\n<html lang=\"en\"><head>\n"
            "<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
            "<title>%s</title>\n<style>%s</style>\n</head>\n<body>\n%s\n"
            "</body></html>\n" % (title, CSS, body))

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)

    print("%s\n  %d figures inlined, %.1f MB, nothing external"
          % (OUT, n, len(html.encode("utf-8")) / 1e6))
    return 0 if n == 7 else 1


if __name__ == "__main__":
    sys.exit(main())
