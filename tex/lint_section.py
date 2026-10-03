#!/usr/bin/env python3
"""Structural lint for paper sections, for when there is no LaTeX locally.

This cluster has no pdflatex -- the paper is compiled on Overleaf -- so a
section can stay syntactically broken for as long as it takes someone to paste
it in and find out. build.sh covers encoding and `% src:` pointers by reading a
build log it cannot produce here; this covers what is checkable WITHOUT a build:
brace balance, environment pairing, undefined refs, labels colliding with the
rest of the paper, math-mode balance, draft markers and non-ASCII.

    ./.venv/bin/python paper/tex/lint_section.py [section.tex ...]

Defaults to every file under paper/tex/sections/. Exits non-zero on any failure.
"""
import glob
import os
import re
import subprocess
import sys

SECTIONS = "paper/tex/sections"


def _escaped(text, i):
    """True if text[i] is escaped, i.e. preceded by an ODD run of backslashes.

    The parity matters and getting it wrong is not academic: `\\{` is a LaTeX
    line break followed by a real opening brace, which a naive "is the previous
    character a backslash" test reads as an escaped brace. That misread made
    this linter report a brace imbalance in an appendix that is in fact
    well-formed.
    """
    n = 0
    j = i - 1
    while j >= 0 and text[j] == "\\":
        n += 1
        j -= 1
    return n % 2 == 1


def strip_comments(text):
    """Drop everything after an unescaped % on each line."""
    out = []
    for ln in text.split("\n"):
        cut = len(ln)
        for i, ch in enumerate(ln):
            if ch == "%" and not _escaped(ln, i):
                cut = i
                break
        out.append(ln[:cut])
    return "\n".join(out)


def labels_in(text):
    return set(re.findall(r"\\label\{([^}]+)\}", text))


def lint(path, other_labels):
    src = open(path).read()
    lines = src.split("\n")
    body = strip_comments(src)
    fails = []

    def err(m):
        fails.append(m)

    depth = 0
    for i, ch in enumerate(body):
        esc = _escaped(body, i)
        if ch == "{" and not esc:
            depth += 1
        elif ch == "}" and not esc:
            depth -= 1
            if depth < 0:
                err("brace depth went negative")
                break
    if depth > 0:
        err("unbalanced braces: %d left open" % depth)

    stack = []
    for n, ln in enumerate(body.split("\n"), 1):
        for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", ln):
            kind, env = m.group(1), m.group(2)
            if kind == "begin":
                stack.append((env, n))
            elif not stack:
                err("line %d: \\end{%s} with nothing open" % (n, env))
            elif stack[-1][0] != env:
                err("line %d: \\end{%s} closes \\begin{%s} opened line %d"
                    % (n, env, stack[-1][0], stack[-1][1]))
                stack.pop()
            else:
                stack.pop()
    if stack:
        err("unclosed environments: "
            + ", ".join("%s@%d" % (e, n) for e, n in stack))

    mine = labels_in(src)
    refs = set(re.findall(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", src))
    missing = sorted(refs - mine - other_labels)
    if missing:
        err("undefined refs: " + ", ".join(missing))
    dupe = sorted(mine & other_labels)
    if dupe:
        err("labels also defined elsewhere: " + ", ".join(dupe))

    dollars = sum(1 for i, ch in enumerate(body)
                  if ch == "$" and not _escaped(body, i))
    if dollars % 2:
        err("odd number of $ (%d): unbalanced math mode" % dollars)

    for marker in (r"\todo", r"\needsgh", "FIXME", "XXX"):
        if marker in src:
            err("draft marker present: %s" % marker)

    nonascii = [n for n, l in enumerate(lines, 1)
                if any(ord(c) > 127 for c in l)]
    if nonascii:
        err("non-ASCII on lines "
            + ", ".join(str(n) for n in nonascii[:6])
            + (" ..." if len(nonascii) > 6 else ""))

    # Every % src: pointer must name a file that exists, same rule as build.sh.
    # CAVEAT worth knowing before acting on a failure here: a worktree
    # materialises only TRACKED files, so a pointer to real evidence that was
    # never committed reads as missing in a worktree and present in the main
    # checkout. That is still a finding -- the convention exists so a number is
    # defensible in a rebuttal, and an uncommitted file is not available to a
    # co-author or a fresh clone -- but the fix is to commit it, not to
    # re-measure. The message says which case it is.
    for ptr in re.findall(r"% src: ([^\s@,]+)", src):
        if not ptr.endswith((".json", ".log", ".py")):
            continue
        if "*" in ptr or ptr == "<file>.json":
            continue
        if os.path.exists(ptr) or os.path.exists(os.path.join("paper", ptr)):
            continue
        rc = subprocess.run(["git", "ls-files", "--error-unmatch", "--", ptr],
                            capture_output=True).returncode
        state = "tracked but absent" if rc == 0 else "NOT IN VERSION CONTROL"
        err("%% src: %s -- missing here, %s" % (ptr, state))

    return fails, len(mine), len(refs)


def main():
    targets = sys.argv[1:] or sorted(glob.glob(os.path.join(SECTIONS, "*.tex")))
    if not targets:
        print("no sections found under %s" % SECTIONS)
        return 1
    allfiles = sorted(glob.glob(os.path.join(SECTIONS, "*.tex")))
    if os.path.exists("paper/tex/main.tex"):
        allfiles.append("paper/tex/main.tex")

    total = 0
    for t in targets:
        other = set()
        for f in allfiles:
            if os.path.abspath(f) != os.path.abspath(t):
                other |= labels_in(open(f).read())
        fails, nlab, nref = lint(t, other)
        total += len(fails)
        name = os.path.basename(t)
        if fails:
            print("FAIL %-26s %d problem(s)" % (name, len(fails)))
            for m in fails:
                print("       %s" % m)
        else:
            print("ok   %-26s %2d labels, %2d refs" % (name, nlab, nref))

    print("\nTEX LINT %s" % ("PASS" if total == 0 else "FAIL (%d)" % total))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
