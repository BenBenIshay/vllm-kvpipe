#!/usr/bin/env bash
# Compile the paper locally with tectonic and report the errors that matter.
#
#   ./build.sh          compile, scan the log, leave the PDF in paper/out/
#   ./build.sh --open   also copy it to paper/out/main.pdf and print the path
#
# Tectonic is self-contained and resolves bibtex itself, so this needs no TeX
# Live install -- only network access on first run, to fetch packages.
set -euo pipefail

PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${PAPER_BUILD:-$PAPER_DIR/out}"

command -v tectonic >/dev/null || {
  echo "error: tectonic not on PATH" >&2; exit 1; }

# Stage the Overleaf-flat layout that main.tex expects.
rm -rf "$BUILD"; mkdir -p "$BUILD/figs"
cp "$PAPER_DIR/tex/main.tex" "$PAPER_DIR/tex/refs.bib" "$BUILD/"
cp -r "$PAPER_DIR/tex/sections" "$BUILD/"
cp "$PAPER_DIR"/figs/*.pdf "$BUILD/figs/" 2>/dev/null || true

cd "$BUILD"
if ! tectonic -X compile --keep-logs main.tex > compile.out 2>&1; then
  echo "BUILD FAILED"; grep -iE "^error|error:" compile.out | head -20; exit 1
fi

status=0
scan() {  # scan <label> <grep pattern>
  local hits
  hits="$(grep -E "$2" main.log 2>/dev/null || true)"
  if [[ -n "$hits" ]]; then
    echo "== $1 =="; echo "$hits" | sed 's/^/  /' | head -20; status=1
  fi
}

# Things that put a "??" in the PDF or silently drop a citation.
scan "UNDEFINED REFERENCES"  "Reference \`[^']*' on page .* undefined"
scan "UNDEFINED CITATIONS"   "Citation \`[^']*' on page .* undefined"
scan "MULTIPLY-DEFINED"      "Label \`[^']*' multiply defined"
scan "MISSING FILES"         "File \`[^']*' not found"

# ---- source-level gates (do not need the log) --------------------------------
src_status=0
note() { echo "== $1 =="; shift; printf '  %s\n' "$@"; src_status=1; }

# Raw non-ASCII breaks a pdfLaTeX build: main.tex loads no inputenc, and the
# author kit may not either. tectonic (XeTeX) hides this, so gate on it here.
mapfile -t nonascii < <(grep -rnP --include='*.tex' '[^\x00-\x7F]' "$PAPER_DIR/tex" 2>/dev/null | head -10)
[[ ${#nonascii[@]} -gt 0 ]] && note "NON-ASCII IN SOURCE" "${nonascii[@]}"

# Every numeric claim carries "% src: <file> @ <sha>". A src naming a file that
# does not exist is worse than no src at all: it looks defensible and is not.
missing=()
while IFS= read -r line; do
  f="${line#*% src: }"; f="${f%% @ *}"; f="${f%%,*}"; f="$(echo "$f" | xargs)"
  [[ "$f" == *.json || "$f" == *.log || "$f" == *.py ]] || continue
  [[ "$f" == *'*'* || "$f" == '<file>.json' ]] && continue   # globs + the convention example
  [[ -e "$PAPER_DIR/../$f" || -e "$PAPER_DIR/$f" || -e "$f" ]] || missing+=("${line%%:*}: $f")
done < <(grep -rn --include='*.tex' '% src:' "$PAPER_DIR/tex" 2>/dev/null)
[[ ${#missing[@]} -gt 0 ]] && note "DEAD % src: POINTER" "${missing[@]}"

# \todo / \needsgh error out under \draftfalse by design; catch them earlier.
mapfile -t marks < <(grep -rn --include='*.tex' '\\todo{\|\\needsgh{' "$PAPER_DIR/tex" 2>/dev/null | head -10)
[[ ${#marks[@]} -gt 0 ]] && note "UNRESOLVED DRAFT MARKER" "${marks[@]}"

[[ $src_status -eq 1 ]] && status=1

overfull="$(grep -c "Overfull \\\\hbox" compile.out 2>/dev/null || echo 0)"
# Tectonic writes an .xdv first, so the page count lands in main.log, not stdout.
pages="$(grep -oE "Output written on [^(]*\(([0-9]+) pages" main.log 2>/dev/null \
         | grep -oE "[0-9]+ pages" | tail -1)"; pages="${pages:-? pages}"

echo
if [[ $status -eq 0 ]]; then
  echo "OK -- refs, citations, encoding and % src: pointers all clean."
else
  echo "^ fix the above; they render as ?? or vanish in the PDF."
fi
echo "$pages, $overfull overfull hbox warning(s) (cosmetic)"
echo "PDF: $BUILD/main.pdf"
exit $status
