#!/usr/bin/env bash
# Build the v2 paper with tectonic. Output: paper/v2/out/main.pdf.
#   ./build.sh            compile; exit 1 on errors or undefined refs/citations
#   ./build.sh --figs     also rasterise every figure page to out/fig-*.png, to LOOK at
#                       (needs pymupdf; set PYTHON to an interpreter that has it)
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$D/out"; rm -rf "$OUT"; mkdir -p "$OUT"
cp -r "$D/main.tex" "$D/refs.bib" "$D/sections" "$D/figs" "$OUT/"
cd "$OUT"
if ! tectonic -X compile --keep-logs main.tex > compile.out 2>&1; then
  echo "BUILD FAILED"; grep -iE "error" compile.out | head -20; exit 1
fi
status=0
for pat in "Reference \`[^']*' on page .* undefined" "Citation \`[^']*' on page .* undefined" \
           "multiply defined" "Overfull \\\\hbox \([0-9]{2,}"; do
  if grep -E "$pat" main.log >/dev/null 2>&1; then
    echo "== $pat"; grep -E "$pat" main.log | head -10; status=1; fi
done
if [[ "${1:-}" == "--figs" ]]; then
  "${PYTHON:-python3}" -I "$D/figs/rasterise.py" "$OUT/main.pdf" "$OUT"
fi
exit $status
