#!/usr/bin/env bash
# Where each section starts, and how many pages it eats, against a 10-page
# MLSys body budget. Run after ./build.sh.
#
# The budget row is the allocation typical of systems/ML papers at this length:
# the evaluation is the largest single section, and everything before it exists
# to make the evaluation legible. If the evaluation does not start around p7,
# the front of the paper is too long -- that is the number to watch.
set -euo pipefail
PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$(mktemp -d)"; trap 'rm -rf "$BUILD"' EXIT
mkdir -p "$BUILD/figs"
cp "$PAPER_DIR/tex/main.tex" "$PAPER_DIR/tex/refs.bib" "$BUILD/"
cp -r "$PAPER_DIR/tex/sections" "$BUILD/"
cp "$PAPER_DIR"/figs/*.pdf "$BUILD/figs/" 2>/dev/null || true
# Mark where the bibliography starts, so section 8's span is section 8 and not
# section 8 plus the references. Done on the build copy only.
sed -i 's|^\\bibliography{refs}|\\phantomsection\\label{app:refs}\n\\bibliography{refs}|' "$BUILD/main.tex"
cd "$BUILD"
tectonic -X compile --keep-intermediates main.tex >/dev/null 2>&1 || {
  echo "compile failed -- run ./build.sh first" >&2; exit 1; }
python3 - "$BUILD/main.aux" <<'PY'
import re, sys, pathlib
aux = pathlib.Path(sys.argv[1]).read_text()
pg = {m.group(1): int(m.group(2)) for m in
      re.finditer(r"\\newlabel\{(sec:[a-z]+|app:[a-z]+)\}\{\{[^{}]*\}\{(\d+)\}", aux)}
BODY = [("1 Intro","sec:intro",1.25), ("2 Background","sec:background",1.00),
        ("3 Model","sec:model",1.75), ("4 Design","sec:design",1.50),
        ("5 Impl","sec:impl",0.50), ("6 Evaluation","sec:eval",3.25),
        ("7 Related","sec:related",0.75), ("8 Discussion","sec:discussion",0.75)]
refs = pg.get("app:refs", 0)
app = min([v for k, v in pg.items() if k.startswith("app:") and k != "app:refs"] or [0])
print(f"{'section':<15}{'starts':>7}{'pages':>7}{'budget':>8}")
tot = 0.0
for i, (nm, lb, want) in enumerate(BODY):
    st = pg.get(lb)
    nxt = pg.get(BODY[i+1][1]) if i + 1 < len(BODY) else refs
    has = (nxt - st) if (st and nxt) else 0
    tot += has
    over = has - want
    print(f"{nm:<15}{st or '?':>7}{has:>7}{want:>8.2f}"
          f"{'   <-- +%.1f' % over if over >= 1 else ''}")
print(f"{'BODY':<15}{'':>7}{tot:>7.0f}{sum(w for _,_,w in BODY):>8.2f}"
      f"   (references p{refs}, appendix p{app})")
ev = pg.get("sec:eval")
print(f"\nevaluation starts p{ev} (budget p7); body {tot:.0f}pp against 10pp")
sys.exit(1 if tot > 10 else 0)
PY
