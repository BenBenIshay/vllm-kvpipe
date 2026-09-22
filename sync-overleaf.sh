#!/usr/bin/env bash
# Keep an Overleaf project in sync with paper/tex + paper/figs.
#
# Two modes, see paper/OVERLEAF.md:
#
#   OUTBOX MODE (free Overleaf account) -- no zip re-import, no new project:
#     ./sync-overleaf.sh import <overleaf.zip>   seed/refresh from Overleaf's download
#     ./sync-overleaf.sh stage                   collect ONLY changed files -> outbox
#     ./sync-overleaf.sh mark                    record that the outbox was uploaded
#
#   GIT-BRIDGE MODE (Overleaf Standard/Professional):
#     ./sync-overleaf.sh clone | push [msg] | pull | diff
set -euo pipefail

PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLONE_DIR="${OVERLEAF_CLONE:-$PAPER_DIR/.overleaf}"
OUTBOX="$PAPER_DIR/.overleaf-outbox"
MANIFEST="$PAPER_DIR/.overleaf-manifest"
ID_FILE="$PAPER_DIR/.overleaf-project"

die() { echo "error: $*" >&2; exit 1; }

# The set of files that exist on both sides, as "<local path>|<path in Overleaf>".
# Overleaf keeps the project flat at its root; this repo splits tex/ and figs/.
pairs() {
  echo "tex/main.tex|main.tex"
  echo "tex/refs.bib|refs.bib"
  local f
  for f in "$PAPER_DIR"/tex/sections/*.tex; do
    [[ -e "$f" ]] && echo "tex/sections/$(basename "$f")|sections/$(basename "$f")"
  done
  for f in "$PAPER_DIR"/figs/*.pdf "$PAPER_DIR"/figs/*.png; do
    [[ -e "$f" ]] && echo "figs/$(basename "$f")|figs/$(basename "$f")"
  done
}

hash_of() { sha256sum "$1" | cut -d' ' -f1; }

manifest_hash() {  # manifest_hash <path in Overleaf> -> recorded hash, or empty
  [[ -f "$MANIFEST" ]] || return 0
  awk -F'  ' -v k="$1" '$2 == k { print $1 }' "$MANIFEST"
}

write_manifest() {  # write_manifest <file>
  local dest="$1" local_rel remote_rel
  : > "$dest"
  while IFS='|' read -r local_rel remote_rel; do
    printf '%s  %s\n' "$(hash_of "$PAPER_DIR/$local_rel")" "$remote_rel" >> "$dest"
  done < <(pairs)
}

project_url() {
  [[ -f "$ID_FILE" ]] || { echo ""; return; }
  local id
  id="$(tr -d '[:space:]' < "$ID_FILE" | sed 's#.*/project/##; s#/.*##')"
  [[ -n "$id" ]] && echo "https://www.overleaf.com/project/$id"
}

case "${1:-}" in

  # ---------------------------------------------------------------- outbox mode
  stage)
    rm -rf "$OUTBOX"; mkdir -p "$OUTBOX"
    n=0
    while IFS='|' read -r local_rel remote_rel; do
      src="$PAPER_DIR/$local_rel"
      [[ -f "$src" ]] || continue
      if [[ "$(hash_of "$src")" != "$(manifest_hash "$remote_rel")" ]]; then
        mkdir -p "$OUTBOX/$(dirname "$remote_rel")"
        cp "$src" "$OUTBOX/$remote_rel"
        echo "  changed: $remote_rel"
        n=$((n+1))
      fi
    done < <(pairs)

    if [[ $n -eq 0 ]]; then
      rm -rf "$OUTBOX"
      echo "Overleaf is already up to date -- nothing to upload."
      exit 0
    fi

    write_manifest "$MANIFEST.pending"
    url="$(project_url)"
    cat <<EOF

$n file(s) staged in: $OUTBOX

To apply them in Overleaf (project keeps its URL, history and comments):
  1. Open ${url:-your Overleaf project}
  2. Drag each staged file onto the MATCHING folder in the file tree:
       $OUTBOX/*.tex, refs.bib  -> project root
       $OUTBOX/sections/*       -> sections/
       $OUTBOX/figs/*           -> figs/
     Overleaf asks to confirm the overwrite; accept.
  3. Back here, run:  ./sync-overleaf.sh mark

Files NOT in the list above are unchanged -- do not re-upload them.
EOF
    ;;

  mark)
    [[ -f "$MANIFEST.pending" ]] || die "nothing staged -- run: $0 stage"
    mv "$MANIFEST.pending" "$MANIFEST"
    rm -rf "$OUTBOX"
    echo "recorded. Overleaf and paper/tex are now in sync."
    ;;

  zip)
    # Full-project zip, the original hand-off format. Overleaf's "Upload project"
    # creates a NEW project from this; to keep the existing one, use `stage`.
    out="${2:-$PAPER_DIR/kvtiering-mlsys2027.zip}"
    tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
    mkdir -p "$tmp/overleaf/sections" "$tmp/overleaf/figs"
    cp "$PAPER_DIR/tex/main.tex" "$PAPER_DIR/tex/refs.bib" "$tmp/overleaf/"
    [[ -f "$PAPER_DIR/tex/README.md" ]] && cp "$PAPER_DIR/tex/README.md" "$tmp/overleaf/"
    cp "$PAPER_DIR"/tex/sections/*.tex "$tmp/overleaf/sections/"
    cp "$PAPER_DIR"/figs/*.pdf "$tmp/overleaf/figs/" 2>/dev/null || true
    rm -f "$out"
    (cd "$tmp" && zip -qr "$out" overleaf)
    echo "wrote $out ($(cd "$tmp" && find overleaf -type f | wc -l) files)"
    ;;

  baseline)
    # Record what Overleaf currently holds, WITHOUT touching local files, so that
    # `stage` reports a real delta. Use when local is ahead of Overleaf.
    zip="${2:-}"
    [[ -f "$zip" ]] || die "usage: $0 baseline <project.zip>"
    tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
    unzip -q "$zip" -d "$tmp"
    root="$tmp"; [[ -f "$tmp/overleaf/main.tex" ]] && root="$tmp/overleaf"
    [[ -f "$root/main.tex" ]] || die "no main.tex at the top of $zip"
    : > "$MANIFEST"
    while IFS= read -r f; do
      printf '%s  %s\n' "$(hash_of "$f")" "${f#$root/}" >> "$MANIFEST"
    done < <(find "$root" -type f \( -name '*.tex' -o -name '*.bib' -o -name '*.pdf' -o -name '*.png' \))
    echo "baseline recorded from $zip ($(wc -l < "$MANIFEST") files). Next: $0 stage"
    ;;

  import)
    # NOTE: overwrites paper/tex from the zip. Local edits are recoverable via git.
    zip="${2:-}"
    [[ -f "$zip" ]] || die "usage: $0 import <project.zip>  (Overleaf -> Menu -> Download -> Source)"
    tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
    unzip -q "$zip" -d "$tmp"
    # Overleaf zips are flat at the root; ours nests under overleaf/.
    root="$tmp"; [[ -f "$tmp/overleaf/main.tex" ]] && root="$tmp/overleaf"
    [[ -f "$root/main.tex" ]] || die "no main.tex at the top of $zip"

    cp "$root/main.tex" "$root/refs.bib" "$PAPER_DIR/tex/"
    rsync -a --delete "$root/sections/" "$PAPER_DIR/tex/sections/"
    # Figures are generated locally from figs/*.py; take new ones, delete nothing.
    [[ -d "$root/figs" ]] && rsync -a --include='*.pdf' --include='*.png' --exclude='*' \
      "$root/figs/" "$PAPER_DIR/figs/"

    write_manifest "$MANIFEST"
    rm -rf "$OUTBOX" "$MANIFEST.pending"
    echo "imported $zip. Review with: git -C '$PAPER_DIR' diff"
    ;;

  # ------------------------------------------------------------ git-bridge mode
  clone)
    url="$(project_url)"; [[ -n "$url" ]] || die "missing $ID_FILE"
    [[ -e "$CLONE_DIR" ]] && die "$CLONE_DIR already exists"
    git clone "https://git.overleaf.com/$(basename "$url")" "$CLONE_DIR"
    ;;

  push|pull|diff)
    [[ -d "$CLONE_DIR/.git" ]] || die "no git-bridge clone -- free Overleaf accounts use: $0 stage"
    git -C "$CLONE_DIR" pull --rebase
    if [[ "$1" == "pull" ]]; then
      cp "$CLONE_DIR/main.tex" "$CLONE_DIR/refs.bib" "$PAPER_DIR/tex/"
      rsync -a --delete "$CLONE_DIR/sections/" "$PAPER_DIR/tex/sections/"
      write_manifest "$MANIFEST"; echo "pulled."; exit 0
    fi
    cp "$PAPER_DIR/tex/main.tex" "$PAPER_DIR/tex/refs.bib" "$CLONE_DIR/"
    mkdir -p "$CLONE_DIR/sections" "$CLONE_DIR/figs"
    rsync -a --delete "$PAPER_DIR/tex/sections/" "$CLONE_DIR/sections/"
    rsync -a --include='*.pdf' --include='*.png' --exclude='*' "$PAPER_DIR/figs/" "$CLONE_DIR/figs/"
    git -C "$CLONE_DIR" add -A
    if [[ "$1" == "diff" ]]; then
      git -C "$CLONE_DIR" --no-pager diff --cached --stat; git -C "$CLONE_DIR" reset >/dev/null; exit 0
    fi
    git -C "$CLONE_DIR" diff --cached --quiet && { echo "up to date."; exit 0; }
    git -C "$CLONE_DIR" commit -q -m "${2:-Sync from vllm-kvpipe $(git -C "$PAPER_DIR" rev-parse --short HEAD)}"
    git -C "$CLONE_DIR" push
    write_manifest "$MANIFEST"; echo "pushed."
    ;;

  *) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
