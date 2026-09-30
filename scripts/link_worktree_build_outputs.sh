#!/bin/bash
# Make a fresh git worktree of this repo actually runnable.
#
# THE TRAP. An editable install keeps its compiled extensions in the SOURCE
# tree, and `git worktree add` only materialises TRACKED files -- build outputs
# are gitignored. So a new worktree has zero `.so` files where the main tree has
# five, and `import vllm` fails outright. The failure reads like a code bug, not
# a setup one, and it costs a GPU allocation to discover if you submit first.
#
# THE FIX, as WORKTREE_SETUP.md describes for vllm-kvpipeline-66: symlink the
# build outputs. They are products, identical for both trees, and copying them
# is not sensible -- `_vllm_fa3_C.abi3.so` alone is 1.1 GB.
#
# Why this computes the set instead of carrying a list: vllm-kvpipeline-66 has
# 22 such symlinks, but a worktree made today needs 56, because several
# `third_party/triton_kernels` subdirectories postdate that setup. A hardcoded
# list silently leaves the newer ones unresolved.
#
# CONSEQUENCE: rebuilding the C extensions in the main tree changes every
# worktree linked this way. Usually desirable; know it before debugging a
# mystery.
#
# Note the OTHER half of worktree setup, which this script does NOT do: the
# venv installs vllm editable and the finder hardcodes
#   MAPPING = {'vllm': '/mnt/beegfs/benishay/vllm-kvpipe/vllm'}
# as a MetaPathFinder consulted BEFORE sys.path, so PYTHONPATH cannot override
# it. Sharing the main venv means editing code here and running the main tree's
# code -- which produced two wrong diagnoses on 2026-09-04. Each worktree needs
# its own venv with that MAPPING repointed. Verify with
#   ./.venv/bin/python -c "import vllm; print(vllm.__file__)"
#
# Usage:  bash scripts/link_worktree_build_outputs.sh [main_tree] [worktree]
set -u

MAIN=${1:-/mnt/beegfs/benishay/vllm-kvpipe}
WT=${2:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}

if [ "$(readlink -f "$MAIN")" = "$(readlink -f "$WT")" ]; then
    echo "refusing to link a tree into itself: $WT"
    exit 1
fi
[ -d "$MAIN/vllm" ] || { echo "no $MAIN/vllm"; exit 1; }
[ -d "$WT/vllm" ] || { echo "no $WT/vllm"; exit 1; }

echo "main: $MAIN"
echo "into: $WT"

n=0
cd "$MAIN/vllm" || exit 1
while read -r f; do
    [ -z "$f" ] && continue
    mkdir -p "$(dirname "$WT/vllm/$f")"
    ln -sfn "$MAIN/vllm/$f" "$WT/vllm/$f"
    n=$((n + 1))
done < <(find . -type f ! -path '*__pycache__*' ! -name '*.pyc' \
           | while read -r g; do
               [ -e "$WT/vllm/$g" ] || echo "${g#./}"
             done)

echo "linked $n build outputs"
echo ".so in worktree: $(find "$WT/vllm" -name '*.so' | wc -l) "\
     "(main has $(find "$MAIN/vllm" -name '*.so' | wc -l))"
echo
echo "git status should stay clean -- all of these are gitignored:"
git -C "$WT" status --short | head -5
