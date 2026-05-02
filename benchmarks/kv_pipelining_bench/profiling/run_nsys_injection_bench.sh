#!/usr/bin/env bash
# Profile a command with Nsight Systems (CUDA + OSRT). Requires: nsys in PATH.
set -euo pipefail
OUT="${NSYS_OUT:-$PWD/nsys_injection}"
if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command...>" >&2
  exit 1
fi
exec nsys profile -o "$OUT" --trace=cuda,nvtx,osrt --force-overwrite true -- "$@"
