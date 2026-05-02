#!/usr/bin/env bash
# Sample GPU counters while a PID runs.
set -euo pipefail
PID="${1:?pid}"
INT="${2:-1}"
if ! kill -0 "$PID" 2>/dev/null; then
  echo "pid $PID not running" >&2
  exit 1
fi
exec nvidia-smi dmon -s pucvmet -d "$INT" -p "$PID"
