#!/usr/bin/env bash
# Phase 4: Qwen-32B hide-boundary sweep (longer decode → larger hide window).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
export CONC_LIST="${CONC_LIST:-1 4 8 12 16 20 24 28 32}"
export GPU_UTIL="${GPU_UTIL:-0.90}"
OUT="${1:-/mnt/beegfs/benishay/paper_runs/20260705/hybrid_kv/hide_boundary_qwen32b}"
PORT_BASE="${2:-9800}"
exec bash "$(dirname "$0")/run_hide_boundary_sweep.sh" "$OUT" "$PORT_BASE"
