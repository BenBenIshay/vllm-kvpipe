#!/usr/bin/env bash
# Tiny per-cell runs to validate sweep protocol (full decode, comparable setup).
#
# Usage:
#   bash benchmarks/kv_pipelining_bench/run_k_window_apple_to_apple_validate.sh
#   REF_CSV=/path/to/k_window_sweep_summary.csv OUT=/tmp/validate bash ...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BENCH="${ROOT}/benchmarks/kv_pipelining_bench"
REF_CSV="${REF_CSV:-/data/ben/kv_phase_a_internal/open_loop_eval_v1/k_window_sweep_20260517_084927/k_window_sweep_summary.csv}"
OUT="${OUT:-/data/ben/kv_phase_a_internal/open_loop_eval_v1/k_window_sweep_20260517_084927/apple_to_apple_validate_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
K_WINDOWS="${K_WINDOWS:-2 4 8 16}"
MODES="${MODES:-baseline real_prefill}"
MIN_FREE="${MIN_FREE:-0.85}"
GPU_UTIL="${GPU_UTIL:-0.85}"
# Same shape as sweep but shorter for speed
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"
PORT_BASE="${PORT_BASE:-8300}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/ben/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUT"

echo "=== GPU preflight (need >= ${MIN_FREE} free) ==="
"$PY" "$BENCH/gpu_preflight.py" \
  --min-free-fraction "$MIN_FREE" \
  --gpu-memory-utilization "$GPU_UTIL"

port="$PORT_BASE"
for kw in $K_WINDOWS; do
  for mode in $MODES; do
    cell_dir="$OUT/kw${kw}_${mode}"
    mkdir -p "$cell_dir"
    echo ""
    echo "########################################"
    echo "# validate k_window=$kw mode=$mode port=$port"
    echo "########################################"
    "$PY" "$BENCH/decode_only_bench.py" \
      --model "$MODEL" \
      --modes "$mode" \
      --concurrency 1 \
      --input-len "$INPUT_LEN" \
      --output-len "$OUTPUT_LEN" \
      --k-window "$kw" \
      --streaming-throughput \
      --gpu-memory-utilization "$GPU_UTIL" \
      --require-gpu-free-fraction "$MIN_FREE" \
      --result-dir "$cell_dir" \
      --port "$port" \
      --warmup-requests 1 \
      --itl-steady-skip "${ITL_STEADY_SKIP:-1}" \
      2>&1 | tee "$cell_dir/run.log"
    port=$((port + 1))
    sleep 3
  done
done

echo ""
echo "=== Compare to reference CSV ==="
REF_CSV="$REF_CSV" OUT="$OUT" OUTPUT_LEN="$OUTPUT_LEN" "$PY" - <<'PY'
import csv
import json
import os
import sys
from pathlib import Path

ref_csv = Path(os.environ["REF_CSV"])
out = Path(os.environ["OUT"])
output_len = int(os.environ["OUTPUT_LEN"])
# ITL may differ with output_len=32 vs 256; check token count + order of magnitude
ref: dict[tuple[int, str, int], dict] = {}
with ref_csv.open() as f:
    for row in csv.DictReader(f):
        if row["mode"] == "baseline" and int(row["K"]) != 1:
            continue
        if int(row["K"]) != 1:
            continue
        key = (int(row["k_window"]), row["mode"])
        ref[key] = {
            "mean_itl_ms": float(row["mean_itl_ms"]) if row["mean_itl_ms"] else None,
            "throughput": float(row["throughput_tok_s"]) if row["throughput_tok_s"] else None,
        }

failures: list[str] = []
rows: list[dict] = []
for cell in sorted(out.glob("kw*_*")):
    name = cell.name  # kw16_real_prefill
    parts = name.split("_", 1)
    kw = int(parts[0][2:])
    mode = parts[1]
    jpath = cell / f"results_{mode}.json"
    if not jpath.is_file():
        failures.append(f"{name}: missing {jpath.name}")
        continue
    data = json.loads(jpath.read_text())
    tp = (data.get("throughput") or [{}])[0]
    tokens = int(tp.get("total_output_tokens") or 0)
    itl_ms = None
    if tp.get("concurrent_mean_itl_s_mean") is not None:
        itl_ms = float(tp["concurrent_mean_itl_s_mean"]) * 1000.0
    ok_tokens = tokens >= int(0.9 * output_len)
    if not ok_tokens:
        failures.append(f"{name}: tokens={tokens} expected>={int(0.9 * output_len)}")
    log = cell / f"server_{mode}.log"
    if mode == "real_prefill" and log.is_file():
        text = log.read_text(errors="replace")
        if "enable_prefix_caching': True" in text or "enable_prefix_caching=True" in text:
            failures.append(f"{name}: prefix caching enabled in server log")
        if "enable_prefix_caching': False" not in text and "enable_prefix_caching=False" not in text:
            failures.append(f"{name}: cannot confirm prefix caching disabled")
    ref_row = ref.get((kw, mode))
    ratio_note = ""
    if ref_row and itl_ms and ref_row["mean_itl_ms"]:
        ratio = itl_ms / ref_row["mean_itl_ms"]
        ratio_note = f"itl_ratio_vs_sweep={ratio:.2f}"
        # Tiny decode: allow wide band; catch bogus ~12ms kw16 or 10x drift
        if mode == "real_prefill" and kw == 16 and itl_ms < 30:
            failures.append(f"{name}: ITL {itl_ms:.1f}ms suspiciously low (short-decode bug?)")
        if ratio > 5.0 or ratio < 0.15:
            failures.append(
                f"{name}: ITL {itl_ms:.1f}ms vs sweep {ref_row['mean_itl_ms']:.1f}ms ratio={ratio:.2f}"
            )
    rows.append({
        "k_window": kw,
        "mode": mode,
        "tokens": tokens,
        "mean_itl_ms": itl_ms,
        "ref_itl_ms": ref_row["mean_itl_ms"] if ref_row else None,
        "note": ratio_note,
    })

report = out / "validate_report.json"
report.write_text(json.dumps({"cells": rows, "failures": failures}, indent=2))
print(f"Wrote {report}")
for r in rows:
    print(
        f"  kw={r['k_window']:2d} {r['mode']:12s} tokens={r['tokens']:3d} "
        f"itl={r['mean_itl_ms']:.2f}ms  ref={r['ref_itl_ms']}  {r['note']}"
    )
if failures:
    print("\nFAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("\nAll validation cells passed.")
PY
