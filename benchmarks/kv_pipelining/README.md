# KV Pipelining OOM Sweep

This directory contains a simple sweep script to compare decode-only
concurrency limits with and without KV cache pipelining.

## Usage

Baseline:

```
benchmarks/kv_pipelining/oom_sweep.sh baseline
```

Pipelined:

```
benchmarks/kv_pipelining/oom_sweep.sh pipelined
```

## Environment variables

- `MODEL` (default: `meta-llama/Llama-2-7b-hf`)
- `PORT` (default: `8000`)
- `INPUT_LEN` (default: `4096`)
- `OUTPUT_LEN` (default: `256`)
- `MAX_MODEL_LEN` (default: `8192`)
- `GPU_UTIL` (default: `0.90`)
- `START_CONCURRENCY` (default: `8`)
- `END_CONCURRENCY` (default: `128`)
- `STEP` (default: `8`)
- `K_WINDOW` (default: `32`)

## Notes

- This script assumes **decode-only** runs (no prefill on the server).
- Use `--no-enable-prefix-caching` for correctness with pipelining (`--enable-prefix-caching false` incorrectly enables caching in vLLM).
- Increase `K_WINDOW` for slower interconnects (PCIe).
