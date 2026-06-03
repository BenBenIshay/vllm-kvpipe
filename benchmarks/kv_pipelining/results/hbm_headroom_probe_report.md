# Decode HBM headroom probe — results

**Date:** 2026-04-05  
**Host:** local run (`ben` workspace)  
**Script:** `benchmarks/kv_pipelining/decode_hbm_headroom_probe.py`

## Purpose

Validate how much a **memory-bound GPU kernel** (streaming `torch.add` on large tensors) slows when **concurrent pinned H2D DMA** runs on a separate CUDA stream — a proxy for **KV prefetch / CPU→GPU copies** overlapping decode-style memory traffic.

## Environment

| Item | Value |
|------|--------|
| GPU | NVIDIA H100 NVL (95 830 MiB) |
| PyTorch | 2.9.1+cu126 |
| CUDA available | Yes |
| Probe settings | `--tensor-mb 256`, `--iters 200`, `--warmup 20`, `float32` |

**Reference peak HBM** in the script’s lookup table for “H100” is **3350 GB/s** (product brief order-of-magnitude). Measured **solo “effective” GB/s** can exceed this (see Caveats).

## Results — DMA sweep

Fixed: **256 MiB** tensors, **200** iterations per phase, **32 MiB** H2D per iteration unless noted.

| DMA per iter (MiB) | Solo kernel effective (GB/s) | Overlap kernel effective (GB/s) | Ratio (overlap / solo) | Notes |
|--------------------|-------------------------------|-----------------------------------|-------------------------|--------|
| 0 | 3594 | — | — | No concurrent DMA |
| 8 | 3592 | 3535 | **0.984** | Minimal impact |
| 16 | 3594 | 2664 | **0.741** | Noticeable contention |
| 32 | 3592 | 1350 | **0.376** | Strong contention |
| 64 | 3584 | 679 | **0.190** | Severe contention |

**Solo H2D-only** (32 MiB × 200 iters, copy stream only): **~56.5 GB/s** — consistent with **PCIe Gen5**-limited host→device traffic for this pattern, not HBM peak.

## Interpretation

1. **Concurrent H2D and a memory-bound kernel share the memory subsystem.** On this H100 NVL, small overlapping transfers (**8 MiB/iter**) barely reduce effective kernel bandwidth; larger per-iteration copies (**32–64 MiB**) **sharply** reduce it.

2. **For KV pipelining / DDR offload:** prefer **smaller, steadier** prefetches (or fewer layers resident off-GPU per step) so per-iteration DMA stays in the **low-contention** regime, unless profiling shows spare effective bandwidth.

3. **Capacity vs bandwidth:** this probe does **not** measure VRAM exhaustion; pair with `capacity_test.py` or `decode_only_bench.py` for **OOM / concurrency** limits.

## Caveats

- **“Effective GB/s” for the add kernel** uses bytes moved = 2 loads + 1 store per element; **solo numbers above nominal HBM peak** likely reflect **L2/cache reuse**, **measurement granularity**, and **not** a rigorous DRAM-only counter. Use **relative** comparisons (ratio overlap vs solo) as the primary signal.
- Real **decode** kernels differ (attention, fused ops, launch gaps). This is a **synthetic lower bound** on sensitivity to overlap.
- **H2D ~56 GB/s** is **PCIe-bound**, not HBM-bound; overlap still competes for **HBM** on the GPU side of the copy engine.

## Raw outputs

| File | Contents |
|------|-----------|
| `hbm_headroom_probe_run.json` | Single run (`dma_mb_per_iter=32`), JSON |
| `hbm_headroom_probe_run.txt` | Same run, human-readable stdout |
| `hbm_headroom_dma_sweep.json` | DMA sweep 0–64 MiB (numeric fields as measured) |

## Commands to reproduce

```bash
source .venv/bin/activate
cd /home/ben/vllm

# Full JSON (default-style run)
python benchmarks/kv_pipelining/decode_hbm_headroom_probe.py \
  --tensor-mb 256 --dma-mb-per-iter 32 --iters 200 --warmup 20 --print-peak --json

# Sweep (example)
for dma in 0 8 16 32 64; do
  python benchmarks/kv_pipelining/decode_hbm_headroom_probe.py \
    --tensor-mb 256 --dma-mb-per-iter $dma --iters 200 --warmup 20 --print-peak --json
done
```
