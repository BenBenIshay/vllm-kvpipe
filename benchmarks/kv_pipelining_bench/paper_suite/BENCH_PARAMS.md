# Paper Benchmark Parameters (lit-validated)

Parameters for the layer-wise KV pipelining paper suite, validated against three
KV-offload / sparse-attention papers. We do **exact full-KV offload** (no token
dropping), so we are *more* PCIe-bound than sparse methods - which is why batch
amortization (C4) and the SLO-aware scheduler (C6) are the contribution, not a
sparsity trick.

## Related-work anchors

| Paper | Models | Context | Batch / concurrency | HW | Headline |
|-------|--------|---------|---------------------|----|----------|
| RetroInfer (arXiv 2505.02922) | Llama-3.1-8B (GQA-4), Qwen2.5-7B (GQA-7), Llama3-8B-1048K | 4K-128K (RULER), up to 1M (NIAH) | batch fixed (no vLLM sched); A100-80GB OOM past **batch 4 @ 128K**; BW saturates past batch 3 | 1x A100-80GB, PCIe4 x16 (32 GB/s) | 4.5x vs full-attn in HBM; 10.5x vs sparse w/ CPU KV |
| InfiniGen (arXiv 2406.19707) | OPT-6.7B/13B/30B, Llama-2-7B/13B | 2K-4K | batch 16/20; 1920 in / 128 out | FlexGen offload | up to 3.0x |
| LiteCache (arXiv 2511.14510) | Llama3-8B, Qwen2.5-14B | 4K-1M | varying | **H100** + A40 | +67-224% decode tput on H100; CPU cache mgmt = **45-68%** of GPU kernel time |

## Adopted grid (result-collection phase)

- **Context lengths:** 4K, 8K, 16K, 32K, 64K, 128K (RULER ladder); optional 256K-1M offload-only stress.
- **Output length:** 128 (matches InfiniGen).
- **Concurrency:** capacity claims sweep batch 1 -> baseline OOM wall at fixed long context (single-digit at 128K per RetroInfer); C4 amortization keeps the 1-64 ladder at moderate ISL.
- **K window:** 4 for micro-claims; sweep {4,6,8} for capacity/SLO.

### Models (layer-count axis, per-layer KV ~held constant)

| Model | Layers | KV heads x head_dim | KV/layer/token | KV/token total | Role |
|-------|--------|---------------------|----------------|----------------|------|
| Qwen2.5-0.5B | 24 | 2 x 64 (GQA) | ~0.5 KB | ~12 KB | C3/C4 micro + **smoke/correctness** |
| Llama-3.1-8B | 32 | 8 x 128 | **4 KB** | 128 KB | C1/C2/C5/C6 primary; controlled twin of 32B |
| Qwen2.5-32B | **64** | 8 x 128 | **4 KB** | 256 KB | C1/C2 sharpest capacity wall; 64-layer endpoint |
| Qwen2.5-7B (optional) | 28 | 4 x 128 | 2 KB | 56 KB | architecture variety only |

**Controlled comparison:** Llama-3.1-8B (32L) vs Qwen2.5-32B (64L) have identical
4 KB/layer/token but 2x layers -> isolates layer-count/orchestration scaling from
transfer bytes (second axis alongside the R1~=R3 fake-transfer result).

**32B caveats:** fp16 weights ~64 GB of the 94 GB H100 (gpu_memory_utilization
~0.9, ~30 GB KV headroom); pinned CpuKvTable reserves host DDR ~ full KV footprint
(confirm host RAM); longer load/compile. At conc=1 the 32B baseline ITL is
weight-BW-dominated, so report C3 as absolute per-forward orchestration ms +
host-hook/GPU-kernel ratio (LiteCache-style), not just % of baseline ITL.

## Two-phase scope

- **Now (shared GPU):** develop scripts + validate correctness with `--smoke`
  params on **Qwen2.5-0.5B only** (short context, conc 1-2, tiny prompt counts).
  No result collection.
- **Later (free GPU):** run the adopted grid above for publishable numbers.

## Smoke params (Qwen2.5-0.5B only)

| Knob | Smoke value |
|------|-------------|
| model | Qwen/Qwen2.5-0.5B-Instruct |
| gpu_memory_utilization | 0.30 |
| max_model_len | 2048 |
| k_window | 4 |
| concurrency (capacity) | 2 4 |
| num_prompts (capacity) | 12 |
| input/output len (capacity) | 256 / 16 |
| goodput SLOs | 200 ms |
| goodput rates | 4 req/s |
| goodput num_prompts | 12 |
