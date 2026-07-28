# Hybrid tiered-KV — measured results (2026-07-28, Llama-3.1-8B / A100 80GB PCIe)

Fresh sweep on nova06. Fair baseline is `vanilla` (R0, full-HBM residency), NOT
the strawman `stock` arm. Data + figures:
`/mnt/beegfs/benishay/paper_runs/20260728/hybrid_m1/combined/`.
Config: input 512 / output 128, k_window 4, `--enforce-eager`, prefix caching on.

## 1. Hide-boundary: R0 (full HBM) vs R1 (all-CPU rolling K-window)

R1 keeps only a rolling 4-of-32 layer window of KV in HBM and streams the rest
from CPU per layer. If the transfer overlaps decode compute, R1 ITL ≈ R0 ITL.

| conc | R0 ITL ms | R1 ITL ms | Δ (R1−R0) ms | hidden? |
|---:|---:|---:|---:|:--:|
| 1  | 10.46 | 10.90 | +0.44 | yes |
| 4  | 10.73 | 11.22 | +0.49 | yes |
| 8  | 10.92 | 11.36 | +0.44 | yes |
| 16 | 11.87 | 12.05 | +0.18 | yes |
| 32 | 13.22 | 13.47 | +0.24 | yes |
| 48 | 14.81 |  (a)  |  —    |  —  |
| 64 | 15.13 |  (a)  |  —    |  —  |

**R1 transfer is fully hidden through conc 32** (Δ ≤ 0.5 ms, SLO 2 ms). Fig:
`fig_itl_vs_conc.png`. (a) conc 48/64 R1 cells failed this run (high-concurrency
pipelining fragility, ok=1/48, 0/64); the July-5 `hide_boundary_v2` sweep (same
S/K, on a larger-RAM allocation) reached conc 64 with R1 ITL 15.07/15.38 ms vs
R0 ~14.8/15.1 — still hidden. So the hidden regime extends to at least 64.

## 2. Composite goodput projection (measured curves + roofline)

Hybrid serves `B_hbm` HBM-resident + `B_off` CPU-tier reqs together:
`tok/s ≈ B_hbm/ITL_R0(B_hbm) + B_off/ITL_R1(B_hbm+B_off)`, with `B_off` from the
PCIe/NVLink hide roofline (`vllm/v1/hybrid_kv_roofline.max_offload_batch`).

| B_hbm | B_off (A100) | gain A100 | B_off (GH200) | gain GH200 |
|---:|---:|---:|---:|---:|
| 8  | 3 | 1.35x | 62 | 7.28x |
| 16 | 3 | 1.18x | 67 | 4.69x |
| 32 | 3 | 1.09x | 75 | 3.30x |
| 48 | 4 | 1.09x | 84 | 2.92x |
| 64 | 4 | 1.07x | 86 | 2.51x |

Fig: `fig_composite_goodput.png`. **On A100/PCIe4 (X_eff≈25 GB/s) the win is
small (≈1.1x) — the roofline says B_off≈3-4.** On GH200/NVLink-C2C
(X_eff≈480 GB/s) B_off≈60-86 and the composite goodput is **2.5-4.7x** R0-only
at realistic batch. This is the thesis: the feature's value lives on fast
interconnect; A100 is the conservative floor and its composite is already
bounded by the two measured curves.

## 3. Status of the mixed-batch dual-KV forward (M1 core)

NOT delivered this session, by deliberate decision. Measuring the composite
*directly* (one mixed forward) requires a dual-region per-layer KV allocator +
two-pass attention — a global memory-layout change (each Attention layer today
binds one KV pool; the rolling-K vs full-N layout is a per-model-instance
decision). Full analysis + implementation plan + verification ladder in
`MIXED_BATCH_DUAL_KV_DESIGN.md`. On A100 its marginal value over the two-curve
bound above is small; its payoff is on fast fabric, so it is scoped as the next
step rather than rushed into an unvalidatable state.

## Reproduce

```
# on the GPU node (srun --jobid=<J> --overlap), from repo root:
export VLLM_KVPIPE_BATCHED_D2H=1 VLLM_KVPIPE_STEP_PLAN=1
# R0: .venv/bin/python .../pipelined_decode_fast_bench.py --modes vanilla \
#   --concurrency 1 4 8 16 32 48 64 --input-len 512 --output-len 128 --max-model-len 1024 ...
# R1: same with --modes pipelined --no-inflate-kv-pool --pipelined-gpu-mem-cap 0.35
#   (64GB-cgroup nodes: shrink the block pool or CpuKvTable host-OOMs — see reference-run-infra)
# then: analyze_hide_boundary.py <dir> ; composite_projection.py <dir>
```
