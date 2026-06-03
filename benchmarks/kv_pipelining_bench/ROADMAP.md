# KV layer-wise pipelining — maintained roadmap

Single session-to-session plan for the PrefillStaging / K+1 decode path.
Supersedes scattered notes in repo-root `NEXT_MOVE_PLAN.md` (historical) and
consolidates open items from `CONTINUATION_NOTES.md`, `DECODE_BENCH_SESSION_NOTES.md`.

**Technical reference:** [`TRANSFER_AND_OVERLAP_MODEL.md`](TRANSFER_AND_OVERLAP_MODEL.md)

---

## Goals

1. **Correctness** — K+1 shared-tensor decode matches baseline outputs on real prefill.
2. **Fair measurement** — baseline vs feature comparisons separate TTFT from steady decode ITL.
3. **Paper claims** — higher useful **throughput under ITL SLO** via smaller GPU KV footprint (k_window), not “same ITL as baseline.”
4. **k_window latency** — larger k_window should hide DDR/PCIe stalls when data movement dominates; verify in implementation + Llama-scale runs.

---

## Measurement protocol (mandatory for published numbers)

| Rule | Detail |
|------|--------|
| GPU preflight | `gpu_preflight.py` or `--require-gpu-free-fraction 0.85` before runs |
| Prefix caching | **`--no-enable-prefix-caching`** for primary results (never `--enable-prefix-caching false`) |
| Warmup | `--warmup-requests 2` before latency/throughput measurement |
| Full decode | `stream_options.include_usage`; fail if completion &lt; 90% of `max_tokens` |
| **TTFT vs ITL** | See below |

### Fair TTFT / ITL (baseline vs feature)

**Baseline:** after prefill, all layer KV is on GPU before the decode loop.

**Feature:** before first decode token, `start_load_kv` prefetches the first `k_window` layers from CPU → GPU. That work belongs in **TTFT** (time to first token), not in inter-token ITL.

| Metric | Definition |
|--------|------------|
| **TTFT** | Request start → first streaming chunk (prefill + initial k_window H2D + first forward) |
| **Mean ITL** | Mean gap between consecutive output chunks (all gaps) |
| **Mean ITL steady** | Same, but drop first `--itl-steady-skip` gaps (default **1**) so the first post-TTFT interval (pipeline fill) is not compared to baseline steady decode |

For **k_window sweeps**, report **mean_ITL_steady** as the primary ITL column; keep raw mean ITL for debugging.

Optional later: server-side marker for “decode pipeline ready” to split TTFT into prefill vs k_window fill.

### Prefix caching (dual protocol)

| Column | Use |
|--------|-----|
| **PC=OFF** | Our claims, k_window curves, overlap optimization |
| **PC=ON** | Head-to-head only vs external papers that leave caching enabled — **separate table/plot, never mixed** |

---

## Trusted results (2026-05-17)

- **Protocol fix:** prefix-caching CLI bug invalidated original kw=16 (and made kw=2/4/8 feature rows not comparable).
- **Apples-to-apples (PC=OFF, 256 tok):** kw=8 and kw=16 feature ITL both ~**146–148 ms**; baseline ~**7 ms**.
- **Qwen conc=1:** measured ITL ≫ PCIe H2D floor → **sync/overhead bound**, not bytes-bound (see `kv_transfer_budget.py`).
- **Tools:** `run_k_window_sweep.sh`, `run_k_window_apple_to_apple_validate.sh`, `gpu_preflight.py`.

---

## Open hypotheses

### H1 — k_window should improve latency when PCIe-bound

Theory: while computing layer `i`, prefetch layers `i+1…i+k` so decode rarely waits on DDR.

**If flat ITL on Qwen at all k:** may be sync-bound, not proof k_window is useless.

**Audit (Phase 2):** pipeline debug log, nsys kw=4/8/16, confirm first K layers ready before layer 0; measure stall fraction vs k on **Llama-8B conc=16**.

### H2 — Implementation may not overlap effectively

Suspects: `wait_for_layer_save` before each chained prefetch; `wait_for_saves` / `delta_queue.join`; per-layer host work.

---

## Backlog

### P0 — Measurement refresh (done 2026-05-18)

- [x] Full `run_k_window_sweep.sh` PC=OFF, warmup=2, steady ITL — `k_window_sweep_20260518_120552/`
- [x] `run_k_window_apple_to_apple_validate.sh` — all 8 cells passed

### P1 — k_window / overlap audit (done)

- [x] Pipeline debug kw=4 vs 16 — initial K prefetch verified
- [x] Nsight kw=4/8/16 — sync-bound on Qwen
- [x] Report: `/data/ben/kv_phase_a_internal/open_loop_eval_v1/k_window_audit_20260518/P1_KWINDOW_AUDIT.md`

### P2 — Overlap optimization (implemented 2026-05-19)

- [x] Audit identifies P2 targets (wait_for_layer_save chain, wait_for_saves, delta_queue)
- [x] **Code:** decode-only `wait_for_saves(deferred=True)`; GPU save waits on transfer stream in prefetch; per-layer CPU delta readiness before H2D (`kv_cache_pipelining.py`, `prefill_staging_pipelined_connector.py`)
- [x] **Qwen re-profile (2026-05-19):** `p2_measurement_20260519/` — steady ITL **~146–156 ms** conc=1 (was ~180–191), **~154–171 ms** conc=16 (was ~186–216); conc=16 throughput **+20–24%**. See `P2_RESULTS.md` in that dir.
- [ ] Optional: nsys re-profile post-P2; Llama-8B sweep with P2 code

### P3 — Llama-8B (in progress)

- [ ] k_window sweep — `llama8_k_window_sweep_20260518/` (running)
- [ ] Max concurrency under ITL SLO vs k_window (after sweep)

### P4 — Prefix caching comparison (done)

- [x] kw=8 PC-OFF vs PC-ON row — `k_window_sweep_20260518_120552/pc_on_comparison/`
- Note: with correct protocol both ~113–117 ms steady ITL at 256 tok (not old ~22 ms artifact)

### P5 — Deferred

- [ ] Concurrency crossover / mechanistic admission E2E until ITL protocol stable
- [ ] K+1 shared-tensor correctness at k &lt; num_layers (see CONTINUATION_NOTES §6)

### P6 — Decode-only P/D: CpuKvTable + direct K (2026-05, **active**)

**Plan doc:** [`KV_PD_DECODE_PLAN.md`](KV_PD_DECODE_PLAN.md)  
**Stage-to-CPU focus:** [`STAGE_TO_CPU_OPTIMIZATION.md`](STAGE_TO_CPU_OPTIMIZATION.md)

- [x] K-only GPU tensors + `inflate_kv_pool`; `DirectKWindowMixin`; upfront `CpuKvTable` (Option A DDR sizing)
- [x] `LayerwisePipelinedConnector` + `PipelinedDecodeBenchConnector` on shared path (no connector `WindowPool`)
- [ ] **P0** Stage-to-CPU: one-token D2H, less host overhead, narrower prefetch (see optimization doc)
- [ ] **P1** `PrefillStagingPipelinedConnector` — same mixin + table on decode
- [ ] **P2** Disjoint GPU scratch + parallel D2H/H2D on transfer stream (non-overlapping addresses)
- [ ] **P3** Dense CPU row layout (DDR / clarity; not primary ITL lever)
- [ ] **P4** nsys + Llama-scale validation; correctness with delta path on

---

## Commands cheat sheet

```bash
# GPU preflight
python benchmarks/kv_pipelining_bench/gpu_preflight.py \
  --min-free-fraction 0.85 --gpu-memory-utilization 0.85

# Full k_window sweep (primary)
bash benchmarks/kv_pipelining_bench/run_k_window_sweep.sh

# KV byte budget
python benchmarks/kv_pipelining_bench/kv_transfer_budget.py \
  --model Qwen/Qwen2.5-0.5B-Instruct --input-len 1024 --output-len 256 --concurrency 16

# Validate cells
bash benchmarks/kv_pipelining_bench/run_k_window_apple_to_apple_validate.sh
```

---

## Related files

| File | Role |
|------|------|
| `decode_only_bench.py` | Primary bench + steady ITL |
| `prefill_staging_pipelined_connector.py` | K+1 connector |
| `kv_cache_pipelining.py` | CPU/GPU staging + orchestrator |
| `CONTINUATION_NOTES.md` | Historical debug log (K+1 correctness) |
| `KV_PD_DECODE_PLAN.md` | Decode-only CpuKvTable + backlog (P6) |
| `STAGE_TO_CPU_OPTIMIZATION.md` | Ordered stage-to-CPU experiments |
| `NEXT_MOVE_PLAN.md` | **Superseded** — see this ROADMAP |
