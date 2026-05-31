# KV Pipelining Optimizations — Summary (Paper Notes)

Decode-only layer-wise KV pipelining for a P/D **decode node**: keep **K** full layer KV tensors in HBM (rolling window), back the rest in **CPU DDR**, and overlap decode compute with per-layer D2H (new token) and H2D (prefetch layer *i+K*).

**Bench:** `pipelined_decode_fast_bench.py` · **Connectors:** `PipelinedDecodeBenchConnector`, `LayerwisePipelinedConnector`  
**Related docs:** [`KV_PD_DECODE_PLAN.md`](KV_PD_DECODE_PLAN.md), [`STAGE_TO_CPU_OPTIMIZATION.md`](STAGE_TO_CPU_OPTIMIZATION.md)

---

## 1. System architecture

| Component | Role |
|-----------|------|
| **K-only GPU allocator** + optional `inflate_kv_pool` | Same HBM KV budget; larger logical block pool via *num_layers/K* inflation (capacity mode) |
| **`CpuKvTable`** | Upfront pinned DDR table (Option A sizing from unique GPU KV tensor bytes at `register_kv_caches`) |
| **`DirectKWindowMixin`** | Bind layers to allocator rolling K tensors (no connector `WindowPool`) |
| **Connectors** | Hooks: `start_load_kv` → layer loop → `save_kv_layer` → `wait_for_save` |

**Correctness invariant:** After attention on layer *i*, D2H for layer *i* must complete (or use a disjoint scratch snapshot) before H2D prefetch into the rolling GPU slot reused by layer *i+K*.

**Lean ITL benchmarking:** `--no-inflate-kv-pool` avoids multi-minute / 100+ GiB CPU pin at startup; measures transfer + hook path, not max capacity.

---

## 2. Per-token / single-request path (conc = 1)

| ID | Name | What we did | Main files |
|----|------|-------------|------------|
| **Step 1** | One-token D2H | n=1 fast path copies `kv[phys, off]` → CPU, not a full page (~16× fewer bytes/layer on Qwen 0.5B) | `vllm/v1/cpu_kv_table.py` (`_copy_n1_token`, `stage_delta_host_slot`) |
| **Step 3a** | One transfer-stream sync per step | `sync_decode_transfer_stream_once()` — single `wait_stream(compute)` per forward, not 24× | `vllm/v1/kv_cache_pipelining.py` |
| **Step 3b** | Dual CUDA streams | D2H on `_stream`; H2D prefetch on `_prefetch_stream` for overlap after snapshot | `kv_cache_pipelining.py`, `cpu_kv_table.py` |
| **Step 3c** | Bootstrap prefetch | Prefetch first **K** layers at `wait_for_save`; skip in `start_load_kv` if layer already `ready` | `kv_pipelining_direct_window.py`, connectors |
| **Step 3d** | Lean `wait_for_layer_load` | Direct `wait_event` when heavy profiling/logging off | connectors |
| **Step 3e** | Preallocated CUDA events | Per-layer `save` / `snap` / `ready` event pools (no per-step event alloc) | `kv_cache_pipelining.py` |
| **Host slot cache** | Resolve slot once per step (n=1) | `resolve_n1_host_slot` + `begin_decode_n1_step` — avoid repeated GPU scalar sync | `kv_cache_pipelining.py` |
| **Phase H (5b)** | Delta D2H scratch (**default on**) | Per rolling slot: `main[phys,off] →` tiny HBM scratch → D2H; prefetch waits `snap_event` only so D2H and H2D can overlap | `cpu_kv_table.py`, `kv_pipelining_direct_window.py`; env `VLLM_KVPIPE_DELTA_SCRATCH=1` |
| **Phase H (5c)** | Overlap-aware block prefetch | `delta_phys` / block-run aware H2D when scratch path off | `kv_cache_pipelining.py` |
| **P2 (decode deferral)** | Deferred `wait_for_saves` | Decode-only: do not block compute stream at end of step; ordering via per-layer events + delta scatter queue | `kv_cache_pipelining.py` |

**Documented regression (off):** Phase H **5a** block scratch-prefetch (`VLLM_KVPIPE_SCRATCH_PREFETCH=1`) — ~6.3 ms steady ITL regression; not used in production path.

**Step 4 (prefetch scope):** For conc=1, prefetch uses scheduler-active physical blocks, not the full inflated pool. Multi-concurrency union trim left as future work.

---

## 3. Multi-request / batched decode (conc ≥ 4)

| ID | Name | What we did | Main files |
|----|------|-------------|------------|
| **B1** | Batched per-layer D2H | For N tokens in one forward: **one gather + one D2H + CPU scatter** per layer (not N separate copies) | `cpu_kv_table.py` (`enqueue_batched_n1_delta_via_scratch`), `stage_decode_batched_layer`, batched GPU/CPU scratch slabs in `kv_pipelining_direct_window.py` |
| **B2** | Step-level transfer plan | **Prepare once** at `start_load_kv`: resolve `slot_mapping` from forward-level attention metadata; **thin** `save_kv_layer` → `_issue_decode_layer_step` (stage + prefetch only) | `vllm/v1/kv_pipelining_step.py`, `kv_pipelining_direct_window.py`, connectors |

**Evaluation toggles:**

- `VLLM_KVPIPE_BATCHED_D2H=1` (default) — enable B1  
- `VLLM_KVPIPE_STEP_PLAN=1` (default) — enable B2  

**Critical coupling:** B1 without B2 (or per-layer metadata reporting `num_actual_tokens=1` while batch has N>1) falls back to slow per-token staging (~10× ITL at conc=8 in validation). B2 without B1 still ~90 ms at conc≥4. **Both required** for batched decode performance.

**`prepare_decode_batch` fix:** Do not slice `slot_mapping` to `num_actual_tokens` when that value is 1 but the batch has multiple slots (use full `slot_mapping.numel()` for batching decisions).

---

## 4. Measurement and tooling

| Item | Purpose |
|------|---------|
| **Profiler v2** (`VLLM_KVPIPE_PROFILE=1`) | Per-forward `KVPIPE_FWD` / `KVPIPE_STEADY`: `gpu_stall_sum`, `save_host`, `start_load`, `bootstrap`, etc. **Not** wall-clock ITL |
| **`pipelined_decode_fast_bench.py`** | Dummy-KV decode bench; isolated (5-run) + concurrent throughput ITL |
| **`--enforce-eager`** | CUDA graphs off (required for per-layer KV connector hooks) |
| **`VLLM_KVPIPE_SKIP_DELTA_STAGE=1`** | Diagnostic only — skips D2H; invalid for production |
| **`sweep_b1_b2_aligned.sh`** | Factorial B1/B2 validation aligned with K4 sweep (conc 1/4/8, isolated on) |

---

## 5. Measured outcomes (Qwen2.5-0.5B-Instruct, K=4, enforce-eager, no inflate pool)

Aligned validation: `/tmp/kv_b1_b2_aligned_v2/` (gpu_memory_utilization=0.72). K4 reference: `/tmp/kv_profile_v2/sweep/K4/`.

| Configuration | conc=1 steady ITL | conc=8 steady ITL | Notes |
|---------------|-------------------|-------------------|--------|
| **Baseline** (`DecodeBenchConnector`, GPU KV) | **6.5 ms** | **7.0 ms** | No CPU table / pipelining |
| **K4 reference pipelined** | **8.1 ms** | **17.8 ms** | Pre B1/B2 explicit flags; gpu 0.90 |
| **B1+B2 (`both`, defaults)** | **9.0 ms** | **12.8 ms** | Production path |
| **Neither flag** | 8.9 ms | **123 ms** | Per-token fallback at conc≥4 |
| **B1 only** | 8.7 ms | **125 ms** | Missing B2 routing |
| **B2 only** | 8.8 ms | **129 ms** | Missing B1 batched D2H |

**Claims supported by data:**

- Layer-wise CPU-backed KV with K-window HBM **materially improves multi-request decode ITL** when B1+B2 are enabled (~18 ms → ~13 ms vs early pipelined reference; ~10× vs disabled flags at conc=8).
- **Single-request** overhead vs GPU-only baseline (~2–2.5 ms) remains after Step 1 + Phase H + stream amortization; B1/B2 do not target that gap.
- B1 and B2 are **complementary** at conc≥4; neither alone is sufficient.

**Do not over-claim:** The roadmap target **6.8–7.0 ms @ conc=1** after “B1+B2” was optimistic; those optimizations address **batching and step-level planning**, not the full conc=1 hook/prefetch tax. Per-token work (Step 1, Phase H, Step 3) is largely done; remaining conc=1 gap is likely **fixed per-forward overhead** (`start_load_kv`, 24× hooks, rolling-K orchestration).

---

## 6. Bottleneck classes (reference)

| Class | Symptom | Addressed by |
|-------|---------|--------------|
| D2H bytes (full page) | High PCIe / memcpy per layer | **Step 1**, Phase H 5b |
| Per-layer host sync | 24× `wait_stream` | **Step 3a** |
| Transfer stream serialization | Save blocks prefetch | **Phase H 5b** (dual stream + snap) |
| Multi-token D2H | N copies per layer at conc≥4 | **B1** |
| Wrong batch routing / late prepare | ~90–220 ms ITL at conc≥4 | **B2** + plan-based issue path |
| Hook / `start_load` overhead | conc=1 gap vs baseline | **Open** — profile with `KVPIPE_PROFILE` |

---

## 7. Suggested next work (conc = 1)

1. Profile conc=1 only: `VLLM_KVPIPE_PROFILE=1`, compare `start_load`, `save_host`, `gpu_stall_sum` vs wall ITL.  
2. Attack **`start_load_kv` / bootstrap** if `start_load` dominates.  
3. Reduce **24× connector hook** cost (lighter fast path or fewer invocations).  
4. Audit **rolling-K bind/prefetch** for redundant work when `ready_event` already set.  
5. Use **conc≥4** to validate any change that touches B1/B2; use **conc=1** as regression guard (~8–9 ms, not worse than ~10 ms).

---

## 8. Environment variables (quick reference)

| Variable | Default | Meaning |
|----------|---------|---------|
| `VLLM_KVPIPE_BATCHED_D2H` | `1` | B1 batched gather+D2H |
| `VLLM_KVPIPE_STEP_PLAN` | `1` | B2 step prepare + thin issue |
| `VLLM_KVPIPE_DELTA_SCRATCH` | `1` | Phase H 5b delta scratch path |
| `VLLM_KVPIPE_SCRATCH_PREFETCH` | `0` | Phase H 5a (regression if on) |
| `VLLM_KVPIPE_PROFILE` | `0` | Per-forward hook profiler |
| `VLLM_KVPIPE_SKIP_DELTA_STAGE` | `0` | Diagnostic: no D2H |

---

*Last updated from development session May 2026. Re-run aligned sweep: `benchmarks/kv_pipelining_bench/sweep_b1_b2_aligned.sh`.*
