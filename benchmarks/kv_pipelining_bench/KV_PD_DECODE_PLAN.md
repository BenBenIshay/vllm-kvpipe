# Decode-only P/D KV plan — CpuKvTable + direct K tensors

Maintained plan for **decode-only** layer-wise pipelining (`decode_only_pipelining`,
`inflate_kv_pool`, `kv_k_window=K`). Targets a dedicated D node with upfront DDR.

**Technical reference:** [`TRANSFER_AND_OVERLAP_MODEL.md`](TRANSFER_AND_OVERLAP_MODEL.md)  
**Bench:** [`pipelined_decode_fast_bench.py`](pipelined_decode_fast_bench.py) + `PipelinedDecodeBenchConnector` / `LayerwisePipelinedConnector`

---

## Target architecture

| Resource | Role |
|----------|------|
| **GPU HBM** | **K** rolling layer tensors (same total KV budget as baseline); inflated block pool → ~`num_layers/K` more concurrent decodes |
| **CPU DDR** | **One upfront table** at init: `cpu_bytes ≈ (num_layers / K) × hbm_kv_bytes` (**Option A**: sum unique GPU KV tensor bytes at `register_kv_caches`) |
| **Per layer row** | Pinned buffer per logical layer; partitioned by **req_slot × logical_block** (not dynamic per-admit `malloc` in hot path) |

**Admission:** assign `req_slot` + map `block_table[req][logical_b] → physical_P`; release slot on finish.

---

## Correctness rules (do not violate)

1. **Rolling slot reuse:** Layer `i` and `i+K` share GPU tensor `i % K`. After attention on layer `i`, **D2H of layer `i`’s new KV** must complete (or read from a disjoint scratch region) **before** H2D prefetch of layer `i+K` overwrites that GPU memory.
2. **`save_kv_layer` on decode** means **stage new token(s)** via `slot_mapping` — **not** full-layer dump (full copy only on **seed** / prefill handoff).
3. **No end-of-step batch** of all layer saves: deferring all D2H until `wait_for_save` loses layer `i`’s GPU state once slot `i+K` prefetches.
4. **Layer `i+K`’s new token** does not exist before layer `i+K` runs; the hazard is **clobbering layer `i`’s just-written GPU data**, not `i+K`’s future token.

---

## Completed (2026-05)

| Item | Location |
|------|----------|
| K-only allocator + `inflate_kv_pool` on uniform path | `vllm/v1/core/kv_cache_utils.py` |
| `CpuKvTable` upfront alloc (Option A sizing) | `vllm/v1/cpu_kv_table.py` |
| `DirectKWindowMixin` (no connector `WindowPool`) | `vllm/v1/kv_pipelining_direct_window.py` |
| `LayerwisePipelinedConnector` refactor | `layerwise_pipelined_connector.py` |
| `PipelinedDecodeBenchConnector` on shared path | `pipelined_decode_bench_connector.py` |
| `LayerwisePipeliningState` table seed / delta / prefetch | `vllm/v1/kv_cache_pipelining.py` |

### Preliminary bench (Qwen2.5-0.5B, `gpu_mem=0.2`, K=4, inflate on, in=64, out=32, conc=1)

| Mode | Throughput | Mean ITL |
|------|------------|----------|
| Baseline (`DecodeBenchConnector`) | ~100 tok/s | ~3.6 ms |
| Pipelined + CpuKvTable | ~6–7 tok/s | ~58–77 ms |
| Pipelined + `VLLM_KVPIPE_SKIP_DELTA_STAGE=1` | ~60 tok/s | ~7.4 ms |

**Interpretation:** H2D wait (`wait_for_layer_load`) is ~µs; **`stage_delta_to_cpu` / `save_kv_layer` path dominates** (~4.9–5.9 ms × 24 layers/step host-reported). Skip-delta is diagnostic only (CPU table stale).

---

## Backlog (prioritized)

### P0 — Stage-to-CPU hot path (next focus)

See **[`STAGE_TO_CPU_OPTIMIZATION.md`](STAGE_TO_CPU_OPTIMIZATION.md)** for ordered experiments and metrics.

Summary:

- Copy **one token** (or minimal slots), not a full **page** per layer.
- Cut **per-layer Python / CUDA event** overhead (24×/step).
- **Prefetch scope:** only blocks needed for layer `i+K`, not all active physical blocks every binding.
- Profile transfer stream (nsys) after each change.

**Explicitly lower priority for ITL:** “smaller CPU row” tensor shapes — saves **DDR** and clarifies layout; **does not** shrink bytes moved if page-sized D2H stays the same.

---

### P1 — `PrefillStagingPipelinedConnector` alignment

- Inherit **`DirectKWindowMixin`** + **`CpuKvTable`** on decode (same as `LayerwisePipelinedConnector`).
- Keep prefill-specific **slice staging** and metadata; only unify backing store + K binding.
- Re-run `validate_qa.py` / `eval_pd_decode.py` after integration.

---

### P2 — Disjoint GPU scratch + parallel D2H/H2D (Phase H)

**Goal:** Overlap transfer work on the **same rolling slot** without address races.

**Design:**

- Reserve a **per-request scratch** region in the GPU tensor (fixed slots / blocks) where attention writes **this step’s** new KV.
- **Main** region holds prefetched history for the layer about to execute.
- On `save_kv_layer(i)` after attention:
  - **D2H:** scratch → CPU row `i` (small)
  - **H2D:** CPU row `i+K` → main region (may be larger)
  - Queue both on **transfer stream** without `wait_event(save_i)` **if regions are disjoint**
- Before attention on layer `i+K`: compute stream waits only on **prefetch ready** into main.

**Requires:** contract with **`slot_mapping` / `reshape_and_cache`** (or post-attn gather into scratch).

**Value:** Moderate — hides **serial** D2H→H2D on transfer stream when prefetch is large; does **not** remove 24× host hooks.

---

### P3 — Dense CPU row layout (memory, not primary ITL)

- Allocate rows as `num_req_slots × max_blocks_per_req` from **max concurrent admits** and `max_model_len`, not full inflated `num_blocks` index space.
- **Upfront** table at worker init (no per-admit heap alloc in decode hot path).
- Motivation: **DDR footprint** and logical clarity for P/D node; measure ITL only if pinned-RAM pressure suspected.

---

### P4 — Measurement & validation

- [ ] nsys: transfer stream, `save_event` / `ready_event` chains post-P0
- [ ] Llama-8B + longer output sweeps with fair baseline
- [ ] Log **bytes/layer** D2H and H2D per step (not just `save_submit` wall time)
- [ ] Correctness: pipelined vs baseline outputs with delta path **on**

---

## Anti-patterns (learned)

| Idea | Verdict |
|------|---------|
| Batch all layer D2H at `wait_for_save` | **Breaks** slot reuse correctness |
| `SKIP_DELTA_STAGE` for production | **Diagnostic only** |
| Smaller CPU rows alone | **DDR/layout**; won’t fix ~5 ms/layer if copy + hooks unchanged |
| Parallel save + prefetch on **same** GPU addresses | **Race** unless scratch/main split or D2H completes first |
| End-of-step batch without per-slot ordering | Same as first row |

---

## Related env / metrics

| Env | Purpose |
|-----|---------|
| `VLLM_KVPIPE_SKIP_DELTA_STAGE=1` | Isolate delta D2H cost (incorrect for prod) |
| `VLLM_KVPIPE_WAIT_LOG=0` | Reduce wait_for_layer log noise |

Connector logs: `CpuKvTable: hbm_kv=… cpu_table=… ratio=…`, `save_submit=… avg … ms`.
