# Stage-to-CPU optimization — next steps

Focus: minimize **`save_kv_layer` → `stage_delta_to_cpu`** cost on the decode hot path
(PipelinedDecodeBench / LayerwisePipelinedConnector + `CpuKvTable`).

**Current signal (Qwen 0.5B, K=4, conc=1):**

- `save_submit` ≈ **4.9–5.9 ms per call** × **24 layers** ≈ **118 ms**/token if serial on host
- Measured mean ITL ≈ **58–77 ms** (some overlap with compute / transfer)
- `VLLM_KVPIPE_SKIP_DELTA_STAGE=1` → **~7.4 ms ITL** (~60 tok/s) → **delta path ≈ 50–70 ms**/token of the gap

**What `save_submit` measures:** `perf_counter` around **`stage_delta_to_cpu` only** (not `_prefetch_after_save`). See `pipelined_decode_bench_connector.py` `save_kv_layer`.

---

## What happens today (decode, table seeded)

1. Attention writes new token via **`slot_mapping`** into GPU rolling tensor.
2. `CpuKvTable.stage_delta` **fast path** (`n==1`):
   - Record event on **compute stream**; transfer stream waits.
   - **D2H:** full **page** `kv_gpu[:, phys, :]` → `cpu_buf[:, tbl_block, :]` (~`block_size` slots, often 16× larger than one token).
   - `table_block_gpu[phys]` lookup + tensor slice on **inflated** GPU tensor (~569k blocks in row geometry).
3. `_prefetch_after_save(i)`: **H2D** active physical blocks for layer `i+K` into **same** GPU slot, after `wait_event(save_i)` on transfer stream.

Bottleneck classes:

| Class | Symptom | Skip-delta test |
|-------|---------|-----------------|
| A. Per-layer **host** overhead | 24× hooks, events, Python | Still ~7 ms ITL (hooks + prefetch remain) |
| B. **D2H bytes** / page copy | PCIe + memcpy | Partially removed if skip |
| C. **Transfer stream ordering** | save then prefetch serialized on slot | Phase H (scratch) |
| D. **Prefetch volume** | Many blocks H2D per new binding | Independent of save size |

---

## Optimization roadmap (ordered)

### Step 1 — Copy one token, not one page (highest ROI, lowest risk)

**Change:** In `CpuKvTable.stage_delta` fast path, D2H only  
`kv_gpu[:, phys, off]` → `cpu_buf[:, tbl_block, off]`  
(not `[:, phys, :]` entire page).

**Bytes (Qwen 0.5B):** ~512 B/layer vs ~8 KiB/page → ~16× less per layer, ~40 KiB/step vs ~192 KiB.

**Correctness:** Same logical slot; CPU table still gets the new token.

**Measure:**

- `save_submit` avg ms/call
- mean ITL / tok/s vs baseline
- Optional: log `d2h_bytes_per_step` in connector

**Risk:** Low.

---

### Step 2 — Instrument bytes and transfer-stream timeline

Add counters (debug build or step 100 log):

- `d2h_bytes_layer`, `h2d_prefetch_bytes_layer`
- Whether fast path hit (`n_tokens==1`)
- Transfer stream: time from `save_event` recorded to prefetch `ready_event`

**Tool:** `nsys profile` on 1 request, 32 decode steps — confirm compute vs transfer overlap.

**Goal:** Verify Step 1 dominates; decide if Step 4 (prefetch scope) is next.

---

### Step 3 — Reduce per-layer host overhead (implemented May 2026)

1. **One `wait_stream(compute)` per decode step** (`sync_decode_transfer_stream_once`), not 24×.
2. **H2D on `_prefetch_stream`**, D2H on `_stream` — overlap after delta scratch snapshot.
3. **Bootstrap prefetch** of first `K` layers at `wait_for_save` + skip if `ready_event` already set in `start_load_kv`.
4. **Lean `wait_for_layer_load`** — direct `wait_event` when profiling off.
5. **`VLLM_KVPIPE_WAIT_LOG=0`** default; **`VLLM_KVPIPE_PROFILE_SAVE=1`** optional.

**Remaining gap vs ~3.6 ms baseline:** ~24× Python hook invocations (`maybe_transfer_kv_layer`) + ~20 in-forward prefetches; `SKIP_DELTA` floor ~4.8 ms. Closing to baseline likely needs **step-level transfer API** (fewer hooks), not smaller copies alone.

**Risk:** Low for items 1–5.

---

### Step 4 — Narrow H2D prefetch scope

**Status:** Mostly already true for **conc=1** — see note below. Revisit for **multi-request** batches.

**Today:** `decode_physical_block_ids` = all physical blocks in `_active_req_blocks` (scheduler-allocated), not the full inflated pool (~569k). For one request, that set grows with sequence length (~`ceil(seq_len/block_size)` blocks).

**Not the same as:** trimming by `decode_max_seq_len` if the scheduler pre-allocates many blocks at admit time. For incremental block growth, active set ≈ needed blocks.

**Multi-conc:** union of all requests' allocated blocks — possible future trim per request.

---

### Step 5 — Phase H: transfer overlap

**5a — Prefetch scratch (legacy, off):** `VLLM_KVPIPE_SCRATCH_PREFETCH=1` — H2D to block scratch + merge to main. **~6.3 ms** steady ITL (regression).

**5b — Delta D2H scratch (default on):** `VLLM_KVPIPE_DELTA_SCRATCH=1`

- Per rolling slot: **one-token** HBM scratch (~KiB total for K=4).
- On save: `main[phys,off] → scratch` → D2H(scratch→DDR); `snap_event` after snapshot.
- Prefetch on **`_prefetch_stream`**: waits **`snap_event` only** (not full `save_event`), then H2D → **main** — D2H and H2D can overlap after snapshot.
- **Bench (Qwen 0.5B, conc=1):** steady ITL ~**4.7–4.8 ms** — matches path without delta scratch; **no ~1 ms drop** on this workload (tiny PCIe + 3 blocks; overlap hides little).

Disable: `VLLM_KVPIPE_DELTA_SCRATCH=0`.

**Profiling v2 (`VLLM_KVPIPE_PROFILE=1`):** per-forward (not cumulative) lines in server log:

- `KVPIPE_FWD` — **this token only**: `gpu_stall_sum` (sum of 24 layer `wait_event` stalls on compute stream), p50/p99/max, `worst_layers=[layer:ms,...]`, host `save_host` / `start_load` / `rolling_pf`.
- `KVPIPE_STEADY` — rolling window (`VLLM_KVPIPE_PROFILE_WINDOW`, default 32) of per-forward `gpu_stall_sum` mean/p99/max. **Do not equate stall_sum with ITL** (most layers ~0; tails + attn dominate).

Env: `VLLM_KVPIPE_PROFILE_EVERY=10` (log every N decode steps). For sweeps use `--no-inflate-kv-pool` to avoid multi-minute CPU table pin at startup.

**5c — Overlap-aware block prefetch:** still active when delta scratch off (`delta_phys` split).

---

### Step 6 — PrefillStaging + production connectors

Not stage-to-CPU on bench only — align `PrefillStagingPipelinedConnector` so decode uses same table + direct K path (plan P1).

---

## Suggested experiment matrix

| Run | Config | Pass criteria |
|-----|--------|---------------|
| Baseline | `DecodeBenchConnector` | ~100 tok/s Qwen conc=1 reference |
| A0 | Current CpuKvTable | Documented ~6–7 tok/s |
| A1 | Step 1 one-token D2H | `save_submit` ↓; ITL ↓ materially |
| A2 | A1 + Step 4 narrow prefetch | ITL ↓ further; correctness OK |
| A3 | A2 + Step 3 event pool | `save_submit` ↓ toward &lt;1 ms/call target |
| B0 | `SKIP_DELTA_STAGE=1` on A2 | Upper bound without D2H |

**Target (aspirational):** steady ITL within **2× baseline** (~7 ms) with delta **on** and correctness — may require Steps 1+3+4, not layout-only changes.

---

## Discussion checklist (before coding)

1. **Page vs token copy** — OK to stage only `slot_mapping` offsets (may include spec-decode multi-slot later)?
2. **Prefetch minimal blocks** — use merged `decode_block_runs` capped by `ceil(max_seq_len/block_size)`?
3. **Scratch region** — worth HBM cost on H100 decode-only node vs serial transfer stream?
4. **Batched enqueue** — one worker thread issuing ordered copy list per step vs 24 Python calls?

---

## Code touchpoints

| File | Change |
|------|--------|
| `vllm/v1/cpu_kv_table.py` | `stage_delta` token vs page copy |
| `vllm/v1/kv_cache_pipelining.py` | `_stage_delta_via_table`, optional batched enqueue |
| `pipelined_decode_bench_connector.py` | metrics, prefetch block scope |
| `layerwise_pipelined_connector.py` | same paths as bench |
