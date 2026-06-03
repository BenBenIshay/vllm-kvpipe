# Layer-KV pipelining: transfer model and overlap (P0)

This note fixes vocabulary for evals and profiling. It matches
`PrefillStagingPipelinedConnector` + K+1 layout in `kv_cache_utils.py`.

## Physical layout

- **`num_layers` logical layers**, **`k_window + 1` physical GPU tensors**.
- Layer `i` uses tensor `i % (k_window + 1)`.
- Full per-request KV history lives in **pinned CPU** after prefill; GPU holds only
  the current window slots in the shared pool (same **`num_blocks`** as baseline,
  smaller per-tensor page size).

## Per decode forward (one batched step, all active requests)

For each logical layer `i` (in order):

1. **`wait_for_layer(i)`** — compute stream waits until this layer’s CPU→GPU
   prefetch finished.
2. **Attention** — reads KV in the physical tensor for layer `i`.
3. **`save_kv_layer(i)`** — **D2H**, block mode by default:
   only physical blocks touched by new token slots (~one block per request per layer).
4. **`plan_prefetch(i)`** — schedule CPU→GPU for upcoming layers that will reuse
   other physical tensors; before each prefetch, **`wait_for_layer_save`** on the
   tensor’s previous occupant.

End of forward: **`wait_for_saves()`** — all GPU→CPU events + **`delta_queue.join()`**.

## H2D: “full layer” is expected, not a bug

When layer `i` is loaded into tensor `t = i % (k+1)`, that tensor may still hold
**another layer’s** GPU data from ~`k_window` steps earlier in the same pass.
The connector **must reload layer `i`’s full active KV** from the CPU buffer for
the physical block rows in use (`kv_block_runs` = merge of active requests’ block
ids).

That is **full sequence history for those blocks**, not a single new block.

**Incremental H2D on prefetch** would imply keeping the rest of the sequence
resident in HBM between layer visits — that defeats K+1 memory savings (you would
need either all layers on GPU or a larger resident window).

**Incremental D2H on save** *is* what we do (block / delta mode): only write new
slots back to CPU.

## Byte budget (estimate)

Use `kv_transfer_budget.py`:

```bash
.venv/bin/python benchmarks/kv_pipelining_bench/kv_transfer_budget.py \
  --model Qwen/Qwen2.5-0.5B-Instruct --input-len 1024 --output-len 256 \
  --concurrency 1
```

Per batched decode step (rough):

| Term | Scale |
|------|--------|
| H2D / layer | `(merged active blocks) × bytes/block` |
| H2D / step | `× num_layers` |
| D2H / step | `× num_layers × ~1 block × bytes/block × concurrency` |

`bytes/block = 2 × block_size × num_kv_heads × head_dim × dtype_bytes`.

The admission predictor’s `k_window × 2 × heads × dim` formula is **not** the
connector’s per-step volume.

## Why mean ITL can exceed the PCIe lower bound (especially K=1)

Theoretical H2D time = `(H2D bytes) / effective_BW`. Measured mean ITL can be
much larger because:

1. **Overlap is incomplete** — H2D per step is often comparable to or larger than
   compute; hiding needs copy stream far ahead of attention.
2. **Sync on the critical path** — `wait_for_layer` (required at layer boundary),
   block-mode host work (`slot_mapping` → CPU). **P2 (2026-05-19):** decode-only
   steps defer `wait_for_saves` / `delta_queue.join`; chained prefetches wait
   prior GPU saves on the **transfer stream**, and H2D waits per-layer CPU delta
   readiness instead of a global end-of-forward barrier.
3. **Prefetch fan-out** — up to `k_window` layers prefetched after each layer wait.
4. **Real prefill path** — staging full layers to CPU on prefill (not repeated each
   decode token, but part of feature E2E).

**K=1 profiling (P1)** should classify time into: attention, memcpy, stream waits.

## What we are *not* claiming

- Same **K** and same mean ITL as baseline (we pay paging).
- Tail-only H2D as the primary optimization (conflicts with tensor reuse).

## What we *are* claiming (later, after optimization)

- Lower **GPU KV footprint** → sustain **higher useful concurrency / throughput**
  under an ITL SLO (admission + crossover sweeps — deferred until P3 tuning done).

## Related tools

| Tool | Purpose |
|------|---------|
| `kv_transfer_budget.py` | Theoretical MiB/step |
| `probe_one_decode.py` + `VLLM_PIPELINE_DEBUG_LOG=1` | Log `kv_block_runs`, save mode |
| `parse_pipeline_debug_log.py` | Summarize debug log |
| `run_profile_k1_overhead.sh` | Nsight Systems trace (P1) |
| `run_k_window_sweep.sh` | P3 `k_window` grid |
