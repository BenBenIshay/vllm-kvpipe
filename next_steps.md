# Next Steps: KV Cache Pipelining

This file captures the recommended next steps for the decode-only KV cache
pipelining feature. It is intended as a working checklist for continued
development and evaluation.

## 1) End-to-end decode validation (H100)  [DONE]

- Run a decode-only workload with:
  - `--enable-kv-pipelining`
  - `--kv-k-window <K>`
  - `--free-original-kv-cache`
  - `--kv-transfer-config '{"kv_connector":"LayerwisePipelinedConnector"}'`
  - `--enable-prefix-caching false`
- Verify:
  - Output correctness (tokens match baseline)  -- correctness_test.py
  - GPU memory reduction vs baseline
  - Decode latency impact and throughput

## 2) Telemetry & accounting

- Log window activity:
  - number of window binds / evictions per step
  - bytes transferred CPU-GPU per step
  - average overlap time (copy stream vs compute stream)

## 3) Guardrails & config validation  [DONE]

- Emit warnings if:
  - pipelining enabled without LayerwisePipelinedConnector
  - prefix caching enabled while pipelining enabled
  - prefill occurs after freeing original KV buffers

## 4) Allocator integration (deeper memory savings)  [DONE]

With `--enable-kv-pipelining --free-original-kv-cache`, only `kv_k_window`
layers reside on GPU at any moment during decode.  The allocator now inflates
the block pool by `num_layers / kv_k_window` so the scheduler can admit more
concurrent requests.

Implementation details:
- `get_kv_cache_config_from_groups()` computes both `num_blocks_base` (physical
  per-layer tensor size) and `num_blocks_pipelined` (inflated pool size).
- `KVCacheConfig.num_blocks` = inflated count, `.num_physical_blocks` = base.
- Per-layer KV cache tensors are allocated at the base size (fit in GPU memory).
- The connector creates window buffers at the inflated size so block IDs up to
  `num_blocks_pipelined - 1` are valid during decode.
- CPU pinned buffers are pre-allocated at the inflated size via
  `ensure_cpu_buffer()` to avoid costly re-creation on the second step.

Constraints:
- Prefill block IDs must stay < num_physical_blocks.  This is naturally
  satisfied because the free list allocates low IDs first and physical pools
  are typically 10k+ blocks (plenty for concurrent prefills).
- k_window must be << num_layers to see benefit (e.g., 4 for 32-layer Llama).
- CPU memory: each layer needs `num_blocks_pipelined * page_size` pinned bytes.

## 5) Expand tests

- Add a decode-only integration test (smoke):
  - ensure no sync in hot path
  - assert connector metadata is built in decode
- Add regression coverage for `--free-original-kv-cache`.
- Add allocator inflation unit tests.

## 6) Run capacity_test.py on H100 with Llama-3.1-8B

```
python benchmarks/kv_pipelining/capacity_test.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --k-window 4 \
    --input-len 512 \
    --output-len 512 \
    --start-concurrency 4 \
    --end-concurrency 128 \
    --step 4 \
    --modes baseline pipelined_free
```

Input/output lengths match realistic workloads (ShareGPT-like).  During
prefill, block IDs stay within num_physical_blocks because the free list
allocates low IDs first and physical pools are typically 10k+ blocks.

Expected: `pipelined_free` supports ~8x more concurrent requests than baseline
(32 layers / k_window=4 = 8x block inflation).

## 7) Future: dynamic block pool expansion

Current approach inflates the pool at init time.  A more robust approach would
dynamically expand the pool after the first decode step frees the original
tensors.  This would allow prefill to use high block IDs too (no
num_physical_blocks constraint).