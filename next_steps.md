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

## 7) Decode-only P/D disaggregation mode  [DONE]

With Prefill/Decode disaggregation, the decode node never performs local
prefill — KV caches arrive pre-computed from a remote prefill node.  This
simplifies and amplifies the pipelining benefit:

- **No `num_physical_blocks` constraint**: since there is no local prefill,
  any block ID is safe.  The allocator uses `num_physical_blocks = None`.
- **Minimal per-layer GPU tensors**: only tiny placeholders (32 blocks) are
  allocated for model-init compatibility.  The full GPU KV memory budget goes
  to the `k_window` window buffers.
- **Maximum inflation**: available memory is divided only by the pipelined
  page size (`total_page_size * k_window / num_layers`), yielding the
  largest possible block pool.

Usage:
```
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B \
    --decode-only-pipelining \
    --kv-k-window 4 \
    --kv-transfer-config '{"kv_connector":"LayerwisePipelinedConnector","kv_role":"kv_both"}' \
    --enable-prefix-caching false
```

`--decode-only-pipelining` auto-enables `--enable-kv-pipelining` and
`--free-original-kv-cache`.  The connector never restores original (tiny)
per-layer buffers and always uses inflated window buffers.

Benchmark:
```
python benchmarks/kv_pipelining/capacity_test.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --k-window 4 \
    --input-len 512 --output-len 512 \
    --modes baseline pipelined_free decode_only
```

## 8) Self-contained decode-only benchmark (no P/D stack)  [DONE]

For academic performance evaluation, we skip the full P/D transfer integration
and instead simulate the decode-only scenario locally:

1. **PipelinedDecodeBenchConnector**: combines `DecodeBenchConnector` (dummy KV
   fill) with `LayerwisePipelinedConnector` (CPU→GPU pipelining).
   - Scheduler side claims all prompt tokens as externally available.
   - Worker side fills KV on GPU with dummy values, stages everything to CPU
     pinned memory, frees GPU tensors, then switches to layer-wise pipelining.
   - All subsequent decode steps use pure CPU→GPU pipelining through window
     buffers — exactly the performance path a real P/D decode node would take.

2. **decode_only_bench.py** (`benchmarks/kv_pipelining_bench/`): end-to-end
   benchmark comparing baseline (all KV on GPU) vs pipelined decode-only.
   Measures TTFT, ITL (mean + p99), throughput, and max concurrency.

Files:
- `vllm/distributed/kv_transfer/kv_connector/v1/pipelined_decode_bench_connector.py`
- `benchmarks/kv_pipelining_bench/decode_only_bench.py`
- `tests/v1/kv_connector/unit/test_pipelined_decode_bench_connector.py`

Usage (baseline vs pipelined):
```
python benchmarks/kv_pipelining_bench/decode_only_bench.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --k-window 4 \
    --input-len 512 \
    --output-len 128 \
    --concurrency 1 4 16 64 \
    --modes baseline pipelined
```

Pipelined mode server flags:
```
--decode-only-pipelining \
--kv-k-window 4 \
--kv-transfer-config '{"kv_connector":"PipelinedDecodeBenchConnector","kv_role":"kv_both"}' \
--enable-prefix-caching false
```

Expected results:
- **Latency**: small ITL overhead from CPU→GPU copy stream (~5-15% on H100 PCIe).
- **Concurrency**: ~num_layers/k_window improvement (e.g., 8x for 32-layer
  Llama with k_window=4) since only k_window layers of KV reside on GPU.

## 9) Real-prefill pipelined decode (single-GPU paper benchmark)  [DONE]

For academically rigorous evaluation on a single H100, we need real model
prefill (not dummy KV fill) followed by pipelined decode.  This enables
correctness validation and realistic latency measurement with real datasets
(LongBench, ShareGPT, vLLM random).

### PrefillStagingPipelinedConnector

A new connector that lets the model run **real prefill** computation, then
transparently transitions to layer-wise CPU→GPU pipelined decode:

  - **Scheduler side**: returns 0 from `get_num_new_matched_tokens` — the
    scheduler treats the request as a normal prefill (no external token
    claims).
  - **Worker side — prefill step**: the model computes real attention.
    After each layer's forward pass, `save_kv_layer` asynchronously stages
    the real KV output from GPU to CPU pinned memory via a CUDA copy-stream.
  - **Worker side — decode step**: `start_load_kv` binds K-window GPU
    buffers and prefetches from CPU.  `wait_for_layer_load` blocks until
    the prefetch completes and triggers ahead-of-time prefetch for the
    next layers.  `save_kv_layer` stages the decode-token delta back to
    CPU.
  - **Mixed steps** (prefill + decode in the same batch): the connector
    keeps original GPU KV tensors (no window binding) and stages full
    layers to CPU.  Pipelining activates only on pure decode steps.

Files:
- `vllm/distributed/kv_transfer/kv_connector/v1/prefill_staging_pipelined_connector.py`
- `tests/v1/kv_connector/unit/test_prefill_staging_pipelined_connector.py`

Usage (real prefill + pipelined decode):
```
vllm serve <model> \
    --enable-kv-pipelining \
    --kv-k-window 4 \
    --free-original-kv-cache \
    --kv-transfer-config '{"kv_connector":"PrefillStagingPipelinedConnector","kv_role":"kv_both"}' \
    --enable-prefix-caching false
```

Benchmark (baseline vs real_prefill):
```
python benchmarks/kv_pipelining_bench/decode_only_bench.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --k-window 4 \
    --input-len 512 --output-len 128 \
    --concurrency 1 4 16 64 \
    --modes baseline real_prefill
```

Three-mode comparison (dummy-fill + real-prefill + baseline):
```
python benchmarks/kv_pipelining_bench/decode_only_bench.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --k-window 4 \
    --input-len 512 --output-len 128 \
    --concurrency 1 4 16 64 \
    --modes baseline pipelined real_prefill
```

Expected results:
- **Correctness**: real_prefill output should match baseline (identical tokens
  at temperature=0) since the attention computation is the same.
- **TTFT**: real_prefill has slightly higher TTFT than baseline due to the
  async GPU→CPU staging during prefill (~5-10% overhead).
- **ITL**: small overhead from CPU→GPU copy stream (~5-15% on H100 PCIe).
- **Capacity**: with `--free-original-kv-cache`, original per-layer GPU tensors
  are freed during decode, reclaiming GPU memory.

### Paper evaluation plan

1. **Correctness test**: run a small batch at temperature=0 with both
   `baseline` and `real_prefill`, compare output tokens.
2. **Latency benchmark**: single-request streaming to measure TTFT and ITL
   with real datasets (LongBench, ShareGPT, random).
3. **Throughput benchmark**: sweep concurrency levels (1, 4, 16, 64, 128)
   to measure throughput and find the max concurrency before OOM.
4. **Capacity test**: use `pipelined` mode (dummy fill + block inflation via
   `--decode-only-pipelining`) to measure the maximum number of concurrent
   decode requests (~num_layers/k_window improvement).

## 10) Comprehensive Evaluation Benchmark  [DONE]

A single evaluation script that compares baseline (DecodeBenchConnector,
all KV on GPU) vs pipelined (PrefillStagingPipelinedConnector, real prefill
+ CPU→GPU layer-wise decode) across four workloads using `vllm bench serve`
as the metrics-collection client.

### Workloads

1. **random** — synthetic prompts of fixed token length (vLLM built-in
   random dataset).
2. **sharegpt** — real multi-turn QA conversations from ShareGPT.
3. **longbench** — long-context QA from THUDM/LongBench (auto-download
   from HuggingFace, converted to JSONL for the custom dataset loader).
4. **multi_round** — emulated multi-turn chat with growing context
   (sequential turns with increasing input length per "session").

### Metrics (decode-focused)

- TTFT, TPOT, ITL (mean + p99), E2EL — all in milliseconds
- Output throughput (tok/s)
- Request throughput (req/s)
- Max concurrency before failure

### Correctness validation

`--correctness` sends identical prompts at temperature=0 to both modes
and compares outputs token-by-token.

### Files

- `benchmarks/kv_pipelining_bench/eval_kv_pipelining.py`

### Usage

```bash
# Quick random-only test:
python benchmarks/kv_pipelining_bench/eval_kv_pipelining.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --workloads random \
    --modes baseline pipelined \
    --k-window 4

# Full paper evaluation (all workloads + correctness):
python benchmarks/kv_pipelining_bench/eval_kv_pipelining.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --workloads random sharegpt longbench multi_round \
    --modes baseline pipelined \
    --k-window 4 \
    --concurrency 1 4 16 64 \
    --correctness

# ShareGPT with local dataset:
python benchmarks/kv_pipelining_bench/eval_kv_pipelining.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --workloads sharegpt \
    --sharegpt-path /data/ShareGPT_V3_unfiltered.json \
    --modes baseline pipelined
```

### Output

Results saved to `--result-dir` (default `/tmp/kv_pipelining_eval/`):
```
<result-dir>/
  <workload>/
    results_<mode>_conc<N>.json   — raw vllm bench serve output
    server_<mode>.log             — server stdout/stderr
  correctness/
    correctness_results.json      — token-level comparison
  summary.json                    — aggregated cross-mode comparison
```

## 11) Future: dynamic block pool expansion

Current approach inflates the pool at init time.  A more robust approach would
dynamically expand the pool after the first decode step frees the original
tensors.  This would allow prefill to use high block IDs too (no
num_physical_blocks constraint).

## 12) Future: P/D transfer integration

Wire decode-only pipelining to vLLM's P2P NCCL / NiXL / LMCache connectors
so KV arriving from a real prefill node flows directly into CPU pinned
memory → layer-wise pipeline → GPU window buffers without intermediate
GPU allocation.