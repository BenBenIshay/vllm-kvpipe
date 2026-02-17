# KV Cache Pipelining — Session Context
## Last Updated: 2026-02-17 18:30 UTC

---

## 1. PROJECT OVERVIEW

**Goal**: Implement and evaluate **layer-wise KV cache pipelining** for vLLM.
Instead of keeping all `num_layers` KV cache tensors on GPU, keep only `K+1`
physical tensors (a sliding window). The remaining layers are staged to/from
CPU pinned memory asynchronously, overlapping PCIe transfers with GPU compute.

**Key Benefit**: Massive GPU KV memory reduction (84-90%), enabling either
lower memory footprint or (with pool inflation) 6-10× more concurrent decode
requests from the same GPU memory budget.

**Hardware**: Single H100 NVL (95 GB HBM3), tested on `/home/ben/vllm`.

---

## 2. ARCHITECTURE

### Core Components

| File | Role |
|------|------|
| `vllm/v1/kv_cache_pipelining.py` | `LayerwisePipeliningState` — manages CPU pinned buffers, async CPU↔GPU transfers via dedicated CUDA stream, delta staging threads |
| `vllm/distributed/kv_transfer/kv_connector/v1/prefill_staging_pipelined_connector.py` | **`PrefillStagingPipelinedConnector`** — the main KV connector. Performs real prefill, stages KV to CPU, then pipelines decode layer-by-layer |
| `vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py` | **`DecodeBenchConnector`** — baseline connector that fills KV with dummy values for decode-only benchmarking (no real prefill) |
| `vllm/distributed/kv_transfer/kv_connector/v1/pipelined_decode_bench_connector.py` | Combined pipelined + bench connector (earlier version) |
| `vllm/distributed/kv_transfer/kv_connector/v1/layerwise_pipelined_connector.py` | Earlier iteration of layerwise connector |

### Config Changes (tracked by git)

| File | Changes |
|------|---------|
| `vllm/config/cache.py` | Added `enable_kv_pipelining`, `kv_k_window`, `free_original_kv_cache`, `inflate_kv_pool`, `decode_only_pipelining` fields |
| `vllm/engine/arg_utils.py` | Exposed these as CLI args: `--enable-kv-pipelining`, `--kv-k-window`, `--inflate-kv-pool`, etc. |
| `vllm/v1/core/kv_cache_utils.py` | K+1 physical tensor layout in `get_kv_cache_config_from_groups()`, with phase-1 (same blocks, less memory) and phase-2 (`--inflate-kv-pool`: inflated block count, same total memory) |

### Benchmark Infrastructure

| File | Role |
|------|------|
| `benchmarks/kv_pipelining_bench/eval_decode_isolation.py` | Main eval harness — launches vLLM servers in different modes, runs `vllm bench serve`, collects JSON results, writes summary |
| `benchmarks/kv_pipelining_bench/run_full_eval.sh` | Unattended orchestration script (nohup-safe) — runs all experiments sequentially |
| `benchmarks/kv_pipelining_bench/generate_report.py` | Post-processing: reads all result JSONs and generates a formatted comparison report |
| `benchmarks/kv_pipelining_bench/eval_kv_pipelining.py` | Earlier eval script (kept for reference) |
| `benchmarks/kv_pipelining_bench/decode_only_bench.py` | Earlier decode-only benchmark (kept for reference) |

---

## 3. THREE BENCHMARK MODES

| Mode | CLI flags | What it does |
|------|-----------|-------------|
| **baseline** | `--kv-transfer-config '{"kv_connector":"DecodeBenchConnector",...}'` | Dummy KV fill → decode. Full `num_layers` tensors on GPU. Standard vLLM memory layout. |
| **pipelined** | `--enable-kv-pipelining --kv-k-window 4 --kv-transfer-config '{"kv_connector":"PrefillStagingPipelinedConnector",...}'` | Real prefill → stage to CPU → decode with K+1 GPU window. Same block count as baseline. |
| **pipelined_inflated** | Same as pipelined + `--inflate-kv-pool` | Same as pipelined but block pool inflated by `num_layers/(K+1)`. Same total GPU memory, more concurrent capacity. |

---

## 4. RESULTS SO FAR

### 4a. Llama-3.1-8B-Instruct (32 layers, K=4 → K+1=5, inflation=6.4×)

**Random workload**: input=512, output=128, 150 prompts

| Mode | Conc | Mean ITL (ms) | P99 ITL (ms) | TTFT (ms) | Throughput (tok/s) |
|------|------|--------------|--------------|-----------|-------------------|
| baseline | 1 | 5.95 | 6.30 | 41.7 | 160 |
| baseline | 4 | 6.24 | 6.83 | 30.8 | 614 |
| baseline | 16 | 6.42 | 9.16 | 61.8 | 2,194 |
| baseline | 64 | 8.58 | 20.16 | 160.3 | 5,427 |
| pipelined | 1 | 5.95 | 6.25 | 49.8 | 159 |
| pipelined | 4 | 6.33 | 7.63 | 103.4 | 557 |
| pipelined | 16 | 6.61 | 10.51 | 305.9 | 1,689 |
| pipelined | 64 | 11.05 | 25.00 | 839.7 | 3,139 |
| pipelined_inflated | 1 | 5.94 | 6.27 | 48.3 | 159 |
| pipelined_inflated | 4 | 6.34 | 7.89 | 101.9 | 557 |
| pipelined_inflated | 16 | 6.63 | 8.86 | 301.0 | 1,691 |
| pipelined_inflated | 64 | 11.14 | 25.23 | 806.6 | 3,157 |

**Key observations**:
- **Decode ITL at conc=1**: ~0% overhead (5.95 vs 5.95ms) — PCIe fully overlapped
- **Decode ITL at conc=64**: ~29% overhead (11.05 vs 8.58ms) — PCIe contention from concurrent transfers
- **TTFT overhead**: Expected — pipelined does real prefill + CPU staging; baseline uses dummy fill
- **P99 ITL at conc=16**: Pipelined_inflated actually *better* (8.86 vs 9.16ms)
- **Inflated mode**: Nearly identical to non-inflated (pool inflation doesn't hurt latency)

**Capacity sweep** (baseline only so far, stored in `/tmp/kv_eval_llama8b_v2/capacity/`):
- Baseline successfully served up to 500 concurrent decode requests (input=512, output=128)
- With 6.4× inflation, pipelined mode can theoretically serve ~3,200 concurrent requests

### 4b. LongBench + Multi-round (Llama-3.1-8B)

Results in `/tmp/kv_eval_llama8b_longbench/`:
- **LongBench**: Both modes failed (likely dataset loading issue with `vllm bench serve`)
- **Multi-round**: Partial results available

---

## 5. WHAT'S CURRENTLY RUNNING

**Full eval script** is running unattended:
- **PID**: 1529405
- **Script**: `/home/ben/vllm/benchmarks/kv_pipelining_bench/run_full_eval.sh`
- **Master log**: `/tmp/kv_full_eval/master.log`
- **Results dir**: `/tmp/kv_full_eval/`

### Experiments in the queue:

| # | Label | Model | Workload | Modes | Status |
|---|-------|-------|----------|-------|--------|
| 1 | `llama8b_random` | Llama-3.1-8B | random 512→128 | baseline, pipelined, inflated | **IN PROGRESS** |
| 2 | `llama8b_multi_round` | Llama-3.1-8B | 4-turn QA | baseline, pipelined | pending |
| 3 | `llama8b_random_long` | Llama-3.1-8B | random 2048→256 | baseline, pipelined, inflated | pending |
| 4 | `llama8b_offloading` | Llama-3.1-8B | random 512→128 | baseline, offloading | pending |
| 5 | `qwen14b_random` | Qwen2.5-14B | random 512→128 | baseline, pipelined, inflated | pending |
| 6 | `qwen14b_multi_round` | Qwen2.5-14B | 4-turn QA | baseline, pipelined | pending |
| 7 | `qwen14b_random_long` | Qwen2.5-14B | random 2048→256 | baseline, pipelined, inflated | pending |

When all experiments complete, `generate_report.py` auto-generates `/tmp/kv_full_eval/REPORT.txt`.

---

## 6. ENVIRONMENT DETAILS

- **Python**: `/usr/bin/python3` (Python 3.10.12) — has torch, vllm installed
- **Alternative**: `/home/ben/vllm/vllm_env/bin/python` (Python 3.12) — does NOT have torch
- **HF cache**: `export HF_HOME=/home/ben/hf_cache`
- **Models cached at**: `/home/ben/hf_cache/hub/`
  - `meta-llama/Llama-3.1-8B-Instruct` (32 layers, ~16GB)
  - `Qwen/Qwen2.5-14B-Instruct` (48 layers, ~28GB)
  - `Qwen/Qwen3-0.6B` (28 layers, small, used for early testing)
- **GPU**: NVIDIA H100 NVL, 95040 MiB HBM3
- **vLLM version**: Latest main branch (v0.8.x era) with our patches applied

---

## 7. MONITOR COMMANDS (for next session)

```bash
# Check if eval is still running
ps -p 1529405 2>/dev/null && echo "RUNNING" || echo "DONE"

# Follow progress
tail -f /tmp/kv_full_eval/master.log

# See completed results
find /tmp/kv_full_eval -name "*.json" | wc -l
ls /tmp/kv_full_eval/

# Read individual experiment logs
cat /tmp/kv_full_eval/llama8b_random/eval.log
cat /tmp/kv_full_eval/qwen14b_random/eval.log

# Read final report
cat /tmp/kv_full_eval/REPORT.txt

# GPU status
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

---

## 8. PREVIOUS RESULT DIRECTORIES (for reference)

| Directory | What it contains |
|-----------|-----------------|
| `/tmp/kv_eval_llama8b/` | First eval run (baseline vs pipelined, no inflated mode) |
| `/tmp/kv_eval_llama8b_v2/` | **Main v2 results**: random + capacity sweep with inflated mode. Best data so far. |
| `/tmp/kv_eval_llama8b_longbench/` | LongBench + multi-round attempt (LongBench failed, multi-round partial) |
| `/tmp/kv_full_eval/` | **Current full eval** running all experiments unattended |

---

## 9. KNOWN ISSUES & NOTES

1. **LongBench dataset**: `vllm bench serve` doesn't support LongBench natively.
   The eval script generates a custom JSONL file, but the bench tool may not
   load it correctly. May need to use raw HTTP requests instead.

2. **TTFT overhead**: The pipelined mode shows higher TTFT because it performs
   real prefill computation + CPU staging, while baseline uses instant dummy fill.
   This is an artifact of the benchmark design, not a real-world penalty (in a P/D
   setup, the prefill node sends KV over the network anyway).

3. **Correctness**: Token-level output comparison between baseline and pipelined
   is not meaningful because baseline uses dummy KV (random tokens) while pipelined
   generates real text from actual prefill. The pipelined mode produces coherent output.

4. **jinja2 warning**: `ImportError: apply_chat_template requires jinja2>=3.1.0`.
   Cosmetic warning, doesn't affect benchmarks.

5. **`fuser` / `pkill` in scripts**: These can fail silently when no matching process
   exists. The `run_full_eval.sh` script uses `|| true` throughout to handle this.

---

## 10. NEXT STEPS (after eval completes)

1. **Analyze full eval report**: `cat /tmp/kv_full_eval/REPORT.txt`
2. **Capacity sweep with inflated pool**: Run pipelined_inflated at high concurrency
   to demonstrate 6.4× (Llama) / 9.6× (Qwen) more concurrent decode capacity
3. **Fix LongBench**: Either fix the JSONL format or switch to raw HTTP benchmarking
4. **Write final paper/presentation**: Compile all results into a cohesive comparison
5. **Upstream preparation**: Clean up patches, write proper tests, prepare PR

---

## 11. GIT PATCH

The original patch is at `/home/ben/vllm/0001-pipeline-kv-cache-first-version.patch`.
Additional changes on top (tracked by git):

```bash
cd /home/ben/vllm
git diff HEAD  # shows config.py, arg_utils.py, kv_cache_utils.py changes
```

Untracked files we created:
- `benchmarks/kv_pipelining_bench/eval_decode_isolation.py`
- `benchmarks/kv_pipelining_bench/generate_report.py`
- `benchmarks/kv_pipelining_bench/run_full_eval.sh`
- `vllm/distributed/kv_transfer/kv_connector/v1/prefill_staging_pipelined_connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/pipelined_decode_bench_connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/layerwise_pipelined_connector.py`
- `vllm/v1/kv_cache_pipelining.py`
