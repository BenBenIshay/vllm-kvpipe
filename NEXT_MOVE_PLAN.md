# KV Layer-Wise Pipelining — Next Move Plan

> **Superseded (2026-05-17):** Use [`benchmarks/kv_pipelining_bench/ROADMAP.md`](benchmarks/kv_pipelining_bench/ROADMAP.md) as the maintained plan.

**Date**: 2026-03-18  
**Context**: Resuming after a setup issue. This plan summarizes project state and concrete next steps.

---

## 1. Project state (from SESSION_CONTEXT.md & next_steps.md)

- **Goal**: Layer-wise KV cache pipelining — keep only K+1 layers on GPU, stage the rest in CPU pinned memory; optionally inflate block pool for 6–10x more concurrent decode.
- **Status**: Implementation and v1/v2 evals are **done**. The `_slice_kv_layer` correctness fix (use `decode_max_block_id` across all active requests) was only in **PipelinedDecodeBenchConnector** (fake KV). **PrefillStagingPipelinedConnector** (real prefill + pipelined decode) was still using per-sequence `max_seq_len`, so **real Q&A workloads produced wrong outputs**. As of 2026-03-18 the same fix has been applied to PrefillStagingPipelinedConnector (see §7 below).
- **Results**: K-sweep (best K=6), capacity sweeps, LMCache comparison, post-fix ITL, and P/D eval on Llama-8B were completed on the previous setup. Result dirs (`/tmp/kv_full_eval_v2/`, `/tmp/kv_pd_eval/`, etc.) are **not** present on this machine — either `/tmp` was cleared or this is a different environment.

---

## 2. Setup verification (do this first)

Current environment: system Python 3.12.3 at `/usr/bin/python3` has no `pip` or `venv` (externally-managed / no `python3.12-venv`). Use the automated setup script after installing the system package once.

**One-time system dependency (required):**
```bash
sudo apt install -y python3.12-venv
```

**Automated setup (from repo root):**
```bash
cd /home/ben/vllm
bash scripts/setup_kv_pipelining_env.sh
```
This script: creates `.venv`, installs PyTorch 2.9.1 (CUDA 12.6), installs [requirements/build.txt](requirements/build.txt) into the venv, then installs vLLM in editable mode with **`pip install -e . --no-build-isolation`** (avoids a second PyTorch copy in pip’s isolated build env, which can OOM). It sets **`MAX_JOBS` to `max(1, nproc/2)`** and **`NVCC_THREADS=1`** by default (override in the environment if needed). It adds matplotlib/numpy/aiohttp for benchmarks and runs a smoke test. If the vLLM build fails, install build tools: `sudo apt install -y cmake ninja build-essential` and ensure CUDA toolkit (nvcc) is available.

**Activate and smoke test** (after setup):
   ```bash
   source /home/ben/vllm/.venv/bin/activate
   python -c "
   from vllm.distributed.kv_transfer.kv_connector.v1.prefill_staging_pipelined_connector import PrefillStagingPipelinedConnector
   from vllm.v1.kv_cache_pipelining import LayerwisePipeliningState
   from vllm.config.cache import CacheConfig
   print('KV pipelining imports OK')
   "
   ```

3. **Real-workload sanity check (do this before any heavy eval)**  
   Last time, a real Q&A test showed the pipelined path produced wrong outputs (only fake-KV bench path was correct). The fix has been applied to **PrefillStagingPipelinedConnector**; validate with:
   ```bash
   source .venv/bin/activate
   python benchmarks/kv_pipelining_bench/validate_qa.py \
       --model meta-llama/Llama-3.1-8B-Instruct \
       --k-window 6
   ```
   This runs baseline vs pipelined on real Q&A prompts and compares outputs (exact match + similarity). **If this fails, debug before running P/D evals or benchmarks.**

---

## 3. Immediate next steps (in order)

| # | Task | Command / notes |
|---|------|------------------|
| 1 | **Fix environment** | Install torch + vLLM (and optional deps) in a venv/conda; run smoke test above. |
| 2 | **Real-workload correctness** | Run `validate_qa.py` (real Q&A, baseline vs pipelined). **First priority** — ensures real prefill + pipelined decode is correct before any other evals. |
| 3 | **Run P/D eval on Qwen-14B** | Two-model comparison (Llama-8B was already done in prior session). |
| 4 | **Regenerate publication graphs** | If you have or re-run v2 eval data; optionally extend `generate_graphs.py` to consume P/D result dirs for TTFT/E2E/P99 E2E plots. |

### 2. Real-workload correctness: `validate_qa.py`

Sends real Q&A prompts to baseline and pipelined servers, compares responses (exact match + similarity). Use temperature=0 for deterministic comparison.

```bash
# Llama-8B
python benchmarks/kv_pipelining_bench/validate_qa.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --k-window 6

# Qwen-14B
python benchmarks/kv_pipelining_bench/validate_qa.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --k-window 6
```

If outputs differ, inspect the printed diffs and server logs (including any "MISMATCH" from the connector’s GPU-vs-CPU check).

### 3. Run P/D eval on Qwen-14B

From `next_steps.md` §14:

```bash
python benchmarks/kv_pipelining_bench/eval_pd_decode.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --k-window 6 \
    --concurrency 64 128 256 384 512 \
    --num-prompts 200 \
    --result-dir /tmp/kv_pd_eval/qwen14b
```

Uses **PrefillStagingPipelinedConnector** with `--inflate-kv-pool` (real prefill + inflated capacity). Saves `results_baseline_conc*.json`, `results_pipelined_conc*.json`, and `summary.json`.

### 4. Regenerate / extend graphs

- **Current**: `generate_graphs.py` expects v2 layout under `--result-root` (e.g. `llama8b_ksweep_k6`, `qwen14b_capacity_k6`, etc.) and produces ITL, throughput, SLO, cost, K-sweep heatmap, GPU util, memory/capacity.
- **If v2 data exists elsewhere**: Copy or symlink it under a result root and run:
  ```bash
  python benchmarks/kv_pipelining_bench/generate_graphs.py \
      --result-root /path/to/kv_full_eval_v2 \
      --output-dir /path/to/kv_full_eval_v2/graphs
  ```
- **P/D-specific plots**: To add TTFT / E2E / P99 E2E from P/D eval, extend `generate_graphs.py` to:
  - Accept a P/D result dir (e.g. `/tmp/kv_pd_eval/llama8b` or `qwen14b`).
  - Load `results_baseline_conc*.json` and `results_pipelined_conc*.json`.
  - Plot mean TTFT, mean E2EL, P99 E2EL vs concurrency (baseline vs pipelined).

---

## 4. Paper / narrative

- **Emphasize**: Capacity advantage (TTFT, E2E, P99 E2E, max concurrency), not only ITL.
- **State limitation**: Transfer volume vs concurrency trade-off (see SESSION_CONTEXT.md §6); at very high concurrency, correct full-layer transfer volume can dominate; K≈6 is a good compromise.

---

## 5. Longer term

- **Adaptive K**: Tune K from concurrency (e.g. lower K at low conc, higher at high conc).
- **Upstream**: Clean patches, tests, and PR for vLLM (see original patch `0001-pipeline-kv-cache-first-version.patch` and current `git diff HEAD`).

---

## 6. Key file reference

| Purpose | Path |
|--------|------|
| Session context & results summary | `SESSION_CONTEXT.md` |
| Detailed next steps checklist | `next_steps.md` |
| Design | `docs/design/kv_cache_pipelining.md` |
| Main connector (real prefill + pipelined decode) | `vllm/distributed/kv_transfer/kv_connector/v1/prefill_staging_pipelined_connector.py` |
| Pipelining state (CPU buffers, async transfers) | `vllm/v1/kv_cache_pipelining.py` |
| P/D eval script | `benchmarks/kv_pipelining_bench/eval_pd_decode.py` |
| Graph generator | `benchmarks/kv_pipelining_bench/generate_graphs.py` |
| Config (CLI + cache) | `vllm/config/cache.py`, `vllm/engine/arg_utils.py` |
| Real Q&A correctness | `benchmarks/kv_pipelining_bench/validate_qa.py` |

---

## 7. Fix applied 2026-03-18 (real prefill correctness)

**PrefillStagingPipelinedConnector** was missing the same `_slice_kv_layer` fix that **PipelinedDecodeBenchConnector** had: during decode we must copy `decode_max_block_id + 1` blocks (max block ID across *all* active requests), not `cdiv(max_seq_len, block_size)` (blocks for a single sequence). With multiple requests, block IDs are scattered across the pool; using only the per-sequence count caused wrong/stale data in the K+1 window and wrong outputs on real Q&A.

**Changes in** `prefill_staging_pipelined_connector.py`:

- **Metadata**: Added `decode_max_block_id: int | None` to `PrefillStagingPipelinedConnectorMetadata`.
- **Scheduler**: Added `_active_req_blocks: dict[str, set[int]]`. In `update_state_after_alloc` record all block IDs for the request. In `build_connector_meta` merge `scheduled_cached_reqs.new_block_ids` into `_active_req_blocks`, remove `finished_req_ids`, and set `decode_max_block_id = max(active block IDs)`. In `request_finished` remove the request from `_active_req_blocks`.
- **Worker**: In `start_load_kv` set `self._decode_max_block_id = metadata.decode_max_block_id`. In `_slice_kv_layer` use `decode_max_block_id + 1` when present (decode), else keep `cdiv(max_seq_len, block_size)` (prefill). Clear `_decode_max_block_id` in `wait_for_save`.
