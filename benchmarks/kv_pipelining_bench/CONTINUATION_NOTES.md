# KV pipelining — session notes & continuation guide

This document records work done while validating **PrefillStagingPipelinedConnector** correctness, scaling concurrency, and running **P/D decode evaluation**. Use it to resume debugging **K+1 shared-tensor decode** issues.

---

## 1. Code changes (repository)

### 1.1 `vllm/v1/kv_cache_pipelining.py` — `stage_delta_to_cpu` + `save_event`

**Problem:** `wait_for_layer_save()` in the pipelined connector only waits on `state.save_event`. That event was set by **`stage_to_cpu`**, not by **`stage_delta_to_cpu`** (steady decode path). For shared physical GPU tensors, `wait_for_layer_save(prev_occupant)` was often a **no-op** during decode, allowing **prefetch** to race with **in-flight delta staging**.

**Change:** After the async copies on the pipelining stream, record **`state.save_event`** **before** `ready_event` (background scatter still uses `ready_event` afterward).

**Status:** Necessary for ordering; **not sufficient** for full correctness — **`validate_qa` with `--k-window 6` still produced garbage** (~0.04 avg similarity vs baseline). Further root-cause work is required on the **K+1 shared** path.

### 1.2 `vllm/distributed/kv_transfer/kv_connector/v1/prefill_staging_pipelined_connector.py` (from earlier thread)

- **`_physical_kv_is_shared`** from unique `data_ptr()` counts.
- When **not** shared (`kv_k_window >= num_layers`): **skip** decode prefetch / delta path that would overwrite per-layer GPU KV.
- Layer list alignment via **`kv_cache_groups[*].layer_names`** where applicable.

### 1.3 `benchmarks/kv_pipelining_bench/validate_qa.py` (from earlier thread)

- **`--enforce-eager` / `--no-enforce-eager`** (default: eager) for both servers.

### 1.4 `benchmarks/kv_pipelining_bench/eval_pd_decode.py` (this session)

- **`--enforce-eager` / `--no-enforce-eager`** (default: eager), appended to `vllm` CLI like `validate_qa.py`.
- **`--inflate-kv-pool` / `--no-inflate-kv-pool`** (default: inflate). When **false**, the script does **not** pass `--inflate-kv-pool` to the pipelined server (aligns with default `validate_qa` behavior).

---

## 2. Test matrix & outcomes

**Model:** `Qwen/Qwen2.5-0.5B-Instruct` (24 transformer layers in KV group).

**Common flags:** `--enforce-eager`, `--gpu-memory-utilization 0.70` (see §4).

### 2.1 `validate_qa.py`

| Configuration | Outcome | Notes |
|---------------|---------|--------|
| `--k-window 6` (K+1 sharing **on**) | **FAIL** | Pipelined text **nonsense** vs baseline; ~**0.04** avg similarity, **0** exact matches. Log: `shared_across_layers=True`, **7** physical tensors. |
| `--k-window 24` (`k_window >= num_layers` → **no** K+1 inflation in allocator) | **PASS** (Phase 1 + conc=2) | **30/30** exact, similarity **1.0** (Phase 1); Phase 2 conc=2 **PASS** overall. |
| `--k-window 24`, `--concurrency 1 4 8` | **Partial** | Phase 1 **PASS**. Phase 2 **fails overall verdict** because **`min_similarity` &lt; 0.50** while **avg** stays **~0.91–0.95**. Worst prompts often differ in **wording** (e.g. `context_02`), not K=6-style garbage. |

**Verdict logic** (in `validate_qa.py`): `similarity_threshold_pass()` requires **both** `avg_similarity > 0.90` **and** `min_similarity > 0.50`.

### 2.2 `eval_pd_decode.py`

| Configuration | Outcome |
|----------------|---------|
| Default **`--inflate-kv-pool`**, K=6, bench load | **EngineCore died** during/after warmup; bench saw **connection refused**. Very large **inflated** block pool + **PrefillStaging** CPU staging suspected (**OOM / resource** or secondary failure). |
| **`--no-inflate-kv-pool`**, `--enforce-eager`, K=6, conc 16 & 32, 24 prompts | **Completed** baseline + pipelined; **24/24** successes each. Results under a directory like `results/pd_eval_noinflate_*`. |

---

## 3. Artifact locations (on the machine where tests ran)

Paths are **examples** from the session; rerun commands below to regenerate.

- K=6 validation failure: `results/kv_qa_validation/run_20260321_230108/`
- K=24 sanity (conc 1,2): `results/kv_qa_validation/sanity_kw24_20260321_231831/`
- K=24 conc 1,4,8: `results/kv_qa_validation/kw24_conc148_20260321_232307/`
- Eval no-inflate: `results/pd_eval_noinflate_20260321_234314/` (`summary.json`, `console.log`, `results_*_conc*.json`)

Server logs in each `validate_qa` / `eval_pd_decode` result dir: `server_baseline_*.log`, `server_pipelined_*.log`.

---

## 4. Environment / ops notes

- **`--gpu-memory-utilization 0.78`** failed when **another process** held ~25 GiB on the GPU: vLLM checks **free** VRAM vs **utilization × total** and can abort startup. **`0.70`** was used successfully afterward.
- Use **`python -u`** or **`PYTHONUNBUFFERED=1`** for live logs when piping to `tee`.

---

## 5. Reproduction commands

**Correctness (recommended safe mode, no K+1 sharing):**

```bash
cd /path/to/vllm
PYTHONUNBUFFERED=1 python -u benchmarks/kv_pipelining_bench/validate_qa.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --k-window 24 \
  --concurrency 1 2 \
  --gpu-memory-utilization 0.70 \
  --result-dir /tmp/kv_qa_validation_rerun
```

**Reproduce K+1 failure (debug target):**

```bash
PYTHONUNBUFFERED=1 python -u benchmarks/kv_pipelining_bench/validate_qa.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --k-window 6 \
  --concurrency 1 \
  --gpu-memory-utilization 0.70 \
  --result-dir /tmp/kv_qa_kw6_debug
```

**Eval (stable on this setup):**

```bash
PYTHONUNBUFFERED=1 python -u benchmarks/kv_pipelining_bench/eval_pd_decode.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --k-window 6 \
  --no-inflate-kv-pool \
  --enforce-eager \
  --concurrency 16 32 \
  --num-prompts 24 \
  --gpu-memory-utilization 0.70 \
  --result-dir /tmp/pd_eval_rerun
```

---

## 6. Next steps — root-cause debugging (prioritized)

### A. K+1 **shared** decode correctness (`k_window < num_layers`, e.g. K=6)

Symptoms: **garbled** pipelined outputs vs baseline; **not** minor paraphrase.

**Hypotheses to verify:**

1. **Prefetch / CPU buffer / slot_mapping** — wrong blocks or slots when multiple logical layers alias one physical tensor.
2. **`decode_max_block_id` / `_slice_kv_layer`** — cross-request or pool indexing wrong under sharing.
3. **`plan_prefetch` / `wait_for_layer_save` ordering** — remaining races beyond `save_event` (e.g. **compute vs copy stream**, or **per-layer CPU state** vs **shared GPU** identity).
4. **Uniform cross-layer KV** (`allocate_uniform_kv_caches`) — same `Tensor` referenced by multiple `layer_name` keys; confirm connector always uses **logical** layer CPU buffers vs **physical** tensor targets consistently.

**Concrete actions:**

- Add **temporary asserts** or **logging** of `(layer_name, tensor.data_ptr(), step_id)` around `prefetch_to_gpu` / `stage_delta_to_cpu` / `wait_for_layer_load`.
- Compare **one** forward step against **non-pipelined** CPU reference for a **tiny** fixed prompt (unit-style) on **one** layer tensor.
- Re-read **`PipelinedDecodeBenchConnector`** / **`LayerwisePipelinedConnector`** — they use **window buffers** instead of in-place shared tensors; diff may highlight missing invariant.

### B. `validate_qa` Phase 2 **min similarity** &lt; 0.5 with **`k-window 24`**

Here outputs are often **both** reasonable; failure is **threshold** / **two-server** / **truncation** sensitivity.

**Actions:**

- Run **two baseline-only** runs at same conc and compare pairwise similarity to estimate **natural variance**.
- Consider **phase-specific** thresholds, **higher `max_tokens`**, or **excluding** `finish_reason=length` pairs from **min** statistics.
- Treat as **evaluation methodology** unless new evidence of a **scheduler/block** bug.

### C. **`eval_pd_decode` with `--inflate-kv-pool`**

**Actions:**

- Reproduce with **smaller** `--num-prompts`** / **lower concurrency** to see if failure is **RAM OOM** (pinned CPU) vs **GPU** vs **assert**.
- Watch **dmesg** / **host RAM** during run; reduce **inflation** or **pinned budget** in code if confirmed.

---

## 7. Related files (quick index)

| Area | File |
|------|------|
| Pipelining state, delta/save events | `vllm/v1/kv_cache_pipelining.py` |
| Prefill + pipelined decode connector | `vllm/distributed/kv_transfer/kv_connector/v1/prefill_staging_pipelined_connector.py` |
| K+1 allocator | `vllm/v1/core/kv_cache_utils.py` (`_get_pipelining_block_inflation`, etc.) |
| Attention hooks | `vllm/attention/utils/kv_transfer_utils.py` |
| QA harness | `benchmarks/kv_pipelining_bench/validate_qa.py` |
| P/D eval harness | `benchmarks/kv_pipelining_bench/eval_pd_decode.py` |
| Uniform KV allocation | `vllm/v1/worker/kv_connector_model_runner_mixin.py` (`allocate_uniform_kv_caches`) |

---

## 8. Root-cause narrowing (2026-03-22)

1. **`VLLM_PIPELINE_FULL_DECODE_STAGE=1`** (see `prefill_staging_pipelined_connector.py`) forces **full `stage_to_cpu`** on every decode step, bypassing **`stage_delta_to_cpu`**. **`validate_qa` still fails** (~0.03 avg similarity). So the primary bug is **not** isolated to the delta scatter path alone.

   **`VLLM_PIPELINE_DEBUG_LOG=1`** turns on **`logger.info`** lines from the same connector (`start_load_kv`, prefetch batch, `wait_for_layer_load`, `save_kv_layer`): forward id, `layers_to_prefetch`, slice bounds / `data_ptr`, `plan_prefetch`, and decode slot-mapping summary. Use when running **`validate_qa.py`** to compare scheduling vs attention writes.

2. **Unit tests** `tests/v1/kv_connector/unit/test_kv_cache_pipelining_roundtrip.py` verify that **`LayerwisePipeliningState`** full round-trip and delta scatter into the CPU buffer match reference indexing on synthetic tensors (both **pass**). Isolated copy math is therefore **unlikely** to be wrong.

3. **Conclusion:** Suspect **connector scheduling** relative to the **K+1 shared physical tensors**: **`_slice_kv_layer` bounds**, **prefetch/wait ordering** vs **which layer owns which tensor**, or **metadata** (`decode_max_block_id`, `layers_to_prefetch`) diverging from what attention wrote. **Instrumentation:** set **`VLLM_PIPELINE_DEBUG_LOG=1`** and inspect a short **`validate_qa`** run (see env var above).

---

*Last updated from the debugging/validation session(s) described above. Update this file as root causes are confirmed or fixes land.*
