# Decode-only fast bench — session notes (continuation)

Use this file to resume work on **dummy-KV pipelined decode** tuning and baseline
comparison. Last consolidated: **2026-03-28**.

## Goals (this thread)

1. Run **decode-only** pipelined benchmarks reliably on a **shared / memory-tight** GPU.
2. Sweep **decreasing K** (`kv-k-window`) to pick a good **K** under limited resources.
3. Capture **baseline** reference numbers and clarify what the harness actually measures.

## Main scripts

| Script | Role |
|--------|------|
| `pipelined_decode_fast_bench.py` | Dummy KV → CPU staging → pipelined decode (`PipelinedDecodeBenchConnector`). Optional **baseline** (`DecodeBenchConnector`). |
| `find_optimal_k.py` | Runs the fast bench once per **K**, recommends **K** by max throughput with **p99 ITL** within tolerance of the **first K** in the list. |

Project Python: **`./.venv/bin/python`** (system `python3` may lack `torch`).

Hub downloads: use **`HF_HUB_ENABLE_HF_TRANSFER=0`** unless `hf_transfer` is installed.

## Defaults / behavior we set (code)

- **`--enforce-eager` default True** — avoids CUDA graph / `dummy_run` mismatch with the bench connector; use `--no-enforce-eager` only if you accept graph issues.
- **GPU memory monitor**: higher **`--memory-warn-used-frac` (default 0.995)**, **`--memory-monitor-start-delay` (default 90s)** — reduces false aborts when HBM is legitimately high during load.
- **`--free-memory-fraction` default 0.6** — auto `--gpu-memory-utilization` uses **60% of free HBM** (expressed as `fraction × free/total`, clamped) to leave headroom on shared GPUs.
- **`find_optimal_k.py`**:
  - **`--lite`**: K sweep **6 → 4 → 2**, concurrency **4 / 8 / 16**, measure at **16**, prompts **256 / 64** tokens (lighter than full defaults).
  - **`result_dir` is created early** in `main()` so `| tee .../run.log` works once the process starts; still safest: `mkdir -p <result-dir>` before `tee`.

## K sweep we ran (`--lite`)

```bash
cd /home/ben/vllm
mkdir -p /data/ben/kv_fast_bench_run/k_sweep_lite
PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
  ./.venv/bin/python benchmarks/kv_pipelining_bench/find_optimal_k.py \
  --lite \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --result-dir /data/ben/kv_fast_bench_run/k_sweep_lite \
  --port 8201 --startup-timeout 1200 --request-timeout 300 \
  2>&1 | tee /data/ben/kv_fast_bench_run/k_sweep_lite/run.log
```

**Output:** `/data/ben/kv_fast_bench_run/k_sweep_lite/optimal_k_report.json`

**Outcome (same machine, measure concurrency 16, p99 ITL tolerance 5% vs ref K=6):**

- **Recommended K: 6** (best throughput among K within p99 bound).
- K=4 slightly over p99 bound vs K=6; K=2 in bound but lower throughput than K=6.

## Baseline reference run (single-stream ITL caveat)

We ran **baseline only** (all KV on GPU, dummy fill) with high **throughput** concurrency:

```bash
mkdir -p /data/ben/kv_fast_bench_run/baseline_itl_ref
PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
  ./.venv/bin/python benchmarks/kv_pipelining_bench/pipelined_decode_fast_bench.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --modes baseline \
  --k-window 6 \
  --concurrency 32 64 128 256 \
  --input-len 256 --output-len 64 \
  --result-dir /data/ben/kv_fast_bench_run/baseline_itl_ref \
  --port 8202 --startup-timeout 1200 --request-timeout 300
```

**Results file:** `/data/ben/kv_fast_bench_run/baseline_itl_ref/results_baseline.json`

**Important:** **`latency` ITL (mean / p99)** in this harness is from **single-request streaming** (5 measured runs), **not** under N concurrent decodes. The **throughput** section **is** under the listed concurrency but **non-streaming** — it does not report per-token ITL at load.

Approximate baseline **unloaded** ITL from that run: **mean ~5.4 ms**, **p99 ~115 ms** (see JSON for exact values).

For **loaded** ITL at high concurrency (apples-to-apples with “many concurrent streams”), use **`vllm bench serve`**, **`eval_kv_pipelining.py`**, or extend the bench with **concurrent streaming** ITL (not implemented as of this note).

## Other useful paths

- Earlier fast bench outputs: `/data/ben/kv_fast_bench_run/` (`manifest.json`, `summary_all.json`, various `run*.log`).
- Full project plan / prior evals: repo root **`SESSION_CONTEXT.md`**, **`next_steps.md`**.

## Suggested next steps when you resume

1. Re-run **`find_optimal_k`** without `--lite` (`--k-values 6 5 4 3 2`, tune `--concurrency` / `--measure-concurrency`) on a **larger model** if GPU allows.
2. Run **`pipelined_decode_fast_bench.py --with-baseline`** (or `--modes pipelined baseline`) at fixed **K** for side-by-side JSON under one `result-dir`.
3. If you need **ITL under high concurrency**: integrate **`vllm bench`** metrics or add concurrent streaming ITL to `decode_only_bench.py` / fast bench.
4. Keep **`PYTHONUNBUFFERED=1`** when piping logs so output is not delayed by buffering.
