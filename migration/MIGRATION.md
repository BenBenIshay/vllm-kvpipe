# Migration Runbook — rebuild vLLM + layer-wise KV pipelining on a new server

For the agent setting up the new machine. Goal: reproduce the working
`/home/ben/vllm` environment (vLLM source build + KV pipelining work + benchmarks).

See `ENVIRONMENT.md` for the exact versions/hardware this was captured from, and
`pip_freeze.txt` for the full dependency list.

## 0. Prerequisites on the new server

- NVIDIA GPU + driver supporting CUDA 12.6; verify `nvidia-smi`.
- System packages:
  ```bash
  sudo apt update
  sudo apt install -y python3.12 python3.12-venv python3.12-dev \
      cmake ninja-build build-essential git
  ```
- A large non-quota disk for caches (default `/data/ben`); set
  `KV_PIPELINING_DATA_ROOT` if it lives elsewhere.

## 1. Get the code

This repo is a vLLM **source checkout** with our changes. Two equivalent paths:

**Path A — clone the migration repo (recommended; everything is in git):**
```bash
git clone https://github.com/BenBenIshay/<REPO>.git vllm
cd vllm
```

**Path B — base + patch (if only the base commit was pushed):**
```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
git checkout 46c975eaaef899ac02c6b0f57ea92f331a6bf45e   # our base commit
git apply migration/uncommitted_changes.patch            # 34 tracked-file edits
# then copy the untracked custom files listed in migration/custom_files_manifest.txt
```

## 2. Provision secrets (NOT in git)

```bash
printf '%s' '<YOUR_HF_TOKEN>' > hf_token        # gated models (Llama-3.1-8B)
# or: huggingface-cli login
```

## 3. Build the environment (one command)

The repo ships a complete builder:
```bash
bash scripts/setup_kv_pipelining_env.sh
```
It creates `.venv`, installs torch 2.9.1+cu126, builds vLLM editable
(`--no-build-isolation`), installs benchmark deps, and runs an import smoke test.
Tune build memory with `MAX_JOBS=...` / `NVCC_THREADS=...`.

## 4. Per-session env (every new shell)

```bash
source .venv/bin/activate
export KV_PIPELINING_DATA_ROOT=/your/big/disk   # if not /data/ben
source scripts/env_kv_pipelining.sh
export HF_HUB_CACHE="$KV_PIPELINING_DATA_ROOT/huggingface"   # used by bench scripts
```

> vLLM imports only with the repo root on the path. Benchmark wrappers export
> `PYTHONPATH=<repo-root>` for the servers they launch; if you run a server by
> hand, `cd` to the repo root or set `PYTHONPATH`.

## 5. Verify the environment

```bash
# import + CUDA check (done by the setup script too):
.venv/bin/python -c "import torch; from vllm.v1.kv_cache_pipelining import LayerwisePipeliningState; print('OK', torch.cuda.is_available())"

# end-to-end correctness + suite smoke on the small model (~5 min):
python benchmarks/kv_pipelining_bench/validate_qa.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --k-window 4 --concurrency 1 8
SMOKE=1 bash benchmarks/kv_pipelining_bench/paper_suite/run_all.sh /tmp/paper_smoke
```

Expected: validate_qa PASS (conc=1 bit-exact; conc=8 within baseline
self-consistency band), and `run_all.sh` writes `/tmp/paper_smoke/PAPER_RESULTS.md`
with C1/C2 + C6 artifacts.

## 6. Paper test suite (the deliverable)

See `benchmarks/kv_pipelining_bench/paper_suite/README.md` — it maps the six
paper claims (C1–C6) to scripts and has the full free-GPU collection commands.

## Custom code map (what our work added/changed)

Core engine (changed/added under `vllm/`):
- `vllm/v1/kv_cache_pipelining.py`, `kv_pipelining_step.py`,
  `kv_pipelining_direct_window.py`, `kv_pipelining_profile.py`, `cpu_kv_table.py`
- connectors: `vllm/distributed/kv_transfer/kv_connector/v1/`
  `prefill_staging_pipelined_connector.py`, `pipelined_decode_bench_connector.py`
- scheduler: `vllm/v1/core/sched/scheduler.py`,
  `vllm/v1/core/sched/transfer_aware_admission.py` (SLO-aware admission)
- `vllm/v1/core/kv_cache_utils.py`, `arg_utils.py`, `config/scheduler.py`,
  worker/model-runner files, metrics.

Benchmarks / paper suite (`benchmarks/kv_pipelining_bench/`):
- `paper_suite/` — orchestrated C1–C6 suite (this is the test harness).
- `validate_qa.py` (+ `paper_suite/qa_nondeterminism_control.py`) — correctness.
- `eval_pd_decode.py`, `decode_only_bench.py`, `feature_on_k_sweep.py`,
  `admission_predictor.py`, `bench_admission_policy_e2e.py`,
  `run_injection_sweep_v5.py`, `prove_*.sh`, `summarize_*`, `plot_*`.

Tests: `tests/v1/core/test_transfer_aware_admission.py`,
`tests/v1/kv_connector/unit/test_kv_cache_pipelining_roundtrip.py`, others.
