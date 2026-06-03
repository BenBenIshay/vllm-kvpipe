# Migration — Environment Manifest

Snapshot of the current working server, for rebuilding the vLLM + layer-wise KV
pipelining environment on a new machine. Captured from `/home/ben/vllm`.

## Host / hardware (current server)

| Item | Value |
|------|-------|
| OS | Ubuntu 24.04.4 LTS |
| GPU | NVIDIA H100 NVL (94 GB) |
| NVIDIA driver | 595.71.05 |
| CUDA (torch wheels) | 12.6 |
| Python | 3.12.3 |

The new server needs an NVIDIA GPU with a driver supporting CUDA 12.6, Python
3.12, and enough host RAM for the pinned CpuKvTable (≈ full KV footprint of the
largest model you offload; budget tens of GB).

## vLLM source state

| Item | Value |
|------|-------|
| Upstream remote | `https://github.com/vllm-project/vllm.git` |
| Base tag | `v0.14.0rc1` (HEAD is `v0.14.0rc1-91-g46c975eaa`) |
| HEAD commit | `46c975eaaef899ac02c6b0f57ea92f331a6bf45e` — "Add decode-only KV pipelining perf path with B1/B2 batching and step plan." |
| Reported version | `0.14.0rc2.dev88+gee1e3c104` |
| Install mode | editable / source build (`pip install -e . --no-build-isolation`) |

Our work = the HEAD commit (already committed) **+** uncommitted delta:
- **34 tracked files** with real content changes (see `uncommitted_changes.patch`).
- **~155 untracked custom files** (benchmarks, scheduler, tests, scripts; see
  `custom_files_manifest.txt`).

> Note: the working tree also shows ~3,700 files as "modified", but those are
> executable-bit (filemode 100644→100755) changes only — not content. Ignore them
> (`git -c core.fileMode=false ...`). The patch and manifest already exclude them.

## Key pinned dependencies

Full list in `pip_freeze.txt`. Critical pins:

| Package | Version |
|---------|---------|
| torch | 2.9.1+cu126 |
| torchvision | 0.24.1+cu126 |
| torchaudio | 2.9.1+cu126 |
| triton | 3.5.1 |
| transformers | 4.57.6 |
| numpy | 2.2.6 |
| aiohttp | 3.13.3 |
| nvidia-nccl-cu12 | 2.27.5 |

Install torch from `https://download.pytorch.org/whl/cu126` (the setup script does this).

## Environment variables / cache paths (per-session)

Set by `scripts/env_kv_pipelining.sh` (root `KV_PIPELINING_DATA_ROOT`, default
`/data/ben`) — point this at a large non-quota disk on the new server:

| Var | Default | Purpose |
|-----|---------|---------|
| `HF_HOME` | `$ROOT/hf_cache` | Hugging Face cache |
| `HF_HUB_CACHE` | `/data/ben/huggingface` | model snapshots (used by benchmark scripts) |
| `VLLM_CACHE_ROOT` | `$ROOT/vllm_cache` | vLLM compile/metadata cache |
| `APPTAINER_CACHEDIR` | `$ROOT/apptainer_cache` | container layer cache |

Diagnostic env flags used by the pipelining work:
- `VLLM_KVPIPE_FAKE_TRANSFER=1` — zero-byte transfer (isolates orchestration cost).

## Secrets — NOT in this repo (provision manually on the new server)

- **`hf_token`** (repo root) — Hugging Face token for gated models
  (Llama-3.1-8B, etc.). Excluded via `.gitignore`. On the new server, recreate it:
  `printf '%s' '<HF_TOKEN>' > hf_token` (or `huggingface-cli login`).

## Large artifacts — NOT in this repo

- **`docker/vllm-research.sif`** (8.3 GB Apptainer image) — rebuild from
  `docker/vllm-research.def` if you use the container path.

## Models exercised

- `Qwen/Qwen2.5-0.5B-Instruct` (smoke/correctness)
- `meta-llama/Llama-3.1-8B-Instruct` (primary; gated → needs HF token)
- `Qwen/Qwen2.5-32B-Instruct`, `Qwen/Qwen2.5-14B-Instruct` (capacity/scale)
