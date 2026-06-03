# Apptainer workflow (recommended for dev)

## Why not `vllm/vllm-openai` for building from source?

The `vllm-openai` image is oriented toward **running** vLLM. Its CUDA layout often **omits** full math library headers (`cusparse.h`, `cublas_v2.h`, etc.), so **compiling** the mounted tree fails even though inference works.

For **editable / from-source** work (KV pipelining, CUDA kernels), use a **PyTorch CUDA *devel*** image so `/usr/local/cuda/include` is complete.

## Default image

`docker://pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel`

This matches **PyTorch 2.9.1 + CUDA 12.6** (same wheel family as `scripts/setup_kv_pipelining_env.sh`).

Override with `VLLM_DOCKER_REF` / `VLLM_APPTAINER_SIF` if your site requires another tag.

## Scripts

| Script | Purpose |
|--------|---------|
| [vllm_apptainer_dev.sh](vllm_apptainer_dev.sh) | Pull image (first time), then `apptainer shell/exec` with repo at `/workspace/vllm`, `--nv`. |
| [apptainer_install_editable.sh](apptainer_install_editable.sh) | `pip install -e . --no-build-isolation` inside the image (full native build). |
| [setup_apptainer_git_jammy.sh](setup_apptainer_git_jammy.sh) | Optional (legacy): Jammy `git` bind for slim images. Prefer **Miniconda git** installed by `apptainer_install_editable.sh` into `~/.local/share/vllm-apptainer/miniconda3`. |

## Usage

```bash
cd /path/to/vllm
bash scripts/setup_apptainer_git_jammy.sh   # only if git is missing inside the container
bash scripts/apptainer_install_editable.sh  # long compile
bash scripts/vllm_apptainer_dev.sh          # interactive shell
```

Environment: `MAX_JOBS`, `NVCC_THREADS`, `SETUPTOOLS_SCM_PRETEND_VERSION`, `CMAKE_ARGS` as needed.
