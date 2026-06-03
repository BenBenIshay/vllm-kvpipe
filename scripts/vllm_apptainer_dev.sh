#!/usr/bin/env bash
# vLLM development via Apptainer: bind-mount this repo at /workspace/vllm (matches Dockerfile.dev).
#
# First run downloads the image (large). Override image path or Docker ref with:
#   VLLM_APPTAINER_SIF=/path/to/image.sif
#   VLLM_DOCKER_REF=docker://pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel
#
# Default is a PyTorch *devel* image (full CUDA headers) so `pip install -e .` can compile.
# The slimmer vllm/vllm-openai image is poor for source builds — see APPTAINER.md.
#
# Usage:
#   bash scripts/vllm_apptainer_dev.sh                    # interactive shell, GPU, cwd /workspace/vllm
#   bash scripts/vllm_apptainer_dev.sh python -c "import vllm; print(vllm.__version__)"
#
# Typical editable install (KV / C++ work) inside the shell:
#   pip install -e . --no-build-isolation
# Optional: export MAX_JOBS=$(($(nproc)/2)) first on shared hosts.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_DIR="${VLLM_APPTAINER_IMAGE_DIR:-${HOME}/.local/share/vllm-apptainer}"
IMAGE="${VLLM_APPTAINER_SIF:-${IMAGE_DIR}/pytorch_2.9.1_cuda126_devel.sif}"
DOCKER_REF="${VLLM_DOCKER_REF:-docker://pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel}"

mkdir -p "${IMAGE_DIR}"
if [[ ! -f "${IMAGE}" ]]; then
  echo "=== First-time pull: ${DOCKER_REF} -> ${IMAGE} (this can take a while and use several GB) ==="
  apptainer pull "${IMAGE}" "${DOCKER_REF}"
fi

# CMake FetchContent clones CUTLASS with `git`; the vllm-openai image often has no git. Host `git` may
# need a newer glibc than the image (Jammy = 2.35). Prefer Jammy .deb extracted under
# ~/.local/share/vllm-apptainer/git-jammy (see scripts/setup_apptainer_git_jammy.sh).
GIT_JAMMY="${VLLM_APPTAINER_GIT_ROOT:-${HOME}/.local/share/vllm-apptainer/git-jammy}"
COMMON=(--nv --bind "${REPO_ROOT}:/workspace/vllm" --pwd /workspace/vllm)
if [[ -x "${GIT_JAMMY}/usr/bin/git" ]]; then
  COMMON+=(--bind "${GIT_JAMMY}/usr/bin/git:/usr/bin/git")
  COMMON+=(--bind "${GIT_JAMMY}/usr/lib/git-core:/usr/lib/git-core")
fi

if [[ $# -eq 0 ]]; then
  exec apptainer shell "${COMMON[@]}" "${IMAGE}"
else
  exec apptainer exec "${COMMON[@]}" "${IMAGE}" "$@"
fi
