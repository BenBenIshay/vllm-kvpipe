#!/usr/bin/env bash
# Run a full editable install of the mounted repo inside the Apptainer image (CUDA/C++ dev).
# Uses the same image as vllm_apptainer_dev.sh. Safe parallelism: MAX_JOBS defaults to half CPUs.
#
# setuptools-scm needs a version string; git inside the image may be missing or incompatible with
# a bind-mounted host git (glibc mismatch). We derive the version from the host repo with git.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export MAX_JOBS="${MAX_JOBS:-$(($(nproc) / 2))}"
[[ "${MAX_JOBS}" -lt 1 ]] && export MAX_JOBS=1
export NVCC_THREADS="${NVCC_THREADS:-1}"

if [[ -z "${SETUPTOOLS_SCM_PRETEND_VERSION:-}" ]]; then
  export SETUPTOOLS_SCM_PRETEND_VERSION="$(cd "${REPO_ROOT}" && python3 -c "
import re, subprocess
try:
    d = subprocess.check_output(
        ['git', 'describe', '--tags', '--long'], text=True, cwd='.').strip()
except Exception:
    print('0.0.0.dev0')
    raise SystemExit(0)
d = re.sub(r'^v', '', d)
# 0.14.0rc1-86-g765159108 -> 0.14.0rc1.dev86 (PEP 440)
m = re.match(r'^(.+)-([0-9]+)-g[0-9a-f]+\$', d)
if m:
    print(f\"{m.group(1)}.dev{m.group(2)}\")
else:
    print(d)
")"
fi
if [[ -z "${SETUPTOOLS_SCM_PRETEND_VERSION}" ]]; then
  export SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0.dev0"
fi

# Avoid git+HTTPS inside minimal containers: pre-clone CUTLASS on the host (see CMakeLists CUTLASS_REVISION).
CUTLASS_DIR="${REPO_ROOT}/.deps/cutlass-src"
if [[ ! -f "${CUTLASS_DIR}/CMakeLists.txt" ]]; then
  mkdir -p "${REPO_ROOT}/.deps"
  echo "Cloning CUTLASS into ${CUTLASS_DIR} (one-time, uses host git)..."
  git clone --depth 1 --branch v4.2.1 https://github.com/nvidia/cutlass.git "${CUTLASS_DIR}"
fi

# User-writable git (conda-forge): Apptainer user cannot `conda install` into /opt/conda; images often lack working HTTPS git.
MC="${HOME}/.local/share/vllm-apptainer/miniconda3"
if [[ ! -x "${MC}/bin/git" ]]; then
  echo "One-time: installing Miniconda + git to ${MC} (for FetchContent clones inside the container)..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda-vllm.sh
  bash /tmp/miniconda-vllm.sh -b -p "${MC}"
  "${MC}/bin/conda" install -y -c conda-forge git
fi

echo "Installing vLLM editable in Apptainer (MAX_JOBS=${MAX_JOBS} NVCC_THREADS=${NVCC_THREADS})..."
echo "  SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION}"
export SETUPTOOLS_SCM_PRETEND_VERSION
# Some CUDA images only ship versioned libnvrtc.so.12; CMake's FindCUDAToolkit may set CUDA_nvrtc_LIBRARY to NOTFOUND.
export CMAKE_ARGS="${CMAKE_ARGS:--DCUDA_nvrtc_LIBRARY=/usr/local/cuda/lib64/libnvrtc.so.12}"
exec "${SCRIPT_DIR}/vllm_apptainer_dev.sh" bash -c "
  set -e
  export MAX_JOBS=${MAX_JOBS} NVCC_THREADS=${NVCC_THREADS}
  export CMAKE_ARGS=\"\${CMAKE_ARGS}\"
  cd /workspace/vllm
  export PATH=\"${HOME}/.local/share/vllm-apptainer/miniconda3/bin:\${PATH}\"
  export LD_LIBRARY_PATH=\"${HOME}/.local/share/vllm-apptainer/miniconda3/lib:\${LD_LIBRARY_PATH:-}\"
  command -v git && git --version
  export VLLM_CUTLASS_SRC_DIR=/workspace/vllm/.deps/cutlass-src
  python3 -m pip install -U pip
  # PEP 517 metadata step needs setuptools-scm, cmake, grpcio-tools, etc. (matches pyproject [build-system]).
  python3 -m pip install --user -r requirements/build.txt
  export PATH=\"\${HOME}/.local/bin:\${PATH}\"
  command -v cmake
  pip install -e . --no-build-isolation
"
