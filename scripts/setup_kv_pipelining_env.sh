#!/usr/bin/env bash
# Setup Python environment for KV cache pipelining (vLLM + torch + benchmarks).
# Run from repo root: bash scripts/setup_kv_pipelining_env.sh
#
# One-time system dependencies:
#   sudo apt install -y python3.12-venv   # if venv creation fails
#   sudo apt install -y python3.12-dev    # required for vLLM native build (CMake Python headers)
#   sudo apt install -y cmake ninja build-essential  # for building vLLM native extensions
#
# Build load limits (safe defaults; override anytime):
#   MAX_JOBS   — parallel compile jobs for vLLM's CMake/Ninja build. If unset, defaults to
#              max(1, nproc/2) to reduce RAM pressure vs using all CPUs.
#   NVCC_THREADS — forwarded by vLLM setup.py; defaults to 1 here for predictable memory use.

set -e
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"
# Pip-installed cmake/ninja live in the venv; build subprocesses must find them on PATH.
export PATH="${VENV_DIR}/bin:${PATH}"

echo "=== KV Pipelining environment setup ==="
echo "  Repo root: ${REPO_ROOT}"
echo "  Venv:      ${VENV_DIR}"
echo ""

# ---------------------------------------------------------------------------
# 1. Create virtual environment
# ---------------------------------------------------------------------------
if [[ ! -x "${VENV_DIR}/bin/python" ]] || [[ ! -x "${VENV_DIR}/bin/pip" ]]; then
  if [[ -d "${VENV_DIR}" ]]; then
    echo "Removing incomplete venv at ${VENV_DIR} ..."
    rm -rf "${VENV_DIR}"
  fi
  echo "Creating virtual environment at ${VENV_DIR} ..."
  if ! python3 -m venv "${VENV_DIR}"; then
    echo ""
    echo "ERROR: Could not create venv. On Debian/Ubuntu run:"
    echo "  sudo apt install -y python3.12-venv"
    echo ""
    echo "Then re-run this script."
    exit 1
  fi
  echo "  Done."
else
  echo "Using existing venv: ${VENV_DIR}"
fi

# ---------------------------------------------------------------------------
# 2. Upgrade pip
# ---------------------------------------------------------------------------
"${PIP}" install --upgrade pip -q
echo "  pip: $("${PIP}" --version)"

# ---------------------------------------------------------------------------
# 3. Install PyTorch with CUDA (so vLLM setup detects CUDA and builds correctly)
# ---------------------------------------------------------------------------
echo ""
echo "Installing PyTorch 2.9.1 with CUDA 12 ..."
"${PIP}" install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu126

# ---------------------------------------------------------------------------
# 4. Install vLLM in editable mode (pulls deps from requirements/cuda.txt + common.txt)
# ---------------------------------------------------------------------------
# Require Python dev headers so CMake find_package(Python Development) succeeds.
PYTHON_VER=$("${PYTHON}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_HEADER="/usr/include/python${PYTHON_VER}/Python.h"
if [[ ! -f "${PYTHON_HEADER}" ]]; then
  echo ""
  echo "ERROR: Python development headers not found (required for vLLM native build)."
  echo "  Expected: ${PYTHON_HEADER}"
  echo ""
  echo "On Debian/Ubuntu run:"
  echo "  sudo apt install -y python${PYTHON_VER}-dev"
  echo ""
  echo "Then re-run this script."
  exit 1
fi

# ---------------------------------------------------------------------------
# 4a. Limit parallel compilation (vLLM reads MAX_JOBS / NVCC_THREADS — see setup.py)
# ---------------------------------------------------------------------------
if [[ -z "${MAX_JOBS:-}" ]]; then
  _mj="$(($(nproc) / 2))"
  [[ "${_mj}" -lt 1 ]] && _mj=1
  export MAX_JOBS="${_mj}"
fi
export NVCC_THREADS="${NVCC_THREADS:-1}"

echo ""
echo "  Build limits: MAX_JOBS=${MAX_JOBS}  NVCC_THREADS=${NVCC_THREADS}"
echo "    (override with MAX_JOBS=... and/or NVCC_THREADS=... in the environment)"

# ---------------------------------------------------------------------------
# 4b. Build-backend deps in the venv, then editable install without pip build isolation.
#     Default isolated `pip install -e .` can provision a *second* torch in a temp build env
#     (see pyproject.toml [build-system] requires), which spikes RAM. Upstream documents
#     `pip install --no-build-isolation -e .` after installing torch (gpu.cuda.inc.md).
# ---------------------------------------------------------------------------
echo ""
echo "Installing build dependencies (requirements/build.txt) ..."
"${PIP}" install -r "${REPO_ROOT}/requirements/build.txt"

echo ""
echo "Installing vLLM in editable mode (may take a long time; compiles native code) ..."
# If CMake complains about a cache from another path, remove: rm -rf .deps build
"${PIP}" install -e . --no-build-isolation
# If build fails, ensure: cmake, ninja, build-essential, and CUDA toolkit (nvcc) are installed.

# ---------------------------------------------------------------------------
# 5. Optional: benchmark and graph deps
# ---------------------------------------------------------------------------
echo ""
echo "Installing optional deps for benchmarks (matplotlib, aiohttp for validate_qa) ..."
"${PIP}" install matplotlib numpy aiohttp -q
# Optional LMCache comparison: pip install lmcache==0.3.14

# ---------------------------------------------------------------------------
# 6. Smoke test
# ---------------------------------------------------------------------------
echo ""
echo "Smoke test (KV pipelining imports) ..."
"${PYTHON}" -c "
from vllm.distributed.kv_transfer.kv_connector.v1.prefill_staging_pipelined_connector import PrefillStagingPipelinedConnector
from vllm.v1.kv_cache_pipelining import LayerwisePipeliningState
from vllm.config.cache import CacheConfig
print('  KV pipelining imports OK')
import torch
print('  torch.cuda.is_available():', torch.cuda.is_available())
"

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Every new session (before benchmarks / vLLM), load cache paths so Hugging Face"
echo "and vLLM metadata do not use your home quota (default: /data/ben):"
echo "  source ${REPO_ROOT}/scripts/env_kv_pipelining.sh"
echo ""
echo "Then run the real-workload sanity check:"
echo "  python benchmarks/kv_pipelining_bench/validate_qa.py --model meta-llama/Llama-3.1-8B-Instruct --k-window 6"
echo ""
