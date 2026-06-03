#!/usr/bin/env bash
# Run an interactive Apptainer shell equivalent to `docker compose up` for vllm-research
# (see ../docker-compose.yml). Adjust paths if your checkout is not /home/ben/vllm.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SIF="${VLLM_RESEARCH_SIF:-${REPO_ROOT}/docker/vllm-research.sif}"
HF_CACHE_HOST="${HF_HOME:-${HOME}/.cache/huggingface}"

if [[ ! -f "${SIF}" ]]; then
  echo "Missing ${SIF}. Build it first:"
  echo "  cd ${REPO_ROOT} && apptainer build docker/vllm-research.sif docker/vllm-research.def"
  exit 1
fi

mkdir -p "${HF_CACHE_HOST}"

exec apptainer shell --nv \
  --bind "${REPO_ROOT}:/workspace/vllm" \
  --bind "${HF_CACHE_HOST}:/root/.cache/huggingface" \
  --pwd /workspace/vllm \
  "${SIF}"
