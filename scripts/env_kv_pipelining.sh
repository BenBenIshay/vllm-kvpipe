#!/usr/bin/env bash
# KV pipelining — per-session environment (source this file in every new shell).
#
#   source /path/to/vllm/scripts/env_kv_pipelining.sh
#
# Puts large caches on a non-quota path (default /data/ben) so Hugging Face and
# vLLM metadata do not fill your home 100G quota. Override the root if needed:
#   export KV_PIPELINING_DATA_ROOT=/your/big/disk/path
#   source scripts/env_kv_pipelining.sh
#
# Optional: same root is used for Apptainer layer cache (APPTAINER_CACHEDIR) so
# future container pulls do not land under ~/.apptainer on home.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file instead of executing it:" >&2
  echo "  source ${BASH_SOURCE[0]}" >&2
  exit 1
fi

_KV_ROOT="${KV_PIPELINING_DATA_ROOT:-/data/ben}"

if [[ -d "${_KV_ROOT}" ]]; then
  export HF_HOME="${_KV_ROOT}/hf_cache"
  export VLLM_CACHE_ROOT="${_KV_ROOT}/vllm_cache"
  mkdir -p "${VLLM_CACHE_ROOT}" "${HF_HOME}" 2>/dev/null || true

  # Apptainer/Singularity: keep blob/oci-tmp off home (saves tens of GB over time)
  export APPTAINER_CACHEDIR="${_KV_ROOT}/apptainer_cache"
  mkdir -p "${APPTAINER_CACHEDIR}" 2>/dev/null || true
else
  echo "[env_kv_pipelining] WARN: KV_PIPELINING_DATA_ROOT (${_KV_ROOT}) is not a directory — not setting HF_HOME / VLLM_CACHE_ROOT / APPTAINER_CACHEDIR." >&2
  echo "[env_kv_pipelining]      Create it or export KV_PIPELINING_DATA_ROOT to a large filesystem." >&2
fi

unset _KV_ROOT
