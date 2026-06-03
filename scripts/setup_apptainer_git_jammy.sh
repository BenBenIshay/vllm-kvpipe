#!/usr/bin/env bash
# One-time: extract Ubuntu 22.04 (Jammy) git into ~/.local/share/vllm-apptainer/git-jammy
# so Apptainer can bind-mount a glibc-2.35-compatible git (matches vllm-openai image).
# Safe to re-run; skips if usr/bin/git already exists.

set -euo pipefail
ROOT="${VLLM_APPTAINER_GIT_ROOT:-${HOME}/.local/share/vllm-apptainer/git-jammy}"
DEB_URL="${VLLM_GIT_JAMMY_DEB_URL:-http://archive.ubuntu.com/ubuntu/pool/main/g/git/git_2.34.1-1ubuntu1.17_amd64.deb}"

mkdir -p "${ROOT}"
cd "${ROOT}"
if [[ -x usr/bin/git ]]; then
  echo "Jammy git already present at ${ROOT}/usr/bin/git"
  usr/bin/git --version
  exit 0
fi

echo "Downloading ${DEB_URL} ..."
wget -q --show-progress "${DEB_URL}" -O git.deb
dpkg-deb -x git.deb .
rm -f git.deb
usr/bin/git --version
echo "Done. vllm_apptainer_dev.sh will bind-mount this git when present."
