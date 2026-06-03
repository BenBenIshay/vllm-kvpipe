#!/bin/bash
# Build Apptainer image from definition file (no Docker or root required)
# Usage: ./build_apptainer.sh [--fakeroot]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create requirements.txt if it doesn't exist (needed for %files section)
if [ ! -f requirements.txt ]; then
  echo "# Create this file or run create_reqs_file.py to add Python dependencies" > requirements.txt
  echo "Note: Created placeholder requirements.txt - add your packages or run create_reqs_file.py"
fi

# Build with fakeroot if --fakeroot passed (for systems without root)
# Requires: subuid/subgid configured for your user (ask admin)
if [ "$1" == "--fakeroot" ]; then
  apptainer build --fakeroot vllm-research.sif vllm-research.def
else
  # Standard build (may require root or configured unprivileged build)
  apptainer build vllm-research.sif vllm-research.def
fi

echo "Build complete: vllm-research.sif"
