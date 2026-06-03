#!/bin/bash
# Start the Manar container as a background instance
# Usage: ./new_start_cont.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use user tmp dir to avoid filling /tmp (limited space on /)
# Override with env vars if needed, e.g. MANAR_TMPDIR
MANAR_TMP="${MANAR_TMPDIR:-/home/ben/tmp}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$MANAR_TMP}"
export TMPDIR="${TMPDIR:-$MANAR_TMP}"
export SINGULARITY_LOCALCACHEDIR="${SINGULARITY_LOCALCACHEDIR:-$MANAR_TMP}"
mkdir -p "$MANAR_TMP"

# Overlay path - directory (override with MANAR_OVERLAY_PATH)
# Directory overlay avoids loop device; no size limit (uses disk space)
OVERLAY_PATH="${MANAR_OVERLAY_PATH:-/data/ben/overlays/vllm_overlay}"
INSTANCE_NAME="vllm-research"

mkdir -p "$OVERLAY_PATH"

apptainer instance start --nv --fakeroot --overlay "$OVERLAY_PATH" \
  --bind ../:/workspace \
  --bind /data:/data \
  vllm-research.sif "$INSTANCE_NAME"

echo "Instance $INSTANCE_NAME started."
echo "Attach with: ./attach_cont.sh"
echo "Overlay (persistent): $OVERLAY_PATH"
echo "Temp/rootfs dir: $MANAR_TMP"
