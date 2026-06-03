#!/bin/bash
# Attach to a running Manar container instance (get a shell)
# Usage: ./attach_cont.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INSTANCE_NAME="vllm-research"

apptainer exec --userns instance://"$INSTANCE_NAME" bash -c "cd /workspace && exec bash"
