# Source this to get a working nvcc on a cluster with no CUDA toolkit.
# The pip package nvidia-cuda-nvcc-cu12 ships ONLY ptxas -- no nvcc driver, no
# cicc -- so FlashInfer's JIT and DirectKV's SM90a kernel cannot build from it.
# These are NVIDIA's per-component redistributables (~80 MB, no root needed).
export CUDA_HOME=/mnt/beegfs/benishay/cuda-12.8
export PATH="$CUDA_HOME/bin:/mnt/beegfs/benishay/vllm-kvpipe/.venv/bin:$PATH"  # .venv/bin carries ninja, which FlashInfer JIT shells out to
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
