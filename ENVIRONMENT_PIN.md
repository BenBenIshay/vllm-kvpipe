# Environment pin — recorded 2026-09-05T16:09:01+03:00

Written for the GH200 rental: the target box must reproduce this, and
`nvcc` must be present (a CUDA `devel` image, not `runtime`) or we lose
the FlashInfer arm of 4.2 and all of 4.5.

## Versions (A100 reference environment)
```
python       3.12.13
torch        2.9.1+cu128
cuda (torch) 12.8
cudnn        91002
platform     Linux-4.18.0-425.3.1.el8.x86_64-x86_64-with-glibc2.28
device       <no gpu on this host>
```

## Key packages
```
numpy                             2.2.6
safetensors                       0.7.0
tokenizers                        0.22.2
torch                             2.9.1
transformers                      4.57.6
triton                            3.5.1
vllm                              0.1.dev13032+g3661a60c1.d20260603.cu128 /mnt/beegfs/benishay/vllm-kvpipe
```

## Toolchain present on the A100 cluster
```
nvcc:   ABSENT — this is why FlashInfer JIT and DirectKV were unmeasurable here
ptxas:  absent
gcc:    gcc (GCC) 8.5.0 20210514 (Red Hat 8.5.0-15)
```

## Two corrections to the raw output above

* **`pip list` reports vllm's location as `/mnt/beegfs/benishay/vllm-kvpipe`
  even from the `vllm-kvpipeline-66` worktree.** That is stale dist-info
  metadata: repointing the worktree venv rewrote the editable *finder*, which
  is what imports actually use, but not the recorded install path. Imports were
  verified directly (`vllm.__file__` resolves inside the worktree). Do not
  trust `pip list` to tell you which tree you are running.
* **`ptxas` is not on `PATH`, but it does exist** — shipped inside the nvidia
  pip wheels at
  `.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas`.
  So the A100 gap is specifically `nvcc`, not the whole toolchain.

## What the GH200 image must provide

| need | why |
|---|---|
| `nvcc` (CUDA `devel` image) | FlashInfer JIT-compiles attention kernels at runtime; DirectKV is SM90a CUTLASS/CuTe source. Verify with `nvcc --version` in minute one. |
| torch 2.9.1+cu128 or a deliberate, recorded upgrade | the A100 numbers were taken on it; changing it silently makes the comparison cross-substrate |
| disk for 72.7 GB of weights + the pinned host pool | the pool is tens of GB at high rider counts (`gh200_preflight.py` sizes it) |
