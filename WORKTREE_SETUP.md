# vllm-kvpipeline-66 — how this tree is wired

A git worktree of `/mnt/beegfs/benishay/vllm-kvpipe` on branch `kvpipe-66`,
created 2026-09-05 so engine work can proceed while another session writes the
paper in the main checkout. Both share one `.git`, so commits and branches are
visible across them, but each has its own working tree and index — which is the
point: a `git add -A` in one can no longer sweep up the other's files.

## Why it has its OWN .venv (8.5 GB, copied not shared)

The venv installs vllm *editable*, and the finder hardcodes an absolute path:

    MAPPING = {'vllm': '/mnt/beegfs/benishay/vllm-kvpipe/vllm'}

It is a MetaPathFinder, consulted **before** `sys.path`, so `PYTHONPATH`
cannot override it — verified, not assumed. Sharing the main venv would mean
editing `vllm/` here, running the engine, and silently executing the OLD code
from the main tree. Runs would look healthy and measure the wrong thing, which
is the exact failure mode that produced two wrong diagnoses on 2026-09-04.

So this tree has its own venv with the finder repointed here. Use
`./.venv/bin/python` — never the main tree's.

Proven on 2026-09-05: a marker added to `vllm/v1/hybrid_pool_admit.py` was
visible to this venv and invisible to the main tree.

## What is SHARED (and therefore not isolated)

* **Compiled extensions are symlinks** into the main tree — ~2 GB of `.so`
  (`_vllm_fa3_C` alone is 1.1 GB). They are build outputs, identical for both.
  Consequence: **rebuilding the C extensions in the main tree changes this tree
  too.** Usually desirable; know it before debugging a mystery.
* **Generated Python** is symlinked the same way (`_version.py`, the grpc pb2
  files, `third_party/triton_kernels/`, `vllm_flash_attn/*.py`). If the build
  regenerates any of these, this tree follows.
* `HF_HUB_CACHE` and `/mnt/beegfs/benishay/paper_runs` are shared on purpose —
  no model re-downloads, and results land in one place.

## Gotchas

* **cwd matters.** Python puts the cwd on `sys.path`, so running the main
  tree's python from *this* directory (or vice versa) imports whichever `vllm/`
  is in the cwd, bypassing the finder entirely. Always run from the tree you
  mean, with that tree's venv. This bit me once while verifying the setup.
* beegfs does **not** support hard links, so `cp -al` for the venv is not an
  option; the copy is real.
* SLURM allocations are path-independent — the existing job works from here.
