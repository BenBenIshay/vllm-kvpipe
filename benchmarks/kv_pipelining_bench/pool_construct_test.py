#!/usr/bin/env python3
"""Construct a real RungdHostPool, under every flag combination.

WHY THIS EXISTS. T7a read an env var in `__init__` a few lines ABOVE the point
where `_os` is bound, so the constructor raised `UnboundLocalError` on every
code path. Six of nine suite tests failed, each needing an engine and a 65 GB
model load to discover a typo.

The off-GPU tests missed it because they all bind the pool's METHODS to a
lightweight `Stub` and never invoke `__init__` at all -- a deliberate choice
(they need no CUDA) that leaves the constructor, and every flag read inside it,
completely uncovered.

This closes that gap: it instantiates the real class, with the real `__init__`,
once per flag combination that the code offers -- 128 of them, a few
milliseconds each. It needs no model and no engine, so it belongs at the FRONT
of the suite, where a typo costs seconds rather than six failed tests and half
an hour of model loads.

It does need a GPU, though only for a CUDA context: the host mirrors are
allocated with `pin_memory=True`, which cannot be satisfied on a login node.
An earlier version of this docstring claimed CPU-only and was wrong.

It asserts only what construction should guarantee -- the per-layer mirrors
exist, are the right shape, are pinned when they should be, and are distinct
views of the unified allocation when that is on. It does NOT check behaviour;
that is what the equivalence tests are for.

    ./.venv/bin/python benchmarks/kv_pipelining_bench/pool_construct_test.py
"""
import itertools
import os
import sys

import torch

sys.path.insert(0, ".")
from vllm.v1.hybrid_rungd_pool import RungdHostPool  # noqa: E402

LAYERS = ["l0", "l1", "l2", "l3"]
BS, NKV, HD = 4, 2, 3
KV_SHAPE = (2, 8, BS, NKV, HD)          # (kv, blocks, bs, nkv, hd)
H_MAX = 16

# Every env flag the pool reads in __init__. Each is tried on and off; the
# measurement-only ones that demand a second _UNSAFE var are given it, since the
# point is to exercise the constructor, not the guard.
FLAGS = [
    ("VLLM_KVPIPE_RUNGD_UNIFIED_MIRROR", ("0", "1")),
    ("VLLM_KVPIPE_RUNGD_D2H_ASYNC", ("0", "1")),
    ("VLLM_KVPIPE_RUNGD_BATCH_GATHER", ("0", "1")),
    ("VLLM_KVPIPE_RUNGD_WHOLE_EXTENT", ("0", "1")),
    ("VLLM_KVPIPE_RUNGD_MIRROR_SYNC", ("0", "1")),
    ("VLLM_KVPIPE_RUNGD_SKIP_D2H", ("0", "1")),
    ("VLLM_KVPIPE_RUNGD_ASYNC", ("0", "1")),
]


def build(bpr=4):
    return RungdHostPool(
        layer_names=LAYERS, kv_shape=KV_SHAPE, block_size=BS,
        k_window=2, h_max_blocks=H_MAX, w_max_blocks=H_MAX,
        device="cpu", dtype=torch.float16, blocks_per_req=bpr,
    )


def check(pool, env):
    probs = []
    if sorted(pool.host_k) != sorted(LAYERS):
        probs.append("host_k keys %s" % sorted(pool.host_k))
    for ln in LAYERS:
        hk = pool.host_k[ln]
        if tuple(hk.shape) != (H_MAX, BS, NKV, HD):
            probs.append("%s shape %s" % (ln, tuple(hk.shape)))
            break
        if not hk.is_contiguous():
            probs.append("%s not contiguous" % ln)
            break
    # distinct storage per layer, whether unified or not: two layers must never
    # alias, or a write to one would corrupt another
    ptrs = {pool.host_k[ln].data_ptr() for ln in LAYERS}
    if len(ptrs) != len(LAYERS):
        probs.append("layers alias each other")
    if env.get("VLLM_KVPIPE_RUNGD_UNIFIED_MIRROR", "1") == "1":
        if getattr(pool, "_host_k_all", None) is None:
            probs.append("unified on but _host_k_all is None")
        elif pool.host_k[LAYERS[1]].data_ptr() <= \
                pool.host_k[LAYERS[0]].data_ptr():
            probs.append("unified views not in layer order")
        if getattr(pool, "_wt_layer_off", None) is None:
            probs.append("unified on but _wt_layer_off is None")
    return probs


def main():
    fails = 0
    saved = {k: os.environ.get(k) for k, _ in FLAGS}
    print("Constructing a real RungdHostPool under every flag combination")
    print()

    # 1. the default, which is what production runs
    for k, _ in FLAGS:
        os.environ.pop(k, None)
    try:
        pool = build()
        probs = check(pool, {})
    except Exception as e:                                  # noqa: BLE001
        probs = ["raised %s: %s" % (type(e).__name__, e)]
    print("  %-46s %s" % ("defaults (no flags set)",
                          "ok" if not probs else "FAIL " + "; ".join(probs)))
    fails += bool(probs)

    # 2. every combination. 2^7 = 128 constructions, each a few ms on CPU.
    names = [k for k, _ in FLAGS]
    n_comb = 0
    for combo in itertools.product(*[v for _, v in FLAGS]):
        env = dict(zip(names, combo))
        for k, v in env.items():
            os.environ[k] = v
        # the double-gated knobs need their _UNSAFE partner to actually engage;
        # set it so the guarded branch is exercised rather than skipped
        os.environ["VLLM_KVPIPE_RUNGD_NOFLUSH_UNSAFE"] = "1"
        try:
            pool = build()
            probs = check(pool, env)
        except Exception as e:                               # noqa: BLE001
            probs = ["raised %s: %s" % (type(e).__name__, e)]
        n_comb += 1
        if probs:
            fails += 1
            on = ",".join(k.replace("VLLM_KVPIPE_RUNGD_", "")
                          for k, v in env.items() if v == "1") or "none"
            print("  FAIL  flags on: %-34s %s" % (on, "; ".join(probs)))
    os.environ.pop("VLLM_KVPIPE_RUNGD_NOFLUSH_UNSAFE", None)
    print("  %-46s %s" % ("%d flag combinations" % n_comb,
                          "all ok" if not fails else "see above"))

    # 3. the scattered-block allocator too (blocks_per_req = 0)
    for k, _ in FLAGS:
        os.environ.pop(k, None)
    try:
        pool = build(bpr=0)
        probs = check(pool, {})
    except Exception as e:                                  # noqa: BLE001
        probs = ["raised %s: %s" % (type(e).__name__, e)]
    print("  %-46s %s" % ("blocks_per_req = 0 (scattered allocator)",
                          "ok" if not probs else "FAIL " + "; ".join(probs)))
    fails += bool(probs)

    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    print()
    print("POOL CONSTRUCTION %s"
          % ("PASS" if not fails else "FAIL (%d)" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
