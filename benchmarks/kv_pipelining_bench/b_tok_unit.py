#!/usr/bin/env python3
"""B1's KV bytes/token must come from the model, not from a flag.

Every harness here passed `--hybrid-b-tok 131072` while running Qwen2.5-32B,
whose real figure is 262144 (64 layers x 8 KV heads x 128 head_dim x 2 (K,V)
x 2 bytes). B1 therefore charged each host-tier request HALF its interconnect
cost and would admit about twice as many riders as can hide inside a step --
silently, and in exactly the direction that breaks the ITL claim.

No GPU, no server: this drives the resolver against duck-typed configs.
"""
from __future__ import annotations

import logging
import sys

sys.path.insert(0, "/mnt/beegfs/benishay/vllm-kvpipe")

import torch

from vllm.v1.core.sched.transfer_aware_admission import (
    TransferAwareAdmissionController,
)


class _Model:
    def __init__(self, layers, kv_heads, head_dim, dtype=torch.bfloat16):
        self._l, self._h, self._d, self.dtype = layers, kv_heads, head_dim, dtype

    def get_num_layers(self, _pc):      return self._l
    def get_num_kv_heads(self, _pc):    return self._h
    def get_head_size(self):            return self._d


class _Cache:
    cache_dtype = "auto"


class _VllmConfig:
    def __init__(self, model):
        self.model_config = model
        self.parallel_config = object()
        self.cache_config = _Cache()


class _SchedCfg:
    def __init__(self, b_tok=0):
        self.hybrid_b_tok = b_tok
        self.hybrid_itl_ms = 45.0
        self.hybrid_pcie_gbps = 25.1
        self.hybrid_b_hbm_max = 6
        self.enable_hybrid_kv_tiers = True


QWEN32B = _VllmConfig(_Model(64, 8, 128))
FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: got {got}, want {want}")
    if not ok:
        FAILURES.append(name)


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(f"{record.levelname}:{record.getMessage()}")


def main() -> int:
    log = logging.getLogger("vllm.v1.core.sched.transfer_aware_admission")
    cap = _Capture()
    log.addHandler(cap)
    log.setLevel(logging.DEBUG)

    print("1. derived from the model, ignoring an absent flag")
    c = TransferAwareAdmissionController(_SchedCfg(0), QWEN32B)
    check("qwen32b b_tok", c.kv_bytes_per_token(), 262144)

    print("2. a WRONG explicit flag loses to the model, and is reported")
    cap.records.clear()
    c = TransferAwareAdmissionController(_SchedCfg(131072), QWEN32B)
    check("stale flag overridden", c.kv_bytes_per_token(), 262144)
    check("mismatch logged at ERROR",
          any(r.startswith("ERROR") and "disagrees" in r for r in cap.records),
          True)

    print("3. no model config -> keep the explicit value, do not guess")
    c = TransferAwareAdmissionController(_SchedCfg(131072), None)
    check("explicit kept", c.kv_bytes_per_token(), 131072)

    print("4. nothing at all -> documented fallback, loudly")
    cap.records.clear()
    c = TransferAwareAdmissionController(_SchedCfg(0), None)
    check("fallback", c.kv_bytes_per_token(), 32 * 2 * 8 * 128 * 2)
    check("fallback warned",
          any(r.startswith("WARNING") for r in cap.records), True)

    print("5. an 8B-class model really is 131072 (the flag was right ONCE)")
    c = TransferAwareAdmissionController(
        _SchedCfg(0), _VllmConfig(_Model(32, 8, 128)))
    check("8b b_tok", c.kv_bytes_per_token(), 131072)

    print("6. the rider budget this implies at a 45 ms step / 25.1 GB/s")
    from vllm.v1.hybrid_kv_roofline import max_offload_batch
    riders = max_offload_batch(2048, 45.0, 25.1, 262144)
    riders_wrong = max_offload_batch(2048, 45.0, 25.1, 131072)
    print(f"      correct b_tok admits {riders} riders at ctx 2048; "
          f"the stale flag would have admitted {riders_wrong}")
    check("stale flag doubles the admitted riders", riders_wrong, 2 * riders)

    print("=" * 66)
    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        return 1
    print("all b_tok checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
