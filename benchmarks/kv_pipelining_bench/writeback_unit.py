#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does the swap-semantics emulation move the bytes a swap design would?

The paper claims that a *swap*-based offload pays a return leg a *streaming*
one does not: if the offloaded unit is OWNED by one side, a unit that has just
had this token's K,V appended is dirty and must be persisted before its
device-side buffer is reused. That claim currently rests on reading another
system's swap protocol. `RungdHostPool.writeback_layer` lets us measure it on
our own hardware instead -- but only if the emulation is faithful, which is
what this test checks. Three properties, none of which are obvious:

  1. DISABLED BY DEFAULT and a genuine no-op. If the flag off still cost
     something, every number we have already published would be suspect.
  2. FAITHFUL: it writes back the SAME extent the refresh read in -- same runs,
     same granularity, opposite direction. An emulation that wrote less would
     understate the cost we are trying to attribute to swap semantics, which
     would flatter us.
  3. LOSSLESS: writing a slot back and re-reading it is the identity when the
     slot is unmodified. A writeback that corrupted the mirror would turn a
     measurement arm into a correctness bug.

CPU-only for the tensors' logic, but needs a GPU for the rolling region.

  srun ... .venv/bin/python benchmarks/kv_pipelining_bench/writeback_unit.py
"""

from __future__ import annotations

import functools
import os
import sys

print = functools.partial(print, flush=True)  # noqa: A001

sys.path.insert(0, "/mnt/beegfs/benishay/vllm-kvpipe")

LAYERS = ["l0", "l1", "l2", "l3"]
BS, NKV, HD = 16, 2, 8
K_WIN, H_MAX, W_MAX = 2, 64, 32


def _pool(writeback: bool):
    os.environ["VLLM_KVPIPE_RUNGD_WRITEBACK"] = "1" if writeback else "0"
    import importlib

    import torch
    import vllm.v1.hybrid_rungd_pool as m
    importlib.reload(m)
    return m.RungdHostPool(
        layer_names=LAYERS,
        kv_shape=(2, H_MAX, BS, NKV, HD),
        block_size=BS, k_window=K_WIN, h_max_blocks=H_MAX, w_max_blocks=W_MAX,
        device="cuda", dtype=torch.float16,
    ), torch


def _seed(pool, torch, reqs):
    """Give each request blocks and fill its host mirror deterministically."""
    for rid, nblk in reqs:
        pool.ensure_req(rid, nblk)
    for li, ln in enumerate(LAYERS):
        for rid, _ in reqs:
            for j, hb in enumerate(pool.req_blocks[rid]):
                v = float(li + 1) * 100 + hash(rid) % 7 * 10 + j
                pool.host_k[ln][hb].fill_(v)
                pool.host_v[ln][hb].fill_(-v)


def main() -> int:
    fails = []

    # ---- 1. off by default is a true no-op -------------------------------
    pool, torch = _pool(writeback=False)
    reqs = [("A", 2), ("B", 1), ("C", 3)]
    _seed(pool, torch, reqs)
    pool.build_step(["A", "B", "C"], [BS * 2, BS, BS * 3])
    before = pool.host_k["l0"].clone()
    pool.stream_layer(0)
    # corrupt the slot: a faithful no-op must NOT propagate this to the mirror
    pool.rolling_k[0].fill_(999.0)
    moved = pool.writeback_layer(0)
    if moved != 0:
        fails.append(f"disabled writeback moved {moved} bytes")
    if not torch.equal(before, pool.host_k["l0"]):
        fails.append("disabled writeback MUTATED the host mirror")
    print(f"1. disabled: moved={moved} mirror_unchanged="
          f"{torch.equal(before, pool.host_k['l0'])}")

    # ---- 2. enabled moves exactly the refreshed extent --------------------
    pool, torch = _pool(writeback=True)
    _seed(pool, torch, reqs)
    pool.build_step(["A", "B", "C"], [BS * 2, BS, BS * 3])
    w = pool._w_cur
    pool.stream_layer(0)
    moved = pool.writeback_layer(0)
    # 6 blocks x BS x NKV x HD x 2 tensors (K and V) x 2 bytes (fp16)
    expect = w * BS * NKV * HD * 2 * 2
    ok2 = moved == expect
    if not ok2:
        fails.append(f"writeback moved {moved} bytes, expected {expect}")
    print(f"2. enabled:  W={w} moved={moved} expected={expect} ok={ok2}")

    # symmetry: the same bytes the refresh brought in
    read_bytes = sum(n for _, _, n in pool._runs) * BS * NKV * HD * 2 * 2
    if read_bytes != moved:
        fails.append(f"asymmetric: read {read_bytes} wrote {moved}")
    print(f"   refresh read {read_bytes} bytes; writeback wrote {moved}")

    # ---- 3. round trip is the identity -----------------------------------
    pool, torch = _pool(writeback=True)
    _seed(pool, torch, reqs)
    pool.build_step(["A", "B", "C"], [BS * 2, BS, BS * 3])
    for ln in LAYERS:
        pass
    golden = {ln: (pool.host_k[ln].clone(), pool.host_v[ln].clone())
              for ln in LAYERS}
    for li in range(len(LAYERS)):
        pool.stream_layer(li)
        pool.writeback_layer(li)
    torch.cuda.synchronize()
    exact = all(torch.equal(golden[ln][0], pool.host_k[ln])
                and torch.equal(golden[ln][1], pool.host_v[ln])
                for ln in LAYERS)
    if not exact:
        fails.append("round trip changed the host mirror")
    print(f"3. round trip is identity: {exact}")

    # ---- 4. accumulator, for harness reporting ---------------------------
    print(f"4. writeback_bytes accumulator = {pool.writeback_bytes}")
    if pool.writeback_bytes <= 0:
        fails.append("accumulator not updated")

    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        return 1
    print("PASS - emulation is off by default, faithful in extent, and lossless")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
