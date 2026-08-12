#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bitwise test for the host->resident restore path (promotion's foundation).

Promotion (host tier -> HBM tier) is only correct if the resident blocks hold
the request's FULL history afterwards. This checks the round trip:

    resident --seed_req--> host --restore_req--> resident

must be bitwise identical, including when the request's host blocks and its
resident blocks are at completely different indices (they always are), and when
the restore lands in blocks that previously held someone else's data.
"""

from __future__ import annotations

import sys

import torch


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from vllm.v1.hybrid_rungd_pool import RungdHostPool

    BS, NKV, HD = 16, 4, 64
    NBLK, K = 256, 2
    BPR = 8
    HMAX = BPR * 4
    layers = ["L0", "L1", "L2"]
    kv_shape = (2, NBLK, BS, NKV, HD)

    pool = RungdHostPool(layers, kv_shape, BS, K, HMAX, HMAX, dev,
                         torch.float16, blocks_per_req=BPR)
    resident = {
        ln: torch.randn(kv_shape, dtype=torch.float16, device=dev)
        for ln in layers
    }
    pool.resident = resident

    fails = []

    # Request occupies resident blocks scattered on purpose.
    res_ids = [37, 5, 200, 91, 12, 150, 63, 7]
    rid = "req-a"
    pool.ensure_req(rid, len(res_ids))
    truth = {ln: resident[ln].index_select(
        1, torch.tensor(res_ids, device=dev)).clone() for ln in layers}

    # 1) seed host from resident
    pool.seed_req(rid, res_ids)

    # 2) clobber the resident blocks (simulate reuse while host-tier)
    for ln in layers:
        idx = torch.tensor(res_ids, device=dev)
        resident[ln].index_copy_(
            1, idx, torch.randn_like(truth[ln]))
    clobbered = all(
        not torch.equal(
            resident[ln].index_select(1, torch.tensor(res_ids, device=dev)),
            truth[ln])
        for ln in layers
    )
    print(f"[1] resident blocks clobbered while host-tier: {clobbered}")
    if not clobbered:
        fails.append("clobber step did not actually change resident")

    # 3) restore into DIFFERENT resident blocks (as a real promotion would)
    new_ids = [101, 102, 3, 44, 77, 180, 21, 250]
    pool.restore_req(rid, new_ids)
    got = {ln: resident[ln].index_select(
        1, torch.tensor(new_ids, device=dev)) for ln in layers}
    ok = all(torch.equal(got[ln], truth[ln]) for ln in layers)
    print(f"[2] restore into fresh blocks is bitwise-exact: {ok}")
    if not ok:
        for ln in layers:
            if not torch.equal(got[ln], truth[ln]):
                d = (got[ln].float() - truth[ln].float()).abs().max().item()
                print(f"      layer {ln}: max|diff|={d}")
        fails.append("host->resident restore is not bitwise exact")

    # 4) a second request must be unaffected (no cross-request bleed)
    rid2 = "req-b"
    res2 = [9, 10, 11, 12]
    pool.ensure_req(rid2, len(res2))
    truth2 = {ln: resident[ln].index_select(
        1, torch.tensor(res2, device=dev)).clone() for ln in layers}
    pool.seed_req(rid2, res2)
    pool.restore_req(rid, new_ids)          # restore A again
    ok2 = all(
        torch.equal(
            resident[ln].index_select(1, torch.tensor(res2, device=dev)),
            truth2[ln])
        for ln in layers
    )
    print(f"[3] other request's resident blocks untouched: {ok2}")
    if not ok2:
        fails.append("restore bled into another request's blocks")

    # 5) after restore the request is no longer host-authoritative
    ok3 = rid not in pool.req_seeded
    print(f"[4] restored request cleared from req_seeded "
          f"(a later demotion re-seeds): {ok3}")
    if not ok3:
        fails.append("restored request still marked seeded")

    print("=" * 64)
    if fails:
        print("RESTORE ROUNDTRIP FAIL:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("RESTORE ROUNDTRIP PASS: host->resident restore is bitwise-exact "
          "into arbitrary blocks and isolated between requests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
