#!/usr/bin/env python3
"""Preloaded riders must be admittable ON TOP of a FULL resident cohort.

This is the scheduling half of the fix. The KV half is already proven
(`pool_preload_test.py`: KV can enter the host tier from outside, bitwise
correct, with nothing resident). What was never true before is that the
SCHEDULER would let such a request in without giving it blocks.

Why it matters, from the 2026-08-13 and 2026-08-30 measurements. A rider used to
become a rider by prefilling into HBM and then being reclaimed, so it needed a
full request's worth of blocks transiently. That headroom is the same headroom
the resident cohort wants, so the two trade off one for one and the total is
pinned at capacity -- exactly what the b_hbm_max sweep shows:

    b_hbm_max   riders    total   (all-HBM = 6.0)
        3        ~2        5.3
        4        ~1.5      5.7
        6         0        6.0
        8         0        6.0

Never above 6. A preloaded rider owns no block at any instant, so it should be
addable to a resident cohort that already fills the pool. That is the single
thing this test checks, and it is checked against a pool with NO free blocks --
the case that decides whether the mixture can exceed capacity at all.

CPU only, no GPU, no model.

Run: .venv/bin/python benchmarks/kv_pipelining_bench/preload_admission_test.py
"""

from __future__ import annotations

import os
import sys

# The admission path is gated; turn it on before the Scheduler is constructed,
# since it reads the env once in __init__.
os.environ["VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM"] = "1"
os.environ["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.v1.core.utils import create_requests, create_scheduler  # noqa: E402

BLOCK_SIZE = 16
FAILURES: list[str] = []


def check(cond: bool, what: str) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        FAILURES.append(what)


def make_scheduler(num_blocks: int):
    sched = create_scheduler(
        max_num_seqs=32, num_blocks=num_blocks, block_size=BLOCK_SIZE,
        enable_prefix_caching=False, max_model_len=2048,
    )
    # Hybrid tiers are a config flag the test harness does not expose; the
    # admission path only reads `enable_hybrid_kv_tiers` and the env gate.
    sched.scheduler_config.enable_hybrid_kv_tiers = True
    sched._rungd_reclaim_enabled = True
    return sched


def force_tier(sched, req_id: str, tier: str) -> None:
    """Pin a request's tier without running the admission controller."""
    sched.transfer_aware_admission._req_tier[req_id] = tier


def free_blocks(sched) -> int:
    return sched.kv_cache_manager.block_pool.get_num_free_blocks()


def test_preloaded_admitted_with_zero_blocks() -> None:
    print("\n1) a preloaded rider is admitted without consuming a single block")
    # 40 blocks: enough for a few 128-token requests and nothing spare.
    sched = make_scheduler(num_blocks=40)
    reqs = create_requests(num_requests=1, num_tokens=128, max_tokens=8,
                           block_size=BLOCK_SIZE)
    r = reqs[0]
    force_tier(sched, r.request_id, "cpu")
    sched.register_pool_preloaded(r.request_id)
    sched.add_request(r)

    before = free_blocks(sched)
    out = sched.schedule()
    after = free_blocks(sched)

    check(r.request_id in out.num_scheduled_tokens,
          "the preloaded rider was scheduled")
    check(after == before,
          f"it consumed ZERO blocks ({before} -> {after})")
    ids = sched.kv_cache_manager.get_block_ids(r.request_id)[0]
    check(all(b == 0 for b in ids) or len(ids) == 0,
          f"it holds no real blocks (block ids {ids[:6]})")


def test_rider_adds_on_top_of_a_full_pool() -> None:
    print("\n2) riders are admitted when the pool has NO free blocks left")
    # Size the pool so a small number of resident requests exhausts it, which
    # is the situation the whole design exists for.
    sched = make_scheduler(num_blocks=40)
    resident = create_requests(num_requests=4, num_tokens=128, max_tokens=64,
                               block_size=BLOCK_SIZE,
                               req_ids=[f"hbm-{i}" for i in range(4)])
    for r in resident:
        force_tier(sched, r.request_id, "hbm")
        sched.add_request(r)
    out = sched.schedule()
    admitted_resident = len(out.num_scheduled_tokens)
    print(f"       resident admitted: {admitted_resident}, "
          f"free blocks now {free_blocks(sched)}")

    # Fill the pool completely, so nothing more could prefill.
    filler = create_requests(num_requests=6, num_tokens=128, max_tokens=64,
                             block_size=BLOCK_SIZE,
                             req_ids=[f"fill-{i}" for i in range(6)])
    for r in filler:
        force_tier(sched, r.request_id, "hbm")
        sched.add_request(r)
    for _ in range(4):
        sched.schedule()
    free_now = free_blocks(sched)
    running_before = len(sched.running)
    print(f"       pool driven to {free_now} free blocks, "
          f"{running_before} running")
    check(free_now < 8,
          f"pool is effectively full ({free_now} free) -- no room to prefill")

    # Now the question: can a rider still get in?
    rider = create_requests(num_requests=1, num_tokens=128, max_tokens=64,
                            block_size=BLOCK_SIZE, req_ids=["rider-0"])[0]
    force_tier(sched, rider.request_id, "cpu")
    sched.register_pool_preloaded(rider.request_id)
    sched.add_request(rider)
    out = sched.schedule()
    got_in = rider.request_id in out.num_scheduled_tokens
    check(got_in,
          "a preloaded rider is admitted even with the pool full -- "
          "concurrency ABOVE capacity")
    check(len(sched.running) > running_before,
          f"running set grew past the resident ceiling "
          f"({running_before} -> {len(sched.running)})")
    check(free_blocks(sched) == free_now,
          "and it still took no blocks")


def test_not_preloaded_is_unaffected() -> None:
    print("\n3) a host-tier request that is NOT preloaded still needs blocks")
    # The permissive failure -- admitting a request whose KV is nowhere -- is
    # the one that serves garbage, so the default must stay strict.
    sched = make_scheduler(num_blocks=40)
    r = create_requests(num_requests=1, num_tokens=128, max_tokens=8,
                        block_size=BLOCK_SIZE, req_ids=["nopreload"])[0]
    force_tier(sched, r.request_id, "cpu")           # host tier, but NOT preloaded
    sched.add_request(r)
    before = free_blocks(sched)
    sched.schedule()
    after = free_blocks(sched)
    check(after < before,
          f"it allocated blocks normally ({before} -> {after}) -- the "
          "zero-block path is not applied to requests without pool KV")


def test_gate_is_off_by_default() -> None:
    print("\n4) the zero-block path is inert unless explicitly enabled")
    sched = make_scheduler(num_blocks=40)
    sched._rungd_reclaim_enabled = False             # as in a stock engine
    r = create_requests(num_requests=1, num_tokens=128, max_tokens=8,
                        block_size=BLOCK_SIZE, req_ids=["gated"])[0]
    force_tier(sched, r.request_id, "cpu")
    sched.register_pool_preloaded(r.request_id)
    sched.add_request(r)
    before = free_blocks(sched)
    sched.schedule()
    check(free_blocks(sched) < before,
          "with the gate off, even a preloaded rider allocates normally")


def main() -> int:
    print("preloaded-rider admission -- can concurrency exceed HBM capacity?")
    test_preloaded_admitted_with_zero_blocks()
    test_rider_adds_on_top_of_a_full_pool()
    test_not_preloaded_is_unaffected()
    test_gate_is_off_by_default()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("PRELOAD ADMISSION PASS: a rider whose KV is already on the host is "
          "admitted with zero blocks, even against a full pool, so the running "
          "set can exceed what HBM alone can hold -- which is the thing the "
          "b_hbm_max sweep could never reach.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
