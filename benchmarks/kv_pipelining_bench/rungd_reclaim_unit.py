"""Unit test for the hybrid-KV prefill reclaim (paper/BLOCKER_PREFILL_RESIDENCY.md).

The blocker: capping a host-tier request's block GROWTH leaves its entire
prefill resident, so offloading it frees almost nothing. The fix is
`KVCacheManager.reclaim_tier_blocks` (free the whole block list, leaving null
placeholders) plus `refill_tier_blocks` (put real blocks back for promotion).

This exercises them against a REAL `KVCacheManager` -- no GPU, no server -- so
the allocator-level invariants are checked in seconds:

  1. reclaim actually returns the blocks to the pool (the whole point);
  2. it is idempotent, and the request's list keeps its length in nulls;
  3. `num_real_blocks` reports 0 afterwards, so the promotion gate cannot be
     satisfied by null placeholders;
  4. a reclaimed request's blocks are genuinely reusable by another request;
  5. refill restores every position, all-or-nothing under pool pressure;
  6. freeing a request that holds nulls does not double-free.

Run: .venv/bin/python benchmarks/kv_pipelining_bench/rungd_reclaim_unit.py
"""

import sys
from collections.abc import Callable

import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

BLOCK_SIZE = 16
FAILURES: list[str] = []


def check(cond: bool, what: str) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        FAILURES.append(what)


def make_kv_cache_config(num_blocks: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )


def make_request(request_id: str, num_tokens: int, hash_fn: Callable = sha256):
    # Distinct token ids per request so nothing collides in the prefix cache.
    salt = int(request_id.split("-")[-1]) * 100_000
    return Request(
        request_id=request_id,
        prompt_token_ids=[salt + i for i in range(num_tokens)],
        sampling_params=SamplingParams(max_tokens=17),
        pooling_params=None,
        eos_token_id=100,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, hash_fn),
    )


def new_manager(num_blocks: int, enable_caching: bool = True) -> KVCacheManager:
    return KVCacheManager(
        make_kv_cache_config(num_blocks),
        max_model_len=8192,
        enable_caching=enable_caching,
        hash_block_size=BLOCK_SIZE,
    )


def admit(manager: KVCacheManager, request: Request) -> int:
    """Run a request through prefill allocation; return blocks it now holds."""
    computed, num_computed = manager.get_computed_blocks(request)
    manager.allocate_slots(
        request,
        request.num_tokens - num_computed,
        new_computed_blocks=computed,
        num_lookahead_tokens=0,
    )
    request.num_computed_tokens = request.num_tokens
    return len(manager.get_block_ids(request.request_id)[0])


def test_reclaim_frees_the_prefill() -> None:
    print("\n1) reclaim returns a host-tier request's prefill to the pool")
    manager = new_manager(num_blocks=64)
    # 8 blocks of prefill -- the shape the blocker is about (long prompt,
    # short generation), where capping growth alone frees essentially nothing.
    request = make_request("req-1", 8 * BLOCK_SIZE)
    held = admit(manager, request)
    free_after_prefill = manager.block_pool.get_num_free_blocks()
    check(held == 8, f"request holds its 8 prefill blocks (got {held})")

    freed = manager.reclaim_tier_blocks("req-1")
    free_after_reclaim = manager.block_pool.get_num_free_blocks()
    check(freed == 8, f"reclaim reports 8 blocks freed (got {freed})")
    check(
        free_after_reclaim - free_after_prefill == 8,
        f"pool really regained 8 blocks "
        f"({free_after_prefill} -> {free_after_reclaim})",
    )
    check(
        len(manager.get_block_ids("req-1")[0]) == 8,
        "the block list keeps its length (positions are nulled, not dropped)",
    )
    check(
        manager.num_real_blocks("req-1") == 0,
        "num_real_blocks is 0, so the promotion gate cannot be fooled by nulls",
    )

    again = manager.reclaim_tier_blocks("req-1")
    check(again == 0, f"reclaim is idempotent (second call freed {again})")
    check(
        manager.block_pool.get_num_free_blocks() == free_after_reclaim,
        "the second reclaim did not double-free",
    )


def test_reclaimed_blocks_are_reusable() -> None:
    print("\n2) the reclaimed blocks are genuinely available to other requests")
    # A pool that fits exactly two 8-block requests plus vLLM's null block.
    manager = new_manager(num_blocks=17, enable_caching=False)
    first = make_request("req-1", 8 * BLOCK_SIZE)
    admit(manager, first)
    second = make_request("req-2", 8 * BLOCK_SIZE)
    admit(manager, second)
    check(
        manager.block_pool.get_num_free_blocks() == 0,
        "pool is exactly full with two resident requests",
    )

    third = make_request("req-3", 8 * BLOCK_SIZE)
    computed, num_computed = manager.get_computed_blocks(third)
    refused = manager.allocate_slots(
        third, third.num_tokens - num_computed, new_computed_blocks=computed
    )
    check(refused is None, "a third request is refused while the pool is full")

    # Offload req-1 to the host tier: this is the capacity the paper claims.
    manager.reclaim_tier_blocks("req-1")
    computed, num_computed = manager.get_computed_blocks(third)
    admitted = manager.allocate_slots(
        third, third.num_tokens - num_computed, new_computed_blocks=computed
    )
    check(
        admitted is not None,
        "after reclaiming one host-tier request, a THIRD request is admitted "
        "-- serve-more at a fixed pool",
    )
    third_ids = set(manager.get_block_ids("req-3")[0])
    first_ids = set(b for b in manager.get_block_ids("req-1")[0] if b != 0)
    check(
        not (third_ids & first_ids),
        "the new request's blocks do not overlap anything req-1 still claims",
    )


def test_refill_for_promotion() -> None:
    print("\n3) refill restores every nulled position, all-or-nothing")
    manager = new_manager(num_blocks=64)
    request = make_request("req-1", 8 * BLOCK_SIZE)
    admit(manager, request)
    original = list(manager.get_block_ids("req-1")[0])
    manager.reclaim_tier_blocks("req-1")

    ids = manager.refill_tier_blocks("req-1")
    check(ids is not None, "refill succeeds when the pool has room")
    assert ids is not None
    check(len(ids) == len(original), f"refill returns all 8 positions ({len(ids)})")
    check(all(b != 0 for b in ids), "no null placeholder is left behind")
    check(
        manager.num_real_blocks("req-1") == 8,
        "num_real_blocks is 8 again, so the promotion gate now passes",
    )
    check(
        ids == manager.get_block_ids("req-1")[0],
        "the returned row matches what the manager holds (what the runner "
        "overwrites its block table with)",
    )

    print("\n4) refill refuses rather than half-promoting a request")
    manager = new_manager(num_blocks=20, enable_caching=False)
    victim = make_request("req-1", 8 * BLOCK_SIZE)
    admit(manager, victim)
    manager.reclaim_tier_blocks("req-1")
    # Soak up the pool so a full refill cannot be satisfied.
    hog = make_request("req-2", 15 * BLOCK_SIZE)
    admit(manager, hog)
    free_before = manager.block_pool.get_num_free_blocks()
    check(free_before < 8, f"fewer than 8 blocks free ({free_before})")
    refused = manager.refill_tier_blocks("req-1")
    check(refused is None, "refill refuses when it cannot cover every hole")
    check(
        manager.block_pool.get_num_free_blocks() == free_before,
        "the refused refill consumed nothing (the request stays host-tier)",
    )
    check(
        manager.num_real_blocks("req-1") == 0,
        "the refused refill left no partially-filled row",
    )


def test_free_after_reclaim() -> None:
    print("\n5) finishing a reclaimed request does not double-free")
    manager = new_manager(num_blocks=64, enable_caching=False)
    request = make_request("req-1", 8 * BLOCK_SIZE)
    admit(manager, request)
    manager.reclaim_tier_blocks("req-1")
    free_before = manager.block_pool.get_num_free_blocks()
    manager.free(request)
    free_after = manager.block_pool.get_num_free_blocks()
    check(
        free_after == free_before,
        f"the request end returned nothing extra ({free_before} -> {free_after})",
    )
    # A fresh request of the same size must still be servable.
    nxt = make_request("req-2", 8 * BLOCK_SIZE)
    check(admit(manager, nxt) == 8, "the pool still hands out 8 blocks afterwards")


def main() -> int:
    init_none_hash(sha256)
    print("hybrid-KV prefill reclaim -- allocator unit test")
    test_reclaim_frees_the_prefill()
    test_reclaimed_blocks_are_reusable()
    test_refill_for_promotion()
    test_free_after_reclaim()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
