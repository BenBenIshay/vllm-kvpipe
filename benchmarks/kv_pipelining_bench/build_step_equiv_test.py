#!/usr/bin/env python3
"""build_step must be BITWISE identical to the implementation it replaced.

The rewrite turns the block-table rebuild from O(r * S/bs) Python tensor
element assignments into O(r) slice assignments, by using the guarantee that
`ensure_req` gives each request one contiguous extent. That is a real change to
a hot path that feeds attention, so equivalence is asserted rather than argued:
this file carries the PRE-CHANGE algorithm verbatim as a reference and compares
against it.

No GPU. The pool's constructor needs CUDA for its pinned host mirrors, but
`build_step` is pure CPU logic, so the real methods are bound to a lightweight
stand-in carrying only the attributes they touch. That keeps the test runnable
when the GPU is busy -- and the same technique already caught a wiring bug in
the purity gate that a hand-built-input test had missed.

    ./.venv/bin/python benchmarks/kv_pipelining_bench/build_step_equiv_test.py
"""
import sys

import torch

sys.path.insert(0, ".")
from vllm.v1.hybrid_rungd_pool import RungdHostPool, _runs_from_sorted  # noqa: E402


# --------------------------------------------------------------------------
# The pre-change algorithm, copied verbatim from the commit before the rewrite
# (a90ed1b49:vllm/v1/hybrid_rungd_pool.py). Do not "improve" this: its whole
# job is to be what shipped.
# --------------------------------------------------------------------------
def reference(req_blocks, active_req_ids, block_size):
    host_set = set()
    for r in active_req_ids:
        host_set.update(req_blocks.get(r, []))
    active_host = sorted(host_set)
    compact = {b: i for i, b in enumerate(active_host)}
    w_cur = len(active_host)
    runs = []
    if active_host:
        start = prev = active_host[0]
        cstart = 0
        for i, hb in enumerate(active_host[1:], 1):
            if hb != prev + 1:
                runs.append((start, cstart, prev - start + 1))
                start, cstart = hb, i
            prev = hb
        runs.append((start, cstart, prev - start + 1))
    max_blk = max((len(req_blocks.get(r, [])) for r in active_req_ids),
                  default=1)
    bt = torch.zeros(len(active_req_ids), max_blk, dtype=torch.int32)
    for row, r in enumerate(active_req_ids):
        for j, hb in enumerate(req_blocks.get(r, [])):
            bt[row, j] = compact[hb]
    return bt, runs, w_cur, compact


class Stub:
    """Only what build_step / _plan_extents / compact_of actually read."""

    def __init__(self, req_blocks, block_size=16, w_max=4096, layers=4):
        self.req_blocks = dict(req_blocks)
        self.block_size = block_size
        self.w_max = w_max
        self.layers = ["l%d" % i for i in range(layers)]
        self._xfer_mode = "real"
        self._runs = []
        self._compact = {}
        self._req_cbase = {}
        self._w_cur = 0
        self._events = {}
        self._ar = torch.arange(w_max, dtype=torch.int32)

    # bind the real implementations. `_active_host` is a property, so it has
    # to be a CLASS attribute to work at all -- which is itself worth asserting,
    # since it was a plain attribute before the rewrite.
    build_step = RungdHostPool.build_step
    _plan_extents = RungdHostPool._plan_extents
    compact_of = RungdHostPool.compact_of
    _active_host = RungdHostPool._active_host


def extents(spec, bpr):
    """{req: [base, base+1, ...]} with one contiguous extent each, as the
    allocator produces (`base = extent_index * blocks_per_req`)."""
    return {rid: [slot * bpr + i for i in range(n)]
            for rid, (slot, n) in spec.items()}


CASES = []


def case(name, req_blocks, active, expect_fast, bs=16):
    CASES.append((name, req_blocks, active, expect_fast, bs))


# (a) the normal case: contiguous extents, equal lengths -> fast path
case("contiguous, equal lengths",
     extents({"a": (0, 8), "b": (1, 8), "c": (2, 8)}, 32), ["a", "b", "c"], True)

# (b) ragged lengths -> the padding must stay zero exactly where a request
#     owns fewer blocks, which is the case most likely to break
case("contiguous, ragged lengths",
     extents({"a": (0, 3), "b": (1, 9), "c": (2, 1)}, 32), ["a", "b", "c"], True)

# (c) ADJACENT extents must merge into ONE run, or _issue_runs issues more
#     copies than it needs to
case("adjacent extents merge",
     {"a": list(range(0, 8)), "b": list(range(8, 16))}, ["a", "b"], True)

# (d) active order != allocation order: the compact numbering follows the
#     SORTED host blocks, not the cohort order
case("active order reversed",
     extents({"a": (0, 4), "b": (1, 4), "c": (2, 4)}, 32), ["c", "a", "b"], True)

# (e) a subset of the allocated requests is active
case("subset active",
     extents({"a": (0, 4), "b": (1, 4), "c": (2, 4)}, 32), ["b"], True)

# (f) single request
case("single request", extents({"a": (3, 5)}, 32), ["a"], True)

# (g) empty cohort
case("empty cohort", {}, [], True)

# (h) FALLBACK: free-list allocation, blocks scattered
case("fragmented free list",
     {"a": [7, 2, 19, 3], "b": [11, 5, 6]}, ["a", "b"], False)

# (i) FALLBACK: a cohort member with no blocks at all
case("member with no blocks",
     {"a": list(range(0, 4)), "b": []}, ["a", "b"], False)

# (j) FALLBACK: a request whose blocks are contiguous but DESCENDING
case("descending blocks",
     {"a": [5, 4, 3, 2]}, ["a"], False)

# (k) FALLBACK: overlapping extents. Should never happen, but the old code
#     took a set union and this one sums lengths, so they would disagree.
case("overlapping extents",
     {"a": list(range(0, 8)), "b": list(range(4, 12))}, ["a", "b"], False)

# (l) a large context, where the old path did 2048 tensor __setitem__ calls
case("ctx 8192 equivalent, 4 riders",
     extents({"r%d" % i: (i, 512) for i in range(4)}, 512),
     ["r%d" % i for i in range(4)], True)


def main():
    fails = 0
    print("build_step equivalence against the pre-change implementation")
    print()
    for name, req_blocks, active, expect_fast, bs in CASES:
        want_bt, want_runs, want_w, want_compact = reference(
            req_blocks, active, bs)
        pool = Stub(req_blocks, block_size=bs)
        seq = [bs * len(req_blocks.get(r, [])) for r in active]
        got_bt, got_seq = pool.build_step(active, seq)

        prob = []
        if got_bt.shape != want_bt.shape:
            prob.append("shape %s != %s" % (tuple(got_bt.shape),
                                            tuple(want_bt.shape)))
        elif not torch.equal(got_bt, want_bt):
            prob.append("block table differs")
        if got_bt.dtype != want_bt.dtype:
            prob.append("dtype %s != %s" % (got_bt.dtype, want_bt.dtype))
        if pool._w_cur != want_w:
            prob.append("W %d != %d" % (pool._w_cur, want_w))
        if pool._runs != want_runs:
            prob.append("runs %s != %s" % (pool._runs, want_runs))

        took_fast = bool(pool._req_cbase) or (not req_blocks)
        if took_fast != expect_fast:
            prob.append("took %s path, expected %s"
                        % ("fast" if took_fast else "general",
                           "fast" if expect_fast else "general"))

        # compact_of must agree at EVERY token position of every active request
        for r in active:
            nblk = len(req_blocks.get(r, []))
            for pos in range(nblk * bs):
                try:
                    got = pool.compact_of(r, pos)
                except Exception as e:                  # noqa: BLE001
                    prob.append("compact_of(%s,%d) raised %s" % (r, pos, e))
                    break
                exp = (want_compact[req_blocks[r][pos // bs]], pos % bs)
                if got != exp:
                    prob.append("compact_of(%s,%d) %s != %s" % (r, pos, got, exp))
                    break

        # the runs must tile [0, W) exactly, which is what _issue_runs assumes
        covered = sorted(cs for _hs, cs, _ln in pool._runs)
        tot = sum(ln for _hs, _cs, ln in pool._runs)
        if tot != pool._w_cur:
            prob.append("runs cover %d blocks, W is %d" % (tot, pool._w_cur))
        acc = 0
        for hs, cs, ln in pool._runs:
            if cs != acc:
                prob.append("run compact_start %d != running total %d" % (cs, acc))
                break
            acc += ln

        if prob:
            fails += 1
            print("  FAIL %-32s %s" % (name, "; ".join(prob)))
        else:
            print("  ok   %-32s %s path, W=%-5d runs=%-3d bt=%s"
                  % (name, "fast" if took_fast else "general", pool._w_cur,
                     len(pool._runs), tuple(got_bt.shape)))

    # _active_host must still reconstruct, since it is now a derived property
    pool = Stub(extents({"a": (0, 4), "b": (2, 4)}, 8))
    pool.build_step(["a", "b"], [64, 64])
    want = sorted(set([0, 1, 2, 3] + [16, 17, 18, 19]))
    if pool._active_host != want:
        fails += 1
        print("  FAIL %-32s %s != %s" % ("_active_host reconstructs",
                                         pool._active_host, want))
    else:
        print("  ok   %-32s %s" % ("_active_host reconstructs", want))

    # and the setter must still work, since __init__ and callers may use it
    pool._active_host = [5, 6, 7, 20]
    if pool._runs != _runs_from_sorted([5, 6, 7, 20]):
        fails += 1
        print("  FAIL %-32s" % "_active_host setter")
    else:
        print("  ok   %-32s %s" % ("_active_host setter", pool._runs))

    print()
    print("BUILD-STEP EQUIVALENCE %s"
          % ("PASS (%d cases)" % (len(CASES) + 2) if not fails
             else "FAIL (%d)" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
