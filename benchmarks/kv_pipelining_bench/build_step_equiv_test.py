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
        # T2 state. build_step now flushes deferred mirror writes first, so the
        # stand-in has to carry enough for that call to be real rather than
        # stubbed out -- otherwise the test would pass while the ordering it is
        # supposed to protect was broken.
        self._wt_dirty = False
        self._wt_noflush = False
        self._wt_pending = []
        self._wt_k = None
        self._wt_v = None
        self._wt_event = None
        self.wt_flushes = 0
        self.wt_deferred_rows = 0
        self.host_k = {}
        self.host_v = {}
        # T3 state: build_step now also plans the batched gather.
        self.h_max = w_max
        self._batch_plan = None
        self._batch_gather = True
        self.batched_steps = 0
        self.unbatched_steps = 0
        # T5 state, flag OFF. These 18 cases are the reference for the PER-RUN
        # numbering, so this file is what proves T5 changes nothing when it is
        # not enabled; the whole-extent path has its own test
        # (whole_extent_equiv_test.py) because it deliberately renumbers the
        # compact region and so cannot be checked against this reference.
        self._whole_extent = False
        self._whole_extent_max_slack = 0.5
        self._bpr = 0
        self._req_extent = {}
        self.whole_extent_steps = 0
        self.whole_extent_refused = 0
        # T6 state. flush_token_writes now orders itself after the mirror's
        # READERS, so the stand-in needs both handles for that call to be real
        # rather than stubbed -- the ordering is the correctness fix, and a test
        # that skipped it would pass while the thing it protects was broken.
        self._copy_stream = None
        self._mirror_ev = None
        # Set explicitly even though the two guards above short-circuit before
        # reading it: relying on short-circuit order would make this stub pass
        # for a reason unrelated to what it is testing.
        self._no_mirror_sync = False

    # bind the real implementations. `_active_host` is a property, so it has
    # to be a CLASS attribute to work at all -- which is itself worth asserting,
    # since it was a plain attribute before the rewrite.
    build_step = RungdHostPool.build_step
    _plan_extents = RungdHostPool._plan_extents
    compact_of = RungdHostPool.compact_of
    _active_host = RungdHostPool._active_host
    flush_token_writes = RungdHostPool.flush_token_writes
    _plan_batched_gather = RungdHostPool._plan_batched_gather


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

    # ---------------------------------------------------------------- T2
    # The deferred mirror write. The async copy needs CUDA, but the SCATTER is
    # pure CPU, so the part that can be wrong on the host is testable here:
    # whether the deferred rows land in the right places, whether the flush is
    # idempotent, and whether build_step performs it BEFORE anything could read
    # the mirror -- which is the entire correctness argument for T2.
    bs_t, nkv, hd, L = 4, 2, 3, 3
    pool = Stub(extents({"a": (0, 2), "b": (1, 2)}, 4), block_size=bs_t)
    pool.layers = ["l%d" % i for i in range(L)]
    for ln in pool.layers:
        pool.host_k[ln] = torch.zeros(8, bs_t, nkv, hd)
        pool.host_v[ln] = torch.zeros(8, bs_t, nkv, hd)
    pool._wt_k = torch.zeros(L, 2, nkv, hd)
    pool._wt_v = torch.zeros(L, 2, nkv, hd)

    t2_fail = 0

    # a flush with nothing pending must do nothing and claim nothing
    if pool.flush_token_writes() != 0 or pool.wt_flushes != 0:
        t2_fail += 1
        print("  FAIL %-32s flush was not a no-op when clean" % "T2 idempotent")
    else:
        print("  ok   %-32s no-op when nothing pending" % "T2 idempotent")

    # stage distinguishable values and defer them
    for li in range(L):
        pool._wt_k[li, 0].fill_(100 + li)
        pool._wt_k[li, 1].fill_(200 + li)
        pool._wt_v[li, 0].fill_(-(100 + li))
        pool._wt_v[li, 1].fill_(-(200 + li))
        # rider a's token at flat row 1, rider b's at flat row 5
        pool._wt_pending.append((li, torch.tensor([1, 5]), 2))
    pool._wt_dirty = True

    rows = pool.flush_token_writes()
    probs = []
    if rows != L * 2:
        probs.append("applied %d rows, expected %d" % (rows, L * 2))
    for li, ln in enumerate(pool.layers):
        flat_k = pool.host_k[ln].view(8 * bs_t, nkv, hd)
        flat_v = pool.host_v[ln].view(8 * bs_t, nkv, hd)
        if not torch.all(flat_k[1] == 100 + li):
            probs.append("layer %d row 1 K wrong" % li)
        if not torch.all(flat_k[5] == 200 + li):
            probs.append("layer %d row 5 K wrong" % li)
        if not torch.all(flat_v[1] == -(100 + li)):
            probs.append("layer %d row 1 V wrong" % li)
        # every other row must be untouched
        touched = [r for r in range(8 * bs_t)
                   if r not in (1, 5) and flat_k[r].abs().sum() != 0]
        if touched:
            probs.append("layer %d scattered into %s" % (li, touched[:4]))
    if pool._wt_dirty or pool._wt_pending:
        probs.append("still dirty after flushing")
    if probs:
        t2_fail += 1
        print("  FAIL %-32s %s" % ("T2 deferred scatter", "; ".join(probs)))
    else:
        print("  ok   %-32s %d rows over %d layers, nothing else touched"
              % ("T2 deferred scatter", rows, L))

    # a second flush must be a no-op, not a re-apply
    again = pool.flush_token_writes()
    if again != 0:
        t2_fail += 1
        print("  FAIL %-32s re-applied %d rows" % ("T2 flush twice", again))
    else:
        print("  ok   %-32s second flush applied nothing" % "T2 flush twice")

    # AND THE ORDERING: build_step must flush before it does anything else,
    # because _issue_runs reads the mirror host-to-device later in the step.
    pool._wt_pending.append((0, torch.tensor([2]), 1))
    pool._wt_k[0, 0].fill_(777.0)
    pool._wt_dirty = True
    pool.build_step(["a", "b"], [8, 8])
    flat = pool.host_k[pool.layers[0]].view(8 * bs_t, nkv, hd)
    if pool._wt_dirty or not torch.all(flat[2] == 777.0):
        t2_fail += 1
        print("  FAIL %-32s build_step did not flush first" % "T2 ordering")
    else:
        print("  ok   %-32s build_step flushed before proceeding"
              % "T2 ordering")

    fails += t2_fail

    print()
    print("BUILD-STEP EQUIVALENCE %s"
          % ("PASS (%d cases)" % (len(CASES) + 6) if not fails
             else "FAIL (%d)" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
