#!/usr/bin/env python3
"""T3's batched gather must move BITWISE-identical bytes to the per-run loop.

The per-layer gather copies each run from the host mirror into the rolling slot.
T3 replaces that loop with ONE strided copy per tensor when the runs are evenly
spaced and equal length, cutting launches from 2*L*r to 2*L. Getting the stride
arithmetic wrong would stream the wrong KV **with no error raised**, so this
asserts the two produce identical slots rather than arguing the views are right.

CPU only: the copy is a plain `torch.Tensor.copy_` between views, so the index
arithmetic is testable without CUDA. What this does NOT cover is the
`non_blocking=True` behaviour on a real device, which the suite covers.

    ./.venv/bin/python benchmarks/kv_pipelining_bench/batch_gather_equiv_test.py
"""
import sys

import torch

sys.path.insert(0, ".")
from vllm.v1.hybrid_rungd_pool import RungdHostPool  # noqa: E402

BS, NKV, HD = 4, 2, 3


class Stub:
    def __init__(self, h_max, runs, w_max=512):
        self.h_max = h_max
        self.w_max = w_max
        self._runs = list(runs)
        self._batch_plan = None
        self._batch_gather = True
        self.batched_steps = 0
        self.unbatched_steps = 0

    _plan_batched_gather = RungdHostPool._plan_batched_gather
    _strided_views = RungdHostPool._strided_views


def per_run_gather(host, slot, runs):
    """The pre-T3 loop, which is the reference."""
    out = slot.clone()
    for hs, cs, n in runs:
        out.narrow(0, cs, n).copy_(host.narrow(0, hs, n))
    return out


def batched_gather(pool, host, slot):
    out = slot.clone()
    dst, src = pool._strided_views(host, out)
    dst.copy_(src)
    return out


CASES = [
    # (label, h_max, runs, expect_batched)
    # runs are (host_start, compact_start, length)
    ("3 riders, stride 10, len 4",
     64, [(0, 0, 4), (10, 4, 4), (20, 8, 4)], True),
    ("8 riders, stride 141, len 33 (ctx 512 shape)",
     1200, [(i * 141, i * 33, 33) for i in range(8)], True),
    ("2 riders, minimum batchable",
     64, [(5, 0, 6), (20, 6, 6)], True),
    ("stride == length (adjacent, should have merged)",
     64, [(0, 0, 4), (4, 4, 4)], True),
    # --- must NOT batch ---
    ("single run", 64, [(3, 0, 5)], False),
    ("unequal lengths", 64, [(0, 0, 4), (10, 4, 6)], False),
    ("uneven spacing", 64, [(0, 0, 4), (10, 4, 4), (25, 8, 4)], False),
    ("compact starts not cumulative", 64, [(0, 0, 4), (10, 9, 4)], False),
    ("overlapping sources", 64, [(0, 0, 8), (4, 8, 8)], False),
    # h_max 20 makes the last window end at 24, past the mirror. (At h_max 24
    # it ends exactly AT the mirror and is correctly batched -- my first
    # version of this case asserted the opposite and the code was right.)
    ("last window past the mirror", 20, [(0, 0, 8), (16, 8, 8)], False),
]


def main():
    fails = 0
    print("T3 batched gather vs the per-run loop")
    print()
    for label, h_max, runs, expect in CASES:
        pool = Stub(h_max, runs)
        plan = pool._plan_batched_gather()
        got = plan is not None
        if got != expect:
            fails += 1
            print("  FAIL %-46s planned=%s expected=%s"
                  % (label, got, expect))
            continue
        if not got:
            print("  ok   %-46s correctly NOT batched" % label)
            continue

        pool._batch_plan = plan
        host = torch.arange(
            h_max * BS * NKV * HD, dtype=torch.float32
        ).view(h_max, BS, NKV, HD)
        w = sum(n for _, _, n in runs)
        slot = torch.full((pool.w_max, BS, NKV, HD), -1.0)

        want = per_run_gather(host, slot, runs)
        have = batched_gather(pool, host, slot)

        probs = []
        if not torch.equal(want, have):
            bad = (want != have).any(dim=(1, 2, 3)).nonzero().flatten().tolist()
            probs.append("slots differ at rows %s" % bad[:6])
        # the region beyond W must be untouched by either
        if not torch.all(have.narrow(0, w, pool.w_max - w) == -1.0):
            probs.append("wrote past W=%d" % w)
        if probs:
            fails += 1
            print("  FAIL %-46s %s" % (label, "; ".join(probs)))
        else:
            hs0, stride, n0, cnt = plan
            print("  ok   %-46s batched, %d runs -> 1 copy (stride %d, len %d)"
                  % (label, cnt, stride, n0))

    # launch accounting: the whole point is the count
    print()
    runs = [(i * 141, i * 33, 33) for i in range(8)]
    pool = Stub(1200, runs)
    pool._batch_plan = pool._plan_batched_gather()
    L = 64
    before, after = 2 * L * len(runs), 2 * L
    print("  launch count at L=%d, %d riders: %d -> %d  (%.0fx fewer)"
          % (L, len(runs), before, after, before / after))

    print()
    print("T3 GATHER EQUIVALENCE %s"
          % ("PASS (%d cases)" % len(CASES) if not fails else "FAIL (%d)" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
