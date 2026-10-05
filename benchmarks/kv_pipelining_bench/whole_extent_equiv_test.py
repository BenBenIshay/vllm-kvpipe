#!/usr/bin/env python3
"""T5's whole-extent runs must resolve every block to the SAME host block.

T5 changes the compact numbering: runs are extent-length rather than used-block
length, so W grows and every compact id moves. So the test CANNOT be "the block
table is unchanged" -- it will not be, and asserting that would reject a correct
change. The invariant that actually matters is end-to-end:

    for every request and every one of its logical blocks j, the rolling slot
    row that the block table sends attention to must contain the KV of host
    block req_blocks[rid][j].

So this builds a host mirror whose block `b` holds the value `b`, runs the
planned runs to fill the rolling slot exactly as `_issue_runs` does, and then
reads back through the block table. If the index arithmetic is wrong anywhere --
extent base, compact base, run merge, or the arange slice -- a row resolves to
the wrong host block and that is caught here rather than as a silent accuracy
loss in the engine.

Also asserted, because they are the point of the change rather than side
effects: the runs MERGE to one when the extents are consecutive, the launch
count therefore falls to 2 per layer, and the slack guard REFUSES a loose extent
instead of quietly moving four times the bytes.

CPU only, no CUDA needed.

    ./.venv/bin/python benchmarks/kv_pipelining_bench/whole_extent_equiv_test.py
"""
import sys

import torch

sys.path.insert(0, ".")
from vllm.v1.hybrid_rungd_pool import RungdHostPool  # noqa: E402


class Stub:
    """Just enough pool to drive _plan_extents and the gather."""

    def __init__(self, bpr, assignment, used, whole, max_slack=0.5,
                 w_max=4096):
        self._bpr = bpr
        self._whole_extent = whole
        self._whole_extent_max_slack = max_slack
        self.w_max = w_max
        self.whole_extent_steps = 0
        self.whole_extent_refused = 0
        # assignment: rid -> extent slot;  used: rid -> blocks filled
        self._req_extent = dict(assignment)
        self.req_blocks = {}
        for rid, slot in assignment.items():
            base = slot * bpr
            self.req_blocks[rid] = [base + i for i in range(used[rid])]
        self._ar = torch.arange(w_max, dtype=torch.int32)

    _plan_extents = RungdHostPool._plan_extents


def gather(runs, w, h_max):
    """Fill a [w] slot from a mirror where block b holds value b, as
    _issue_runs does: one contiguous copy per run."""
    host = torch.arange(h_max, dtype=torch.int64)
    slot = torch.full((w,), -1, dtype=torch.int64)
    for hs, cs, n in runs:
        slot.narrow(0, cs, n).copy_(host.narrow(0, hs, n))
    return slot


def block_table(stub, order, row_cbase):
    n_req = len(order)
    max_blk = max(len(stub.req_blocks[r]) for r in order)
    bt = torch.zeros(n_req, max_blk, dtype=torch.int32)
    for row, rid in enumerate(order):
        n = len(stub.req_blocks[rid])
        cb = row_cbase[row]
        bt[row, :n] = stub._ar[cb:cb + n]
    return bt


CASES = [
    # (label, bpr, {rid: slot}, {rid: used}, expect_merged_runs)
    ("4 riders, consecutive slots, tight extent",
     41, {"a": 0, "b": 1, "c": 2, "d": 3},
     {"a": 33, "b": 33, "c": 33, "d": 33}, 1),
    ("8 riders, consecutive, ragged use",
     41, {chr(97 + i): i for i in range(8)},
     dict(zip([chr(97 + i) for i in range(8)],
              [33, 34, 33, 35, 33, 41, 33, 40])), 1),
    ("slots out of order -- must still merge after the sort",
     41, {"a": 2, "b": 0, "c": 1}, {"a": 33, "b": 33, "c": 33}, 1),
    ("a gap in the slots -- two runs, not one",
     41, {"a": 0, "b": 1, "c": 3}, {"a": 33, "b": 33, "c": 33}, 2),
    ("single rider",
     41, {"a": 0}, {"a": 33}, 1),
    ("extent exactly full",
     41, {"a": 0, "b": 1}, {"a": 41, "b": 41}, 1),
    ("two separate pairs",
     41, {"a": 0, "b": 1, "c": 5, "d": 6},
     {"a": 33, "b": 33, "c": 33, "d": 33}, 2),
]


def main():
    fails = 0
    print("T5 whole-extent runs: does every block still resolve correctly?")
    print()

    for label, bpr, slots, used, want_runs in CASES:
        order = sorted(slots)
        stub = Stub(bpr, slots, used, whole=True)
        plan = stub._plan_extents(order)
        if plan is None:
            fails += 1
            print("  FAIL %-52s plan returned None" % label)
            continue
        runs, row_cbase, w = plan
        h_max = (max(slots.values()) + 1) * bpr

        probs = []
        if len(runs) != want_runs:
            probs.append("got %d runs, wanted %d" % (len(runs), want_runs))
        if w != len(order) * bpr:
            probs.append("W=%d, wanted %d" % (w, len(order) * bpr))

        # THE INVARIANT: resolve every logical block through the block table.
        slot = gather(runs, w, h_max)
        bt = block_table(stub, order, row_cbase)
        for row, rid in enumerate(order):
            for j, want_host in enumerate(stub.req_blocks[rid]):
                got = int(slot[int(bt[row, j])])
                if got != want_host:
                    probs.append("%s blk %d -> host %d, wanted %d"
                                 % (rid, j, got, want_host))
                    break

        # and the rows the block table never points at are the slack
        pointed = set()
        for row, rid in enumerate(order):
            n = len(stub.req_blocks[rid])
            pointed.update(int(x) for x in bt[row, :n])
        slack = w - sum(len(stub.req_blocks[r]) for r in order)
        if len(pointed) != w - slack:
            probs.append("block table points at %d of %d rows, expected %d"
                         % (len(pointed), w, w - slack))

        if probs:
            fails += 1
            print("  FAIL %-52s %s" % (label, "; ".join(probs[:3])))
        else:
            print("  ok   %-52s %d run(s), W=%d, %d launches/layer"
                  % (label, len(runs), w, 2 * len(runs)))

    # --- the per-run path must be untouched when the flag is off -----------
    print()
    for label, bpr, slots, used, _ in CASES:
        order = sorted(slots)
        off = Stub(bpr, slots, used, whole=False)
        plan = off._plan_extents(order)
        if plan is None:
            fails += 1
            print("  FAIL flag-off %-44s plan returned None" % label)
            continue
        runs, row_cbase, w = plan
        want_w = sum(used[r] for r in order)
        h_max = (max(slots.values()) + 1) * bpr
        slot = gather(runs, w, h_max)
        bt = block_table(off, order, row_cbase)
        bad = None
        for row, rid in enumerate(order):
            for j, want_host in enumerate(off.req_blocks[rid]):
                if int(slot[int(bt[row, j])]) != want_host:
                    bad = "%s blk %d" % (rid, j)
                    break
        if w != want_w or bad:
            fails += 1
            print("  FAIL flag-off %-44s W=%d want %d, bad=%s"
                  % (label, w, want_w, bad))
        else:
            print("  ok   flag-off %-44s W=%d (used blocks only), %d runs"
                  % (label, w, len(runs)))

    # --- the slack guard must REFUSE a loose extent ------------------------
    print()
    loose = Stub(141, {"a": 0, "b": 1, "c": 2}, {"a": 33, "b": 33, "c": 33},
                 whole=True)
    runs, row_cbase, w = loose._plan_extents(["a", "b", "c"])
    refused = loose.whole_extent_refused == 1 and w == 99
    if not refused:
        fails += 1
        print("  FAIL slack guard: refused=%d W=%d (wanted refused=1, W=99)"
              % (loose.whole_extent_refused, w))
    else:
        print("  ok   slack guard refuses bpr=141 with 33 used (76%% slack):")
        print("       falls back to %d runs, W=%d, per-block numbering"
              % (len(runs), w))

    tight = Stub(41, {"a": 0, "b": 1, "c": 2}, {"a": 33, "b": 33, "c": 33},
                 whole=True)
    _r, _c, w2 = tight._plan_extents(["a", "b", "c"])
    good = tight.whole_extent_steps == 1 and w2 == 123 and len(_r) == 1
    if not good:
        fails += 1
        print("  FAIL tight extent not accepted: steps=%d W=%d runs=%d"
              % (tight.whole_extent_steps, w2, len(_r)))
    else:
        print("  ok   tight extent accepted (20%% slack): 1 run, W=123")

    print()
    n = len(CASES) * 2 + 2
    print("T5 WHOLE-EXTENT EQUIVALENCE %s"
          % ("PASS (%d checks)" % n if not fails else "FAIL (%d)" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
