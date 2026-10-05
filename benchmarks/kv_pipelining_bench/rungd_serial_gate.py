#!/usr/bin/env python3
"""Does issuing the prompts SERIALLY give a gate that is actually an equality test?

`rungd_test_selfconsistency.py` established that the existing gate is not one:
two runs of plain vLLM -- no pool, no feature -- produced different text for 2 of
3 prompts. So "pool != vanilla" was never evidence about the pool.

The suspected mechanism is that `hybrid_rungd_real` issues its three prompts
CONCURRENTLY (`ThreadPoolExecutor`), so the engine batches whatever is in flight;
batch composition sets the reduction order inside attention and the MLP, and
greedy decoding turns a last-bit logit difference into a different token. If that
is the mechanism, issuing one prompt at a time should make each side
self-consistent.

This tests exactly that, and it is written as a NEW file rather than an edit to
the existing gate for two reasons: the self-consistency job was still importing
that module when this was written, and keeping both lets the serial and
concurrent results be compared on one node without a checkout dance.

Reading it:

  * vanilla self-consistent under serial issue  => mechanism CONFIRMED, and the
    fix for the gate is to issue serially. The existing gate should then be
    changed and the suite's end-to-end claim can be re-established.
  * vanilla STILL self-inconsistent            => the non-determinism is not
    batch composition, and a deterministic text-level gate is not available by
    this route. Correctness would then have to rest on the bitwise tests, which
    is where it already rests in practice.

Note what is NOT claimed either way: that pool == vanilla. The pool changes
batch composition on purpose -- serving more requests per step is the feature --
so even with serial issue the two sum in different orders and exact text
equality is not guaranteed in principle. The question here is only whether each
side is reproducible, which is the precondition for any comparison at all.

    ./.venv/bin/python benchmarks/kv_pipelining_bench/rungd_serial_gate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hybrid_rungd_real as H  # noqa: E402


def run_serial(port: int, pool: bool) -> dict:
    """H.run, but one prompt at a time so batch composition is pinned."""
    proc = H.launch(port, pool)
    try:
        if not H.wait_ready(port):
            return {p: "<no-start>" for p in H.PROMPTS}
        return {p: H.complete(port, p) for p in H.PROMPTS}
    finally:
        H.terminate_engine(proc, label=__name__)


def first_diff(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def compare(label, a, b):
    bad = [p for p in H.PROMPTS if a[p] != b[p]]
    if not bad:
        print("  [SAME] %-22s all %d prompts identical" % (label,
                                                           len(H.PROMPTS)))
        return True
    print("  [DIFF] %-22s %d of %d prompts differ"
          % (label, len(bad), len(H.PROMPTS)))
    for p in bad:
        i = first_diff(a[p], b[p])
        print("      first diff at char %d of %d (%.0f%% through)"
              % (i, len(a[p]), i / max(1, len(a[p])) * 100))
        lo = max(0, i - 25)
        print("        run A: ...%r" % a[p][lo:i + 25])
        print("        run B: ...%r" % b[p][lo:i + 25])
    return False


def main():
    reps = int(os.environ.get("SERIAL_GATE_REPS", "2"))
    print("Serial-issue gate: is each side reproducible?")
    print("model %s, max_tokens %d, %d prompts, ONE AT A TIME, %d repeats"
          % (H.MODEL, H.MAX_TOKENS, len(H.PROMPTS), reps))
    print()

    port = 9961
    vruns, pruns = [], []
    for i in range(reps):
        print("=== vanilla serial, run %d ===" % (i + 1), flush=True)
        vruns.append(run_serial(port, pool=False))
        port += 1
    for i in range(reps):
        print("=== pool serial, run %d ===" % (i + 1), flush=True)
        pruns.append(run_serial(port, pool=True))
        port += 1

    allv = {}
    for d in vruns + pruns:
        allv.update(d)
    if any(x.startswith("<no-start>") or x == "<err>" for x in allv.values()):
        print("INCONCLUSIVE: a server did not start / errored.")
        return 2

    print()
    print("=" * 72)
    vok = all(compare("vanilla %d vs %d" % (1, i + 2), vruns[0], vruns[i + 1])
              for i in range(len(vruns) - 1))
    pok = all(compare("pool %d vs %d" % (1, i + 2), pruns[0], pruns[i + 1])
              for i in range(len(pruns) - 1))
    compare("pool vs vanilla", pruns[0], vruns[0])
    print("=" * 72)
    print()

    if vok and pok:
        print("VERDICT: SERIAL ISSUE MAKES BOTH SIDES REPRODUCIBLE.")
        print()
        print("That confirms the mechanism: the concurrent issue in the")
        print("existing gate let batch composition vary, which varied the")
        print("reduction order and so the last bits of the logits, and greedy")
        print("decoding turned that into a different token. The fix for the")
        print("gate is to issue serially.")
        print()
        print("It does NOT follow that pool == vanilla. The pool changes batch")
        print("composition deliberately, so the two still sum in different")
        print("orders; see the pool-vs-vanilla line above for what that costs")
        print("on this run.")
        return 0

    print("VERDICT: SERIAL ISSUE IS NOT ENOUGH.")
    print()
    print("  vanilla reproducible: %s" % vok)
    print("  pool    reproducible: %s" % pok)
    print()
    print("So the non-determinism is not (only) batch composition, and a")
    print("deterministic text-level gate is not available by this route.")
    print("Correctness then rests on the bitwise tests -- which is where it")
    print("already rests in practice: build_step_equiv, whole_extent_equiv,")
    print("restore_roundtrip, host_pool, and T2's negative control.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
