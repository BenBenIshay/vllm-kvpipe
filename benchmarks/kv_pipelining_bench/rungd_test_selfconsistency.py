#!/usr/bin/env python3
"""Is the token-identity test a valid equality test at all? Check it against itself.

`hybrid_rungd_real.py` declares the feature correct when the pool's text equals
vanilla's, and the whole overhead campaign was signed off against that gate. A
rate study then failed on the arm where the feature's async path is OFF -- the
suite's own configuration -- so the gate is flaky, and before attributing that
to anything in the pool the test's own premise has to be checked.

THE PREMISE IS THAT BOTH SIDES ARE DETERMINISTIC. There is a specific reason to
doubt it: the test issues its three prompts CONCURRENTLY
(`ThreadPoolExecutor(max_workers=len(PROMPTS))`), so the engine batches whatever
happens to be in flight. Batch composition changes the reduction order inside
attention and the MLP, which changes logits in the last bits, and under greedy
decoding a near-tie argmax then flips. If that happens, two runs of the SAME
configuration disagree and "pool != vanilla" says nothing about the pool.

So this compares like with like:

    vanilla vs vanilla      -- must be identical if the test is valid at all
    pool    vs pool         -- same question on the other side

Reading it:

  * Either self-comparison differing => THE TEST IS NOT AN EQUALITY TEST. Every
    "suite 9/9" in this campaign was a coin that happened to land right, and the
    pool-vs-vanilla failures are not evidence of a fault in the pool.
  * Both self-comparisons identical, pool-vs-vanilla differing => the difference
    is real and attributable to the feature, and the earlier investigation
    should resume.

Deliberately NOT importing anything that mutates hybrid_rungd_real: it reuses
that module's server plumbing read-only, on its own ports, so it can run beside
the suite.

    ./.venv/bin/python benchmarks/kv_pipelining_bench/rungd_test_selfconsistency.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hybrid_rungd_real as H  # noqa: E402


def first_diff(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def compare(label, a, b):
    """True when identical."""
    bad = [p for p in H.PROMPTS if a[p] != b[p]]
    if not bad:
        print("  [SAME] %-22s all %d prompts identical" % (label, len(H.PROMPTS)))
        return True
    print("  [DIFF] %-22s %d of %d prompts differ" % (label, len(bad),
                                                      len(H.PROMPTS)))
    for p in bad:
        i = first_diff(a[p], b[p])
        print("      %r" % p[:34])
        print("        first diff at char %d of %d (%.0f%% through)"
              % (i, len(a[p]), i / max(1, len(a[p])) * 100))
        lo = max(0, i - 25)
        print("        run A: ...%r" % a[p][lo:i + 25])
        print("        run B: ...%r" % b[p][lo:i + 25])
    return False


def main():
    print("Self-consistency of the token-identity gate")
    print("model %s, max_tokens %d, %d prompts issued CONCURRENTLY"
          % (H.MODEL, H.MAX_TOKENS, len(H.PROMPTS)))
    print()

    print("=== vanilla, run A ===", flush=True)
    v1 = H.run(9951, pool=False)
    print("=== vanilla, run B ===", flush=True)
    v2 = H.run(9952, pool=False)
    print("=== pool, run A ===", flush=True)
    p1 = H.run(9953, pool=True)
    print("=== pool, run B ===", flush=True)
    p2 = H.run(9954, pool=True)

    allv = {**v1, **v2, **p1, **p2}
    if any(x.startswith("<no-start>") or x == "<err>" for x in allv.values()):
        print("INCONCLUSIVE: a server did not start / errored.")
        return 2

    print()
    print("=" * 72)
    vok = compare("vanilla vs vanilla", v1, v2)
    pok = compare("pool vs pool", p1, p2)
    xok = compare("pool vs vanilla", p1, v1)
    print("=" * 72)
    print()

    if not vok or not pok:
        print("VERDICT: THE GATE IS NOT AN EQUALITY TEST.")
        print()
        print("A configuration disagrees with ITSELF, so comparing pool against")
        print("vanilla cannot establish correctness in either direction. The")
        print("most likely source is the concurrent issue of the prompts: batch")
        print("composition varies run to run, which changes reduction order and")
        print("so the last bits of the logits, and greedy decoding turns a")
        print("near-tie into a different token.")
        print()
        print("CONSEQUENCE FOR THE CAMPAIGN: every 'suite 9/9' cited as")
        print("correctness evidence -- including for T1 and T2 -- was a test")
        print("that can pass or fail by chance. The feature may well be")
        print("correct; this gate was never capable of showing it.")
        print()
        print("WHAT WOULD BE A VALID GATE: issue the prompts one at a time so")
        print("batch composition is fixed; or compare logits with a tolerance")
        print("rather than argmax'd text; or compare the KV tensors the pool")
        print("streams against the resident ones directly, which is what the")
        print("mechanism-level tests (restore_roundtrip, host_pool) already do")
        print("and why those are the trustworthy half of the suite.")
        return 1

    if xok:
        print("VERDICT: both sides self-consistent AND pool == vanilla here.")
        print("The gate looks valid on this run and the pool agreed with it.")
        return 0

    print("VERDICT: both sides SELF-CONSISTENT but pool != vanilla.")
    print("That makes the difference real and attributable to the feature.")
    print("The FP-tie-breaking explanation is then wrong and the fault should")
    print("be hunted in the pool after all.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
