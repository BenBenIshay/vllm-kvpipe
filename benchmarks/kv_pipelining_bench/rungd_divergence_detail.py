#!/usr/bin/env python3
"""WHERE does the async path diverge, and what does the divergence look like?

`hybrid_rungd_real.py` compares full texts (`v[p] == b[p]`) but prints only the
first 46 characters, so a DIFF whose printed halves look identical tells us
nothing about severity -- and that is exactly what the two async failures looked
like. With MAX_TOKENS=32, 46 characters is roughly the first eleven tokens, so
those divergences were LATE.

That distinction decides what the bug is:

  * EARLY divergence, continuation turns to garbage
        -> KV corruption. Something is streaming the wrong bytes.
  * LATE divergence, both continuations stay fluent and differ by a word
        -> a near-tie argmax flipped. The pool path sums attention over two
           sources (resident + streamed) in a different ORDER from vanilla, so
           small floating-point differences are guaranteed by construction,
           independent of any race. Strict token identity can then fail by
           chance, and the failure is a property of the TEST rather than of the
           feature.

The second case would also explain why the failure is intermittent, why the
negative control passed twice, and why the hazard counter -- which reads 100% --
does not predict it.

This prints the full texts, the first differing character, and the first
differing token-ish word, for each prompt and for a chosen configuration.

    # the configuration that failed
    VLLM_KVPIPE_RUNGD_ASYNC=1 ./.venv/bin/python \\
        benchmarks/kv_pipelining_bench/rungd_divergence_detail.py

Reuses hybrid_rungd_real's own server plumbing so the two cannot drift apart.
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


def word_diff(a: str, b: str):
    wa, wb = a.split(), b.split()
    for i in range(min(len(wa), len(wb))):
        if wa[i] != wb[i]:
            return i, wa[i], wb[i]
    if len(wa) != len(wb):
        i = min(len(wa), len(wb))
        return i, (wa[i] if i < len(wa) else "<end>"), \
               (wb[i] if i < len(wb) else "<end>")
    return None


def main():
    print("async=%s  no_mirror_sync=%s  d2h_async=%s"
          % (os.environ.get("VLLM_KVPIPE_RUNGD_ASYNC", "0"),
             os.environ.get("VLLM_KVPIPE_RUNGD_NO_MIRROR_SYNC", "0"),
             os.environ.get("VLLM_KVPIPE_RUNGD_D2H_ASYNC", "1")))
    print("model %s, max_tokens %d" % (H.MODEL, H.MAX_TOKENS))
    print()

    # Different ports from hybrid_rungd_real's 9931/9932, so this can run
    # alongside it without one server's port binding failing the other.
    print("=== VANILLA (no tiers, real KV) ===", flush=True)
    v = H.run(9941, pool=False)
    print("=== POOL (hybrid tiers, no connector, RUNGD_FREE=1) ===", flush=True)
    b = H.run(9942, pool=True)
    if any(x.startswith("<no-start>") or x == "<err>"
           for x in {**v, **b}.values()):
        print("INCONCLUSIVE: a server did not start / errored.")
        return 2
    bad = 0
    for p in H.PROMPTS:
        a, c = v[p], b[p]
        if a == c:
            print("[OK  ] %r" % p[:40])
            continue
        bad += 1
        i = first_diff(a, c)
        frac = i / max(1, len(a))
        print("[DIFF] %r" % p[:40])
        print("   first differing char: %d of %d  (%.0f%% through)"
              % (i, len(a), frac * 100))
        wd = word_diff(a, c)
        if wd:
            print("   first differing word: #%d  vanilla=%r  pool=%r"
                  % (wd[0], wd[1], wd[2]))
        print("   vanilla: %r" % a)
        print("   pool   : %r" % c)
        print("   ...context around the split:")
        lo = max(0, i - 30)
        print("     vanilla: ...%r" % a[lo:i + 30])
        print("     pool   : ...%r" % c[lo:i + 30])
        print()

    print("=" * 72)
    if not bad:
        print("NO DIVERGENCE on this run. For an intermittent fault that is")
        print("one sample and settles nothing -- see the rate study.")
    else:
        print("%d of %d prompts diverged." % (bad, len(H.PROMPTS)))
        print()
        print("HOW TO READ IT. A divergence late in the string, with both")
        print("continuations still fluent and differing by a word, is a flipped")
        print("near-tie: the pool sums attention over two KV sources in a")
        print("different order from vanilla, so small FP differences exist by")
        print("construction. An early divergence into garbage is KV corruption.")
        print("These demand opposite responses, which is why the position is")
        print("printed rather than just the verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
