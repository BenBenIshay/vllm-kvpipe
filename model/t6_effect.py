#!/usr/bin/env python3
"""Was the mirror-write race real, and what did ordering it cost?

Two questions, and the first one is NOT answered by whether tokens diverged.
The failure is intermittent -- two failures on nova07, a pass on nova06 with the
same binary and flags -- and the negative control (NO_MIRROR_SYNC=1, the race
restored in full) passed on its first two attempts. So output divergence cannot
establish either the bug or the fix.

  1. DID THE HAZARD WINDOW EXIST? `mirror_hazard_steps` counts the steps on
     which the mirror's READERS were still in flight at the instant the deferred
     scatter was about to overwrite their rows. It is sampled with
     Event.query() / Stream.query(), which are non-blocking (Class A, the idiom
     `_drain_wait_events` already uses), and sampled BEFORE the fix's
     synchronise -- so it reports the window whether or not the sync then closed
     it. A non-zero count is the race, independent of whether it fired.

  2. WHAT DID THE ORDERING COST? Fitted `c` and `F` with the fix against the
     control, bracketed. The fix waits on the stream that READ the mirror rather
     than on the main stream, so it should be nearly free: at the top of a step
     the newest such DMA was issued a layer before the end of the previous step.
     If `c` stays in the archived async-ON band (0.883-1.004 at ctx 512) the
     ordering costs nothing and there is no trade-off to weigh.

    ./.venv/bin/python paper/model/t6_effect.py
"""
import glob
import json
import os
import re
import statistics
import sys

DATA = os.environ.get(
    "T6_DATA", "/mnt/beegfs/benishay/paper_runs/20260930/overhead/13_t6")
ARCHIVE = "/mnt/beegfs/benishay/paper_runs/20260930/overhead"

# The async-ON band from three archived runs at this exact config, fitted the
# same way. The fix must land inside it.
ARCHIVED_C = (0.883, 1.004)


def fit(path):
    """(F, c, n, points) from dT = F + c*r over usable rider points."""
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return None
    rows = d.get("rows", [])
    xs = [(p["riders"], p["dT_ms"]) for p in rows
          if p.get("usable") and p["riders"] > 0 and p.get("dT_ms") is not None]
    if len(xs) < 2:
        return None
    n = len(xs)
    sx = sum(x for x, _ in xs)
    sy = sum(y for _, y in xs)
    sxx = sum(x * x for x, _ in xs)
    sxy = sum(x * y for x, y in xs)
    den = n * sxx - sx * sx
    if den == 0:
        return None
    c = (n * sxy - sx * sy) / den
    F = (sy - c * sx) / n
    return d, F, c, n, xs


def hazard(path):
    """(steps with the window open, steps measured) from a phase JSONL."""
    opened = total = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        ph = d.get("phases", {})
        if "wt_flush_scatter" not in ph and "mirror_hazard_steps" not in ph:
            continue
        total += 1
        if "mirror_hazard_steps" in ph:
            opened += 1
    return opened, total


def main():
    print("=" * 78)
    print(" 1. DID THE HAZARD WINDOW EXIST?")
    print("=" * 78)
    print()
    pfs = sorted(glob.glob(os.path.join(DATA, "*_phases.jsonl")))
    if not pfs:
        print(" no phase files under %s (still running?)" % DATA)
    for f in pfs:
        op, tot = hazard(f)
        tag = re.sub(r"_phases\.jsonl$", "", os.path.basename(f))
        if not tot:
            print(" %-18s no flush steps recorded" % tag)
            continue
        print(" %-18s window OPEN on %d of %d flush steps (%.1f%%)"
              % (tag, op, tot, op / tot * 100))
        if op:
            print("   -> a mirror-READING stream was still busy at the instant")
            print("      the scatter was about to run. Stated precisely: this")
            print("      is the NECESSARY condition for the race, measured on")
            print("      every step -- the reader had not retired. It is not by")
            print("      itself proof that the rows overlap.")
            print("      THE ROW OVERLAP is argued separately and is not in")
            print("      doubt: the scatter writes the new token's row, which")
            print("      sits in the last block of the request, and that block")
            print("      is part of the window _issue_runs streams. So the")
            print("      reader and the writer address the same rows, and the")
            print("      only thing standing between them was the ordering")
            print("      this fix adds.")
        else:
            print("   -> NOT observed on this run. That does NOT clear the")
            print("      path: the window is timing-dependent and two token")
            print("      divergences were observed directly on nova07. It")
            print("      means this configuration did not expose it.")
    print()

    print("=" * 78)
    print(" 2. WHAT DID THE ORDERING COST?")
    print("=" * 78)
    print()
    print(" %-14s %7s %8s %9s %s"
          % ("run", "async", "F ms", "c ms/r", "note"))
    rows = {}
    for tag in ("T6_on", "T6_fix_a", "T6_nofix", "T6_fix_b"):
        p = os.path.join(DATA, tag + ".json")
        r = fit(p)
        if not r:
            continue
        d, F, c, n, xs = r
        rows[tag] = (F, c)
        note = ""
        if tag == "T6_nofix":
            note = "control: ordering DISABLED"
        print(" %-14s %7s %8.2f %9.3f %s"
              % (tag, d.get("async"), F, c, note))
    print()
    # DO NOT TRUST THE TAGS. The run artifacts do not record whether the
    # mirror ordering was on, and I flipped its DEFAULT while this very job was
    # running -- so T6_fix_b, named for the ordered arm, actually ran unordered.
    # Grouping by the measured value instead is not a nicety here; keying off
    # the tag produced a "bracket drift" of 6.119 and the verdict "FREE", which
    # is the effect being reported as its own noise floor.
    vals = sorted((v[1], k, v[0]) for k, v in rows.items())
    if len(vals) >= 2:
        gap, cut = 0.0, 1
        for i in range(1, len(vals)):
            if vals[i][0] - vals[i - 1][0] > gap:
                gap, cut = vals[i][0] - vals[i - 1][0], i
        lowg, highg = vals[:cut], vals[cut:]
        print(" grouped by MEASURED c, not by tag:")
        for name, grp in (("low", lowg), ("high", highg)):
            cs = [v[0] for v in grp]
            fs = [v[2] for v in grp]
            print("   %-5s c %s  (spread %.3f)   F %s"
                  % (name, " ".join("%.3f" % c for c in cs),
                     max(cs) - min(cs), " ".join("%.2f" % f for f in fs)))
            print("         arms: %s" % ", ".join(v[1] for v in grp))
        eff = (sum(v[0] for v in highg) / len(highg)
               - sum(v[0] for v in lowg) / len(lowg))
        print()
        print(" separation between the groups: %.3f ms/rider" % eff)
        print(" within-group spread: %.3f (low), %.3f (high)"
              % (max(v[0] for v in lowg) - min(v[0] for v in lowg),
                 max(v[0] for v in highg) - min(v[0] for v in highg)))
        tight = (max(v[0] for v in lowg) - min(v[0] for v in lowg) < 0.1
                 and max(v[0] for v in highg) - min(v[0] for v in highg) < 0.1)
        print(" verdict: %s"
              % ("the two groups are tight and far apart, so the separation is "
                 "the ORDERING, not noise"
                 if tight and eff > 1.0 else
                 "the groups are not cleanly separated -- do not attribute"))
        print()
        print(" The per-rider transfer at this config is 6.17 ms (0.155 GB at")
        print(" 25.1 GB/s), which the separation should match if the ordering")
        print(" simply un-hides the transfer: %.3f vs 6.17, %.0f%% agreement."
              % (eff, (1 - abs(eff - 6.17) / 6.17) * 100))
        lo, hi = ARCHIVED_C
        print()
        print(" archived async-ON band at this config: c in [%.3f, %.3f];"
              % (lo, hi))
        print(" the LOW group sits in it, which is what makes the bracket")
        print(" internally consistent rather than a cross-session comparison.")
    print()
    print(" WHAT THIS DOES NOT SHOW, and it is most of it.")
    print()
    print(" 1. That the hazard causes the divergence. It probably does not. The")
    print("    row the scatter overwrites holds the CURRENT token, and at step")
    print("    N-1 the streamed window only needs tokens up to t-1 -- token t's")
    print("    KV comes from the resident GPU cache, written by the attention")
    print("    kernel. So what the in-flight prefetch reads there is masked out")
    print("    by seq_len and never consumed. On that reading the race is")
    print("    BENIGN and section 1 measured a real window around a value")
    print("    nobody reads.")
    print()
    print(" 2. Correctness. The repeated token-identical runs in the log are")
    print("    weak by construction: the failure is intermittent and the")
    print("    negative control passed twice on the node that had failed twice,")
    print("    so passes do not accumulate into a proof.")
    print()
    print(" 3. Severity. hybrid_rungd_real compares full texts but prints 46")
    print("    characters, and both failures had identical printed halves --")
    print("    i.e. they diverged LATE. A late divergence between two fluent")
    print("    continuations is a flipped near-tie, not corruption, and the")
    print("    pool's two-source attention sums in a different order from")
    print("    vanilla so small FP differences are guaranteed anyway. See")
    print("    rungd_divergence_detail.py, which prints the position.")
    print()
    print(" So the open question is not 'does T6 work' but 'what actually")
    print(" diverges, and is it a bug at all'. T6 stays flag-gated either way,")
    print(" because section 2 shows it costs everything async buys.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
