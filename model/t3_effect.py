#!/usr/bin/env python3
"""Did T3 cut the launch count in the engine, and what did it buy?

Two separate questions, and the first gates the second. T3 replaces the per-run
H2D loop with one strided copy per tensor per layer, so `h2d_launches` should
read 2 per layer-step instead of 2*r. If the regularity check never fires inside
the real engine then T3 is a no-op however well it tests off-GPU -- which is
exactly how T1 could have failed -- so the launch count is checked before any
timing is read.

It also attributes the post-T2 residual F, which T2 left at 0.048 ms per layer,
65% of what remains. `write_token_d2h_async` times the async copies and
`wt_flush_scatter` the deferred scatter; without that split there is nothing to
aim the next change at.

    ./.venv/bin/python paper/model/t3_effect.py
"""
import glob
import json
import os
import re
import statistics
import sys

DATA = os.environ.get(
    "OPT_DATA", "/mnt/beegfs/benishay/paper_runs/20260930/overhead/10_t3")

WARMUP_PER_GROUP = 3


def purity_ok(path):
    rec = re.sub(r"_phases\.jsonl$", ".json", path)
    if not os.path.exists(rec):
        return True, "no run record"
    try:
        d = json.load(open(rec))
    except ValueError:
        return True, "unreadable"
    tot, use = d.get("n_rows"), d.get("n_rows_usable")
    if tot and use is not None and use < tot:
        return False, "gate rejected %d of %d rows" % (tot - use, tot)
    return True, "%s/%s" % (use, tot)


def read(path):
    groups = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        c = d.get("ctx") or {}
        k = c.get("n_active")
        if k is None:
            continue
        ph = d.get("phases", {})
        groups.setdefault(k, []).append({
            "step": d.get("step"),
            "n_runs": c.get("n_runs"),
            "launches": (ph.get("h2d_launches") or {}).get("calls"),
            "batched": "h2d_batched_steps" in ph,
            "issue": (ph.get("issue_runs") or {}).get("host_ms"),
            "d2h_async": (ph.get("write_token_d2h_async") or {}).get("host_ms"),
            "flush": (ph.get("wt_flush_scatter") or {}).get("host_ms"),
            "wt": (ph.get("write_token") or {}).get("host_ms"),
            "layers": c.get("layers"),
        })
    return groups


def main():
    files = sorted(glob.glob(os.path.join(DATA, "T3_*_phases.jsonl")))
    if not files:
        print("no T3 phase files under %s (still running?)" % DATA)
        return 0

    print("=" * 84)
    print(" 1. DID THE LAUNCH COUNT DROP IN THE ENGINE?")
    print("=" * 84)
    print()
    verdict = {}
    for f in files:
        tag = re.sub(r"_phases\.jsonl$", "", os.path.basename(f))
        ok, note = purity_ok(f)
        if not ok:
            print(" %-16s DISCARDED -- %s" % (tag, note))
            continue
        g = read(f)
        if not g:
            continue
        print(" %s" % tag)
        print("   %7s %7s %10s %12s %9s"
              % ("riders", "n_runs", "launches", "expect 2*r", "batched"))
        for k in sorted(g):
            recs = sorted(g[k], key=lambda r: r["step"] or 0)[WARMUP_PER_GROUP:]
            if len(recs) < 5:
                continue
            lau = [r["launches"] for r in recs if r["launches"] is not None]
            if not lau:
                continue
            med = statistics.median(lau)
            nr = statistics.median([r["n_runs"] for r in recs if r["n_runs"]])
            L = recs[0]["layers"] or 64
            nb = sum(1 for r in recs if r["batched"])
            print("   %7d %7.0f %10.0f %12.0f %6d/%d"
                  % (k, nr, med, 2 * nr * L, nb, len(recs)))
            verdict.setdefault(tag, []).append((k, med, 2 * nr * L, nb, len(recs)))
        print()

    on = [t for t in verdict if "_on" in t]
    off = [t for t in verdict if "_off" in t]
    if on and off:
        t_on, t_off = on[0], off[0]
        print(" %s vs %s, launches per step:" % (t_on, t_off))
        d_on = dict((k, v) for k, v, _, _, _ in verdict[t_on])
        d_off = dict((k, v) for k, v, _, _, _ in verdict[t_off])
        for k in sorted(set(d_on) & set(d_off)):
            if d_on[k]:
                print("   riders %-3d  T3 on %-6.0f  T3 off %-6.0f  %.0fx fewer"
                      % (k, d_on[k], d_off[k], d_off[k] / d_on[k]))
        print()

    print("=" * 84)
    print(" 2. WHERE THE REMAINING PER-LAYER HOST COST GOES")
    print("=" * 84)
    print()
    print(" T2 left 0.048 ms per layer of D2H, 65% of the residual F. This")
    print(" splits it between the async copies and the deferred scatter.")
    print()
    print(" READ ONLY THE LOW-RIDER ROWS. These are host-side brackets, and a")
    print(" copy issued with non_blocking=True still blocks the host once the")
    print(" copy stream is saturated -- which is what being AT the knee means.")
    print(" At ctx 512 the knee is 8 riders, and issue_runs then reads 257 ms")
    print(" per step against a 65 ms step: that is bus wait, not launch cost.")
    print(" The same artefact was already documented for write_token at 21 ms.")
    print(" The trustworthy measure of T3 is the DIFFERENCED step time in the")
    print(" bracketed on/off/on arms, not these timers.")
    print()
    for f in files:
        tag = re.sub(r"_phases\.jsonl$", "", os.path.basename(f))
        ok, _ = purity_ok(f)
        if not ok:
            continue
        g = read(f)
        rows = []
        for k in sorted(g):
            recs = sorted(g[k], key=lambda r: r["step"] or 0)[WARMUP_PER_GROUP:]
            if len(recs) < 5:
                continue
            L = recs[0]["layers"] or 64

            def med(key):
                v = [r[key] for r in recs if r[key] is not None]
                return statistics.median(v) if v else None
            rows.append((k, L, med("issue"), med("d2h_async"), med("flush"),
                         med("wt")))
        if not rows:
            continue
        print(" %s" % tag)
        print("   %7s %10s %12s %12s %12s"
              % ("riders", "issue_runs", "d2h_async", "flush_scatter",
                 "write_token"))
        for k, L, iss, da, fl, wt in rows:
            print("   %7d %10s %12s %12s %12s"
                  % (k,
                     "-" if iss is None else "%.3f" % iss,
                     "-" if da is None else "%.3f" % da,
                     "-" if fl is None else "%.3f" % fl,
                     "-" if wt is None else "%.3f" % wt))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
