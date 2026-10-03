#!/usr/bin/env python3
"""Did T1 remove the per-block term's scaling, or only shrink it?

The campaign measured the block-table rebuild at 0.00765 ms per rider BLOCK per
step, by varying block size at fixed bytes. T1 rewrote it to one slice
assignment per REQUEST, which should make it O(r) and independent of the block
count. This reads the Class A host timer `build_step_bt_python` and tests that.

The sharp test is BLOCK SIZE, not context or rider count: both of those move the
block count as well, so only block size at fixed context and cohort isolates the
claim. A 4x block-size change should now barely move the timer, where before it
moved it ~4x.

TWO THINGS THE RECORDS MAKE EASY TO GET WRONG, both of which bit a first version
of this script:

  * `w_blocks` INCREMENTS EVERY STEP as the sequence generates, so it cannot be
    part of a grouping key -- every group ends up with one or two records. It is
    a per-record value, and the per-block figure has to be computed per record
    before any median is taken.
  * step numbers are GLOBAL across the whole sweep, not per rider count, so a
    fixed "discard steps below N" warmup filter discards nothing useful. The
    first step at each new cohort size carries real first-touch cost -- 0.224 ms
    against a steady 0.027 at r=1 -- so the warmup has to be dropped per group.

    ./.venv/bin/python paper/model/t1_effect.py
"""
import glob
import json
import os
import re
import statistics
import sys

DATA = os.environ.get(
    "OPT_DATA", "/mnt/beegfs/benishay/paper_runs/20260930/overhead/08_opt")

BASELINE_MS_PER_BLOCK = 0.00765   # the pre-T1 coefficient, from slice B/B2
WARMUP_PER_GROUP = 3              # first-touch steps at each new cohort size


def purity_ok(path):
    """(ok, note) from the arm's companion run record.

    The phase JSONL carries no purity verdict, so an arm whose rows the gate
    rejected would otherwise be read as if it were clean. The campaign's rule
    is that a point failing the gate is DISCARDED, not adjusted, and that has
    to apply to this analysis too: T1_c2048_bs64 had all three rows rejected
    for prefill leakage, because block rounding at bs 64 inflates each
    request's allocation from 2256 to 2304 tokens and 16 residents then exceed
    the pool, tipping it into eviction and recompute. Its r=0 baseline is
    already contaminated (61.42 ms against 53.76 at bs 16).
    """
    rec = re.sub(r"_phases\.jsonl$", ".json", path)
    if not os.path.exists(rec):
        return True, "no run record found"
    try:
        d = json.load(open(rec))
    except ValueError:
        return True, "run record unreadable"
    tot, use = d.get("n_rows"), d.get("n_rows_usable")
    if tot and use is not None and use < tot:
        return False, "purity gate rejected %d of %d rows" % (tot - use, tot)
    return True, "%s/%s rows usable" % (use, tot)


def read_arm(path):
    """Per-step records grouped by (n_active, block_size, fast_path)."""
    groups = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ph = d.get("phases", {}).get("build_step_bt_python")
            if not ph:
                continue
            c = d.get("ctx") or {}
            key = (c.get("n_active"), c.get("block_size"),
                   c.get("build_fast_path"))
            groups.setdefault(key, []).append(
                {"step": d.get("step"), "ms": ph["host_ms"],
                 "w": c.get("w_blocks"), "max_blk": c.get("max_blk")})
    return groups


def summarise(groups):
    rows = []
    for (n_active, bs, fast), recs in groups.items():
        recs.sort(key=lambda r: r["step"] or 0)
        recs = recs[WARMUP_PER_GROUP:]
        if len(recs) < 5:
            continue
        ms = [r["ms"] for r in recs]
        per_blk = [r["ms"] / r["w"] for r in recs if r["w"]]
        ws = [r["w"] for r in recs if r["w"]]
        rows.append({
            "n_active": n_active, "bs": bs, "fast": fast,
            "n_steps": len(recs),
            "ms": statistics.median(ms),
            "ms_max": max(ms),
            "w_med": statistics.median(ws) if ws else None,
            "ms_per_block": statistics.median(per_blk) if per_blk else None,
        })
    rows.sort(key=lambda r: (r["bs"] or 0, r["n_active"] or 0))
    return rows


def main():
    files = sorted(glob.glob(os.path.join(DATA, "T1_*_phases.jsonl")))
    if not files:
        print("no T1 phase files under %s (the arms may still be running)" % DATA)
        return 0

    print("=" * 78)
    print(" T1: build_step_bt_python, the block-table rebuild")
    print("=" * 78)
    print()
    print(" pre-T1 coefficient %.5f ms per block per step; first %d steps of"
          % (BASELINE_MS_PER_BLOCK, WARMUP_PER_GROUP))
    print(" each cohort size dropped as first-touch.")
    print()

    allrows = {}
    for f in files:
        tag = re.sub(r"_phases\.jsonl$", "", os.path.basename(f))
        ok, note = purity_ok(f)
        if not ok:
            print(" %-18s DISCARDED -- %s" % (tag, note))
            print()
            continue
        rows = summarise(read_arm(f))
        if not rows:
            print(" %-18s no usable records yet" % tag)
            continue
        allrows[tag] = rows
        print(" %s" % tag)
        print("   %8s %4s %6s %6s %5s %11s %11s %12s %9s"
              % ("n_active", "bs", "blocks", "steps", "path", "bt ms med",
                 "bt ms max", "ms/block", "vs pre-T1"))
        for r in rows:
            spd = (BASELINE_MS_PER_BLOCK / r["ms_per_block"]
                   if r["ms_per_block"] else float("nan"))
            print("   %8s %4s %6s %6d %5s %11.4f %11.4f %12.6f %8.1fx"
                  % (r["n_active"], r["bs"], r["w_med"], r["n_steps"],
                     "fast" if r["fast"] else "GEN", r["ms"], r["ms_max"],
                     r["ms_per_block"] or 0.0, spd))
        print()

    print("=" * 78)
    print(" THE SHARP TEST -- block size at fixed context and cohort")
    print("=" * 78)
    print()
    print(" T1 removes the dependence on the BLOCK COUNT, so at fixed context")
    print(" and cohort a 4x block-size change should leave the timer nearly")
    print(" unchanged. Before T1 it was one tensor __setitem__ per block, so it")
    print(" would have fallen ~4x when the block count fell 4x.")
    print()
    for a, b, lab in (("T1_c512_bs16", "T1_c512_bs64", "ctx 512"),
                      ("T1_c2048_bs16", "T1_c2048_bs64", "ctx 2048")):
        if a not in allrows or b not in allrows:
            continue
        ra = {r["n_active"]: r for r in allrows[a]}
        rb = {r["n_active"]: r for r in allrows[b]}
        shared = sorted(set(ra) & set(rb))
        if not shared:
            continue
        print(" %s" % lab)
        print("   %8s %10s %10s %9s %9s %9s %s"
              % ("n_active", "bs16 ms", "bs64 ms", "bs16 blk", "bs64 blk",
                 "ms ratio", "verdict"))
        for n in shared:
            x, y = ra[n], rb[n]
            ratio = y["ms"] / x["ms"] if x["ms"] else float("nan")
            # blocks fall ~4x; if the cost tracked blocks, ms would too
            verd = ("block-count dependence GONE" if ratio > 0.7
                    else "still tracks blocks")
            print("   %8s %10.4f %10.4f %9s %9s %8.2fx %s"
                  % (n, x["ms"], y["ms"], x["w_med"], y["w_med"], ratio, verd))
        print()

    print("=" * 78)
    print(" THE RIDER AXIS -- O(r) now, instead of O(r * blocks)")
    print("=" * 78)
    for tag in sorted(allrows):
        rows = allrows[tag]
        if len(rows) < 3:
            continue
        base = min(r["n_active"] for r in rows)
        print()
        print(" %s" % tag)
        print("   %8s %7s %6s %11s %13s %14s"
              % ("n_active", "riders", "blocks", "bt ms", "ms per rider",
                 "pre-T1 would be"))
        for r in rows:
            riders = (r["n_active"] or 0) - base + 1
            pre = BASELINE_MS_PER_BLOCK * (r["w_med"] or 0)
            print("   %8s %7d %6s %11.4f %13s %14.4f"
                  % (r["n_active"], riders, r["w_med"], r["ms"],
                     "%.4f" % (r["ms"] / riders) if riders else "-", pre))
        # is the slope flat in blocks?
        if len(rows) >= 3:
            lo, hi = rows[0], rows[-1]
            if lo["w_med"] and hi["w_med"] and lo["ms"]:
                blk_ratio = hi["w_med"] / lo["w_med"]
                ms_ratio = hi["ms"] / lo["ms"]
                print("   blocks x%.1f, bt time x%.1f -- pre-T1 these would"
                      % (blk_ratio, ms_ratio))
                print("   have been equal, since the cost was per block.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
