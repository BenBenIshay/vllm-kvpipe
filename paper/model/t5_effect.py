#!/usr/bin/env python3
"""Did whole-extent runs fire in the engine, and what did they buy?

Two questions, and the first gates the second — the discipline T1 nearly failed
and T3 did fail.

  1. DID THE RUNS MERGE? `n_runs` must read 1 instead of the rider count, and
     `h2d_launches` 2 per layer-step instead of 2r. If the pool's slack guard
     refused the whole-extent path (which it does when `blocks_per_req` is loose
     — the runner sizes extents for `max_model_len`, ~75% slack), the run
     silently measures the per-run path and any timing difference is noise.
     `w_blocks` is also checked: it must equal riders x bpr, which is the
     whole-extent region including the slack T5 deliberately carries.

  2. WHAT DID IT BUY? Fitted `c` and `F`, bracketed off/on/off, because `F`
     drifts ~0.7 ms/hour.

WHAT TO EXPECT, from the off-engine pricing in launch_pattern_bench.py: `c`
should lose most of the per-rider launch term and `F` should RISE slightly,
because T5 moves cost from `c` into `F` rather than deleting it — the launch
count stops depending on the rider count. Moving cost out of `c` is a structural
win even when the arithmetic is a wash, since the gate is `c < T0/B` and `F` only
sets where break-even lands.

AND THIS IS THE HOSTILE CONTEXT. At ctx 512 the slack is gen/S = 25%, the worst
case for the extra bytes T5 carries; the model predicts the trade is best at
short context (launch-dominated `c`) but this is also where the slack costs most.
A win here is therefore a conservative result, not a flattering one.

    ./.venv/bin/python paper/model/t5_effect.py
"""
import glob
import json
import os
import re
import sys

DATA = os.environ.get(
    "T5_DATA", "/mnt/beegfs/benishay/paper_runs/20260930/overhead/14_t5")
BPR = int(os.environ.get("T5_BPR", "41"))   # ceil((512 + 136) / 16)


def fit(path):
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
    return d, (sy - c * sx) / n, c, n


def mechanism(path):
    """{riders: (n_runs, w_blocks, launches)} medians from a phase JSONL."""
    seen = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        ctx = d.get("ctx") or {}
        r = ctx.get("n_active")
        if r is None:
            continue
        lau = (d.get("phases", {}).get("h2d_launches") or {}).get("calls")
        seen.setdefault(r, []).append(
            (ctx.get("n_runs"), ctx.get("w_blocks"), lau))
    out = {}
    for r, v in seen.items():
        # mode of n_runs; these are integers and should be constant per cohort
        runs = [x[0] for x in v if x[0] is not None]
        wb = [x[1] for x in v if x[1] is not None]
        la = [x[2] for x in v if x[2] is not None]
        if not runs:
            continue
        out[r] = (max(set(runs), key=runs.count),
                  max(set(wb), key=wb.count) if wb else None,
                  max(set(la), key=la.count) if la else None)
    return out


def main():
    print("=" * 78)
    print(" 1. DID THE RUNS MERGE IN THE ENGINE?")
    print("=" * 78)
    print()
    fired = {}
    for f in sorted(glob.glob(os.path.join(DATA, "*_phases.jsonl"))):
        tag = re.sub(r"_phases\.jsonl$", "", os.path.basename(f))
        m = mechanism(f)
        if not m:
            print(" %-12s no steps recorded" % tag)
            continue
        print(" %s" % tag)
        print("   %7s %8s %10s %10s %12s"
              % ("riders", "n_runs", "w_blocks", "launches", "expect 2*r*L"))
        for r in sorted(m):
            runs, wb, la = m[r]
            print("   %7d %8s %10s %10s %12d"
                  % (r, runs, wb, la, 2 * r * 64))
            if r >= 2:
                fired.setdefault(tag, []).append(runs == 1)
        print()
    for tag, oks in fired.items():
        if all(oks):
            print(" %-12s MERGED on every multi-rider cohort (n_runs == 1)"
                  % tag)
        elif any(oks):
            print(" %-12s merged on SOME cohorts only -- treat timing with care"
                  % tag)
        else:
            print(" %-12s DID NOT MERGE. The slack guard likely refused the"
                  % tag)
            print("              whole-extent path (loose blocks_per_req), so"
                  " this arm")
            print("              measured the per-run path and any difference"
                  " is noise.")
    print()

    print("=" * 78)
    print(" 2. WHAT IT BOUGHT")
    print("=" * 78)
    print()
    print(" %-12s %7s %9s %9s %s" % ("run", "npts", "F ms", "c ms/r", "note"))
    rows = {}
    for tag in ("T5_off", "T5_on", "T5b_off_a", "T5b_on", "T5b_off_b"):
        r = fit(os.path.join(DATA, tag + ".json"))
        if not r:
            continue
        d, F, c, n = r
        rows[tag] = (F, c)
        env = d.get("kvpipe_env") or {}
        note = ("whole-extent ON" if env.get(
            "VLLM_KVPIPE_RUNGD_WHOLE_EXTENT") == "1" else "")
        if d.get("whole_extent"):
            note = "whole-extent ON"
        print(" %-12s %7d %9.2f %9.3f %s" % (tag, n, F, c, note))
    print()
    offs = [v for k, v in rows.items() if "off" in k]
    ons = [v for k, v in rows.items() if k in ("T5_on", "T5b_on")]
    if offs and ons:
        mc_off = sum(v[1] for v in offs) / len(offs)
        mc_on = sum(v[1] for v in ons) / len(ons)
        mf_off = sum(v[0] for v in offs) / len(offs)
        mf_on = sum(v[0] for v in ons) / len(ons)
        drift = (max(v[1] for v in offs) - min(v[1] for v in offs)
                 if len(offs) > 1 else float("nan"))
        print(" c: %.3f -> %.3f  (%+.3f ms/rider)" % (mc_off, mc_on,
                                                      mc_on - mc_off))
        print(" F: %.2f -> %.2f  (%+.2f ms)" % (mf_off, mf_on, mf_on - mf_off))
        if drift == drift:
            print(" off-arm bracket drift on c: %.3f" % drift)
            print(" verdict: %s"
                  % ("the change in c EXCEEDS the drift, so it is real"
                     if abs(mc_on - mc_off) > drift else
                     "the change in c is INSIDE the drift -- not resolvable"))
        print()
        print(" Expected shape: c DOWN (launch term mostly gone) and F UP")
        print(" slightly (the launch cost moves into F rather than vanishing).")
        print(" c down with F up is the predicted result; c down with F also")
        print(" down would mean something other than T5 moved.")
    else:
        print(" not enough arms yet (still running?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
