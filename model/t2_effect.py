#!/usr/bin/env python3
"""What did T2 actually buy, measured on both paths?

T2 defers write_token's host-side mirror scatter to the next step behind one
event wait, so 2L BLOCKING device-to-host copies per step become 2L async ones.
Phase 0 priced the synchronisation it removes at 8.47 ms of F. This reads the
bracketed before/after and says what was realised.

BRACKETED, because the effect is ~8 ms and F drifts ~0.7 ms per hour of GPU
work: async_a / blocking / async_b, so drift is bounded INSIDE the comparison
rather than by appeal to a separate measurement. Both arms are numerically
valid -- unlike the probe and skip arms of Phase 0, both of these update the
mirror -- so the purity gate applies normally and the step times are real.

Phase 0's 8.47 ms is the CEILING for T2, not its prediction: that arm skipped
the scatter entirely, and T2 still performs it, just later. probe - skip put the
async copy at 0.98 ms, and the deferred scatter is 2L CPU index_copy_ calls on
top. So the expected realisation is roughly 8.47 minus the scatter.

    ./.venv/bin/python paper/model/t2_effect.py
"""
import json
import os
import sys

DATA = os.environ.get(
    "OPT_DATA", "/mnt/beegfs/benishay/paper_runs/20260930/overhead/09_t2")

PHASE0_PRIZE = 8.47      # real - probe, the synchronisation ceiling
PHASE0_ASYNC = 0.98      # probe - skip, the async copy a fix still pays

GROUPS = [
    ("ctx 512", [("T2_async_a", "T2 on, before"),
                 ("T2_blocking", "T2 OFF, the old blocking path"),
                 ("T2_async_b", "T2 on, after")]),
    ("ctx 2048", [("T2_c2048_async", "T2 on"),
                  ("T2_c2048_blocking", "T2 OFF")]),
]


def _ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    f = my - c * mx
    return f, c, max(abs(f + c * x - y) for x, y in zip(xs, ys))


def fit(tag):
    path = os.path.join(DATA, tag + ".json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    rows = [r for r in d["rows"] if r.get("usable")]
    base = [r for r in rows if r["riders"] == 0]
    if not base:
        return None
    t0 = base[0]["step_ms_direct"] or base[0]["step_ms"]
    xs, ys, lk = [], [], None
    for r in sorted(rows, key=lambda r: r["riders"]):
        n = r["riders"]
        if not n:
            continue
        lk = r["link_ms"] / n
        t = r["step_ms_direct"] or r["step_ms"]
        if t is not None and n < t0 / lk:
            xs.append(n)
            ys.append(t - t0)
    if len(xs) < 2:
        return None
    F, c, res = _ols(xs, ys)
    return {"F": F, "c": c, "resid": res, "n": len(xs), "t0": t0,
            "async": d.get("d2h_async"),
            "valid": d.get("numerically_valid"),
            "usable": "%s/%s" % (d.get("n_rows_usable"), d.get("n_rows"))}


def main():
    print("=" * 78)
    print(" T2: what the deferred mirror write bought")
    print("=" * 78)

    for label, arms in GROUPS:
        got = {}
        print()
        print(" %s" % label)
        print("   %-20s %-30s %8s %8s %8s %8s"
              % ("arm", "path", "F", "c", "T0", "rows"))
        for tag, desc in arms:
            a = fit(tag)
            if a is None:
                print("   %-20s %-30s %8s" % (tag, desc, "pending"))
                continue
            got[tag] = a
            print("   %-20s %-30s %8.2f %8.3f %8.2f %8s"
                  % (tag, desc, a["F"], a["c"], a["t0"], a["usable"]))

        on = [t for t, _ in arms if t in got and got[t]["async"]]
        off = [t for t, _ in arms if t in got and got[t]["async"] is False]
        if not on or not off:
            continue

        f_on = sum(got[t]["F"] for t in on) / len(on)
        f_off = got[off[0]]["F"]
        saved = f_off - f_on
        print()
        print("   F blocking %.2f -> async %.2f    SAVED %.2f ms"
              % (f_off, f_on, saved))
        if len(on) == 2:
            drift = abs(got[on[0]]["F"] - got[on[1]]["F"])
            print("   bracket drift %.2f ms, a ratio of %.1fx against the effect"
                  % (drift, (saved / drift) if drift else float("inf")))
            if drift and saved / drift < 3:
                print("   NOT A CLEAN SEPARATION -- do not act on this without")
                print("   repeating the bracket.")
        # c should barely move: the mirror write is per-layer
        cs = [got[t]["c"] for t, _ in arms if t in got]
        spread = (max(cs) - min(cs)) / (sum(cs) / len(cs))
        print("   c spread %.0f%% across the arms -- %s"
              % (100 * spread,
                 "consistent with a per-LAYER cost" if spread < 0.25
                 else "larger than a per-layer cost should give"))

        if label == "ctx 512":
            print()
            print("   against Phase 0: the synchronisation ceiling was %.2f ms"
                  % PHASE0_PRIZE)
            print("   realised %.2f ms, i.e. %.0f%% of it. The shortfall is the"
                  % (saved, 100 * saved / PHASE0_PRIZE))
            print("   deferred scatter, which T2 still performs and the probe")
            print("   arm did not -- 2L CPU index_copy_ calls per step.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
