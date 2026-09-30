#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Is a measured window actually pure decode, on the tier split we asked for?

WHY THIS EXISTS. Every rider-cost number is a difference of step times, and the
difference is only a rider cost if the two runs differ ONLY in riders. Three
ways that has silently failed in this repo:

1. **Local prefill leaked into the window.** This is a P/D decode-node feature.
   A co-located run at ctx 2048 spent 58-68% of its wall clock on prefill, and
   the apparent loss was mostly prefill slowing down. `--connector` is what
   makes the node decode-only -- and it is OFF by default, so the correct setup
   is the one you have to remember to ask for.

2. **Refused riders became residents.** `max_num_seqs = batch + max(riders)`, so
   a request the controller declines to make a rider is still served, as a
   RESIDENT. The batch grows, the step lengthens, and the harness charges it to
   riders. `paper_runs/20260916/rider_curve/curve_marginal_eta.json` is this:
   r=4 and r=8 admitted ZERO riders (`stream_gb_per_step = 0.0`,
   `tiers = {hbm: 16}`) while the step went 53.5 -> 101.5 -> 103.9 ms.

3. **The rider count was not the requested one.** Without the connector a rider
   cannot be created while HBM is full (its prefill needs resident blocks), so
   riders cap at 2 however many are offered -- and `curve_32b_v2.json` reports
   r=4/6/8 rows whose `stream_gb_per_step` FALLS from 1.115 to 0.543, i.e.
   fewer bytes at a higher nominal rider count.

So purity is ASSERTED per step, not assumed from the flags. A point that fails
is discarded and logged, never adjusted.

INSTRUMENTATION CLASS: A (non-perturbing), and MEASURED rather than asserted.
This wraps `Scheduler.schedule()` and per step does one `perf_counter_ns`, one
pass over the scheduled-token dict, one tally of the tier table and two `len()`
calls. No CUDA call, no event, no synchronisation, nothing that can reorder the
stream.

Cost, measured on this machine at a realistic 22-request batch:

    perf_counter_ns()      439 ns per call, once per step
    whole _record() body   28.5 us per step
      -> 0.053% of a 53.5 ms step, 0.063% of a 45 ms one

Forty times below the 2% perturbation gate, and the clock read is the cheapest
part of it. The end-to-end bound is better still: with this monitor running on
every step, M0 arm A reproduced the published curve -- taken with no monitor at
all -- to 0.35%, which is inside run-to-run noise.

So it is safe to leave on for every run, which is the point: a gate you have to
remember to switch on is a gate that will be off for the run that needed it.

Usage:
    from decode_purity import DecodePurityMonitor
    mon = DecodePurityMonitor(llm); mon.attach()
    ...
    mon.start_window()
    llm.generate(...)
    w = mon.stop_window()
    v = w.verdict(b_res=16, riders=2)
    if not v["ok"]: ...discard...
"""

from __future__ import annotations

import functools
import statistics
import time
from dataclasses import dataclass, field

print = functools.partial(print, flush=True)  # noqa: A001

# The steady window is found from the data (see Window.ramp_end), not by
# trimming a fixed number of steps: the ramp length depends on the rider count
# -- measured at about 2(r-1) extra steps -- so a constant trim is either too
# small at high r or discards real steps at low r.


def _sched_of(llm):
    core = llm.llm_engine.engine_core
    inner = getattr(core, "engine_core", core)
    return inner.scheduler


def _tier_counts(sched) -> dict:
    """Requests per tier, from the admission controller's own table."""
    try:
        adm = sched.transfer_aware_admission
        tiers = getattr(adm, "_req_tier", {}) or {}
    except Exception:                                          # noqa: BLE001
        return {}
    out: dict = {}
    for t in tiers.values():
        out[t] = out.get(t, 0) + 1
    return out


@dataclass
class Window:
    """Per-step records for one measured generation."""

    steps: list = field(default_factory=list)

    # Each step is a dict:
    #   running, waiting, tokens, extra (tokens beyond 1/req), hbm, cpu

    def ramp_end(self, b_res: int, riders: int) -> int | None:
        """First step at which the requested cohort is fully admitted.

        Admission STAGGERS: at r riders the engine queues up to r-1 requests
        while it works up to the full cohort, so the first several steps run a
        smaller batch. Those steps are legitimate -- the estimator is built to
        cancel them -- but only if BOTH the short and the long run clear the
        ramp. That is checked in `measure_step`, not here; this just reports
        where the ramp ends.
        """
        want = b_res + riders
        for i, s in enumerate(self.steps):
            if (s["waiting"] == 0 and s["running"] == want
                    and s["hbm"] == b_res and s["cpu"] == riders):
                return i
        return None

    def _steady(self, b_res: int, riders: int) -> list:
        """Steps from the end of the ramp to the last full-cohort step.

        The trailing steps are the drain, where requests have finished and the
        batch shrinks; they are as legitimate and as irrelevant as the ramp.
        """
        start = self.ramp_end(b_res, riders)
        if start is None:
            return []
        want = b_res + riders
        end = None
        for i in range(len(self.steps) - 1, start - 1, -1):
            if self.steps[i]["running"] == want:
                end = i
                break
        if end is None:
            return []
        return self.steps[start:end + 1]

    def step_ms_direct(self, b_res: int, riders: int) -> dict:
        """Step time measured DIRECTLY over the steady window, no differencing.

        WHY THIS EXISTS. The established estimator differences two generation
        lengths and relies on everything that is not a decode step appearing
        identically in both runs so it cancels. Ben's objection to leaning on
        that: a cost which is out of scope should be *impossible* here, not
        merely argued to subtract. And the argument does have a hole -- riders
        are seeded when they are ADMITTED, i.e. during the ramp, so a short run
        that ends before the last rider is admitted pays less seeding than the
        long run and the difference lands in the numerator. Same root cause as
        the ramp residue itself.

        Since the monitor already runs on every step, the interval between
        consecutive `schedule()` calls IS the step period. Taking the median of
        those over the steady window only means the ramp, the drain, the
        seeding and the prefill are excluded by CONSTRUCTION: they are not in
        the window, so there is nothing to cancel.

        CAVEAT, and it is why this does not simply replace the other estimator:
        this is the host-side step period. vLLM v1's engine loop is synchronous
        per step (it waits for sampled token ids), so it tracks the true step
        time closely -- but the two estimators measure subtly different things
        and are reported side by side. Agreement between them is the evidence;
        a divergence is a finding in itself.
        """
        st = self._steady(b_res, riders)
        ts = [s["t"] for s in st if s.get("t") is not None]
        if len(ts) < 4:
            return {"step_ms_direct": None, "n_intervals": max(0, len(ts) - 1)}
        d = sorted((ts[i + 1] - ts[i]) / 1e6 for i in range(len(ts) - 1))
        n = len(d)
        return {
            "step_ms_direct": statistics.median(d),
            "n_intervals": n,
            "step_ms_direct_p10": d[int(0.10 * n)],
            "step_ms_direct_p90": d[min(n - 1, int(0.90 * n))],
            # Spread matters: a wide one means the "steady" window is not
            # steady, whatever the counters say.
            "step_ms_direct_iqr": d[min(n - 1, int(0.75 * n))] - d[int(0.25 * n)],
        }

    def summary(self, b_res: int = 0, riders: int = 0) -> dict:
        if not self.steps:
            return {"n_steps": 0, "n_steady": 0}
        st = self._steady(b_res, riders)
        run = [s["running"] for s in st] or [0]
        return {
            "n_steps": len(self.steps),
            "n_steady": len(st),
            "ramp_end": self.ramp_end(b_res, riders),
            # prefill leakage, over the WHOLE window: a single prefill step
            # anywhere in it is a contamination, ramp or not.
            "prefill_steps": sum(1 for s in self.steps if s["extra"] > 0),
            "prefill_tokens": sum(s["extra"] for s in self.steps),
            # queueing: reported for the whole window (so the ramp is visible)
            # AND for the steady window, which is the one that must be clean.
            "max_waiting": max(s["waiting"] for s in self.steps),
            "max_waiting_steady": max((s["waiting"] for s in st), default=0),
            "running_min": min(run),
            "running_max": max(run),
            "running_mode": max(set(run), key=run.count),
            "hbm_max": max((s["hbm"] for s in st), default=0),
            "cpu_max": max((s["cpu"] for s in st), default=0),
            "hbm_mode": max({s["hbm"] for s in st} or {0},
                            key=[s["hbm"] for s in st].count),
            "cpu_mode": max({s["cpu"] for s in st} or {0},
                            key=[s["cpu"] for s in st].count),
            # The differencing-free estimator. Excludes ramp, drain, seeding and
            # prefill by construction rather than by cancellation.
            **self.step_ms_direct(b_res, riders),
        }

    def verdict(self, b_res: int, riders: int) -> dict:
        """The conditions of the P/D decode-purity gate.

        Queueing is judged on the STEADY window, not the whole run. Queueing
        during the initial admission ramp is expected (`max_waiting = r-1` at r
        riders) and cancels in the differencing estimator -- provided both runs
        clear the ramp, which `measure_step` checks separately. Queueing that
        persists into steady state is fatal, because then the batch is not the
        one being charged.
        """
        s = self.summary(b_res, riders)
        if not s["n_steps"]:
            return {"ok": False, "violations": ["no steps recorded"], **s}

        v = []
        if s["prefill_tokens"] > 0:
            v.append(
                f"prefill leaked into the window: {s['prefill_tokens']} tokens "
                f"over {s['prefill_steps']} steps (need --connector for "
                f"decode-node emulation)")
        if s["ramp_end"] is None:
            v.append(
                f"the requested cohort was NEVER fully admitted: no step had "
                f"waiting=0 with running={b_res + riders}, hbm={b_res}, "
                f"cpu={riders} (peak seen: hbm={s['hbm_max']}, "
                f"cpu={s['cpu_max']})")
            return {"ok": False, "violations": v, **s}
        if s["n_steady"] < 8:
            v.append(
                f"only {s['n_steady']} steady steps after the ramp (ends at "
                f"step {s['ramp_end']} of {s['n_steps']}): too few to be a "
                f"step-time measurement -- raise --long-len")
        if s["max_waiting_steady"] > 0:
            v.append(
                f"requests queued IN STEADY STATE: "
                f"max_waiting={s['max_waiting_steady']} -- the batch is not "
                f"the one being charged")
        if s["cpu_mode"] != riders:
            v.append(
                f"rider count is not the requested one: cpu_mode="
                f"{s['cpu_mode']} (max {s['cpu_max']}), requested {riders}")
        if s["hbm_mode"] != b_res:
            v.append(
                f"resident cohort is not the requested one: hbm_mode="
                f"{s['hbm_mode']} (max {s['hbm_max']}), requested {b_res} "
                f"-- refused riders may be running as residents")
        want = b_res + riders
        if s["running_mode"] != want:
            v.append(
                f"batch is not B_res+r in steady state: running_mode="
                f"{s['running_mode']} (min {s['running_min']}, max "
                f"{s['running_max']}), expected {want}")

        return {"ok": not v, "violations": v, **s}


class DecodePurityMonitor:
    """Wraps `Scheduler.schedule()` to record per-step purity evidence."""

    def __init__(self, llm):
        self.sched = _sched_of(llm)
        self._orig = None
        self._win: Window | None = None

    def attach(self) -> None:
        if self._orig is not None:
            return
        sched = self.sched
        self._orig = sched.schedule

        @functools.wraps(self._orig)
        def wrapped(*a, **kw):
            out = self._orig(*a, **kw)
            win = self._win
            if win is not None:
                try:
                    self._record(out)
                except Exception:                              # noqa: BLE001
                    # Never let the gate break the run it is watching.
                    pass
            return out

        sched.schedule = wrapped

    def detach(self) -> None:
        if self._orig is not None:
            self.sched.schedule = self._orig
            self._orig = None

    def _record(self, out) -> None:
        # Named t_ns, not t: the token-count comprehension below binds `t` as its
        # loop variable. Python 3 scopes that separately so it does not actually
        # clobber this, but the collision is a trap and cost one run already.
        t_ns = time.perf_counter_ns()
        nst = getattr(out, "num_scheduled_tokens", None) or {}
        total = int(getattr(out, "total_num_scheduled_tokens", 0) or 0)
        if not nst and not total:
            return  # an empty schedule step carries no evidence either way
        # A decode step gives every running request exactly one token. Anything
        # beyond that is prefill (or spec-decode, which this feature does not
        # use).
        extra = sum(max(0, int(t) - 1) for t in nst.values())
        tiers = _tier_counts(self.sched)
        try:
            waiting = len(self.sched.waiting)
        except Exception:                                      # noqa: BLE001
            waiting = -1
        self._win.steps.append({
            "t": t_ns,          # feeds step_ms_direct; omitted once, silently
            "running": len(self.sched.running),
            "waiting": waiting,
            "tokens": total,
            "extra": extra,
            "hbm": int(tiers.get("hbm", 0)),
            "cpu": int(tiers.get("cpu", 0)),
        })

    def start_window(self) -> None:
        self._win = Window()

    def stop_window(self) -> Window:
        w = self._win or Window()
        self._win = None
        return w


def _selftest() -> int:
    """Synthetic windows covering every condition and the real failure shapes.

    Run: python decode_purity.py --selftest
    """
    def mk(ramp, steady, drain, b_res=16, r=6, steady_wait=0, extra=0,
           clean_first=0, hbm=None, cpu=None):
        w = Window()
        want = b_res + r
        for i in range(ramp):
            partial = max(0, r - (ramp - i))
            w.steps.append({"running": b_res + partial, "waiting": r - 1,
                            "tokens": 1, "extra": extra, "hbm": b_res,
                            "cpu": partial})
        for i in range(steady):
            wait = 0 if i < clean_first else steady_wait
            w.steps.append({"running": want, "waiting": wait, "tokens": want,
                            "extra": 0,
                            "hbm": b_res if hbm is None else hbm,
                            "cpu": r if cpu is None else cpu})
        for _ in range(drain):
            w.steps.append({"running": 3, "waiting": 0, "tokens": 3,
                            "extra": 0, "hbm": 3, "cpu": 0})
        return w

    cases = [
        # (label, window, b_res, riders, expect_ok, expect_substring)
        ("ramp queueing then clean steady state",
         mk(10, 126, 10), 16, 6, True, None),
        ("queueing that appears AFTER a clean start",
         mk(10, 126, 10, steady_wait=2, clean_first=20), 16, 6, False,
         "queued IN STEADY STATE"),
        ("queueing on every step: cohort never clean",
         mk(10, 126, 10, steady_wait=2), 16, 6, False,
         "NEVER fully admitted"),
        ("cohort never reaches the requested size",
         mk(10, 0, 10), 16, 6, False, "NEVER fully admitted"),
        ("too few steady steps to be a step time",
         mk(10, 5, 5), 16, 6, False, "too few"),
        ("prefill leaked into the window",
         mk(10, 126, 10, extra=2047), 16, 6, False, "prefill leaked"),
        ("refused riders became residents",
         mk(0, 126, 10, hbm=20, cpu=0), 16, 4, False, "NEVER fully admitted"),
        ("rider count pinned below the request",
         mk(0, 126, 10, hbm=16, cpu=2), 16, 4, False, "NEVER fully admitted"),
        ("clean run with no riders at all",
         mk(0, 130, 6, r=0), 16, 0, True, None),
    ]

    passed = 0
    for label, win, b_res, riders, want_ok, want_sub in cases:
        v = win.verdict(b_res, riders)
        good = (v["ok"] == want_ok)
        if want_sub is not None:
            good = good and any(want_sub in x for x in v["violations"])
        passed += good
        print(f"  [{'ok ' if good else 'BAD'}] {label}")
        if not good:
            print(f"        got ok={v['ok']} violations={v['violations']}")
    # The differencing-free estimator: plant known step intervals and check it
    # recovers them from the STEADY window only, ignoring fast ramp steps and a
    # slow drain. If it averaged the whole window it would be pulled off.
    total = len(cases)
    w = Window()
    t = 0
    for i in range(10):                      # ramp: fast, 20 ms, partial cohort
        t += 20_000_000
        w.steps.append({"t": t, "running": 16 + i - 4 if i >= 4 else 16,
                        "waiting": 5, "tokens": 1, "extra": 0,
                        "hbm": 16, "cpu": max(0, i - 4)})
    for _ in range(100):                     # steady: 50 ms, full cohort
        t += 50_000_000
        w.steps.append({"t": t, "running": 22, "waiting": 0, "tokens": 22,
                        "extra": 0, "hbm": 16, "cpu": 6})
    for _ in range(10):                      # drain: slow, 90 ms, small batch
        t += 90_000_000
        w.steps.append({"t": t, "running": 3, "waiting": 0, "tokens": 3,
                        "extra": 0, "hbm": 3, "cpu": 0})
    d = w.step_ms_direct(16, 6)
    got = d.get("step_ms_direct")
    good = got is not None and abs(got - 50.0) < 0.5
    total += 1
    passed += good
    print(f"  [{'ok ' if good else 'BAD'}] direct estimator recovers the steady "
          f"step (planted 50.0 ms, ramp 20, drain 90) -> "
          f"{'None' if got is None else f'{got:.2f}'} ms "
          f"over {d.get('n_intervals')} intervals")

    # WIRING TEST. The cases above build step dicts by hand, so they exercise
    # `Window` but never `_record` -- which is how a missing "t" key shipped and
    # silently disabled step_ms_direct for a whole run while the selftest stayed
    # green. This drives the real recording path through fakes.
    class _FakeQ(list):
        pass

    class _FakeAdm:
        def __init__(self, tiers):
            self._req_tier = tiers

    class _FakeSched:
        def __init__(self, n_res, n_rid):
            self.running = list(range(n_res + n_rid))
            self.waiting = _FakeQ()
            self.transfer_aware_admission = _FakeAdm(
                {f"r{i}": ("hbm" if i < n_res else "cpu")
                 for i in range(n_res + n_rid)})

        def schedule(self):
            n = len(self.running)
            class _Out:
                num_scheduled_tokens = {f"r{i}": 1 for i in range(n)}
                total_num_scheduled_tokens = n
            return _Out()

    class _FakeLLM:
        def __init__(self, sched):
            inner = type("I", (), {"scheduler": sched})()
            core = type("C", (), {"engine_core": inner})()
            self.llm_engine = type("E", (), {"engine_core": core})()

    sched = _FakeSched(16, 6)
    mon = DecodePurityMonitor(_FakeLLM(sched))
    mon.attach()
    mon.start_window()
    for _ in range(40):
        sched.schedule()
    win = mon.stop_window()
    mon.detach()

    keys = set(win.steps[0]) if win.steps else set()
    need = {"t", "running", "waiting", "tokens", "extra", "hbm", "cpu"}
    v = win.verdict(16, 6)
    wiring_ok = (len(win.steps) == 40 and need <= keys and v["ok"]
                 and v.get("step_ms_direct") is not None)
    total += 1
    passed += wiring_ok
    print(f"  [{'ok ' if wiring_ok else 'BAD'}] _record wiring: {len(win.steps)} "
          f"steps, keys {'complete' if need <= keys else 'MISSING ' + str(need - keys)}"
          f", verdict ok={v['ok']}, step_ms_direct="
          f"{'present' if v.get('step_ms_direct') is not None else 'MISSING'}")

    print(f"\nPURITY GATE SELFTEST {'PASS' if passed == total else 'FAIL'} "
          f"({passed}/{total})")
    return 0 if passed == total else 1


def report(verdict: dict, label: str = "") -> bool:
    """Print a one-or-few-line verdict. Returns True if the point is usable."""
    tag = f"[{label}] " if label else ""
    if verdict["ok"]:
        print(f"{tag}PURITY OK  steps={verdict['n_steps']} "
              f"running={verdict['running_mode']} "
              f"hbm={verdict['hbm_mode']} cpu={verdict['cpu_mode']} "
              f"waiting_max={verdict['max_waiting']} prefill_tok=0")
        return True
    print(f"{tag}PURITY FAIL steps={verdict.get('n_steps')} -- DISCARDING")
    for x in verdict["violations"]:
        print(f"{tag}  - {x}")
    return False


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
