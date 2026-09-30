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

INSTRUMENTATION CLASS: A (non-perturbing). This wraps `Scheduler.schedule()`
and does one dict pass plus two `len()` calls per step. No CUDA calls, no
events, no synchronisation, nothing that can reorder the stream. Steps are tens
of milliseconds; this is nanoseconds. It is safe to leave on for every run,
which is the point -- a gate you have to remember to switch on is a gate that
will be off for the run that needed it.

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
from dataclasses import dataclass, field

print = functools.partial(print, flush=True)  # noqa: A001

# Steps to ignore at each end of a window when judging steady state. Admission
# ramps up over the first few steps and drains at the end; those steps are real
# but they are not the steady cohort, and the step-time estimator already
# cancels them by differencing two generation lengths.
RAMP_STEPS = 3


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

    def _steady(self) -> list:
        if len(self.steps) <= 2 * RAMP_STEPS:
            return list(self.steps)
        return self.steps[RAMP_STEPS:-RAMP_STEPS]

    def summary(self) -> dict:
        st = self._steady()
        if not self.steps:
            return {"n_steps": 0, "n_steady": 0}
        run = [s["running"] for s in st] or [0]
        return {
            "n_steps": len(self.steps),
            "n_steady": len(st),
            # prefill leakage, over the WHOLE window: a single prefill step
            # anywhere in it is a contamination, ramp or not.
            "prefill_steps": sum(1 for s in self.steps if s["extra"] > 0),
            "prefill_tokens": sum(s["extra"] for s in self.steps),
            # queueing, whole window for the same reason
            "max_waiting": max(s["waiting"] for s in self.steps),
            # cohort stability, steady portion only
            "running_min": min(run),
            "running_max": max(run),
            "running_mode": max(set(run), key=run.count),
            "hbm_max": max((s["hbm"] for s in st), default=0),
            "cpu_max": max((s["cpu"] for s in st), default=0),
            "hbm_mode": max({s["hbm"] for s in st} or {0},
                            key=[s["hbm"] for s in st].count),
            "cpu_mode": max({s["cpu"] for s in st} or {0},
                            key=[s["cpu"] for s in st].count),
        }

    def verdict(self, b_res: int, riders: int) -> dict:
        """The four conditions of the P/D decode-purity gate."""
        s = self.summary()
        if not s["n_steps"]:
            return {"ok": False, "violations": ["no steps recorded"], **s}

        v = []
        if s["prefill_tokens"] > 0:
            v.append(
                f"prefill leaked into the window: {s['prefill_tokens']} tokens "
                f"over {s['prefill_steps']} steps (need --connector for "
                f"decode-node emulation)")
        if s["max_waiting"] > 0:
            v.append(
                f"requests queued: max_waiting={s['max_waiting']} (wall time "
                f"includes queue drain, so the step estimate is not a step)")
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
                f"batch is not B_res+r: running_mode={s['running_mode']} "
                f"(min {s['running_min']}, max {s['running_max']}), "
                f"expected {want}")

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
