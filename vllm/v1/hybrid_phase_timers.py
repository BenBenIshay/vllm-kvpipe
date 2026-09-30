# SPDX-License-Identifier: Apache-2.0
"""Per-phase timers for the rung-d host-tier path.

WHY. The residual `F` -- the per-step cost of using the host path at all -- is
attributed in the paper to "roughly 640 extra kernel launches per step, about
ten per layer", on the evidence of an operator trace taken at ONE rider count.
Nothing in the rung-d hot path is instrumented: `DecodeHookProfile` only sees
the connector's hooks, and rung-d does its work in the attention impl and the
pool, which those hooks never touch. So the breakdown of `F` has never been
measured, only inferred.

WHAT CLASS OF INSTRUMENT THIS IS, which matters more than what it measures.

  Class A -- host issue time (`VLLM_KVPIPE_PHASE_TIMERS=1`).
      `perf_counter_ns` around each phase. No CUDA call, no event, no
      synchronisation, nothing that can reorder the stream or drain the
      pipeline. For a launch-bound cost this is the RIGHT primary metric: the
      thing we are trying to size IS host time spent issuing work.

  Class B -- device time. **NOT IMPLEMENTED, deliberately.**
      `VLLM_KVPIPE_PHASE_TIMERS=2` sets a `dev` flag that nothing reads, so
      level 2 executes exactly the level-1 path. That was originally an
      oversight -- the docstring described CUDA event pairs that were never
      written -- and it is left unimplemented on purpose, for two reasons.
      Host issue time is the right primary metric for a launch-bound cost, and
      the byte-versus-launch split that device timing would be used for is
      already measured directly by the three XFER_MODE arms
      (`real - minbytes` is the byte cost, `minbytes - nobus` the launch cost),
      without any stream operation at all.

      The oversight paid for itself: because level 2 is level 1, the t1/t2 arms
      of M1 were an unintended PURE REPLICATE, which is how the run-to-run
      noise floor got measured -- +/-0.47% per step, and +/-2.9% on a fitted
      `F`, +/-8% on a fitted `c` at six rider points. Without it, the
      instrumentation's 0.69% per-step effect would have been mistaken for a
      coefficient shift larger than noise.

      If device timing is ever needed, record events on the current stream and
      drain them non-blocking via `Event.query()`; never wait inline.

This distinction is not pedantry. This repo's existing profiler has misled the
same investigation twice, in opposite directions:

  * it put `torch.cuda.synchronize()` inside its timing wrapper, which made
    async double-buffering look worthless;
  * its CPU column charged `write_token` for the model's own compute, producing
    a "21 ms blocking D2H" that was worth ~4 ms at a realistic queue depth.

So: nothing here waits inline, and every number this produces is cross-checked
against an operator-count diff (which is Class A by construction) before it is
believed. Absolute coefficients are fitted from UNTIMED runs; these timers
supply the shares.

OUTPUT. One JSON object per decode step to the path in
`VLLM_KVPIPE_PHASE_TIMERS_OUT` (default: stderr as `KVPIPE_PHASE ...`), holding
the phase name -> (calls, host_ns) map plus the step's rider/layer context.

Usage in the hot path -- two calls, guarded by one attribute load when off:

    from vllm.v1.hybrid_phase_timers import PT
    t = PT.now()            # returns None when disabled
    ...work...
    PT.add("issue_runs", t) # no-op when t is None
"""

from __future__ import annotations

import json
import os
import sys
import time


class _PhaseTimers:
    """Accumulates host (and optionally device) time per named phase."""

    __slots__ = ("level", "on", "dev", "_acc", "_ev", "_steps", "_out",
                 "_ctx", "_fh", "_every", "_n")

    def __init__(self) -> None:
        try:
            self.level = int(os.environ.get("VLLM_KVPIPE_PHASE_TIMERS", "0"))
        except ValueError:
            self.level = 0
        self.on = self.level >= 1
        # Set but intentionally unread: see the class-B note in the module
        # docstring. Level 2 is level 1.
        self.dev = self.level >= 2
        self._acc: dict[str, list] = {}
        self._ev: list = []
        self._ctx: dict = {}
        self._out = os.environ.get("VLLM_KVPIPE_PHASE_TIMERS_OUT", "")
        self._fh = None
        try:
            self._every = max(1, int(
                os.environ.get("VLLM_KVPIPE_PHASE_TIMERS_EVERY", "1")))
        except ValueError:
            self._every = 1
        self._n = 0
        if self.on:
            note = " (level 2 == level 1: device timing not implemented)" \
                if self.dev else ""
            print(f"*** hybrid phase timers ON, level={self.level}, host time "
                  f"only{note}, every={self._every}", flush=True)

    # -- hot path ---------------------------------------------------------
    def now(self):
        """Timestamp, or None when disabled (so `add` becomes a no-op)."""
        if not self.on:
            return None
        return time.perf_counter_ns()

    def add(self, name: str, t0) -> None:
        if t0 is None:
            return
        dt = time.perf_counter_ns() - t0
        a = self._acc.get(name)
        if a is None:
            self._acc[name] = [1, dt]
        else:
            a[0] += 1
            a[1] += dt

    def count(self, name: str, n: int = 1) -> None:
        """A pure counter -- launches issued, runs, blocks. Always class A."""
        if not self.on:
            return
        a = self._acc.get(name)
        if a is None:
            self._acc[name] = [n, 0]
        else:
            a[0] += n

    def context(self, **kw) -> None:
        """Per-step context: rider count, layer count, block count, bytes."""
        if not self.on:
            return
        self._ctx.update(kw)

    # -- step boundary ----------------------------------------------------
    def step_end(self) -> None:
        """Emit this step's record and reset. Called once per decode forward."""
        if not self.on or not self._acc:
            return
        self._n += 1
        if self._n % self._every:
            self._acc.clear()
            self._ctx.clear()
            return
        rec = {
            "step": self._n,
            "ctx": dict(self._ctx),
            "phases": {k: {"calls": v[0], "host_ms": v[1] / 1e6}
                       for k, v in self._acc.items()},
            "host_total_ms": sum(v[1] for v in self._acc.values()) / 1e6,
        }
        self._acc.clear()
        self._ctx.clear()
        line = json.dumps(rec, separators=(",", ":"))
        if self._out:
            if self._fh is None:
                self._fh = open(self._out, "a")           # noqa: SIM115
            self._fh.write(line + "\n")
            self._fh.flush()
        else:
            print("KVPIPE_PHASE " + line, file=sys.stderr, flush=True)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


PT = _PhaseTimers()
