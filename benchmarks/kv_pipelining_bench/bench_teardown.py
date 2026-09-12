#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Kill an engine so that SLURM does not have to.

On 2026-09-06 nova06 and nova07 were both drained with reason **"Kill task
failed"** — SLURM's way of saying a job's processes would not die. Both are
nodes this suite had been running on all day. The cause is visible in the
harnesses: twenty of them tear an engine down with

    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    time.sleep(4)

one polite signal, a fixed wait, and then an assumption. A vLLM engine wedged
in a CUDA call ignores SIGTERM, the harness moves on, the process outlives the
job step, and the node leaves service.

Losing a lab node for a day is the cheap version. The expensive version is a
rented GH200 that must be released and re-provisioned mid-experiment.

**Checking that the parent exited is not enough, and that difference is the
whole bug.** vLLM's API server is the parent; the EngineCore worker is its
child in the same process group. The parent dies on SIGTERM and the teardown
looks clean while the worker, stuck in CUDA, survives — and the survivor is
what SLURM cannot kill. Verified by test: a parent that terminates while its
child ignores SIGTERM leaves a process behind, and a parent-only check calls
that success.

So this signals the GROUP, verifies the GROUP is empty, escalates, verifies
again, and says so loudly when something survives SIGKILL — that state is what
precedes a drain, and it is far better learned at the end of a run than from
`sinfo -R`.
"""

from __future__ import annotations

import os
import signal
import time


def _reap(proc) -> None:
    """Collect the exit status so the child does not linger as a zombie.

    Matters here beyond tidiness: a zombie still occupies the process table, so
    an unreaped parent makes the group look occupied forever.
    """
    try:
        proc.wait(timeout=5)
    except Exception:                                          # noqa: BLE001
        pass


def terminate_engine(proc, grace_s: float = 15.0, kill_s: float = 10.0,
                     label: str = "engine", verbose: bool = True) -> str:
    """Terminate `proc`'s whole process group and confirm it is gone.

    Returns "already-dead", "term", "kill", or "SURVIVED-SIGKILL". The last is
    not recoverable from inside the harness -- a process in uninterruptible
    sleep cannot be killed at all -- so it is reported rather than swallowed,
    and the run should not be read as clean.
    """
    if proc is None or proc.poll() is not None and _no_group(proc):
        return "already-dead"
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        _reap(proc)
        return "already-dead"

    def group_empty() -> bool:
        if proc.poll() is not None:
            _reap(proc)
        try:
            os.killpg(pgid, 0)          # signal 0 is an existence probe
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    def empty_within(seconds: float) -> bool:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if group_empty():
                return True
            time.sleep(0.25)
        return group_empty()

    if group_empty():
        return "already-dead"

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return "already-dead"
    except Exception:                                          # noqa: BLE001
        pass
    if empty_within(grace_s):
        return "term"

    if verbose:
        print(f"  [teardown] {label}: process group {pgid} still populated "
              f"{grace_s:.0f}s after SIGTERM, sending SIGKILL", flush=True)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return "already-dead"
    except Exception:                                          # noqa: BLE001
        pass
    if empty_within(kill_s):
        return "kill"

    print(f"  [teardown] !! {label}: process group {pgid} SURVIVED SIGKILL. A "
          "process in uninterruptible sleep (typically inside a CUDA call) "
          "cannot be killed, and SLURM will drain this node with 'Kill task "
          "failed' when the step ends. Treat this run's results as suspect and "
          "check `sinfo -R`.", flush=True)
    return "SURVIVED-SIGKILL"


def _no_group(proc) -> bool:
    try:
        os.killpg(os.getpgid(proc.pid), 0)
    except Exception:                                          # noqa: BLE001
        return True
    return False
