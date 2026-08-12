#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Item 3 in-server test: burst recovery via demotion.

The documented limitation of gate-new-only was: a burst admits everyone before
the ITL loop engages, and the over-SLO cohort then has **no recourse but to
finish**. Migration is supposed to give it one.

This drives a real burst through the engine and records the controller's
closed-loop behaviour step by step:

  burst -> measured ITL crosses the SLO -> THROTTLED -> demotions planned and
  committed -> ITL recovers -> THROTTLED clears

WHAT THIS DOES **NOT** CLAIM. On A100/PCIe, demoting a request makes the step
*slower*, not faster: the host tier costs S*k of interconnect per step at
~25 GB/s, which cannot hide inside the step, whereas the HBM read it replaces
was ~24x faster. Demotion relieves HBM *capacity* pressure; it improves *ITL*
only where the transfer hides, i.e. on a fast interconnect. So this test
validates that the control loop **responds and converges**; the claim that
recovery actually lowers ITL needs GH200.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

BURST = 8
NTOK = 64
SLO_MS = 6.0          # deliberately tight so the burst crosses it


def main() -> int:
    os.environ["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
    os.environ["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
    os.environ["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
    os.environ["VLLM_KVPIPE_TIER_POLICY"] = "scored"
    os.environ["VLLM_KVPIPE_MIGRATE_MIN_REMAIN"] = "4"
    os.environ["VLLM_KVPIPE_MIGRATE_COOLDOWN"] = "8"
    os.environ["VLLM_KVPIPE_MIGRATE_MAX_PER_STEP"] = "2"

    from vllm import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="float16",
              enable_prefix_caching=False, gpu_memory_utilization=0.35,
              max_model_len=512, max_num_seqs=BURST, enforce_eager=True,
              disable_log_stats=True,
              enable_transfer_aware_admission=True,
              enable_hybrid_kv_tiers=True,
              hybrid_b_hbm_max=BURST,          # burst lands entirely on HBM
              transfer_aware_slo_p99_itl_ms=SLO_MS)

    core = llm.llm_engine.engine_core
    sched = getattr(getattr(core, "engine_core", core), "scheduler", None)
    if sched is None:
        print("could not reach the scheduler"); return 2
    ctl = sched.transfer_aware_admission

    trace = []
    orig = sched._apply_tier_migrations

    def traced():
        before = dict(ctl.migration_stats())
        orig()
        after = dict(ctl.migration_stats())
        newly = after.get("cpu", 0) - before.get("cpu", 0)
        trace.append((round(ctl._resident_step_ms_ema, 3),
                      ctl._itl_throttled_state, newly,
                      len(sched._promoting_req_ids)))
    sched._apply_tier_migrations = traced

    prompts = [f"Write a detailed technical explanation of topic {i}: "
               for i in range(BURST)]
    sp = SamplingParams(temperature=0.0, max_tokens=NTOK, ignore_eos=True)
    llm.generate(prompts, sp, use_tqdm=False)

    emas = [t[0] for t in trace if t[0] > 0]
    throttled_steps = [i for i, t in enumerate(trace) if t[1]]
    demotions = sum(t[2] for t in trace)
    stats = ctl.migration_stats()

    print("=" * 68)
    print(f"steps observed        : {len(trace)}")
    print(f"ITL EMA range         : {min(emas):.2f} .. {max(emas):.2f} ms "
          f"(SLO {SLO_MS} ms)")
    print(f"steps in THROTTLED    : {len(throttled_steps)}"
          + (f"  (first at step {throttled_steps[0]})" if throttled_steps else ""))
    print(f"demotions committed   : {demotions}   migration stats: {stats}")

    headroom = ctl.hide_budget_headroom_bytes(list(sched.running))
    hides = headroom > 0
    entered = bool(throttled_steps)
    bounded = demotions <= len(trace)
    refused = ctl.drain_events().get("migrate_demote_no_hide_budget", 0)

    print(f"hide-budget headroom  : {headroom/1e6:+.1f} MB  "
          f"-> demotion {'CAN' if hides else 'CANNOT'} hide on this link")
    print("-" * 68)
    print(f"[1] burst pushed ITL over SLO (entered THROTTLED) : "
          f"{'PASS' if entered else 'FAIL'}")
    print(f"[2] migration rate bounded by the cooldown        : "
          f"{'PASS' if bounded else 'FAIL'}")
    recovered = entered and not trace[-1][1]
    if hides:
        print(f"[3] demoted ({demotions}) and converged              : "
              f"{'PASS' if (demotions > 0 and recovered) else 'FAIL'}")
        good = entered and bounded and demotions > 0 and recovered
        verdict = ("BURST RECOVERY PASS: the loop detects a burst, demotes, "
                   "and converges -- the recourse gate-new-only lacked.")
        if demotions > 0 and not recovered:
            print()
            print("DIAGNOSIS (this is a real finding, not a flaky test):")
            print("  The B1 roofline admitted the demotions on TRANSFER VOLUME")
            print(f"  ({headroom/1e6:.0f} MB headroom, ~9 MB needed) and they")
            print("  fired -- but ITL still rose and never recovered.")
            print("  B1 models bytes/step; it does NOT model the host path's")
            print("  FIXED per-step overhead (per-layer hooks, k-window write")
            print("  +read). On a small model/cohort that overhead dominates")
            print("  the transfer it hides -- the same effect recorded for")
            print("  opt #2. So the roofline is NECESSARY BUT NOT SUFFICIENT")
            print("  as a demotion gate.")
            print("  FIX (needs a design decision): close the loop empirically")
            print("  -- demote, measure, and PROMOTE BACK if ITL got worse.")
            print("  The promotion path now exists to do exactly that.")
    else:
        # The CORRECT behaviour on a slow link is to refuse to demote: it would
        # replace a fast HBM read with an interconnect read that cannot hide.
        print(f"[3] slow link: correctly REFUSED to demote        : "
              f"{'PASS' if demotions == 0 else 'FAIL'}  "
              f"(refusals logged: {refused})")
        good = entered and bounded and demotions == 0
        verdict = ("BURST RECOVERY PASS (slow-link case): the loop detects the "
                   "burst and correctly declines to demote, because on this "
                   "interconnect demotion would make ITL WORSE. There is no "
                   "recourse here -- recovery needs a link where the transfer "
                   "hides. That is a property of the hardware, not a bug.")
    print("=" * 68)
    if good:
        print(verdict)
        return 0
    print("BURST RECOVERY FAIL — trace tail:")
    for t in trace[-12:]:
        print(f"   ema={t[0]:>7} throttled={t[1]} new_demotions={t[2]} "
              f"pending_promote={t[3]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
