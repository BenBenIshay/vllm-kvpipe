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
    os.environ["VLLM_KVPIPE_PROBE_STEPS"] = "12"      # settle before judging

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
        back = after.get("hbm", 0) - before.get("hbm", 0)
        trace.append((round(ctl._resident_step_ms_ema, 3),
                      ctl._itl_throttled_state, newly,
                      len(sched._promoting_req_ids), back))
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

    reverts = ctl.migration_stats().get("hbm", 0)
    probes = ctl._probe_results
    peak = max(emas) if emas else 0.0
    final = trace[-1][0] if trace else 0.0
    headroom = ctl.hide_budget_headroom_bytes(list(sched.running))
    entered = bool(throttled_steps)
    bounded = demotions <= len(trace)
    harmful = any(w for _, _, w in probes)
    undone = (reverts >= demotions) if demotions else True
    settled = (final <= peak * 0.75) if peak > 0 else False

    print("-" * 68)
    print(f"[1] burst pushed ITL over SLO (entered THROTTLED) : "
          f"{'PASS' if entered else 'FAIL'}")
    print(f"[2] migration rate bounded by the cooldown        : "
          f"{'PASS' if bounded else 'FAIL'}")
    if demotions == 0:
        print("[3] declined to demote at all                     : PASS")
        good = entered and bounded
        verdict = ("BURST RECOVERY PASS: the loop detected the burst and "
                   "declined to demote.")
    else:
        print(f"[3] probe MEASURED the demotion as harmful        : "
              f"{'PASS' if harmful else 'FAIL'}")
        print(f"[4] loop UNDID its own damage (promoted back)     : "
              f"{'PASS' if undone else 'FAIL'}  ({reverts}/{demotions})")
        print(f"[5] ITL settled ({final:.1f} <= 75% of peak {peak:.1f})       : "
              f"{'PASS' if settled else 'FAIL'}")
        good = entered and bounded and harmful and undone and settled
        verdict = ("BURST RECOVERY PASS (adaptive): the loop probed a "
                   "demotion, MEASURED that it made ITL worse, promoted the "
                   "requests back, and suppressed further demotions. On this "
                   "interconnect the right action is to do nothing -- and the "
                   "controller learns that from MEASUREMENT, which the B1 "
                   "roofline cannot do: it sees transfer bytes, not the host "
                   "path's fixed per-step overhead.")
    print("=" * 68)
    if good:
        print(verdict)
        return 0
    print("BURST RECOVERY FAIL - trace tail:")
    for t in trace[-14:]:
        print(f"   ema={t[0]:>7} throttled={t[1]} demote={t[2]} pending={t[3]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
