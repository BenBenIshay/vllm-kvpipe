#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic unit test for Item 3 — tier migration decision logic.

No GPU, no server. Drives the controller's ITL signal directly and checks the
three anti-thrash guards: hysteresis, the amortisation cost model, and the
post-migration cooldown.

The headline check is [5]: under an ADVERSARIAL oscillating ITL signal the
migration rate must stay bounded. The IEEE CAL letter reports that token-level
migration thrashes below ~60% attention sparsity; our claim is that
request-level migration does not, because whole requests move once at
controlled moments rather than pages chasing shifting token importance.
"""

from __future__ import annotations

import os
import sys


class _SP:
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens


class _Req:
    def __init__(self, rid, max_tokens=1024, done=0, ctx=2048):
        self.request_id = rid
        self.num_prompt_tokens = ctx
        self.num_computed_tokens = ctx
        self.num_output_tokens = done
        self.sampling_params = _SP(max_tokens)


class _Cfg:
    enable_transfer_aware_admission = True
    transfer_aware_slo_p99_itl_ms = 20.0
    transfer_aware_predictor_csv = None
    hybrid_b_hbm_max = 4
    hybrid_itl_ms = 20.0
    hybrid_b_tok = 131072
    hybrid_pcie_gbps = 25.1
    hybrid_seq_tokens = 2048
    enable_hybrid_kv_tiers = True


def drive(c, ms, n=30):
    """Advance the controller: n scheduling steps at the given step time.

    Advancing the STEP COUNTER matters, not just the ITL: the controller
    refuses to migrate on a cold EMA (VLLM_KVPIPE_MIGRATE_WARMUP, default 24),
    because acting before a steady state exists gives the probe a garbage
    baseline -- which is exactly what we measured in-server."""
    for _ in range(n):
        c.begin_schedule_step()
        c.observe_step_time(ms, True)


def main() -> int:
    os.environ["VLLM_KVPIPE_TIER_POLICY"] = "scored"
    os.environ["VLLM_KVPIPE_MIGRATE_COOLDOWN"] = "64"
    from vllm.v1.core.sched.transfer_aware_admission import (
        TransferAwareAdmissionController as C,
    )
    fails = []

    # ---- 1. over SLO -> demote from HBM ---------------------------------
    c = C(_Cfg())
    reqs = [_Req(f"r{i}") for i in range(4)]
    for r in reqs:
        c._req_tier[r.request_id] = "hbm"
        c.note_active(r.request_id, now=0.0)
    drive(c, 30.0)                      # well over the 20 ms SLO
    acts = c.plan_migrations(reqs, now=100.0)
    ok = len(acts) >= 1 and all(a[1] == "demote" for a in acts)
    print(f"[1] over SLO -> demote: {acts} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("no demotion under SLO breach")

    # ---- 2. comfortably under SLO -> promote back ------------------------
    c = C(_Cfg())
    reqs = [_Req(f"c{i}") for i in range(3)]
    for r in reqs:
        c._req_tier[r.request_id] = "cpu"
    drive(c, 5.0)                       # far below 0.70*SLO = 14 ms
    acts = c.plan_migrations(reqs, now=100.0)
    ok = len(acts) >= 1 and all(a[1] == "promote" for a in acts)
    print(f"[2] under SLO -> promote: {acts} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("no promotion when ITL has headroom")

    # ---- 3. hysteresis dead-band: no action between watermarks -----------
    c = C(_Cfg())
    reqs = [_Req(f"h{i}") for i in range(3)]
    for r in reqs:
        c._req_tier[r.request_id] = "hbm"
    drive(c, 17.0)                      # between 0.70*SLO and 0.95*SLO
    acts = c.plan_migrations(reqs, now=100.0)
    ok = acts == []
    print(f"[3] hysteresis dead-band quiet: {acts} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"migrated inside the dead band: {acts}")

    # ---- 4. cost model refuses a nearly-finished request -----------------
    c = C(_Cfg())
    almost = _Req("almost", max_tokens=1024, done=1020)   # 4 tokens left
    c._req_tier["almost"] = "hbm"
    drive(c, 30.0)
    acts = c.plan_migrations([almost], now=100.0)
    ok = acts == []
    print(f"[4] cost model blocks almost-done (4 tok left): {acts} -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("migrated a request that cannot amortise the move")

    # ---- 5. ADVERSARIAL oscillation -> bounded migration rate -------------
    c = C(_Cfg())
    reqs = [_Req(f"o{i}", max_tokens=100000) for i in range(6)]
    for i, r in enumerate(reqs):
        c._req_tier[r.request_id] = "hbm" if i < 3 else "cpu"
    total = 0
    for step in range(400):
        c.begin_schedule_step()
        c.observe_step_time(40.0 if (step // 5) % 2 == 0 else 3.0, True)
        for rid, act in c.plan_migrations(reqs, now=float(step)):
            c.note_migrated(rid, "cpu" if act == "demote" else "hbm")
            total += 1
    rate = total / 400.0
    # cooldown 64 steps, 6 requests -> at most ~6/64 per step even if the
    # signal flips every step
    ok = rate <= 6.0 / 64.0 + 1e-9
    print(f"[5] adversarial oscillation over 400 steps: {total} migrations "
          f"({rate:.4f}/step, bound {6/64:.4f}) -> {'PASS' if ok else 'FAIL'}")
    print(f"    stats={c.migration_stats()}")
    if not ok:
        fails.append(f"thrash: {rate:.3f} migrations/step exceeds the cooldown bound")

    # ---- 5b. ADAPTIVE PROBE: revert when the demotion made ITL worse ------
    os.environ["VLLM_KVPIPE_PROBE_STEPS"] = "4"
    c = C(_Cfg())
    reqs = [_Req(f"p{i}", max_tokens=100000) for i in range(3)]
    for r in reqs:
        c._req_tier[r.request_id] = "hbm"
    drive(c, 30.0)                       # over SLO -> should demote
    c.begin_schedule_step()
    acts = c.plan_migrations(reqs, now=1.0)
    demoted = [a for a in acts if a[1] == "demote"]
    for rid, _ in demoted:
        c.note_migrated(rid, "cpu")
    # ITL gets WORSE after the demotion (what we measured on A100)
    for _ in range(6):
        c.begin_schedule_step()
        c.observe_step_time(60.0, True)
    acts2 = c.plan_migrations(reqs, now=2.0)
    reverted = [a for a in acts2 if a[1] == "promote"]
    ok = bool(demoted) and len(reverted) == len(demoted)
    print(f"[5b] probe reverts a harmful demotion: demoted={len(demoted)} "
          f"reverted={len(reverted)} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"probe did not revert: demoted={demoted} acts2={acts2}")
    # and further demotions are suppressed
    for rid, _ in reverted:
        c.note_migrated(rid, "hbm")
    for _ in range(6):
        c.begin_schedule_step()
        c.observe_step_time(60.0, True)
    acts3 = c.plan_migrations(reqs, now=3.0)
    ok2 = not any(a[1] == "demote" for a in acts3)
    print(f"[5c] after harm, demotion is suppressed: {acts3} -> "
          f"{'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        fails.append("kept demoting after learning it was harmful")

    # ---- 5d. probe KEEPS a demotion that helped --------------------------
    c = C(_Cfg())
    reqs = [_Req(f"g{i}", max_tokens=100000) for i in range(3)]
    for r in reqs:
        c._req_tier[r.request_id] = "hbm"
    drive(c, 30.0)
    c.begin_schedule_step()
    acts = c.plan_migrations(reqs, now=1.0)
    for rid, _ in [a for a in acts if a[1] == "demote"]:
        c.note_migrated(rid, "cpu")
    for _ in range(6):                    # ITL IMPROVES after the demotion
        c.begin_schedule_step()
        c.observe_step_time(8.0, True)
    acts4 = c.plan_migrations(reqs, now=2.0)
    ok3 = not any(a[1] == "promote" for a in acts4)
    print(f"[5d] probe keeps a helpful demotion (no revert): {acts4} -> "
          f"{'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        fails.append(f"reverted a demotion that helped: {acts4}")

    # ---- 6. disabled by default (arrival policy) -------------------------
    os.environ["VLLM_KVPIPE_TIER_POLICY"] = "arrival"
    c = C(_Cfg())
    r = _Req("x")
    c._req_tier["x"] = "hbm"
    drive(c, 40.0)
    acts = c.plan_migrations([r], now=100.0)
    ok = acts == []
    print(f"[6] arrival policy -> migration off: {acts} -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("migration active under the arrival baseline")

    print("=" * 64)
    if fails:
        print("MIGRATION UNIT FAIL:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("MIGRATION UNIT PASS: demotes over SLO, promotes with headroom, "
          "quiet in the dead band, refuses un-amortisable moves, and stays "
          "bounded under adversarial oscillation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
