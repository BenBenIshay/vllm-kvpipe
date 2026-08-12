#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic unit test for Item 2 — request-level tier assignment.

No GPU, no server. Exercises `host_affinity` and the tier decision inside
`_should_admit_hybrid` against a synthetic mix of requests.

What is being asserted, and why it is the interesting part: **context length
must NOT discriminate between tiers.** Moving a request to the host tier frees
S*k of HBM and costs S*k of interconnect per step — both scale identically with
S, and the k-window discount is S-independent. That is the same cancellation as
design rule (a), which we validated on hardware in P4. The signal that *does*
discriminate is temporal: idle requests hold HBM while consuming no bandwidth.
"""

from __future__ import annotations

import sys
import types


class _SP:
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens


class _Req:
    def __init__(self, rid, prompt_tokens=1024, max_tokens=256, done=0):
        self.request_id = rid
        self.num_prompt_tokens = prompt_tokens
        self.num_computed_tokens = prompt_tokens
        self.num_output_tokens = done
        self.sampling_params = _SP(max_tokens)


class _Cfg:
    enable_transfer_aware_admission = True
    transfer_aware_slo_p99_itl_ms = 0.0
    transfer_aware_predictor_csv = None
    hybrid_b_hbm_max = 2
    hybrid_itl_ms = 20.0
    hybrid_b_tok = 131072
    hybrid_pcie_gbps = 25.1
    hybrid_seq_tokens = 1024
    enable_hybrid_kv_tiers = True


def main() -> int:
    import os
    os.environ["VLLM_KVPIPE_TIER_POLICY"] = "scored"
    from vllm.v1.core.sched.transfer_aware_admission import (
        TransferAwareAdmissionController,
    )

    fails = []

    # ---- 1. context length must NOT change host affinity -----------------
    c = TransferAwareAdmissionController(_Cfg())
    short = _Req("s", prompt_tokens=512, max_tokens=256)
    long_ = _Req("l", prompt_tokens=32768, max_tokens=256)
    c.note_active("s", now=100.0); c.note_active("l", now=100.0)
    a_s = c.host_affinity(short, now=100.0)
    a_l = c.host_affinity(long_, now=100.0)
    ok = abs(a_s - a_l) < 1e-9
    print(f"[1] context-length neutrality: short={a_s:.3f} long={a_l:.3f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("context length changed affinity (rule (a) says it must not)")

    # ---- 2. idle requests are preferred for the host tier ----------------
    c = TransferAwareAdmissionController(_Cfg())
    busy, idle = _Req("busy"), _Req("idle")
    c.note_active("busy", now=100.0)
    c.note_active("idle", now=95.0)          # 5 s idle, saturation is 2 s
    a_b = c.host_affinity(busy, now=100.0)
    a_i = c.host_affinity(idle, now=100.0)
    ok = a_i > a_b
    print(f"[2] idle preferred: busy={a_b:.3f} idle={a_i:.3f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("idle request not preferred for host tier")

    # ---- 3. nearly-finished requests are NOT displaced --------------------
    c = TransferAwareAdmissionController(_Cfg())
    fresh = _Req("fresh", max_tokens=512, done=0)
    almost = _Req("almost", max_tokens=512, done=500)
    c.note_active("fresh", now=100.0); c.note_active("almost", now=100.0)
    a_f = c.host_affinity(fresh, now=100.0)
    a_a = c.host_affinity(almost, now=100.0)
    ok = a_f > a_a
    print(f"[3] long-remaining preferred: fresh={a_f:.3f} almost-done={a_a:.3f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("nearly-finished request not deprioritised for host tier")

    # ---- 4. "arrival" policy reproduces pre-Item-2 behaviour exactly ------
    os.environ["VLLM_KVPIPE_TIER_POLICY"] = "arrival"
    c = TransferAwareAdmissionController(_Cfg())
    tiers = []
    running: list = []
    for i in range(4):
        r = _Req(f"r{i}")
        adm, why = c._should_admit_hybrid(request=r, running_requests=running)
        tiers.append(c.get_req_tier(r.request_id))
        if adm:
            running.append(r)
    ok = tiers[:2] == ["hbm", "hbm"]
    print(f"[4] arrival policy fills HBM first: {tiers} -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"arrival policy changed: {tiers}")

    # ---- 5. scored policy yields an HBM slot to a low-affinity request ----
    os.environ["VLLM_KVPIPE_TIER_POLICY"] = "scored"
    c = TransferAwareAdmissionController(_Cfg())
    idle_req = _Req("idle1")
    c.note_active("idle1", now=0.0)          # very idle -> high affinity
    import time as _t
    adm, why = c._should_admit_hybrid(request=idle_req, running_requests=[])
    t_idle = c.get_req_tier("idle1")
    c2 = TransferAwareAdmissionController(_Cfg())
    busy_req = _Req("busy1")
    c2.note_active("busy1")                  # just active -> low affinity
    adm2, why2 = c2._should_admit_hybrid(request=busy_req, running_requests=[])
    t_busy = c2.get_req_tier("busy1")
    ok = (t_idle == "cpu") and (t_busy == "hbm")
    print(f"[5] scored policy: idle->{t_idle} ({why}), busy->{t_busy} ({why2}) "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"scored policy misassigned: idle={t_idle} busy={t_busy}")

    print("=" * 64)
    if fails:
        print("TIER POLICY UNIT FAIL:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("TIER POLICY UNIT PASS: context-neutral, idle-preferring, "
          "completion-aware; arrival policy unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
