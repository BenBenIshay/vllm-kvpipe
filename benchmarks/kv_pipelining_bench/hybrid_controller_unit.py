#!/usr/bin/env python3
"""Deterministic unit test of the ITL-preserving admission controller.

Tests the decision logic directly (no server, no noise):
  B1 seq-length-aware roofline: with a running CPU cohort whose cumulative
     transfer exceeds a TIGHT hide budget, a new CPU-tier admission is REFUSED;
     with a LOOSE budget it is admitted.
  B2 closed-loop guardrail: feeding measured step times above the SLO flips the
     controller to THROTTLED (refuses new CPU admissions); step times back under
     the low watermark recover it.
"""
from __future__ import annotations

from types import SimpleNamespace

from vllm.v1.core.sched.transfer_aware_admission import (
    TransferAwareAdmissionController,
)

B_TOK = 24576  # Qwen2.5-0.5B bytes/token (24 layers * 2 * 2 kv heads * 64 * 2B)


def cfg(itl_ms: float, slo_ms: float) -> SimpleNamespace:
    return SimpleNamespace(
        enable_transfer_aware_admission=True,
        enable_hybrid_kv_tiers=True,
        hybrid_b_hbm_max=0,
        hybrid_itl_ms=itl_ms,
        hybrid_b_tok=B_TOK,
        hybrid_pcie_gbps=25.1,
        hybrid_seq_tokens=256,
        transfer_aware_slo_p99_itl_ms=slo_ms,
    )


def req(rid: str, computed: int, prompt: int) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=rid, num_computed_tokens=computed, num_prompt_tokens=prompt
    )


def running_cpu(ctrl, n: int, seq: int):
    reqs = [req(f"r{i}", seq, seq) for i in range(n)]
    for r in reqs:
        ctrl._req_tier[r.request_id] = "cpu"
    return reqs


def main() -> None:
    ok_all = True

    # --- B1: budget gates on cumulative seq-length transfer -----------------
    # itl_ms=0.2 -> budget ~ (0.2/1000)*25.1e9*0.9 = 4.52 MB = ~184 token-slots.
    tight = TransferAwareAdmissionController(cfg(itl_ms=0.2, slo_ms=0.0))
    run_t = running_cpu(tight, n=10, seq=50)  # 500 token-slots >> 184 budget
    ok_t, why_t = tight.should_admit_from_waiting(
        request=req("new", 0, 10), running_requests=run_t, token_budget=10_000)
    b1_tight = (not ok_t) and why_t == "hybrid_offload_cap"

    loose = TransferAwareAdmissionController(cfg(itl_ms=40.0, slo_ms=0.0))
    run_l = running_cpu(loose, n=10, seq=50)   # 500 token-slots << huge budget
    ok_l, why_l = loose.should_admit_from_waiting(
        request=req("new", 0, 10), running_requests=run_l, token_budget=10_000)
    b1_loose = ok_l and why_l == "hybrid_cpu"

    print(f"[B1] tight budget refuses new CPU: {b1_tight} ({why_t})")
    print(f"[B1] loose budget admits new CPU:  {b1_loose} ({why_l})")

    # --- B2: measured-ITL guardrail throttles + recovers --------------------
    g = TransferAwareAdmissionController(cfg(itl_ms=40.0, slo_ms=20.0))  # loose budget
    for _ in range(10):
        g.observe_step_time(50.0, is_decode_only=True)   # 50ms > 20ms SLO
    b2_throttled = g._itl_throttled_state
    ok_g, why_g = g.should_admit_from_waiting(
        request=req("new", 0, 10), running_requests=[], token_budget=10_000)
    b2_refuse = (not ok_g) and why_g == "hybrid_itl_throttled"
    for _ in range(30):
        g.observe_step_time(5.0, is_decode_only=True)    # 5ms < 20*0.85 recover
    b2_recover = not g._itl_throttled_state
    # prefill steps must be ignored (not counted toward the EMA/state)
    g2 = TransferAwareAdmissionController(cfg(itl_ms=40.0, slo_ms=20.0))
    for _ in range(10):
        g2.observe_step_time(500.0, is_decode_only=False)  # prefill -> ignored
    b2_ignore_prefill = not g2._itl_throttled_state

    print(f"[B2] step>SLO -> throttled: {b2_throttled}")
    print(f"[B2] throttled -> refuses new CPU: {b2_refuse} ({why_g})")
    print(f"[B2] step<low-watermark -> recovers: {b2_recover}")
    print(f"[B2] prefill steps ignored: {b2_ignore_prefill}")

    ok_all = all([b1_tight, b1_loose, b2_throttled, b2_refuse, b2_recover,
                  b2_ignore_prefill])
    print("=" * 60)
    if ok_all:
        print("CONTROLLER UNIT PASS: B1 seq-aware roofline gates on the hide "
              "budget and B2 measured-ITL guardrail throttles/recovers on the "
              "SLO — the ITL-preserving admission controller is correct.")
    else:
        print("CONTROLLER UNIT FAIL — inspect the flags above.")


if __name__ == "__main__":
    main()
