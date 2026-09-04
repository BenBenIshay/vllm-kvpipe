#!/usr/bin/env python3
"""B1's hide budget must be measured, and must not measure itself.

The budget is "how long one forward iteration lasts" -- how much interconnect
time a rider's transfer can hide behind. It used to be `--hybrid-itl-ms`, a
hand-set constant every harness had to guess, and guessing it high (100000,
"capacity question, not an SLO question") silently disabled the gate. It is a
property of the machine and the model, so it is now observed.

The trap this guards is circularity. If the budget were taken from the live
step time, a step already slowed by rider transfer would enlarge the budget
that admitted those riders, and the loop would inflate itself until the ITL
collapsed. Only steps carrying NO rider are sampled.

No GPU, no server.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/mnt/beegfs/benishay/vllm-kvpipe")

from vllm.v1.core.sched.transfer_aware_admission import (
    TransferAwareAdmissionController,
)

FAILURES: list[str] = []
STEP_MS = 45.0          # a plausible Qwen-32B decode step on A100
CTX = 2048
B_TOK = 262144          # Qwen-32B KV bytes/token


class Cfg:
    hybrid_b_tok = B_TOK
    hybrid_itl_ms = 100000.0        # the "gate disabled" value, on purpose
    hybrid_pcie_gbps = 25.1
    hybrid_b_hbm_max = 2
    enable_hybrid_kv_tiers = True
    enable_transfer_aware_admission = True
    transfer_aware_slo_p99_itl_ms = 0.0
    transfer_aware_predictor_csv = None


class Req:
    def __init__(self, rid: str, n: int) -> None:
        self.request_id = rid
        self.num_prompt_tokens = n
        self.num_computed_tokens = n
        self.sampling_params = None


def check(cond: bool, what: str) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        FAILURES.append(what)


def admit_until_refused(c, ctx: int, limit: int = 64):
    run: list[Req] = []
    for i in range(limit):
        r = Req(f"r{i}", ctx)
        ok, _why = c.should_admit_from_waiting(
            request=r, running_requests=run, token_budget=1 << 30)
        if not ok:
            break
        run.append(r)
    tiers = {t: sum(1 for x in run
                    if c.get_req_tier(x.request_id) == t)
             for t in ("hbm", "cpu")}
    return len(run), tiers


def main() -> int:
    print("1. before any clean sample, the configured value still governs")
    c = TransferAwareAdmissionController(Cfg())
    check(c.hide_budget_ms() == 100000.0 and c.hide_budget_source() ==
          "configured", "falls back to --hybrid-itl-ms until measured")

    print("2. clean resident-only steps become the budget")
    for _ in range(10):
        c.observe_step_time(STEP_MS, True, 0)
    check(abs(c.hide_budget_ms() - STEP_MS) < 1e-6
          and c.hide_budget_source() == "measured",
          f"budget = {c.hide_budget_ms():.2f} ms from clean steps")

    print("3. rider-carrying steps must NOT enlarge the budget (circularity)")
    for _ in range(50):
        c.observe_step_time(20 * STEP_MS, True, 4)
    check(abs(c.hide_budget_ms() - STEP_MS) < 1e-6,
          f"budget still {c.hide_budget_ms():.2f} ms after 50 steps at "
          f"{20 * STEP_MS:.0f} ms carrying 4 riders")

    print("4. prefill steps are still ignored")
    before = c.hide_budget_ms()
    for _ in range(20):
        c.observe_step_time(3000.0, False, 0)
    check(abs(c.hide_budget_ms() - before) < 1e-6,
          "a 3 s prefill step does not become the hide budget")

    print("5. the admitted rider count matches the roofline by hand")
    n, tiers = admit_until_refused(c, CTX)
    budget_bytes = (STEP_MS / 1000.0) * 25.1e9
    by_hand = int(budget_bytes * 0.9 // (CTX * B_TOK))   # 10% default margin
    print(f"      admitted {n} total {tiers}; hand roofline says "
          f"{by_hand} riders at ctx {CTX}")
    check(tiers["cpu"] == by_hand,
          f"riders admitted ({tiers['cpu']}) == roofline ({by_hand})")
    check(tiers["hbm"] == Cfg.hybrid_b_hbm_max,
          "the resident cohort is unaffected by the gate")

    print("6. a longer context admits proportionally fewer riders")
    c2 = TransferAwareAdmissionController(Cfg())
    for _ in range(10):
        c2.observe_step_time(STEP_MS, True, 0)
    _n2, t2 = admit_until_refused(c2, CTX * 4)
    check(t2["cpu"] * 4 <= tiers["cpu"] + 1,
          f"ctx {CTX * 4} admits {t2['cpu']} riders vs {tiers['cpu']} at "
          f"ctx {CTX} -- the budget is bytes, not requests")

    print("7. the measurement can be turned off")
    import os
    os.environ["VLLM_KVPIPE_B1_MEASURED"] = "0"
    c3 = TransferAwareAdmissionController(Cfg())
    for _ in range(10):
        c3.observe_step_time(STEP_MS, True, 0)
    check(c3.hide_budget_ms() == 100000.0,
          "VLLM_KVPIPE_B1_MEASURED=0 restores the configured value")
    os.environ.pop("VLLM_KVPIPE_B1_MEASURED")

    print("=" * 70)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("B1 MEASURED-BUDGET PASS: the hide budget comes from real forward "
          "iterations, ignores the steps its own riders slowed down, and "
          "admits exactly the rider count the roofline predicts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
