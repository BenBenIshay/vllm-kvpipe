# rung-d optimizations + ITL-preserving admission controller (2026-08-08)

Follow-up to the rung-d serve-more feature (RESULTS_RUNGD_2026-08-07.md): the
pool optimizations and the adaptive ITL-preserving admission controller. A100 /
Qwen2.5-0.5B, all env-gated.

## Optimizations

| # | What | Status | Result |
|---|---|---|---|
| 1 | Vectorize `write_token` (batched `index_copy_`, no per-row loop) | done | correct (standalone bitwise + real-KV == vanilla) |
| 2 | Async double-buffered host→rolling stream (side stream + per-layer pinned staging + events) | done, **opt-in** (`VLLM_KVPIPE_RUNGD_ASYNC=1`) | correct; on A100+0.5B+small cohort it did **not** help (wall 18.4s vs 16.4s sync) |

**Honest finding on #2:** async streaming hides the transfer, which only helps in
the **transfer-bound** regime (large cohort / many layers / high-BW C2C). On A100
with a 0.5B model and a small cohort (W≈48 blocks) the per-layer prefetch
bookkeeping *exceeded* the tiny PCIe transfer it hid. So it is opt-in (default
sync path unregressed) and stays available, bitwise-validated, for GH200/large.
The real A100-small per-step bottleneck is Python overhead + the double
attention (standard pass over CPU rows, then override); **opt #5 (skip the
wasted standard pass over CPU rows)** is the likely A100 win and is not yet built.

## ITL-preserving admission controller (B1 + B2)

Replaces the static `off_max = budget // fixed_512` with an adaptive controller
(no manual `off_max`/`itl` tuning), in `transfer_aware_admission.py` +
`scheduler.py`.

- **B1 — seq-length-aware roofline:** admit a new CPU-tier request while the
  running cohort's cumulative per-step transfer stays under the hide budget:
  `b_tok · Σ(running cpu num_computed_tokens) + new ≤ (itl_ms/1000)·x_eff·1e9·(1−margin)`.
  Uses each request's **actual** sequence length and is re-evaluated on every
  admission, so the parallelizable count shrinks as running sequences grow.
- **B2 — closed-loop guardrail:** the scheduler feeds the measured engine step
  time (≈ resident ITL, decode-only steps) each step; an EMA + OK/THROTTLED
  hysteresis around `transfer_aware_slo_p99_itl_ms` stops admitting new CPU-tier
  requests when live ITL nears the SLO (gate-new-only; no demotion, per design).

### Validation

- **Deterministic unit test** (`hybrid_controller_unit.py`, no server): **PASS** —
  B1 refuses a new CPU admission when the cohort exceeds a tight budget and
  admits under a loose one; B2 throttles when step time > SLO, refuses new CPU,
  recovers below the low watermark, and ignores prefill steps.
- **In-server** (`hybrid_controller_check.py`, staggered arrivals, real KV):
  loose budget sustains **13 concurrent CPU** vs tight **10**, both 24/24
  correct — the controller adapts parallelism to the budget with no manual
  tuning. (Gating is on cumulative token-slots, so the workload's natural
  concurrency bounds the visible spread; the unit test is the definitive check.)

### Known limitation (chosen policy)

Gate-new-only means a **burst** admits everyone before the loop engages, and a
long-running cohort can drift over SLO with no recourse but to finish
(demotion is the documented future extension). Staggered arrivals + a binding
budget are needed to observe the adaptation.
