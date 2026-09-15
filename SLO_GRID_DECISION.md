# SLO grid for GH200 experiment 4.1 — decided in advance

`GH200_EXPERIMENT_PLAN.md` §2 requires this fixed before renting, so we are not
tuning on a metered box. The plan proposes **{30, 40, 50, 75, 100} ms**.

## The problem with a grid of absolute numbers

An SLO below the arm's own rider-free decode step is unsatisfiable by *every*
arm, all-HBM included. Such a point does not measure a trade-off; it measures
which arm fails first, at full price.

Two of the five proposed points are at risk. Measured on A100, Qwen-32B, ctx
2048, 8 concurrent: the rider-free step is **~49 ms**. The 70B at 8K will be
slower per step, not faster — 80 layers instead of 64, and 4x the context. A
weights-bound floor alone is ~72.7 GB / ~3 TB/s ≈ **24 ms** on GH200 HBM3,
before any attention or KV traffic. So a 30 ms SLO, and plausibly 40 ms, may sit
under the floor for all three arms.

## Decision: anchor the grid to the MEASURED baseline

**Hour 1 of the rental**: run the all-HBM arm alone at the target context and
record its median ITL. Call it `T0`.

**Then set the grid to `T0 x {1.0, 1.2, 1.5, 2.0, 3.0}`**, rounded to 5 ms.

Rationale:
* `1.0 x T0` — the tightest SLO all-HBM can meet. Riders must show they cost
  nothing here, or the SLO-preserving claim is false at its hardest point.
* `1.2` and `1.5` — the regime the paper's claim lives in: a little latency
  traded for capacity.
* `2.0` and `3.0` — loose enough that the interconnect, not the SLO, binds.
  This is where the mixture should win by the largest margin, and where a
  failure would be most damaging.

If `T0` lands near 50 ms the grid is ~{50, 60, 75, 100, 150}, which overlaps
the plan's upper points while dropping the two that would have measured
nothing.

## Correction 2026-09-15: T0 must be measured in the units the SLO enforces

`T0` was specified as the all-HBM arm's **median ITL**. The gate it feeds
scores **goodput**, and `vllm bench serve` defines a request as meeting a TPOT
target using `(latency - ttft) / (output_len - 1)` — a per-request mean, not a
pooled inter-token latency. On the same A100 dry run those differ by 1.7x:

    median pooled ITL   5.85 ms
    median TPOT         3.39 ms

The gap is a definition, not noise: a streamed chunk can carry more than one
token, so `len(itl)` under-counts tokens while `sum(itl)` stays correct, and
dividing by the true token count gives a smaller per-token figure.

Anchoring the grid to pooled ITL and then enforcing it on TPOT sets **every**
point ~1.7x looser than intended. The damage lands hardest on `1.0 x T0`, the
point whose entire purpose is to be the tightest target all-HBM can meet: it
would silently become a target the mixture can also pass, and the rationale
above ("riders must show they cost nothing here") would no longer be tested.

**Decided: `T0` = median per-request TPOT of the all-HBM arm at its peak.**
`gate_goodput.py --t0-metric itl` reproduces the old wording if a comparison is
ever needed; the JSON records which was used. The same correction applies to
the tail condition — `--tail-metric` defaults to p99 TPOT, for the same reason.

**Related asymmetry, worth deciding before 4.0 runs.** The plan's gate reads
"p99 ITL(C) no worse than 1.5 x T0". On the dry run arm A's *own* p99 pooled
ITL was 27.95 ms against an 8.78 ms SLO it met on 64/64 requests. A condition
the baseline fails by 3x is not a gate, it is a disqualification of one arm.
The harness now reports whether the baseline meets the same condition and says
so when it does not; judge the tail arm-relatively.

## Cost of getting this wrong

Five SLO points x three arms is the bulk of 4.1's 24 GPU-h. Two dead points is
~40% of the experiment spent learning that a latency target was unreachable —
which arithmetic could have told us for free.

## Guardrail already in place

`endpoint_comparison.py` now treats the SLO as a **gate**, not part of the
score: an arm that beats the endpoints on goodput while breaching the latency
target it ran under is reported as `SLO BREACHED`, not as a win. So a
too-tight point is at least labelled rather than silently scored.
