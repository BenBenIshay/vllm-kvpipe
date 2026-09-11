# Who we compare against, what they measure, and what we must change

Written 2026-09-11 at Ben's direction: prove the mechanism is valuable first,
then compare against the *right* competitors — which requires knowing who they
are and what benchmarks they run.

## 1. The nine cited systems, sorted by whether they can be a competitor at all

| system | what it is | can we run it head-to-head? |
|---|---|---|
| **Pie** (arXiv 2411.09317) | layer-partitioned KV offload on GH200, adaptive expansion factor | **YES — the one real competitor.** Same hardware, same goal, different partition axis |
| **DirectKV** (OSDI'26) | zero-copy GPU attention over CPU-resident KV, SM90a | **NOT a competitor — a substrate.** Its own §8.1: "does not change scheduling, batching, or KV-management policies" |
| OmniServe | closest prior *policy* work | **No.** CPU-side attention, PCIe A100 + RoCE, externally supplied service class — trades compute-for-capacity, not bandwidth-for-capacity |
| FlexGen | GPU/CPU/disk placement by offline LP | **No.** Offline-profiled, throughput-oriented batch, not online serving |
| DAK | per-operation offload via TMA | **No.** Takes a global offload ratio as *input* — the quantity a policy must produce |
| NEO, ShadowKV, CALKV | reduce/sparsify/quantize the KV | **No.** Orthogonal: they shrink KV, we relocate it. Composable, not competing |
| "bottlenecks" trace study | evidence that capacity binds | Premise, not a system |

**Conclusion: the competitor set is Pie, and only Pie.** DirectKV belongs in the
paper as a *substrate* comparison (do we compose with it, does it make us
faster) rather than as a rival. Everything else is positioned in §7 prose and
does not need a run.

This is worth stating plainly because it shrinks the evaluation: we owe the
reader one head-to-head, not five.

## 2. What Pie measures — and every way our harnesses differ

From Pie §5.1, read directly:

| axis | Pie | us, today |
|---|---|---|
| models | OPT-13B, OPT-30B, Llama-13B | Qwen2.5-32B |
| hardware | GH200, 96 GB HBM3, 480 GB LPDDR5X | A100 PCIe → GH200 |
| workload | **ShareGPT + Alpaca**, real texts | **synthetic fixed-length token ranges** |
| arrivals | **Poisson at a request rate** (open loop) | **all-at-once** (closed loop) |
| throughput | output tokens ÷ duration | same |
| latency | **queuing latency and compute latency, reported separately** | median/p99 ITL, conflated |
| baseline | vLLM, FlexGen | stock vLLM (all-HBM) |
| headline | **1.9× over vLLM** | — |

**Three mismatches that would make a comparison unsound:**

1. **Synthetic vs real prompts.** Ours are uniform-length integer ranges. Real
   ShareGPT traces are heavy-tailed in both input and output length, which is
   exactly the variation a *per-request* tier policy should exploit — a uniform
   workload hides our own advantage.
2. **Closed vs open loop.** Firing N requests at once and measuring what happens
   is not the same experiment as offering λ requests/second. It also caused a
   concrete error already: `steady_concurrency` rewarded slow arms precisely
   because the load was closed-loop and fixed.
3. **Conflated latency.** Pie separates *queuing* from *compute* latency. Ours
   does not, and the distinction is load-bearing for us: admitting more riders
   should *reduce* queuing while slightly raising compute, and a single ITL
   number hides that trade entirely.

## 3. What we should use instead — and it already exists

`vllm bench serve` (`vllm/benchmarks/serve.py`) ships in our own fork and has:

* `--dataset-name sharegpt` (also BurstGPT, MLPerf, Sonnet, Random)
* `--request-rate` with **Poisson** arrivals, plus `--burstiness`
* `--goodput` — **SLO-aware goodput**, which is our primary metric in 4.1
* standard TTFT / TPOT / ITL percentile reporting, split properly

**We have been building bespoke harnesses while the community-standard one sits
in the tree** — and it is the harness Pie's own baseline (vLLM) is measured
with. Using it makes our numbers directly comparable to Pie's published 1.9×
instead of merely adjacent to it.

Our custom harnesses remain the right tool for *mechanism* work (bitwise gates,
rider-cost curves, fixed-cost hunts) where we need to control rider count
exactly. The split should be deliberate:

* **mechanism claims** → our harnesses (control, precision)
* **serving claims** → `vllm bench serve` (comparability)

## 4. Preparation required before the rental

- [ ] **Cache ShareGPT** (and optionally Alpaca). Not present in
      `/mnt/beegfs/benishay/huggingface`; downloading on a metered box is waste.
- [ ] **Dry-run `vllm bench serve` against our engine on A100** with
      `--dataset-name sharegpt --request-rate <λ> --goodput ttft:...,tpot:...`,
      confirming the hybrid path runs under it and the tier histogram shows
      riders.
- [ ] **Decide λ grid** by the same rule as the SLO grid: anchor to the measured
      saturation point of the all-HBM arm, not round numbers.
- [ ] **Restore or rewrite `open_loop_rate_sweep.py`** — it is a broken symlink
      to `/data/ben/kv_phase_a_internal/`, i.e. we *had* an open-loop harness
      and lost it.
- [ ] **Decide the Pie comparison mode.** Pie is open-source; if it builds on
      GH200 we run it directly. If not, the honest fallback is to run *our*
      system on *their* workload (ShareGPT/Alpaca, Poisson, same models where
      possible) and compare against their published 1.9× over a shared vLLM
      baseline — stating clearly that the vLLM baselines differ in version.

## 5. Consequence for the experiment plan

`all_host` is dropped as an arm (Ben, 2026-09-11). It is not a system anyone
deploys, its capacity is arithmetic rather than a discovery, and DirectKV —
which *is* the all-host endpoint — is a substrate we compose with, not a rival.

The sequence becomes: **prove the mechanism (4.0) → characterise it (4.2, 4.2b)
→ show the policy is load-bearing (4.3) → compare against Pie on Pie's own
workload.**
