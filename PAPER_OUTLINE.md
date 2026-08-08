# Paper outline — hybrid tiered-KV "free-rider decoding" (target: MLSys)

Status: skeleton to start writing. Maps our BUILT+MEASURED results to figures and
marks what is writable now vs blocked on the GH200 experiment. Companion to the
strategy in memory (`project-hybrid-kv-tiers`, `state-hybrid-kv-execution`) and
`~/.claude/plans/effervescent-orbiting-scroll.md`.

## Title (pick/refine)
1. "Free-Rider Decoding: Serving Off-HBM-KV Requests on the Amortized Weight Read"
2. "Riding the Weight Read: Interconnect-Aware Admission for Mixed HBM/Host-KV Decode"

## One-line thesis
LLM decode is weight-bandwidth-bound, so a decode batch has spare attention/compute
headroom per weight read. In ONE weight-shared forward we serve HBM-resident requests
+ requests whose KV lives on host and streams to the GPU (attention stays on-GPU);
the extra requests "ride free" on the amortized weight read, and an interconnect-
parameterized, SLO-targeted admission controller decides how many fit without
degrading the resident cohort's ITL — worthwhile on NVLink-C2C, not on PCIe.

## Contributions (the defensible conjunction — none novel alone)
1. **Free-rider framing + mixed forward:** weights read 1×, attention split (HBM rows
   attend resident, host-tier rows attend a k_window-shared rolling region streamed
   from host). Contrast NEO/FastDecode which move attention *compute* to the CPU and
   whose gains *decay* as the GPU strengthens (T4 7.5×→H100 14%) — that trend is our
   motivation.
2. **Host-block-id KV pool + scheduler reclaim:** host-tier requests consume ~0 main-
   pool blocks; the freed HBM serves more requests. Measured **2.67× concurrent at
   the same HBM, token-identical to vanilla**.
3. **Interconnect-parameterized, SLO-targeted admission controller:** a seq-length-
   aware roofline + a closed-loop measured-ITL guardrail that decides how many host-
   tier "free riders" to admit without breaking resident ITL. C2C-worthwhile, PCIe-not.
4. Implementation in vLLM v1 + evaluation: A100 capacity (measured) and GH200 C2C
   goodput/latency crossover (the headline).

## Section plan
1. Introduction
2. Background & Motivation — weight-BW-bound decode; the NEO decay trend; PCIe→C2C
   roofline crossover (analytical).
3. Design — mixed weight-shared forward; host-block-id pool (seed/grow/stream/write-
   redirect/compact-read); scheduler cap; ITL admission controller (B1 roofline + B2
   guardrail).
4. Implementation — vLLM v1, env-gated; correctness-first pool + optimizations.
5. Evaluation (matrix below).
6. Related Work — positioning table.
7. Discussion & Limitations — gate-new-only (no demotion); PCIe barely helps (agrees
   with "Bottlenecks"); why C2C is load-bearing.

## Figure / table plan  (✅ = data in hand, 🔲 = needs GH200)
- ✅ **Correctness ladder** (Table): rung-c compacted-attention bitwise; real-KV token-
   identical to vanilla; rung-d served (harnesses committed).
- ✅ **Capacity / serve-more** (Fig): 2.67× concurrent at a fixed 160-block pool; CPU-
   tier main-pool blocks 126→48. (`RESULTS_RUNGD_2026-08-07.md`)
- ✅ **Occupancy opportunity** (Fig): CPU tier holds 38.5% of in-use main-pool blocks.
- ✅ **ITL controller** (Fig): B1 budget-driven adaptation (unit + staggered in-server);
   B2 guardrail throttle/recover. (`RESULTS_OPT_CONTROLLER_2026-08-08.md`)
- ✅ **Optimization** (Table): opt #5 rung-d wall 16.4s→12.8s (~22%).
- ✅ **Hide boundary** (Fig): R0 vs R1 ITL within +0.5ms through conc 32 (July A100).
- 🔲 **HEADLINE C2C crossover** (Fig): goodput/ITL vs interconnect X_eff, PCIe→C2C —
   the load-bearing result; on GH200.
- 🔲 **NEO same-hardware comparison** (Fig): streamed-KV-to-GPU vs CPU-attention at
   high BW (the "isn't this NEO?" rebuttal); on GH200.
- 🔲 **Sweeps**: concurrency across the B_hbm/B_off boundary; model size; seq length.

## Writable NOW (A100 + design + related work), before GH200
Intro, Background/Motivation, Design, Implementation, the six ✅ evaluation subsections,
Related Work, Limitations. This is ~70% of the paper draft.

## Blocked on GH200 (the load-bearing 30%)
Headline C2C crossover + the same-hardware NEO comparison. Per the strategy, without a
measured C2C result the paper is pre-empted by NEO (MLSys'25) + "Bottlenecks", and our
only measurable regime (A100/PCIe ~1.1×) agrees with the skeptics.

## What I need from you (to unblock)
1. **GH200 decision + rough timeline** — rent now (headline, MLSys-tier) vs write the
   A100-floor draft first and rent later. This is the single biggest gate.
2. **Author list + affiliations** for the header.
3. **Confirm the contribution framing** above (the 3-way conjunction) — or edit it.
4. **Where we write** — set up an Overleaf project with the `mlsys2026` style and share
   edit access, or should I scaffold LaTeX in-repo (`paper/`) with a placeholder class?
5. **NEO-style baseline scope** — implement a CPU-attention offload baseline (needed to
   answer "isn't this NEO?" apples-to-apples on GH200), or approximate/cite? Real eng.
6. **MLSys 2026 CFP dates** (abstract/paper deadlines) — to pace the plan; please confirm.
7. **Cite-check** the two unverified 2026 preprints before we rely on them (arXiv
   2601.19910 "Bottlenecks", 2506.03296 APEX).
