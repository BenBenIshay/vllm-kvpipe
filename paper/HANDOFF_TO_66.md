# Handoff: sessions consolidated, continue in `vllm-kvpipe-66`

Written 2026-09-12 by `vllm-kvpipe-03` (the paper session), which is standing
down. Ben consolidated to one session. A cross-session message with this content
was sent twice and held for approval both times, so this file is the reliable
copy — git is the channel that has actually worked between us.

## The merge

**`b8ae9ffe7` on `hybrid-kv-tiers`.** From `kvpipe-66`:

```bash
git merge hybrid-kv-tiers      # should be clean
```

The split had diverged 15 commits (yours) against 6 (mine), both editing
`GH200_EXPERIMENT_PLAN.md`, `policy_ablation.py` and `hybrid_rungd_pool.py`.
Conflicts resolved **in your favour wherever you had made a decision**.

### `policy_ablation.py` — yours, plus one addition

Your version won outright. Your **vacuity guard** and **inert-baseline guard**
are better than what I had, and your finding that `uniform_25` "beat" us by
14.5% while `uniform_50` at our own cap was 2.9% — *almost all of that headline
was the allocation, not the policy* — is exactly the kind of thing that ships as
a result if nobody checks.

I added one thing on top: a **byte-matching flag**. Matching `b_hbm_max`
controls the *allocation*; it does not control the *traffic*. An arm can sit at
the same cap and still move 2× the volume, and the interconnect only sees bytes.
It prints `[UNMATCHED BYTES]` and says the delta is not reportable as a policy
result.

### `GH200_EXPERIMENT_PLAN.md` — yours from §4.4 onward

Your decisions with Ben stand: 70B dropped, fp8 pair dropped, negative control
moved to the A100, Pie head-to-head added as the only real competitor. Budget
reconciled **76–102 → 74–82 GPU-h**. Your duplicate §4.6 heading (DirectKV) is
renumbered §4.7.

## Two things I added that you have not seen

### §4.3b — the return leg (4 GPU-h)

`RungdHostPool.writeback_layer`, gated `VLLM_KVPIPE_RUNGD_WRITEBACK=1`, **off by
default**. Writes each layer's whole rolling slot back to the host mirror — the
exact inverse of the refresh, same runs, same granularity — to measure what
*swap* semantics costs versus *streaming*, policy held fixed.

Validated at both levels:

- `writeback_unit.py`: disabled is a true no-op (does not propagate even a
  deliberately corrupted slot), enabled moves exactly the extent the refresh
  read, round trip is lossless.
- Full engine, Qwen2.5-0.5B: both arms read **0.602 GB**, swapback totals
  **1.205 GB = exactly 2.00×**, token output identical.

That confirms **on our own hardware** the factor-of-two the paper claims about
Pie, which until now rested only on reading their swap controller.

**Do not quote the −7.2% tok/s from that run.** 0.5B over 0.6 s with 25 sampled
steps is apparatus validation, not a measurement. The *traffic* ratio is
structural and transfers; the *throughput* cost is what §4.3b measures.

### §4.2c — a workload blocker that gates your §4.3

`host_affinity` scores requests on **temporal** signals — idle time and
estimated remaining generation — because context length provably cannot
discriminate. Every harness we have generates **identical** requests: same
prompt length, `max_tokens == min_tokens` with `ignore_eos`, all arriving in one
burst. So `idle_score` is 0 for everyone and `rem_score` is identical for
everyone.

⇒ `host_affinity` returns the same value for every request, and the `scored` arm
is **byte-for-byte the `arrival` arm**. §4.3 on that workload cannot show a
difference that exists. Same species as your vacuity guard, different cause:
a setup that manufactures evidence against our own contribution.

**The fix is your §4.5, not a bespoke generator.** ShareGPT/Alpaca with Poisson
arrivals via `vllm bench serve` carries the variation this needs — generation
lengths differ per request, multi-turn sessions have real gaps — so your
Pie-workload work and this gate are one job.

**Gate before §4.3 runs:** report `host_affinity` min/median/max across the
running set and require a real spread. Near-zero spread ⇒ the workload is still
inert ⇒ do not run §4.3.

## Paper text I had to change because your decisions invalidated mine

I had built a *"weight precision is the clean lever on HBM/C"* argument in §3
and a matching evaluation slot around the fp8/fp16 pair. With §4.4b dropped,
both now state plainly that **the `HBM/C` term is asserted, not measured**, that
the law is validated across **bandwidth** instead, and that a reviewer asking
for that axis is asking for the experiment we chose not to run. Your own plan
text made the same point; the paper now matches it. The 70B confirmatory slot is
removed from §6.1 with the reason (B_cap ≈ 7 against a 15% threshold).

## State you are inheriting

| | |
|---|---|
| `paper/tex/` | §1, 2, 3, 4, 6, 7, 8.5 are real prose. **§5 implementation is still a skeleton.** |
| Build | `~/bin/tectonic -X compile main.tex --outdir <dir>` — there is no LaTeX on this cluster otherwise. Builds clean: no undefined refs, no multiply-defined labels. |
| `paper/ARCHITECTURE.md` | The system described from the code, with file/function names against every claim. Includes the execution model (no SM partitioning; two sequential flash calls over disjoint row subsets) and the Pie comparison. |
| `paper/data/` @ `f3df05bdb` | Frozen A100 dataset with a manifest. Figures build from it and nothing else. |
| Overleaf | `paper/kvtiering-mlsys2027.zip`, regenerated on each paper commit. |
| `\todo` / `\needsgh` | Become **compile errors** under `\draftfalse`, so nothing unresolved can ship. |

**Pie has no public artifact** — checked the paper, the first author's page, and
the arXiv listing. The comparison is running *our* system on *their* workload,
not running their code.

**SLURM 2174626 on nova06** was allocated for the smoke test. Take it or release
it; I will not touch the GPU again.

Everything else is in the commit messages.
