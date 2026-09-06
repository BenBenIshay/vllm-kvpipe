# PAPER_OUTLINE.md — RETIRED 2026-09-06

**Do not use this document.** It is retired, not merely stale: several of its
claims are now known to be wrong, and it was written under a framing the paper
no longer uses.

The full 408-line original is preserved in git at **`2bfc6ca5e`**:

```bash
git show 2bfc6ca5e:paper/PAPER_OUTLINE.md
```

## Where each part went

| Part of the outline | Now lives in |
|---|---|
| §2 thesis, §3 analytical spine | `paper/tex/sections/03-model.tex` — written, with the numbers regenerated from the committed model |
| §4 positioning table | `paper/tex/sections/07-related.tex` — written, and every DirectKV quote **verified against the PDF** (`/mnt/beegfs/benishay/papers/`) |
| §5 contributions | `paper/tex/sections/01-intro.tex` — restated under the memory-system framing |
| §6 assets, §7 development plan, §8 section plan, §11 open decisions | `paper/PAPER_PLAN_2026-09-05.md` |
| §9 fallback paper, §10 risks | `paper/PAPER_PLAN_2026-09-05.md` §4 (branch on the gate) and §6 (attack surface) |
| GH200 specifics | `paper/GH200_EXPERIMENT_PLAN.md` |

## Why it was retired rather than updated

Four of its load-bearing statements are now false. Recorded here so that anyone
who does open the git version knows what to distrust.

1. **Framing.** It positions the work as *"the policy layer for host-memory
   KV"* — a layer on top of DirectKV's kernel. That framing is structurally
   subordinate and invites "so you wrote the scheduler they skipped". The paper
   is now a **heterogeneous memory system for LLM decode**: HBM and host DRAM
   managed as one KV memory, of which placement is one component.

2. **`eff_w = 0.78`** (§7, P4 status). Wrong; the settled value is **0.75**,
   which is what `paper/model/killer_v2.py` uses and what §3 of the paper
   reports. The outline contradicts itself — its own §"measured parameters"
   says 0.75.

3. **The gain figures ×1.22 / ×1.46 / ×1.59.** These came from a
   re-parameterisation that was never committed, and the model in the
   repository did not reproduce them. Regenerated 2026-09-06 from the committed
   script at the **measured** 72.7 GB fp8 checkpoint: see `tab:gain` in
   `03-model.tex`. They are also labelled "capacity multiplier" there when they
   are throughput gains — three different quantities under one name.

4. **Model choice.** It targets Llama-70B as the headline. Superseded: 70B
   gives `B_cap = 7.2` at 8K, so one request is 14% against a 15% kill
   threshold. **Qwen2.5-32B is the primary model**, with the fp8/fp16 pair as
   the clean lever on `HBM/C`. See `GH200_EXPERIMENT_PLAN.md` §1.

Also superseded but recorded elsewhere: the "2.67× serve-more" result (a
short-prompt/long-generation special case), the P1 plan to integrate DirectKV's
kernel (SM90a, uncompilable on this cluster), and the losslessness contribution
(cut — exact placement is what moving bytes does, and arguing for it invites
the doubt it was meant to settle).

## Live documents

- `paper/PAPER_PLAN_2026-09-05.md` — venue, claim ladder, schedule, figures, attack surface
- `paper/GH200_EXPERIMENT_PLAN.md` — the rental
- `paper/GH200_HANDOFF.md` — brief for the session running the GH200 track
- `paper/BLOCKER_PREFILL_RESIDENCY.md` — the measurement narrative, Aug 12 → Sep 6
- `paper/tex/` — the paper itself
