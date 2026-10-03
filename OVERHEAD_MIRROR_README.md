# Overhead characterisation — where everything is

**Everything in this directory is a copy.** The originals live on the branch
`perblock-d2h-opt` (which carries `overhead-characterisation` in its history),
inside a worktree at
`/mnt/beegfs/benishay/vllm-kvpipe/.claude/worktrees/overhead-char/`, which is
**not visible from the main checkout** — a worktree materialises only its own
files, and `/mnt/beegfs/benishay/vllm-kvpipe/paper/` will not show any of them.
That is why these copies exist. If a file here looks stale, the branch is the
source of truth.

Branch head as of this copy: **`ae4b51d00`** on **`perblock-d2h-opt`** (the
optimisation work). The characterisation campaign's own baseline is
**`a90ed1b49`** on `overhead-characterisation`, which is what the optimisation
results are measured against. Not pushed — there are no GitHub
credentials in the run environment — so **this directory is the copy that
survives if the worktree is deleted.**

## Read these

| file | what it is |
|---|---|
| `OVERHEAD_CHARACTERISATION.md` | **The report.** How `c` scales, and whether a faster link buys throughput. Start here; **§9 is what was then built and what it measured.** |
| `OPT_PROPOSAL_PERBLOCK_D2H.md` | The optimisation proposal, now **EXECUTED**. Kept as the design record; its four decisions are resolved in a table at the top, with a note on what it got wrong. |
| `09-overhead.tex` | **The paper chapter.** Deliberately *not* `\input` into `main.tex` — that is a page-budget decision, one line when you want it. In the branch it is at `paper/tex/sections/09-overhead.tex`. |
| `OVERHEAD_PROGRESS.md` | The chronological campaign log. Superseded by the report; kept for the dead ends and corrections. |

## Scripts, if you want to regenerate a number

Run these from the worktree root (they need the repo's `.venv` and the data
under `../02_law` etc.):

| file | regenerates |
|---|---|
| `c_scaling.py` | report §3.1, §3.1b, §4 — the per-arm table, the slope band, the route sensitivity. `--selftest` is 10/10. |
| `amortization.py` | report §2 — the rider-sharing question. `--window` is the primary diagnostic; `--m7` the high-concurrency sweep. |
| `overhead_model.py` | report §0, §3.3, §5, §7 — the link sweep, the composition, the envelope. |
| `overhead_fit.py` | the model-selection fit with bootstrap CIs and the knee / model / job-offset filters. |
| `lint_section.py` | structural lint for the paper sections (no pdflatex on this cluster). |
| `m7_amortization.sbatch` | the M7 sweep itself, with both predictions registered in its header. |
| `t1_effect.py` | T1's effect on the block-table rebuild, from the Class A host timer. |
| `t2_effect.py` | T2's bracketed before/after on `F`. |
| `phase0.py` | the probe arm that priced the synchronisation and decided T2. |
| `make_manifest.py` | regenerates `paper/data/overhead/MANIFEST.json`. |

## Raw data

`00_baseline/ 00b_ramp/ 01_perturbation/ 02_law/ 03_d2h/ 04_stall/
07_amortization/ 08_opt/ 09_t2/` — the harness JSONs, plus the sbatch `.out` logs at this
level. The branch also carries these under `paper/data/overhead/` with a
`MANIFEST.json` giving each record's config, source commit, purity verdict and
a sha256 prefix (75 records; 0 from a modified tree; 1 non-connector run, which
is the deliberate M0 control).

## Two items that need you

1. **`check_tables.py`** — I added `128` to its bandwidth allowlist, because its
   "one value per quantity" guard globs every file in `sections/` including the
   unwired chapter and flagged the PCIe6 sweep point. That extends the set for a
   genuinely new link rather than loosening a pin on an existing quantity, and
   it exits 0. Alternative, if you prefer: have the guard skip sections that
   `main.tex` does not `\input`.
2. **A dead evidence pointer in the paper, pre-existing.**
   `06-evaluation.tex` and `B-correctness.tex` both cite
   `paper/data/baseline_integrity_20260912.log`, which exists in your main
   checkout but is **not in version control** — live for you, dead for a
   co-author or a fresh clone. The fix is to commit it, not to re-measure, so I
   left it alone.
