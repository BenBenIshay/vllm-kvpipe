# Paper source — working practice

Target: **MLSys 2027, deadline 2026-10-30.** Plan: `../PAPER_PLAN_2026-09-05.md`.

## Where things live

| | where | edited in |
|---|---|---|
| Prose (`.tex`) | this directory, in git | Overleaf, synced back here |
| Data (`.json`) | `../data/` | never — written once by a freeze |
| Figure scripts | `../figs/*.py` | git only |
| Figure PDFs | `../figs/*.pdf` | built locally, uploaded to Overleaf |

Overleaf is the editor; **git is the source of truth.** Import this directory
as a zip, or link Overleaf to the repo. Never edit `../data/` or `../figs/*.py`
in Overleaf — a number that exists only in a web editor cannot be traced to the
run that produced it.

## The `% src:` rule

Every numeric claim in the text carries a comment naming where it came from:

```latex
ITL is linear in total KV tokens with $R^2 = 0.975$.
% src: paper/model/p4_llama8b_a100.json @ 2bfc6ca5e
```

This is not bureaucracy. Numbers in this project have been measured, retracted
and re-measured repeatedly — the `+12%` concurrency claim, the `b_tok` that was
2× wrong for twenty runs, the "89% did not hide" reading of a single point, the
"break-even ~7 riders" that ignored queueing. A number without a source is a
number we cannot defend in a rebuttal, and we will not remember which run it
came from in eight weeks.

## Building figures

Figures are built from a **frozen** dataset — one commit, one node, one run —
recorded in `../data/manifest.json`. Never from scraped logs.

```bash
python ../figs/f4_rider_cost.py --data ../data/hide_efficiency.json   --out ../figs/f4.pdf
python ../figs/f5_capacity.py   --data ../data/preload_capacity.json --out ../figs/f5.pdf
```

Both scripts **refuse to plot a dataset containing a failed arm**. A failed arm
still writes its JSON, with `{"error": "server did not start"}` in every row;
one of those was nearly frozen and plotted on 2026-09-05.

Check `manifest.json` before quoting anything:

```json
{ "valid": true, "commit": "...", "dirty_files": 0, "dirty_end": 0 }
```

`valid: false` means a step failed or the tree changed mid-run, so the arms may
not share code. Re-run on a quiet tree.

## Drafting aids

`\todo{...}` and `\needsgh{...}` mark unwritten prose and slots that need GH200
numbers. Both render in colour while `\drafttrue`; setting `\draftfalse`
converts them into **compile errors**, so an unresolved slot cannot ship by
accident. Set it before generating the submission PDF.

## The style file

`main.tex` uses `mlsys2027.sty` if present and falls back to `article` with
1-inch margins otherwise, so the draft compiles before the author kit is
dropped in. **Get the real one before the first full-length compile** — the
fallback has a different text block, so page-count estimates from it are
meaningless.

## Section status

| file | state |
|---|---|
| `03-model.tex` | **drafted** — the analytical spine; survives every branch |
| `07-related.tex` | **drafted** — positioning and the endpoint argument |
| `02-background.tex` | skeleton; must state the P/D system model up front |
| `04-design.tex`, `05-implementation.tex` | skeleton, W1 |
| `06-evaluation.tex` | skeleton; the **only** section that forks on the GH200 gate |
| `01-intro.tex`, `08-discussion.tex`, `00-abstract.tex` | skeleton, W5/W7 |

Write order is deliberate: the branch-independent sections first, so a failed
GH200 gate costs one section rather than a paper.
