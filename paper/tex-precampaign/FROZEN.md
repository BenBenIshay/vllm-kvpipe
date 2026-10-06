# Frozen: the paper as it stood BEFORE the overhead-campaign revision

This is a verbatim copy of `paper/tex/` at commit **`78912d6c7`**, taken
2026-10-06, so the revision that follows can be read against what it replaced.
The matching build is `paper/out-precampaign/main.pdf` (25 pages, body 11.0pp,
`build.sh` exit 0). Tag: `paper-precampaign-20261006`.

**Do not edit anything here, and do not build from it.** It exists to be compared
against, nothing else. The live paper is `paper/tex/`.

## Why the paper needed revising

Every coefficient in this version was fitted over rider sweeps whose top points
sat AT the knee, which raises the fitted slope and lowers the fitted intercept
from a single cause. Re-measuring below the knee on the shipped code gave:

| | this frozen version | measured |
|---|---|---|
| `c` (per rider) | 2.2–2.9 ms | **0.61** ms at L=80 |
| `F` (fixed) | 12.3 ms | **4.24** ms at L=80 |
| the law in `r` | "not affine ... fits a negative fixed cost" | affine, both coefficients well identified |
| what `F` is made of | ~640 kernel launches, "launch-bound" | ~56% is the per-layer second attention call |

Three claims in this version are not merely stale but wrong in mechanism:

1. **`06-evaluation.tex:149-161`** says the per-rider cost is *not* affine and
   that `ΔT = F + c·r` "fits a negative fixed cost", quoting non-monotone
   residuals 18.1 / 8.6 / 13.5 ms. Below the knee the law is affine and the
   coefficients repeat to 1.5% and 10%. The non-monotonicity was the knee.
2. **`05-implementation.tex:53-67`** attributes `F` to ~640 extra kernel
   launches. T5 cut the launch count eightfold with no resolvable effect on
   either coefficient, so launch count is not what `F` is.
3. **`03-model.tex`** has no `r*` (knee) formula at all, although the knee is
   what actually decides whether the feature pays — and because riders are
   integers, the gain is a sawtooth in context that this version cannot express.

## Where the superseding numbers live

- `paper/OVERHEAD_CHARACTERISATION.md` — the report, with provenance per number
- `paper/model/overhead_model.py` — `FINAL`, `closing_state()`, selftest 27/27.
  `COEFF` in that same file is the superseded across-the-knee fit, kept for
  audit; never re-derive from it
- `paper/model/operating_point.py` — the integer-rider sawtooth and the
  operating-point recommendation, selftest 7/7

## One thing this freeze is NOT evidence of

`build.sh` did not pass on this tree as committed. Its `% src:` gate failed on
`paper/data/baseline_integrity_20260912.log`, which `06-evaluation.tex` and
`B-correctness.tex` both cite and which had never been committed — `*.log` in the
generic Django block of `.gitignore` had silently swallowed it. A second cited
log passed only because a stray local copy existed and would have failed on any
fresh clone. Both are now tracked and the ignore rule is negated for
`paper/data`. So the exit-0 build above required that fix first; the frozen
*sources* are otherwise untouched.
