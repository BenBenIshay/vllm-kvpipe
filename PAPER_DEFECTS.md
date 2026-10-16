# Paper defects found while writing the design document

Found 2026-10-14/15 while reconstructing the architecture from the code. Each is a
place where a committed file asserts something the project has since retracted, or
states something the code contradicts. Priority order; item 1 is the one that would
do real damage at review.

**1 `07-related.tex:24` makes a provably false claim about an OSDI paper.** It says
of DirectKV "they pay no device memory and get no overlap". DirectKV's abstract claims
"warp-level pipelining to overlap KV fetching, computation, and write-back" and it has
a section measuring it. Our own `F-coherence.tex:19` states the correct version -- a
window of one tile, with almost no arithmetic to hide behind. **Fix the body to the
appendix's wording.** The same sentence also carries the retracted "hide about 90% of
the transfer", which appears again at `C-related.tex:238`. Most damaging item in the
paper.

**2 The SuperInfer claim is sourced to an abstract.** `refs.bib` says so explicitly
and says to read the body first; there is no local copy. The claim it supports -- that
a rotated-out request is not generating -- is the one separating us from the closest
competitor.

**3 `H-ridercost.tex:128` overstates the obstacle to fusing.** It claims a fused
call needs "the rider cohort's table rebuilt per layer on the hot path". The table is
built **once per step** and reused by every layer (§4.2); a fused formulation needs
`k` tables per step, not a hot-path rebuild. Same error in
`OVERHEAD_CHARACTERISATION.md:816`. This makes our own backlog look harder than it is.

**4 `04-design.tex:143` mislabels a prefetch ablation as a `k>1` ablation**, and
`k=4` is unjustified. Both arms of the 0.92-vs-7.20 comparison ran `k_window=4`; what
varied was the side-stream prefetch. And the one `k` sweep we have reads
61.24 / 62.10 / 61.91 ms at `r=1` for `k` = 2 / 4 / 8 -- **`k=2` marginally best**, all
within drift -- while `killer_v2.py` charges every rider `4/64` of device memory for
`k=4`.

**5 `H-ridercost.tex` still says "across seven arms the per-rider cost repeats to
1.5%".** Across twelve arms of one configuration it spans 0.414-0.538, i.e. **+-12%**.

**6 `08-discussion.tex:53`** -- "bounded below by 0.83 ms of unavoidable copies".
Not a bound: that is our launch pattern's cost on this host, and the count floor is
`O(1)` per step (§6.2, C7c).

**7 `overhead_model.py:112` and `fusion_bound.py:37`** still carry **1.94** for the
second attention call. That was a *residual* -- `F` minus the mirror write minus the
streaming -- then **attributed** to the call. Direct ablation says the call is **1.21**.
So the residual is not simply wrong; **~0.7 ms of it is something else and is
currently unattributed**, and §6.2's C6 ordering operations are the obvious candidate.
Needs a decision, not a blind edit.

**8 `REVIEW_PROCESS_20261008.md`** was never amended after the
retraction-of-the-retraction, so it reads as though the headline is withdrawn to a
range including no benefit. The settled position hours later is x1.2-1.4 for the two
weight-heavy configurations.

**9 `E-implementation.tex`** still presents the launch-bound account of `F` as fact,
contradicting `05-implementation.tex` and `06-evaluation.tex` -- and contradicting T5,
which cut launches eightfold and moved nothing.

**10 `ORCHESTRATION_TAX_PLAN.md`** header still reads "Status: active" with the dead
+62%/+105% claim; the nullification is 300 lines down. `09-overhead.tex` is correctly
headed stale and excluded from the build -- listed so it is not "fixed" by accident.
