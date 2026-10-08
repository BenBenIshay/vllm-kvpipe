# The review process, 2026-10-07/08 — what was run, why, and what it found

A record of the adversarial-review cycle on the paper and the GPU experiment it
forced. Written because the outcome was a **retraction of the paper's headline**,
and the chain of reasoning that produced it should be auditable rather than
summarised in a commit message.

Commits `dc5bc589f .. 177e1afee` (13). GPU jobs 2182024, 2182025, 2182795.

---

## 1. Why there was a process at all

Ben reviewed the first page and found it poor: a vacuous metric in the abstract, a
claim that is table stakes presented as a result, an abstract twice the length of
the papers we cite, an unscoped opening sentence, hardware-specific numbers in
general prose, and an introduction that described our architecture before
positioning the field. He asked for a tier-A reviewer agent to be stood up and
iterated against until the paper was defensible.

Three review rounds ran. **The score never moved off 2 (weak reject, confidence 4.)**
Presentation reached a 3 by the reviewer's own assessment after round 1; the blocker
throughout was substance, and in round 3 it became a measurement question.

---

## 2. What each round found

### Round 1 — the paper as it stood

Score 2. Three claims were **false**, and I had written two of them in the previous
session:

| claim | refuted by |
|---|---|
| "None serves both tiers in the same step" (bolded, abstract) | BOOST serves ~9% of accesses from host *concurrently* with HBM — and our own §7 says so, written a day after the abstract claim |
| "output token-identical to the unmodified engine" | our own §6.8: greedy decoding matched stock vLLM "6 of 6 prompts in one round and **4 of 6** in another" |
| the regime needs "a context long enough that KV rivals" the weights | §3.2's headline result is that **context length cancels** — a self-contradiction 40 lines apart |

Plus: the 13-page body contained **one** figure, a results plot in the intro, with
all five diagrams in Appendix D — so the introduction sent readers to page 22 for
the core mechanism; `tab:gain`'s caption said "2048 context" while its own `% src:`
said `S=8192`; and the 80 ms target's stated justification ("the first round figure
above ITL₀ + F") is arithmetically wrong, since the four sums are 48/47/37/37.

**I found one the reviewer missed, and it was on page 1.** `f6`, the intro's only
figure, is generated from the across-the-knee dataset this paper *retracts*, and
annotates a band "overhead the model does not carry" — under a caption I had
rewritten to claim the law **is** affine below the knee. Plot and caption
contradicted each other. Removed.

### Round 2 — the first serious challenge

Score 2, unchanged. The reviewer differenced `ΔT(2) − ΔT(1)`, in which `F` cancels
identically, and observed that at ctx 1080/1440 it gives 2.40 and 2.90 ms against
our shipped `c = 0.52` — while the law we had **retracted** predicts 2.19 and 2.85.
If right, Table 2 collapses.

I checked it. Both those contexts sit at `ℓ/T₀` of 0.48 and 0.63, so context and
transfer-exposure are confounded there, and the retracted law was *itself* fitted
across the knee — making it a proxy for exposure, not context. I produced a
low-context test (128 → 256, `ℓ/T₀` under 0.12) showing −3% where the rival needs
+53%, and put it in §6.3 as a falsification test.

**That table was wrong, and round 3 caught it.**

### Round 3 — the reviewer audited our own JSONs

Score 2. Four charges, **all four reproduced**:

1. **My falsification table was not reproducible from its own cited files.** The
   1.17 ms averaged a campaign-15 arm with a campaign-21 arm whose `F` differs by
   2×, then compared that against a campaign-15 arm. Campaign-matched it is
   0.975 → 1.135, i.e. **+16%, not −3%**, and the two ctx-128 replicates differ by
   **0.285 ms** — larger in absolute terms than the effect it claimed to exclude.
   Our own `% src:` convention exposed it: three files cited for a two-cell table.
2. **`c` rises with context even with the bytes removed** — `MB_c128` 1.072 →
   `MB_c256` 1.190, +11%, same job, `ℓ/T₀ = 0.007` in both.
3. **"across seven arms `c` repeats to 1.5%"** is false: across twelve arms of one
   configuration `c` spans 0.414–0.538, i.e. **±12%**.
4. **`F(L) = 0.053L` is unsupported.** On the only depth sweep we have, `F/L` runs
   0.058, 0.071, 0.095 at L = 16/32/64 — **rising 64%**, not constant. That
   assumption was made by dividing one depth's measurement by its depth, and
   Table 2 charges it at L = 80.

The reviewer also **withdrew** their round-2 claim that Table 2 extrapolates past
its fit window: `F = 3.4` and `c = 0.52` do come from one regression over
`r ∈ [1,8]`, and `B_off = 6.17` sits inside it. That part of the paper was right.

---

## 3. The experiment: M20 (job 2182795)

Round 3 left one question deciding the paper: **does `c` have a context term?** Two
prior attempts to answer it from the archive were confounded — one by exposure, one
by mixing campaigns — so it was measured directly.

**Design.** `ΔT(2) − ΔT(1)` cancels `F` identically. Sweep context 128 / 256 / 512
at fixed depth, batch and link; **three replicates per cell in one job**,
interleaved across contexts rather than blocked so machine drift could not imitate a
context effect; then repeat with the bytes removed (`--xfer-mode minbytes`) so
transfer and host work separate at every context. `ℓ/T₀` at r=2 stays ≤ 0.23,
inside the paper's own below-the-knee gate, so exposure could not explain a rise.

**Predictions recorded before the run**, in the sbatch's commit message: flat `c`
gives 0% across the ladder; the two-point `minbytes` trend ~+25%; the withdrawn
context law +161%.

15 arms, 51 minutes on nova07.

### What it found

Fitted over a **matched rider range** at every context, `r ∈ [1,4]`:

| `c` (ms) | ctx 128 | ctx 256 | ctx 512 |
|---|---|---|---|
| bytes sent | 0.41 | 0.51 | **1.21** |
| bytes removed | 0.43 | 0.19 | 0.99 |

**`c` roughly triples over a fourfold context increase, at `ℓ/T₀ ≤ 0.23` — inside
our own gate.** So the gate is not conservative enough, and the per-request cost
measured at 128 tokens is not the cost at the context Table 2 evaluates.

**A confound in my own first analysis pass, found before reporting.** The gate
filters on *total* link time, so it admits eight rider points at ctx 128, five at
256 and **two** at 512 — comparing those fits compares an eight-point regression
against a two-point difference. `paper/model/c_context.py` now fits a fixed rider
range and prints both estimators.

**What it does not settle.** The bytes-removed row cannot apportion the rise between
transfer leakage and host work: two replicates rather than three, and because `ΔT`
is concave its value moves with the fit range (ctx-512 reads 0.43 over eight points,
0.99 over four). The rise is solid; its mechanism is not.

---

## 4. The consequence

Table 2 charges `c = 0.52` at 8192 tokens — four octaves past the largest context
measured, at `ℓ/T₀ = 0.93` rather than 0.23. Freezing the unhidden transfer fraction
at the largest value measured (11.6% of a request's link time) and applying it at
the target puts `c` near 1.2 ms and the Llama-70B row at **×1.11**; letting the
fraction continue to climb, as it does across our three contexts, takes it **below
unity**.

**The point estimates are withdrawn.** The abstract and §6.5 now give a range that
includes no benefit — roughly **×0.98 to ×1.4** — rather than ×1.25 and ×1.37.

### What survives, and it is not nothing

- the **mechanism**, measured: zero-device-block admission, co-serving in one pass
- the **knee**, confirmed on both sides at two contexts, and the finding that two
  contexts collapse onto one curve against `ℓ/T₀` with the boundary at unity
- the **ordering** by `HBM/Ckv`, unchanged under every scenario in the sensitivity
  table
- the **cost structure** and its failure modes, including four optimisations whose
  mechanism fired and whose benefit did not

### What would close it

The same ladder continued to the target context. **It needs more device memory, not
a faster link** — a materially cheaper ask than a GH200, and the first experiment to
run with either.

---

## 5. Honest assessment of the process

Three rounds did not move the score, and that is the correct outcome rather than a
failure of the rounds: rounds 1 and 2 fixed presentation, which was never the
blocker, and round 3 located the blocker in a coefficient rather than in prose.
The adversarial reviewer found three false claims I had written and one I had
introduced *during* the review; I found one it missed. The measurement that was run
to defend the headline refuted it instead, which is the outcome the campaign's own
record should have led us to expect — four of nine optimisations in the earlier
campaign had a mechanism that fired and a benefit that did not.

The paper is weaker and correct. It was stronger and wrong.
