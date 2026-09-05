# Handoff: GH200 track

Written 2026-09-05 by the paper session (`vllm-kvpipe-03`). Ben split the work:
**one session owns the GH200 track, the other owns the paper.** This is the
brief for whoever holds the GH200 track. It duplicates a cross-session message
that may not have been delivered — treat this file as the authoritative copy.

## Why this exists

Two sessions collided in this repo on 2026-09-05: a freeze was nearly launched
onto a GPU that already had a sweep in flight, and one session's commit swept up
the other's uncommitted `.tex` files. A boundary is cheaper than a merge.

## Ownership

| | owner |
|---|---|
| `benchmarks/kv_pipelining_bench/*`, `paper_runs/*`, `paper/GH200_EXPERIMENT_PLAN.md` | **GH200 track** |
| `paper/tex/`, `paper/figs/`, `paper/data/`, `paper/PAPER_PLAN_2026-09-05.md` | **paper track** |

If the GH200 track needs a number changed in the paper, message the paper
session rather than editing `paper/tex/`.

## Three things that changed on 2026-09-05

### 1. C2C is 450 GB/s, not 900 (commit `f65738c4a`)

900 is the **bidirectional** figure; a rider's KV stream is unidirectional
host-to-device. Every C2C projection in `BLOCKER_PREFILL_RESIDENCY.md` was
optimistic by exactly 2× and is corrected in place.

- per-rider collapse ~36× → **~18×**; riders that hide ~75 → **~37**.
- `gh200_preflight.py`'s docstring said 36× / ~70 riders → now 18× / ~37. **The
  script itself was right** because it *measures* the link instead of assuming
  one; only the comment was wrong.
- One correction **reverses its sentence**: 49 GB was "~54 ms on C2C, so ~90
  riders sits right at the hiding limit". At 450 it is ~109 ms, so 90 riders is
  ~2.4× **over**. The ~75 and ~90 estimates never agreed; both were read
  against the wrong constant.
- **Not affected:** the gain law already sets `x=450` in
  `paper/model/killer_v2.py`, so the endpoint predictions stand.

### 2. Setup trap #2 is now backwards and would handicap arm C (commit `c125c4aa1`)

`BLOCKER_PREFILL_RESIDENCY.md` says `b_hbm_max` must be **below** the measured
all-HBM capacity. That predates zero-block preload admission (2026-08-30).
Riders no longer own a block, so they **add** rather than displace — the 8 → 48
result was itself obtained at `b_hbm_max` = the measured **ceiling**, which the
old rule forbids.

**Correct rule: `b_hbm_max` = the MEASURED all-HBM ceiling, riders on top.**

The surviving half of that trap is the diagnostic: if the mixture's tier
histogram shows no `cpu` entries, no rider was created and the arms are
identical *by construction*, not by measurement. Check the histogram before
believing any comparison.

### 3. Experiment 4.2b added — the rider cost curve (6 GPU-h)

It was missing, and figure F8 (the crossover: PCIe point + C2C point + the model
line through both) cannot be drawn without it.

**Sweep `--caps 1,2,3,4,6,8,12,16,24,32`, not the A100 default of 1,2,3.** Two
runs of those same three points fitted `12.29 + 4.45` and `9.55 + 4.66` ms
because the cap=2 point moved 69.92 → 65.93. That split is exactly what sets
C2C break-even, since the fixed term is what a faster link does **not** shrink.

Ordering: run it **right after 4.1**, ahead of the optional phases. It is the
one experiment that survives a failed 4.1 — the fallback crossover paper is
4.2b plus the frozen A100 curve.

## Frozen dataset — do not overwrite

`paper/data/` @ `f3df05bdb`, `manifest.valid = true`. All figures build from it
and nothing else.

    rider cost   49.99 / 65.66 / 65.93 / 74.98 ms at 0/1/2/3 riders
    capacity     all-HBM peak 8, mixture peak 48 (42 delivered, 42 reclaims)
    goodput      57.1 -> 55.8 tok/s, -2.3%

Two corrections it produced, both relevant to the cost-curve work:

- **Break-even is NOT reached within 3 riders.** tok/s 57.5 / 54.7 / 55.8 / 57.0
  — three riders sit 0.9% *below* rider-free. The record's "empirical break-even
  3–4" is really "parity approached at 3, crossing never observed".
- **The fixed/slope split is unstable at three points** (see 4.2b above).

## GPU coordination, learned the hard way

**A poll of `nvidia-smi --query-compute-apps` is not a lock.** It reads 0 while
a harness is *between engines* — one arm finished, the next not yet started.
Two runs then share the GPU and both are invalid. Check for live **processes**:

```bash
ps -u $USER -o args= | grep -cE "[.]venv/bin/python.*(kv_pipelining_bench|vllm[.]entrypoints)|^VLLM::EngineCore"
```

The `[.]` bracket matters: a plain pattern self-matches the shell asking.

**Run a multi-step sequence inside ONE `srun`.** Per-step `srun` from a
login-node parent orphans children that keep running and keep writing to the
output paths after the parent is killed.

**`rider_hide_efficiency.py` now takes `--logdir`** (commit `96e98059b`). Its
output directory was hardcoded to `paper_runs/20260904`, so a rerun silently
overwrote the sweep quoted in commit `38e88917c`. Use `--logdir` for anything
new; `20260904` holds published numbers now.

## Work package while the box is unbooked

Ben books ~Sep 19. Rental time is for measuring, not debugging.

1. **`endpoint_comparison.py` still uses `steady_concurrency` as its headline
   metric.** It is a mean over sampled steps, so a *slower* arm holds requests
   longer and scores *higher* — it produced the retracted "+12% concurrency".
   Fix before the rental.
2. **Close `GH200_EXPERIMENT_PLAN.md` §2:** dry-run every experiment script on
   A100 with Qwen-0.5B so only constants change on the rented box; pin the
   environment; decide the SLO grid in advance.
3. `decode_only_bench.py` has the right P/D shape but no mixture mode.
4. Ben's own actions: book the box (**CUDA `devel` image — verify `nvcc` in
   minute one**) and pre-stage the 72.7 GB fp8 checkpoint.

## One request from the paper side

`paper/model/killer_v2.py` still assumes **70 GB** of weights where the fp8
checkpoint measures **72.7 GB**. That lowers $C$ from 22 to 19.3 GB and moves
the projected gain **up**. §3's table currently carries a blocking TODO because
the committed script does not reproduce the published numbers. Whoever owns the
model should update it and regenerate — see `paper/tex/sections/03-model.tex`.
