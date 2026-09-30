# Overhead characterisation — live progress log

Campaign: decompose `F` (the per-step cost of using the host path) into its
static / per-rider / per-byte / per-block parts, then project it onto NVLink-C2C.
Plan: `~/.claude/plans/fluttering-dreaming-kite.md`. Measurement only — no
feature optimisation.

**Read the "Open concerns" section first.** It is where anything that might need
Ben's steer is parked, newest first. Everything else is append-only history.

Branch `overhead-characterisation`, worktree `.claude/worktrees/overhead-char`,
forked from `05b32de56` (`hybrid-kv-tiers`).

**Where to read this.** The work happens in an isolated worktree, so the copy
in git is at
`.claude/worktrees/overhead-char/paper/OVERHEAD_PROGRESS.md` — not at
`paper/OVERHEAD_PROGRESS.md` in the main checkout. A mirror is kept at
`/mnt/beegfs/benishay/paper_runs/20260930/overhead/OVERHEAD_PROGRESS.md`,
next to the campaign's run outputs, and updated whenever this file is.

---

## Open concerns

### C1 — ANSWERED 2026-09-30. Measured: the rigs differ 5-8%, with inconsistent sign.

Arm A (`--connector`) vs arm B (co-located), same config, same session:

| r | arm A dT | arm B dT | B vs A | A GB/step | B GB/step |
|---|---|---|---|---|---|
| 2 | 17.91 | 16.95 | **-5.4%** | 1.115 | 1.115 |
| 4 | 43.90 | 47.20 | **+7.5%** | 2.230 | 2.290 |
| 6 | 92.93 | 98.35 | **+5.8%** | 3.379 | 3.464 |

Arm B is heavily contaminated, as expected — 32,749 to 45,031 prefill tokens over
5 to 10 steps, `max_waiting` 12 to 18, and even its r=0 baseline impure. Yet the
differencing estimator cancels most of that: what survives is 5-8%, **and the
sign is not consistent**, so it is bias rather than a correctable offset.

Verdict: material for a paper whose headline margin is "a few milliseconds", but
it does not by itself flip a conclusion. The bigger threat is C4.

**A correction to my own earlier claim.** I wrote that co-located "structurally
cannot exceed 2 riders", citing the harness's `--connector` help text ("riders
cap at 2 however many are offered") and the archived `curve_32b_v2.json`. This
fresh co-located run created 4 and 6 riders without difficulty — `cpu: 4` and
`cpu: 6` with GB/step rising 1.115 -> 2.290 -> 3.464. So the cap is
configuration-dependent, not structural, and whatever pinned the archived run at
2 was something else. The substantive point stands unchanged: that archived
dataset's r>=4 rows are still invalid, because its GB/step *fell* from 1.115 to
0.543 while this run shows it need not.

### C1-orig — the original concern, for the record

**The paper's `F = 12.3 ms` anchor comes from a co-located run, and rests on
about one usable rider point.**

It is derived from `curve_32b_v2.json`, which ran **without** `--connector`, i.e.
co-located, not the P/D-emulated decode-node setup ground rule 1 requires. The
same file is the source of the `0.86x` operating point quoted in the intro, §6
and the EuroHPC application.

Worse, its rider axis is not real above r=2: at r=4, 6 and 8 the engine admitted
only **2** riders, and `stream_gb_per_step` *falls* from 1.115 (r=2) to 0.543
(r=4) — fewer bytes moved at a higher nominal rider count. So r=4/6/8 measure
degradation at a fixed 2 riders, exactly as that file's own provenance warning
says. Usable rider points: r=1 and r=2.

I am proceeding as planned (M0 re-measures both arms), but flagging now because
if the connector A/B shows a material gap then **the `0.86x` figure and
Figure 1's cost curve are both affected**, and that is a paper-level decision,
not mine.

### C4 — OPEN. **The most consequential finding so far. Paper-level.**

**The paper's "not affine" result may be an admission-ramp artifact, not a
superlinear mechanism.**

Arm A reproduced the published decode-node curve to within **0.35%**
(53.45 / 71.36 / 97.35 / 146.38 ms against 53.512 / 71.632 / 97.546 / 146.886),
so the substrate is pinned and what follows applies to the published data too.

But the purity gate failed every rider row on one condition: **requests were
queued**, `max_waiting = r - 1`.

| r | running min/mode/max | tiers | max_waiting | steps | prefill |
|---|---|---|---|---|---|
| 0 | 16/16/16 | hbm 16 | 0 | 136 | 0 |
| 2 | 18/18/18 | hbm 16, cpu 2 | 1 | 138 | 0 |
| 4 | 3/20/20 | hbm 16, cpu 4 | 3 | 142 | 0 |
| 6 | 3/22/22 | hbm 16, cpu 6 | 5 | 146 | 0 |

The tier composition is *correct* — riders are created as requested, and the
modal batch is exactly `B_res + r`. The problem is the ramp: admission staggers,
so the step count grows 136 -> 146, about `2(r-1)` extra steps.

**Why that breaks the estimator.** Step time is
`(wall_long - wall_short) / (136 - 8)`, which assumes prefill, ramp and teardown
are identical in both runs and cancel. At r=6 the ramp needs ~10 steps while the
**short run is only 8 tokens long** — so the short run never reaches steady
state, its ramp is truncated relative to the long run's, and the residue
`R_long - R_short` lands in the numerator. The step is therefore
**overestimated, by an amount that grows with r**.

That is precisely the shape §6.5 reports as the paper's "not affine" evidence:
per-rider cost rising 9.06 -> 11.01 -> 15.56 ms at r = 2, 4, 6, from which it
concludes `dT = F + cr` "fits a negative fixed cost and the fixed-versus-slope
split is not a meaningful decomposition". If the rise is ramp residue, that
conclusion is an artifact — and so is the `F = -23.4 ms` the affine fit returns.

**It is directly testable and cheap:** raise `--short-len` until the short run
clears the ramp (48 instead of 8) and see whether the rise survives. Queued at
the top of the next job. I am NOT assuming the artifact explanation is right —
the rise may be real, and the test is what decides it.

**Consequence if it is an artifact.** The "not affine" claim, Figure 1's cost
curve, the negative-`F` argument, and the choice to model a kink rather than an
affine law would all need revisiting. That is a paper decision, not mine.

### C3 — OPEN, low risk, will be cross-checked in M3.

**Does the `minbytes` arm really keep the launch cost intact?** It issues the
same `2*L*R` `cudaMemcpyAsync` calls but each moves one block row instead of
`n`, so the per-byte term goes to ~0 while every launch is still made. The
assumption is that a memcpy's launch overhead is independent of its size, which
is true to first order but not exactly: very small transfers can be scheduled
differently by the DMA engine.

This is why M3's standalone launch-pattern microbenchmark is not optional — it
measures per-launch cost against transfer size directly, engine-free, so the
`minbytes` reading can be checked against it rather than trusted. If the two
disagree, the microbenchmark wins and `minbytes` is reported as a bound rather
than a measurement.

### C2 — OPEN. Methodological; already handled by the plan's gate.

**At least one archived curve silently measured resident-batch growth as "rider
cost".** See F1(e): `curve_marginal_eta.json` has r=4 and r=8 with **zero riders
admitted** (`stream_gb_per_step = 0.0`, `tiers = {hbm: 16}`) yet step time rises
53.5 -> 101.5 -> 103.9 ms.

The mechanism is almost certainly that `max_num_seqs = batch + max(riders)`, so a
request the controller refuses as a rider is still served **as a resident** — the
batch grows, the step lengthens, and the harness attributes it to riders.

This is precisely the contamination Ben warned about. The decode-purity gate's
`n_hbm == B_res` assertion catches it, which is why that assertion is not
optional. No action needed beyond keeping the gate.

---

## History

### 2026-09-30 — M0 started

**Setup.** Isolated worktree created from `hybrid-kv-tiers` HEAD, *not* from
`origin/main` — the tooling default would have branched from `origin/main` and
silently dropped the entire fork (this branch has no upstream and has never been
pushed). Own `.venv` copied (8.6 GB, a real copy; beegfs has no hardlinks) so the
editable-install `MetaPathFinder` can be repointed at this tree. That finder
hardcodes `MAPPING = {'vllm': '/mnt/beegfs/benishay/vllm-kvpipe/vllm'}` and is
consulted **before** `sys.path`, so `PYTHONPATH` cannot override it; sharing the
main venv would mean editing code here and measuring the main tree's code, the
failure that produced two wrong diagnoses on 2026-09-04.

Cluster state at start: no jobs queued, `nova06`/`nova07` idle (`supernova`,
1x A100-80 PCIe, ~1 TB host RAM, 64 CPU). 1.4 TB free on beegfs. Environment:
Python 3.12.13, torch 2.9.1+cu128, CUDA 12.8.

**F1 — audit of the eight archived rider curves in
`paper_runs/20260916/rider_curve/`, done before spending any GPU time.**

Connector status recovered from the run logs
(`grep -c PipelinedDecodeBenchConnector`):

| file | connector (P/D) | tier accounting | verdict |
|---|---|---|---|
| `curve_a100_final.json` | **ON** | `cpu` = 2/4/6, tracks the request | **the only clean rider axis** |
| `curve_32b_v2.json` | OFF | `cpu` pinned at 2 for r=4/6/8 | source of `F=12.3` and `0.86x`; only r=1,2 usable |
| `curve_32b.json` | OFF | broken (`{hbm: 1}`) | garbage: r=1 step 21.7 ms vs a 54.2 baseline |
| `curve_final_a100.json` | ON | `cpu` = 2/4/6 | rejected twin; r=2 **below** baseline (47.99 vs 53.41) |
| `curve_32b_stage12.json` | ON | `cpu` pinned at 1-2 | r=2 **below** baseline (47.71 vs 51.56) |
| `curve_marginal.json` | ON | `cpu` pinned at 1-2 | rider axis not real |
| `curve_marginal_eta.json` | ON | **zero riders** at r=4,8 | see concern C2 |
| `curve_32b_decode_node.json` | ON | `cpu` pinned at 1-2 | rider axis not real above r=2 |

Consequences, stated plainly:

* **(a) Exactly one archived dataset has a genuine rider axis** —
  `curve_a100_final.json`, connector on. Its affine fit is
  `F = -23.4 ms, c = 18.8 ms`. That negative fixed cost is what §6.5 reports as
  evidence the affine split "is not a meaningful decomposition".
* **(b)** The `F = 12.3 ms` the model actually uses comes from a *different*,
  co-located run whose rider axis is not real above r=2. So the paper's two
  overhead claims are not merely inconsistent in value — they come from datasets
  differing in the P/D setup **and** in whether the rider count is real.
* **(c)** Below-baseline r=2 samples appear in two files. The harness already
  discards samples below the rider-free baseline, so these are medians that still
  landed low — the differencing estimator is noisy at this operating point. That
  is independent motivation for measuring below the knee, where `dT` is large
  relative to its own error.
* **(d)** Verified against the transcriptions the paper cites:
  `curve_a100_final` = 53.512 / 71.632 / 97.546 / 146.886 ms with per-rider
  9.06 / 11.01 / 15.56 matches `paper/data/rider_cost_phase0.json` exactly;
  `curve_32b_v2` = 54.10 / 70.79 / 113.33 / 114.79 matches
  `rider_cost_4p2b.json`. Replacing the transcriptions with the real artifacts is
  therefore safe, and is still planned.

**Decisions taken without asking**

1. Branched the worktree from `hybrid-kv-tiers` HEAD rather than the tooling
   default `origin/main`; the default was simply wrong here.
2. Paid the 8.6 GB venv copy instead of a `sys.meta_path` hack to redirect the
   `vllm` import. The hack works but fails *silently*, and silent failure here
   means measuring the wrong code.
3. Audited the archived curves before booking GPU time. That turned up F1, which
   changes what M0 has to prove.

**Next**

* Finish the venv, repoint the finder, verify `import vllm` resolves into this
  tree.
* Allocate `nova07`; `run_suite.sh` -> 9/9; confirm `ulimit -l` is `unlimited`.
* Build the decode-purity gate checker, then run the connector A/B at the 4.2b
  config with the gate live (concern C1).

### 2026-09-30 — decode-purity gate landed (`8fd0fbc34`)

Asserts the four conditions per step by wrapping `Scheduler.schedule()`, which
gives per-step ground truth rather than the tier sampler's 50 ms sampling gap.
Class A: one dict pass and two `len()` calls per step, no CUDA, nothing that can
reorder the stream — cheap enough to leave on for every run, which is the point,
since a gate you have to switch on will be off for the run that needed it.

A rep failing the gate is dropped before the estimator sees it, not adjusted: it
is not a noisy measurement of the right thing, it is a measurement of a
different thing. Rows carry `usable`, which is what the fitter filters on.

Selftest is 6/6 and covers all four conditions plus the three real failure
shapes from F1 — and confirms that legitimate admission ramp at the window edges
still passes, which matters because the estimator's short run is mostly ramp.
The short run therefore gets only the contamination checks (prefill, queueing);
the long run, which carries the decode steps the estimate is made of, gets all
four.

Also: `--connector` is now required unless `--allow-colocated` is passed, and
the output JSON records rig, connector, force_riders, dtype, repeats, block
size, k-window and the usable row count — so no future audit has to grep logs
the way this one did.

### 2026-09-30 — M1 instrumentation landed (`9e3b8f3d3`)

**Design change I made without asking, and it is an improvement on the plan.**
The plan said "extend `FAKE_TRANSFER` to rung-d", i.e. one arm that skips the
bus copies. That arm conflates two costs which extrapolate in opposite
directions, so `VLLM_KVPIPE_RUNGD_XFER_MODE` takes three values instead:

| arm | what crosses the bus | what it is |
|---|---|---|
| `real` | everything | the design |
| `minbytes` | same `2*L*R` launches, one block row each | **an infinitely fast link** |
| `nobus` | nothing; loop and views still run | the unreachable floor |

`real - minbytes` is the byte cost, `minbytes - nobus` is the launch cost, and
`nobus - baseline` is everything else the host path costs. That is the entire
decomposition the C2C projection turns on, **measured directly rather than
fitted** — and `minbytes` is a hard ceiling for any interconnect, because no
link removes a launch. See C3 for the one assumption it makes.

The pool now also counts `stream_runs` (H2D launches issued) and
`stream_bytes_nominal` (what `real` would have moved). Launch count is the
link-invariant quantity, so it is counted rather than inferred from rider count.

**Phase timers** (`VLLM_KVPIPE_PHASE_TIMERS=1`) cover the previously
uninstrumented phases: `build_step`, `stream_layer`, `issue_runs`,
`write_token`, `prepare_layer`, the rider gather/attend/scatter, and the
runner's per-step split setup. Level 1 is host issue time only — no CUDA call,
no event, no sync — which for a launch-bound cost is the right primary metric,
since the quantity being sized *is* host time spent issuing work. Level 2 adds
device time and is Class B.

Two phases are timed separately because they are **named artifacts, not
intrinsic costs**, and a projection carrying them understates the design:
`write_token`'s two pageable (blocking) D2H copies per layer — 64
host/device serialisation points per step for trivial bytes — and `build_step`'s
Python per-block block-table build.

**Sweep axes added:** `--block-size` (separates per-block from per-byte at fixed
bytes — nothing in the current model distinguishes them), `--k-window`,
`--xfer-mode`, `--rungd-layers`. The layer knob is double-gated behind
`_UNSAFE=1`, resolved once and cached, and both measurement imports in
`flash_attn.py` are hoisted to module level — an `os.environ` read per layer
would be 64 dict lookups per step on a path whose whole subject is small
per-layer costs.

Every numerically-invalid arm prints a banner and sets `numerically_valid:
false` in the output JSON, so none can be quoted for correctness or tokens/s.

### 2026-09-30 — a worktree trap caught before it cost an allocation

A fresh `git worktree` cannot run this engine, and it fails in a way that looks
like a code bug rather than a setup one. An editable install keeps its compiled
extensions in the **source tree**, and `git worktree add` only materialises
tracked files — build outputs are gitignored. So the new tree had **zero** `.so`
files against the main tree's five, and `import vllm` would have failed outright.

Checked before submitting, not after. `paper_runs` would have shown a job that
died in the first minute with an import error.

Fixed the way `WORKTREE_SETUP.md` describes for `vllm-kvpipeline-66`: symlink the
build outputs, which are products identical for both trees (`_vllm_fa3_C` alone
is 1.1 GB, so copying them is not sensible). But rather than replicate that
tree's list, I computed the set actually missing here — **56 files, against the
22 symlinks `-66` carries**, because several `third_party/triton_kernels`
subdirectories postdate that setup. Copying the older list would have left
`matmul_ogs_details`, `numerics_details`, `tensor_details`, `topk_details` and
the `vllm_flash_attn` triton ops unresolved.

`git status` stays clean afterwards, confirming all 56 are gitignored.

**Consequence to remember:** rebuilding the C extensions in the main tree now
changes this tree too. Usually desirable; worth knowing before debugging a
mystery.

**Status:** all instrumented files compile; gate selftest 6/6; layer-knob gating
verified standalone (off / ignored-without-unsafe / active / no-op when the
limit exceeds depth). **Nothing has run on a GPU yet** — the worktree's venv
copy (8.6 GB, needed so the editable finder resolves `vllm` to this tree rather
than the main one) has been running ~70 minutes, currently inside `torch`. No
measurement can start until it finishes and the tree-identity gate passes.
