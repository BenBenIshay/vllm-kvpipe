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

### C1 — OPEN. Needs a decision eventually; not blocking M0.

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

**Status:** all instrumented files compile; gate selftest 6/6; layer-knob gating
verified standalone (off / ignored-without-unsafe / active / no-op when the
limit exceeds depth). **Nothing has run on a GPU yet** — the worktree's venv
copy (8.6 GB, needed so the editable finder resolves `vllm` to this tree rather
than the main one) has been running ~70 minutes, currently inside `torch`. No
measurement can start until it finishes and the tree-identity gate passes.
