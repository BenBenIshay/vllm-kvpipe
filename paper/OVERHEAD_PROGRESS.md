# Overhead characterisation — live progress log

Campaign: decompose `F` (the per-step cost of using the host path) into its
static / per-rider / per-byte / per-block parts, then project it onto NVLink-C2C.
Plan: `~/.claude/plans/fluttering-dreaming-kite.md`. Measurement only — no
feature optimisation.

**Read the "Open concerns" section first.** It is where anything that might need
Ben's steer is parked, newest first. Everything else is append-only history.

Branch `overhead-characterisation`, worktree `.claude/worktrees/overhead-char`,
forked from `05b32de56` (`hybrid-kv-tiers`).

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
