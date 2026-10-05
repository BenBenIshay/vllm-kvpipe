#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Experiment 4.2b: what does one rider actually cost per decode step?

WHY THIS IS THE CRITICAL PATH. Everything the admission logic decides rests on
one function -- the step-time cost of adding a rider, `dT(r)` -- and we have
never measured it. Three separate estimates in one week (0%, 23%, 38%, then 88%
hidden) came from differencing goodput runs that also contained prefill, and
they disagreed with each other because they were measuring the workload, not
the mechanism.

Worse, the arithmetic does not close. At 7 riders, streaming each rider's whole
KV every step is 3.99 GB, which at 25.1 GB/s needs 159 ms of link time -- but
the measured step was 71.9 ms in total. You cannot move 159 ms of bytes inside
a 71.9 ms step however well you overlap: the link is the constraint. So either
fewer riders stream than are admitted, or the implementation moves less per
step than the model assumes. Until that is settled, `dT` is fitted to a
mechanism nobody has verified.

So this measures BOTH, together:
  * dT(r)      -- the decode step time as a function of rider count
  * bytes(r)   -- what actually crossed the link, from the pool's own counters

and derives eta(r) from their ratio instead of assuming it.

DESIGN -- three choices that matter.

1. DECODE ONLY. This is a P/D decode-node feature, so prefill must not be in
   the measurement. A co-located run at ctx 2048 spent 58-68% of its wall clock
   on prefill, and arm C's apparent loss was almost entirely its prefill
   slowing down. Here the step time is obtained by DIFFERENCING two generation
   lengths at the same batch: `(wall_long - wall_short) / (len_long -
   len_short)`. Prefill, admission, scheduling ramp and teardown are identical
   in both and cancel exactly. Nothing needs to be excluded by hand.

2. RIDERS ARE ADDED, NOT CARVED OUT -- and the first version of this file got
   that wrong. Holding the batch at B and setting `b_hbm_max = B - r` converts
   residents INTO riders: the batch never grows, so the measurement charges
   riders their full cost while denying them the concurrency that is the whole
   point, and decode throughput fell monotonically (295 -> 112 tok/s) purely
   by construction. Here `b_hbm_max = N` stays fixed at the resident capacity
   and `N + r` prompts are submitted, so the r extra requests become riders on
   TOP of a full HBM. Cost and benefit are then both in the measurement.
   Riders are still set rather than solicited -- this experiment measures the
   input the controller should decide from, so it must not depend on it.

3. ONE ENGINE FOR THE WHOLE SWEEP. `hybrid_b_hbm_max` is read per admission, so
   it can be changed between calls; a 32B model load is ~4 minutes and the
   sweep would otherwise spend an hour reloading weights it already has.

Usage:
    python rider_cost_curve.py --model Qwen/Qwen2.5-32B-Instruct \\
        --ctx 2048 --batch 16 --riders 0,1,2,4,6,8
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
import sys
import time

print = functools.partial(print, flush=True)  # noqa: A001

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

# In-process, so the harness can read the pool's byte counters and retune
# admission between runs.
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")

from decode_purity import DecodePurityMonitor  # noqa: E402
from decode_purity import report as purity_report  # noqa: E402

# Set once the engine exists. None = gate not attached (unit-test paths only);
# a measured point must never be reported with the gate detached.
_PURITY: DecodePurityMonitor | None = None


def _git(*args) -> str:
    import subprocess
    try:
        return subprocess.run(("git", "-C", ROOT, *args), check=True,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                          # noqa: BLE001
        return ""


def _git_sha() -> str:
    return _git("rev-parse", "HEAD")[:12]


def _git_dirty() -> bool:
    """True if the tree differs from HEAD. A dirty tree means the SHA does not
    identify the code that ran, which is worse than no SHA at all -- so it is
    recorded rather than hidden."""
    return bool(_git("status", "--porcelain", "--", "benchmarks", "vllm"))


def find_pool(llm):
    try:
        core = llm.llm_engine.engine_core
        inner = getattr(core, "engine_core", core)
        runner = inner.model_executor.driver_worker.worker.model_runner
        return getattr(runner, "_rungd_pool", None)
    except Exception:                                          # noqa: BLE001
        return None


def pool_bytes(llm) -> tuple[int, int]:
    p = find_pool(llm)
    if p is None:
        return 0, 0
    return (int(getattr(p, "stream_bytes", 0) or 0),
            int(getattr(p, "writeback_bytes", 0) or 0))


def pool_counters(llm) -> dict:
    """Pool counters that matter for the overhead law.

    `stream_runs` is the H2D memcpy LAUNCH count. It is the quantity a faster
    link does not change -- 2 per run per layer, and the run count is the active
    rider count when extents are contiguous -- so it has to be counted rather
    than inferred from the rider count or the bytes. `nominal` is what the real
    path would have moved, so an arm that deliberately moves less can still be
    charged the link time the real path would have paid.
    """
    p = find_pool(llm)
    if p is None:
        return {"stream": 0, "writeback": 0, "nominal": 0, "runs": 0}
    return {
        "stream": int(getattr(p, "stream_bytes", 0) or 0),
        "writeback": int(getattr(p, "writeback_bytes", 0) or 0),
        "nominal": int(getattr(p, "stream_bytes_nominal", 0) or 0),
        "runs": int(getattr(p, "stream_runs", 0) or 0),
    }


def tier_counts(llm) -> dict:
    """How many requests the scheduler actually placed on each tier."""
    try:
        core = llm.llm_engine.engine_core
        inner = getattr(core, "engine_core", core)
        adm = inner.scheduler.transfer_aware_admission
        tiers = getattr(adm, "_req_tier", {}) or {}
    except Exception:                                          # noqa: BLE001
        return {}
    out: dict = {}
    for t in tiers.values():
        out[t] = out.get(t, 0) + 1
    return out


def run_once(llm, prompts, out_len: int, seed: int) -> dict:
    """One generation, with the tier split sampled WHILE it runs.

    Sampling after `generate` returns reads an empty table: the scheduler pops
    a request's tier when it finishes, so the first version of this reported
    `{'hbm': 1}` for a batch of sixteen and could not confirm any rider had
    been created at all.
    """
    import threading

    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=out_len,
                        min_tokens=out_len, ignore_eos=True, seed=seed)
    peak: dict = {}
    stop = threading.Event()

    def sampler():
        while not stop.wait(0.05):
            c = tier_counts(llm)
            if sum(c.values()) >= sum(peak.values()):
                peak.clear()
                peak.update(c)

    th = threading.Thread(target=sampler, daemon=True)
    c_before = pool_counters(llm)
    if _PURITY is not None:
        _PURITY.start_window()
    t0 = time.perf_counter()
    th.start()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    stop.set()
    th.join(timeout=1.0)
    c_after = pool_counters(llm)
    win = _PURITY.stop_window() if _PURITY is not None else None
    return {"wall_s": wall, "out_len": out_len,
            "stream_bytes": c_after["stream"] - c_before["stream"],
            "writeback_bytes": c_after["writeback"] - c_before["writeback"],
            "nominal_bytes": c_after["nominal"] - c_before["nominal"],
            "stream_runs": c_after["runs"] - c_before["runs"],
            "completed": len(outs), "tiers": dict(peak),
            "purity_window": win}


def measure_step(llm, prompts, a, r: int) -> dict:
    """Step time and per-step link traffic at exactly r riders.

    The difference of two generation lengths removes everything that is not a
    decode step. Both runs prefill the same prompts at the same batch size, so
    prefill, scheduling and teardown appear in both and subtract out.
    """
    short = run_once(llm, prompts, a.short_len, a.seed)
    long_ = run_once(llm, prompts, a.long_len, a.seed)
    dtoks = a.long_len - a.short_len
    step_ms = (long_["wall_s"] - short["wall_s"]) / dtoks * 1000.0
    d_stream = long_["stream_bytes"] - short["stream_bytes"]
    d_wb = long_["writeback_bytes"] - short["writeback_bytes"]
    d_nom = long_["nominal_bytes"] - short["nominal_bytes"]
    d_runs = long_["stream_runs"] - short["stream_runs"]

    # DECODE-PURITY GATE (ground rule 1). The long run carries the decode steps
    # the estimate is made of, so it must satisfy all four conditions. The short
    # run is 8 tokens, most of which is admission ramp, so its steady cohort is
    # not a meaningful quantity -- but contamination is, and a prefill or a
    # queued request there corrupts the subtraction just as badly.
    purity = None
    if long_["purity_window"] is not None:
        v_long = long_["purity_window"].verdict(a.batch, r)
        v_short = short["purity_window"].verdict(a.batch, r)

        # THE CROSS-RUN CONDITION, and it is the one that actually threatens
        # this estimator. Differencing assumes the ramp is IDENTICAL in both
        # runs so that it cancels. Admission staggers by about 2(r-1) steps, so
        # at r=6 the ramp needs ~10 steps while `--short-len 8` gives the short
        # run only 8 -- it never reaches steady state, its ramp is truncated
        # relative to the long run's, and the residue R_long - R_short lands in
        # the numerator. The step is then OVERESTIMATED by an amount that grows
        # with r, which is exactly the shape the paper reads as "not affine".
        cross = []
        if v_short.get("ramp_end") is None:
            cross.append(
                f"the SHORT run never reached steady state (ramp not cleared "
                f"in {v_short.get('n_steps')} steps at r={r}), so the ramp does "
                f"not cancel and the step is overestimated -- raise "
                f"--short-len above the ramp length (~2(r-1) steps)")
        elif v_long.get("ramp_end") is not None:
            # Both cleared it; the residue is the DIFFERENCE in ramp length.
            dr = abs(v_long["ramp_end"] - v_short["ramp_end"])
            if dr > 2:
                cross.append(
                    f"ramp length differs between the two runs by {dr} steps "
                    f"(short ends at {v_short['ramp_end']}, long at "
                    f"{v_long['ramp_end']}): that difference does not cancel")

        short_contam = [x for x in v_short["violations"]
                        if x.startswith(("prefill leaked", "requests queued IN"))]
        purity = {
            "ok": v_long["ok"] and not short_contam and not cross,
            "long": v_long,
            "short": {k: v_short.get(k) for k in
                      ("n_steps", "n_steady", "ramp_end", "max_waiting",
                       "max_waiting_steady", "prefill_tokens")},
            "short_contamination": short_contam,
            "cross_run": cross,
        }

    return {
        "riders_requested": r,
        "step_ms": step_ms,
        "purity": purity,
        "stream_bytes_per_step": d_stream / dtoks,
        "writeback_bytes_per_step": d_wb / dtoks,
        "nominal_bytes_per_step": d_nom / dtoks,
        "h2d_launches_per_step": d_runs / dtoks,
        "tiers_short": short["tiers"], "tiers_long": long_["tiers"],
        "wall_short_s": short["wall_s"], "wall_long_s": long_["wall_s"],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--batch", type=int, default=16,
                   help="RESIDENT requests (HBM capacity). Riders are added on "
                        "top, so the batch is N + r.")
    p.add_argument("--riders", default="0,1,2,4,6,8")
    p.add_argument("--short-len", type=int, default=8)
    p.add_argument("--long-len", type=int, default=72)
    p.add_argument("--repeats", type=int, default=3,
                   help="3 minimum: with 2 the median is the mean, so a single "
                        "bad differencing sample corrupts the point")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--link-gbps", type=float, default=0.0,
                   help="effective host->device GB/s. 0 = MEASURE it here. "
                        "Never leave the A100's 25.1 on a C2C machine: every "
                        "eta and every projection scales with it.")
    p.add_argument("--force-riders", action="store_true", default=True,
                   help="run exactly the requested rider count by disabling "
                        "B1's gate for the measurement (default on)")
    p.add_argument("--no-force-riders", dest="force_riders",
                   action="store_false")
    p.add_argument("--connector", action="store_true",
                   help="DECODE-NODE emulation: attach "
                        "PipelinedDecodeBenchConnector, whose "
                        "get_num_new_matched_tokens reports all but one prompt "
                        "token as already computed. vLLM then SKIPS the prefill "
                        "forward pass and the connector fills those KV blocks "
                        "directly -- which is what a P/D decode node sees when "
                        "KV arrives from the prefill node. Without it the node "
                        "prefills locally, and a rider cannot be created while "
                        "HBM is full because its prefill needs resident blocks "
                        "(measured: riders cap at 2 however many are offered).")
    # ---- sweep axes the overhead law needs, which this harness did not expose
    p.add_argument("--block-size", type=int, default=0,
                   help="KV block size (0 = engine default 16). Swept at FIXED "
                        "ctx and rider count to separate a per-BLOCK cost from "
                        "a per-BYTE one: halving the block size doubles the "
                        "block count while the bytes stay put. A per-block cost "
                        "is host work a faster link does not touch; a per-byte "
                        "cost shrinks with bandwidth. Nothing else in the model "
                        "distinguishes them.")
    p.add_argument("--k-window", type=int, default=0,
                   help="rolling slots (0 = engine default 4). The residual-cost "
                        "analysis names batching ACROSS layers as the way past "
                        "the per-layer issue structure, and these slots are what "
                        "it would use -- so the overhead's sensitivity to it is "
                        "worth measuring rather than assuming.")
    p.add_argument("--xfer-mode", default="real",
                   choices=("real", "minbytes", "nobus"),
                   help="VLLM_KVPIPE_RUNGD_XFER_MODE. 'minbytes' keeps every "
                        "H2D launch but moves one block row per copy, which is "
                        "an infinitely fast link and therefore the CEILING for "
                        "any interconnect; 'nobus' issues no bus copy at all, "
                        "the unreachable floor. real-minbytes is the byte cost, "
                        "minbytes-nobus the launch cost. Non-real modes are "
                        "NUMERICALLY INVALID -- timing only.")
    p.add_argument("--skip-d2h", action="store_true",
                   help="skip write_token's host-mirror write, so the blocking "
                        "pageable D2H is priced by DIFFERENCING against a real "
                        "run rather than by timing the host around it. Timing "
                        "the host does not work here: it reads 21.4 ms per step "
                        "for two copies of one token's KV, which is the host "
                        "blocked on the GPU, not a cost. NUMERICALLY INVALID.")
    p.add_argument("--batch-gather", action="store_true",
                   help="enable T3, the batched strided per-layer gather. OFF "
                        "by default because it is a 6x REGRESSION: it is "
                        "correct and does cut launches from 2*L*r to 2*L, but "
                        "torch materialises the strided pinned view on the HOST "
                        "first, so the transfer runs at 2.3-4.3 GB/s instead of "
                        "the link's 25.1. Kept only for re-measuring that.")
    p.add_argument("--no-d2h-async", action="store_true",
                   help="restore the OLD blocking mirror write (T2 off). The "
                        "async deferred write is the default; this is what the "
                        "before/after comparison needs. Numerically VALID "
                        "either way -- both paths update the mirror.")
    p.add_argument("--d2h-probe", action="store_true",
                   help="PHASE 0: make write_token's mirror copy NON-BLOCKING "
                        "into pinned staging and skip the scatter. Same launch "
                        "count and bytes as --real, so real-minus-probe is the "
                        "synchronisation cost on its own -- the prize a correct "
                        "async fix could actually realise, as distinct from "
                        "--skip-d2h which removes the scatter too and so only "
                        "bounds it from above. NUMERICALLY INVALID.")
    p.add_argument("--rungd-layers", type=int, default=0,
                   help="restrict the rider host path to the first N layers so "
                        "the per-layer slope can be measured inside one model "
                        "at fixed head geometry (0 = all). NUMERICALLY INVALID "
                        "when set; implies the _UNSAFE acknowledgement.")
    p.add_argument("--allow-colocated", action="store_true",
                   help="permit running WITHOUT --connector. Required, because "
                        "co-located is not the setup this feature targets: the "
                        "node prefills locally, a rider cannot be created while "
                        "HBM is full, and riders cap at 2 however many are "
                        "offered. The only legitimate use is a deliberate A/B "
                        "against the decode-node rig.")
    p.add_argument("--rungd-async", action="store_true", default=True)
    p.add_argument("--no-rungd-async", dest="rungd_async", action="store_false")
    p.add_argument("--whole-extent", action="store_true",
                   help="T5: build the per-step runs from whole EXTENTS so "
                        "they merge into one contiguous copy per layer, 2 "
                        "launches instead of 2r. Needs --rungd-bpr to be set "
                        "tight or the pool's slack guard refuses it, because "
                        "the default extent is sized for max_model_len and "
                        "whole-extent copying would then move 4x the bytes.")
    p.add_argument("--rungd-bpr", type=int, default=0,
                   help="blocks_per_req override. Set to "
                        "ceil((ctx + long_len) / block_size) for a tight "
                        "extent. A request needing more RAISES, which is "
                        "deliberate -- it must not silently truncate.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="rider_cost_curve.json")
    p.add_argument("--outdir",
                   default="/mnt/beegfs/benishay/paper_runs/20260916/rider_curve")
    a = p.parse_args()

    # GROUND RULE 1: P/D-emulated, feature on the D node. Enforced here rather
    # than trusted, because --connector is opt-in and the archived dataset
    # behind the paper's F anchor was run without it.
    if not a.connector and not a.allow_colocated:
        p.error("refusing to run co-located: pass --connector for the "
                "decode-node rig, or --allow-colocated if you really mean to "
                "measure the co-located case (A/B only).")
    if not a.connector:
        print("!" * 66)
        print("! CO-LOCATED RIG -- local prefill, riders cap at 2. This is NOT")
        print("! the setup the feature targets. A/B use only.")
        print("!" * 66)

    os.makedirs(a.outdir, exist_ok=True)

    riders = [int(x) for x in a.riders.split(",") if x != ""]

    os.environ["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
    os.environ["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
    os.environ["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
    os.environ["VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM"] = "1"
    os.environ["VLLM_KVPIPE_TIER_POLICY"] = "arrival"
    if a.force_riders:
        # This experiment measures the COST at a chosen rider count; B1 deciding
        # a different one would make the curve a picture of the controller
        # rather than of the mechanism. `hide_budget_ms()` prefers the MEASURED
        # resident-only step, so raising --hybrid-itl-ms alone does not lift the
        # gate -- the measured-budget path has to be switched off too. Without
        # both, A100 pinned every point at 2 riders.
        os.environ["VLLM_KVPIPE_B1_MEASURED"] = "0"
        os.environ.pop("VLLM_KVPIPE_ADMIT_MARGINAL", None)
    if a.rungd_async:
        os.environ["VLLM_KVPIPE_RUNGD_ASYNC"] = "1"
    os.environ["VLLM_KVPIPE_RUNGD_XFER_MODE"] = a.xfer_mode
    if a.skip_d2h:
        os.environ["VLLM_KVPIPE_RUNGD_SKIP_D2H"] = "1"
    if a.batch_gather:
        os.environ["VLLM_KVPIPE_RUNGD_BATCH_GATHER"] = "1"
    if a.whole_extent:
        os.environ["VLLM_KVPIPE_RUNGD_WHOLE_EXTENT"] = "1"
    if a.rungd_bpr > 0:
        os.environ["VLLM_KVPIPE_RUNGD_BPR"] = str(a.rungd_bpr)
    if a.no_d2h_async:
        os.environ["VLLM_KVPIPE_RUNGD_D2H_ASYNC"] = "0"
    if a.d2h_probe:
        os.environ["VLLM_KVPIPE_RUNGD_D2H_PROBE"] = "1"
        os.environ["VLLM_KVPIPE_RUNGD_D2H_PROBE_UNSAFE"] = "1"
    if a.rungd_layers > 0:
        os.environ["VLLM_KVPIPE_RUNGD_LAYERS"] = str(a.rungd_layers)
        os.environ["VLLM_KVPIPE_RUNGD_LAYERS_UNSAFE"] = "1"
    if (a.xfer_mode != "real" or a.rungd_layers > 0 or a.skip_d2h
            or a.d2h_probe):
        print("!" * 66)
        print("! NUMERICALLY INVALID ARM: "
              f"xfer_mode={a.xfer_mode} layers={a.rungd_layers or 'all'}")
        print("! Generated tokens are meaningless. TIMING ONLY -- this arm must")
        print("! never be cited for correctness, quality or tokens/s.")
        print("!" * 66)
    tok = os.path.join(ROOT, "hf_token")
    if os.path.exists(tok) and not os.environ.get("HF_TOKEN"):
        with open(tok) as fh:
            os.environ["HF_TOKEN"] = fh.read().strip()

    if a.link_gbps <= 0:
        from gh200_preflight import measure_h2d_gbps
        a.link_gbps = measure_h2d_gbps()
        print(f"measured pinned H2D bandwidth: {a.link_gbps:.1f} GB/s")

    print("=" * 66)
    print(f" 4.2b RIDER COST CURVE   {a.model}")
    print(f" ctx {a.ctx}  batch {a.batch} (fixed)  riders {riders}")
    print(f" step time by differencing {a.short_len} vs {a.long_len} tokens "
          f"-- prefill cancels")
    print(f" rig: {'DECODE NODE (connector, prefill skipped)' if a.connector else 'co-located (local prefill)'}")
    print("=" * 66)

    import numpy as np
    from vllm import LLM

    rng = np.random.default_rng(a.seed)
    vocab_lo, vocab_hi = 10, 20000
    all_prompts = [{"prompt_token_ids":
                    rng.integers(vocab_lo, vocab_hi, size=a.ctx).tolist()}
                   for _ in range(a.batch + max(riders))]

    extra: dict = {}
    if a.connector:
        from vllm.config import KVTransferConfig
        extra["kv_transfer_config"] = KVTransferConfig(
            kv_connector="PipelinedDecodeBenchConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"fill_mean": 0.015, "fill_std": 0.0})

    if a.block_size:
        extra["block_size"] = a.block_size
    if a.k_window:
        extra["kv_k_window"] = a.k_window

    llm = LLM(model=a.model, dtype=a.dtype,
              enable_prefix_caching=False,
              gpu_memory_utilization=a.gpu_mem_util,
              max_model_len=a.ctx + a.long_len + 64,
              max_num_seqs=a.batch + max(riders),
              enforce_eager=True, disable_log_stats=False,
              enable_transfer_aware_admission=True,
              enable_hybrid_kv_tiers=True,
              hybrid_b_hbm_max=a.batch,
              hybrid_itl_ms=1e9,            # the SWEEP sets riders, not B1
              transfer_aware_slo_p99_itl_ms=0.0,
              hybrid_pcie_gbps=a.link_gbps, **extra)
    cfg = llm.llm_engine.vllm_config.scheduler_config

    # Attach the decode-purity gate before any measurement. Class A: per-step
    # counting only, no CUDA calls, so it is on for every run rather than being
    # a flag someone forgets for the run that needed it.
    global _PURITY
    _PURITY = DecodePurityMonitor(llm)
    _PURITY.attach()
    print("decode-purity gate attached (per-step, non-perturbing)")

    print("\nwarming up (one throwaway generation)")
    run_once(llm, all_prompts[:a.batch], a.short_len, a.seed)

    rows: list[dict] = []
    for r in riders:
        # Residents stay at capacity; the extra r requests become riders.
        cfg.hybrid_b_hbm_max = a.batch
        prompts = all_prompts[:a.batch + r]

        # THROWAWAY AT EACH RIDER COUNT. The engine is warmed once at startup,
        # not after each change of rider count, and the first measurement at a
        # new count absorbs one-off work -- host-pool extents and pinned buffers
        # for the new riders. Measured at ctx 512: rep1 at r=1 came out 38.70 ms
        # against 61.60-61.78 for reps 2-5, i.e. BELOW the 49.29 ms rider-free
        # baseline. The below-baseline guard happened to catch that one because
        # it was too FAST; a too-slow first rep would pass straight through, so
        # relying on the guard is not enough.
        measure_step(llm, prompts, a, r)

        reps = []
        for i in range(a.repeats):
            m = measure_step(llm, prompts, a, r)
            reps.append(m)
            print(f"  r={r:2d} rep{i + 1}: step {m['step_ms']:7.2f} ms, "
                  f"stream {m['stream_bytes_per_step'] / 1e9:6.3f} GB/step, "
                  f"back {m['writeback_bytes_per_step'] / 1e9:5.3f}, "
                  f"tiers {m['tiers_long']}")
            if m["purity"] is not None:
                purity_report(m["purity"]["long"], f"r={r} rep{i + 1}")
                for x in m["purity"]["short_contamination"]:
                    print(f"  [r={r} rep{i + 1}] short-run contamination: {x}")
                for x in m["purity"]["cross_run"]:
                    print(f"  [r={r} rep{i + 1}] CROSS-RUN: {x}")
        # ROBUST ESTIMATOR, and it needs to be. Differencing two wall times
        # occasionally yields an implausible step -- a slow short run makes the
        # difference too small (measured: 24.73 ms against a 53.4 ms baseline,
        # i.e. riders apparently making the step FASTER). With two repeats the
        # median is just the mean, so one such sample dragged the r=2 point from
        # 71 ms to 48 ms and the fit returned F = -56 ms. Discard samples below
        # the rider-free baseline -- adding work cannot shorten a step -- then
        # take the median of what remains.
        # Purity first, robustness second. A rep that failed the gate is not a
        # noisy measurement of the right thing, it is a measurement of a
        # different thing, so it is dropped before the estimator sees it.
        pure = [m for m in reps
                if m["purity"] is None or m["purity"]["ok"]]
        n_impure = len(reps) - len(pure)
        if n_impure:
            print(f"      DISCARDED {n_impure}/{len(reps)} rep(s) on the "
                  f"decode-purity gate")
        usable = bool(pure)
        est = pure if pure else reps

        cand = [m["step_ms"] for m in est]
        if rows:
            floor = rows[0]["step_ms"] * 0.95
            kept = [x for x in cand if x >= floor]
            if len(kept) < len(cand):
                print(f"      discarded {len(cand) - len(kept)} implausible "
                      f"sample(s) below the {floor:.1f} ms baseline")
            cand = kept or cand
        step = statistics.median(cand)
        # SECOND, INDEPENDENT ESTIMATOR: the median inter-step interval over the
        # steady window, from the purity monitor's own per-step timestamps. It
        # needs no short run, so ramp, drain, rider seeding and prefill are
        # excluded by construction instead of by an argument that they cancel --
        # and that argument has a hole, since riders are seeded when ADMITTED
        # and a short run can end before the last one is. Agreement between the
        # two estimators is the evidence; a divergence is itself a finding.
        direct = [(m["purity"]["long"] or {}).get("step_ms_direct")
                  for m in est if m.get("purity")]
        direct = [x for x in direct if x is not None]
        step_direct = statistics.median(direct) if direct else None

        gbs = statistics.median([m["stream_bytes_per_step"] for m in est])
        gbw = statistics.median([m["writeback_bytes_per_step"] for m in est])
        rows.append({"riders": r, "step_ms": step,
                     "step_ms_direct": step_direct,
                     "step_ms_delta_pct": (
                         (step_direct - step) / step * 100.0
                         if step_direct and step else None),
                     "stream_gb_per_step": gbs / 1e9,
                     "writeback_gb_per_step": gbw / 1e9,
                     "nominal_gb_per_step": statistics.median(
                         [m["nominal_bytes_per_step"] for m in est]) / 1e9,
                     "h2d_launches_per_step": statistics.median(
                         [m["h2d_launches_per_step"] for m in est]),
                     "tiers": est[-1]["tiers_long"],
                     # `usable` is what downstream fitting must filter on. A row
                     # with usable=False is reported so the failure is visible,
                     # but it is NOT a measurement of the rider cost at this r.
                     "usable": usable,
                     "n_reps": len(reps), "n_impure": n_impure,
                     "reps": reps})

    base = rows[0]["step_ms"] if rows else float("nan")
    print("\n" + "=" * 66)
    print(f"{'riders':>7} {'step ms':>9} {'dT ms':>8} {'dT/rider':>9} "
          f"{'GB/step':>9} {'link ms':>9} {'eta':>6}")
    for row in rows:
        r = row["riders"]
        dt = row["step_ms"] - base
        link_ms = row["stream_gb_per_step"] * 1e9 / (a.link_gbps * 1e9) * 1000
        eta = (1 - dt / link_ms) if link_ms > 0 else float("nan")
        row.update(dT_ms=dt, dT_per_rider_ms=(dt / r if r else 0.0),
                   link_ms=link_ms, eta=eta)
        mark = "" if row.get("usable", True) else "  <-- IMPURE, not a rider cost"
        sd = row.get("step_ms_direct")
        dpct = row.get("step_ms_delta_pct")
        two = (f" | direct {sd:7.2f} ({dpct:+.1f}%)"
               if sd is not None and dpct is not None else "")
        print(f"{r:>7} {row['step_ms']:>9.2f} {dt:>8.2f} "
              f"{(dt / r if r else 0):>9.2f} {row['stream_gb_per_step']:>9.3f} "
              f"{link_ms:>9.2f} {eta:>6.2f}{two}{mark}")

    # ------------------------------------------------------------------ fit
    # THE DELIVERABLE, and it must be on screen in hour one rather than fitted
    # afterwards. dT(r) = F + c*r, and the two terms behave completely
    # differently:
    #   F  a FIXED per-step cost charged the moment ANY rider exists -- per-layer
    #      hooks, the pool attention override, the two-pass path. It is
    #      uncorrelated with bytes moved (0.5B: 7.4 ms against 0.27 ms of
    #      transfer per rider) and a faster link does NOT shrink it.
    #   c  the per-rider cost: the unhidden part of that rider's transfer, which
    #      DOES scale with the link.
    # Their ratio decides everything, because F has to be amortised:
    #   gain(r) = (1 + r/N) / (1 + (F + c*r)/T) > 1   iff   r > F/(T/N - c)
    # So there is a MINIMUM rider count below which the feature cannot pay, and
    # a controller admitting one rider at a time pays F alone on the first and
    # never climbs out of it.
    # ------------------------------------------------------------ cost model
    # MEASURED SHAPE, and it is not linear -- dT/rider RISES (9.1 -> 11.0 ->
    # 15.6 ms at r = 2, 4, 6), so fitting dT = F + c*r returns a negative fixed
    # cost and means nothing. What the data fits is a KINK:
    #
    #     step  ~=  max(T_compute, r * S * b_tok / X)  +  overhead
    #
    # Below the knee the riders' aggregate transfer still fits inside a step and
    # is (imperfectly) hidden; above it the step IS the transfer and every extra
    # rider costs its full link time. The knee is
    #
    #     r_knee = T_compute * X / (S * b_tok)
    #
    # which is exactly B1's byte budget -- so B1 identifies the right boundary,
    # and the thing it misses is that even BELOW the knee the overhead is real.
    # A100 measured: knee 2.4 riders, overhead 6.6-18.1 ms.
    print("\n" + "=" * 66)
    print(" COST MODEL   step ~= max(T_compute, T_transfer) + overhead")
    out_fit: dict = {}
    per_rider_gb = next((row["stream_gb_per_step"] / row["riders"]
                         for row in rows if row["riders"] > 0
                         and row["stream_gb_per_step"] > 0), 0.0)
    if per_rider_gb <= 0:
        print("  no rider traffic recorded -- cannot model. Check --connector "
              "and that GB/step > 0.")
    else:
        per_rider_ms = per_rider_gb * 1e9 / (a.link_gbps * 1e9) * 1000.0
        knee = base / per_rider_ms
        print(f"  per-rider transfer      = {per_rider_ms:7.2f} ms "
              f"({per_rider_gb:.3f} GB at {a.link_gbps:.1f} GB/s)")
        print(f"  compute step            = {base:7.2f} ms")
        print(f"  KNEE r = T/per-rider    = {knee:7.2f} riders "
              f"(beyond this the step IS the transfer)")
        oh = []
        for row in rows:
            r = row["riders"]
            if not r:
                continue
            bound = max(base, r * per_rider_ms)
            o = row["step_ms"] - bound
            oh.append(o)
            print(f"    r={r:2d}: max(compute, transfer) = {bound:7.2f} ms, "
                  f"measured {row['step_ms']:7.2f} -> overhead {o:6.2f} ms")
        ovh = statistics.median(oh) if oh else float("nan")
        out_fit = {"per_rider_ms": per_rider_ms, "knee_riders": knee,
                   "overhead_ms": ovh, "compute_step_ms": base}
        print(f"  overhead (median)       = {ovh:7.2f} ms")
        # NEVER PROJECT PAST THE DATA. The knee can sit far beyond the sweep
        # (0.5B: knee 31.6 riders from points measured only to r=4), and the
        # overhead is not guaranteed to stay flat out there -- extrapolating
        # gave "2.06x at 31 riders" from two measurements. Cap the operating
        # point at the largest rider count actually measured and say so.
        r_measured_max = max(row["riders"] for row in rows)
        r_best = max(1, min(int(knee), r_measured_max))
        if knee > r_measured_max:
            print(f"\n  NOTE: the knee ({knee:.1f} riders) lies BEYOND the "
                  f"sweep (max {r_measured_max}). The best operating point is "
                  f"not yet measured -- re-run with --riders up to "
                  f"{int(knee) + 2} before quoting a gain.")
        gain = ((1 + r_best / a.batch)
                / (1 + (max(0.0, r_best * per_rider_ms - base) + ovh) / base))
        out_fit["r_best"] = r_best
        out_fit["gain_at_r_best"] = gain
        print(f"\n  best operating point: r = {r_best} riders -> "
              f"{gain:.2f}x decode throughput")
        if gain > 1.0:
            print(f"  VERDICT: the feature PAYS here, best at ~{r_best} riders.")
        else:
            print(f"  VERDICT: it does not pay on this substrate -- at the knee "
                  f"({knee:.1f} riders) the overhead alone ({ovh:.1f} ms on a "
                  f"{base:.1f} ms step) exceeds what {r_best} extra requests "
                  f"buy. A faster link raises the knee; it does not remove the "
                  f"overhead.")

    print("\nmarginal rule (per measured point):")
    for row in rows:
        r = row["riders"]
        if not r:
            continue
        share = row["dT_per_rider_ms"] / row["step_ms"]
        thresh = 1.0 / (a.batch + r)
        tput = (a.batch + r) / row["step_ms"]
        base_tput = a.batch / base
        print(f"  r={r:2d}: dT/step {share:.4f} vs 1/B {thresh:.4f} -> "
              f"{'admit' if share < thresh else 'STOP'}   "
              f"decode tok/s {tput * 1000:6.1f} vs {base_tput * 1000:6.1f} "
              f"({tput / base_tput:.3f}x)")

    # PROVENANCE. The eight curves archived in paper_runs/20260916 record
    # neither the rig nor the forcing, so which of them was the decode-node
    # emulation had to be recovered afterwards by grepping the run logs for
    # "PipelinedDecodeBenchConnector" -- and it turned out the two datasets the
    # paper quotes disagree on it. Never again: the JSON says what it was.
    out = {"fit": out_fit, "model": a.model, "ctx": a.ctx, "batch": a.batch,
           "link_gbps": a.link_gbps, "async": a.rungd_async,
           "whole_extent": a.whole_extent, "rungd_bpr": a.rungd_bpr,
           "short_len": a.short_len, "long_len": a.long_len,
           "connector": bool(a.connector),
           "rig": ("decode_node_pd_emulated" if a.connector
                   else "co_located_local_prefill"),
           "force_riders": bool(a.force_riders),
           "dtype": a.dtype, "repeats": a.repeats,
           "gpu_mem_util": a.gpu_mem_util,
           "block_size": a.block_size or 16,
           "k_window": a.k_window or 4,
           "xfer_mode": a.xfer_mode,
           "skip_d2h": bool(a.skip_d2h),
           "d2h_probe": bool(a.d2h_probe),
           "d2h_async": not bool(a.no_d2h_async),
           "batch_gather": bool(a.batch_gather),
           "rungd_layers": a.rungd_layers or 0,
           "numerically_valid": (a.xfer_mode == "real"
                                 and a.rungd_layers == 0
                                 and not a.skip_d2h
                                 and not a.d2h_probe),
           "purity_gate": "per-step, decode_purity.DecodePurityMonitor",
           # CODE PROVENANCE. Each arm of a sweep is a fresh process, so editing
           # the harness while a job runs means later arms pick up the change and
           # the job is no longer internally consistent. That happened on
           # 2026-09-30: t0_c512 ran without the per-rider warmup and without a
           # working direct estimator, t0_c256 ran with both, and which was which
           # had to be inferred. Now every artifact states the code that made it.
           "git_commit": _git_sha(),
           "git_dirty": _git_dirty(),
           # JOB IDENTITY. F carries a between-job offset of 1.78 ms (14%),
           # larger than its within-job spread of 0.58 ms (5%) -- one job read
           # 1.2 ms high on every arm it contained. So only within-job contrasts
           # are trustworthy, and a pooled fit must model the job as a variance
           # component rather than launder it into the coefficients. Recorded
           # here so that never has to be reconstructed from log filenames
           # again, which is how it had to be done for the runs before this.
           "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
           "hostname": os.environ.get("SLURMD_NODENAME",
                                      os.uname().nodename),
           "n_rows_usable": sum(1 for x in rows if x.get("usable")),
           "n_rows": len(rows),
           "rows": rows}
    path = os.path.join(a.outdir, a.out)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
