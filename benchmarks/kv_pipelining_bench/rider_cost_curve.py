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
    s_before, w_before = pool_bytes(llm)
    t0 = time.perf_counter()
    th.start()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    stop.set()
    th.join(timeout=1.0)
    s_after, w_after = pool_bytes(llm)
    return {"wall_s": wall, "out_len": out_len,
            "stream_bytes": s_after - s_before,
            "writeback_bytes": w_after - w_before,
            "completed": len(outs), "tiers": dict(peak)}


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
    return {
        "riders_requested": r,
        "step_ms": step_ms,
        "stream_bytes_per_step": d_stream / dtoks,
        "writeback_bytes_per_step": d_wb / dtoks,
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
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--link-gbps", type=float, default=25.1)
    p.add_argument("--rungd-async", action="store_true", default=True)
    p.add_argument("--no-rungd-async", dest="rungd_async", action="store_false")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="rider_cost_curve.json")
    p.add_argument("--outdir",
                   default="/mnt/beegfs/benishay/paper_runs/20260916/rider_curve")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    riders = [int(x) for x in a.riders.split(",") if x != ""]

    os.environ["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
    os.environ["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
    os.environ["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
    os.environ["VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM"] = "1"
    os.environ["VLLM_KVPIPE_TIER_POLICY"] = "arrival"
    if a.rungd_async:
        os.environ["VLLM_KVPIPE_RUNGD_ASYNC"] = "1"
    tok = os.path.join(ROOT, "hf_token")
    if os.path.exists(tok) and not os.environ.get("HF_TOKEN"):
        with open(tok) as fh:
            os.environ["HF_TOKEN"] = fh.read().strip()

    print("=" * 66)
    print(f" 4.2b RIDER COST CURVE   {a.model}")
    print(f" ctx {a.ctx}  batch {a.batch} (fixed)  riders {riders}")
    print(f" step time by differencing {a.short_len} vs {a.long_len} tokens "
          f"-- prefill cancels")
    print("=" * 66)

    import numpy as np
    from vllm import LLM

    rng = np.random.default_rng(a.seed)
    vocab_lo, vocab_hi = 10, 20000
    all_prompts = [{"prompt_token_ids":
                    rng.integers(vocab_lo, vocab_hi, size=a.ctx).tolist()}
                   for _ in range(a.batch + max(riders))]

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
              hybrid_pcie_gbps=a.link_gbps)
    cfg = llm.llm_engine.vllm_config.scheduler_config

    print("\nwarming up (one throwaway generation)")
    run_once(llm, all_prompts[:a.batch], a.short_len, a.seed)

    rows: list[dict] = []
    for r in riders:
        # Residents stay at capacity; the extra r requests become riders.
        cfg.hybrid_b_hbm_max = a.batch
        prompts = all_prompts[:a.batch + r]
        reps = []
        for i in range(a.repeats):
            m = measure_step(llm, prompts, a, r)
            reps.append(m)
            print(f"  r={r:2d} rep{i + 1}: step {m['step_ms']:7.2f} ms, "
                  f"stream {m['stream_bytes_per_step'] / 1e9:6.3f} GB/step, "
                  f"back {m['writeback_bytes_per_step'] / 1e9:5.3f}, "
                  f"tiers {m['tiers_long']}")
        step = statistics.median([m["step_ms"] for m in reps])
        gbs = statistics.median([m["stream_bytes_per_step"] for m in reps])
        gbw = statistics.median([m["writeback_bytes_per_step"] for m in reps])
        rows.append({"riders": r, "step_ms": step,
                     "stream_gb_per_step": gbs / 1e9,
                     "writeback_gb_per_step": gbw / 1e9,
                     "tiers": reps[-1]["tiers_long"], "reps": reps})

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
        print(f"{r:>7} {row['step_ms']:>9.2f} {dt:>8.2f} "
              f"{(dt / r if r else 0):>9.2f} {row['stream_gb_per_step']:>9.3f} "
              f"{link_ms:>9.2f} {eta:>6.2f}")

    # The admission rule this curve exists to feed: adding a rider raises
    # throughput while its step cost is smaller than the share of the step an
    # average request already occupies.
    print("\nmarginal rule:  admit while dT/step < 1/(B_resident + r)")
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

    out = {"model": a.model, "ctx": a.ctx, "batch": a.batch,
           "link_gbps": a.link_gbps, "async": a.rungd_async,
           "short_len": a.short_len, "long_len": a.long_len, "rows": rows}
    path = os.path.join(a.outdir, a.out)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
