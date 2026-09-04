#!/usr/bin/env python3
"""What are those 64 layers actually DOING when a rider exists?

The fixed cost is now well characterised and still unexplained: ~37 ms of ITL
appears the moment the first rider is admitted, and stays flat whether there
are 2 or 8 residents and whether there are 1 or 3 riders. Over 64 layers that
is ~0.58 ms a layer of work that is sensitive to the rider path EXISTING and
insensitive to how much data flows through it.

Three hypotheses are already dead -- the batch-wide gather/scatter (flat in
resident count), `build_step` (~1 ms), and `write_token`'s blocking D2H (~4 ms
at realistic queue depth, not the 21 ms an aggregate timer suggested). Two of
those three were killed only after an aggregate timer pointed the wrong way, so
this stops inferring from totals and asks the profiler which operators run.

Method: same engine, same prompts, decode steps traced with torch.profiler,
once with NO rider and once with ONE. Diff the operator tables. An operator
that appears only in the rider run, or grows by a few ms, is the answer; if the
extra time is spread thinly over hundreds of tiny launches, that is the
launch-bound story and the fix is to do less per layer.

Run: srun ... .venv/bin/python \
       benchmarks/kv_pipelining_bench/rider_step_trace.py
"""

from __future__ import annotations

import argparse
import functools
import json
import os

print = functools.partial(print, flush=True)  # noqa: A001

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

LOGDIR = "/mnt/beegfs/benishay/paper_runs/20260904"


def run(args, riders: int) -> dict:
    import numpy as np
    import torch
    from torch.profiler import ProfilerActivity, profile
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    os.environ["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
    os.environ["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
    os.environ["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
    os.environ["VLLM_KVPIPE_TIER_POLICY"] = "arrival"
    # b_hbm_max = n_req puts everything on HBM (no rider); n_req-1 leaves
    # exactly one request on the host tier.
    n_req = args.residents + (1 if riders else 0)
    hbm_max = n_req if riders == 0 else args.residents

    llm = LLM(model=args.model, dtype="float16", enable_prefix_caching=False,
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.ctx + args.out_len + 64,
              max_num_seqs=n_req, enforce_eager=True, disable_log_stats=True,
              enable_transfer_aware_admission=True, enable_hybrid_kv_tiers=True,
              hybrid_b_hbm_max=hbm_max,
              hybrid_itl_ms=1e9, transfer_aware_slo_p99_itl_ms=0.0)

    rng = np.random.default_rng(7)
    vocab = llm.llm_engine.model_config.get_vocab_size()
    prompts = [TokensPrompt(prompt_token_ids=rng.integers(
        10, vocab - 10, size=args.ctx, dtype=np.int64).tolist())
        for _ in range(n_req)]

    # Warm up so the pool exists and prefill is behind us; only steady-state
    # decode should land in the trace.
    llm.generate(prompts, SamplingParams(max_tokens=8, temperature=0.0,
                                         ignore_eos=True, min_tokens=8),
                 use_tqdm=False)

    sp = SamplingParams(temperature=0.0, max_tokens=args.steps,
                        ignore_eos=True, min_tokens=args.steps)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        llm.generate(prompts, sp, use_tqdm=False)
    torch.cuda.synchronize()

    rows = []
    for e in prof.key_averages():
        rows.append({
            "name": e.key,
            "count": int(e.count),
            "cuda_ms": float(getattr(e, "self_device_time_total",
                                     getattr(e, "self_cuda_time_total", 0.0))) / 1000.0,
            "cpu_ms": float(e.self_cpu_time_total) / 1000.0,
        })
    del llm
    return {"riders": riders, "n_req": n_req, "steps": args.steps, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--residents", type=int, default=4)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--out-len", type=int, default=64)
    ap.add_argument("--gpu-mem-util", type=float, default=0.83)
    ap.add_argument("--single", type=int, default=-1, help="internal")
    ap.add_argument("--out", default=f"{LOGDIR}/rider_step_trace.json")
    args = ap.parse_args()

    if args.single >= 0:
        with open(args.out, "w") as f:
            json.dump(run(args, args.single), f)
        return 0

    import subprocess
    import sys
    results = {}
    for riders in (0, 1):
        tmp = f"{LOGDIR}/trace_r{riders}.json"
        cmd = [sys.executable, "-u", os.path.abspath(__file__),
               "--single", str(riders), "--out", tmp, "--model", args.model,
               "--ctx", str(args.ctx), "--residents", str(args.residents),
               "--steps", str(args.steps),
               "--gpu-mem-util", str(args.gpu_mem_util)]
        print(f"[launch] riders={riders}")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(f"[done  ] riders={riders} rc={r.returncode}")
        if r.returncode != 0:
            print("\n".join((r.stdout + r.stderr).splitlines()[-12:]))
            return 1
        results[riders] = json.load(open(tmp))

    steps = args.steps
    a = {r["name"]: r for r in results[0]["rows"]}
    b = {r["name"]: r for r in results[1]["rows"]}
    diffs = []
    for name, rb in b.items():
        ra = a.get(name, {"cuda_ms": 0.0, "cpu_ms": 0.0, "count": 0})
        d_cuda = (rb["cuda_ms"] - ra["cuda_ms"]) / steps
        d_cpu = (rb["cpu_ms"] - ra["cpu_ms"]) / steps
        diffs.append((d_cuda + d_cpu, d_cuda, d_cpu, name,
                      rb["count"] / steps, ra["count"] / steps))
    diffs.sort(reverse=True)

    with open(args.out, "w") as f:
        json.dump({"no_rider": results[0], "one_rider": results[1]}, f)

    print("=" * 96)
    print(f"per-STEP delta introduced by ONE rider ({args.residents} residents, "
          f"{steps} steps traced)")
    print(f"{'total ms':>9} {'cuda ms':>8} {'cpu ms':>8} {'calls/step':>11} "
          f"{'was':>7}  operator")
    for tot, dc, dp, name, nb, na in diffs[:25]:
        if abs(tot) < 0.05:
            continue
        print(f"{tot:>9.2f} {dc:>8.2f} {dp:>8.2f} {nb:>11.1f} {na:>7.1f}  "
              f"{name[:58]}")
    print("=" * 96)
    tot_all = sum(d[0] for d in diffs)
    top10 = sum(d[0] for d in diffs[:10])
    launches = sum(d[4] - d[5] for d in diffs if d[0] > 0)
    print(f"total added per step: {tot_all:.1f} ms across "
          f"{launches:.0f} extra operator calls/step")
    if diffs and diffs[0][0] > 0.35 * tot_all:
        print(f"CONCENTRATED: '{diffs[0][3][:50]}' alone is "
              f"{diffs[0][0] / tot_all * 100:.0f}% of it -- fix that one thing.")
    elif top10 < 0.5 * tot_all:
        print("DIFFUSE: no operator dominates and the top 10 are under half the "
              "total, so this is launch-bound overhead spread over many tiny "
              "calls. The fix is to do FEWER things per layer, not to make any "
              "one of them faster.")
    else:
        print("Moderately concentrated: the top few operators carry most of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
