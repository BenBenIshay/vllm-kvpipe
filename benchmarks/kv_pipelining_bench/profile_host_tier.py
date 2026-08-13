#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Where does the host tier's time actually go, and what scales superlinearly?

Motivation: the `all_host` arm completes in 3.2 s at 6 requests x 512 ctx but
exceeded 900 s at 20 x 1024 -- only ~13x the work. Something is superlinear and
neither the one-time seeding cost nor the per-step transfer volume obviously
explains a >20x factor.

Method: wrap every RungdHostPool entry point with a counter+timer, run the same
engine at increasing scale, and print a per-phase breakdown plus the observed
scaling exponent. Phases:

  seed_layer    one-time resident->host copy, per layer per request
  build_step    per-step compaction + block-table construction (PYTHON loops)
  stream_layer  per-layer host->rolling H2D (the transfer we actually want)
  write_token   per-layer append of this step's K/V

A phase whose TIME grows much faster than its CALL COUNT is the culprit.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time

print = functools.partial(print, flush=True)  # noqa: A001

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

PHASES = ("seed_layer", "build_step", "stream_layer", "write_token")


def instrument(pool, stats):
    import torch

    for name in PHASES:
        orig = getattr(pool, name, None)
        if orig is None:
            continue

        def make(orig=orig, name=name):
            @functools.wraps(orig)
            def wrapper(*a, **kw):
                # CUDA events, NOT torch.cuda.synchronize().
                #
                # This used to sync inside the wrapper "so the attribution is
                # real". It did the opposite, twice over. A synchronize waits
                # for ALL outstanding GPU work, so the phase absorbed compute
                # queued before it -- stream_layer measured 24.4 ms/call when a
                # clean microbenchmark of everything it does totals 6.4 ms. And
                # a full sync after every call makes double-buffered prefetch
                # impossible BY CONSTRUCTION, so the profile "showed" that
                # VLLM_KVPIPE_RUNGD_ASYNC=1 bought nothing -- it could not have.
                #
                # A pair of events brackets only the work queued between them,
                # and needs no sync until the end of the run.
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                t0 = time.perf_counter()
                out = orig(*a, **kw)
                cpu_dt = time.perf_counter() - t0
                end.record()
                s = stats.setdefault(name, [0, 0.0, 0.0, []])
                s[0] += 1
                s[2] += cpu_dt          # CPU time in the call (gather, python)
                s[3].append((start, end))
                return out
            return wrapper
        setattr(pool, name, make())


def resolve(stats):
    """Turn the recorded event pairs into GPU milliseconds. One sync, at the
    end, so nothing above was serialized while measuring."""
    import torch
    torch.cuda.synchronize()
    for name, s in stats.items():
        if len(s) < 4:
            continue
        s[1] = sum(a.elapsed_time(b) / 1000.0 for a, b in s[3])
        s[3] = []


def run(args, n_req: int) -> dict:
    import numpy as np
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    os.environ["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
    os.environ["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
    os.environ["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
    os.environ["VLLM_KVPIPE_TIER_POLICY"] = "arrival"

    llm = LLM(model=args.model, dtype="float16", enable_prefix_caching=False,
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.ctx + args.out_len + 64,
              max_num_seqs=n_req, enforce_eager=True, disable_log_stats=True,
              enable_transfer_aware_admission=True, enable_hybrid_kv_tiers=True,
              hybrid_b_hbm_max=0,           # every request on the host tier
              hybrid_itl_ms=1e9, transfer_aware_slo_p99_itl_ms=0.0)

    rng = np.random.default_rng(7)
    vocab = llm.llm_engine.model_config.get_vocab_size()
    prompts = [TokensPrompt(prompt_token_ids=rng.integers(
        10, vocab - 10, size=args.ctx, dtype=np.int64).tolist())
        for _ in range(n_req)]
    sp = SamplingParams(temperature=0.0, max_tokens=args.out_len,
                        ignore_eos=True, min_tokens=args.out_len)

    stats: dict = {}
    # the pool is created lazily on the first hybrid step; patch after a warmup
    llm.generate(prompts[:1], SamplingParams(max_tokens=2, temperature=0.0,
                                             ignore_eos=True), use_tqdm=False)
    runner = llm.llm_engine.model_executor.driver_worker.model_runner \
        if hasattr(llm.llm_engine, "model_executor") else None
    pool = getattr(runner, "_rungd_pool", None) if runner else None
    if pool is None:
        print("  !! could not reach the pool; is RUNGD_FREE active?")
    else:
        instrument(pool, stats)

    t0 = time.perf_counter()
    llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    resolve(stats)          # single sync; converts event pairs to GPU seconds
    del llm
    return dict(n_req=n_req, wall_s=wall, stats=stats)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--out-len", type=int, default=8)
    p.add_argument("--reqs", default="4,8,16")
    p.add_argument("--gpu-mem-util", type=float, default=0.92)
    p.add_argument("--single", type=int, default=0, help="internal")
    p.add_argument("--out", default="profile_host_tier.json")
    args = p.parse_args()

    if args.single:
        r = run(args, args.single)
        with open(args.out, "w") as f:
            json.dump(r, f)
        return 0

    import subprocess
    import sys
    import tempfile
    reqs = [int(x) for x in args.reqs.split(",")]
    rows = []
    for n in reqs:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        cmd = [sys.executable, "-u", os.path.abspath(__file__),
               "--single", str(n), "--out", tmp,
               "--model", args.model, "--ctx", str(args.ctx),
               "--out-len", str(args.out_len),
               "--gpu-mem-util", str(args.gpu_mem_util)]
        print(f"[launch] n_req={n} ...")
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
        print(f"[done  ] n_req={n} in {time.time()-t0:.0f}s rc={r.returncode}")
        if r.returncode != 0:
            print("\n".join((r.stdout + r.stderr).splitlines()[-8:]))
            continue
        rows.append(json.load(open(tmp)))

    print("\n" + "=" * 78)
    print(f"{'n_req':>6} {'wall':>9} | " +
          " | ".join(f"{ph:>12}" for ph in PHASES))
    # CPU time is broken out because it is the number that matters here: the
    # host->staging gather is a CPU memcpy, and no interconnect makes it faster.
    print(f"{'':>6} {'':>9} | " +
          " | ".join(f"{'calls gpu/cpu':>12}" for _ in PHASES))
    for r in rows:
        st = r["stats"]
        cells = []
        for ph in PHASES:
            v = st.get(ph, [0, 0.0, 0.0])
            c, t, cpu = v[0], v[1], (v[2] if len(v) > 2 else 0.0)
            cells.append(f"{c:>4}/{t*1000:>4.0f}/{cpu*1000:>4.0f}")
        print(f"{r['n_req']:>6} {r['wall_s']:>8.1f}s | " +
              " | ".join(f"{c:>12}" for c in cells))

    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        ratio_n = b["n_req"] / a["n_req"]
        print(f"\nscaling from n={a['n_req']} to n={b['n_req']} "
              f"({ratio_n:.1f}x the requests):")
        print(f"  wall            {b['wall_s']/max(a['wall_s'],1e-9):>7.1f}x")
        for ph in PHASES:
            va = a["stats"].get(ph, [0, 0.0, 0.0])
            vb = b["stats"].get(ph, [0, 0.0, 0.0])
            ca, ta = va[0], va[1] + (va[2] if len(va) > 2 else 0.0)
            cb, tb = vb[0], vb[1] + (vb[2] if len(vb) > 2 else 0.0)
            if ta > 0:
                print(f"  {ph:<14} calls {cb/max(ca,1):>5.1f}x   "
                      f"time {tb/ta:>7.1f}x"
                      + ("   <<< SUPERLINEAR" if tb / ta > ratio_n * 1.6 else ""))
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out}")
    print("A phase whose TIME grows much faster than its CALL COUNT is the bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
