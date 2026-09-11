#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Is the POLICY load-bearing, or would uniform offloading do just as well?

This is the experiment that decides whether we have a *policy* paper. The
endpoint comparison (endpoint_comparison.py) shows the mixture beats both
endpoints; this shows that HOW you pick the mixture matters. If uniform
offloading at the same offload ratio does as well as our tiered admission, the
policy layer is decoration and the contribution collapses to the analysis.

Four arms, all on the SAME substrate and the SAME offered load:

  uniform_25 / uniform_50   fixed fraction of requests on the host tier, chosen
                            by arrival order -- "DirectKV with a dial", i.e.
                            what you would build without a policy
  arrival                   our B1/B2 controller decides HOW MANY, but WHICH is
                            still arrival order (the pre-Item-2 baseline)
  scored                    + request-level tier assignment (Item 2)
  scored_migrate            + adaptive migration with the probe (Item 3)

Each arm adds exactly one mechanism, so the deltas are attributable.

It also records the ADAPTIVE PROBE VERDICT, which is a prediction we can be
wrong about in public: on A100/PCIe the probe measured demotion as HARMFUL and
correctly reverted. On C2C it should measure BENEFICIAL and keep the demotion.
If it still says harmful on a fast link, our host path has a fixed per-step
overhead that survives the interconnect -- important either way.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time

# A multi-hour phase that block-buffers stdout is indistinguishable
# from a hang. Make every print line-flushed.
print = functools.partial(print, flush=True)  # noqa: A001

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

ARMS = ("uniform_25", "uniform_50", "arrival", "scored", "scored_migrate",
        "scored_swapback")

# Arms that emulate SWAP semantics: the whole rolling slot is written back to
# the host mirror after each layer, as a design whose offloaded unit is OWNED
# by one side must do. See RungdHostPool.writeback_layer. Everything else is
# identical, so the delta is the return leg and nothing else.
SWAPBACK_ARMS = ("scored_swapback",)


def configure(arm: str, args, slo_ms: float) -> dict:
    for k in ("VLLM_KVPIPE_HYBRID_RUNGD", "VLLM_KVPIPE_HYBRID_RUNGD_FREE",
              "VLLM_KVPIPE_HYBRID_TWOPASS"):
        os.environ.pop(k, None)
    os.environ["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
    os.environ["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
    os.environ["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
    # migration only in the last arm
    os.environ["VLLM_KVPIPE_TIER_POLICY"] = (
        "scored" if arm in ("scored", "scored_migrate", "scored_swapback")
        else "arrival")
    if arm == "scored_migrate":
        os.environ["VLLM_KVPIPE_MIGRATE_WARMUP"] = str(args.migrate_warmup)
        os.environ["VLLM_KVPIPE_PROBE_STEPS"] = str(args.probe_steps)
    else:
        # disable migration by making warmup unreachable
        os.environ["VLLM_KVPIPE_MIGRATE_WARMUP"] = "100000000"

    os.environ["VLLM_KVPIPE_RUNGD_WRITEBACK"] = (
        "1" if arm in SWAPBACK_ARMS else "0")

    extra = dict(enable_transfer_aware_admission=True,
                 enable_hybrid_kv_tiers=True,
                 transfer_aware_slo_p99_itl_ms=slo_ms,
                 hybrid_itl_ms=slo_ms,
                 hybrid_pcie_gbps=args.link_gbps)
    if arm.startswith("uniform_"):
        frac = int(arm.split("_")[1]) / 100.0
        # fixed fraction on host: cap HBM slots at (1-frac) of the offered load
        extra["hybrid_b_hbm_max"] = max(1, int(round(args.offered * (1 - frac))))
    else:
        extra["hybrid_b_hbm_max"] = args.b_hbm_max
    return extra


def _find_pool(llm):
    """Reach the worker's RungdHostPool to read its traffic counters.

    In-process only (V1 multiprocessing is disabled above), and deliberately
    tolerant: if the path changes, the harness reports 0 bytes rather than
    dying mid-sweep. A missing counter shows up as an obviously-wrong zero in
    the output, which is safer than an exception after an hour of GPU time.
    """
    try:
        core = llm.llm_engine.engine_core
        inner = getattr(core, "engine_core", core)
        runner = inner.model_executor.driver_worker.worker.model_runner
        return getattr(runner, "_rungd_pool", None)
    except Exception:
        return None


def _teardown(llm) -> None:
    """Fully release the GPU between arms.

    Each arm builds its own engine. `del llm` alone does NOT reliably free the
    weights + KV pool before the next engine starts profiling free memory, and
    the next build then dies with "Free memory ... is less than desired GPU
    memory utilization". Harmless with a 0.5B model; fatal at 70B, where every
    engine holds 70+ GB -- exactly the kind of thing that burns rental hours.
    """
    import gc

    import torch
    try:
        core = llm.llm_engine.engine_core
        inner = getattr(core, "engine_core", core)
        for meth in ("shutdown", "close"):
            fn = getattr(inner, meth, None) or getattr(core, meth, None)
            if callable(fn):
                fn()
                break
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()


def run_arm(arm: str, args, slo_ms: float) -> dict:
    import numpy as np
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    extra = configure(arm, args, slo_ms)
    llm = LLM(model=args.model, dtype=args.dtype, enable_prefix_caching=False,
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.ctx + args.out_len + 64,
              max_num_seqs=args.max_num_seqs, enforce_eager=args.eager,
              disable_log_stats=True, **extra)
    core = llm.llm_engine.engine_core
    sched = getattr(getattr(core, "engine_core", core), "scheduler", None)
    ctl = sched.transfer_aware_admission if sched else None

    rng = np.random.default_rng(1234)
    vocab = llm.llm_engine.model_config.get_vocab_size()
    prompts = [TokensPrompt(prompt_token_ids=rng.integers(
        10, vocab - 10, size=args.ctx, dtype=np.int64).tolist())
        for _ in range(args.offered)]
    sp = SamplingParams(temperature=0.0, max_tokens=args.out_len,
                        ignore_eos=True, min_tokens=args.out_len)

    conc: list[int] = []
    if sched is not None:
        orig = sched.schedule

        def sampled(*a, **kw):
            out = orig(*a, **kw)
            conc.append(len(sched.running))
            return out
        sched.schedule = sampled

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    toks = sum(len(o.outputs[0].token_ids) for o in outs)
    warm = conc[len(conc) // 4:] if len(conc) > 8 else conc
    probes = list(getattr(ctl, "_probe_results", [])) if ctl else []
    # BYTES, not request counts. "Uniform vs selective at the same offload
    # ratio" is only a controlled comparison if the two arms actually move
    # comparable volume; matching the NUMBER of host-tier requests does not
    # ensure that, because the policies pick requests of different contexts.
    # Report it so the match is verifiable instead of assumed.
    pool = _find_pool(llm)
    host_bytes = int(getattr(pool, "stream_bytes", 0) or 0) if pool else 0
    wb_bytes = int(getattr(pool, "writeback_bytes", 0) or 0) if pool else 0
    res = dict(arm=arm, slo_ms=slo_ms, wall_s=wall,
               tokens_per_s=toks / wall,
               steady_concurrency=(sum(warm) / len(warm)) if warm else 0.0,
               migrations=(ctl.migration_stats() if ctl else {}),
               probe_verdicts=[{"baseline_ms": b, "after_ms": a, "worse": w}
                               for b, a, w in probes],
               sampled_steps=len(conc),
               host_read_bytes=host_bytes,
               writeback_bytes=wb_bytes,
               link_bytes_total=host_bytes + wb_bytes,
               token_ids=[tuple(o.outputs[0].token_ids) for o in outs])
    _teardown(llm)
    return res


def _run_arm_subprocess(arm: str, slo: float, args) -> dict:
    """Run one arm in a FRESH process.

    NOTE on cost: unlike endpoint_comparison.py -- which sweeps SLOs INSIDE a
    single engine because the SLO is only a scheduler setting -- the arms here
    differ in ENGINE configuration (hybrid_b_hbm_max, tier policy), so one model
    load per arm is unavoidable. 5 arms x ~19 min at 70B is ~1.6 h; budget for
    it rather than being surprised by it.

    In-process teardown is not sufficient: vLLM keeps the CUDA context (and the
    weights + KV pool) alive after the engine is dropped, so the next arm's
    memory profiling sees a nearly-full GPU and refuses to start. Harmless at
    0.5B, fatal at 70B. A subprocess per arm makes cleanup unconditional.
    """
    import subprocess
    import sys
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    cmd = [sys.executable, os.path.abspath(__file__),
           "--single-arm", arm, "--arm-slo", str(slo), "--out", tmp]
    for k, v in vars(args).items():
        if k in ("single_arm", "arm_slo", "out", "arms", "slos", "slo"):
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd += [flag, str(v)]
    import time as _t
    print(f"  [launch] arm={arm} ...")
    _t0 = _t.time()
    cmd.insert(1, "-u")   # child must not block-buffer either
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  [done  ] arm={arm} in {_t.time()-_t0:.0f}s rc={r.returncode}")
    if r.returncode != 0:
        tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
        raise RuntimeError(f"arm {arm} failed:\n  " + "\n  ".join(tail))
    with open(tmp) as fh:
        return json.load(fh)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--out-len", type=int, default=64)
    p.add_argument("--offered", type=int, default=16)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--b-hbm-max", type=int, default=8)
    p.add_argument("--gpu-mem-util", type=float, default=0.35)
    p.add_argument("--link-gbps", type=float, default=25.1)
    p.add_argument("--slo", type=float, default=50.0)
    p.add_argument("--probe-steps", type=int, default=16)
    p.add_argument("--migrate-warmup", type=int, default=24)
    p.add_argument("--eager", action="store_true", default=True)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--single-arm", default="",
                   help="internal: run ONE arm and dump json "
                        "(each arm runs in its own process so the "
                        "CUDA context is fully released)")
    p.add_argument("--arm-slo", type=float, default=0.0,
                   help="internal: SLO for --single-arm")
    p.add_argument("--out", default="policy_ablation.json")
    args = p.parse_args()

    if args.single_arm:
        r = run_arm(args.single_arm, args, args.arm_slo or args.slo)
        with open(args.out, "w") as f:
            json.dump(r, f)
        return 0

    arms = [a for a in args.arms.split(",") if a]
    results, ref = [], None
    print(f"model={args.model} ctx={args.ctx} offered={args.offered} "
          f"SLO={args.slo} ms link={args.link_gbps} GB/s\n")
    print(f"{'arm':<16} {'tok/s':>9} {'wall':>8} {'GB/step in':>11} "
          f"{'GB/step back':>13} {'migrations':>18}  tokens")
    print("(steady_concurrency is NOT shown: it is a mean over sampled steps, "
          "so a SLOWER arm holds\n requests longer and scores HIGHER. It "
          "produced a retracted claim. Compare tok/s.)\n")
    for arm in arms:
        r = _run_arm_subprocess(arm, args.slo, args)
        if ref is None:
            ref = r["token_ids"]
            chk = "ref"
        else:
            chk = "OK" if r["token_ids"] == ref else "**DIVERGED**"
        r["token_check"] = chk
        r.pop("token_ids")
        results.append(r)
        steps = max(1, int(r.get("sampled_steps") or 0)) or 1
        gin = r.get("host_read_bytes", 0) / 1e9 / steps
        gbk = r.get("writeback_bytes", 0) / 1e9 / steps
        print(f"{arm:<16} {r['tokens_per_s']:>9.1f} {r['wall_s']:>7.1f}s "
              f"{gin:>11.3f} {gbk:>13.3f} "
              f"{str(r['migrations']):>18}  {chk}")
        for v in r["probe_verdicts"]:
            print(f"                 probe: {v['baseline_ms']:.1f} -> "
                  f"{v['after_ms']:.1f} ms  worse={v['worse']}")

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)

    by = {r["arm"]: r for r in results}
    print("\n" + "=" * 72)
    # Compare on THROUGHPUT. steady_concurrency rewards the slower arm (it is
    # a mean over sampled steps, so an arm that holds requests longer scores
    # higher); it produced a claim we had to retract.
    base_arm = max((a for a in by if a.startswith("uniform_")),
                   key=lambda a: by[a]["tokens_per_s"], default=None)
    if base_arm:
        base = by[base_arm]["tokens_per_s"]
        b_in = by[base_arm].get("host_read_bytes", 0)
        print(f"best uniform arm: {base_arm} at {base:.1f} tok/s")
        for a in ("arrival", "scored", "scored_migrate"):
            if a in by:
                d = (by[a]["tokens_per_s"] / base - 1) * 100 if base else 0.0
                # The comparison is only controlled if the arms moved similar
                # volume. Say so when they did not, rather than reporting a
                # delta that is really a difference in offload ratio.
                a_in = by[a].get("host_read_bytes", 0)
                ratio = (a_in / b_in) if b_in else 0.0
                flag = "" if 0.8 <= ratio <= 1.25 else \
                    f"  [UNMATCHED: moved {ratio:.2f}x the bytes]"
                print(f"{a:<16} vs {base_arm}: {d:+6.1f}% tok/s{flag}")
    print("\nTHE QUESTION: at a MATCHED interconnect budget, does choosing")
    print("which requests to offload beat spreading the same bytes uniformly?")
    print("If not, the policy layer is decoration and the contribution is the")
    print("analysis, not the system.")

    # The return leg: swap semantics vs streaming, policy held fixed.
    if "scored" in by and "scored_swapback" in by:
        a, b = by["scored"], by["scored_swapback"]
        t_a = a.get("link_bytes_total", 0) or 1
        t_b = b.get("link_bytes_total", 0) or 1
        print("\n" + "-" * 72)
        print("RETURN LEG (swap semantics vs streaming, same policy):")
        print(f"  streaming : {a['tokens_per_s']:7.1f} tok/s   "
              f"link {t_a/1e9:8.2f} GB total")
        print(f"  swap-back : {b['tokens_per_s']:7.1f} tok/s   "
              f"link {t_b/1e9:8.2f} GB total  ({t_b/t_a:.2f}x)")
        print(f"  cost of persisting the offloaded unit: "
              f"{(b['tokens_per_s']/a['tokens_per_s'] - 1)*100:+.1f}% tok/s")
        print("  A design whose offloaded unit is OWNED by one side pays this;")
        print("  one whose device copy is a CACHE does not. Measured, not")
        print("  inferred from another system's protocol.")
    verdicts = [v for r in results for v in r["probe_verdicts"]]
    if verdicts:
        worse = sum(1 for v in verdicts if v["worse"])
        print(f"\nadaptive probe: {len(verdicts)} verdicts, {worse} said the "
              f"demotion was HARMFUL")
        print("  on A100/PCIe we measured HARMFUL (correctly reverted);")
        print("  on C2C this should flip to BENEFICIAL. If it does not, the")
        print("  host path has fixed per-step overhead that survives a fast link.")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
