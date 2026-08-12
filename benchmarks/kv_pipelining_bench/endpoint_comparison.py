#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""THE decisive experiment: all-HBM vs all-host vs mixture.

This is what the GH200 rental buys. Everything else we can measure on A100.

The claim under test (modelled, never measured): at a fixed HBM budget and a
fixed ITL SLO, serving some requests from HBM *and* some from host memory beats
either endpoint. For Llama-70B fp8 @8K on a GH200-96 the model predicts:

    all-HBM  8 requests     all-host  5      MIXTURE  13

Both endpoints exist in the wild. Stock vLLM/SGLang is 0% host. DirectKV
(OSDI'26) is 100% host for every request, always -- which is why it leaves the
non-weight HBM idle by construction. Nobody runs the interior point.

DESIGN NOTES (why it is set up this way)

* All three arms use the SAME substrate (our StagedHostBackend). This is a
  POLICY comparison, not a mechanism comparison -- otherwise a difference could
  be attributed to kernels rather than placement.
* Primary metric is REQUESTS SERVED AT A FIXED SLO, not raw throughput. Given a
  measured-grounded ceiling of ~x1.39, goodput-at-SLO is the honest framing.
* We sweep the SLO. Our applicability gate says capacity only binds ABOVE the
  full-pool ITL, so the tightest SLO should show NO benefit. That negative is a
  prediction and is worth as much as the positive.
* Every arm is checked token-identical against a no-tiers reference. An arm that
  is fast because it is wrong is worthless.

Dry-run on A100 with a small model; on GH200 only --model/--contexts change.
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

ARMS = ("all_hbm", "all_host", "mixture")


def build_llm(arm: str, args, slo_ms: float):
    """One engine per arm. b_hbm_max is what selects the arm."""
    from vllm import LLM

    for k in ("VLLM_KVPIPE_HYBRID_RUNGD", "VLLM_KVPIPE_HYBRID_RUNGD_FREE",
              "VLLM_KVPIPE_HYBRID_TWOPASS"):
        os.environ.pop(k, None)

    extra: dict = {}
    if arm == "all_hbm":
        # Stock behaviour: tiers entirely off.
        pass
    else:
        os.environ["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
        os.environ["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
        os.environ["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
        os.environ["VLLM_KVPIPE_TIER_POLICY"] = (
            "scored" if arm == "mixture" else "arrival")
        extra = dict(
            enable_transfer_aware_admission=True,
            enable_hybrid_kv_tiers=True,
            transfer_aware_slo_p99_itl_ms=slo_ms,
            hybrid_itl_ms=slo_ms,
            hybrid_pcie_gbps=args.link_gbps,
            # all_host: 0 HBM-tier slots -> every request is host-tier.
            # mixture : let the policy + B1/B2 decide the split.
            hybrid_b_hbm_max=0 if arm == "all_host" else args.b_hbm_max,
        )
    return LLM(model=args.model, dtype=args.dtype,
               enable_prefix_caching=False,
               gpu_memory_utilization=args.gpu_mem_util,
               max_model_len=args.ctx + args.out_len + 64,
               max_num_seqs=args.max_num_seqs,
               enforce_eager=args.eager, disable_log_stats=True,
               **extra)


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


def run_arm(arm: str, args, slo_ms: float):
    """Closed-loop: offer a large burst, measure what is actually served."""
    import numpy as np
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    llm = build_llm(arm, args, slo_ms)
    core = llm.llm_engine.engine_core
    sched = getattr(getattr(core, "engine_core", core), "scheduler", None)

    rng = np.random.default_rng(1234)
    vocab = llm.llm_engine.model_config.get_vocab_size()
    prompts = [
        TokensPrompt(prompt_token_ids=rng.integers(
            10, vocab - 10, size=args.ctx, dtype=np.int64).tolist())
        for _ in range(args.offered)
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=args.out_len,
                        ignore_eos=True, min_tokens=args.out_len)

    # steady-state concurrency sampler
    conc: list[int] = []
    tiers_seen: dict[str, int] = {}
    if sched is not None:
        orig = sched.schedule

        def sampled(*a, **kw):
            out = orig(*a, **kw)
            conc.append(len(sched.running))
            if arm != "all_hbm":
                ctl = sched.transfer_aware_admission
                for r in sched.running:
                    t = ctl.get_req_tier(r.request_id) or "hbm"
                    tiers_seen[t] = tiers_seen.get(t, 0) + 1
            return out
        sched.schedule = sampled

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0

    toks = sum(len(o.outputs[0].token_ids) for o in outs)
    warm = conc[len(conc) // 4:] if len(conc) > 8 else conc   # drop ramp-up
    steady = (sum(warm) / len(warm)) if warm else 0.0
    itl_ms = wall / max(1, args.out_len) * 1000.0
    res = dict(arm=arm, slo_ms=slo_ms, wall_s=wall,
               steady_concurrency=steady, peak_concurrency=max(conc or [0]),
               tokens=toks, tokens_per_s=toks / wall,
               itl_proxy_ms=itl_ms, tier_samples=tiers_seen,
               token_ids=[tuple(o.outputs[0].token_ids) for o in outs])
    _teardown(llm)
    return res


def _run_arm_subprocess(arm: str, slo: float, args) -> dict:
    """Run one arm in a FRESH process.

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
    r = subprocess.run(cmd, capture_output=True, text=True)
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
    p.add_argument("--offered", type=int, default=16,
                   help="requests offered; must exceed what HBM can hold")
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--b-hbm-max", type=int, default=8)
    p.add_argument("--gpu-mem-util", type=float, default=0.35)
    p.add_argument("--link-gbps", type=float, default=25.1,
                   help="A100 PCIe4 ~25; GH200 C2C ~450 (MEASURE IT FIRST)")
    p.add_argument("--slos", default="30,40,50,75,100")
    p.add_argument("--eager", action="store_true", default=True)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--single-arm", default="",
                   help="internal: run ONE arm and dump json "
                        "(each arm runs in its own process so the "
                        "CUDA context is fully released)")
    p.add_argument("--arm-slo", type=float, default=0.0,
                   help="internal: SLO for --single-arm")
    p.add_argument("--out", default="endpoint_comparison.json")
    args = p.parse_args()

    if args.single_arm:
        r = run_arm(args.single_arm, args, args.arm_slo)
        with open(args.out, "w") as f:
            json.dump(r, f)
        return 0

    slos = [float(x) for x in args.slos.split(",")]
    arms = [a for a in args.arms.split(",") if a]
    results = []

    print(f"model={args.model} ctx={args.ctx} offered={args.offered} "
          f"link={args.link_gbps} GB/s")
    print(f"{'SLO':>6} {'arm':<10} {'steady conc':>12} {'tok/s':>9} "
          f"{'wall':>8}  tiers")
    ref: dict = {}
    for slo in slos:
        for arm in arms:
            r = _run_arm_subprocess(arm, slo, args)
            key = arm if arm != "mixture" else "mixture"
            # correctness: every arm must match the all-HBM reference exactly
            if arm == "all_hbm":
                ref[slo] = r["token_ids"]
                ok = "ref"
            else:
                ok = "OK" if r["token_ids"] == ref.get(slo) else "**DIVERGED**"
            r["token_check"] = ok
            r.pop("token_ids")
            results.append(r)
            print(f"{slo:>6.0f} {arm:<10} {r['steady_concurrency']:>12.1f} "
                  f"{r['tokens_per_s']:>9.1f} {r['wall_s']:>7.1f}s  "
                  f"{r['tier_samples']}  {ok}")

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)

    # VACUITY GUARD. If the all-HBM arm already sustains everything offered,
    # HBM capacity was never the binding constraint and NO arm can differ --
    # the comparison proves nothing. This is the same failure mode that made an
    # earlier migration test vacuously "pass", so it is checked, not assumed.
    vacuous = []
    for slo in slos:
        a = next((r for r in results
                  if r["slo_ms"] == slo and r["arm"] == "all_hbm"), None)
        if a and a["steady_concurrency"] >= args.offered * 0.95:
            vacuous.append(slo)
    if vacuous:
        print("\n" + "!" * 72)
        print(f"VACUOUS at SLO {vacuous}: the all-HBM arm sustained "
              f"{args.offered} concurrent, i.e. the ENTIRE offered load.")
        print("HBM capacity was never binding, so the arms cannot differ and")
        print("this comparison measures nothing. Raise --offered (or --ctx, or")
        print("lower --gpu-mem-util) until all-HBM saturates BELOW the offered")
        print("load, then re-run.")
        print("!" * 72)

    print("\n" + "=" * 72)
    for slo in slos:
        by = {r["arm"]: r for r in results if r["slo_ms"] == slo}
        if len(by) < 3:
            continue
        a = by["all_hbm"]["steady_concurrency"]
        b = by["all_host"]["steady_concurrency"]
        c = by["mixture"]["steady_concurrency"]
        best_end = max(a, b)
        gain = (c / best_end - 1) * 100 if best_end else 0.0
        verdict = ("MIXTURE WINS" if gain >= 15 else
                   "marginal (<15%)" if gain > 0 else "MIXTURE LOSES")
        print(f"SLO {slo:>5.0f} ms: all-HBM {a:5.1f} | all-host {b:5.1f} | "
              f"mixture {c:5.1f}  -> {gain:+6.1f}% vs best endpoint  [{verdict}]")
    print("=" * 72)
    print("Kill criteria (agreed): mixture must beat BOTH endpoints by >=15% "
          "at matched SLO, else fall back to the characterization paper.")
    if vacuous:
        print("NOTE: results above are VACUOUS at SLO "
              f"{vacuous} -- see the warning; do not interpret them.")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
