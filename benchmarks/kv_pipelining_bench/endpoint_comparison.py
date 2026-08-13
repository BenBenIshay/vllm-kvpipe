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

ARMS = ("all_hbm", "all_host", "mixture")


def build_llm(arm: str, args, slo_ms: float):
    """One engine per arm. b_hbm_max is what selects the arm."""
    from vllm import LLM

    for k in ("VLLM_KVPIPE_HYBRID_RUNGD", "VLLM_KVPIPE_HYBRID_RUNGD_FREE",
              "VLLM_KVPIPE_HYBRID_TWOPASS", "VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM"):
        os.environ.pop(k, None)

    extra: dict = {}
    if arm == "all_hbm":
        # Stock behaviour: tiers entirely off.
        pass
    else:
        os.environ["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
        os.environ["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
        os.environ["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
        # Reclaim a host-tier request's prefill blocks once they are seeded to
        # host. Without it the request keeps its whole prefill resident, frees
        # only what it would have grown during decode, and still pays S*k of
        # interconnect every step -- strictly worse than leaving it in HBM.
        # `--reclaim 0` reproduces that (paper/BLOCKER_PREFILL_RESIDENCY.md).
        if args.reclaim:
            os.environ["VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM"] = "1"
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
        if arm == "all_host":
            # DirectKV does NOT gate admission -- it puts every request's KV on
            # the host and runs. Leaving B1's transfer budget active with zero
            # HBM-tier slots means requests beyond the budget have NO fallback
            # tier and queue forever (observed: this arm hung indefinitely at 0%
            # GPU). Model the real system: admit everything and let the measured
            # performance be whatever it is.
            extra["hybrid_itl_ms"] = 1e9                   # unlimited B1 budget
            extra["transfer_aware_slo_p99_itl_ms"] = 0.0   # disables B2
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


def run_arm_sweep(arm: str, args, slos: list) -> list:
    """Build ONE engine for this arm, then measure every SLO point in-process.

    The SLO is only a scheduler setting -- it does not change the model or the
    KV pool -- so rebuilding the engine per SLO would reload 72.7 GB of weights
    for nothing (~19 min/load measured on this cluster's FS; 15 loads would be
    ~5 h of a 24 h budget). Subprocess isolation is kept BETWEEN arms, which is
    where it is actually needed, since arms differ in memory layout.
    """
    llm = build_llm(arm, args, slos[0] if slos else 50.0)
    out = []
    for slo in slos:
        _set_slo(llm, slo)
        out.append(_measure(arm, args, llm, slo))
    _teardown(llm)
    return out


def _set_slo(llm, slo_ms: float) -> None:
    """Retune the running scheduler to a new SLO and clear adaptive state.

    Controller state (ITL EMA, throttled flag, migration probe, counters) is
    reset so each SLO point is an independent measurement rather than
    inheriting the previous point's history.
    """
    core = llm.llm_engine.engine_core
    sched = getattr(getattr(core, "engine_core", core), "scheduler", None)
    if sched is None:
        return
    cfg = sched.scheduler_config
    cfg.transfer_aware_slo_p99_itl_ms = slo_ms
    cfg.hybrid_itl_ms = slo_ms
    ctl = getattr(sched, "transfer_aware_admission", None)
    if ctl is not None:
        ctl._resident_step_ms_ema = 0.0
        ctl._itl_throttled_state = False
        ctl._ema_window = []
        ctl._probe = None
        ctl._probe_results = []
        ctl._migration_counts = {}
        ctl._migrated_at_step = {}
        ctl._demote_suppressed_until = 0
    if hasattr(sched, "_promoting_req_ids"):
        sched._promoting_req_ids.clear()


def _measure(arm: str, args, llm, slo_ms: float):
    """Closed-loop: offer a large burst, measure what is actually served."""
    import numpy as np
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

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
    orig = None
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
    try:
        outs = llm.generate(prompts, sp, use_tqdm=False)
    finally:
        # restore: _measure is called once per SLO point on the SAME engine, and
        # re-wrapping an already-wrapped schedule() would nest the samplers.
        if sched is not None:
            sched.schedule = orig
    wall = time.perf_counter() - t0

    toks = sum(len(o.outputs[0].token_ids) for o in outs)
    warm = conc[len(conc) // 4:] if len(conc) > 8 else conc   # drop ramp-up
    steady = (sum(warm) / len(warm)) if warm else 0.0
    itl_ms = wall / max(1, args.out_len) * 1000.0
    return dict(arm=arm, slo_ms=slo_ms, wall_s=wall,
                steady_concurrency=steady, peak_concurrency=max(conc or [0]),
                tokens=toks, tokens_per_s=toks / wall,
                itl_proxy_ms=itl_ms, tier_samples=tiers_seen,
                token_ids=[tuple(o.outputs[0].token_ids) for o in outs])


def _run_arm_subprocess(arm: str, slos: list, args) -> list:
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
           "--single-arm", arm,
           "--arm-slos", ",".join(str(x) for x in slos), "--out", tmp]
    for k, v in vars(args).items():
        if k in ("single_arm", "arm_slos", "out", "arms", "slos", "slo"):
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
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=args.arm_timeout_s)
    except subprocess.TimeoutExpired:
        # A hung arm must fail FAST -- on a metered box a silent hang is the
        # most expensive failure mode available -- but it must NOT take the
        # arms that already succeeded with it. Raising here discarded a
        # completed all_hbm and mixture because all_host timed out, i.e. it
        # threw away the entire comparison over the one arm we already know is
        # slow on PCIe by construction. Report the timeout as this arm's result
        # and let the caller decide.
        print(f"  [TIMEOUT] arm={arm} exceeded --arm-timeout-s="
              f"{args.arm_timeout_s}s; recording as unfinished and continuing "
              "(check admission gating / queueing)")
        return {"arm": arm, "failed": "timeout",
                "timeout_s": args.arm_timeout_s}
    print(f"  [done  ] arm={arm} in {_t.time()-_t0:.0f}s rc={r.returncode}")
    if r.returncode != 0:
        tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
        print(f"  [FAILED ] arm={arm} rc={r.returncode}:\n    "
              + "\n    ".join(tail))
        return {"arm": arm, "failed": f"rc={r.returncode}",
                "tail": tail}
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
    p.add_argument("--arm-timeout-s", type=int, default=3600,
                   help="watchdog: fail an arm rather than let it hang")
    p.add_argument("--single-arm", default="",
                   help="internal: run ONE arm and dump json "
                        "(each arm runs in its own process so the "
                        "CUDA context is fully released)")
    p.add_argument("--arm-slos", default="",
                   help="internal: comma-separated SLOs for --single-arm. The "
                        "arm builds ONE engine and sweeps them in-process; the "
                        "SLO is only a scheduler setting, so reloading 72.7 GB "
                        "of weights per SLO point would be pure waste "
                        "(~19 min/load measured on this cluster).")
    p.add_argument("--reclaim", type=int, default=1,
                   help="free a host-tier request's prefill blocks once seeded "
                        "to host (default on). 0 reproduces the pre-fix "
                        "behaviour, where offloading frees almost nothing.")
    p.add_argument("--out", default="endpoint_comparison.json")
    args = p.parse_args()

    if args.single_arm:
        slo_list = [float(x) for x in args.arm_slos.split(",") if x]
        rs = run_arm_sweep(args.single_arm, args, slo_list)
        with open(args.out, "w") as f:
            json.dump(rs, f)
        return 0

    slos = [float(x) for x in args.slos.split(",")]
    arms = [a for a in args.arms.split(",") if a]
    results = []

    print(f"model={args.model} ctx={args.ctx} offered={args.offered} "
          f"link={args.link_gbps} GB/s")
    print(f"{'SLO':>6} {'arm':<10} {'steady conc':>12} {'tok/s':>9} "
          f"{'wall':>8}  tiers")
    # ARM-major: one subprocess (=> one model load) per arm, SLOs swept inside.
    ref: dict = {}
    by_arm: dict = {}
    for arm in arms:
        by_arm[arm] = _run_arm_subprocess(arm, slos, args)
    # An arm that timed out or crashed returns a dict, not the per-SLO list.
    # Drop it from the table (it is reported above) rather than losing the
    # arms that did finish.
    unfinished = {a: r for a, r in by_arm.items() if isinstance(r, dict)}
    arms = [a for a in arms if a not in unfinished]
    for slo in slos:
        for arm in arms:
            r = next(x for x in by_arm[arm] if x["slo_ms"] == slo)
            if arm == "all_hbm":
                ref[slo] = r["token_ids"]
                ok = "ref"
            elif slo not in ref:
                # No all_hbm reference in this run (e.g. --arms all_host alone).
                # Reporting DIVERGED here would be a FALSE correctness alarm,
                # and a false alarm is worse than no check: it teaches you to
                # distrust the checker.
                ok = "no-ref"
            else:
                ok = "OK" if r["token_ids"] == ref[slo] else "**DIVERGED**"
            r["token_check"] = ok
            r.pop("token_ids")
            results.append(r)
            print(f"{slo:>6.0f} {arm:<10} {r['steady_concurrency']:>12.1f} "
                  f"{r['tokens_per_s']:>9.1f} {r['wall_s']:>7.1f}s  "
                  f"{r['tier_samples']}  {ok}")

    if unfinished:
        for a, r in unfinished.items():
            print(f"{'':>6} {a:<10} {'--':>12} {'--':>9} {'--':>8}  "
                  f"DID NOT FINISH ({r.get('failed')})")

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results,
                   "unfinished": unfinished}, f, indent=2)

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
        if "all_hbm" not in by or "mixture" not in by:
            print(f"SLO {slo:>5.0f} ms: cannot compare -- missing "
                  f"{sorted({'all_hbm', 'mixture'} - set(by))}")
            continue
        a = by["all_hbm"]["steady_concurrency"]
        c = by["mixture"]["steady_concurrency"]
        # A missing all_host must NOT be silently read as an endpoint of 0 --
        # that would flatter the mixture. Compare against the endpoints we
        # actually measured, and say which those are.
        if "all_host" in by:
            b = by["all_host"]["steady_concurrency"]
            best_end, against = max(a, b), "best endpoint"
            host_col = f"{b:5.1f}"
        else:
            best_end, against = a, "all-HBM only (all-host DID NOT FINISH)"
            host_col = "  n/a"
        gain = (c / best_end - 1) * 100 if best_end else 0.0
        verdict = ("MIXTURE WINS" if gain >= 15 else
                   "marginal (<15%)" if gain > 0 else "MIXTURE LOSES")
        print(f"SLO {slo:>5.0f} ms: all-HBM {a:5.1f} | all-host {host_col} | "
              f"mixture {c:5.1f}  -> {gain:+6.1f}% vs {against}  [{verdict}]")
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
