#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Experiment 4.2c: the heterogeneous workload, and the gate that says it works.

WHY THIS EXISTS, AND WHY IT IS A GATE RATHER THAN A SETUP SCRIPT. §4.3 asks
whether our placement policy is load-bearing: `VLLM_KVPIPE_TIER_POLICY=scored`
against `arrival`, 16 GPU-hours of rented machine. `host_affinity` scores a
request on TEMPORAL signals -- seconds idle since its last token, and tokens of
generation remaining -- because context length provably cannot discriminate
between the tiers (the same cancellation as design rule (a); freeing S*k of HBM
costs S*k of interconnect per step, and both scale with S).

Every harness in this directory generates requests that are identical to each
other: one burst, `max_tokens == min_tokens`, `ignore_eos`. On that workload
`idle_score` is 0 for everyone and `rem_score` is the same for everyone, so
`host_affinity` returns one number for the whole running set and `scored` is
**byte-for-byte** `arrival`. The ablation would then measure nothing and report
it as "+0.0%: the policy layer is decoration" -- a plausible number, produced by
a workload with nothing in it to choose between. That is the failure this file
prevents, and it is worth more than the hours because a null result from an
inert setup is not a null result, it is a mistake that reads like one.

WHAT IT RUNS. Not a bespoke generator: the plan's own instruction is to use
§4.5's workload, and `benchmarks/multi_turn/benchmark_serving_multi_turn.py`
already ships in this tree with exactly the three properties 4.2c asks for --
per-turn output lengths drawn from a distribution (so remaining generation
differs across the running set), multi-turn conversations whose client sleeps
between turns (real idle gaps, which a single-shot dataset cannot produce), and
Poisson arrivals (a burst admits everyone before any gate engages, which has
already produced two inconclusive runs here).

THE CONTROL IS THE POINT. A gate that only ever sees the heterogeneous workload
cannot distinguish "the workload discriminates" from "the measurement always
says it does". So the default runs BOTH: the heterogeneous workload and a
homogeneous control (one turn, constant output length, burst arrival). The
control must show ~zero spread. If both show spread, the trace is noise and the
gate is not measuring what it claims; that is reported as INCONCLUSIVE rather
than as a pass.

Requires the server-side trace `VLLM_KVPIPE_AFFINITY_TRACE=1`, which this
harness sets, and which logs per-step min/p25/p50/p75/max over the running set.

Exit codes:  0 the workload discriminates (§4.3 may run)
             1 INERT -- do not run §4.3 on it
             2 INCONCLUSIVE (no samples, server failure, or control also spread)

Usage:
    python affinity_spread_gate.py --model Qwen/Qwen2.5-0.5B-Instruct \\
        --text-file /mnt/beegfs/benishay/bench_data/pg1184.txt \\
        --outdir /mnt/beegfs/benishay/paper_runs/20260915/affinity
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import statistics
import subprocess
import sys

print = functools.partial(print, flush=True)  # noqa: A001

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from gate_goodput import Server, tier_stats  # noqa: E402

MULTI_TURN = os.path.join(ROOT, "benchmarks", "multi_turn",
                          "benchmark_serving_multi_turn.py")

_TRACE = re.compile(
    r"AFFINITY_TRACE: n=(\d+) min=([\d.]+) p25=([\d.]+) p50=([\d.]+) "
    r"p75=([\d.]+) max=([\d.]+) spread=([\d.]+) idle_spread=([\d.]+) "
    r"rem_spread=([\d.]+) w_idle=([\d.]+)")


# --------------------------------------------------------------------------
# Workload configs
# --------------------------------------------------------------------------
def hetero_config(a) -> dict:
    """Heterogeneous: varied generation lengths, several turns, idle gaps.

    `prompt_output.num_tokens` is the knob that moves `rem_score`: a lognormal
    with a wide tail means some requests in the running set have hundreds of
    tokens left to generate while others are nearly done. The multi-turn
    structure plus `--request-rate` is what moves `idle_score`: between turns a
    client sleeps, and a conversation that is waiting for its user holds KV
    while consuming no bandwidth -- the case the policy exists for.
    """
    return {
        "filetype": "generate_conversations",
        "num_conversations": a.conversations,
        "text_files": [os.path.basename(a.text_file)],
        "print_stats": False,
        "prompt_input": {
            "num_turns": {"distribution": "uniform", "min": 2, "max": 6},
            "common_prefix_num_tokens": {"distribution": "constant", "value": 0},
            "prefix_num_tokens": {"distribution": "lognormal",
                                  "average": a.prefix_avg, "max": a.prefix_max},
            "num_tokens": {"distribution": "uniform", "min": 32, "max": 128},
        },
        "prompt_output": {
            "num_tokens": {"distribution": "lognormal",
                           "average": a.gen_avg, "max": a.gen_max},
        },
    }


def control_config(a) -> dict:
    """Homogeneous control: what every existing harness in this directory does.

    One turn, constant prompt, constant generation length. Run with
    `--request-rate 0` (burst). If the gate reports spread on THIS, the spread
    it reports on the heterogeneous workload means nothing.
    """
    return {
        "filetype": "generate_conversations",
        "num_conversations": a.conversations,
        "text_files": [os.path.basename(a.text_file)],
        "print_stats": False,
        "prompt_input": {
            "num_turns": {"distribution": "constant", "value": 1},
            "common_prefix_num_tokens": {"distribution": "constant", "value": 0},
            "prefix_num_tokens": {"distribution": "constant", "value": 256},
            "num_tokens": {"distribution": "constant", "value": 64},
        },
        "prompt_output": {
            "num_tokens": {"distribution": "constant", "value": a.gen_avg},
        },
    }


# --------------------------------------------------------------------------
def parse_trace(server_log: str, min_running: int) -> dict:
    """Per-step affinity spread, aggregated over the run.

    Steps with fewer than `min_running` requests are dropped: the spread over a
    single request is 0 by construction and would dilute the aggregate towards
    "inert" for reasons that have nothing to do with the workload.
    """
    spreads, iqrs, mins, maxs, meds, ns = [], [], [], [], [], []
    idle_sp, rem_sp, w_idle = [], [], []
    try:
        with open(server_log, errors="replace") as fh:
            for line in fh:
                m = _TRACE.search(line)
                if not m:
                    continue
                n = int(m.group(1))
                if n < min_running:
                    continue
                lo, p25, p50, p75, hi, sp = (float(m.group(i)) for i in range(2, 8))
                ns.append(n)
                mins.append(lo)
                meds.append(p50)
                maxs.append(hi)
                iqrs.append(p75 - p25)
                spreads.append(sp)
                idle_sp.append(float(m.group(8)))
                rem_sp.append(float(m.group(9)))
                w_idle.append(float(m.group(10)))
    except OSError:
        pass
    if not spreads:
        return {"samples": 0}
    return {
        "median_idle_spread": statistics.median(idle_sp),
        "median_rem_spread": statistics.median(rem_sp),
        "w_idle": statistics.median(w_idle),
        "samples": len(spreads),
        "mean_running": statistics.mean(ns),
        "median_spread": statistics.median(spreads),
        "max_spread": max(spreads),
        "median_iqr": statistics.median(iqrs),
        "min_score": min(mins),
        "median_score": statistics.median(meds),
        "max_score": max(maxs),
        "steps_with_spread": sum(1 for s in spreads if s > 0.01),
        "frac_steps_with_spread": sum(1 for s in spreads if s > 0.01) / len(spreads),
    }


def run_workload(a, label: str, cfg: dict, rate: float) -> dict:
    """One server + one multi-turn run; returns the parsed trace."""
    cfg_path = os.path.join(a.outdir, f"conv_{label}.json")
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)

    # The generator resolves `text_files` relative to its own working
    # directory, so it runs from the directory holding the corpus.
    text_dir = os.path.dirname(os.path.abspath(a.text_file))

    with Server("C", a, a.slo_ms, f"affinity_{label}") as srv:
        if not srv.ready:
            print(f"  server did not start; see {srv.log}")
            return {"error": "server not ready", "server_log": srv.log}
        cmd = [
            a.python, MULTI_TURN,
            "-m", a.model,
            "-u", f"http://127.0.0.1:{a.port}",
            "-i", cfg_path,
            "-n", str(a.max_requests),
            "-p", str(a.clients),
            "--request-rate", str(rate),
            "--seed", str(a.seed),
        ]
        log = os.path.join(a.outdir, f"client_{label}.log")
        with open(log, "w") as fh:
            fh.write("# " + " ".join(cmd) + "\n")
            fh.flush()
            try:
                rc = subprocess.run(cmd, cwd=text_dir, stdout=fh,
                                    stderr=subprocess.STDOUT,
                                    timeout=a.client_timeout_s).returncode
            except subprocess.TimeoutExpired:
                # Never propagate past the server context manager: an exception
                # here leaves an engine holding the GPU.
                fh.write(f"\n# CLIENT TIMEOUT after {a.client_timeout_s}s\n")
                rc = -1
        trace = parse_trace(srv.log, a.min_running)
        trace["tiers"] = tier_stats(srv.log)
        trace["client_rc"] = rc
        trace["client_log"] = log
        trace["server_log"] = srv.log
    return trace


def report(label: str, t: dict) -> None:
    if t.get("error"):
        print(f"  {label}: FAILED ({t['error']})")
        return
    if not t.get("samples"):
        print(f"  {label}: no AFFINITY_TRACE samples "
              f"(instrumentation off, or no step had >= min-running requests)")
        return
    print(f"  {label}: {t['samples']} steps, mean running {t['mean_running']:.1f}")
    print(f"      host_affinity   min {t['min_score']:.3f}   "
          f"median {t['median_score']:.3f}   max {t['max_score']:.3f}")
    print(f"      per-step spread  median {t['median_spread']:.3f}   "
          f"max {t['max_spread']:.3f}   IQR {t['median_iqr']:.3f}")
    print(f"      carried by:  idle {t['median_idle_spread']:.3f} "
          f"(weight {t['w_idle']:.2f})   remaining-generation "
          f"{t['median_rem_spread']:.3f} (weight {1 - t['w_idle']:.2f})")
    print(f"      steps with any spread: {t['frac_steps_with_spread'] * 100:.1f}%")
    print(f"      riders seen (max n_cpu): {t['tiers']['max_n_cpu']}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--python", default=os.path.join(ROOT, ".venv/bin/python"))
    p.add_argument("--text-file",
                   default="/mnt/beegfs/benishay/bench_data/pg1184.txt",
                   help="corpus the conversation generator samples from")
    # Defaults are the configuration MEASURED to pass on 2026-09-15, not a
    # guess. The first attempt (16 conversations, 6 clients, rate 0.5,
    # generation 128/768) came back INERT at spread 0.029 with a mean running
    # set of 2.5 requests -- too few concurrent requests to differ, and too
    # narrow a generation distribution. At 32/16/1.0 with generation 256/2048
    # the mean running set is 9.7 and the spread is 0.202. Both runs are in
    # paper_runs/20260915/.
    p.add_argument("--conversations", type=int, default=32)
    p.add_argument("--max-requests", type=int, default=128)
    p.add_argument("--clients", type=int, default=16)
    p.add_argument("--request-rate", type=float, default=1.0,
                   help="Poisson rate PER CLIENT; also the think-time between "
                        "turns, which is what creates the idle signal")
    p.add_argument("--gen-avg", type=int, default=256)
    p.add_argument("--gen-max", type=int, default=2048)
    p.add_argument("--prefix-avg", type=int, default=512)
    p.add_argument("--prefix-max", type=int, default=2048)
    p.add_argument("--min-running", type=int, default=4)
    p.add_argument("--min-spread", type=float, default=0.15,
                   help="median per-step (max-min) required of the workload. "
                        "Calibrated against a CEILING of 0.30, not 1.0: the "
                        "idle component is weighted 0.70 and is identically "
                        "zero whenever each turn arrives as its own request "
                        "(a waiting session holds no running request), so the "
                        "reachable maximum is (1 - w_idle) x 1.0. 0.15 is "
                        "therefore half of what the signal can actually do.")
    p.add_argument("--min-iqr", type=float, default=0.05,
                   help="median per-step (p75-p25) required: guards against a "
                        "spread carried by one outlier request")
    p.add_argument("--control-max-spread", type=float, default=0.05,
                   help="the homogeneous control must stay below this")
    p.add_argument("--skip-control", action="store_true")
    p.add_argument("--seed", type=int, default=99)
    # server knobs, shared with gate_goodput.Server
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--out-len", type=int, default=0)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--num-gpu-blocks-override", type=int, default=0)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--b-hbm-max", type=int, default=4)
    p.add_argument("--link-gbps", type=float, default=25.1)
    p.add_argument("--tier-policy", default="scored", choices=("scored", "arrival"))
    p.add_argument("--slo-ms", type=float, default=1e9,
                   help="B1 byte budget for this run. The default is "
                        "deliberately unlimited: 4.2c asks whether the "
                        "WORKLOAD discriminates, so admission must not be the "
                        "thing limiting riders.")
    p.add_argument("--port", type=int, default=8679)
    p.add_argument("--server-timeout-s", type=float, default=1800.0)
    p.add_argument("--client-timeout-s", type=float, default=3600.0)
    p.add_argument("--outdir", default="/mnt/beegfs/benishay/paper_runs/affinity")
    p.add_argument("--out", default="affinity_spread.json")
    a = p.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    tok = os.path.join(ROOT, "hf_token")
    if os.path.exists(tok) and not os.environ.get("HF_TOKEN"):
        with open(tok) as fh:
            os.environ["HF_TOKEN"] = fh.read().strip()
    if not os.path.exists(a.text_file):
        print(f"missing corpus {a.text_file} -- fetch it once, e.g.\n"
              f"  curl -sSL -o {a.text_file} "
              f"https://www.gutenberg.org/cache/epub/1184/pg1184.txt")
        return 2
    if not os.path.exists(MULTI_TURN):
        print(f"missing {MULTI_TURN}")
        return 2
    # The trace is what the gate reads; set it here so no one has to remember.
    os.environ["VLLM_KVPIPE_AFFINITY_TRACE"] = "1"

    print("=" * 62)
    print(" 4.2c WORKLOAD GATE   host_affinity spread")
    print(f" model={a.model}  conversations={a.conversations} "
          f"clients={a.clients} rate={a.request_rate}/client")
    print("=" * 62)

    print("\n[1/2] heterogeneous workload (multi-turn, Poisson, varied lengths)")
    het = run_workload(a, "hetero", hetero_config(a), a.request_rate)
    report("hetero", het)

    ctrl = None
    if not a.skip_control:
        print("\n[2/2] homogeneous control (single turn, constant length, burst)")
        ctrl = run_workload(a, "control", control_config(a), 0.0)
        report("control", ctrl)

    out = {"model": a.model, "hetero": het, "control": ctrl,
           "thresholds": {"min_spread": a.min_spread, "min_iqr": a.min_iqr,
                          "control_max_spread": a.control_max_spread}}

    print("\nverdict")
    rc = 2
    if het.get("error") or not het.get("samples"):
        print("  INCONCLUSIVE: the heterogeneous run produced no trace samples.")
    elif ctrl is not None and ctrl.get("samples") and \
            ctrl["median_spread"] > a.control_max_spread:
        print(f"  INCONCLUSIVE: the homogeneous CONTROL also shows spread "
              f"({ctrl['median_spread']:.3f} > {a.control_max_spread}). The "
              f"trace is not measuring what it claims, so a pass on the "
              f"heterogeneous workload would prove nothing.")
    elif het["median_spread"] >= a.min_spread and het["median_iqr"] >= a.min_iqr:
        print(f"  PASS: median per-step spread {het['median_spread']:.3f} "
              f">= {a.min_spread}, IQR {het['median_iqr']:.3f} >= {a.min_iqr}"
              + (f"; control {ctrl['median_spread']:.3f}"
                 if ctrl and ctrl.get("samples") else ""))
        print("  The running set contains requests the policy can choose "
              "between. §4.3 is meaningful on this workload.")
        rc = 0
    else:
        print(f"  INERT: median spread {het['median_spread']:.3f} "
              f"(need {a.min_spread}), IQR {het['median_iqr']:.3f} "
              f"(need {a.min_iqr}).")
        print("  `scored` will be byte-for-byte `arrival` here. Do NOT spend "
              "§4.3's hours on this workload -- widen the generation-length "
              "distribution or lengthen the think-time first.")
        rc = 1

    out["exit_code"] = rc
    path = os.path.join(a.outdir, a.out)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
