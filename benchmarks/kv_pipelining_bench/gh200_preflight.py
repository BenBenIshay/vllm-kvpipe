#!/usr/bin/env python3
"""Everything that must be TRUE about a machine before the rider runs mean anything.

Written for a GH200 rental, where a misconfiguration is expensive rather than
merely annoying. Every check here corresponds to a mistake already made once on
A100:

  * `--hybrid-pcie-gbps 25.1` is hardcoded in every harness in this directory.
    Left alone on C2C, B1's byte budget is 18x too small and it admits ~2
    riders instead of ~37 -- the whole trip would measure PCIe-era behaviour on
    a machine rented to escape it. This MEASURES the link instead.
    (18x, not 36x: NVLink-C2C's headline 900 GB/s is BIDIRECTIONAL, and a
    rider's KV stream is unidirectional host-to-device, so 450 GB/s is the
    bound. Measuring rather than assuming is what makes this check right
    either way -- and the measured figure will land below 450 as well.)
  * `--hybrid-b-tok 131072` was passed while running a 64-layer model, so B1
    charged every rider half its real transfer. bytes/token is derived here
    from the model and cross-checked.
  * The host mirrors are PINNED. At high rider counts they are tens of GB, and
    a pinned allocation that fails or crawls is not obvious from a log.

Nothing here needs the hybrid path to work; it is arithmetic plus two probes,
so it can be run the moment a machine is available.

Run: srun ... .venv/bin/python benchmarks/kv_pipelining_bench/gh200_preflight.py \
       --model Qwen/Qwen2.5-32B-Instruct --ctx 2048 --target-riders 48
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

FAIL: list[str] = []
WARN: list[str] = []


def ok(cond: bool, what: str, detail: str = "", warn_only: bool = False) -> None:
    tag = "PASS" if cond else ("WARN" if warn_only else "FAIL")
    print(f"  [{tag}] {what}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        (WARN if warn_only else FAIL).append(what)


def measure_h2d_gbps(mb: int = 256, iters: int = 20) -> float:
    """Pinned host->device bandwidth, the quantity B1's budget is denominated in.

    Pinned and non_blocking, because that is exactly how `_issue_runs` moves KV;
    a pageable measurement would understate C2C badly.
    """
    n = mb * 1024 * 1024 // 2
    host = torch.empty(n, dtype=torch.float16, pin_memory=True)
    dev = torch.empty(n, dtype=torch.float16, device="cuda")
    for _ in range(3):
        dev.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        dev.copy_(host, non_blocking=True)
    t1.record()
    torch.cuda.synchronize()
    secs = t0.elapsed_time(t1) / 1000.0 / iters
    return (n * 2) / secs / 1e9


def kv_bytes_per_token(model: str) -> tuple[int, dict]:
    from transformers import AutoConfig

    c = AutoConfig.from_pretrained(model, trust_remote_code=True)
    L = int(c.num_hidden_layers)
    H = int(getattr(c, "num_key_value_heads", None) or c.num_attention_heads)
    D = int(getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads)
    return L * 2 * H * D * 2, {"layers": L, "kv_heads": H, "head_dim": D}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--out-len", type=int, default=64)
    ap.add_argument("--target-riders", type=int, default=48)
    ap.add_argument("--residents", type=int, default=8)
    ap.add_argument("--step-ms", type=float, default=49.0,
                    help="expected rider-free decode step; the hide budget is "
                         "denominated in it. Re-run with the MEASURED value "
                         "once the control arm has run on this machine.")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--weights-gb", type=float, default=72.7,
                    help="checkpoint size on the device; 72.7 measured for the "
                         "fp8 70B. KV space is what is left after this.")
    ap.add_argument("--hbm-gb", type=float, default=0.0,
                    help="override the measured HBM, to check the arithmetic "
                         "for a machine you are not on yet")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90,
                    help="must match what the experiment will actually run "
                         "with; capacity is very sensitive to it")
    ap.add_argument("--planned-offered", type=int, default=0,
                    help="the --offered you intend to run 4.1 with; checked "
                         "for vacuity")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print(f"=== preflight: {args.model} ctx={args.ctx} "
          f"target {args.target_riders} riders + {args.residents} residents ===")

    print("\n1. device")
    ok(torch.cuda.is_available(), "CUDA is available")
    if not torch.cuda.is_available():
        print("cannot continue without a GPU")
        return 2
    name = torch.cuda.get_device_name(0)
    hbm = torch.cuda.get_device_properties(0).total_memory
    print(f"      {name}, {hbm / 1e9:.0f} GB HBM")

    print("\n2. interconnect (MEASURED, not assumed)")
    gbps = measure_h2d_gbps()
    print(f"      pinned H2D: {gbps:.1f} GB/s")
    is_c2c = gbps > 200
    ok(gbps > 5.0, "link bandwidth is plausible", f"{gbps:.1f} GB/s")
    if is_c2c:
        ok(True, "looks like NVLink-C2C", f"{gbps:.0f} GB/s")
    else:
        ok(False, "this is a PCIe-class link, not C2C",
           f"{gbps:.1f} GB/s -- expected >200 on GH200", warn_only=True)
    print(f"\n      >>> pass --hybrid-pcie-gbps {gbps:.1f} to every harness, or "
          f"export VLLM_KVPIPE_PCIE_GBPS={gbps:.1f}")
    print("      (the harnesses default to 25.1, which on C2C would make B1 "
          "admit ~18x too few riders)")

    print("\n3. KV geometry")
    try:
        b_tok, geo = kv_bytes_per_token(args.model)
    except Exception as e:                                    # noqa: BLE001
        print(f"  [FAIL] could not read the model config: {e}")
        return 2
    print(f"      {geo['layers']}L x {geo['kv_heads']}kv x {geo['head_dim']}d "
          f"x 2 (K,V) x 2B = {b_tok} bytes/token")
    ok(b_tok > 0, "bytes/token derived from the model")
    ok(b_tok != 131072 or geo["layers"] == 32,
       "bytes/token is not the stale 131072 for a non-32-layer model")

    print("\n4. the rider budget this implies")
    per_rider = args.ctx * b_tok
    budget = (args.step_ms / 1000.0) * gbps * 1e9 * 0.9
    allowed = int(budget // per_rider)
    unoverlapped_ms = per_rider / gbps / 1e9 * 1000
    print(f"      a rider at ctx {args.ctx} is {per_rider / 1e9:.3f} GB/step "
          f"= {unoverlapped_ms:.2f} ms unoverlapped")
    print(f"      B1 budget at a {args.step_ms:.0f} ms step: "
          f"{budget / 1e9:.2f} GB  ->  {allowed} riders")
    ok(allowed >= args.target_riders,
       f"B1 will admit the {args.target_riders} riders being targeted",
       f"it will admit {allowed}")

    print("\n5. host memory for the pinned mirrors")
    total_reqs = args.target_riders + args.residents
    max_len = args.ctx + args.out_len + 64
    bpr = max(1, math.ceil(max_len / args.block_size))
    h_max = total_reqs * bpr + 8
    mirror_bytes = h_max * args.block_size * b_tok
    print(f"      h_max = {total_reqs} reqs x {bpr} blocks + 8 = {h_max} blocks")
    print(f"      pinned mirrors (K+V, all layers): {mirror_bytes / 1e9:.1f} GB")
    try:
        avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except Exception:                                          # noqa: BLE001
        avail = 0
    if avail:
        print(f"      host memory available: {avail / 1e9:.0f} GB")
        ok(mirror_bytes < avail * 0.6,
           "pinned mirrors fit comfortably in host memory",
           f"{mirror_bytes / 1e9:.1f} GB of {avail / 1e9:.0f} GB")
    cg = "/sys/fs/cgroup/memory.max"
    if os.path.exists(cg):
        try:
            raw = open(cg).read().strip()
            if raw != "max":
                lim = int(raw)
                print(f"      cgroup memory limit: {lim / 1e9:.0f} GB")
                ok(mirror_bytes < lim * 0.6,
                   "pinned mirrors fit inside the cgroup limit",
                   f"{mirror_bytes / 1e9:.1f} GB of {lim / 1e9:.0f} GB")
        except Exception:                                      # noqa: BLE001
            pass

    print("\n6. HBM for the rolling window")
    # w_max is allocated equal to h_max today, k_window slots of K and V.
    rolling = 4 * h_max * args.block_size * b_tok / 64 * 2
    print(f"      k_window=4 slots: ~{rolling / 1e9:.1f} GB "
          f"(w_max is allocated == h_max)")
    ok(rolling < hbm * 0.15, "rolling slots are a small share of HBM",
       f"{rolling / 1e9:.1f} GB of {hbm / 1e9:.0f} GB", warn_only=True)

    print("\n7. does the offered load actually BIND? (vacuity check)")
    # The single most expensive way to waste a rental: run the 24 GPU-h
    # decision experiment at an offered load the all-HBM arm can absorb. Then
    # HBM capacity was never the binding constraint, no arm can differ, and
    # endpoint_comparison's vacuity guard reports it -- AFTER the hours are
    # spent. It is arithmetic, so do it now.
    hbm_gb = args.hbm_gb or hbm / 1e9
    kv_gb = hbm_gb * args.gpu_mem_util - args.weights_gb
    if kv_gb <= 0:
        ok(False, "the weights leave room for a KV pool",
           f"{args.weights_gb} GB of weights vs {hbm_gb:.0f} GB HBM")
    else:
        cap = int(kv_gb * 1e9 // per_rider)
        print(f"      {hbm_gb:.0f} GB HBM x {args.gpu_mem_util} util - "
              f"{args.weights_gb} GB weights = {kv_gb:.1f} GB of KV space")
        print(f"      all-HBM capacity at ctx {args.ctx}: ~{cap} concurrent")
        rec = max(2 * cap, cap + 4)
        print(f"      >>> run 4.1 with --offered >= {rec} "
              f"(~2x capacity) so all-HBM saturates BELOW the offered load")
        ok(cap >= 2, "all-HBM capacity is big enough to compare against",
           f"{cap} concurrent", warn_only=True)
        if args.planned_offered:
            ok(args.planned_offered > cap * 1.05,
               f"the planned --offered {args.planned_offered} will bind",
               f"all-HBM holds ~{cap}; offered must exceed it or the run is "
               "VACUOUS")

    print("\n8. flags to use on this machine")
    print(f"      --hybrid-pcie-gbps {gbps:.1f}")
    print(f"      --max-num-seqs {total_reqs}   (>= residents + riders)")
    print(f"      --hybrid-b-hbm-max {args.residents}")
    print("      --hybrid-b-tok  : DO NOT PASS (derived from the model now)")
    print(f"      VLLM_KVPIPE_RUNGD_HMAX={h_max}  (only if the default is wrong)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"device": name, "hbm_bytes": hbm, "h2d_gbps": gbps,
                       "b_tok": b_tok, "geometry": geo, "riders_allowed": allowed,
                       "h_max": h_max, "mirror_bytes": mirror_bytes,
                       "fail": FAIL, "warn": WARN}, f, indent=2)

    print("\n" + "=" * 72)
    if FAIL:
        print(f"NOT READY ({len(FAIL)}): " + "; ".join(FAIL))
        return 1
    if WARN:
        print(f"READY, with warnings: " + "; ".join(WARN))
        return 0
    print("READY: link measured, KV geometry derived, budget and memory check out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
