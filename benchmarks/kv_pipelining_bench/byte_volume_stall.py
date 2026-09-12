#!/usr/bin/env python3
"""Does the all_host stall live in the MIXTURE path too, or only in all_host?

`GH200_EXPERIMENT_PLAN.md` section 2b: Qwen-32B, 20 requests x 1024 ctx on A100,
`all_host` did not finish in 900 s at ~0% GPU. The arithmetic says ~216 ms/step
(5.37 GB/step over ~25 GB/s), so ~14 s -- a >60x gap, i.e. a bug.

Ben's question was why debug an endpoint nobody would deploy. The answer is
that the stall is not an all_host property. `_issue_runs` and the rolling
window are SHARED with the mixture arm, and the load that drives them is BYTES
PER STEP, which is riders x ctx x 0.26 MB regardless of which arm produced
them. The numbers that matter:

    my clean A100 sweep     16 riders x  256 ctx =  1.07 GB/step   fine
    the reported stall      20 riders x 1024 ctx =  5.37 GB/step   hung
    C2C mixture @8K         ~7 riders x 8192 ctx = 15.03 GB/step   never run
    C2C mixture @2K        ~28 riders x 2048 ctx = 15.03 GB/step   never run

So the rental's mixture arm sits at ~2.8x the per-step volume that already
hangs here, and the mixture has never been run above 3 riders in a real
configuration -- the 16-rider sweep was a deliberate small-context proxy at a
fifth the volume.

This walks BOTH arms up the byte-volume axis on the same engine, so the
question "is it the arm or the volume?" is answered by data rather than
argument:

  * if `mixture` degrades at the same GB/step as `all_host`, the defect is in
    the shared path and MUST be fixed before the rental;
  * if only `all_host` degrades, it is arm-specific, and dropping that arm (as
    the plan now does) is sufficient.

Time-boxed on purpose: a per-config timeout, and the sweep stops at the first
configuration that exceeds it, because the whole point is to find the knee, not
to reproduce a 900 s hang repeatedly.

Run: srun ... .venv/bin/python benchmarks/kv_pipelining_bench/byte_volume_stall.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_teardown import terminate_engine  # noqa: E402

ROOT = "/mnt/beegfs/benishay/vllm-kvpipeline-66"
PY_BIN = f"{ROOT}/.venv/bin/python"
LOGDIR = "/mnt/beegfs/benishay/paper_runs/20260911"
B_TOK = 262144          # Qwen-32B KV bytes/token


def launch(port: int, arm: str, ctx: int, n: int, args) -> subprocess.Popen:
    env = dict(os.environ)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    env["PYTHONUNBUFFERED"] = "1"
    for k in ("VLLM_KVPIPE_HYBRID_RUNGD", "VLLM_KVPIPE_HYBRID_RUNGD_FREE",
              "VLLM_KVPIPE_HYBRID_TWOPASS", "VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM",
              "VLLM_KVPIPE_RUNGD_ASYNC", "VLLM_KVPIPE_RUNGD_PREFILL_STASH"):
        env.pop(k, None)
    cmd = [PY_BIN, "-m", "vllm.entrypoints.openai.api_server",
           "--model", args.model, "--port", str(port),
           "--max-model-len", str(ctx + args.out_len + 64),
           "--gpu-memory-utilization", str(args.gpu_mem_util),
           "--dtype", "float16", "--max-num-seqs", str(n),
           "--enforce-eager", "--no-enable-prefix-caching",
           "--disable-log-requests"]
    env["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
    env["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
    env["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
    env["VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM"] = "1"
    env["VLLM_KVPIPE_RUNGD_ASYNC"] = "1"
    env["VLLM_KVPIPE_RUNGD_PREFILL_STASH"] = "1"
    # all_host: every request on the host tier. mixture: a resident cohort with
    # riders on top -- the arm the rental actually depends on.
    b_hbm = 0 if arm == "all_host" else args.residents
    cmd += ["--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
            "--hybrid-b-hbm-max", str(b_hbm),
            "--hybrid-pcie-gbps", "25.2",
            # Budget off: this asks whether the MECHANISM survives the byte
            # volume, not whether the policy would have allowed it.
            "--hybrid-itl-ms", "1000000",
            "--enable-kv-pipelining", "--kv-k-window", "4"]
    os.makedirs(LOGDIR, exist_ok=True)
    logf = open(f"{LOGDIR}/stall_{arm}_{ctx}_{n}.log", "w")
    return subprocess.Popen(cmd, env=env, stdout=logf,
                            stderr=subprocess.STDOUT, preexec_fn=os.setsid)


def wait_ready(port: int, timeout: int) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def run_one(port: int, arm: str, ctx: int, n: int, args) -> dict:
    gb = n * ctx * B_TOK / 1e9
    print(f"  {arm:<9} {n:>3} x {ctx:>5} = {gb:>6.2f} GB/step ... ",
          end="", flush=True)
    proc = launch(port, arm, ctx, n, args)
    out = {"arm": arm, "ctx": ctx, "n": n, "gb_per_step": round(gb, 2)}
    try:
        if not wait_ready(port, args.startup_timeout):
            out["result"] = "server-did-not-start"
            print("server did not start")
            return out
        import threading
        done, errs = [], []
        lock = threading.Lock()

        def one(i: int) -> None:
            body = json.dumps({
                "model": args.model,
                "prompt": [1000 + i + t for t in range(ctx)],
                "max_tokens": args.out_len, "temperature": 0.0,
                "ignore_eos": True, "min_tokens": args.out_len,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/completions", data=body,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=args.per_config_s) as r:
                    r.read()
                with lock:
                    done.append(i)
            except Exception as e:                            # noqa: BLE001
                with lock:
                    errs.append(f"{type(e).__name__}")

        threads = [threading.Thread(target=one, args=(i,), daemon=True)
                   for i in range(n)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=args.per_config_s)
        wall = time.time() - t0
        out.update(completed=len(done), errors=len(errs),
                   wall_s=round(wall, 1))
        # What SHOULD it cost? bytes/step over the link, times the steps.
        ideal = gb / 25.2 * args.out_len
        out["ideal_s"] = round(ideal, 1)
        out["slowdown_vs_link"] = round(wall / ideal, 1) if ideal else None
        if len(done) < n:
            out["result"] = "STALLED"
            print(f"STALLED ({len(done)}/{n} in {wall:.0f}s; link says "
                  f"~{ideal:.0f}s)")
        else:
            out["result"] = "ok"
            print(f"ok {wall:.0f}s (link says ~{ideal:.0f}s, "
                  f"{out['slowdown_vs_link']}x)")
        return out
    finally:
        terminate_engine(proc, label=f"{arm}-{ctx}-{n}")
        time.sleep(3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--out-len", type=int, default=16)
    ap.add_argument("--residents", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.83)
    ap.add_argument("--startup-timeout", type=int, default=900)
    ap.add_argument("--per-config-s", type=int, default=420,
                    help="time box per configuration; the sweep stops an arm "
                         "after its first breach, since the knee is the "
                         "finding, not the hang")
    ap.add_argument("--configs", default="4x1024,8x1024,12x1024,20x1024,8x2048",
                    help="n x ctx, ascending in bytes/step")
    ap.add_argument("--arms", default="mixture,all_host")
    ap.add_argument("--out", default=f"{LOGDIR}/byte_volume_stall.json")
    args = ap.parse_args()

    cfgs = []
    for c in args.configs.split(","):
        n, ctx = c.lower().split("x")
        cfgs.append((int(n), int(ctx)))
    cfgs.sort(key=lambda t: t[0] * t[1])

    print(f"model={args.model} out_len={args.out_len} "
          f"residents(mixture)={args.residents} timebox={args.per_config_s}s")
    rows, port = [], 9930
    for arm in args.arms.split(","):
        print(f"=== {arm} ===")
        for n, ctx in cfgs:
            r = run_one(port, arm, ctx, n, args)
            port += 1
            rows.append(r)
            if r.get("result") in ("STALLED", "server-did-not-start"):
                print(f"  -> {arm} breached at {r['gb_per_step']} GB/step; "
                      "stopping this arm")
                break

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2)

    knee = {}
    for arm in args.arms.split(","):
        bad = [r for r in rows if r["arm"] == arm and r["result"] != "ok"]
        good = [r for r in rows if r["arm"] == arm and r["result"] == "ok"]
        knee[arm] = {"last_ok_gb": max((r["gb_per_step"] for r in good),
                                       default=None),
                     "first_bad_gb": min((r["gb_per_step"] for r in bad),
                                         default=None)}
    print("\n" + "=" * 70)
    for arm, k in knee.items():
        print(f"{arm:<9} last OK {k['last_ok_gb']} GB/step, "
              f"first STALL {k['first_bad_gb']} GB/step")
    m, a = knee.get("mixture", {}), knee.get("all_host", {})
    if m.get("first_bad_gb") is not None:
        print("\nTHE MIXTURE STALLS TOO. The defect is in the shared rolling/"
              "_issue_runs path, not in the all_host configuration, and the "
              "rental's mixture arm runs at ~15 GB/step -- above this knee. "
              "MUST be fixed before renting.")
        return 1
    if a.get("first_bad_gb") is not None:
        print("\nOnly all_host stalls; the mixture is clean at every volume "
              "tested. Dropping the all_host arm is sufficient -- but note the "
              "mixture was only tested to "
              f"{m.get('last_ok_gb')} GB/step, and C2C runs it at ~15.")
        return 0
    print("\nNeither arm stalled at the volumes tested. The 2026-08-12 report "
          "does not reproduce on current code -- likely already fixed by the "
          "loop-invariant hoists. Re-check the original config before "
          "trusting this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
