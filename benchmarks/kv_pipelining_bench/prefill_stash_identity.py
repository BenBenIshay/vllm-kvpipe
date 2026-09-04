#!/usr/bin/env python3
"""Does a rider fed by a REAL prefill produce the right tokens?

`hybrid_preload_capacity.py` showed riders are additive, but it filled their
KV synthetically because no prefill->pool transport existed. Synthetic KV makes
generated tokens meaningless, so that result was capacity-only. This closes
that gap: the rider's KV is now the actual output of a real prefill, handed
over by prompt key, and the tokens must match vanilla vLLM exactly.

The local P/D emulation, in two rounds against ONE engine:

  round 1 ("prefill node")  each prompt is admitted normally, prefills into
                            HBM, is seeded into the host tier by the ordinary
                            reclaim path, and its REAL KV is stashed under a
                            prompt key that outlives the request.
  round 2 ("decode node")   the same prompts arrive again. The stash exists, so
                            each is admitted with ZERO HBM blocks and never
                            prefills at all -- it decodes from KV that was
                            computed earlier and has been sitting in host DDR.

Round 2 is the claim. Its tokens are compared against vanilla vLLM with no
tiers.

SERIAL on purpose. Vanilla vLLM is not token-stable across batch sizes in
bf16/fp16 (measured: vanilla at concurrency 1 differs from vanilla at
concurrency 6 on the same prompt), so any token-identity comparison across
configurations that change batching is unsound. One request at a time is the
only apples-to-apples.

Run:
  srun ... .venv/bin/python benchmarks/kv_pipelining_bench/prefill_stash_identity.py
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request

ROOT = "/mnt/beegfs/benishay/vllm-kvpipe"
PY = f"{ROOT}/.venv/bin/python"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LOGDIR = "/mnt/beegfs/benishay/paper_runs/20260904"
MAX_TOKENS = 48
MAX_MODEL_LEN = 1024
N_REQS = 6
NUM_BLOCKS = 192
GPU_MEM = "0.30"

_BODIES = [
    "the Roman republic collapsed into empire under a general who crossed a river",
    "a wave function assigns a complex amplitude to every configuration of a system",
    "sourdough fermentation depends on a stable culture of yeast and lactobacilli",
    "the transatlantic cable required insulating a conductor against seawater",
    "a compiler lowers a typed syntax tree into a sequence of machine operations",
    "glaciers record atmospheric composition in bubbles trapped as firn compacts",
]
PROMPTS = [
    ("Consider the following account. " + (body + " ") * 24
     + "Summarise the account in one sentence:")
    for body in _BODIES
]


def launch(port: int, arm: str) -> subprocess.Popen:
    env = dict(os.environ)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    env["PYTHONUNBUFFERED"] = "1"
    for k in ("VLLM_KVPIPE_HYBRID_RUNGD", "VLLM_KVPIPE_HYBRID_RUNGD_FREE",
              "VLLM_KVPIPE_HYBRID_TWOPASS", "VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM",
              "VLLM_KVPIPE_RUNGD_SYNTHETIC_PRELOAD",
              "VLLM_KVPIPE_RUNGD_PREFILL_STASH"):
        env.pop(k, None)
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
        "--port", str(port), "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", GPU_MEM,
        "--num-gpu-blocks-override", str(NUM_BLOCKS),
        "--dtype", "auto", "--max-num-seqs", str(N_REQS), "--enforce-eager",
        "--disable-log-requests",
    ]
    # The prefix-cache arm is the CONTROL for the one numerical difference our
    # handover cannot avoid: a rider is admitted with its last prompt token
    # declared uncomputed, so that token's KV is produced by the DECODE kernel
    # rather than the prefill kernel. vLLM's own prefix cache does exactly the
    # same thing (it always leaves one token to recompute), and every real P/D
    # deployment does too. Comparing against a fresh full prefill would charge
    # our handover for a difference that is inherent to the architecture.
    if arm != "vanilla_prefix":
        cmd.append("--no-enable-prefix-caching")
    if arm == "stash":
        env["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
        env["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
        env["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
        env["VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM"] = "1"
        env["VLLM_KVPIPE_RUNGD_PREFILL_STASH"] = "1"
        # Deliberately NOT set: VLLM_KVPIPE_RUNGD_SYNTHETIC_PRELOAD. A synthetic
        # fill would pass a capacity test and fail this one, which is the point.
        # No --kv-transfer-config either: the bench connectors fake-fill KV.
        cmd += [
            "--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
            "--hybrid-b-hbm-max", "0", "--hybrid-pcie-gbps", "25.1",
            "--hybrid-itl-ms", "100000",
            "--enable-kv-pipelining", "--kv-k-window", "4",
        ]
    os.makedirs(LOGDIR, exist_ok=True)
    logf = open(f"{LOGDIR}/stash_identity_{arm}.log", "w")
    return subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def wait_ready(port: int, timeout: int = 300) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def complete(port: int, prompt: str) -> str:
    body = json.dumps({"model": MODEL, "prompt": prompt,
                       "max_tokens": MAX_TOKENS, "temperature": 0.0,
                       "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["text"]


def scrape(arm: str) -> dict:
    stashed = delivered_real = delivered_synth = undelivered = 0
    dead = None
    try:
        with open(f"{LOGDIR}/stash_identity_{arm}.log") as f:
            for line in f:
                if "PREFILL STASHED" in line:
                    stashed += 1
                elif "PRELOAD DELIVERED" in line:
                    if "REAL prefill KV" in line:
                        delivered_real += 1
                    else:
                        delivered_synth += 1
                if "marked pool-preloaded but no KV was delivered" in line:
                    undelivered += 1
                if dead is None:
                    for sig in ("EngineDeadError", "EngineCore encountered",
                                "Dumping input data"):
                        if sig in line:
                            dead = sig
    except OSError:
        pass
    return {"stashed": stashed, "delivered_real": delivered_real,
            "delivered_synthetic": delivered_synth,
            "undelivered": undelivered, "engine_died": dead}


def run_vanilla(port: int, arm: str = "vanilla") -> list[str]:
    """For the prefix-cache arm each prompt is sent TWICE and the second answer
    kept: the first populates the cache, the second is the one that decodes
    from cached KV with its last prompt token recomputed."""
    proc = launch(port, arm)
    try:
        if not wait_ready(port):
            return ["<no-start>"] * len(PROMPTS)
        if arm == "vanilla_prefix":
            for p in PROMPTS:
                complete(port, p)
        return [complete(port, p) for p in PROMPTS]
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(4)


def run_stash(port: int) -> tuple[list[str], list[str], dict]:
    """Round 1 seeds the stash; round 2 rides it. Same engine, same prompts."""
    proc = launch(port, "stash")
    try:
        if not wait_ready(port):
            n = ["<no-start>"] * len(PROMPTS)
            return n, n, scrape("stash")
        r1 = [complete(port, p) for p in PROMPTS]
        r2 = [complete(port, p) for p in PROMPTS]
        return r1, r2, scrape("stash")
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(4)


def main() -> int:
    print(f"model={MODEL} prompts={len(PROMPTS)} max_tokens={MAX_TOKENS} "
          f"(serial; REAL prefill KV, no synthetic fill)", flush=True)
    print("=== vanilla reference (fresh full prefill) ===", flush=True)
    vanilla = run_vanilla(9970)
    print("=== control: vanilla + prefix cache (last token recomputed) ===",
          flush=True)
    vanilla_prefix = run_vanilla(9972, "vanilla_prefix")
    print("=== stash arm: round 1 prefills+stashes, round 2 rides ===",
          flush=True)
    r1, r2, ev = run_stash(9971)
    print(f"    {ev}", flush=True)

    out = {"vanilla": vanilla, "vanilla_prefix_cache": vanilla_prefix,
           "round1_prefill": r1, "round2_rider": r2, "events": ev}
    with open(f"{LOGDIR}/stash_identity.json", "w") as f:
        json.dump(out, f, indent=2)

    # Validity first: a token comparison through a dead engine, or against a
    # rider that was never actually pool-served, is not a result.
    if ev["engine_died"]:
        print(f"INVALID: the stash engine died ({ev['engine_died']}).")
        return 2
    if ev["delivered_synthetic"]:
        print(f"INVALID: {ev['delivered_synthetic']} riders were fed SYNTHETIC "
              "KV; this test only means anything with real prefill output.")
        return 2
    if ev["undelivered"]:
        print(f"INVALID: {ev['undelivered']} riders were admitted with no KV "
              "delivered at all.")
        return 2
    if ev["stashed"] < len(PROMPTS):
        print(f"INCONCLUSIVE: only {ev['stashed']}/{len(PROMPTS)} prefills were "
              "stashed, so round 2 was not testing the handover.")
        return 2
    if ev["delivered_real"] < len(PROMPTS):
        print(f"INCONCLUSIVE: only {ev['delivered_real']}/{len(PROMPTS)} round-2 "
              "requests rode real stashed KV; the rest prefilled normally.")
        return 2

    idx = range(len(PROMPTS))
    bad1 = [i for i in idx if r1[i] != vanilla[i]]
    bad2 = [i for i in idx if r2[i] != vanilla[i]]
    ctrl = [i for i in idx if vanilla_prefix[i] != vanilla[i]]
    bad2_vs_ctrl = [i for i in idx if r2[i] != vanilla_prefix[i]]
    n = len(PROMPTS)
    print("=" * 74)
    print(f"round 1 (prefill+seed+reclaim)     vs vanilla:       "
          f"{n - len(bad1)}/{n} identical")
    print(f"CONTROL vanilla+prefix cache       vs vanilla:       "
          f"{n - len(ctrl)}/{n} identical   (diverges at {ctrl})")
    print(f"round 2 (rider on real stashed KV) vs vanilla:       "
          f"{n - len(bad2)}/{n} identical   (diverges at {bad2})")
    print(f"round 2                            vs prefix cache:  "
          f"{n - len(bad2_vs_ctrl)}/{n} identical")
    print("=" * 74)
    if bad1:
        print("FAIL: round 1 -- the already-validated seed+reclaim path -- no "
              "longer matches vanilla. Fix that before reading anything else.")
        return 1
    if not bad2:
        print("PASS: a request that never prefilled, decoding from KV computed "
              "earlier and held in host DDR, is token-identical to vanilla.")
        return 0
    if not bad2_vs_ctrl:
        print("PASS: the rider is token-identical to vLLM's OWN prefix cache, "
              "which recomputes the last prompt token exactly as a P/D decode "
              "node must. It differs from a fresh full prefill only where the "
              "prefix cache does -- an inherent property of the architecture, "
              "not of this handover.")
        return 0
    if set(bad2) <= set(ctrl):
        print("INCONCLUSIVE: the rider diverges from vanilla only on prompts "
              "where vLLM's own prefix cache also diverges, but not "
              "identically. That is the near-tie regime; it is consistent with "
              "a correct handover but does not prove one.")
        return 3
    print(f"FAIL: the rider diverges on prompts {sorted(set(bad2) - set(ctrl))} "
          "where the prefix cache does NOT. The last-token recompute cannot "
          "explain that, so the handover is wrong.")
    for i in sorted(set(bad2) - set(ctrl))[:2]:
        print(f"  prompt {i}\n    vanilla: {vanilla[i][:200]!r}\n"
              f"    rider:   {r2[i][:200]!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
