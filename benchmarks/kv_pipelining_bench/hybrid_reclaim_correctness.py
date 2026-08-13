#!/usr/bin/env python3
"""Token-level correctness for the prefill reclaim, in the shape that matters.

`hybrid_rungd_real.py` already checks the pool against vanilla, but with ~10
token prompts: one block per request, so the reclaim frees one block and a
stale-block read would very likely go unnoticed. The blocker
(paper/BLOCKER_PREFILL_RESIDENCY.md) is precisely about LONG PREFILLS -- the
case where the pre-fix implementation kept 89% of a request's KV resident.

So this test is built to break the reclaim if it is breakable:

  * long prompts (tens of blocks each), so the reclaim frees most of the pool;
  * a deliberately SMALL KV pool (--num-gpu-blocks-override), so the freed
    blocks are immediately handed to other requests and overwritten. If the
    host tier were still reading its resident blocks, or writing into them,
    the output would be garbage;
  * every request concurrent and host-tier (b_hbm_max=0);
  * greedy decoding, compared token-for-token against vanilla vLLM with no
    tiers at all.

Reclaim is verified to have actually fired (the engine log records it) -- a
pass with zero reclaims would be vacuous.

Run:
  srun ... .venv/bin/python benchmarks/kv_pipelining_bench/hybrid_reclaim_correctness.py
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = "/mnt/beegfs/benishay/vllm-kvpipe"
PY = f"{ROOT}/.venv/bin/python"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LOGDIR = "/mnt/beegfs/benishay/paper_runs/20260813"
MAX_TOKENS = 48
MAX_MODEL_LEN = 1024
N_REQS = 6
# Small enough that the resident pool cannot hold every request's prefill, so
# reclaimed blocks are genuinely recycled while their owners keep decoding.
NUM_BLOCKS = 192
GPU_MEM = "0.30"

# ~55 tokens of distinct lead-in, then a long shared-shape body, repeated to
# push each prompt to tens of blocks.
_BODIES = [
    "the Roman republic collapsed into empire under a general who crossed a river",
    "a wave function assigns a complex amplitude to every configuration of a system",
    "sourdough fermentation depends on a stable culture of yeast and lactobacilli",
    "the transatlantic cable required insulating a conductor against seawater",
    "glacial moraines record the farthest advance of ice across a valley floor",
    "a compiler lowers a typed syntax tree into a sequence of machine operations",
]
PROMPTS = [
    ("Consider the following account. " + (body + " ") * 24
     + "Summarise the account in one sentence:")
    for body in _BODIES
]


def launch(port: int, reclaim: bool) -> subprocess.Popen:
    env = dict(os.environ)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/mnt/beegfs/benishay/huggingface")
    env["PYTHONUNBUFFERED"] = "1"
    for k in ("VLLM_KVPIPE_HYBRID_RUNGD", "VLLM_KVPIPE_HYBRID_RUNGD_FREE",
              "VLLM_KVPIPE_HYBRID_TWOPASS", "VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM"):
        env.pop(k, None)
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
        "--port", str(port), "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", GPU_MEM,
        "--num-gpu-blocks-override", str(NUM_BLOCKS),
        "--dtype", "auto", "--max-num-seqs", str(N_REQS), "--enforce-eager",
        "--no-enable-prefix-caching", "--disable-log-requests",
    ]
    if reclaim:
        env["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
        env["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
        env["VLLM_KVPIPE_HYBRID_TWOPASS"] = "1"
        env["VLLM_KVPIPE_HYBRID_RUNGD_RECLAIM"] = "1"
        # Tiers via admission only -- NO --kv-transfer-config, so prefill KV is
        # real (the bench connector fake-fills, which invalidates token tests).
        cmd += [
            "--enable-transfer-aware-admission", "--enable-hybrid-kv-tiers",
            "--hybrid-b-hbm-max", "0", "--hybrid-pcie-gbps", "25.1",
            "--hybrid-b-tok", "131072", "--hybrid-itl-ms", "100000",
            "--enable-kv-pipelining", "--kv-k-window", "4",
        ]
    os.makedirs(LOGDIR, exist_ok=True)
    logf = open(f"{LOGDIR}/reclaim_correctness_{port}.log", "w")
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
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": MAX_TOKENS,
                       "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["text"]


def run(port: int, reclaim: bool, serial: bool = False) -> list[str]:
    """serial=True sends the prompts one at a time, so each is generated in a
    batch of one. That is the only way to compare token-for-token across
    configurations that change the batching."""
    if serial:
        proc = launch(port, reclaim)
        try:
            if not wait_ready(port):
                return ["<no-start>"] * len(PROMPTS)
            return [complete(port, p) for p in PROMPTS]
        except Exception as e:                               # noqa: BLE001
            return [f"<err: {e}>"] * len(PROMPTS)
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            time.sleep(4)
    proc = launch(port, reclaim)
    try:
        if not wait_ready(port):
            return ["<no-start>"] * len(PROMPTS)
        with ThreadPoolExecutor(max_workers=len(PROMPTS)) as ex:
            return list(ex.map(lambda p: complete(port, p), PROMPTS))
    except Exception as e:                                   # noqa: BLE001
        return [f"<err: {e}>"] * len(PROMPTS)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(4)


def reclaim_stats(port: int) -> tuple[int, int]:
    """(number of reclaim events, total blocks reclaimed) from the engine log."""
    path = f"{LOGDIR}/reclaim_correctness_{port}.log"
    events, total = 0, 0
    try:
        with open(path) as f:
            for line in f:
                m = re.search(r"RECLAIMED (\d+) main-pool blocks", line)
                if m:
                    events += 1
                    total += int(m.group(1))
    except OSError:
        pass
    return events, total


def first_diff(a: str, b: str) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"char {i}: ...{a[max(0, i - 30):i + 30]!r} vs ...{b[max(0, i - 30):i + 30]!r}"
    return f"one is a prefix of the other (len {len(a)} vs {len(b)})"


def main() -> int:
    print(f"prompts: {len(PROMPTS)}, ~{len(PROMPTS[0].split())} words each, "
          f"pool = {NUM_BLOCKS} blocks", flush=True)
    # NEAR-TIE SCREEN. Greedy decoding is only reproducible for a fixed batch
    # composition: bf16 attention reductions depend on batch order, and a
    # prompt whose top-2 logits are close will flip on that alone. Stock vLLM
    # with no tiers at all was measured giving DIFFERENT text for prompt 1 at
    # concurrency 6 vs concurrency 1, while being perfectly reproducible at
    # each. Since the host tier necessarily changes the batching, comparing it
    # to vanilla-at-concurrency-6 is not a valid identity test for such a
    # prompt -- an earlier version of this script did exactly that and called a
    # near-tie a defect.
    #
    # So: screen each prompt by running vanilla at BOTH concurrencies. Prompts
    # that disagree with themselves are batching-sensitive and are excluded
    # from the identity gate (and reported, not hidden). The rest are compared
    # at full concurrency, which is what exercises the reclaim under load.
    print("=== VANILLA concurrent (no tiers) ===", flush=True)
    van_a = run(9941, reclaim=False)
    print("=== VANILLA serial (near-tie screen: batch of one) ===", flush=True)
    van_b = run(9943, reclaim=False, serial=True)
    print("=== HOST TIER + PREFILL RECLAIM (concurrent) ===", flush=True)
    rec = run(9942, reclaim=True)

    events, blocks = reclaim_stats(9942)
    with open(f"{LOGDIR}/reclaim_correctness_outputs.json", "w") as f:
        json.dump({"vanilla_concurrent": van_a, "vanilla_serial": van_b,
                   "reclaim": rec, "reclaim_events": events,
                   "blocks_reclaimed": blocks}, f, indent=2)

    print("=" * 70)
    bad = [t for t in van_a + van_b + rec if t.startswith("<")]
    if bad:
        print(f"INCONCLUSIVE: {bad[0]}")
        return 2

    # Batching-sensitive = vanilla's own answer changes with concurrency.
    sensitive = [i for i in range(len(PROMPTS)) if van_a[i] != van_b[i]]
    mism = [i for i in range(len(PROMPTS)) if van_a[i] != rec[i]]
    for i in range(len(PROMPTS)):
        if i in sensitive:
            print(f"[SKIP] prompt {i}: batching-sensitive in VANILLA itself "
                  "(concurrent != serial) -- cannot gate token identity on it")
            continue
        tag = "OK  " if van_a[i] == rec[i] else "DIFF"
        print(f"[{tag}] prompt {i}")
        if van_a[i] != rec[i]:
            print(f"       {first_diff(van_a[i], rec[i])}")
    print("=" * 70)
    print(f"reclaim events: {events}, blocks reclaimed: {blocks} "
          f"(pool is {NUM_BLOCKS} blocks)")
    print(f"near-tie screen: {len(sensitive)}/{len(PROMPTS)} prompts are "
          f"batching-sensitive in vanilla {sensitive}")

    if events == 0:
        print("VACUOUS: the reclaim never fired, so this run proves nothing "
              "about it. Check the gating env vars and the tier assignment.")
        return 2
    if events < len(PROMPTS):
        print(f"WARNING: only {events} of {len(PROMPTS)} requests were "
              "reclaimed; the rest never reached the host tier.")

    gated = [i for i in range(len(PROMPTS)) if i not in sensitive]
    if not gated:
        print("VACUOUS: every prompt is batching-sensitive; nothing was gated.")
        return 2
    real = [i for i in mism if i not in sensitive]
    if real:
        print(f"RECLAIM CORRECTNESS FAIL: {len(real)}/{len(gated)} gateable "
              "prompts diverged -- a genuine defect.")
        return 1
    print(f"RECLAIM CORRECTNESS PASS: {len(gated)}/{len(PROMPTS)} prompts are "
          "batching-stable, and on every one of them the host tier -- long "
          "prefills, small pool, freed blocks recycled under load -- generates "
          "tokens IDENTICAL to vanilla vLLM.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
