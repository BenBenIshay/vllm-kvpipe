#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""C2 correctness control: baseline-vs-baseline determinism at a given concurrency.

Sends the validate_qa.py prompt set to ONE standard vLLM server twice and reports
the run-to-run exact-match rate. If exact-match at conc>1 is well below 100%, the
divergence is inherent batched-execution FP nondeterminism in vanilla vLLM, not a
KV-pipelining bug (which is what validate_qa.py's conc=8 baseline-vs-pipelined gap
would otherwise be misread as).

  python qa_nondeterminism_control.py --model Qwen/Qwen2.5-0.5B-Instruct \
      --concurrency 8 --max-tokens 256 --port 8650
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_BENCH_DIR = _THIS.parent
_REPO_ROOT = _BENCH_DIR.parent.parent
sys.path.insert(0, str(_BENCH_DIR))

from validate_qa import QA_PROMPTS, send_all_requests  # noqa: E402


def _wait(port: int, timeout: int) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(2)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--port", type=int, default=8650)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    ap.add_argument("--startup-timeout", type=int, default=600)
    args = ap.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("HF_HUB_CACHE", "/data/ben/huggingface")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model, "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-num-seqs", str(args.concurrency),
        "--enforce-eager", "--disable-log-requests", "--dtype", "auto",
    ]
    log = open("/tmp/qa_control_server.log", "w")
    print(f"launching baseline server on :{args.port} ...", flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid, env=env, cwd=str(_REPO_ROOT))
    try:
        if not _wait(args.port, args.startup_timeout):
            print("server failed to start; see /tmp/qa_control_server.log")
            return
        print("server up; running two identical conc="
              f"{args.concurrency} passes ...", flush=True)
        run_a = asyncio.run(send_all_requests(
            args.port, args.model, QA_PROMPTS, args.max_tokens, args.concurrency))
        time.sleep(2)
        run_b = asyncio.run(send_all_requests(
            args.port, args.model, QA_PROMPTS, args.max_tokens, args.concurrency))
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=20)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass

    a = {r.prompt_id: r.response for r in run_a}
    b = {r.prompt_id: r.response for r in run_b}
    exact = 0
    sims = []
    diverging = []
    for pid in a:
        ra, rb = a[pid], b.get(pid, "")
        sim = difflib.SequenceMatcher(None, ra, rb).ratio()
        sims.append(sim)
        if ra == rb:
            exact += 1
        else:
            diverging.append((pid, sim))
    n = len(a)
    print("\n==================== CONTROL RESULT ====================")
    print(f"model={args.model}  conc={args.concurrency}  prompts={n}")
    print(f"baseline-vs-baseline EXACT match: {exact}/{n} "
          f"({100.0 * exact / n:.1f}%)")
    print(f"avg similarity: {sum(sims) / len(sims):.4f}   "
          f"min similarity: {min(sims):.4f}")
    if diverging:
        print("diverging prompts (run A vs run B, both baseline):")
        for pid, sim in sorted(diverging, key=lambda x: x[1]):
            print(f"  {pid:<12} sim={sim:.3f}")
    print("=======================================================")
    print("Interpretation: a sub-100% exact rate here means vanilla vLLM is "
          "itself nondeterministic at this concurrency (batched-GEMM/attention "
          "reduction order), so the validate_qa.py conc>1 baseline-vs-pipelined "
          "gap is NOT a pipelining correctness bug.")


if __name__ == "__main__":
    main()
