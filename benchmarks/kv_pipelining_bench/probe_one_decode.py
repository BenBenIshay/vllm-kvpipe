#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One streaming request against real_prefill server; for pipeline debug logs."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))
from hf_env import apply_hf_token_from_user_file  # noqa: E402


def _wait_health(port: int, timeout: int) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


def _launch(
    model: str,
    port: int,
    k_window: int,
    log: Path,
    gpu_mem: float,
    *,
    pipeline_debug_log: bool = False,
) -> subprocess.Popen:
    kv = {"kv_connector": "PrefillStagingPipelinedConnector", "kv_role": "kv_both"}
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--max-model-len", "4096",
        "--gpu-memory-utilization", str(gpu_mem),
        "--max-num-seqs", "8",
        "--disable-log-requests",
        "--dtype", "auto",
        "--enforce-eager",
        "--enable-kv-pipelining",
        "--kv-k-window", str(k_window),
        "--free-original-kv-cache",
        "--kv-transfer-config", json.dumps(kv),
        "--no-enable-prefix-caching",
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w")
    fh.write("# " + " ".join(cmd) + "\n")
    fh.flush()
    env = os.environ.copy()
    if pipeline_debug_log:
        env["VLLM_PIPELINE_DEBUG_LOG"] = "1"
    env.setdefault("HF_HUB_CACHE", "/data/ben/huggingface")
    return subprocess.Popen(
        cmd, stdout=fh, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env,
    )


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def _stream_once(port: int, model: str, prompt: str, max_tokens: int) -> None:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        for _ in resp:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--port", type=int, default=8225)
    ap.add_argument("--k-window", type=int, default=4)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=32)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--startup-timeout", type=int, default=300)
    ap.add_argument(
        "--pipeline-debug-log",
        action="store_true",
        help="Set VLLM_PIPELINE_DEBUG_LOG=1 (very verbose; avoid with nsys).",
    )
    args = ap.parse_args()

    apply_hf_token_from_user_file()
    prompt = "x " * max(1, args.input_len)
    proc = _launch(
        args.model,
        args.port,
        args.k_window,
        args.log,
        args.gpu_memory_utilization,
        pipeline_debug_log=args.pipeline_debug_log,
    )
    try:
        if not _wait_health(args.port, args.startup_timeout):
            raise RuntimeError(f"server did not start; see {args.log}")
        _stream_once(args.port, args.model, prompt, args.output_len)
    finally:
        _kill(proc)
    print(f"Wrote {args.log}")


if __name__ == "__main__":
    main()
