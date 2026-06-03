#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end demonstration of the SLO-aware admission policy.

Runs one or more arms back-to-back (stock / policy / mechanistic), all with
layer-KV pipelining feature ON,
both at the same Poisson arrival rate well above sustained capacity:

  arm = "stock"  : --enable-transfer-aware-admission off
                   admits everything; expect SLO blow-out.
  arm = "policy" : --enable-transfer-aware-admission on
                   --transfer-aware-slo-p99-itl-ms <SLO>
                   --transfer-aware-predictor-csv <feature_on_k_sweep.csv>
                   refuses requests whose post-admission predicted p99 ITL
                   would exceed the SLO; expect higher SLO compliance.

Per arm the script collects:
  - per-request TTFT and per-token ITL distributions
  - completed / refused counts (refusals show up as 503 / connection errors)
  - SLO-compliant tokens (= sum of output_tokens for requests whose p99 ITL
    is <= the SLO budget)
  - server-side admission skip counters via /v1/transfer_aware_stats (if
    the endpoint is available; otherwise inferred from refusal rate).

Usage::

    python bench_admission_policy_e2e.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --predictor-csv /data/.../feature_on_k_sweep.csv \
        --slo-p99-itl-ms 200 \
        --request-rate 8 \
        --num-prompts 200 \
        --input-len 1024 --output-len 256 \
        --out /data/.../admission_policy_e2e
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve().parent
_DECODE_ONLY_BENCH = _THIS / "decode_only_bench.py"


def _free_port(start: int = 8300, attempts: int = 50) -> int:
    for off in range(attempts):
        port = start + off
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port")


def _wait_for_server(port: int, timeout: int) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(2)
    return False


def _launch_server(
    *,
    python: str,
    model: str,
    port: int,
    max_model_len: int,
    gpu_mem_util: float,
    max_num_seqs: int,
    k_window: int,
    log_path: Path,
    enable_admission: bool,
    slo_p99_itl_ms: float,
    predictor_csv: str,
    predictor_model: str,
    admission_policy: str,
    baseline_compute_json: str,
    injection_jsonl: str,
    extra_env: dict[str, str] | None,
) -> subprocess.Popen:
    """Start a vLLM API server with feature ON; admission policy optional."""
    kv_config = {
        "kv_connector": "PrefillStagingPipelinedConnector",
        "kv_role": "kv_both",
    }
    cmd = [
        python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--max-num-seqs", str(max_num_seqs),
        "--disable-log-requests",
        "--dtype", "auto",
        "--enforce-eager",
        "--enable-kv-pipelining",
        "--kv-k-window", str(k_window),
        "--free-original-kv-cache",
        "--kv-transfer-config", json.dumps(kv_config),
        "--no-enable-prefix-caching",
    ]
    if enable_admission:
        cmd += [
            "--enable-transfer-aware-admission",
            "--transfer-aware-slo-p99-itl-ms", str(slo_p99_itl_ms),
            "--transfer-aware-predictor-csv", predictor_csv,
        ]
        if predictor_model:
            cmd += ["--transfer-aware-predictor-model", predictor_model]
        if admission_policy:
            cmd += ["--transfer-aware-admission-policy", admission_policy]
        if baseline_compute_json:
            cmd += ["--transfer-aware-baseline-compute-json", baseline_compute_json]
        if injection_jsonl:
            cmd += ["--transfer-aware-injection-jsonl", injection_jsonl]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w")
    log_fh.write(f"# {' '.join(cmd)}\n")
    for k in ("HF_HUB_CACHE", "HF_HOME", "CUDA_VISIBLE_DEVICES"):
        if k in env:
            log_fh.write(f"# env {k}={env[k]}\n")
    log_fh.flush()
    proc = subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid, env=env,
    )
    return proc


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except ProcessLookupError:
                pass
            proc.wait()


def _generate_prompt(input_len: int) -> str:
    """Cheap deterministic prompt of approximately ``input_len`` tokens.

    We use the same approach as decode_only_bench: concatenate filler words
    until the token count is close enough.
    """
    words = ["the", "of", "a", "to", "and", "in", "is", "it", "you", "that"]
    rng = random.Random(0xC0FFEE)
    return " ".join(rng.choice(words) for _ in range(int(input_len * 1.1)))


# ----------------------------------------------------------- async client


@dataclass
class RequestResult:
    arm: str
    request_idx: int
    submitted_at: float
    completed_at: float | None
    success: bool
    refused: bool
    error: str | None
    output_tokens: int
    ttft_s: float | None
    mean_itl_s: float | None
    p99_itl_s: float | None
    itl_samples: list[float]


async def _stream_one(
    session,
    *,
    arm: str,
    idx: int,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
) -> RequestResult:
    submitted = time.monotonic()
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    try:
        async with session.post(url, json=body, timeout=timeout) as resp:
            if resp.status >= 400:
                txt = await resp.text()
                refused = resp.status in (429, 503)
                return RequestResult(
                    arm=arm, request_idx=idx, submitted_at=submitted,
                    completed_at=time.monotonic(), success=False,
                    refused=refused, error=f"HTTP {resp.status}: {txt[:200]}",
                    output_tokens=0, ttft_s=None, mean_itl_s=None, p99_itl_s=None,
                    itl_samples=[],
                )
            samples: list[float] = []
            ttft = None
            tokens = 0
            t_last = submitted
            async for line in resp.content:
                if not line:
                    continue
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta_text = choices[0].get("text") or ""
                if not delta_text:
                    continue
                now = time.monotonic()
                if ttft is None:
                    ttft = now - submitted
                else:
                    samples.append(now - t_last)
                t_last = now
                tokens += 1
            done = time.monotonic()
            mean = (sum(samples) / len(samples)) if samples else None
            p99 = (sorted(samples)[max(0, int(0.99 * (len(samples) - 1)))]
                   if len(samples) >= 2 else mean)
            return RequestResult(
                arm=arm, request_idx=idx, submitted_at=submitted,
                completed_at=done, success=True, refused=False,
                error=None, output_tokens=tokens,
                ttft_s=ttft, mean_itl_s=mean, p99_itl_s=p99,
                itl_samples=samples,
            )
    except Exception as exc:
        return RequestResult(
            arm=arm, request_idx=idx, submitted_at=submitted,
            completed_at=time.monotonic(), success=False, refused=False,
            error=str(exc)[:200], output_tokens=0, ttft_s=None,
            mean_itl_s=None, p99_itl_s=None, itl_samples=[],
        )


async def _run_arm(
    *,
    arm: str,
    port: int,
    model: str,
    prompts: list[str],
    arrival_offsets_s: list[float],
    max_tokens: int,
    timeout: int,
) -> list[RequestResult]:
    """Open-loop arrivals: schedule requests at offsets[i] from t=0."""
    import aiohttp
    url = f"http://127.0.0.1:{port}/v1/completions"
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    async with aiohttp.ClientSession(timeout=timeout_obj, connector=connector) as session:
        t0 = time.monotonic()

        async def _at_offset(idx: int):
            target = t0 + arrival_offsets_s[idx]
            now = time.monotonic()
            if target > now:
                await asyncio.sleep(target - now)
            return await _stream_one(
                session, arm=arm, idx=idx, url=url, model=model,
                prompt=prompts[idx], max_tokens=max_tokens, timeout=timeout,
            )

        results = await asyncio.gather(*[_at_offset(i) for i in range(len(prompts))])
    return results


# --------------------------------------------------------- arrival sequence


def _poisson_offsets(rate: float, n: int, *, seed: int = 0xBEEF) -> list[float]:
    rng = random.Random(seed)
    out: list[float] = []
    t = 0.0
    for _ in range(n):
        gap = rng.expovariate(rate) if rate > 0 else 0.0
        t += gap
        out.append(t)
    return out


# -------------------------------------------------------------- summarize


def _summarize(results: list[RequestResult], *, slo_p99_ms: float) -> dict[str, Any]:
    completed = [r for r in results if r.success]
    refused = [r for r in results if r.refused]
    failed = [r for r in results if not r.success and not r.refused]
    slo_compliant = [r for r in completed
                     if r.p99_itl_s is not None
                     and (r.p99_itl_s * 1000.0) <= slo_p99_ms]

    all_p99 = [r.p99_itl_s for r in completed if r.p99_itl_s is not None]
    all_ttft = [r.ttft_s for r in completed if r.ttft_s is not None]

    if completed:
        wall = max(r.completed_at for r in completed) - min(r.submitted_at for r in completed)
    else:
        wall = 0.0
    total_completed_tokens = sum(r.output_tokens for r in completed)
    total_slo_compliant_tokens = sum(r.output_tokens for r in slo_compliant)

    def _p(seq: list[float], pct: float) -> float | None:
        if not seq:
            return None
        srt = sorted(seq)
        idx = max(0, min(len(srt) - 1, int(pct * (len(srt) - 1))))
        return srt[idx]

    return {
        "n_requests": len(results),
        "completed": len(completed),
        "refused": len(refused),
        "failed_other": len(failed),
        "slo_compliant_completions": len(slo_compliant),
        "slo_compliant_pct_of_completed": (
            100.0 * len(slo_compliant) / len(completed) if completed else 0.0
        ),
        "wall_s": wall,
        "raw_throughput_tok_s": (total_completed_tokens / wall) if wall > 0 else 0.0,
        "slo_compliant_throughput_tok_s": (
            total_slo_compliant_tokens / wall) if wall > 0 else 0.0,
        "p50_p99_itl_ms": _p(all_p99, 0.50) and _p(all_p99, 0.50) * 1000.0,
        "p99_p99_itl_ms": _p(all_p99, 0.99) and _p(all_p99, 0.99) * 1000.0,
        "p50_ttft_ms": _p(all_ttft, 0.50) and _p(all_ttft, 0.50) * 1000.0,
        "p99_ttft_ms": _p(all_ttft, 0.99) and _p(all_ttft, 0.99) * 1000.0,
    }


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# -------------------------------------------------------------- driver


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--predictor-csv", required=True, type=Path)
    ap.add_argument("--predictor-model", default="",
                    help="If empty, the controller picks the first model in the CSV.")
    ap.add_argument("--slo-p99-itl-ms", type=float, default=200.0)
    ap.add_argument("--request-rate", type=float, default=8.0,
                    help="Poisson arrivals per second (open-loop).")
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--k-window", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for JSONL + summary CSV + plots.")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--port-base", type=int, default=8300)
    ap.add_argument("--startup-timeout", type=int, default=600)
    ap.add_argument("--request-timeout", type=int, default=600)
    ap.add_argument("--arms", nargs="+", default=["stock", "policy"])
    ap.add_argument(
        "--baseline-compute-json",
        type=Path,
        default=Path(
            "/data/ben/kv_phase_a_internal/open_loop_eval_v1/"
            "m3_baseline_vs_feature/baseline/results_baseline.json"
        ),
        help="Mechanistic policy: decode_only_bench baseline results JSON.",
    )
    ap.add_argument(
        "--injection-jsonl",
        type=Path,
        default=Path(
            "/data/ben/kv_phase_a_internal/open_loop_eval_v1/"
            "injection_sweep_v5_combined/injection_sweep_v5.jsonl"
        ),
    )
    ap.add_argument("--gpu-id", type=str, default=None)
    return ap.parse_args()


def _arm_admission_config(arm: str, args: argparse.Namespace) -> tuple[bool, str]:
    """Return (enable_admission, transfer_aware_admission_policy) for an arm name."""
    if arm == "stock":
        return False, "combined"
    if arm == "policy":
        return True, "combined"
    if arm == "mechanistic":
        return True, "mechanistic"
    raise ValueError(f"Unknown arm {arm!r}; use stock, policy, or mechanistic")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    sys.path.insert(0, str(_THIS))
    from hf_env import apply_hf_token_from_user_file  # noqa: WPS433
    apply_hf_token_from_user_file()
    extra_env: dict[str, str] = {"HF_HUB_CACHE": "/data/ben/huggingface"}
    if "HF_TOKEN" in os.environ:
        extra_env["HF_TOKEN"] = os.environ["HF_TOKEN"]

    prompts = [_generate_prompt(args.input_len) for _ in range(args.num_prompts)]
    offsets = _poisson_offsets(args.request_rate, args.num_prompts)

    summaries: dict[str, dict] = {}
    raw_rows: list[dict] = []

    for arm in args.arms:
        enable_adm, adm_policy = _arm_admission_config(arm, args)
        port = _free_port(start=args.port_base)
        log_path = args.out / f"server_{arm}.log"
        proc = _launch_server(
            python=args.python, model=args.model, port=port,
            max_model_len=args.max_model_len, gpu_mem_util=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs, k_window=args.k_window, log_path=log_path,
            enable_admission=enable_adm,
            slo_p99_itl_ms=args.slo_p99_itl_ms,
            predictor_csv=str(args.predictor_csv),
            predictor_model=args.predictor_model,
            admission_policy=adm_policy,
            baseline_compute_json=str(args.baseline_compute_json),
            injection_jsonl=str(args.injection_jsonl),
            extra_env=extra_env,
        )
        try:
            print(f"\n=== arm={arm} | port={port} | predictor_csv={args.predictor_csv} "
                  f"| SLO={args.slo_p99_itl_ms} ms ===", flush=True)
            if not _wait_for_server(port, args.startup_timeout):
                raise RuntimeError(f"server failed to start; see {log_path}")
            print("[OK] server up; running open-loop benchmark ...", flush=True)
            t0 = time.monotonic()
            results = asyncio.run(_run_arm(
                arm=arm, port=port, model=args.model, prompts=prompts,
                arrival_offsets_s=offsets, max_tokens=args.output_len,
                timeout=args.request_timeout,
            ))
            wall = time.monotonic() - t0
            print(f"[DONE] arm={arm} wall={wall:.1f}s", flush=True)

            for r in results:
                row = asdict(r)
                row["model"] = args.model
                row["request_rate"] = args.request_rate
                row["slo_p99_itl_ms"] = args.slo_p99_itl_ms
                # Drop heavy itl_samples from raw row to keep JSONL small.
                row.pop("itl_samples", None)
                raw_rows.append(row)

            summaries[arm] = _summarize(results, slo_p99_ms=args.slo_p99_itl_ms)
            print(json.dumps(summaries[arm], indent=2), flush=True)
        finally:
            print(f"  Stopping {arm} server ...", flush=True)
            _kill(proc)
            time.sleep(5)

    _write_jsonl(raw_rows, args.out / "raw_results.jsonl")
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps({
        "args": vars(args) | {"out": str(args.out), "predictor_csv": str(args.predictor_csv)},
        "summaries": summaries,
    }, indent=2, default=str))
    print(f"\nWrote {summary_path}")

    if all(arm in summaries for arm in ("stock", "policy")):
        s, p = summaries["stock"], summaries["policy"]

        def _maybe(v: float | None) -> str:
            return f"{v:.1f}" if isinstance(v, (int, float)) else "n/a"
        print("\n--- comparison ---")
        for k in (
            "completed", "refused", "slo_compliant_completions",
            "slo_compliant_pct_of_completed", "raw_throughput_tok_s",
            "slo_compliant_throughput_tok_s", "p99_p99_itl_ms",
        ):
            print(f"  {k:<35s}  stock={_maybe(s.get(k))}  policy={_maybe(p.get(k))}")


if __name__ == "__main__":
    main()
