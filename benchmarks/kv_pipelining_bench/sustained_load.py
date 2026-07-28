#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Sustained closed-loop load generator (oversubscription sweep)
=============================================================

Keeps ``--offered`` streaming requests continuously in flight against a running
vLLM server for ``--duration`` seconds, then reports **steady-state goodput**
(completed output tokens/sec) and mean ITL over a measurement window that skips
warmup/cooldown. Each request gets a unique prompt prefix so prefix-caching does
NOT collapse them (independent full KV per sequence).

Purpose: drive the long-context baseline PAST the KV-capacity cliff so we can see
whether/where aggregate tok/s degrades and how hard the host CPU works on
scheduling / paged-KV (HBM) management under memory pressure. Pair with
``cpu_gpu_sampler.py`` (running concurrently) and correlate via the steady-window
markers this writes to ``--markers``.

Usage::

    python sustained_load.py --port 8231 --input-len 8192 --output-len 512 \
        --offered 64 --duration 45 --warmup 12 --markers /tmp/run/markers.jsonl
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

_BASE = (
    "The following is a detailed technical analysis of system performance "
    "metrics in large-scale distributed computing environments. "
)


def make_prompt(target_tokens: int, gid: int) -> str:
    body = (_BASE * max(1, (target_tokens * 4) // len(_BASE)))[: target_tokens * 4]
    return f"Session {gid} request id {gid * 7919} unique context marker. " + body


def get_model(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30) as r:
        data = json.loads(r.read())
    return data["data"][0]["id"]


def one_request(
    port: int, model: str, prompt: str, max_tokens: int, timeout: int
) -> dict:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    submit = time.time()
    ttft = None
    toks = 0
    usage_toks = None
    tok_times: list[float] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                ch = obj.get("choices") or []
                if ch and ch[0].get("text"):
                    now = time.time()
                    if ttft is None:
                        ttft = now
                    toks += 1
                    tok_times.append(now)
                if obj.get("usage"):
                    usage_toks = obj["usage"].get("completion_tokens")
    except Exception as e:  # noqa: BLE001
        return {"submit": submit, "ttft": ttft, "end": time.time(),
                "toks": toks, "tok_times": tok_times, "ok": False,
                "err": str(e)[:80]}
    end = time.time()
    return {
        "submit": submit,
        "ttft": ttft,
        "end": end,
        "toks": usage_toks if usage_toks is not None else toks,
        "tok_times": tok_times,
        "ok": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8231)
    ap.add_argument("--model", default=None)
    ap.add_argument("--input-len", type=int, default=8192)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--offered", type=int, required=True)
    ap.add_argument("--duration", type=float, default=45.0)
    ap.add_argument("--warmup", type=float, default=12.0)
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--request-timeout", type=int, default=600)
    ap.add_argument("--markers", required=True)
    args = ap.parse_args()

    model = args.model or get_model(args.port)
    print(f"[load] offered={args.offered} model={model} dur={args.duration}s", flush=True)

    stop = threading.Event()
    results: list[dict] = []
    rlock = threading.Lock()
    gid = [0]
    glock = threading.Lock()

    def worker() -> None:
        while not stop.is_set():
            with glock:
                gid[0] += 1
                myid = gid[0]
            prompt = make_prompt(args.input_len, myid)
            r = one_request(args.port, model, prompt, args.output_len,
                            args.request_timeout)
            with rlock:
                results.append(r)

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.offered) as pool:
        for _ in range(args.offered):
            pool.submit(worker)
        # Live status every 5s.
        next_print = t_start + 5
        while time.time() - t_start < args.duration:
            time.sleep(0.5)
            if time.time() >= next_print:
                with rlock:
                    done = sum(1 for r in results if r["ok"])
                print(f"  [{time.time()-t_start:5.1f}s] completed={done}", flush=True)
                next_print += 5
        stop.set()
    t_end = time.time()

    meas_lo = t_start + args.warmup
    meas_hi = t_end - args.cooldown
    win = max(1e-6, meas_hi - meas_lo)

    # Goodput from actual per-token arrival times within the steady window
    # (robust to long request duration / bursty completions).
    toks_win = 0
    for r in results:
        for tt in r.get("tok_times", ()):
            if meas_lo <= tt <= meas_hi:
                toks_win += 1
    goodput = toks_win / win

    # ITL / TTFT from requests whose decode overlapped the window.
    overlap = [r for r in results if r["ok"] and r["ttft"] is not None
               and r["end"] >= meas_lo and r["ttft"] <= meas_hi]
    itls = [
        (r["end"] - r["ttft"]) / max(1, r["toks"] - 1)
        for r in overlap if r["toks"] > 1
    ]
    mean_itl = sum(itls) / len(itls) if itls else None
    # TTFT only from requests SUBMITTED within the window (queue-wait signal).
    ttfts = [r["ttft"] - r["submit"] for r in results
             if r["ok"] and r["ttft"] is not None
             and meas_lo <= r["submit"] <= meas_hi]
    mean_ttft = sum(ttfts) / len(ttfts) if ttfts else None
    p99_ttft = sorted(ttfts)[int(0.99 * (len(ttfts) - 1))] if ttfts else None
    n_fail = sum(1 for r in results if not r["ok"])

    marker = {
        "offered": args.offered,
        "t_meas_start": meas_lo,
        "t_meas_end": meas_hi,
        "window_s": round(win, 2),
        "goodput_tok_s": round(goodput, 1),
        "completed_in_win": len(overlap),
        "mean_itl_ms": round(mean_itl * 1000, 2) if mean_itl else None,
        "mean_ttft_s": round(mean_ttft, 2) if mean_ttft else None,
        "p99_ttft_s": round(p99_ttft, 2) if p99_ttft else None,
        "total_completed": sum(1 for r in results if r["ok"]),
        "failures": n_fail,
    }
    with open(args.markers, "a", buffering=1) as mf:
        mf.write(json.dumps(marker) + "\n")
    print(
        f"[load] offered={args.offered} goodput={goodput:.0f} tok/s "
        f"ITL={marker['mean_itl_ms']}ms TTFT={marker['mean_ttft_s']}s "
        f"p99TTFT={marker['p99_ttft_s']}s tok_in_win={toks_win} fail={n_fail}",
        flush=True,
    )


if __name__ == "__main__":
    main()
