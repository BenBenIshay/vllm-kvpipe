#!/usr/bin/env python3
"""Experiment A: agentic structured-decoding (FSM) CPU cost.

Drives an ALREADY-RUNNING vanilla vLLM server and compares three arms at each
concurrency:
  - unguided        : plain decode (ignore_eos to fixed length)
  - guided-single   : one FIXED JSON schema for every request (high grammar-cache hit)
  - guided-diverse  : a UNIQUE, complex/nested schema per request (low cache hit,
                      forces xgrammar (re)compile + per-step fillmask)

During each arm's steady-decode window it samples:
  tree_cores      : whole engine process tree CPU (cores)
  busiest_thread  : hottest single thread (cores) -> EngineCore main loop
  worker_cores    : tree minus busiest -> parallel work (xgrammar fillmask pool)
  GPU SM%         : is the GPU stalled waiting on host masks?

Headline = (guided - unguided) deltas in cores / GPU% / tok-s.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_baseline_sweep import Sampler  # noqa: E402
from decode_only_bench import _generate_prompt, _itl_stats  # noqa: E402


def make_schema(complex_: bool, salt: int) -> dict:
    """Fixed schema (cache-friendly) or randomized nested schema (cache-hostile)."""
    rnd = random.Random(salt)

    def fn(prefix: str) -> str:
        return f"{prefix}_{rnd.randint(1000, 9999)}" if complex_ else prefix

    # Large arrays force long generation (~hundreds of tokens) so the steady
    # decode window is measurable even on a tiny, fast model.
    if not complex_:
        rec = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "score": {"type": "integer"},
                "label": {"type": "string"},
                "active": {"type": "boolean"},
            },
            "required": ["name", "score", "label", "active"],
        }
        return {
            "type": "object",
            "properties": {"records": {"type": "array", "items": rec,
                                       "minItems": 50, "maxItems": 50}},
            "required": ["records"],
        }

    # complex + diverse: randomized property names + deeper nesting
    inner = {
        "type": "object",
        "properties": {
            fn("k"): {"type": "string", "minLength": 3},
            fn("v"): {"type": "integer"},
            fn("tags"): {"type": "array", "items": {"type": "string"},
                         "minItems": 2, "maxItems": 4},
        },
        "required": [],
    }
    props = {fn("field"): {"type": "string", "minLength": 3} for _ in range(4)}
    props[fn("items")] = {"type": "array", "items": inner,
                          "minItems": 20, "maxItems": 20}
    return {"type": "object", "properties": props, "required": list(props.keys())}


def send(port: int, model: str, prompt: str, max_tokens: int,
         schema: dict | None, timeout: int) -> dict:
    body: dict[str, Any] = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    if schema is None:
        body["ignore_eos"] = True
    else:
        body["structured_outputs"] = {"json": schema}
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=data,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    tok_times: list[float] = []
    ttft = None
    ct = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            buf = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n\n" in buf:
                    event, buf = buf.split(b"\n\n", 1)
                    for line in event.split(b"\n"):
                        line = line.strip()
                        if not line.startswith(b"data: "):
                            continue
                        ds = line[6:].decode()
                        if ds.strip() == "[DONE]":
                            continue
                        try:
                            d = json.loads(ds)
                        except json.JSONDecodeError:
                            continue
                        u = d.get("usage")
                        if isinstance(u, dict) and u.get("completion_tokens"):
                            ct = int(u["completion_tokens"])
                        ch = d.get("choices", [])
                        if ch and ch[0].get("text"):
                            now = time.perf_counter()
                            if ttft is None:
                                ttft = now - t0
                            tok_times.append(now)
        e2e = time.perf_counter() - t0
        itls = [tok_times[i] - tok_times[i - 1] for i in range(1, len(tok_times))]
        stats = _itl_stats(itls, 1)
        return {"success": len(tok_times) > 0, "ttft_s": ttft, "e2e_s": e2e,
                "mean_itl_steady_s": stats["mean_itl_steady_s"],
                "completion_tokens": ct or len(tok_times),
                "_t0": time.time() - e2e, "_end": time.time()}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": str(e)[:160]}


def analyze(samples, lo, hi):
    win = [s for s in samples if lo <= s["wall"] <= hi]
    if len(win) < 2:
        return None
    dt = win[-1]["wall"] - win[0]["wall"]
    if dt <= 0:
        return None
    tree = (win[-1]["cpu"] - win[0]["cpu"]) / dt
    keys = set(win[0]["threads"]) & set(win[-1]["threads"])
    deltas = sorted(((win[-1]["threads"][k] - win[0]["threads"][k]) / dt)
                    for k in keys)
    busiest = deltas[-1] if deltas else float("nan")
    total_thr = sum(deltas)
    worker = max(0.0, total_thr - busiest)
    sm = [s["sm"] for s in win if s["sm"] == s["sm"]]
    return {"tree_cores": tree, "busiest": busiest, "worker": worker,
            "gpu_sm": sum(sm) / len(sm) if sm else float("nan"),
            "n": len(win)}


def run_arm(port, model, prompt, max_tokens, C, arm, sampler, timeout):
    def one(i):
        if arm == "unguided":
            sch = None
        elif arm == "guided-single":
            sch = make_schema(False, 0)
        else:
            sch = make_schema(True, i + random.randint(0, 10**6))
        return send(port, model, prompt, max_tokens, sch, timeout)

    with ThreadPoolExecutor(max_workers=C) as pool:
        recs = [f.result() for f in [pool.submit(one, i) for i in range(C)]]
    ok = [r for r in recs if r.get("success")]
    if not ok:
        return {"arm": arm, "C": C, "ok": 0,
                "error": (recs[0].get("error") if recs else "?")}
    starts = [r["_t0"] + r["ttft_s"] for r in ok if r.get("ttft_s")]
    ends = [r["_end"] for r in ok]
    lo, hi = max(starts) + 0.4, min(ends) - 0.2
    itls = [r["mean_itl_steady_s"] for r in ok if r.get("mean_itl_steady_s")]
    mitl = sum(itls) / len(itls) if itls else float("nan")
    toks = sum(r["completion_tokens"] for r in ok)
    st = analyze(sampler.samples, lo, hi) or {}
    return {
        "arm": arm, "C": C, "ok": len(ok),
        "mean_itl_ms": round(mitl * 1000, 2) if mitl == mitl else None,
        "decode_tok_s": round(C / mitl, 1) if mitl == mitl and mitl > 0 else 0,
        "gen_toks": toks,
        "tree_cores": round(st.get("tree_cores", float("nan")), 2),
        "busiest": round(st.get("busiest", float("nan")), 2),
        "worker_cores": round(st.get("worker", float("nan")), 2),
        "gpu_sm": round(st.get("gpu_sm", float("nan")), 1),
        "win_s": round(hi - lo, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--root-pid", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--concurrencies", type=int, nargs="+",
                    default=[1, 8, 32, 64])
    ap.add_argument("--arms", nargs="+",
                    default=["unguided", "guided-single", "guided-diverse"])
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"affinity cores: {len(os.sched_getaffinity(0))}")
    prompt = _generate_prompt(args.input_len) + \
        "\nReturn a JSON object that matches the schema.\n"
    sampler = Sampler(args.root_pid, interval=0.5)
    sampler.start()
    rows = []
    hdr = (f"{'arm':15s} {'C':>4} {'ok':>3} {'ITL ms':>7} {'tok/s':>8} "
           f"{'tree':>6} {'busy':>6} {'worker':>7} {'GPU%':>6} {'win_s':>6}")
    print(hdr)
    for C in args.concurrencies:
        for arm in args.arms:
            try:  # warmup
                run_arm(args.port, args.model, prompt, min(48, args.max_tokens),
                        C, arm, sampler, args.timeout)
            except Exception as e:  # noqa: BLE001
                print(f"  warmup {arm} C={C}: {str(e)[:60]}")
            time.sleep(0.4)
            r = run_arm(args.port, args.model, prompt, args.max_tokens, C, arm,
                        sampler, args.timeout)
            rows.append(r)
            if r.get("ok"):
                print(f"{r['arm']:15s} {r['C']:>4} {r['ok']:>3} "
                      f"{str(r['mean_itl_ms']):>7} {r['decode_tok_s']:>8} "
                      f"{r['tree_cores']:>6} {r['busiest']:>6} "
                      f"{r['worker_cores']:>7} {r['gpu_sm']:>6} {r['win_s']:>6}")
            else:
                print(f"{arm:15s} {C:>4} FAILED: {r.get('error')}")
        print("-" * len(hdr))

    sampler.stop()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
