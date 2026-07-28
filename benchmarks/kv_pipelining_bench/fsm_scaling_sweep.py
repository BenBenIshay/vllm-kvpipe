#!/usr/bin/env python3
"""Grammar-complexity x batch sweep: does structured-decoding CPU stay HIDDEN under
a production-size model's forward, or CLIMB toward saturating cores?

Closed-loop (non-streaming) load at each concurrency -> minimal client CPU. Arms
vary the per-step grammar cost while keeping decode work comparable:

  unguided : plain decode (ignore_eos) -- baseline, no grammar work
  simple   : flat scalar/enum array   -- few context-dependent tokens, cache hit
  medium   : nested objects + bounded strings
  complex  : deep nesting + FREE strings + regex + bounded strings (FIXED schema,
             cache hit) -- isolates heavy per-step fillmask (many legal tokens)
  diverse  : same shape as complex but a UNIQUE schema per request (cache-hostile
             -> forces xgrammar (re)compile) -- isolates grammar COMPILE cost

Concurrency sweeps across fill_bitmask_parallel_threshold (128) so the serial->pool
fill transition is visible. During each steady window we sample the server process
tree:
  tree_cores  : whole engine tree CPU (cores)         -> total host cost
  busiest     : hottest single thread (cores)         -> EngineCore step loop
  worker      : tree - busiest (cores)                -> fill/compile thread pools
  GPU SM%     : is the GPU stalled waiting on host masks?
  tok/s       : sustained decode throughput

Read: if (guided - unguided) tree_cores climbs toward the core budget AND GPU%/tok-s
drop as complexity/batch grow -> real CPU-compute bottleneck. If cores stay flat and
GPU% holds -> grammar work hides under the forward (direction killed for big models).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_baseline_sweep import Sampler  # noqa: E402
from decode_only_bench import _generate_prompt  # noqa: E402


def make_schema(level: str, salt: int) -> dict:
    """Graded JSON schemas. `salt` only matters for `diverse` (unique names)."""
    rnd = random.Random(salt)

    def fn(p: str, unique: bool) -> str:
        return f"{p}_{rnd.randint(1000, 9999)}" if unique else p

    if level == "simple":
        # Flat records, only scalars + a tiny enum: almost every position has a
        # small, fixed legal-token set -> adaptive cache resolves it cheaply.
        rec = {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "active": {"type": "boolean"},
                "level": {"type": "string", "enum": ["low", "mid", "high"]},
                "count": {"type": "integer"},
            },
            "required": ["id", "active", "level", "count"],
        }
        return {
            "type": "object",
            "properties": {"rows": {"type": "array", "items": rec,
                                    "minItems": 60, "maxItems": 60}},
            "required": ["rows"],
        }

    if level == "medium":
        inner = {
            "type": "object",
            "properties": {"k": {"type": "string", "minLength": 2, "maxLength": 12},
                           "v": {"type": "integer"}},
            "required": ["k", "v"],
        }
        rec = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 2, "maxLength": 16},
                "meta": inner,
                "score": {"type": "integer"},
            },
            "required": ["name", "meta", "score"],
        }
        return {
            "type": "object",
            "properties": {"items": {"type": "array", "items": rec,
                                     "minItems": 40, "maxItems": 40}},
            "required": ["items"],
        }

    # complex / diverse: deep nesting + FREE strings (no maxLength -> many legal
    # tokens each step) + regex pattern. `diverse` also randomizes property names
    # so every request is a fresh grammar (cache miss -> recompile).
    unique = (level == "diverse")
    inner = {
        "type": "object",
        "properties": {
            fn("code", unique): {"type": "string", "pattern": "[A-Z]{2,4}-[0-9]{2,5}"},
            fn("desc", unique): {"type": "string", "minLength": 5},
            fn("tags", unique): {"type": "array",
                                 "items": {"type": "string", "minLength": 2},
                                 "minItems": 2, "maxItems": 5},
        },
        "required": [],
    }
    rec = {
        "type": "object",
        "properties": {
            fn("title", unique): {"type": "string", "minLength": 5},
            fn("detail", unique): inner,
            fn("num", unique): {"type": "integer"},
        },
        "required": [],
    }
    props = {fn("f", unique): {"type": "string", "minLength": 3} for _ in range(3)}
    props[fn("data", unique)] = {"type": "array", "items": rec,
                                 "minItems": 25, "maxItems": 25}
    return {"type": "object", "properties": props, "required": list(props.keys())}


# Pre-build fixed schemas once (cache-friendly arms) + a pool of unique schemas
# for the diverse arm (built once so the client hot-loop does no schema work).
FIXED = {lvl: make_schema(lvl, 0) for lvl in ("simple", "medium", "complex")}
DIVERSE_POOL = [make_schema("diverse", i) for i in range(512)]


def one_request(port, model, prompt, max_tokens, arm, pick, timeout):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": False}
    if arm == "unguided":
        body["ignore_eos"] = True
    elif arm == "diverse":
        body["structured_outputs"] = {"json": DIVERSE_POOL[pick % len(DIVERSE_POOL)]}
    else:
        body["structured_outputs"] = {"json": FIXED[arm]}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        d = json.loads(resp.read())
    return int(d.get("usage", {}).get("completion_tokens", 0))


def run_closed_loop(port, model, prompt, max_tokens, C, arm, duration, timeout):
    stop = threading.Event()
    counts = [0]
    reqs = [0]
    lock = threading.Lock()

    def worker(wid):
        local = 0
        nreq = 0
        pick = wid
        while not stop.is_set():
            try:
                local += one_request(port, model, prompt, max_tokens, arm, pick,
                                     timeout)
                nreq += 1
                pick += C  # each worker walks the diverse pool distinctly
            except Exception:
                pass
        with lock:
            counts[0] += local
            reqs[0] += nreq

    pool = [threading.Thread(target=worker, args=(i,), daemon=True)
            for i in range(C)]
    t0 = time.time()
    for t in pool:
        t.start()
    time.sleep(duration)
    stop.set()
    for t in pool:
        t.join(timeout=timeout)
    wall = time.time() - t0
    return counts[0] / wall, reqs[0] / wall, t0, t0 + wall


def run_point(port, model, prompt, max_tokens, C, arm, sampler, duration, timeout):
    # warmup (also compiles fixed grammars / primes graphs for this batch)
    run_closed_loop(port, model, prompt, max_tokens, C, arm, 2.5, timeout)
    tps, rps, t0, t1 = run_closed_loop(port, model, prompt, max_tokens, C, arm,
                                       duration, timeout)
    lo, hi = t0 + 0.15 * (t1 - t0), t1 - 0.15 * (t1 - t0)
    win = [s for s in sampler.samples if lo <= s["wall"] <= hi]
    if len(win) >= 2:
        dt = win[-1]["wall"] - win[0]["wall"]
        tree = (win[-1]["cpu"] - win[0]["cpu"]) / dt
        keys = set(win[0]["threads"]) & set(win[-1]["threads"])
        deltas = sorted((win[-1]["threads"][k] - win[0]["threads"][k]) / dt
                        for k in keys)
        busy = deltas[-1] if deltas else float("nan")
        worker = max(0.0, sum(deltas) - busy)
        sm = [s["sm"] for s in win if s["sm"] == s["sm"]]
        gpu = sum(sm) / len(sm) if sm else float("nan")
    else:
        tree = busy = worker = gpu = float("nan")
    return {"arm": arm, "C": C, "tok_s": round(tps, 1), "req_s": round(rps, 2),
            "tree": round(tree, 2), "busy": round(busy, 2),
            "worker": round(worker, 2), "gpu_sm": round(gpu, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--root-pid", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--concurrencies", type=int, nargs="+", default=[32, 128, 256])
    ap.add_argument("--arms", nargs="+",
                    default=["unguided", "simple", "medium", "complex", "diverse"])
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"client affinity cores: {len(os.sched_getaffinity(0))}")
    prompt = _generate_prompt(args.input_len) + \
        "\nReturn a JSON object matching the requested schema.\n"
    sampler = Sampler(args.root_pid, interval=0.4)
    sampler.start()
    rows = []
    hdr = (f"{'arm':10s} {'C':>4} {'tok/s':>9} {'req/s':>7} {'tree':>6} "
           f"{'busy':>6} {'worker':>7} {'GPU%':>6}")
    for C in args.concurrencies:
        print(hdr)
        for arm in args.arms:
            try:
                r = run_point(args.port, args.model, prompt, args.max_tokens, C,
                              arm, sampler, args.duration, args.timeout)
            except Exception as e:  # noqa: BLE001
                print(f"{arm:10s} {C:>4} FAILED: {str(e)[:70]}")
                continue
            rows.append(r)
            print(f"{r['arm']:10s} {r['C']:>4} {r['tok_s']:>9} {r['req_s']:>7} "
                  f"{r['tree']:>6} {r['busy']:>6} {r['worker']:>7} {r['gpu_sm']:>6}")
        print("-" * len(hdr))

    sampler.stop()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
