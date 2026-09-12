#!/usr/bin/env python3
"""What does it COST to turn a prefilled request into a rider?

`paper/PAPER_PLAN_2026-09-05.md` section 6.1 calls this the objection that
decides the paper's reception:

    "Your riders skip prefill -- you are measuring a decode node with free
     input."  -> Measure and report the handover cost rather than assuming it
     away.

The handover is built and proven bitwise (`prefill_stash_roundtrip.py`,
max|diff| = 0), but until now nothing measured what it costs. Publishing a
capacity gain while leaving its entry price unmeasured is exactly the hole a
reviewer opens first.

Three phases, timed separately because they are paid by different parties in a
real deployment:

  seed     resident HBM blocks -> pinned host mirrors. On a real decode node
           this does not happen at all: KV arrives from the prefill node. It is
           charged here because our LOCAL emulation produces the KV itself.
  stash    host mirrors -> a prompt-keyed CPU copy that outlives the request.
           This is the stand-in for "the prefill node's output is retained".
  preload  stash -> pinned host mirrors of a NEW request, the decode-side
           claim. **This is the phase a real P/D decode node actually pays**,
           and it is the number the paper should lead with.

NOT measured, and the paper must say so: the prefill node -> decode node
NETWORK transfer. That is deployment-dependent (NIC, fabric, placement) and
inventing a number for it would be worse than declaring it out of scope.

Reported per request AND amortised over the decode steps a rider serves, since
a one-off cost paid once per request and a per-step cost are not comparable
quantities -- conflating them is how a handover gets "assumed away" in the
other direction.

No engine, no scheduler: drives the pool directly, so the numbers are the data
movement and nothing else.

Run: srun ... .venv/bin/python benchmarks/kv_pipelining_bench/handover_cost.py
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from vllm.v1.hybrid_rungd_pool import RungdHostPool

DT = torch.float16


def timed(fn, *a, **kw) -> float:
    """Wall ms, with the device quiesced on both sides.

    The copies are a mix of device->host, host->host and host->device; timing
    them without a sync would attribute an async copy's cost to whatever ran
    next, which is the mistake that made write_token look like 21 ms.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(*a, **kw)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def build_pool(layers: int, nkv: int, hd: int, bs: int, blocks: int,
               bpr: int) -> RungdHostPool:
    return RungdHostPool(
        layer_names=[f"l{i}" for i in range(layers)],
        kv_shape=(2, blocks, bs, nkv, hd), block_size=bs,
        k_window=4, h_max_blocks=blocks, w_max_blocks=blocks,
        device="cuda", dtype=DT, blocks_per_req=bpr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=64)      # Qwen-32B
    ap.add_argument("--nkv", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--contexts", default="512,1024,2048,4096")
    ap.add_argument("--decode-steps", type=int, default=64,
                    help="steps a rider serves, for amortisation")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    b_tok = args.layers * 2 * args.nkv * args.head_dim * 2
    print(f"{args.layers}L x {args.nkv}kv x {args.head_dim}d -> "
          f"{b_tok} bytes/token of KV")
    print(f"{'ctx':>6} {'KV/req':>9} {'seed':>9} {'stash':>9} {'preload':>9} "
          f"{'GB/s(pre)':>10} {'per step':>9}")

    rows = []
    for ctx in [int(c) for c in args.contexts.split(",")]:
        nb = -(-ctx // args.block_size)
        bpr = nb + 2
        pool = build_pool(args.layers, args.nkv, args.head_dim,
                          args.block_size, bpr * 6, bpr)
        # A resident tensor to seed FROM, exactly as the runner supplies.
        resident = {ln: torch.randn(2, bpr * 6, args.block_size, args.nkv,
                                    args.head_dim, device="cuda", dtype=DT)
                    for ln in pool.layers}
        pool.resident = resident
        rid_blocks = list(range(nb))

        seeds, stashes, preloads = [], [], []
        for r in range(args.reps):
            producer, key = f"p{r}", f"k{r}"
            pool.ensure_req(producer, nb)
            seeds.append(timed(pool.seed_req, producer, rid_blocks))
            stashes.append(timed(pool.stash_from_host, key, producer, ctx))
            pool.free_req(producer)
            preloads.append(timed(pool.preload_from_stash, f"c{r}", key))
            pool.free_req(f"c{r}")
            pool.drop_stash(key)

        med = lambda xs: sorted(xs)[len(xs) // 2]                # noqa: E731
        s, t, p = med(seeds), med(stashes), med(preloads)
        kv_gb = ctx * b_tok / 1e9
        row = {"ctx": ctx, "kv_gb_per_req": round(kv_gb, 4),
               "seed_ms": round(s, 2), "stash_ms": round(t, 2),
               "preload_ms": round(p, 2),
               "preload_gbps": round(kv_gb / (p / 1000.0), 2) if p else None,
               "preload_ms_per_decode_step": round(p / args.decode_steps, 3)}
        rows.append(row)
        print(f"{ctx:>6} {kv_gb:>8.3f}G {s:>8.2f}m {t:>8.2f}m {p:>8.2f}m "
              f"{row['preload_gbps']:>10} {row['preload_ms_per_decode_step']:>8.3f}m")
        del pool, resident
        torch.cuda.empty_cache()

    print()
    print("READING THIS:")
    print("  * `preload` is what a real P/D decode node pays -- the claim side.")
    print("  * `seed` is an artefact of the LOCAL emulation producing its own")
    print("    KV; a real decode node never does it. Reported, not hidden.")
    print("  * the prefill-node -> decode-node NETWORK transfer is NOT here and")
    print("    is deployment-dependent; the paper must say so rather than")
    print("    invent a number.")
    ref = rows[-1]
    print(f"  * amortised: at ctx {ref['ctx']}, preload is "
          f"{ref['preload_ms']:.1f} ms once, i.e. "
          f"{ref['preload_ms_per_decode_step']:.3f} ms per decode step over "
          f"{args.decode_steps} steps.")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "b_tok": b_tok, "rows": rows},
                      f, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
