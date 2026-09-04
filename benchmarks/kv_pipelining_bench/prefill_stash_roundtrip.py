#!/usr/bin/env python3
"""The prefill->pool handover moves KV without changing a bit.

`prefill_stash_identity.py` runs the real engine end to end, but it cannot be
decisive, and it is worth being precise about why. A rider is admitted with its
LAST prompt token declared uncomputed, so that token's KV is produced by the
decode kernel rather than the prefill kernel. That is not a choice we made --
vLLM's own prefix cache does it, and every P/D decode node must. Measured on
Qwen2.5-0.5B, greedy, serial:

    vanilla + prefix cache   vs fresh full prefill   2/6 tokens identical
    our rider on stashed KV  vs fresh full prefill   4/6 tokens identical

vLLM's own cache perturbs MORE prompts than our handover does, and our
divergences are a strict subset of its. So the greedy token stream cannot
separate "the handover corrupted the KV" from "a near-tie flipped", and a
token test that claimed to would be lying.

This separates them. No engine, no scheduler, no sampling: hand real KV in,
stash it, claim it from a different request, and compare BITWISE.

  1. round trip is exact, every layer, every token;
  2. exact across block boundaries and at non-block-aligned lengths, where a
     token-major <-> block-major mapping is most likely to be wrong;
  3. exact after the producing request has been freed AND its host extent
     handed to another request that overwrites it -- the stash must be a copy,
     not a view of blocks that get recycled;
  4. the claimed KV reads back through the ROLLING SLOT path the attention
     kernel actually uses, not merely out of the mirrors.

Run: srun ... .venv/bin/python benchmarks/kv_pipelining_bench/prefill_stash_roundtrip.py
"""

from __future__ import annotations

import torch

from vllm.v1.hybrid_rungd_pool import RungdHostPool

DEV = "cuda"
DT = torch.float16
NUM_LAYERS = 4
BS = 16
NKV = 2
HD = 64
H_MAX = 512
FAILURES: list[str] = []


def check(cond: bool, what: str) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        FAILURES.append(what)


def make_pool(blocks_per_req: int) -> RungdHostPool:
    return RungdHostPool(
        layer_names=[f"l{i}" for i in range(NUM_LAYERS)],
        kv_shape=(2, H_MAX, BS, NKV, HD),
        block_size=BS, k_window=2, h_max_blocks=H_MAX, w_max_blocks=H_MAX,
        device=DEV, dtype=DT, blocks_per_req=blocks_per_req,
    )


def random_kv(n: int, seed: int | None = None
              ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(n if seed is None else seed)
    return {f"l{i}": (torch.randn(n, NKV, HD, dtype=DT),
                      torch.randn(n, NKV, HD, dtype=DT))
            for i in range(NUM_LAYERS)}


def read_back(pool: RungdHostPool, req_id: str, n: int, layer: str):
    """Read a request's KV out of the host mirrors, token-major."""
    hblks = pool.req_blocks[req_id]
    hk, hv = pool.host_k[layer], pool.host_v[layer]
    hk_flat = hk.view(hk.shape[0] * BS, *hk.shape[2:])
    hv_flat = hv.view(hv.shape[0] * BS, *hv.shape[2:])
    k = torch.empty((n, NKV, HD), dtype=DT)
    v = torch.empty((n, NKV, HD), dtype=DT)
    for p in range(n):
        src = hblks[p // BS] * BS + p % BS
        k[p] = hk_flat[src]
        v[p] = hv_flat[src]
    return k, v


def roundtrip(n: int, label: str) -> None:
    """producer preloads real KV -> stash -> a DIFFERENT request claims it."""
    pool = make_pool(blocks_per_req=32)
    kv = random_kv(n)
    pool.preload_req("producer", n, kv)
    pool.stash_from_host("key", "producer", n)
    pool.free_req("producer")

    # Recycle the producer's extent under heavy overwriting before claiming.
    # If the stash held a view rather than a copy, this is what would corrupt it.
    pool.preload_req("squatter", n, random_kv(n, seed=99991))
    pool.free_req("squatter")

    delivered = pool.preload_from_stash("consumer", "key")
    check(delivered == n, f"{label}: delivered {delivered} tokens, expected {n}")
    worst = 0.0
    exact = True
    for ln in pool.layers:
        k, v = read_back(pool, "consumer", n, ln)
        k0, v0 = kv[ln]
        exact = exact and torch.equal(k, k0) and torch.equal(v, v0)
        worst = max(worst, (k - k0).abs().max().item(),
                    (v - v0).abs().max().item())
    check(exact, f"{label}: bitwise identical after recycling (max|diff| "
                 f"{worst:g})")


def rolling_path(n: int) -> None:
    """The claimed KV must be right where attention reads it: the rolling GPU
    slots, after compaction -- not just in the mirrors."""
    pool = make_pool(blocks_per_req=32)
    kv = random_kv(n)
    pool.preload_req("producer", n, kv)
    pool.stash_from_host("key", "producer", n)
    pool.free_req("producer")
    pool.preload_from_stash("consumer", "key")
    pool.build_step(["consumer"], [n])
    exact = True
    worst = 0.0
    for li, ln in enumerate(pool.layers):
        kslot, vslot = pool.stream_layer(li)
        torch.cuda.synchronize()
        k0, v0 = kv[ln]
        for p in range(n):
            c, off = pool.compact_of("consumer", p)
            gk = kslot[c, off].cpu()
            gv = vslot[c, off].cpu()
            exact = exact and torch.equal(gk, k0[p]) and torch.equal(gv, v0[p])
            worst = max(worst, (gk - k0[p]).abs().max().item(),
                        (gv - v0[p]).abs().max().item())
    check(exact, f"n={n}: rolling slots the kernel reads hold the stashed KV "
                 f"bitwise (max|diff| {worst:g})")


def main() -> int:
    print("prefill stash round trip: real KV in, bitwise KV out")
    print("1. block-aligned")
    roundtrip(64, "n=64 (4 whole blocks)")
    print("2. non-block-aligned, spanning many blocks")
    roundtrip(327, "n=327 (20 blocks + 7)")
    print("3. shorter than one block")
    roundtrip(5, "n=5")
    print("4. exactly one token past a block boundary")
    roundtrip(17, "n=17")
    print("5. through the rolling slots attention actually reads")
    rolling_path(327)

    print("=" * 70)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("PASS: the prefill->pool handover is bitwise exact. Any token-level "
          "difference a rider shows is the last-token recompute that P/D "
          "and vLLM's own prefix cache share -- not corruption in transit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
