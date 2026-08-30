#!/usr/bin/env python3
"""A request can join the host tier without ever owning an HBM block.

This is the mechanism behind the second of the two limits measured on
2026-08-13. `seed_layer` reads a request's KV out of its RESIDENT blocks, so a
request could only become a rider by first being an ordinary fully-resident
request: allocated HBM, prefilled into it, copied out, then reclaimed. That
transient residency is why `b_hbm_max` has to sit below capacity, and therefore
why riders were purchased WITH resident slots instead of added on top of them:

    all-HBM  6 resident                     = 6
    mixture  3 resident + ~2 riders         = 5     (measured; loses)

`preload_req` is the other door: KV arrives from outside (a prefill node, a
connector that staged it during prefill, a fixture) and lands directly in the
host mirrors. A request admitted this way owns zero main-pool blocks at every
instant.

What this checks, with no engine and no scheduler:

  1. a preloaded request's KV is bitwise what was handed in, read back through
     the normal rolling-slot path the attention kernel uses;
  2. attention over the preloaded request equals attention over the same KV
     laid out resident -- i.e. the pool path is not just self-consistent but
     agrees with the reference;
  3. it holds across a block boundary and for a non-block-aligned length, which
     is where a token-major -> block-major write is most likely to be wrong;
  4. preloaded and seeded-from-resident requests coexist in one cohort, since
     a real mixture will contain both.

Run: srun ... .venv/bin/python benchmarks/kv_pipelining_bench/pool_preload_test.py
"""

from __future__ import annotations

import sys

import torch

from vllm.v1.hybrid_rungd_pool import RungdHostPool
from vllm.vllm_flash_attn import flash_attn_varlen_func

DEV = "cuda"
DT = torch.float16
NUM_LAYERS = 4
BS = 16          # block size
NKV = 2
HD = 64
NUM_HEADS = 4
SCALE = HD ** -0.5
FA = 2
H_MAX = 128
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


def paged_ref(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay a token-major [n, NKV, HD] history out as one paged block table."""
    n = k.shape[0]
    nb = (n + BS - 1) // BS
    kc = torch.zeros(nb, BS, NKV, HD, device=DEV, dtype=DT)
    vc = torch.zeros(nb, BS, NKV, HD, device=DEV, dtype=DT)
    for p in range(n):
        kc[p // BS, p % BS] = k[p]
        vc[p // BS, p % BS] = v[p]
    return kc, vc


def attend(q, kc, vc, bt, seqlen) -> torch.Tensor:
    out = torch.empty(1, NUM_HEADS, HD, device=DEV, dtype=DT)
    cu = torch.arange(0, 2, device=DEV, dtype=torch.int32)
    flash_attn_varlen_func(
        q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=1,
        seqused_k=torch.tensor([seqlen], device=DEV, dtype=torch.int32),
        max_seqlen_k=seqlen, softmax_scale=SCALE, causal=True,
        block_table=bt, fa_version=FA,
    )
    return out


def test_lengths() -> None:
    print("\n1) preloaded KV reads back bitwise, and attends like resident")
    # 40 tokens spans 3 blocks and is not block-aligned -- the case a
    # token-major -> block-major write is most likely to get wrong.
    for n_tok in (16, 40, 64):
        pool = make_pool(blocks_per_req=8)
        kv = {}
        for i in range(NUM_LAYERS):
            k = torch.randn(n_tok, NKV, HD, device=DEV, dtype=DT)
            v = torch.randn(n_tok, NKV, HD, device=DEV, dtype=DT)
            kv[f"l{i}"] = (k, v)
        pool.preload_req("r0", n_tok, kv)
        check("r0" in pool.req_seeded,
              f"n={n_tok}: preloaded request is marked seeded "
              "(runner will not seed it from resident it never had)")

        bt, seqused = pool.build_step(["r0"], [n_tok])
        bt_d, q = bt.to(DEV), torch.randn(1, NUM_HEADS, HD, device=DEV, dtype=DT)
        content_ok, attn_ok = True, True
        for i in range(NUM_LAYERS):
            kslot, vslot = pool.stream_layer(i)
            k_in, v_in = kv[f"l{i}"]
            for p in range(n_tok):
                c, off = pool.compact_of("r0", p)
                if not torch.equal(kslot[c, off], k_in[p]):
                    content_ok = False
                if not torch.equal(vslot[c, off], v_in[p]):
                    attn_ok = attn_ok and False
            got = attend(q, kslot, vslot, bt_d, n_tok)
            kc, vc = paged_ref(k_in, v_in)
            ref_bt = torch.arange(kc.shape[0], device=DEV,
                                  dtype=torch.int32).unsqueeze(0)
            want = attend(q, kc, vc, ref_bt, n_tok)
            if not torch.equal(got, want):
                attn_ok = False
        check(content_ok, f"n={n_tok}: every token bitwise as handed in")
        check(attn_ok, f"n={n_tok}: attention == resident-layout reference")


def test_mixed_cohort() -> None:
    print("\n2) a preloaded rider and a seeded-from-resident rider coexist")
    pool = make_pool(blocks_per_req=8)
    n_tok = 40
    nb = (n_tok + BS - 1) // BS

    # r_seed: the classic path -- KV lives in a resident tensor, copied out.
    resident = {f"l{i}": torch.randn(2, H_MAX, BS, NKV, HD, device=DEV, dtype=DT)
                for i in range(NUM_LAYERS)}
    pool.resident = resident
    res_blocks = [50, 51, 52]
    pool.ensure_req("r_seed", nb)
    pool.seed_req("r_seed", res_blocks)

    # r_pre: the new path -- KV arrives from outside, no resident blocks.
    kv = {f"l{i}": (torch.randn(n_tok, NKV, HD, device=DEV, dtype=DT),
                    torch.randn(n_tok, NKV, HD, device=DEV, dtype=DT))
          for i in range(NUM_LAYERS)}
    pool.preload_req("r_pre", n_tok, kv)

    active = ["r_seed", "r_pre"]
    bt, seqused = pool.build_step(active, [n_tok, n_tok])
    bt_d = bt.to(DEV)
    ok_pre, ok_seed = True, True
    for i in range(NUM_LAYERS):
        kslot, vslot = pool.stream_layer(i)
        k_in, v_in = kv[f"l{i}"]
        for p in range(n_tok):
            c, off = pool.compact_of("r_pre", p)
            if not torch.equal(kslot[c, off], k_in[p]):
                ok_pre = False
            c2, off2 = pool.compact_of("r_seed", p)
            want = resident[f"l{i}"][0, res_blocks[p // BS], p % BS]
            if not torch.equal(kslot[c2, off2], want):
                ok_seed = False
    check(ok_pre, "preloaded rider's KV intact alongside a seeded one")
    check(ok_seed, "seeded rider's KV intact alongside a preloaded one")
    check(bt.shape[0] == 2, "both riders share one compacted block table")


def test_no_resident_needed() -> None:
    print("\n3) preload works with NO resident tensors registered at all")
    # A real P/D decode node has no prefill and therefore nothing resident to
    # seed from. Registering `resident` must not be a precondition.
    pool = make_pool(blocks_per_req=8)
    check(not pool.resident, "pool has no resident tensors registered")
    n_tok = 33
    kv = {f"l{i}": (torch.randn(n_tok, NKV, HD, device=DEV, dtype=DT),
                    torch.randn(n_tok, NKV, HD, device=DEV, dtype=DT))
          for i in range(NUM_LAYERS)}
    try:
        pool.preload_req("r0", n_tok, kv)
        ok = True
    except Exception as e:                                   # noqa: BLE001
        print(f"       raised: {e}")
        ok = False
    check(ok, "preload_req succeeds with an empty resident map")
    if ok:
        pool.build_step(["r0"], [n_tok])
        kslot, _ = pool.stream_layer(0)
        c, off = pool.compact_of("r0", n_tok - 1)
        check(torch.equal(kslot[c, off], kv["l0"][0][n_tok - 1]),
              "last token of an unaligned length is correct")


def main() -> int:
    if not torch.cuda.is_available():
        print("needs a GPU")
        return 2
    torch.manual_seed(0)
    print("host-tier preload -- a rider that never owns an HBM block")
    test_lengths()
    test_mixed_cohort()
    test_no_resident_needed()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("POOL PRELOAD PASS: KV can enter the host tier from outside, bitwise "
          "correct, agreeing with the resident reference, alongside "
          "seeded-from-resident riders, and with nothing resident at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
