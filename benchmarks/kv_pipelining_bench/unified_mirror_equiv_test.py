#!/usr/bin/env python3
"""T7a's vectorised scatter must land BITWISE where the per-layer loop lands.

T7a puts the L per-layer host mirrors in one pinned allocation and replaces
T2's deferred scatter -- 2*L `index_copy_` calls, one per layer per tensor --
with ONE call per tensor using a vectorised index. The index arithmetic is
`global_row = layer * h_max * block_size + host_pos`, and getting it wrong
would write a rider's new token into **another layer's mirror** with nothing
raised: the shapes all match, the writes all land somewhere valid, and the only
symptom would be wrong attention output several steps later.

So this does not argue the arithmetic is right. It runs both implementations
over the same inputs and asserts the resulting mirrors are identical, including
that nothing outside the intended rows was touched.

Cases cover what the fast path keys off: uniform host_pos (the steady state),
a non-uniform step (must fall back), a gap in the pending layers (must fall
back), a single layer (must fall back), and n < the staging width.

CPU only -- the scatter is a host-side `index_copy_`, so no CUDA needed.

    ./.venv/bin/python benchmarks/kv_pipelining_bench/unified_mirror_equiv_test.py
"""
import sys

import torch

L, H_MAX, BS, NKV, HD = 6, 12, 4, 2, 3


def per_layer_reference(mir_k, mir_v, pending, wt_k, wt_v):
    """Exactly the pre-T7a loop, kept verbatim as the reference."""
    for li, host_pos, n in pending:
        hk, hv = mir_k[li], mir_v[li]
        hk.view(hk.shape[0] * BS, *hk.shape[2:]).index_copy_(
            0, host_pos, wt_k[li, :n])
        hv.view(hv.shape[0] * BS, *hv.shape[2:]).index_copy_(
            0, host_pos, wt_v[li, :n])


def vectorised(mir_k, mir_v, pending, wt_k, wt_v):
    """T7a's fast path, with its own guard conditions."""
    lis = [e[0] for e in pending]
    nl = len(lis)
    pos0, n0 = pending[0][1], pending[0][2]
    uniform = all(e[2] == n0 and torch.equal(e[1], pos0) for e in pending)
    if not (uniform and nl > 1 and lis == list(range(nl))):
        return False
    off = torch.arange(L, dtype=torch.int64) * (H_MAX * BS)
    idx = (pos0.view(1, -1) + off[:nl].view(-1, 1)).reshape(-1)
    mir_k.view(-1, NKV, HD).index_copy_(
        0, idx, wt_k[:nl, :n0].reshape(-1, NKV, HD))
    mir_v.view(-1, NKV, HD).index_copy_(
        0, idx, wt_v[:nl, :n0].reshape(-1, NKV, HD))
    return True


CASES = [
    ("uniform, all layers, full width",
     [(li, torch.tensor([3, 17, 29], dtype=torch.int64), 3) for li in range(L)],
     3, True),
    ("uniform, all layers, n < staging width",
     [(li, torch.tensor([5, 9], dtype=torch.int64), 2) for li in range(L)],
     2, True),
    ("uniform, one rider",
     [(li, torch.tensor([7], dtype=torch.int64), 1) for li in range(L)],
     1, True),
    ("uniform but only the first 3 layers wrote (the layer knob)",
     [(li, torch.tensor([2, 11], dtype=torch.int64), 2) for li in range(3)],
     2, True),
    # --- must FALL BACK ---
    ("non-uniform: one layer writes different rows",
     [(li, torch.tensor([3, 17] if li != 2 else [4, 18], dtype=torch.int64), 2)
      for li in range(L)], 2, False),
    ("non-uniform: one layer writes fewer rows",
     [(li, torch.tensor([3, 17], dtype=torch.int64), 2) for li in range(L - 1)]
     + [(L - 1, torch.tensor([3], dtype=torch.int64), 1)], 2, False),
    ("gap in the pending layers",
     [(li, torch.tensor([3, 17], dtype=torch.int64), 2)
      for li in (0, 1, 3, 4)], 2, False),
    ("single layer",
     [(0, torch.tensor([3, 17], dtype=torch.int64), 2)], 2, False),
]


def main():
    fails = 0
    print("T7a vectorised scatter vs the per-layer loop")
    print()
    for label, pending, nmax, expect_fast in CASES:
        torch.manual_seed(len(label))
        wt_k = torch.randn(L, max(nmax, 3), NKV, HD)
        wt_v = torch.randn(L, max(nmax, 3), NKV, HD)
        # the two mirrors start identical and full of a marker, so an untouched
        # row is distinguishable from a correctly-written one
        base_k = torch.full((L, H_MAX, BS, NKV, HD), -7.0)
        base_v = torch.full((L, H_MAX, BS, NKV, HD), -9.0)
        ref_k, ref_v = base_k.clone(), base_v.clone()
        new_k, new_v = base_k.clone(), base_v.clone()

        per_layer_reference(ref_k, ref_v, pending, wt_k, wt_v)
        took_fast = vectorised(new_k, new_v, pending, wt_k, wt_v)
        if not took_fast:
            per_layer_reference(new_k, new_v, pending, wt_k, wt_v)

        probs = []
        if took_fast != expect_fast:
            probs.append("fast=%s expected %s" % (took_fast, expect_fast))
        if not torch.equal(ref_k, new_k):
            bad = (ref_k != new_k).any(dim=(1, 2, 3, 4)).nonzero().flatten()
            probs.append("K differs in layers %s" % bad.tolist()[:4])
        if not torch.equal(ref_v, new_v):
            probs.append("V differs")
        # and the marker must survive everywhere the reference left it
        untouched = (ref_k == -7.0)
        if not torch.equal(new_k[untouched], ref_k[untouched]):
            probs.append("wrote outside the intended rows")

        if probs:
            fails += 1
            print("  FAIL %-52s %s" % (label, "; ".join(probs)))
        else:
            print("  ok   %-52s %s" % (label,
                  "vectorised, identical" if took_fast else "fell back"))

    # the point of the change: how many index_copy_ calls
    print()
    print("  calls per step at L=%d: per-layer 2L = %d -> vectorised 2"
          % (64, 2 * 64))
    print()
    print("T7a UNIFIED-MIRROR EQUIVALENCE %s"
          % ("PASS (%d cases)" % len(CASES) if not fails
             else "FAIL (%d)" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
