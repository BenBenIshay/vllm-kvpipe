#!/usr/bin/env python3
"""Unit test for HybridCpuStreamer's Step-2 compact rolling region.

Validates, in isolation (no server), that streaming the CPU cohort's per-layer
KV from the host mirror into a small `k_window`-slot compact rolling region
places every block at its compacted id bitwise-correctly, and that the
block-table remap is consistent. This de-risks the data movement before the
server integration (rung c), same pattern as the Step-0 numeric proof.

Run on the GPU node:
  srun --jobid=<J> --overlap --mem=0 --gres=gpu:1 bash -lc \
    'cd /mnt/beegfs/benishay/vllm-kvpipe && .venv/bin/python \
     benchmarks/kv_pipelining_bench/rolling_region_unit_test.py'
"""
from __future__ import annotations

import torch

from vllm.v1.hybrid_cpu_stream import HybridCpuStreamer

DEV = "cuda"
DT = torch.bfloat16


def main() -> None:
    torch.manual_seed(0)
    num_layers = 6
    num_blocks, block_size, nkv, hd = 8, 4, 2, 8
    k_window = 3
    cpu_blocks = [1, 3, 5, 6]          # resident ids the CPU cohort holds
    W = len(cpu_blocks)

    # Synthetic resident KV, one [2, num_blocks, block_size, nkv, hd] per layer.
    layer_names = [f"model.layers.{i}.attn" for i in range(num_layers)]
    kv = {
        ln: torch.randn(2, num_blocks, block_size, nkv, hd, device=DEV, dtype=DT)
        for ln in layer_names
    }

    st = HybridCpuStreamer(kv, layer_names)
    print(f"block_dim detected = {st._bd} (expect 1 for [2,blocks,...])")

    # Stage the cohort's resident blocks -> host mirror (D2H), then build the
    # compact rolling region and fix this step's cohort map.
    st.stage_blocks(cpu_blocks)
    st.build_rolling_region(k_window=k_window, w_max_blocks=W)
    st.set_cpu_cohort(cpu_blocks)

    sorted_ids = torch.tensor(sorted(cpu_blocks), dtype=torch.long, device=DEV)
    ok = True
    for idx, ln in enumerate(layer_names):
        slot = st.load_layer_rolling(ln, idx)          # host mirror -> slot idx%k
        assert slot is not None
        got = slot.index_select(1, torch.arange(W, device=DEV))     # [2, W, ...]
        want = kv[ln].index_select(1, sorted_ids)                   # [2, W, ...]
        eq = torch.equal(got, want)
        ok = ok and eq
        if not eq:
            print(f"  layer {idx} {ln}: MISMATCH max|diff|="
                  f"{(got.float()-want.float()).abs().max().item():.3e}")
    print(f"[rolling load] all {num_layers} layers compacted bitwise-correct: {ok}")

    # Block-table remap: resident ids -> compact [0:W) positions.
    remap = st.remap_rows([[1, 5], [3, 6], [6, 1, 3]])
    expect = [[0, 2], [1, 3], [3, 0, 1]]
    remap_ok = remap == expect
    print(f"[remap] {remap} == {expect}: {remap_ok}")

    print("=" * 60)
    if ok and remap_ok:
        print("ROLLING-REGION UNIT PASS: compacted rolling load + block-table "
              "remap are bitwise-correct -> Step-2 data movement is sound.")
    else:
        print(f"ROLLING-REGION UNIT FAIL: load_ok={ok} remap_ok={remap_ok}")


if __name__ == "__main__":
    main()
