#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print expected KV transfer volumes for pipelined decode (sanity / paper)."""
from __future__ import annotations

import argparse
import math

# HuggingFace-style defaults used in admission_predictor / evals.
MODEL_PROFILES: dict[str, dict] = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "dtype_bytes": 2,
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "num_layers": 24,
        "num_kv_heads": 2,
        "head_dim": 64,
        "dtype_bytes": 2,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "num_layers": 28,
        "num_kv_heads": 2,
        "head_dim": 128,
        "dtype_bytes": 2,
    },
}


def block_bytes(
    *,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> int:
    """K+V bytes for one paged block (vLLM FlashAttention layout)."""
    return 2 * block_size * num_kv_heads * head_dim * dtype_bytes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--k-window", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Active decode requests (for merged prefetch span).")
    ap.add_argument("--pcie-gbps", type=float, default=15.0,
                    help="Assumed effective PCIe BW for lower-bound latency.")
    args = ap.parse_args()

    prof = MODEL_PROFILES.get(args.model)
    if prof is None:
        raise SystemExit(
            f"Unknown model {args.model!r}; add to MODEL_PROFILES in this file."
        )

    seq_tokens = args.input_len + args.output_len
    blocks_per_req = math.ceil(seq_tokens / args.block_size)
    b_bytes = block_bytes(
        block_size=args.block_size,
        num_kv_heads=prof["num_kv_heads"],
        head_dim=prof["head_dim"],
        dtype_bytes=prof["dtype_bytes"],
    )
    per_layer = blocks_per_req * b_bytes
    num_layers = prof["num_layers"]
    stored_cpu_per_req = num_layers * per_layer

    # Connector steady-state (block prefetch uses merged kv_block_runs).
    blocks_merged = blocks_per_req * args.concurrency
    h2d_per_layer = blocks_merged * b_bytes
    h2d_per_token_step = num_layers * h2d_per_layer
    d2h_per_token_step = num_layers * b_bytes * args.concurrency  # ~1 block / req / layer

    pcie_s = h2d_per_token_step / (args.pcie_gbps * 1e9)

    print(f"Model:              {args.model}")
    print(f"Layers / KV heads:  {num_layers} / {prof['num_kv_heads']}")
    print(f"Sequence tokens:    {seq_tokens} ({args.input_len}+{args.output_len})")
    print(f"Blocks/request:     {blocks_per_req} (block_size={args.block_size})")
    print(f"Bytes/block (K+V):  {b_bytes:,} ({b_bytes / 1024:.1f} KiB)")
    print()
    print(f"CPU storage / request (all layers): {stored_cpu_per_req / 1e6:.2f} MB")
    print(f"k_window:           {args.k_window}  (physical tensors: {args.k_window + 1})")
    print(f"Concurrency:        {args.concurrency}")
    print()
    print("Per batched decode step (one new token / request, block-mode estimate):")
    print(f"  H2D per layer (merged spans): {h2d_per_layer / 1e6:.3f} MB")
    print(f"  H2D all layers:               {h2d_per_token_step / 1e6:.2f} MB")
    print(f"  D2H all layers (≈1 block/req): {d2h_per_token_step / 1e6:.3f} MB")
    print(f"  PCIe lower bound (@ {args.pcie_gbps} GB/s): {pcie_s * 1000:.1f} ms (H2D only)")
    print()
    print("Admission_predictor rough formula (k_window * KV per layer) — NOT what")
    print("the connector moves per step:")
    naive = (
        args.k_window
        * 2
        * prof["num_kv_heads"]
        * prof["head_dim"]
        * prof["dtype_bytes"]
    )
    print(f"  naive bytes/step/seq: {naive:,} ({naive / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
