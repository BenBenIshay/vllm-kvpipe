#!/usr/bin/env python3
"""Compute fp16 K+V element count and MiB for one layer, one paged block."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))
from kv_sizing import numel_from_kv_block_one_layer as kv_numel  # noqa: E402

LLAMA_3_1_8B = {"block_tokens": 16, "n_kv_heads": 8, "head_dim": 128}
QWEN2_5_14B = {"block_tokens": 16, "n_kv_heads": 2, "head_dim": 128}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--block-tokens", type=int, default=16)
    p.add_argument("--n-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--preset",
        choices=["", "llama-3.1-8b", "qwen2.5-14b"],
        default="",
    )
    args = p.parse_args()

    if args.preset == "llama-3.1-8b":
        t, h, d = (
            LLAMA_3_1_8B["block_tokens"],
            LLAMA_3_1_8B["n_kv_heads"],
            LLAMA_3_1_8B["head_dim"],
        )
    elif args.preset == "qwen2.5-14b":
        t, h, d = (
            QWEN2_5_14B["block_tokens"],
            QWEN2_5_14B["n_kv_heads"],
            QWEN2_5_14B["head_dim"],
        )
    else:
        t, h, d = args.block_tokens, args.n_kv_heads, args.head_dim

    n = kv_numel(t, h, d)
    mib = n * 2 / (1024 * 1024)
    if args.json:
        print(
            json.dumps(
                {
                    "block_tokens": t,
                    "n_kv_heads": h,
                    "head_dim": d,
                    "numel": n,
                    "mib": round(mib, 4),
                }
            )
        )
    else:
        print(
            f"one layer, one block: numel={n} fp16, ~{mib:.4f} MiB, "
            f"block_tokens={t} n_kv_heads={h} head_dim={d}"
        )


if __name__ == "__main__":
    main()
