#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hybrid tiered KV roofline: B_hbm, B_off_max, derived CPU pool sizing.

Underloaded -> R0 (all KV in HBM).  Overcommitted -> B_hbm HBM-resident requests
plus B_off CPU-tier requests where transfer fits in the decode hide window:

    B_off * S <= ITL_ms/1000 * X_eff / b_tok
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from kv_transfer_budget import MODEL_PROFILES, block_bytes

MODEL_ALIASES: dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

LLAMA8B_R0_ITL_MS: dict[int, float] = {
    1: 10.95,
    4: 11.67,
    8: 12.50,
    16: 14.63,
    32: 16.13,
    64: 18.06,
    256: 24.46,
    512: 42.06,
}

DEFAULT_B_HBM_MAX = 49


@dataclass(frozen=True)
class HybridRow:
    seq_tokens: int
    b_hbm: int
    itl_ms: float
    theta_tok_slots: float
    b_off_max: int
    total_served: int
    cpu_pool_gib: float


def resolve_model(model: str) -> tuple[str, dict]:
    key = MODEL_ALIASES.get(model, model)
    prof = MODEL_PROFILES.get(key)
    if prof is None:
        raise SystemExit(f"Unknown model {model!r}; add to MODEL_PROFILES")
    return key, prof


def bytes_per_token(prof: dict) -> int:
    return (
        prof["num_layers"]
        * 2
        * prof["num_kv_heads"]
        * prof["head_dim"]
        * prof["dtype_bytes"]
    )


def itl_at_batch(batch: int, itl_table: dict[int, float]) -> float:
    if batch in itl_table:
        return itl_table[batch]
    keys = sorted(itl_table)
    if batch <= keys[0]:
        return itl_table[keys[0]]
    if batch >= keys[-1]:
        return itl_table[keys[-1]]
    lo = max(k for k in keys if k <= batch)
    hi = min(k for k in keys if k >= batch)
    if lo == hi:
        return itl_table[lo]
    t = (batch - lo) / (hi - lo)
    return itl_table[lo] * (1 - t) + itl_table[hi] * t


def hide_budget_tok_slots(itl_ms: float, x_eff_gbps: float, b_tok: int) -> float:
    return (itl_ms / 1000.0) * x_eff_gbps * 1e9 / b_tok


def b_off_max(seq_tokens: int, itl_ms: float, x_eff_gbps: float, b_tok: int) -> int:
    theta = hide_budget_tok_slots(itl_ms, x_eff_gbps, b_tok)
    if seq_tokens <= 0:
        return 0
    return max(0, int(theta // seq_tokens))


def cpu_pool_gib(
    *,
    b_off: int,
    seq_tokens: int,
    block_size: int,
    prof: dict,
) -> float:
    if b_off <= 0:
        return 0.0
    b_bytes = block_bytes(
        block_size=block_size,
        num_kv_heads=prof["num_kv_heads"],
        head_dim=prof["head_dim"],
        dtype_bytes=prof["dtype_bytes"],
    )
    blocks = math.ceil(seq_tokens / block_size)
    per_req = prof["num_layers"] * blocks * b_bytes
    return (b_off * per_req) / (1 << 30)


def hybrid_row(
    *,
    seq_tokens: int,
    b_hbm: int,
    x_eff_gbps: float,
    prof: dict,
    block_size: int,
    itl_table: dict[int, float],
) -> HybridRow:
    b_tok = bytes_per_token(prof)
    itl = itl_at_batch(b_hbm, itl_table)
    theta = hide_budget_tok_slots(itl, x_eff_gbps, b_tok)
    off = b_off_max(seq_tokens, itl, x_eff_gbps, b_tok)
    pool = cpu_pool_gib(
        b_off=off, seq_tokens=seq_tokens, block_size=block_size, prof=prof
    )
    return HybridRow(
        seq_tokens=seq_tokens,
        b_hbm=b_hbm,
        itl_ms=itl,
        theta_tok_slots=theta,
        b_off_max=off,
        total_served=b_hbm + off,
        cpu_pool_gib=pool,
    )


def print_table(title: str, rows: list[HybridRow], x_eff: float, b_tok_kib: float) -> None:
    print(f"\n{title}")
    print(f"  X_eff = {x_eff:.1f} GB/s, b_tok = {b_tok_kib:.0f} KiB/token/step")
    print("+--------+--------+--------+----------------+----------+-------------+------------------+")
    print("| S tok  | B_hbm  | ITL ms | Theta tok-slots| B_off max| B_hbm+B_off | CPU pool (GiB)   |")
    print("+--------+--------+--------+----------------+----------+-------------+------------------+")
    for r in rows:
        print(
            f"| {r.seq_tokens:6d} | {r.b_hbm:6d} | {r.itl_ms:6.2f} | "
            f"{r.theta_tok_slots:14.0f} | {r.b_off_max:8d} | {r.total_served:11d} | "
            f"{r.cpu_pool_gib:16.2f} |"
        )
    print("+--------+--------+--------+----------------+----------+-------------+------------------+")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--x-eff-gbps", type=float, default=25.1)
    ap.add_argument("--b-hbm-max", type=int, default=DEFAULT_B_HBM_MAX)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 8320])
    args = ap.parse_args()

    _, prof = resolve_model(args.model)
    b_tok_kib = bytes_per_token(prof) / 1024

    print(f"Model: {args.model}")
    print(f"B_hbm_max: {args.b_hbm_max}")

    rows = [
        hybrid_row(
            seq_tokens=s,
            b_hbm=args.b_hbm_max,
            x_eff_gbps=args.x_eff_gbps,
            prof=prof,
            block_size=args.block_size,
            itl_table=LLAMA8B_R0_ITL_MS,
        )
        for s in args.seq_lens
    ]
    print_table("Hybrid tier sizing (B_hbm at capacity)", rows, args.x_eff_gbps, b_tok_kib)

    print("\nB_off_max vs B_hbm at S=512:")
    print("+--------+--------+----------------+----------+")
    print("| B_hbm  | ITL ms | Theta tok-slots| B_off max|")
    print("+--------+--------+----------------+----------+")
    for b in (1, 4, 8, 16, 32, 49, 64):
        r = hybrid_row(
            seq_tokens=512,
            b_hbm=b,
            x_eff_gbps=args.x_eff_gbps,
            prof=prof,
            block_size=args.block_size,
            itl_table=LLAMA8B_R0_ITL_MS,
        )
        print(
            f"| {r.b_hbm:6d} | {r.itl_ms:6.2f} | {r.theta_tok_slots:14.0f} | "
            f"{r.b_off_max:8d} |"
        )
    print("+--------+--------+----------------+----------+")


if __name__ == "__main__":
    main()
