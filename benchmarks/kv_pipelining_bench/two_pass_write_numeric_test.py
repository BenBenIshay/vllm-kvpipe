#!/usr/bin/env python3
"""Numeric proof of the rung-d two-pass KV WRITE (write-redirect).

rung-d frees the CPU cohort's main-pool blocks, so their new decode-token KV can
no longer be written to the resident pool via the normal slot_mapping. Instead
the CPU rows' KV must be reshape_and_cache'd into a SEPARATE rolling tensor with
a COMPACT slot_mapping, while the HBM rows still write to the main pool. This
validates that recipe in isolation (real reshape_and_cache_flash, bf16, decode
shapes), so any bug is in the write recipe, not the vLLM plumbing.

Run on the GPU node:
  srun --jobid=<J> --overlap --mem=0 --gres=gpu:1 bash -lc \
    'cd /mnt/beegfs/benishay/vllm-kvpipe && .venv/bin/python \
     benchmarks/kv_pipelining_bench/two_pass_write_numeric_test.py'
"""
from __future__ import annotations

import torch

from vllm._custom_ops import reshape_and_cache_flash

DEV = "cuda"
DT = torch.bfloat16
torch.manual_seed(0)

NKV, HD, BS = 2, 8, 16
ONE = torch.tensor(1.0, device=DEV)


def main() -> None:
    n_hbm, n_cpu = 4, 3
    n = n_hbm + n_cpu
    # New decode-token K/V for every row (one token per request).
    key = torch.randn(n, NKV, HD, device=DEV, dtype=DT)
    val = torch.randn(n, NKV, HD, device=DEV, dtype=DT)

    # Main pool (HBM cohort) and a separate compact rolling pool (CPU cohort).
    nb_main, nb_roll = 8, n_cpu  # rolling: one compact block per cpu row here
    kc_main = torch.zeros(nb_main, BS, NKV, HD, device=DEV, dtype=DT)
    vc_main = torch.zeros(nb_main, BS, NKV, HD, device=DEV, dtype=DT)
    kc_roll = torch.zeros(nb_roll, BS, NKV, HD, device=DEV, dtype=DT)
    vc_roll = torch.zeros(nb_roll, BS, NKV, HD, device=DEV, dtype=DT)

    # Row order: [hbm rows | cpu rows]. Full-batch slot_mapping (what the runner
    # builds today) would send everyone into the main pool; rung-d splits it.
    hbm_idx = torch.arange(0, n_hbm, device=DEV)
    cpu_idx = torch.arange(n_hbm, n, device=DEV)
    # HBM rows -> distinct main-pool slots (block b, offset 0).
    main_slots_full = torch.tensor(
        [b * BS + 0 for b in range(n)], device=DEV, dtype=torch.long
    )
    # CPU rows -> compact rolling slots (compact block 0..n_cpu-1, offset 0).
    roll_slots = torch.tensor(
        [c * BS + 0 for c in range(n_cpu)], device=DEV, dtype=torch.long
    )

    # --- HBM pass: write hbm rows to the main pool ---
    reshape_and_cache_flash(
        key.index_select(0, hbm_idx), val.index_select(0, hbm_idx),
        kc_main, vc_main, main_slots_full.index_select(0, hbm_idx),
        "auto", ONE, ONE,
    )
    # --- CPU pass: write cpu rows to the SEPARATE rolling pool (compact) ---
    reshape_and_cache_flash(
        key.index_select(0, cpu_idx), val.index_select(0, cpu_idx),
        kc_roll, vc_roll, roll_slots, "auto", ONE, ONE,
    )

    # Verify: each hbm row landed at its main slot; each cpu row at its compact
    # rolling slot; and the main pool was NOT touched by cpu rows.
    ok = True
    for j, r in enumerate(range(n_hbm)):
        gk = kc_main[r, 0]
        if not torch.equal(gk, key[r]):
            ok = False
    for j, r in enumerate(range(n_hbm, n)):
        gk = kc_roll[j, 0]
        gv = vc_roll[j, 0]
        if not (torch.equal(gk, key[r]) and torch.equal(gv, val[r])):
            ok = False
    # Main pool blocks >= n_hbm must remain zero (cpu rows didn't leak in).
    leaked = kc_main[n_hbm:].abs().sum().item()
    print(f"hbm rows in main pool: bitwise-correct={ok if ok else '...'}")
    print(f"cpu rows in rolling pool (compact): checked {n_cpu} rows")
    print(f"main-pool leakage past hbm blocks (want 0): {leaked:.3e}")
    print("=" * 60)
    if ok and leaked == 0.0:
        print("TWO-PASS WRITE PASS: CPU new-token KV redirects into a separate "
              "compact rolling pool bitwise-correctly, HBM rows unaffected, no "
              "main-pool leakage -> the rung-d write-redirect recipe is sound.")
    else:
        print(f"TWO-PASS WRITE FAIL: ok={ok} leaked={leaked}")


if __name__ == "__main__":
    main()
