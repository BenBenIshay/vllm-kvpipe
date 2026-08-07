# rung-d serve-more: complete + validated on A100 (2026-08-07)

The hybrid CPU tier now serves its requests from a **host-block-id KV pool**
instead of main-pool residency, so the CPU cohort's HBM is reclaimed and more
requests run concurrently — in one weight-shared forward, alongside the resident
HBM cohort. All on A100 80GB / Qwen2.5-0.5B, every path env-gated (default OFF).

## Mechanism — validated bitwise / token-identical

| Check | What it proves | Result |
|---|---|---|
| `rungd_host_pool_test.py` | pool (seed+grow+write+stream+read), multi-req, reuse-immune | bitwise, isolated |
| `=2` self-verify (RUNGD_FREE, NOWRITE=0) | pool attention == resident attention in-server | 576 layer-steps, `max\|diff\|=0` |
| `hybrid_rungd_real.py` (pool vs vanilla, REAL KV) | CPU cohort served entirely from the pool == vanilla generation | **token-identical**, coherent |
| real-KV test WITH the block cap active | reclaim doesn't corrupt | token-identical |

(Token comparison against the `PipelinedDecodeBenchConnector` is invalid — it
fake-fills KV; the pool stores the real projected keys and correctly diverges.
Validation uses `=2` numeric equality + real-KV-no-connector generation.)

## Serve-more — measured (`hybrid_rungd_servemore.py`)

Fixed **160-block** KV pool, 32 concurrent long-generation requests, `b_hbm_max=6`,
admission opened so the KV pool (not the roofline) is the binding constraint:

| | baseline (no rung-d) | **rung-d** |
|---|---|---|
| steady-state concurrent running | 12 | **32** |
| CPU-tier main-pool blocks (peak) | 126 | **48** |
| completed correctly | 32/32 | 32/32 |

**2.67× more concurrent requests at the same HBM**, all producing correct output.
The CPU cohort's decode blocks are reclaimed (48 vs 126) because its KV lives in
the host pool + a `k_window`-shared rolling region (num_layers/k_window ≈ 6× less
HBM per CPU request).

## Honest caveat (expected, and the GH200 motivation)

rung-d is **slower per step on A100** (wall 16.4s vs 6.0s): the correctness-first
pool does per-layer host↔GPU copies over PCIe (~25 GB/s). This is the *capacity*
win, not a *latency* win — the latency/throughput win requires NVLink-C2C
(~480 GB/s, GH200), which is the paper's headline and the reason to rent it. The
copies are also unoptimized (per-row Python loops); batching + async double-
buffering would cut the A100 overhead substantially.

## Components (all committed on `hybrid-kv-tiers`)

- `vllm/v1/hybrid_rungd_pool.py` — host-block-id KV pool.
- `gpu_model_runner._setup_rungd_pool_path` / `_maybe_setup_rungd_pool` — identify
  CPU decode rows on any step; seed from resident; build the compact rolling
  block table; slot_mapping=-1.
- `flash_attn.py::_pool_cpu_override` — standard pass + CPU-decode override
  (stream→write-redirect→compact-read from the pool).
- `scheduler.py` — cap a CPU-tier request's decode-loop block growth (reclaim).
