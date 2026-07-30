# rung-d integration plan: per-request CPU-tier KV routing (serve-more on A100)

Goal: CPU-tier requests consume ~0 main-pool blocks; their KV lives in a
connector-owned rolling-K region (k_window layer-copies) + host, served via the
validated two-pass attention. Result: with a main pool that fits B_hbm resident
requests, serve B_hbm + B_off concurrently, correctly.

Why custom: vLLM's KV config is per-LAYER (`shared_by` → k_window tensors shared
by ALL requests); the block_pool is single-region. Per-REQUEST routing (HBM →
full-N, CPU → rolling-K) is not native. So the CPU cohort needs a connector-
owned KV subsystem. All component recipes are already bitwise-validated:
compact read (rung-c), write-redirect, rolling load, host mirror.

## Memory model
- Main pool: unchanged, HBM cohort, full-N (num_layers tensors).
- Rolling-K region (connector-owned): k_window tensors, each [2, W_max, bs, nkv,
  hd]; layer L uses tensor L%k, streamed from host just before its attention.
- Host mirror: CPU cohort full KV, all layers (HybridCpuStreamer already has it).
- CPU cohort's main-pool blocks: prefill resident, then offloaded to host +
  released; decode consumes ~0 main blocks.

## Increments (each gated VLLM_KVPIPE_HYBRID_RUNGD=1, default OFF; test before next)
1. **Scheduler bypass** — CPU-tier requests, once in decode, stop growing their
   main-pool allocation (cap at their post-prefill footprint, or release +
   pin-to-host). GATE: HBM cohort unaffected (bitwise vs today); pool free-block
   count rises with CPU load. [scheduler.py allocate path + get_req_tier]
2. **Runner rolling block table/slot_mapping for CPU rows** — build the CPU
   cohort's rolling block table (compact, rung-c) + a rolling slot_mapping into
   the rolling-K region. Carry on the split. GATE: shapes/values match a hand
   computation (debug log).
3. **Write-redirect live** — CPU rows' new-token KV → rolling-K tensor(L%k) via a
   second reshape_and_cache (recipe c3de6ada9), then persist to host. GATE:
   numeric — rolling/host hold the new token.
4. **Read live** — CPU two-pass reads rolling-K tensor(L%k), streamed from host
   (load_layer_rolling). GATE: =2 verify CPU-via-rolling == resident bitwise
   (CPU still resident as backstop for this gate).
5. **Free CPU main blocks** — release the CPU cohort's main-pool blocks post-
   prefill (connector offload path). GATE: CPU token output == resident run
   (greedy, distinct prompts); no crash; pool free rises.
6. **Serve-more measurement** — small main pool sized for B_hbm; admit
   B_hbm+B_off; MEASURE served concurrency > B_hbm and correct outputs. This is
   the A100 "beneficial" result. [hybrid_tier_occupancy.py → served-concurrency]

## Status (2026-07-30)
- **Increments 1-4 (read path) DONE + bitwise PASS (commit 5a8a2b202):** CPU cohort
  served from the k_window=4 host-streamed rolling region; =2 verify 576 layer-
  steps `max|diff|=0`. Current-token capture via `save_layer` (resident backstop).
- **Increment 5 (free CPU main blocks) — NEXT, precise path:** use
  `block_pool.free_blocks` + null the main block table for CPU-tier reqs. Safe
  because CPU attention uses the rolling block table, not the main one. Needs:
  seed host once → free old blocks → write-redirect the decode token to host +
  rolling slot (drop the resident write-head) OR keep a 1-block write head. The
  general case (seq grows past seeded blocks) needs a small rolling block
  allocator; a bounded test (generation fits the last block) demonstrates serve-
  more first. Delicate (live-block free) → gate on CPU-token == resident.
- **Increment 6:** small pool fits B_hbm, admit B_hbm+B_off, measure served
  concurrency > B_hbm + correct tokens = the A100 "beneficial" result.
- Node 2053223/nova06. See state-hybrid-kv-execution memory.
