# Mixed-batch dual-KV decode (Solution 1) — design & why the core is an allocator change

Status as of 2026-07-28. Grounded in a full read of the execution path
(gpu_model_runner, gpu_input_batch, block_table, attention/layer, flash_attn,
kv_pipelining_direct_window, pipelined_decode_bench_connector, cpu_kv_table,
kv_cache_utils).

## Goal

Serve, in one decode forward, `B_hbm` **HBM-tier** requests (full KV resident in
HBM across all N layers) together with `B_off` **CPU-tier** requests (only a
rolling K-of-N layer window resident in HBM, the rest streamed from CPU per
layer). Weights are read once for the whole batch (decode is weight-BW-bound —
that is the win). A CPU-tier request then costs ≈ K/N of the HBM a resident
request needs, so the node serves more than HBM alone fits.

## What already exists (committed on `hybrid-kv-tiers`)

- Roofline (`vllm/v1/hybrid_kv_roofline.py`) + admission that tags each request
  `hbm`|`cpu` by the PCIe hide budget (`transfer_aware_admission._should_admit_hybrid`).
- Tier tags plumbed scheduler → `SchedulerOutput.req_kv_tiers` → connector
  (`cpu_decode_req_ids`, `hbm_decode_req_ids`).
- All-HBM fast path: when a step has only hbm reqs, the connector skips all
  pipelining hooks and the model reads full HBM KV (`_hybrid_hbm_only_decode`).
- All-CPU rolling window: the existing `pipelined` / decode-only path.

## The irreducible blocker (verified in code)

The HBM memory **layout is a global, per-model-instance decision, not
per-request** — decided once in `kv_cache_utils.get_kv_cache_config_from_groups`
via `_get_pipelining_block_inflation`:

- **Rolling layout** (decode-only / `pipelined`): allocates only
  `num_physical_tensors = K` (or K+1) `KVCacheTensor`s, each `shared_by` a
  round-robin set of layers (`layer i → tensor i%K`, kv_cache_utils.py 1254-1269).
  This is where the K/N saving comes from — but it means **no** request can be
  fully resident: there is physically no per-layer storage for N layers at once.
- **Full layout** (what `enable_hybrid_kv_tiers` selects today — returns
  `(False,0,False)` at kv_cache_utils.py 1101-1106): one full tensor per layer,
  everyone fully resident → **no capacity win** (CPU-tier reqs still hold full
  HBM blocks).

A single `Attention` layer binds exactly **one** KV pool
(`self.kv_cache[ve]`, attention/layer.py 382/397), and the block table is built
**once per step, shared across all layers in a KV group**
(gpu_model_runner 1741-1742). So "some rows persistent, some rows rolling, in the
same layer" cannot be expressed in the current 1-tensor-per-layer contract.

## Required design — dual-region per-layer tensor + two-pass attention

Carve **each per-layer KV tensor** into two block ranges:

```
per-layer tensor = [ persistent region P |  rolling window region W ]
                     blocks [0 : P)          blocks [P : P+W)
```

- `P` sized for the HBM cohort: `B_hbm_max * blocks_per_req`. HBM-tier requests'
  block-table rows point into `[0:P)`, **static across all N layers** → ordinary
  paged attention over a sub-range. Full residency, never evicted.
- `W` sized for one layer of the CPU cohort's window:
  `~B_off_max * blocks_per_req`. CPU-tier requests' block-table rows point into
  `[P:P+W)` and are **rewritten per layer** as the connector streams layer i's
  CPU-tier KV into `W` (H2D before attention, D2H after). Only the current
  window layer of the CPU cohort is in HBM; all N layers live in `CpuKvTable`.

HBM held ≈ `N·P + N·W` (W reused across layers), vs full residency of everyone
`N·(B_hbm+B_off)·bpr`. Since `W` is one layer of the CPU cohort, this is the
K/N-style saving for the CPU cohort while the HBM cohort stays fully resident.

### The changes this forces (why it is not surgical)

1. **KV-cache config / allocation** (`kv_cache_utils.py`,
   `kv_cache_interface.py`): a new "dual-region" tensor spec — one tensor per
   layer sized `P+W`, with metadata recording the P/W split. New branch in
   `get_kv_cache_config_from_groups` and a new mode in
   `_get_pipelining_block_inflation`.
2. **Block-pool / manager / admission**: the scheduler must allocate HBM-tier
   blocks from `[0:P)` and CPU-tier window blocks from `[P:P+W)` — two logical
   sub-pools with independent caps (ties into the roofline `B_off_max` /
   `hybrid_b_hbm_max`).
3. **Per-request tier tag through `InputBatch`** (mechanical): `tier` array
   written in `add_request` (~342, beside `block_table.add_row`), kept coherent
   in `remove_request` (469), `swap_states` (523, above the pooling early-return
   at 583), `condense` (626, above 708); origin field on `CachedRequestState`
   populated from `SchedulerOutput.req_kv_tiers` in `_update_states`. Reorder
   (`reorder_batch_to_split_decodes_and_prefills` ~816) gains tier as a second
   sort key so hbm rows `[0:n_hbm)` and cpu rows `[n_hbm:)` are contiguous.
4. **Two-pass attention** (per layer, eager only — bench already
   `--enforce-eager`): slice `q/out/cu_seqlens_q/seqused_k/block_table` by tier
   and call `flash_attn_varlen_func` twice — pass 1 HBM rows over `[0:P)`, pass 2
   CPU rows over the rewritten `[P:P+W)` window — with re-based `cu_seqlens_q`.
   `reshape_and_cache_flash` for the new decode token also splits (HBM token → its
   persistent block; CPU token → the window block for this layer). Cascade/DCP
   and the CUDA-graph `update_block_table` fast path each assume a single block
   table and would need the same split or must be disabled.
5. **Per-layer CPU-tier block-table rewrite**: no hook exists today (block table
   built once per step). Either split CPU-tier layers into their own KV-cache
   group so `_build_attn_group_metadata` runs per layer, or add a per-layer
   block-table variant for the CPU cohort keyed off the connector's window bind.
6. **Connector 3-way dispatch**: tier-split `decode_physical_block_ids` to CPU
   blocks only (build side, connector ~943-958) so the roller never touches HBM
   blocks; relax the `CpuKvTable` gate so the table is created in hybrid mode.

Items 1, 2, 4, 5 are the deep, bitwise-correctness-sensitive parts and touch the
most delicate vLLM internals (KV allocation, block manager, attention backend).
Realistic effort: multi-week, with a strict correctness ladder.

## Verification ladder (strict)

- (a) all-HBM cohort == `vanilla` **bitwise** (same logits/tokens).
- (b) all-CPU == current `pipelined` pipeline.
- (c) 1-HBM + 1-CPU mixed: each cohort's outputs match its per-cohort reference.
- (d) overcommit (B_hbm+B_off > HBM-only capacity): runs without OOM and ITL
  stays inside the hide-window.

## Is the direct mixed measurement worth it on A100?

On A100/PCIe4 (X_eff ≈ 25 GB/s) the roofline gives small `B_off` (≈6 at
B_hbm=49), so the A100 composite win is small **and already bounded** by the two
separately-measured curves:
`tok/s_hybrid ≈ B_hbm/ITL_R0(B_hbm) + B_off/ITL_R1(B_off)` (see
`THEORY_VS_MEASURED.md`). The mixed forward mainly *demonstrates* a value the
two-curve bound + roofline already establish. Its real payoff is on fast fabric
(GH200/GB200, X_eff ≈ 480-512 GB/s) where `B_off` is large. This is why the
measurement suite (hide-boundary R0 vs R1 + roofline projection) is the primary
evidence, and the dual-region allocator is the scoped next step whose direct
value lands on fast interconnect.
