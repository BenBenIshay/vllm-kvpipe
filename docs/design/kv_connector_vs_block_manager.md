# KV Connector + KV Block Manager in vLLM — Session Summary

A consolidated reference covering the architecture, programming model, and
system-level tradeoffs of vLLM's KV connector framework and its interaction
with the KV block manager. Written for the layer-wise KV pipelining research
project.

---

## 1. Roles: KV Block Manager vs KV Connector

| Component | Role | Visibility |
|-----------|------|-----------|
| `KVCacheManager` (+ `block_pool`, coordinators) | **Allocator / bookkeeping** for paged KV blocks in HBM. Owns block IDs, prefix cache, eviction, refcounts. | GPU block IDs only. Doesn't know about DDR/NVMe/remote tiers. |
| `KVConnector` (`KVConnectorBase_V1`) | **Data movement & scheduling bridge** that fills/exports the manager-owned HBM blocks across tiers. | Scheduler side: request-level decisions. Worker side: paged HBM tensors + external tier. |

Mental model:

- **Manager** = who owns which HBM blocks.
- **Connector** = how KV bytes get into/out of those blocks across memory tiers (DDR / NVMe / remote GPU).

The connector **does not bypass the GPU/NIC DMA engines** — it **orchestrates** them.

---

## 2. Connector lifecycle (scheduler ↔ worker)

```text
SCHEDULER (CPU)
  get_num_new_matched_tokens(req)          → how many tokens already in external KV
  KVCacheManager.allocate_slots(...)       → reserve GPU block IDs
  update_state_after_alloc(req, blocks)    → connector tracks block IDs
  build_connector_meta(scheduler_output)   → per-step opaque metadata
  request_finished(req, block_ids)         → may delay free until async send done

WORKER (CPU process driving GPU)
  bind_connector_metadata(meta)
  start_load_kv(forward_context)           → kick off async loads (CPU→GPU / NIC→GPU)
  for each attention layer:
      wait_for_layer_load(layer_name)      → ensure that layer's KV is ready
      attention kernel runs on GPU
      save_kv_layer(layer_name, kv, attn)  → kick off async export of new KV
  wait_for_save()                          → drain pending saves
  get_finished(finished_req_ids)           → report which async xfers completed
```

`maybe_transfer_kv_layer` decorates `unified_attention` so the hooks fire
automatically around each layer.

---

## 3. Where code actually runs (CPU vs GPU)

| Layer | Runs on | Notes |
|-------|---------|-------|
| Scheduler logic, connector Python | **CPU** | Single Python process |
| `execute_model`, `model.forward()` | **CPU** | Walks layers, dispatches ops |
| Attention kernels (FlashAttention etc.) | **GPU SMs** | Launched from CPU, executed asynchronously |
| `copy_(..., non_blocking=True)` | CPU enqueues; **GPU copy engine** moves bytes | Used by K+1 pipelining |
| NCCL / NIXL / RDMA | CPU posts work; **NIC + GPU memory** do the bytes | Used by P2P NCCL, NIXL, Mooncake |
| KV tensors | **HBM** (paged) | Same surface attention uses |

Worker = **CPU driver of GPU work**. Attention isn't "Python on GPU"; it's
**CUDA launches enqueued from CPU**.

---

## 4. Connector ecosystem: who runs where

| Connector | Per-layer hooks? | Tier | Data path |
|-----------|------------------|------|-----------|
| `PrefillStagingPipelinedConnector` (this project) | **Yes** | CPU pinned DDR | `copy_` on dedicated `torch.cuda.Stream` (copy engine) |
| `LMCacheConnectorV1` | Optional (`use_layerwise`) | CPU / disk / remote via LMCache | `VLLMPagedMem*` connector scatter/gather (CUDA) |
| `NixlConnector` | No (bulk) | Remote GPU (P/D) | NIXL/UCX over RDMA, can register GPU memory directly |
| `MooncakeConnector` | No (bulk) | Remote node via Mooncake TransferEngine | RDMA, with CPU thread pools + asyncio |
| `P2pNcclConnector` | No (bulk) | Remote GPU | NCCL send/recv, Python injection into KV tensor |
| `OffloadingConnector` | No per-layer | Generic via `OffloadingSpec` | Async transfers per `LoadStoreSpec` |

Common pattern: **scheduler logic in scheduler process; worker logic in
worker process; bytes moved by hardware engines.**

---

## 5. NIXL / GPU-direct demystified

- **GPU-direct** = NIC/RDMA path can read/write **registered GPU HBM**
  without an extra CPU staging copy.
- **CPU still initiates** the transfer (`start_load_kv` posts NIXL
  READ/WRITE).
- Not a different connector API — same `start_load_kv` / `wait_for_save`
  hooks; what changes is the **transport** under the hood.
- vLLM NIXL hooks `wait_for_layer_load` / `save_kv_layer` are **no-ops**
  (bulk per request, not per layer).

---

## 6. CUDA streams, graphs, copy engines

### Streams

- **Logical ordered queue of GPU ops** (kernels, memcpys, events).
- Enqueue is **cheap (µs)**. Multiple streams can run concurrently when no
  false deps.
- Synchronization (`stream.synchronize()`, events) is what costs — drains
  overlap.

### Copy engine (CE)

- Separate hardware from SMs; handles **async memcpy** (H2D/D2H/D2D).
- **Submit:** ~µs. **Transfer:** bytes / bandwidth (e.g. 100 MB ≈ ms over PCIe).
- Many tiny copies → submission overhead dominates. Few large block-runs →
  bandwidth dominates (good).

### CUDA graphs

- Record once, **replay** many ops with far fewer driver submissions.
- Per-layer Python in connectors can disable/limit graphs.

### Sync cost: waiting for work vs mechanism

| Situation | Dominant cost |
|-----------|----------------|
| Sync after large copy/kernel | **Real transfer / kernel time** |
| Sync every layer with full queue | **Pipeline serialization (opportunity cost)** |
| Sync on idle GPU | **Driver latency (µs–tens of µs)** |
| `.item()` / `.tolist()` in hot path | **Drains entire queue ahead** |

"Sync = ms" usually = waiting for **real work**, not the sync call itself.

---

## 7. GIL vs GPU sync

- **GIL** = one Python bytecode thread at a time per process. C++/CUDA calls
  **release the GIL**, so GPU work overlaps with other threads — but only if
  you actually have other threads and avoid syncing.
- **GPU sync** = CPU (or another stream) waits for GPU completion. Cheap when
  work is done, expensive when it forces serialization.
- "Too much Python work" means: per-layer / per-request Python loops,
  `.item()`, fresh tensor allocation, big logging — all on the **main
  forward thread**.

---

## 8. How CUDA enqueue actually reaches the GPU

| Layer | Where |
|--------|--------|
| `libcudart` / PyTorch | **User space (your process)** — builds command descriptors |
| **User-mode driver (UMD)** | User space — batches/queues work |
| **Kernel-mode driver (`nvidia.ko`)** | OS kernel — manages contexts, pinning, GPU VA, submission, fences |
| **GPU command processor + firmware** | On device — schedules CE and SMs from command packets |

Submission path uses **MMIO / doorbells** and DMA of command buffers.
Steady-state submission is **batched**, not necessarily one ioctl per Python
op. **Kernel-mode = OS kernel (CPU privileged code)**, not a GPU kernel.

### RDMA contrast (important nuance)

- **InfiniBand / RoCE verbs:** after setup (MR registration etc.), the
  data-plane post is **fully user-space** (doorbell to NIC via mapped UAR).
- **CUDA:** **user/kernel split** with UMD + KMD; can avoid kernel per op
  via batching, but **not** the same path as verbs.
- **NIXL / Mooncake / GPUDirect RDMA:** combine both — RDMA verbs into
  **registered GPU HBM**. CPU initiates; NIC + GPU do the bytes.

---

## 9. K+1 layer-wise pipelining design (in this context)

- **HBM:** K+1 shared physical KV tensors (round-robin across layers).
- **CPU pinned DDR:** full per-layer history.
- **Dedicated `torch.cuda.Stream`:** async H2D prefetch + D2H staging.
- **Per-layer hooks:**
  - Prefill: `save_kv_layer` → stage to CPU.
  - Decode: `start_load_kv` + `wait_for_layer_load` → prefetch next layers
    from CPU.
- **Scheduler connector:** mostly transparent (returns 0 from
  `get_num_new_matched_tokens`), but tracks per-request GPU block IDs for
  span-based transfer.

This is the canonical example of: **manager allocates fewer HBM blocks;
connector keeps full state in CPU DDR; copy engine + dedicated stream
overlap PCIe with attention compute.**

---

## 10. LMCache vs this design (high-level)

| Dimension | LMCache | K+1 pipelining |
|-----------|----------|----------------------|
| Goal | Cross-request / instance KV reuse via external tiers | Maximize decode throughput at fixed HBM via overlap |
| HBM footprint | Full per-layer (typical) | ~1/(K+1) of layer tensors |
| External tier | Pluggable backend (CPU / disk / remote) | Local pinned DDR |
| Per-layer API | Optional (`use_layerwise` generators) | Required (always layerwise) |
| Transport | `VLLMPagedMem*GPUConnector` + LMCache engine | Direct `copy_` on dedicated CUDA stream |
| Block mapping | `slot_mapping` + chunk alignment | Physical block-id spans |
| Sweet spot | High prefix reuse, long sessions | Memory-bound decode, high concurrency on fixed HBM |

Hybrid idea worth researching: **LMCache for cold prefix + K+1 for hot
decode residency**, with manager owning eviction across both.

---

## 11. Key takeaways (one-liners)

1. **Manager = HBM block IDs; connector = move bytes for those blocks across tiers.**
2. **Connector worker hooks are CPU Python that orchestrate GPU/NIC engines.**
3. **NIXL "GPU-direct" = RDMA into registered HBM**, still CPU-initiated.
4. **CUDA stream = ordered GPU queue.** Enqueue is cheap; sync is what hurts.
5. **CUDA graph = replay many ops with fewer submissions.**
6. **Sync "ms" usually means waiting for real bytes/kernel work, not the call itself.**
7. **GIL = single Python thread for bytecode**; most CUDA calls release it.
8. **Kernel-mode = CPU OS kernel** (`nvidia.ko`); **GPU kernel = function running on SMs**.
9. **CUDA path = UMD + KMD + firmware**; **RDMA verbs path = user-space doorbell after setup** — different stacks with similar goals.
10. **The K+1 design is the "local DDR window" analogue of NIXL's "remote GPU window,"** both built on the same connector contract.

---

## 12. Suggested follow-ups

- Trace one **LMCache `retrieve_layer` step** vs one
  **`prefetch_to_gpu_block_runs`** step side-by-side.
- Profile expected sync points and overlap windows on H100 with Nsight Systems.
- Sketch a hybrid LMCache + K+1 design with tier-aware eviction in
  `KVCacheManager`.

---

## References (in-tree)

- `vllm/distributed/kv_transfer/kv_connector/v1/base.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/prefill_staging_pipelined_connector.py`
- `vllm/v1/kv_cache_pipelining.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/attention/utils/kv_transfer_utils.py`
- `vllm/v1/worker/kv_connector_model_runner_mixin.py`
- `docs/design/kv_cache_pipelining.md`
