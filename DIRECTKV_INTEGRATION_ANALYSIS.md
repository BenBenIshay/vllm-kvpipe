# DirectKV (OSDI'26) — can it host a mixture? Source analysis

Repo: `github.com/shutianluo/DirectKV` (cloned, read 2026-08-12). Paper read in
full. This answers two architectural questions that decide our positioning:

1. Can their kernel be invoked on a **subset** of the batch, or does it assume
   all requests' KV are CPU-resident?
2. Can their **pool** hold some requests on the host while others stay in HBM?

**Both: yes.** Their architecture already does the first, and the second is a
constructor-level change. Ben's concern — that a naive mixture would ruin their
CPU-aware tiling — does not apply, because nothing mixes *inside* a kernel.

---

## 1. Subset invocation — already their normal operating mode

`directkv/directkv_backend.py` header, verbatim:

> "forward_decode buckets requests by 64-aligned seqlen, **one kernel call/bucket**."

And the dispatch itself (`forward_decode`, ~line 560):

```python
sdpa_idxs   = []  # requests where S_past % 64 != 0
kernel_idxs = []  # requests where S_past % 64 == 0
for i in range(bs):
    S_past = meta.seq_lens_cpu[i] - 1
    if S_past % _TILE != 0: sdpa_idxs.append(i)
    else:                   kernel_idxs.append(i)

for i in sdpa_idxs:                     # implementation A
    o_i = self._sdpa_decode_directkv(i, q, k, v, layer, meta)
    output_slots.append((i, o_i))

if kernel_idxs:                         # implementation B, one call per bucket
    buckets = defaultdict(list)
    for i in kernel_idxs:
        buckets[_pad64(meta.seq_lens_cpu[i] - 1)].append(i)
    for S_past_pad, idxs in buckets.items():
        req_ids = [meta.req_pool_indices_cpu[i] for i in idxs]
        K_cpu_batch, V_cpu_batch = self._pool.get_batch_kv(req_ids, lid, S_total_pad)
        O = self._directkv_fwd(...)
```

In **one decode step** they already run **two different attention
implementations over disjoint subsets of the same batch**, plus N bucket calls
to the SM90 kernel, reassembling via `output_slots: List[Tuple[int, Tensor]]`.
The kernel is invoked with an arbitrary `req_ids` list.

**Consequence for us:** a mixture — their kernel on host-tier rows, standard FA
on HBM-resident rows — is *structurally identical to what they already do*.
Their CPU-memory-aware tiling, SMEM pipelining and warp-group producer/consumer
schedule are all **per kernel call**, and each call sees a uniform KV source.
Nothing is disturbed. The cost of the extra call is what we measured in
`DIRECTION_A_GO_NOGO.md` Gate 1: **+4.7%** per request.

## 2. Pool extension — per-request rows make it trivial

`directkv/directkv_request_pool.py`:

```python
class DirectKVRequestPool:
    """Shared pre-allocated CPU-pinned KV pool with per-request row assignment.
       One pool instance per layer. Shape [max_reqs, max_seqlen_pad, NH_kv, D].
       Requests are assigned to rows via a free-list. The kernel can access
       any row as a [1, S_total, NH_kv, head_dim] view without any memory copy."""

    def alloc(self, req_id): row = self._free.pop(); self._req_to_row[req_id] = row
    def free(self, req_id):  ...
    def row_of(self, req_id) -> int: ...
```

The enabler: **placement is per-request, through an explicit `req_id → row`
map, with no cross-request coupling.** Each row is an independent
`[1, S_total, NH_kv, D]` view. Nothing in the pool assumes the whole batch lives
together.

A mixture therefore needs only:
1. a second pool instance with the same interface, allocated in **HBM**
   (`device='cuda'`, no `pin_memory`) — the class is placement-agnostic apart
   from its constructor;
2. tier-aware `alloc(req_id)` routing to one pool or the other;
3. partitioning in `forward_decode` by *which pool the request lives in* —
   the same partitioning pattern they already use for SDPA-vs-kernel;
4. their kernel for host rows, standard FA for HBM rows.

**No kernel modification required.** This is the strongest evidence that our
contribution sits cleanly above their mechanism.

## 3. Constraints found in the artifact (log these; several are load-bearing)

- **SM90a / Hopper only.** README: *"Any Hopper GPU (SM90) can run all unit
  tests."* `directKV_kernel.cuh` is an SM90a warp-group MMA kernel; built on
  FlashAttention-3 with CUTLASS/CuTe Hopper support; uses **TMA**, Hopper-only
  hardware. **It cannot run on A100 at all** — any head-to-head needs Hopper.
- **`head_dim == 128`, bf16 only.**
- **`--disable-radix-cache` required** — no prefix caching. A real deployment
  limitation, since prefix reuse is standard and valuable.
- **`--chunked-prefill-size` must be a multiple of 64.**
- **The fast kernel is gated on `S_past % 64 == 0`.** Their comment gives the
  reason: when `S_past` is not 64-aligned, positions `[S_past:S_past_pad]` in the
  CPU pool hold *uninitialised zeros*, so correctness forces the SDPA fallback.
  **During decode `S_past` grows by 1 per step, so a request is kernel-eligible
  once every 64 steps (~1.6%)**; the other ~98% go through
  `_sdpa_decode_directkv`, invoked in a **per-request Python loop**. Their
  benchmarks use ShareGPT/Alpaca with arbitrary prompt lengths, so nothing keeps
  requests aligned.
  *Caveat:* this is the published artifact, not their measurement harness. We
  cannot check what fraction was on the fast path without running it, which
  needs Hopper. Do **not** assert this about their reported numbers.
- **Row zeroing on admission.** `alloc()` zeroes the request's row across all
  layers: `num_layers × 2 × row_bytes`. For 70B at 8K that is ~2.7 GB of host
  memset **per admission** — meaningful under request churn. Implementation
  detail, easily fixed, but it is in the artifact.

## 4. What this settles for our positioning

- Ben's tiling concern: **resolved.** Mix across kernels, never within one.
- Option "reverse-engineer whether their design admits a mixture": **answered
  yes**, at the source level, for free, with no GPU.
- Option "prove our sliding window beats their zero-copy": **still not
  recommended.** It needs Hopper (their kernel won't run on A100), it is a
  kernel-engineering fight on their home turf, and given the constraints above
  it risks becoming a critique of an *artifact* rather than of an *idea* —
  which reviewers dislike and which is beside our contribution.
- The strongest frame remains Ben's own: **we degenerate to DirectKV in the
  big-IO regime (never worse) and address the middle regime they leave empty
  (HBM idle by construction).** A dominance argument that holds regardless of
  whose mechanism is faster — and we now know their architecture can *be* the
  mechanism.

## 5. What still needs Hopper

- Any measurement involving their kernel.
- The magnitude of the rider benefit (A100/PCIe gives 0.47 riders at 70B).
- Whether the FA2-vs-Triton crossover we found on A100 holds under FA3.
