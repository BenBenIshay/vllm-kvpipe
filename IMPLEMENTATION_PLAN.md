# Implementation plan — request-level KV tiering (approved 2026-08-12)

Goal: have the **policy layer** built, debugged and validated on the A100 we
own, so the GH200 rental spends its time **measuring**, not debugging.

**Approved decision:** build in **vLLM v1 with a swappable mechanism**, not by
porting our policy into SGLang. Rationale: DirectKV's kernel is **SM90a-only**
(TMA/FA-3/CuTe) and this cluster has **no `nvcc`**, so any DirectKV-integration
code is unrunnable and uncompilable here. Everything below is testable today.
See `DIRECTKV_INTEGRATION_ANALYSIS.md` for why the swap is clean when we get
Hopper (their pool is per-request row-allocated; their `forward_decode` already
dispatches disjoint subsets to different attention implementations).

**Scope:** items 1–4. Item 5 (GH200 endpoint comparison) is the rental.

---

## Current assets (what we build on)

| Component | File | State |
|---|---|---|
| Host block-id KV pool | `vllm/v1/hybrid_rungd_pool.py` (245 L) | bitwise-validated |
| CPU-decode attention override | `vllm/v1/attention/backends/flash_attn.py::_pool_cpu_override` | token-identical |
| B1 roofline + B2 ITL guardrail | `vllm/v1/core/sched/transfer_aware_admission.py` | unit + in-server PASS |
| Scheduler block-growth cap | `vllm/v1/core/sched/scheduler.py` (~line 373) | reclaims host-tier HBM |
| Runner wiring | `gpu_model_runner._setup_rungd_pool_path` / `_maybe_setup_rungd_pool` | env-gated |
| Correctness ladder | `hybrid_rungd_real.py`, `rungd_host_pool_test.py`, `=2` self-verify | token-identical / bitwise |

Everything stays **env-gated, default OFF**.

---

## Item 1 — Fix the host-tier transfer path

**Why first:** it is needed under every option, it is the largest measured
inefficiency in our own code, and leaving it in place would contaminate the
GH200 measurement with our defect rather than the design's merit.

**Measured problem** (`DIRECTION_A_GO_NOGO.md` Gate 2, corrected):

| current behaviour | measured |
|---|---|
| pinned H2D (async path, opt-in) | 25.08 GB/s |
| pageable H2D (**default** sync path) | 11.17 GB/s (2.25× slower) |
| **CPU `index_select` gather (both paths, critical path)** | **2.9 GB/s** ← dominant |

**1a. Kill the pageable copy.** `_sync_stream_into` does
`slot.copy_(gathered.to(device))` where `gathered = host[layer].index_select(...)`
is **not pinned**. Gather into the existing per-layer pinned stage buffer
(`self._stage[layer_idx]`) and copy from that, on both the sync and async paths.
*Expected:* 11.17 → ~25 GB/s on the default path.

**1b. Kill the gather.** The gather exists because our host mirror is indexed by
**host block id**, so a cohort's blocks are scattered. DirectKV's pool is
*"contiguous per-request, no scatter/gather"*. Adopt the same: give each
host-tier request a **contiguous host row** (`[max_seqlen_pad, NH_kv, D]` per
layer), assigned from a free-list. Then a cohort's per-layer transfer becomes a
small number of contiguous slices instead of an `index_select` over scattered
blocks.
*Expected:* removes a 2.9 GB/s stage from the critical path entirely.

**Acceptance:**
- `rungd_host_pool_test.py` still bitwise-clean.
- `hybrid_rungd_real.py` still **token-identical** to vanilla.
- Measured per-layer host→device path ≥ 20 GB/s on A100 (from 2.9).
- rung-d serve-more wall time improves measurably vs the 12.8 s baseline.

**Risk:** medium. Changing the host layout touches seed/grow/write/read. The
bitwise + token-identical harness is the gate; no performance number is taken
until both pass.

## Item 2 — Request-level tier assignment

**Gap:** today "which requests are host-tier" is effectively arrival order; the
controller decides only *how many*. This is contribution 2 of the paper and the
thing no prior system does (OmniServe uses an externally-supplied LS/BE class
label; DAK uses a per-*operation* ratio from a given global ratio; DirectKV has
no policy at all).

**Build:** a scorer over per-request observable state, evaluated at admission:
- current context length (bigger ⇒ more HBM freed per demotion, but more
  interconnect per step)
- estimated remaining decode length
- time since last turn (multi-turn / agentic idle gaps)
- marginal HBM cost vs the rider budget the controller has left

Start with a **transparent threshold/linear rule** so the paper can explain it;
only try a learned variant if the simple rule leaves measurable headroom.

**Where:** extend `TransferAwareAdmissionController` with
`choose_tier(request, state) -> Literal["hbm", "host"]`, called from
`scheduler.py` where `should_admit_from_waiting` is consulted today. Keep B1/B2
untouched — they answer *how many*, this answers *which*.

**Acceptance:**
- Deterministic unit test: given a synthetic mix of short/long/idle requests and
  a fixed budget, the scorer produces the intended assignment.
- In-server: token-identical output regardless of tier assignment.
- Ablation hook: `VLLM_KVPIPE_TIER_POLICY={arrival,scored}` so the paper can
  measure policy-vs-arrival directly.

**Risk:** low-medium. The real risk is that the simple rule captures nearly all
the gain — a fine result, but it weakens "policy is load-bearing". Mitigated by
the item-5 ablation, not by rule sophistication.

## Item 3 — Tier migration (promote / demote)

**Gap:** documented limitation today — *gate-new-only*. A burst admits everyone
before the loop engages, and a long-running cohort can drift over SLO with no
recourse but to finish.

**Build:**
- **demote** (HBM → host) under ITL pressure; **promote** (host → HBM) when
  headroom appears
- **hysteresis** on both edges (reuse B2's OK/THROTTLED watermark pattern)
- **migration cost model**: bytes moved vs steps of benefit; refuse migrations
  that cannot amortise before the request's estimated completion
- correctness-preserving handoff reusing the pool's seed / compact-read
  primitives

**Testable claim (a citable disagreement):** the IEEE CAL letter's reactive/LRU
migration is **token**-level and their own results show migration overhead
dominates below ~60% attention sparsity. Our claim is that **request**-level
migration avoids that thrash, because whole requests move once at controlled
moments rather than pages chasing shifting token importance.

**Acceptance:**
- Token-identical output **across migration events** (the hard one).
- No thrash: migrations/second bounded under an adversarial oscillating load.
- Burst scenario: ITL recovers to SLO after a burst, which gate-new-only cannot do.

**Risk:** high — correctness during migration is where the bugs live. Gate every
performance number behind the bitwise/token-identical harness.

## Item 4 — Narrow mechanism interface (DirectKV swap-in)

**Purpose:** make the mechanism replaceable so the GH200 trip is a swap, not a
rewrite.

**Interface** (the minimum the policy layer needs):
```
class KVTierBackend(Protocol):
    def alloc(req_id, num_tokens) -> None          # place a request in this tier
    def free(req_id) -> None
    def write_token(req_id, layer, k, v) -> None   # decode-step append
    def prepare_layer(layer_idx) -> Handle         # make layer readable by attn
    def attend(rows, q, handle) -> Tensor          # tier-specific attention
    def bytes_per_step(req_ids) -> int             # feeds B1's roofline
```

Two implementations:
- `StagedHostBackend` — our rolling k-window + pinned bulk copies (item 1). Runs
  on A100 **today**.
- `DirectKVBackend` — their SM90a kernel + per-request contiguous CPU pool.
  Written against the same interface, **compiled and run only on Hopper.**

The analysis says this is clean: their kernel takes an arbitrary `req_ids`
subset, their pool is per-request row-allocated, and their `forward_decode`
already partitions a batch across two attention implementations. `attend(rows,…)`
maps directly onto that.

**Acceptance:** `StagedHostBackend` passes the full correctness ladder through
the interface with no behaviour change vs today; `DirectKVBackend` is written
and reviewed but **explicitly untested until GH200** (recorded as a known risk).

**Risk:** low for the interface; the DirectKV implementation carries unrunnable-
code risk, which is precisely why it is isolated behind one boundary.

---

## Sequencing and gates

```
Item 1 (transfer path)  ──► GATE: bitwise + token-identical + >=20 GB/s
   └─► Item 4 interface ──► GATE: ladder passes through the interface
        └─► Item 2 (tier assignment) ──► GATE: unit test + token-identical
             └─► Item 3 (migration)  ──► GATE: token-identical ACROSS migration
                                              + bounded thrash + burst recovery
   [then rent GH200] ──► Item 5: endpoint comparison, all-HBM vs all-host vs mixture
```

Items 1 and 4 are low-risk and unlock the rest. Item 3 is the long pole and the
only one that could slip.

## What this plan does NOT do

- It does **not** measure the mixture's magnitude — impossible on A100
  (0.47 riders at 70B/PCIe). That is the rental's job.
- It does **not** claim our mechanism beats DirectKV's. We build the interface so
  either can be the mechanism, and stay out of a kernel-engineering fight.
- It does **not** validate `DirectKVBackend`. Cannot compile SM90a here.

## Standing risks

1. The 8→13 batch increase remains **modelled, not measured** (`DIRECTION_A_GO_NOGO.md`
   Gate 4). Everything downstream inherits it.
2. eff_kv was measured under FA2; Hopper's FA3 may differ, and eff_kv sits in the
   capacity denominator — a better kernel *reduces* our projected gain.
3. FlashInfer remains unevaluated (needs `nvcc`).
