# BLOCKER: the host tier never frees prefill blocks

Found 2026-08-12 while validating that the endpoint-comparison harness can
detect a difference. **This invalidates the current implementation's ability to
demonstrate the paper's central claim, and it must be fixed before the GH200
rental.** It also explains two mysteries at once.

## The defect

`scheduler.py` caps a CPU-tier request's block *growth*
(`_rungd_cap_run -> new_blocks = empty_kv_cache_blocks`). It never releases the
blocks the request already holds. Our own comment says so:

> "cap prevents block *growth* for CPU-tier requests, it does not reclaim"

So a host-tier request keeps its **entire prefill** in HBM forever. The only HBM
it frees is what it *would have grown* during decode.

## Why that guts the model

The gain law assumes a host-tier request costs `f = k_window/L` (~10-25%) of a
resident request's HBM. In reality it costs the full prefill plus the k-window.

Llama-70B fp8, 8K context, C = 19.3 GB:

| generation G | request KV | stays resident | freed |
|---|---|---|---|
| 256 | 2.77 GB | 2.68 GB | 0.08 GB (**3%**) |
| 1024 | 3.02 GB | 2.68 GB | 0.34 GB (**11%**) |
| 4096 | 4.03 GB | 2.68 GB | 1.34 GB (**33%**) |

At G=1024:

| configuration | B_cap |
|---|---|
| all-HBM | 6.4 |
| tiering **as implemented** (prefill stays) | 7.2 — **+12%** |
| tiering **as modelled** (`f`=0.1) | 63.9 |

**Worse than a small gain: a "rider" whose prefill stays resident frees nothing
at all, while still costing `S*k` of interconnect every step.** It is strictly
worse than leaving the request in HBM. The mixture cannot beat all-HBM under
this implementation, at any interconnect speed.

## It is confirmed by measurement, not just by reading code

A100, Qwen-32B, 24 requests offered at 512 ctx / 12 generated, capacity binding
(9.6 of 24 sustained, so the vacuity guard did not fire):

| arm | steady concurrency | tok/s |
|---|---|---|
| all-HBM | 9.6 | 61.4 |
| mixture | **9.6 (identical)** | **25.5 (2.4x slower)** |

Same concurrency, much slower — exactly what the analysis predicts when the
offload frees nothing. At 512 prefill / 12 generated there is essentially
nothing to reclaim.

## It also explains the `all_host` stall

With `hybrid_b_hbm_max=0`, all 20 requests are host-tier — but every one still
needs its **full prefill resident**, i.e. 20 x 1024 = 20,480 tokens. That does
not fit the resident pool, so the scheduler thrashes. One root cause explains
both open mysteries, which is a good sign the diagnosis is right.

## Why the earlier "2.67x serve-more" result did not catch this

That harness used **short prompts with long generation**, where decode-grown
blocks dominate (126 -> 48 blocks). That is the one workload shape where
capping growth reclaims most of a request's HBM. It is a favourable special
case, and the "CPU-tier requests consume ~0 main-pool blocks" framing was
overstated: they consume their whole prefill.

## What has to happen

**Release a host-tier request's prefill blocks once they are seeded to host.**
This is clearly possible — DirectKV keeps 100% of KV on the host — our
implementation simply never does it. Required:

1. after `seed_req` succeeds, return the request's resident blocks to the KV
   cache manager;
2. ensure attention for that request reads **only** from the rolling window
   (the pool override already does this — the standard pass over CPU rows was
   already masked out by opt #5);
3. make promotion re-allocate, which the two-phase commit already does;
4. re-run the full correctness ladder — this changes what memory a request owns,
   so token-identical is the gate.

**Until this lands, the GH200 endpoint comparison would measure ~no concurrency
gain and we would wrongly conclude direction A is dead.** That is the rental
this finding saves.

## Honest status of the paper's claims

- "2.67x concurrent at the same HBM" — true only for short-prompt/long-generation
  workloads. Must be restated with that condition.
- The 7/5/12 prediction assumes near-full offload. **Not yet delivered by our
  implementation.**
- The gain law itself is unaffected: it describes what a *correct* offload
  achieves. The implementation does not yet meet it.
