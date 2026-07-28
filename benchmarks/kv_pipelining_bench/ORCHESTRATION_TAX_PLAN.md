# Orchestration-Tax → DPU Offload: Thesis & Execution Plan

Status: active. Hardware: NVIDIA A100 80GB PCIe. Model: meta-llama/Llama-3.1-8B-Instruct.
Repo HEAD at validated run: f417a0e25.

## Thesis (locked 2026-06-23)

Layer-wise KV orchestration forces the host CPU into the **per-layer pipeline loop**:
every decode step, for each of L layers, the GPU signals progress, the host decides and
issues the next CPU→GPU/GPU→CPU transfer, and synchronizes. The resulting tax is:

1. **Bandwidth-invariant** — R3 (`VLLM_KVPIPE_FAKE_TRANSFER=1`, zero-byte copies,
   `gpu_stall_sum ≈ 0`) still costs +62% (c32) to +105% (c64) ITL at long context.
2. **Block/sparsity-invariant at its floor** — the per-layer control events fire L× per
   step regardless of how many blocks move; sparse attention (RetroInfer/LiteCache) cannot
   escape it and in fact adds per-step block *selection* control.
3. **Not removable by CUDA graphs** when the access pattern is data-dependent (sparse).

A DPU that **monitors GPU layer-progress on-fabric and drives transfers autonomously**
removes the host CPU from this loop → eliminates GPU↔CPU round-trips + host CUDA issue/sync.
This is **acceleration by structural decoupling**, not relocation of compute.

Claim to prove: residual tax ≈ CPU-in-loop GPU↔CPU cost, and it is invariant to bandwidth
and block count. (DPU absolute speedup needs latency model + emulation; no DPU silicon here.)

## Measured so far

- R3 long c32: ITL 27.98 ms vs R0 17.22 ms → tax 10.8 ms/step; `gpu_stall_sum ≈ 0`.
- Instrumented connector host hooks ≈ 0.28 ms/step (start_load 0.23 + wait 0.02 + save 0.02).
- => ~10.5 ms residual is neither GPU transfer-stall nor instrumented hook host time.

### Phase 1a results (2026-06-23, nsys unavailable → dmon + Torch/CUPTI trace)

Repro c32: R0 ITL 16.0 ms, R3(fake) ITL 28.0 ms (tax ~12 ms).

GPU SM-utilization during steady decode (nvidia-smi dmon 1 Hz, overhead-free):
- R0 : SM 97–100%, mem-util 49–59%
- R3 : SM ~92%,     mem-util ~73%
=> R3 is NOT GPU-starved; SM stays busy. 92%×28ms ≈ 25.8 ms GPU-active vs R0
   99%×16ms ≈ 15.8 ms → R3 has ~10 ms MORE GPU-active time + higher mem traffic
   despite zero-byte (fake) copies. Tax ≈ extra GPU work, not a transfer stall.

Torch/CUPTI op inventory (timing overhead-inflated; COUNTS valid). 30 decode steps
captured (`vllm::unified_attention_with_output` ×960 = 32 layers × 30 steps):
- `aten::item`/`_local_scalar_dense` (GPU→CPU sync) ×229,794 BUT time-quantile binned
  to Q0–Q1 only, ZERO in steady Q2–Q3 → ADMISSION artifact (register_physical_blocks /
  ensure_table_block_gpu per-block loops). Inflates TTFT, NOT steady ITL.
- Steady (Q2–Q3) per-step ops, uniform: `aten::copy_`, `Memcpy HtoD (Pageable→Device)`,
  `aten::select`/indexing in large counts → per-layer host-issued small GPU ops +
  small LUT HtoD transfers to assemble KV placement.

Source confirmation: `_fake_transfer_enabled()` guards ONLY the bulk `copy_`
(cpu_kv_table.py 351/427/443/492/519/598/717/752). The per-layer control path in
`stage_delta` (562 `bool(valid.any())`, 565 `bool(valid.all())`, 570 `.cpu().item()`
[n==1], 575 `physical_to_flat.to(device)`, 580 `(table_slots<0).any()`) runs in R3.
=> R3 tax = per-layer host-control + small GPU placement work, bandwidth-invariant.

OPEN QUESTION (linchpin for DPU claim): is the steady tax irreducible per-layer
host round-trip / CPU-in-loop, or optimizable kernel inefficiency (fuse/batch the
small index/copy kernels, cache LUT on GPU, drop redundant valid.any/all)? Resolve
via Phase 1b (eager-R0 vs graph-R0 per-op launch price) + a kernel-count of R3's
extra small ops. Honest framing: R3-as-measured is an UPPER bound on orchestration
tax for the current impl; the DPU argument rests on the irreducible per-layer
control round-trip that remains after software optimization (and that grows under
data-dependent sparse access, where the schedule cannot be statically graphed).

### Phase 1b results (2026-06-23): eager-vs-graph + kernel/memcpy decomposition

New bench mode `vanilla` (no kv-transfer-config -> CUDA graphs allowed) added to
launch_server. c32 long ITL:
- vanilla-graph (R0', TRUE production baseline) : 15.5 ms
- vanilla-eager                                : 16.0 ms   (graphs save 0.5 ms = 3%)
- baseline (DecodeBenchConnector, eager)       : 16.0 ms
- R3-fake (pipelined)                          : 28.0 ms   (tax +12.0 ms = +77%)

=> CUDA graphs save only 0.5 ms at c32: the base decode is NOT launch-bound at
   batch 32 (kernel work hides launch latency). Therefore R3's 12 ms tax is NOT an
   eager-launch penalty graphs could remove; it is structural orchestration WORK.

Steady-window (admission excluded) per decode step, CUPTI counts (counts reliable;
profiler timing overhead-inflated so treated as lower bounds):
            kernels/step   memcpy/step   memcpy-GPU-time/step
  R0 base       603            33               0.06 ms
  R3 fake       507          2,213             2.32 ms
=> NOT extra compute kernels (comparable). The signature is a ~67x storm of tiny
   pageable HtoD memcpies/step = per-layer KV-placement metadata (LUTs, block
   indices, slot mappings) shipped host->device EVERY layer. >=2.3 ms/step pure
   transfer + host work to generate & issue ~2,200 cudaMemcpyAsync/step.

PRELIMINARY (later OVERTURNED) Phase 1 conclusion: tax = per-layer host->device
metadata storm. THIS WAS WRONG -- it was admission contamination of short profile
windows. See CRITICAL CORRECTION below.

### CRITICAL CORRECTION (2026-06-24): the c32 long tax is ATTENTION, not orchestration

Method fix: earlier "steady" CUPTI windows were contaminated by request ADMISSION
(32 reqs admit over many steps; register_physical_blocks does a per-block tbl_gpu
HtoD loop ONCE per req via the bench connector's _stage_all_layers_to_cpu). The
".item() storm" (admission, register/ensure_table_block_gpu) and the "~2,200
memcpy/step storm" were BOTH admission, not steady decode.

Clean steady capture (VLLM_PROFILER_DELAY_ITERS=60, MAX_ITERS=20; step anchor =
reshape_and_cache_flash_kernel @ 32/layer/step => 20 steps both arms):

  device kernel-time/step       R0 baseline   R3 fake     delta
    gemm / projections            11.42 ms     11.38 ms    ~0     (identical model work)
    ATTENTION (flash splitkv)      4.56 ms     15.74 ms   +11.2 ms  <-- THE TAX
    gather/copy (orchestration)    0.43 ms      0.49 ms   +0.06 ms (negligible)
    memcpy                         0.03 ms      0.03 ms    0
    TOTAL device/step             17.14 ms     28.35 ms   +11.2 ms

Device time == real ITL on both arms (R0 17~16ms, R3 28~28ms) => both are fully
GPU-bound, numbers trustworthy. flash_fwd_splitkv: R0 x64/step (split-KV = 2
parallel chunks/layer, 4.3ms) vs R3 x32/step (NO split, 15.6ms, 3.4x slower).

=> At c32 long context the R3 "orchestration tax" is a FlashAttention split-KV
   regression caused by the pipelining KV LAYOUT (rolling K-window /
   --inflate-kv-pool changes block count/shape -> FA picks num_splits=1). The
   genuine per-layer orchestration (gather/copy/memcpy) is ~0.06 ms/step.
   Steady host CPU-in-loop is NOT significant at c32.

IMPLICATIONS for the DPU thesis:
- The headline long-context R3 numbers in DECODE_ENGINE_MATRIX do NOT measure
  orchestration; they measure an attention-kernel-selection artifact. Must not
  build the DPU motivation on them.
- TODO verify cause: re-run R3 with --no-inflate-kv-pool and inspect the FA
  num_splits heuristic vs KV tensor shape/num_blocks in the pipelined layout.
- Genuine orchestration cost must be sought where it actually dominates:
  (i) admission/TTFT (the register_physical_blocks HtoD loop -- real, but bench
      connector impl, optimizable), (ii) small-batch decode (launch-bound regime,
      where graphs DO help and R3 cannot graph), (iii) real R1 DDR<->HBM bandwidth.
- Open: is split-KV loss fundamental to ANY layer-wise CPU-offload KV layout, or
  fixable? If fixable, the long-context tax largely disappears and the thesis must
  rest on small-batch launch overhead + sparse per-step control, not dense c32.

VERIFICATION (2026-06-24): --inflate-kv-pool RULED OUT as cause.
  R0 baseline : device 17.14 ms/step | attention 4.46 ms | flash x96/step
  R3 inflate  : device 28.35 ms/step | attention 15.74 ms | flash x64/step
  R3 NO-inflate: device 28.33 ms/step | attention 15.73 ms | flash x64/step
  => inflate vs no-inflate IDENTICAL. The attention regression is inherent to the
     pipelining rolling-K-window KV layout/binding: R0 gets more split-KV
     parallelism (x96 flash kernels, more SMs) vs R3 (x64, fewer splits) for the
     SAME attention FLOPs -> 3.5x slower attention. Next: inspect how the rolling
     window presents KV to FlashAttention (block_table / seqlens / num_blocks /
     contiguity) vs paged baseline, to see if the layout can be made split-friendly.

### FINAL ROOT CAUSE (2026-07-02): the gap is PREFIX CACHING, not pipelining/FA layout
Both prior conclusions (metadata storm; then split-KV-from-layout) were ARTIFACTS of
the synthetic benchmark. Decisive instrumentation + apples-to-apples controls:

1. Instrumented `flash_attn_varlen_func` inputs (VLLM_FA_SPLIT_DEBUG) for R0 vs R3 at
   the SAME decode step. Every FA argument is BYTE-IDENTICAL:
     q shape, heads_q=32, heads_kv=8, hdim=128, block_size=16,
     kcache=(28713,16,8,128), kstride=(16384,1024,128,1), block_table=(B,576),
     max_seqlen_k, seqused_k, num_splits(=0, FA-auto), scheduler_metadata(=None, FA2).
   With --no-inflate-kv-pool R3's kcache num_blocks (28713) == R0's exactly.
   => There is NO differing FA metadata field. The premise "the layout flips FA to
      num_splits=1" is FALSE.

2. Torch-profiler kernel table, identical load (32 reqs, SAME prompt, c32, ~4.6k ctx,
   25 steady decode steps), R0 baseline vs R3 fake:
     R0: flash_fwd_splitkv x1600 (64/step = 2 kernels/layer) + `merge_attn_states`
         x800 ; attention 7.0 ms/step ; gemm 11.4 ms/step ; total 23.9 ms/step
     R3: flash_fwd_splitkv x800  (32/step = 1 kernel/layer) ; NO merge_attn_states
         attention 17.7 ms/step ; gemm 11.4 ms/step (identical) ; total 30.3 ms/step
   The 2-kernels-per-layer + `merge_attn_states` in R0 is the CASCADE-ATTENTION path,
   not ordinary split-KV. Cascade fires because the 32 requests share an IDENTICAL
   prompt and R0 has prefix caching ON (default); R3's pipelined path forces
   `--no-enable-prefix-caching`, so it cannot dedupe the shared prefix.

3. Apples-to-apples ITL (identical harness: 32 shared prompts, c32, ~4.6k ctx, eager):
     R0 baseline, prefix caching ON  (cascade) : 17.5 ms   <- the "fast" number
     R0 baseline, prefix caching OFF           : 27.7 ms
     R3 pipelined (fake transfer, prefix off)  : 28.2 ms
   => R3 (28.2) ~= R0-prefix-OFF (27.7), within ~2% noise. The pipelining
      ORCHESTRATION TAX IS ~0. The entire ~10-12 ms "tax" was R0's cascade-attention
      speedup from prefix caching on a degenerate all-identical-prompt benchmark.

IMPLICATIONS:
- There is no FlashAttention kernel bug and no layer-pipelining orchestration penalty
  to fix. When CPU<->GPU transfer is fully hidden (fake/infinite BW), R3 ~= R0 at
  matched prefix-caching. This is exactly the "no added latency" result the project
  set out to demonstrate.
- FAIR baseline for all future R0-vs-R3 claims: compare at the SAME prefix-caching
  setting, or use DISTINCT prompts (no shared prefix, so cascade never fires and R0
  cannot pull ahead). The DECODE_ENGINE_MATRIX long-context R0 numbers that used
  shared prompts + prefix caching are NOT a fair baseline for R3.
- Real trade-off worth stating honestly: prefix caching is a genuine serving
  optimization (shared system prompts, agentic multi-turn). Layer-wise CPU-offload
  pipelining currently disables it (`--no-enable-prefix-caching`); reconciling
  pipelining with prefix caching is a separate, real design question -- distinct from
  any per-layer orchestration cost, which is ~0.

### R3 PREFIX CACHING ENABLED + MATRIX RERUN (2026-07-03)

Change: `pipelined_decode_fast_bench.py` now defaults `--prefix-caching-on` for BOTH
baseline and pipelined (removed hardcoded `--no-enable-prefix-caching` on R3).

Validation (R3 fake, c32, S=8192, prefix ON):
  - 32/32 OK, steady ITL 16.4 ms (was ~28 ms with prefix OFF)
  - Server log: Prefix cache hit rate 96.8%

Full matrix rerun (prefix ON both arms, no legacy setup-seed/untimed-prefix flags):
  Root: `/mnt/beegfs/benishay/paper_runs/20260703/orchestration_tax/r0_r3_prefix_on_v1`

  Long S=8192 — R3-R0 ITL tax:
    c1 +5.0% | c4 +3.9% | c8 +3.3% | c16 +1.9% | c32 +2.2% | c64 +2.3%
    (c32: R0 16.13 ms vs R3 16.49 ms — vs old matrix 17.22 vs 27.98 ms)

  Short S=512 — R3-R0 ITL tax:
    c16 +1.2% | c64 +1.6% | c256 +3.9% | c512 +2.2% (512/512 OK after max_model_len fix)
    c512 failure (2026-07-03): max_model_len=9216 → only 383 CpuKvTable slots;
    fix: short grid uses max_model_len=1024 (3452 slots), matching original matrix.

  => With prefix caching ON on both arms, orchestration tax (R3 fake) is ~2-5% ITL
     across all successful cells. Old +62% c32 gap was prefix-caching asymmetry.

## PRODUCTION BASELINE CPU UTILIZATION (R0′ vanilla, CUDA graphs ON) — 2026-06-25
Goal: establish a trustworthy "where we start" — how many CPU cores the production
baseline burns to keep the A100 fed, sweeping concurrency until tok/s saturates.
Config: vanilla vLLM (no KV connector, graphs FULL_AND_PIECEWISE), Llama-3.1-8B,
max_model_len 9216, input 8192 / output 200 (ignore_eos), gpu_mem_util 0.90.
Sampler: engine process-tree CPU (psutil cpu_times) + busiest single thread +
GPU SM% over each concurrency's STEADY-decode window.
ENV CAVEAT: agent sandbox pins launched procs to 8 cores (Cpus_allowed 0-7 on a
64-core box; sched_setaffinity is clamped). NOT liftable here. BUT peak usage was
<2 cores, so the 8-core cap never bound -> the CPU numbers ARE trustworthy.

  C    ITL ms   tok/s    TTFT s   tree_cores  busiest_thr  GPU SM%   KV MiB
  1    10.9      91.7     0.19     1.13        0.99         100.0     73853
  4    11.4     349.9     0.22     1.11        1.00         100.0     73853
  8    12.2     653.3     0.24     1.14        1.00         100.0     73853
  16   14.9    1076.8     0.31     1.63        1.00          97.0     73853
  32   16.4    1951.9     0.35     1.69        0.99          97.4     73853
  48   18.1    2654.1     0.40     1.73        0.99          97.6     73853
  56   18.4    3038.3     0.46     1.75        0.99          97.8     73853
  64+         (queues: TTFT 0.96->1.62s, batch capped ~54 by KV memory)

FINDINGS:
1. A100 is the binding resource: GPU SM 97-100% across the whole sweep. Throughput
   scales 92 -> ~3040 tok/s up to the KV-memory ceiling (~54 concurrent 9216-ctx
   seqs; 73.9 GB KV pinned). Past that, requests QUEUE (TTFT blows up) — tok/s is
   KV-capacity bound, NOT CPU bound. It plateaus, it does not "degrade".
2. Host CPU cost is tiny: whole engine tree uses ~1.1 cores at low load, ~1.75 cores
   at peak (out of 8 available). 4-7x headroom; the sandbox cap never bound.
3. Orchestration is SINGLE-THREADED: the busiest thread is the EngineCore main loop
   (scheduler + block/KV mgmt + CUDA launch), pegged at ~1.0 core (100% of ONE core)
   at EVERY concurrency, including C=1. Everything else (~0.1-0.75 cores) is API
   server / tokenizer / aux threads.
=> R0' baseline costs <2 cores to saturate an A100 at long context; scheduling +
   HBM/KV management is ~1 core of single-threaded work. This is the trustworthy
   starting point. DPU relevance: the host control plane is fundamentally ONE core;
   it has slack at 8192-ctx (GPU-bound), so the CPU-bottleneck thesis must target
   regimes where that single core becomes the bound (small-batch/launch-bound decode,
   admission/TTFT, or many tiny per-layer control ops) — NOT dense long-context decode.

## EXPERIMENT A — AGENTIC STRUCTURED DECODING (FSM) CPU COST — 2026-06-30
Goal: does guided/structured decoding (JSON schema, the core of agentic tool-calling)
make the HOST the bottleneck while trying to push GPU throughput?
Config: vanilla vLLM, CUDA graphs ON, Qwen2.5-0.5B (fast model → host shows through),
xgrammar backend, input 256 / max_tokens 1024 (schemas force large arrays so decode
window is measurable). Server PINNED to cores 2-7 (6c), client PINNED to cores 0-1 (2c)
=> server CPU measured WITHOUT client contention (disjoint cpusets within the 8-core cap).
Arms: unguided | guided-single (one fixed schema, cache-friendly) | guided-diverse
(unique nested schema per request, cache-hostile). cores = server process-tree;
busy = hottest thread (EngineCore step loop); worker = tree-minus-busy (xgrammar pool + detok).

  C   arm             ITL ms   tok/s    tree  busy  worker  GPU%
  1   unguided         1.85     540     1.67  0.98   0.70   99.0
  1   guided-single    1.85     542     1.52  0.98   0.48   99.0
  1   guided-diverse   1.85     540     1.46  0.94   0.49   98.7
  8   unguided         2.19    3661     1.66  0.97   0.67   98.7
  8   guided-single    2.36    3385     1.31  0.98   0.32   91.3
  8   guided-diverse   2.78    2875     1.24  1.00   0.28   86.5
  32  unguided         2.98   10738     1.49  0.97   0.51   91.2
  32  guided-single    4.02    7967     1.26  0.98   0.27   67.7
  32  guided-diverse   4.27    7493     1.24  0.98   0.25   60.8
  64  unguided         5.93   10793     4.65  0.95   3.71   55.0
  64  guided-single    6.58    9725     3.33  0.97   2.36   51.5
  64  guided-diverse   6.72    9525     1.86  0.98   0.88   51.2

FINDINGS:
1. Structured decoding STALLS THE GPU: at C=32 guided cuts throughput ~26-30%
   (10.7k -> 7.5-8.0k tok/s) and GPU util 91% -> 61-68%, worse with schema complexity.
2. The cost lands on the SINGLE EngineCore step thread (busy ~0.98 core in ALL arms):
   get_grammar_bitmask is computed/coordinated on the scheduler step path
   (scheduler.py:1082). Even though the bitmask FILL uses a ThreadPoolExecutor
   (structured_output/__init__.py:63-73), the main loop must gather + submit + WAIT,
   so per-step host time grows -> fewer steps/s -> GPU starves. The pool does NOT
   relieve it (worker cores are LOWER for guided, only because lower tok/s -> less detok).
3. Server CPU/token rises ~15-20% (138 -> 158-165 us/tok) -> modest absolute CPU, but
   because it hits the saturated serial thread it costs ~30% throughput.
4. Grammar COMPILE cost (diverse arm) is front-loaded at admission, outside the steady
   window, so steady-state diverse ~= single in cores (diverse only slightly worse tok/s).
=> Agentic structured output IS a real host-side lever: it makes the single-threaded
   engine loop the throughput limiter and idles the GPU (down to ~61% at C=32).

## EXPERIMENT B1 — SCHEDULING SINGLE-THREAD CEILING — 2026-06-30
Goal: does the single-threaded EngineCore step loop cap throughput before the GPU
saturates (small fast model, many seqs)? Clean non-streaming CLOSED-LOOP client
(minimal client CPU) on cores 0-1; server on cores 2-7. Qwen2.5-0.5B, input 256,
max_tokens 128, 8s sustained per point. sched_probe.py.

   C    tok/s   tree  busy  worker  GPU%
  16    6495    1.72  0.97   0.75   98.9
  32   11597    1.75  0.96   0.78   98.2
  64   19436    1.74  0.96   0.79   96.2
 128   27426    1.72  0.95   0.78   89.1
 256   35176    1.79  0.94   0.85   86.2
 512   34990    3.65  0.94   2.72   73.1

FINDINGS:
1. Throughput SATURATES at ~35k tok/s (C=256->512 flat) while GPU still has headroom
   (86% -> 73%). More batch yields NO more throughput -> not GPU-bound.
2. The EngineCore step thread is pinned at ~0.94-0.97 core (one core, 100%) at EVERY
   concurrency. The serial scheduler/step loop is the throughput ceiling: bigger batch
   => more per-step host work on that one thread => lower step rate, offsetting the
   larger batch (GPU util drops instead of throughput rising).
3. worker cores stay ~0.8 until C=512 where detok/sampling for 512 seqs adds ~2.7 cores.
=> On a fast model the single-threaded control plane caps throughput with ~27% GPU left
   idle. This is the scheduling-bound regime (complements Exp A: structured decoding
   pushes the same single thread into the bottleneck at far lower batch).
NOTE: 8-core sandbox forces client+server to share cores; clean separation needs
disjoint cpusets (server 2-7 / client 0-1) + non-streaming, else client contention
distorts high-C throughput (streaming variant regressed spuriously at C>=128).

## EXPERIMENT B2 — PREFIX-CACHE / KV-BLOCK MANAGEMENT CPU — 2026-06-30
Goal: is block hashing + cache-map + alloc/free/evict a heavy CPU consumer?
Closed-loop non-streaming, 1024-token prefix, 32-token output, prefix caching ON.
shared = same prefix for all (cache hits); unique = unique prefix per request
(full hashing + store + evict). b2_prefix_probe.py. server 2-7 / client 0-1.

  arm     C    tok/s   req/s   tree  busy  worker  GPU%
  shared  16   4954    154.8   1.85  0.95   0.90   83.3
  unique  16   3630    113.4   1.87  0.96   0.91   94.7
  shared  32   8089    252.8   2.01  0.94   1.07   78.1
  unique  32   5021    156.9   2.05  0.96   1.09   96.8

FINDINGS (partly NEGATIVE — important):
1. KV-block management CPU is CHEAP: tree/busy/worker are ~IDENTICAL for unique vs
   shared (2.01 vs 2.05 cores; busy 0.94 vs 0.96). Hashing ~64 blocks/req at ~157 req/s
   adds no measurable cores. Vanilla prefix-cache management is NOT a CPU lever.
2. The unique-vs-shared difference is GPU, not CPU: cache MISS (unique) re-prefills 1024
   tokens on GPU -> GPU 97%, fewer req/s; cache HIT (shared) skips prefill -> GPU drops
   to 78% and the request becomes SCHEDULER-bound (busy pegged 0.94, GPU idle) = same
   single-thread ceiling as B1.
=> In vanilla vLLM, prefix-cache value is GPU-prefill savings; its mgmt CPU is small.
   The CPU-heavy KV-management story lives in OFFLOAD/retention machinery (CPU<->HBM
   staging, per-layer orchestration = this project's R1/R3), NOT in paged-block hashing.

## Plan (nsys/ncu unavailable on host → nsys-free methods)

### Phase 1 — Localize the residual + GPU↔CPU cost (LINCHPIN)
- 1a. **GPU SM-utilization + engine CPU%** during steady decode, R0 vs R3. Low SM% + high
  CPU in R3 = host/launch-bound (CPU pacing GPU). (`nvidia-smi dmon`, no code change.)
- 1b. **eager-R0 vs graph-R0 calibration**: per-op CPU launch cost = (ITL_eager−ITL_graph)/
  (#kernels/step) on a graph-capable baseline.
- 1c. **Component ablation** via env knobs (BATCHED_D2H, STEP_PLAN, SKIP_DELTA_STAGE,
  DELTA_SCRATCH, WRAP_PREFETCH) to attribute tax to save vs prefetch vs step-plan build.
- Decision gate: host-control/round-trip dominant → DPU justified; GPU-stream-bubble only →
  static/graph fix candidate.

### Phase 2 — CUDA-graph baseline + static feasibility
- R0′ = vanilla vLLM + CUDA graphs (true production reference); quantify graph speedup.
- Scope whether R3 copy/event schedule is capturable with index-tensor indirection.

### Phase 3 — Sparsity scaling model
- Sweep: fixed B, L; vary blocks-moved/step (emulate top-k sparse budgets).
- Fit tax = α·n_blocks + C_fixed(L,B); report the **invariant floor** C_fixed (the
  per-layer GPU↔CPU cost sparse cannot escape). Predict RetroInfer/LiteCache tax.

### Phase 5 — DPU-benefit evidence
- Latency model: host round-trip (driver+schedule+doorbell) vs on-fabric poll+DMA.
- Software emulation of autonomous orchestrator: persistent kernel / CUDA device graph /
  decoupled poller thread; show removing CPU-in-loop collapses the residual.

### Synthesis
- dense tax (R3) → root-cause split → graph-fix-vs-DPU decision → sparse generalization →
  DPU as control-plane (selection + scheduling) offload.

---

## Experiment C — Structured decoding: does grammar CPU hide under the forward? (2026-07-02)
Config: vanilla vLLM v1, **Llama-3.1-8B-Instruct** (production size, 128k vocab), CUDA
graphs FULL_AND_PIECEWISE, xgrammar backend (default), `--max-num-seqs 256` (so the
fill-bitmask thread pool exists and batches can cross the 128 parallel threshold).
Server pinned `taskset -c 2-7` (6 cores), closed-loop non-streaming client on cores 0-1.
Driver: `fsm_scaling_sweep.py`. Arms grade per-step grammar cost; concurrency crosses 128.

### C1 — Server sweep (tok/s, cores, GPU SM%)
```
arm         C     tok/s   tree   busy  worker   GPU%
unguided   32    2354    1.14   0.98    0.15  100.0
simple     32    2339    1.10   0.87    0.22  100.0
medium     32     895    1.03   0.79    0.24   37.9   <-- GPU STARVED
complex    32    2330    1.10   0.81    0.30  100.0
diverse    32    1750    2.79   0.44    0.26   65.4   <-- compile cost, tree climbs
unguided  128    6252    1.18   0.97    0.21  100.0
simple    128    6118    1.13   0.77    0.35  100.0
medium    128    1116    1.01   0.94    0.07   17.1   <-- GPU STARVED
complex   128    6144    1.14   0.62    0.51  100.0
diverse   128    3517    3.24   0.35    0.31   58.8   <-- compile cost
unguided  256    7815    1.16   0.95    0.21  100.0
simple    256    7116    1.24   0.84    0.41   90.5
medium    256    2152    1.99   0.28    1.71   23.9   <-- fill pool burns 1.71 cores, still starved
complex   256    7133    1.42   0.81    0.60   90.0
diverse   256    6852    1.64   0.80    0.85   90.2   <-- pool cached -> recovers
```

### C2 — Single-request control (C=1): grammar is FULLY HIDDEN
All arms identical: 400 tok, ~11.3 ms/token, finish=length. So schemas generate equal
length and there is **zero** measurable grammar cost at low concurrency on the 8B — the
GPU forward absorbs it entirely. The effect below is purely concurrency-driven.

### C3 — xgrammar micro-benchmark (per-step fill/accept, µs)
```
grammar     fill mean   fill med   fill p95    fill max     accept
simple          9.0        8.9        9.6        227          0.5
complex        93.2       95.2       98.5        130          0.6
diverse        91.2       95.0       98.8        123          0.6
medium        733.9        5.6     6484.4      13519          0.7   <-- catastrophic tail
```
Isolating the trigger (same schema, only maxLength differs):
```
bounded (maxLength)   fill mean 3972 µs   p95 13213 µs   max 13595 µs
free    (no maxLength) fill mean   87 µs   p95    96 µs   max   122 µs
```

### Verdict — direction CONFIRMED, with a sharp refinement
1. **On a production 8B, structured decoding hides completely at low concurrency**
   (C=1 identical to unguided). Confirms the "big forward absorbs grammar CPU" intuition.
2. **Under concurrency it can severely starve the GPU (SM 17-38%)** — but *whether* it
   does depends on **grammar STRUCTURE, not apparent complexity**:
   - Cheap / hidden: unguided, enums (simple), free-form strings (complex/diverse) →
     fill 9-95 µs → GPU 90-100%.
   - Expensive / exposed: **bounded-length (`maxLength`) strings** → fill spikes to
     6-13 ms → GPU collapses to 17-38%. maxLength defeats xgrammar's cached-mask fast
     path (state depends on remaining length budget → per-token length checks).
3. **FSM advance (`accept_token`) is a non-cost (~0.5 µs).** The cost is per-step FILL,
   plus grammar COMPILE.
4. **Two distinct host costs, both bad for the single-threaded control plane:**
   - **Fill**: serial on the EngineCore main thread for batch ≤128 (adding cores does
     NOT help); on the pool for >128 but barrier-bound (slowest fill gates the step).
     medium at C=256 burns 1.71 worker cores yet still starves GPU to 24%.
   - **Compile**: 300-440 ms per *unique* schema. Diverse/agentic workloads (many
     distinct tool schemas) pay this repeatedly → tree cores climb to 2.8-3.2 and GPU
     stalls (65%) until the schema set is cached.
5. **Realism:** maxLength fields (IDs, codes, names, capped enums) and diverse tool
   schemas are ubiquitous in real agentic/tool-calling traffic → this is not contrived.
   → Structured decoding IS a genuine, concurrency-triggered host-side bottleneck on
   production-size models, and it is exactly the kind of serial/barrier-bound symbolic
   work that neither the CPU (serial, GIL) nor the GPU (divergent) handles well.

### Caveats
- Micro-walk uses deterministic first-allowed-token selection (worst-case-ish path) →
  absolute fill means inflated vs real sampling; but the server run (real sampling)
  corroborates the medium GPU collapse, so the effect is real, not a walk artifact.
- 6 server cores (sandbox). GPU starvation is a per-step latency phenomenon (serial fill
  ≤128), so it is not rescued by more cores below the 128 pool threshold.

### C4 — External validation + retractions (2026-07-02, literature check)
Checked the finding against the literature/community before investing further. Env:
xgrammar **0.1.29** (reproduced the 6-13 ms fill locally).

CONFIRMED (the 10 ms fill is real, not a walk artifact):
- LLGuidance benchmark (Microsoft Research) on xgrammar: masks "under 8µs in half of masks
  we tested, however if it doesn't fit the particular input, the mask computation times can
  run to **tens or hundreds of milliseconds**."
- Real vLLM user ("Grammar CPU bound performance" forum): no-grammar ~8s @ ~100% GPU vs
  grammar ~24s (3x) with **GPU mostly ~15%, spiking to 40%** — matches our 17-38% collapse.
- SqueezeBits benchmark: guided decoding degrades vLLM throughput, "gap becomes more
  apparent with larger batch sizes" — matches our concurrency scaling.
- xgrammar issue #465 / PR #467 "fix the efficiency issue in the repetition structure" —
  bounded repetition (maxLength/maxItems class) is a KNOWN pathology.

CAVEATS THAT WEAKEN THE THESIS:
- Tail, not common case: XGrammar paper reports context-dependent tokens are <1%
  (1134/128k) for Llama-3.1 JSON, typical fill <40µs. The 10 ms hit needs specific shapes
  (bounded repetition, heavy whitespace, many context-dependent tokens).
- Moving target: XGrammar-2 (2026-05) compresses repetition O(n)->O(1) (534ms->5.37ms case)
  and targets tool-calling; LLGuidance ~50µs on-the-fly; SGLang overlaps mask-gen better
  than vLLM. The specific maxLength pathology may already be fixed upstream (we are on the
  older 0.1.29). Building a systems thesis on a patchable library bug is risky.

RETRACTIONS:
- **Compile cost dropped from the data-path thesis.** Tools are registered once and reused;
  compiled grammars are cached. The `diverse` (unique-schema-per-request) arm is artificial;
  compile is a one-time, offline, pre-warmable per-tool cost. Only re-enters the hot path
  for genuinely novel per-request schemas (rare) or cache-eviction thrashing (niche).
- **FSM advance (accept) is a non-cost (~0.5µs).** Not a lever.

### FINAL VERDICT for structured decoding — direction CLOSED
- Narrow claim (maxLength → 10 ms fill): TRUE but FRAGILE (known, patchable library bug).
  Do not anchor on it.
- Durable claim (community-backed): structured-decoding mask-gen is CPU work that does not
  always hide behind the GPU, and the hiding budget shrinks as batch grows and per-token GPU
  time falls (faster GPUs, smaller models, spec decode). Even 40-95µs masks × large batch ÷
  limited serial cores eventually compete with the forward. But this is an **actively
  contested SOTA axis** (xgrammar-2, llguidance, SGLang overlap) — real motivation, but NOT
  unclaimed territory.
- Decision: **do not build the research contribution on structured decoding.** Confirmed
  real but it is a tail phenomenon on a patchable, actively-optimized library path. Parked.
  Artifacts kept: `fsm_scaling_sweep.py`, `fsm_scaling_8b.json`.
