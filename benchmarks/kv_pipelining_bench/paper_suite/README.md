# Layer-wise KV Pipelining — Paper Test Suite

Turnkey suite to produce the six paper claims. Developed and **smoke-validated for
correctness on Qwen2.5-0.5B** (shared GPU). Result collection is deferred to a
free GPU. See `BENCH_PARAMS.md` for the lit-validated parameter grid.

## Storyline (the six claims)

1. **C1 — Full MHA underutilizes the GPU.** All KV in HBM hits a capacity/OOM
   wall in a memory-bound regime, capping concurrency before compute saturates.
2. **C2 — Layer-wise KV pipelining lifts throughput.** Offloading KV to CPU DDR
   with a K-layer GPU window lets the engine sustain more sequences -> higher tput.
3. **C3 — Pipelining adds an ITL tax at low concurrency.** The control-plane
   orchestration (CPU<->GPU sync), not the bytes, dominates at conc=1. Evidence:
   `FAKE_TRANSFER` (R3, zero-byte) ~= real (R1), so it's orchestration, not PCIe.
4. **C4 — The advanced design (B1+B2) amortizes that tax over the batch.** The
   per-sequence amortized feature tax `(R1 - R0)/conc` falls with concurrency and
   becomes negligible vs single-request baseline ITL.
5. **C5 — At high concurrency, real data-movement + HBM-competition costs appear**
   and must be measured (injection sweep) to drive the scheduler.
6. **C6 — An SLO-aware admission scheduler improves goodput.** Using the measured
   ITL(K) predictor, admit/defer to hold p99 ITL under the SLO; goodput beats stock
   across SLOs and offered rates.

## Claim -> script -> output

| Claim | Script | Output | Smoke status |
|-------|--------|--------|--------------|
| C1+C2 | `run_c1_c2_capacity.sh` (wraps `eval_pd_decode.py`) | `results_{baseline,pipelined}_conc*.json`, `summary.json` | validated (baseline + pipelined run, parseable JSON) |
| C3 | `../prove_conc1_thesis.sh` + `../summarize_thesis_conc1.py` | `THESIS_RESULTS.md` | pre-validated (prior work) |
| C4 | `../prove_claim2_conc_sweep.sh` + `../summarize_thesis_conc_sweep.py` | `THESIS_CLAIM2_RESULTS.md` | pre-validated (prior work) |
| C5 | `../run_injection_sweep_v5.py` (in `run_all.sh` full path) | `injection_sweep_v5.jsonl`, `.csv` | existing tool; long (tune via `C5_EXTRA`) |
| C6 | `run_c6_goodput.sh` + `summarize_goodput.py` (wraps `bench_admission_policy_e2e.py`) | `GOODPUT_RESULTS.md`, `goodput_summary.csv` | validated (stock+policy, grid summarized) |
| C6 calibration | `../feature_on_k_sweep.py` -> predictor CSV | `feature_on_k_sweep.csv` | consumer load validated at runtime; CSV schema superset-compatible |
| Correctness | `validate_qa.py` (self-consistency band); control: `qa_nondeterminism_control.py` | exact-match / similarity | see note below |

## Orchestrator

```bash
# Correctness smoke (Qwen2.5-0.5B, tiny params) — runs C1/C2 + C6:
SMOKE=1 bash run_all.sh /tmp/paper_smoke

# Full collection (free GPU) — runs C1/C2, C6, C3, C4, C5; prints index:
bash run_all.sh /data/ben/paper_runs/$(date +%Y%m%d)
# (C5 is long; shorten it with e.g. C5_EXTRA="--smoke" bash run_all.sh ...)
```

`run_all.sh` writes `PAPER_RESULTS.md` (index) and includes a **safe orphan
reaper**: it kills only engine cores orphaned to init (ppid==1), which a crashed
server frontend can leave holding GPU. The ppid==1 filter makes it safe on a
shared GPU (a healthy job keeps its live api_server parent).

## Collection-phase commands (free GPU)

Run from the repo root, or rely on the wrappers exporting `PYTHONPATH` (vLLM is a
source install importable only from the repo root).

```bash
# C1+C2 capacity wall (primary: Llama-3.1-8B 32L; sharpest wall: Qwen2.5-32B 64L)
MODEL=meta-llama/Llama-3.1-8B-Instruct MAX_MODEL_LEN=33000 RIN=8192 \
  CONC="64 128 256 384 512" NUM_PROMPTS=200 GPU_UTIL=0.90 \
  bash run_c1_c2_capacity.sh /data/ben/paper_runs/C1_C2

# C3 orchestration floor (R0/R1/R3 at conc=1)
TRIALS=5 GPU_UTIL=0.60 bash ../prove_conc1_thesis.sh /data/ben/paper_runs/C3 9300

# C4 amortization sweep (per-seq tax vs conc)
TRIALS=3 CONC="1 2 4 8 16 32 64" GPU_UTIL=0.60 \
  bash ../prove_claim2_conc_sweep.sh /data/ben/paper_runs/C4 9400

# C5 data-movement / HBM-competition cost (injection sweep) — long
python ../run_injection_sweep_v5.py   # see its --help for out dir / shapes

# C6 calibration: build the ITL(K) predictor used by the scheduler
python ../feature_on_k_sweep.py \
  --models meta-llama/Llama-3.1-8B-Instruct \
  --k-values 16 32 64 128 192 256 384 \
  --input-len 1024 --output-len 1024 --k-window 4 --repeats 2 \
  --out /data/ben/paper_runs/C6_calib

# C6 goodput (multi-SLO x multi-rate, stock vs policy) using that predictor
PREDICTOR_CSV=/data/ben/paper_runs/C6_calib/feature_on_k_sweep.csv \
  MODEL=meta-llama/Llama-3.1-8B-Instruct \
  SLOS="100 150 200 300" RATES="2 4 8 16 32" NUM_PROMPTS=200 \
  bash run_c6_goodput.sh /data/ben/paper_runs/C6_goodput
```

## Correctness note (C2)

`validate_qa.py` is **bit-exact at conc=1** (30/30) — the pipelining mechanism does
not corrupt KV. Its conc=8 exact-match drop (~63%) is **batched-execution FP
nondeterminism in vanilla vLLM, not a pipelining bug**: the baseline-vs-baseline
control (`qa_nondeterminism_control.py`) self-disagrees ~73% at conc=8, the same
prompts diverge, divergence starts mid-sequence after long identical prefixes, and
all outputs stay coherent.

**`validate_qa.py` now encodes this** (`--baseline-self-consistency`, default on):
at conc>1 it runs a second identical baseline pass and judges the pipelined arm
against the baseline's own self-consistency band (avg-similarity within
`SELF_BAND_AVG_MARGIN`=0.05 and exact-rate within `SELF_BAND_EXACT_MARGIN`=0.10),
plus an absolute corruption floor (`CORRUPTION_FLOOR_AVG`=0.80) that still fails
genuinely broken KV. conc=1 keeps the strict near-exact criterion. Use
`--no-baseline-self-consistency` for the old behavior.

```bash
# Standalone control (one server, two passes, reports run-to-run exact match):
python qa_nondeterminism_control.py --model Qwen/Qwen2.5-0.5B-Instruct --concurrency 8
# Full correctness check now passes legitimately at conc>1:
python ../validate_qa.py --model Qwen/Qwen2.5-0.5B-Instruct --k-window 4 --concurrency 1 8
```

## Environment

- Python: `<repo>/.venv/bin/python`; vLLM imports only with cwd=repo-root or
  `PYTHONPATH=<repo-root>` (the wrappers set this).
- `HF_HUB_CACHE=/data/ben/huggingface`.
- Smoke uses `gpu_memory_utilization=0.30`, `max_model_len=2048`; collection uses
  `0.90` and long context per `BENCH_PARAMS.md`.
