#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P4 — validate the decode-step ITL model on real hardware.

The whole hybrid-KV tiering plan rests on one formula:

    ITL(B, S) = (W + B * S * k) / (BW * eff)

  W  = model weight bytes      k  = KV bytes per token
  B  = decode batch            S  = context length
  BW = peak HBM bandwidth      eff = achieved fraction of peak

Three predictions this script tests:
  A. ITL is LINEAR in B at fixed S, with intercept W/(BW*eff) and
     slope S*k/(BW*eff).
  B. ITL is LINEAR in S at fixed B, same intercept.
  C. At a FULL KV pool, ITL ~= HBM_usable/(BW*eff) -- independent of model
     and context. (This is the load-bearing one: the "rider budget is a
     hardware constant" result and the SLO gate both follow from it.)

Method: differential timing. Each config is run twice, generating N1 and then
N2 tokens; ITL = (T2 - T1) / (N2 - N1). This cancels prefill, warmup, sampling
setup and teardown exactly, leaving only steady-state decode steps.

Contexts are RANDOM DISTINCT token ids and prefix caching is OFF -- a shared
prefix makes KV reads look ~3.7x cheaper than they are and is precisely the
confound that produced our earlier (wrong) weight-bound conclusion.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def kv_bytes_per_token(cfg, tp: int = 1) -> int:
    hf = cfg.hf_text_config
    n_layer = hf.num_hidden_layers
    n_kv = getattr(hf, "num_key_value_heads", hf.num_attention_heads)
    head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
    return int(n_layer * 2 * (n_kv / tp) * head_dim * 2)


def weight_bytes_from_checkpoint(model_id: str) -> int:
    """Sum *.safetensors in the HF cache — robust across vLLM versions."""
    import glob
    try:
        from huggingface_hub import snapshot_download
        d = snapshot_download(model_id, allow_patterns=["*.json"])
    except Exception:
        return 0
    total = 0
    for f in glob.glob(os.path.join(d, "**", "*.safetensors"), recursive=True):
        total += os.path.getsize(f)
    if total:
        return total
    # snapshot dir holds symlinks into ../blobs; walk the whole model cache
    root = os.path.dirname(os.path.dirname(d))
    for f in glob.glob(os.path.join(root, "**", "*.safetensors"), recursive=True):
        try:
            total += os.path.getsize(os.path.realpath(f))
        except OSError:
            pass
    return total


def probe_num_gpu_blocks(llm) -> int | None:
    """num_gpu_blocks moved around across vLLM versions; try known paths."""
    for path in (
        lambda: llm.llm_engine.cache_config.num_gpu_blocks,
        lambda: llm.llm_engine.model_config.cache_config.num_gpu_blocks,
        lambda: llm.llm_engine.engine_core.cache_config.num_gpu_blocks,
        lambda: llm.llm_engine.vllm_config.cache_config.num_gpu_blocks,
    ):
        try:
            v = path()
            if v:
                return int(v)
        except Exception:
            pass
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--batches", default="1,2,4,8,16,24,32,48")
    p.add_argument("--contexts", default="4096")
    p.add_argument("--max-model-len", type=int, default=0,
                   help="0 = max(contexts) + n2 + 64")
    p.add_argument("--max-num-seqs", type=int, default=0, help="0 = max(batches)")
    p.add_argument("--n1", type=int, default=32)
    p.add_argument("--n2", type=int, default=96)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--out", default="p4_results.json")
    p.add_argument("--peak-bw-gbs", type=float, default=1935.0,
                   help="A100 80GB PCIe = 1935; SXM = 2039")
    p.add_argument("--weight-bytes", type=int, default=0,
                   help="0 = sum *.safetensors from the HF cache")
    p.add_argument("--repeats", type=int, default=1,
                   help="independent ITL measurements per config (error bars)")
    args = p.parse_args()

    batches = [int(x) for x in args.batches.split(",") if x]
    contexts = [int(x) for x in args.contexts.split(",") if x]
    max_len = args.max_model_len or (max(contexts) + args.n2 + 64)
    max_seqs = args.max_num_seqs or max(batches)

    import numpy as np
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=args.model,
        dtype="float16",
        enable_prefix_caching=False,       # critical: no shared-prefix discount
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=max_len,
        max_num_seqs=max_seqs,
        enforce_eager=False,               # CUDA graphs on = realistic decode
        disable_log_stats=True,
    )

    cfg = llm.llm_engine.model_config
    k = kv_bytes_per_token(cfg)
    w_bytes = args.weight_bytes or weight_bytes_from_checkpoint(args.model)
    assert w_bytes > 0, "could not determine weight bytes; pass --weight-bytes"

    n_blocks = probe_num_gpu_blocks(llm)
    block_size = cfg.get_head_size() and llm.llm_engine.cache_config.block_size
    pool_tokens = (n_blocks * block_size) if (n_blocks and block_size) else None

    meta = {
        "model": args.model,
        "kv_bytes_per_token": k,
        "weight_bytes": w_bytes,
        "num_gpu_blocks": n_blocks,
        "block_size": block_size,
        "pool_tokens": pool_tokens,
        "pool_bytes": pool_tokens * k if pool_tokens else None,
        "peak_bw_gbs": args.peak_bw_gbs,
        "max_model_len": max_len,
        "max_num_seqs": max_seqs,
        "gpu_mem_util": args.gpu_mem_util,
    }
    print("\n=== config ===")
    for kk, vv in meta.items():
        print(f"  {kk:22} {vv}")
    if pool_tokens:
        print(f"  KV pool = {pool_tokens*k/1e9:.1f} GB "
              f"({pool_tokens/1000:.0f}k tokens)")

    rng = np.random.default_rng(1234)
    vocab = cfg.get_vocab_size()

    def make_prompts(bsz: int, ctx: int):
        # distinct random contexts -> zero prefix sharing between requests
        return [
            TokensPrompt(
                prompt_token_ids=rng.integers(
                    10, vocab - 10, size=ctx, dtype=np.int64).tolist())
            for _ in range(bsz)
        ]

    def run(prompts, ntok: int) -> float:
        # SAME prompts for the N1/N2 pair so prefill cancels exactly
        sp = SamplingParams(temperature=0.0, max_tokens=ntok,
                            ignore_eos=True, min_tokens=ntok)
        t0 = time.perf_counter()
        out = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        got = min(len(o.outputs[0].token_ids) for o in out)
        assert got == ntok, f"expected {ntok} tokens, got {got}"
        return dt

    # warmup: trigger CUDA-graph capture and allocator steady state
    run(make_prompts(min(batches), min(contexts)), 8)

    results = []
    for ctx in contexts:
        for b in batches:
            need = b * (ctx + args.n2)
            if pool_tokens and need > pool_tokens:
                print(f"  SKIP B={b:>3} S={ctx:>6}: needs {need/1000:.0f}k "
                      f"tokens > pool {pool_tokens/1000:.0f}k "
                      f"(would not run as one batch)")
                continue
            prompts = make_prompts(b, ctx)
            samples = []
            for _ in range(args.repeats):
                t1 = run(prompts, args.n1)
                t2 = run(prompts, args.n2)
                samples.append((t2 - t1) / (args.n2 - args.n1) * 1e3)
            itl_ms = sum(samples) / len(samples)
            spread = (max(samples) - min(samples)) if len(samples) > 1 else 0.0
            sd = (sum((s - itl_ms) ** 2 for s in samples)
                  / max(1, len(samples) - 1)) ** 0.5 if len(samples) > 1 else 0.0
            pred_bytes = w_bytes + b * ctx * k
            eff = (pred_bytes / (itl_ms / 1e3)) / (args.peak_bw_gbs * 1e9)
            kv_frac = (b * ctx * k) / pred_bytes
            r = dict(batch=b, ctx=ctx, itl_ms=itl_ms, itl_sd=sd,
                     itl_spread=spread, samples=samples,
                     bytes_per_step=pred_bytes, implied_eff=eff,
                     kv_fraction=kv_frac)
            results.append(r)
            print(f"  B={b:>3} S={ctx:>6}  ITL={itl_ms:7.2f} ms "
                  f"+/-{sd:5.2f} (n={len(samples)}, spread {spread:4.2f})   "
                  f"bytes/step={pred_bytes/1e9:6.2f} GB   "
                  f"implied_eff={eff:5.3f}   KV={100*kv_frac:4.1f}%")

    payload = {"meta": meta, "results": results}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}  ({len(results)} points)")

    del llm
    gc.collect()


if __name__ == "__main__":
    main()
