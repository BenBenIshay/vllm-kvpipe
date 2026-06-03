#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Real Q&A Correctness & Quality Validation
==========================================

Sends a diverse set of real Q&A prompts to both baseline and pipelined
vLLM servers and compares the outputs for correctness.

This validates that KV cache pipelining produces IDENTICAL (or near-identical)
outputs to standard vLLM — i.e., the pipelining mechanism does not corrupt
the KV cache or alter model behaviour.

Tests (see --concurrency; conc=1 always runs first when included):
  1. Phase 1 — single-request (conc=1): one prompt at a time. Catches most
     pipelining bugs without cross-request interference.
  2. Phase 2 — multi-request (conc>1): overlapping requests. Catches block-pool
     / slicing bugs (e.g. decode_max_block_id) that only appear with concurrency.
  3. Quality spot-check: sample outputs for human review.

Usage:
    python benchmarks/kv_pipelining_bench/validate_qa.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --k-window 6

    # Quick check: Phase 1 only (fastest)
    python benchmarks/kv_pipelining_bench/validate_qa.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct --k-window 6 --concurrency 1

    # Full: Phase 1 then multi-request (default: 1 8). By default both servers
    # use --enforce-eager so torch.compile does not elide KV connector hooks.

    python benchmarks/kv_pipelining_bench/validate_qa.py \\
        --model Qwen/Qwen2.5-14B-Instruct --k-window 6 --concurrency 1 8

    # Stop after Phase 1 if thresholds fail (skip Phase 2)
    python benchmarks/kv_pipelining_bench/validate_qa.py \\
        --model ... --concurrency 1 8 --skip-multi-request-if-phase1-fails
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# =========================================================================
# Q&A Prompts — diverse set covering factual, reasoning, code, math, etc.
# =========================================================================

QA_PROMPTS = [
    # --- Factual knowledge ---
    {
        "id": "fact_01",
        "category": "factual",
        "prompt": "What is the capital of Australia? Answer in one sentence.",
    },
    {
        "id": "fact_02",
        "category": "factual",
        "prompt": "Who wrote the novel '1984'? Give the author's full name.",
    },
    {
        "id": "fact_03",
        "category": "factual",
        "prompt": "What is the chemical formula for water? Explain briefly.",
    },
    {
        "id": "fact_04",
        "category": "factual",
        "prompt": "Name the three states of matter and give one example of each.",
    },
    {
        "id": "fact_05",
        "category": "factual",
        "prompt": "What year did the first Moon landing take place, and who were the astronauts?",
    },
    # --- Reasoning ---
    {
        "id": "reason_01",
        "category": "reasoning",
        "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning step by step.",
    },
    {
        "id": "reason_02",
        "category": "reasoning",
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show your work.",
    },
    {
        "id": "reason_03",
        "category": "reasoning",
        "prompt": "You have 8 balls, one of which is heavier. Using a balance scale, what is the minimum number of weighings needed to find the heavy ball? Explain.",
    },
    {
        "id": "reason_04",
        "category": "reasoning",
        "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets? Explain your reasoning.",
    },
    # --- Math ---
    {
        "id": "math_01",
        "category": "math",
        "prompt": "Calculate: (17 * 23) + (45 / 9) - 12. Show each step.",
    },
    {
        "id": "math_02",
        "category": "math",
        "prompt": "What is the derivative of f(x) = 3x^2 + 2x - 5?",
    },
    {
        "id": "math_03",
        "category": "math",
        "prompt": "Solve for x: 2x + 7 = 19. Show your steps.",
    },
    # --- Code ---
    {
        "id": "code_01",
        "category": "code",
        "prompt": "Write a Python function that checks if a string is a palindrome. Include a docstring.",
    },
    {
        "id": "code_02",
        "category": "code",
        "prompt": "Write a Python function to compute the nth Fibonacci number using recursion with memoization.",
    },
    {
        "id": "code_03",
        "category": "code",
        "prompt": "Explain the difference between a list and a tuple in Python. Give one code example for each.",
    },
    # --- Summarization ---
    {
        "id": "summary_01",
        "category": "summarization",
        "prompt": (
            "Summarize the following paragraph in 2 sentences:\n\n"
            "The Industrial Revolution, which took place from the 18th to 19th centuries, "
            "was a period during which predominantly agrarian, rural societies in Europe "
            "and America became industrial and urban. Prior to the Industrial Revolution, "
            "which began in Britain in the late 1700s, manufacturing was often done in "
            "people's homes, using hand tools or basic machines. Industrialization marked "
            "a shift to powered, special-purpose machinery, factories, and mass production. "
            "The iron and textile industries, along with the development of the steam engine, "
            "played central roles in the Industrial Revolution."
        ),
    },
    {
        "id": "summary_02",
        "category": "summarization",
        "prompt": (
            "Summarize the key idea in one sentence:\n\n"
            "Photosynthesis is the process by which green plants and some other organisms "
            "use sunlight to synthesize nutrients from carbon dioxide and water. "
            "Photosynthesis in plants generally involves the green pigment chlorophyll "
            "and generates oxygen as a by-product."
        ),
    },
    # --- Creative / Open-ended ---
    {
        "id": "creative_01",
        "category": "creative",
        "prompt": "Write a haiku about the ocean.",
    },
    {
        "id": "creative_02",
        "category": "creative",
        "prompt": "Explain quantum computing to a 10-year-old in 3 sentences.",
    },
    # --- Multi-step Q&A ---
    {
        "id": "multi_01",
        "category": "multi_step",
        "prompt": (
            "Answer these questions one by one:\n"
            "1. What is the largest planet in our solar system?\n"
            "2. What is its approximate diameter in kilometers?\n"
            "3. How many known moons does it have (approximately)?"
        ),
    },
    {
        "id": "multi_02",
        "category": "multi_step",
        "prompt": (
            "Step by step, explain how to make a peanut butter and jelly sandwich. "
            "List each step numbered."
        ),
    },
    # --- Longer context ---
    {
        "id": "context_01",
        "category": "contextual",
        "prompt": (
            "Read the passage and answer the question.\n\n"
            "Passage: The Great Wall of China is a series of fortifications that were "
            "built across the historical northern borders of ancient Chinese states and "
            "Imperial China as protection against various nomadic groups. Several walls "
            "were built from as early as the 7th century BC, with selective stretches "
            "later joined together by Qin Shi Huang (220-206 BC), the first emperor of "
            "China. Little of the Qin wall remains. Later on, many successive dynasties "
            "built and maintained multiple stretches of border walls. The most well-known "
            "sections were built by the Ming dynasty (1368-1644).\n\n"
            "Question: Which dynasty built the most well-known sections of the Great Wall?"
        ),
    },
    {
        "id": "context_02",
        "category": "contextual",
        "prompt": (
            "Read the passage and answer the question.\n\n"
            "Passage: Machine learning is a subset of artificial intelligence (AI) that "
            "provides systems the ability to automatically learn and improve from "
            "experience without being explicitly programmed. Machine learning focuses on "
            "the development of computer programs that can access data and use it to learn "
            "for themselves. The process begins with observations or data, such as examples, "
            "direct experience, or instruction, in order to look for patterns in data and "
            "make better decisions in the future.\n\n"
            "Question: What is the main goal of machine learning according to the passage?"
        ),
    },
    # --- Edge cases / tricky ---
    {
        "id": "tricky_01",
        "category": "tricky",
        "prompt": "Is the following statement true or false: 'This statement is false.' Explain the paradox.",
    },
    {
        "id": "tricky_02",
        "category": "tricky",
        "prompt": "What comes next in the sequence: 2, 6, 12, 20, 30, ...? Explain the pattern.",
    },
    # --- Translation ---
    {
        "id": "translate_01",
        "category": "translation",
        "prompt": "Translate 'Hello, how are you today?' into French and Spanish.",
    },
    # --- Instruction following ---
    {
        "id": "instruct_01",
        "category": "instruction",
        "prompt": "List exactly 5 prime numbers between 10 and 50.",
    },
    {
        "id": "instruct_02",
        "category": "instruction",
        "prompt": "Give me 3 pros and 3 cons of remote work. Format as bullet points.",
    },
    {
        "id": "instruct_03",
        "category": "instruction",
        "prompt": "Sort these animals by size (smallest to largest): elephant, mouse, cat, whale, dog.",
    },
    # --- General knowledge ---
    {
        "id": "general_01",
        "category": "general",
        "prompt": "What are the three primary colors of light, and what color do you get when you mix all three?",
    },
]

assert len(QA_PROMPTS) == 30, f"Expected 30 prompts, got {len(QA_PROMPTS)}"

# =========================================================================
# Constants
# =========================================================================
DEFAULT_PORT = 8199
MAX_TOKENS = 256
TEMPERATURE = 0.0  # greedy — deterministic for comparison


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real Q&A validation for KV pipelining")
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.72,
        help=(
            "Default 0.72 leaves headroom when other processes hold VRAM fragments. "
            "Raise if the GPU is empty and you want a larger KV reservation."
        ),
    )
    p.add_argument(
        "--free-original-kv-cache",
        dest="free_original_kv_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Pass --free-original-kv-cache to the pipelined server only. "
            "Try --free-original-kv-cache if pipelined outputs diverge from baseline "
            "(some layouts reclaim per-layer tensors during decode)."
        ),
    )
    p.add_argument(
        "--no-chunked-prefill-pipelined",
        dest="no_chunked_prefill_pipelined",
        action="store_true",
        default=False,
        help=(
            "Pass --no-enable-chunked-prefill to the pipelined server only "
            "(correctness debugging; less representative of production schedules)."
        ),
    )
    p.add_argument("--k-window", type=int, default=6)
    p.add_argument(
        "--inflate-kv-pool",
        action="store_true",
        help=(
            "Pass --inflate-kv-pool to the pipelined server (capacity mode; "
            "uses much more CPU pinned memory — off by default for correctness runs)."
        ),
    )
    p.add_argument("--startup-timeout", type=int, default=600)
    p.add_argument("--result-dir", default="/tmp/kv_qa_validation")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Use only the first N prompts from the built-in set (faster smoke test). "
            "Default: all prompts."
        ),
    )
    p.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 8],
        metavar="N",
        help=(
            "Concurrency levels to run. Order is normalized: conc=1 runs first "
            "(single-request), then other values ascending (multi-request). "
            "Default: 1 8. Use `--concurrency 1` for the minimal correctness pass."
        ),
    )
    p.add_argument(
        "--skip-multi-request-if-phase1-fails",
        action="store_true",
        help=(
            "If Phase 1 (conc=1) finishes but fails similarity thresholds, "
            "do not run any conc>1 phase."
        ),
    )
    p.add_argument(
        "--enforce-eager",
        dest="enforce_eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pass --enforce-eager to vLLM for both baseline and pipelined servers "
            "(disables torch.compile / Inductor; required for apples-to-apples KV "
            "connector correctness vs default vLLM, which can fuse across attention). "
            "Use --no-enforce-eager only if debugging performance."
        ),
    )
    p.add_argument(
        "--strict-pd-emulation",
        action="store_true",
        default=False,
        help=(
            "Debug mode: approximate strict P/D separation by forcing "
            "--no-enable-chunked-prefill on pipelined runs and issuing an "
            "unmeasured prefill-focused warmup pass before measured requests. "
            "This is a harness-level approximation, not true disaggregated RPC."
        ),
    )
    p.add_argument(
        "--prefill-warmup-max-tokens",
        type=int,
        default=1,
        help=(
            "max_tokens used by the prefill-focused warmup pass when "
            "--strict-pd-emulation is enabled."
        ),
    )
    p.add_argument(
        "--baseline-self-consistency",
        dest="baseline_self_consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "At conc>1, run a second identical baseline pass to measure vanilla "
            "vLLM's own run-to-run nondeterminism, and judge the pipelined arm "
            "against that band instead of near-exact match (the conc>1 exact-match "
            "drop is batched-execution FP nondeterminism, not a pipelining bug). "
            "Use --no-baseline-self-consistency for the old strict criterion."
        ),
    )
    return p.parse_args()


def ordered_concurrency_levels(levels: list[int]) -> list[int]:
    """conc=1 first when present, then remaining unique values ascending."""
    if not levels:
        return [1, 8]
    uniq = list(dict.fromkeys(levels))
    rest = sorted(x for x in uniq if x != 1)
    out: list[int] = []
    if 1 in uniq:
        out.append(1)
    out.extend(rest)
    return out


def phase_label(concurrency: int) -> str:
    if concurrency == 1:
        return "Phase 1 — single-request (conc=1)"
    return f"Phase 2 — multi-request (conc={concurrency})"


def similarity_threshold_pass(summary: dict) -> bool:
    """Strict criteria (avg > 0.90, min > 0.50). Used at conc=1 where vanilla
    vLLM is deterministic, so pipelining must match baseline near-exactly."""
    if summary.get("total_prompts", 0) == 0:
        return False
    return summary["avg_similarity"] > 0.90 and summary["min_similarity"] > 0.50


# At conc>1, vanilla vLLM is itself nondeterministic (batched-GEMM/attention
# reduction order changes which argmax flips), so requiring near-exact match
# pipelined-vs-baseline is a false negative. Instead we judge the pipelined
# arm against the BASELINE's own run-to-run self-consistency band, plus an
# absolute corruption floor that still catches genuinely broken KV.
SELF_BAND_AVG_MARGIN = 0.05   # pipelined avg_sim may trail baseline self-sim by this
SELF_BAND_EXACT_MARGIN = 0.10  # pipelined exact-rate may trail baseline self-exact by this (fraction)
CORRUPTION_FLOOR_AVG = 0.80   # avg_sim at/below this => real corruption regardless of band


def band_pass(summary: dict, baseline_self: dict | None, concurrency: int) -> bool:
    """Pass/fail for a pipelined-vs-baseline comparison.

    conc<=1 (or no self-consistency reference): strict thresholds.
    conc>1: pass if the pipelined arm stays within the baseline's own
    self-consistency band (avg similarity and exact-match rate), and clears the
    absolute corruption floor.
    """
    if concurrency <= 1 or not baseline_self:
        return similarity_threshold_pass(summary)
    n = summary.get("total_prompts", 0)
    if n == 0:
        return False
    if summary["avg_similarity"] <= CORRUPTION_FLOOR_AVG:
        return False
    self_n = baseline_self.get("total_prompts", 0) or n
    self_avg = baseline_self.get("avg_similarity", 1.0)
    self_exact_rate = baseline_self.get("exact_matches", 0) / self_n if self_n else 1.0
    pipe_exact_rate = summary.get("exact_matches", 0) / n
    avg_ok = summary["avg_similarity"] >= self_avg - SELF_BAND_AVG_MARGIN
    exact_ok = pipe_exact_rate >= self_exact_rate - SELF_BAND_EXACT_MARGIN
    return bool(avg_ok and exact_ok)


# =========================================================================
# Server management (reused from eval scripts)
# =========================================================================

def _cleanup_port(port: int) -> None:
    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"],
                       capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    time.sleep(2)


def _wait_for_server(port: int, timeout: int) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(2)
    return False


def _kill_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        except Exception:
            pass


def launch_server(
    args: argparse.Namespace,
    mode: str,
    log_file: str,
) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--disable-log-requests",
        "--dtype", "auto",
        "--max-num-seqs", "64",
    ]
    if getattr(args, "enforce_eager", True):
        cmd.append("--enforce-eager")

    if mode == "pipelined":
        kv_config = {
            "kv_connector": "PrefillStagingPipelinedConnector",
            "kv_role": "kv_both",
        }
        cmd += [
            "--enable-kv-pipelining",
            "--kv-k-window", str(args.k_window),
            "--kv-transfer-config", json.dumps(kv_config),
            "--no-enable-prefix-caching",
        ]
        if args.inflate_kv_pool:
            cmd.append("--inflate-kv-pool")
        if getattr(args, "free_original_kv_cache", False):
            cmd.append("--free-original-kv-cache")
        # Chunked prefill + pipelined connector has been a source of misclassified
        # steps; single-shot prefill simplifies the schedule for correctness runs.
        if mode == "pipelined" and getattr(
            args, "no_chunked_prefill_pipelined", False
        ):
            cmd.append("--no-enable-chunked-prefill")

    log_fh = open(log_file, "w")
    proc = subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT, preexec_fn=os.setsid,
    )
    return proc


# =========================================================================
# Request sending via OpenAI API
# =========================================================================

@dataclass
class QAResult:
    prompt_id: str
    category: str
    prompt: str
    response: str
    ttft_ms: float
    e2e_ms: float
    tokens_generated: int
    finish_reason: str


async def send_single_request(
    session,
    port: int,
    model: str,
    prompt_info: dict,
    max_tokens: int,
) -> QAResult:
    """Send a single chat completion request and return the result."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt_info["prompt"]},
        ],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": False,
    }

    t0 = time.monotonic()
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
    e2e = (time.monotonic() - t0) * 1000.0

    if "error" in data:
        return QAResult(
            prompt_id=prompt_info["id"],
            category=prompt_info["category"],
            prompt=prompt_info["prompt"],
            response=f"[ERROR] {data['error']}",
            ttft_ms=-1,
            e2e_ms=e2e,
            tokens_generated=0,
            finish_reason="error",
        )

    choice = data["choices"][0]
    usage = data.get("usage", {})
    response_text = choice["message"]["content"]
    completion_tokens = usage.get("completion_tokens", len(response_text.split()))

    return QAResult(
        prompt_id=prompt_info["id"],
        category=prompt_info["category"],
        prompt=prompt_info["prompt"],
        response=response_text,
        ttft_ms=-1,  # non-streaming — not available
        e2e_ms=e2e,
        tokens_generated=completion_tokens,
        finish_reason=choice.get("finish_reason", "unknown"),
    )


async def send_all_requests(
    port: int,
    model: str,
    prompts: list[dict],
    max_tokens: int,
    concurrency: int,
) -> list[QAResult]:
    """Send prompts with the given concurrency level."""
    import aiohttp

    sem = asyncio.Semaphore(concurrency)
    results: list[QAResult] = []

    async def _bounded(session, prompt_info):
        async with sem:
            return await send_single_request(
                session, port, model, prompt_info, max_tokens
            )

    # Pipelined decode can be much slower than baseline (CPU staging + PCIe).
    timeout = aiohttp.ClientTimeout(total=900, sock_read=900, sock_connect=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [_bounded(session, p) for p in prompts]
        results = await asyncio.gather(*tasks)

    return list(results)


def run_qa_batch(
    port: int,
    model: str,
    prompts: list[dict],
    max_tokens: int,
    concurrency: int,
) -> list[QAResult]:
    """Synchronous wrapper for async request sending."""
    return asyncio.run(
        send_all_requests(port, model, prompts, max_tokens, concurrency)
    )


def maybe_run_prefill_warmup(
    args: argparse.Namespace,
    port: int,
    prompts: list[dict],
    server_label: str,
) -> None:
    """Run an unmeasured warmup pass to isolate prompt processing behavior.

    In strict P/D emulation mode, we run a small warmup with low max_tokens
    before each measured phase. This is a harness-level approximation of
    separating prefill-heavy work from decode-heavy work.
    """
    if not getattr(args, "strict_pd_emulation", False):
        return
    warmup_tokens = max(0, int(getattr(args, "prefill_warmup_max_tokens", 1)))
    print(
        f"  [warmup] {server_label}: prefill-focused pass "
        f"(conc=1, max_tokens={warmup_tokens}) ..."
    )
    _ = run_qa_batch(
        port=port,
        model=args.model,
        prompts=prompts,
        max_tokens=warmup_tokens,
        concurrency=1,
    )


# =========================================================================
# Comparison logic
# =========================================================================

def compute_similarity(a: str, b: str) -> float:
    """Compute sequence similarity ratio between two strings (0.0 - 1.0)."""
    return difflib.SequenceMatcher(None, a, b).ratio()


def compare_results(
    baseline_results: list[QAResult],
    pipelined_results: list[QAResult],
    label: str,
) -> dict:
    """Compare baseline vs pipelined results, return summary stats."""
    # Build lookup by prompt_id
    base_map = {r.prompt_id: r for r in baseline_results}
    pipe_map = {r.prompt_id: r for r in pipelined_results}

    exact_matches = 0
    high_similarity = 0  # > 0.95
    similarities = []
    comparisons = []

    all_ids = sorted(set(base_map.keys()) | set(pipe_map.keys()))

    for pid in all_ids:
        b = base_map.get(pid)
        p = pipe_map.get(pid)
        if not b or not p:
            comparisons.append({
                "id": pid,
                "status": "missing",
                "note": f"baseline={'present' if b else 'missing'}, pipelined={'present' if p else 'missing'}",
            })
            continue

        sim = compute_similarity(b.response, p.response)
        similarities.append(sim)
        exact = b.response == p.response
        if exact:
            exact_matches += 1
        if sim > 0.95:
            high_similarity += 1

        comparisons.append({
            "id": pid,
            "category": b.category,
            "exact_match": exact,
            "similarity": round(sim, 4),
            "baseline_len": len(b.response),
            "pipelined_len": len(p.response),
            "baseline_tokens": b.tokens_generated,
            "pipelined_tokens": p.tokens_generated,
            "baseline_e2e_ms": round(b.e2e_ms, 1),
            "pipelined_e2e_ms": round(p.e2e_ms, 1),
            "baseline_finish": b.finish_reason,
            "pipelined_finish": p.finish_reason,
        })

    n = len(similarities)
    avg_sim = sum(similarities) / n if n > 0 else 0.0

    summary = {
        "label": label,
        "total_prompts": n,
        "exact_matches": exact_matches,
        "exact_match_rate": f"{exact_matches}/{n} ({100*exact_matches/n:.1f}%)" if n else "N/A",
        "high_similarity_count": high_similarity,
        "high_similarity_rate": f"{high_similarity}/{n} ({100*high_similarity/n:.1f}%)" if n else "N/A",
        "avg_similarity": round(avg_sim, 4),
        "min_similarity": round(min(similarities), 4) if similarities else 0,
        "comparisons": comparisons,
    }
    return summary


def print_comparison_report(summary: dict, baseline_results: list[QAResult], pipelined_results: list[QAResult]) -> None:
    """Print a detailed comparison report."""
    base_map = {r.prompt_id: r for r in baseline_results}
    pipe_map = {r.prompt_id: r for r in pipelined_results}

    label = summary["label"]
    print(f"\n{'=' * 90}")
    print(f"  {label}")
    print(f"{'=' * 90}")
    print(f"  Total prompts:        {summary['total_prompts']}")
    print(f"  Exact matches:        {summary['exact_match_rate']}")
    print(f"  High similarity (>95%): {summary['high_similarity_rate']}")
    print(f"  Average similarity:   {summary['avg_similarity']:.4f}")
    print(f"  Min similarity:       {summary['min_similarity']:.4f}")

    # Per-prompt details
    print(f"\n  {'ID':<15s} {'Category':<14s} {'Match':>6s} {'Sim':>7s} {'Base E2E':>10s} {'Pipe E2E':>10s} {'Base tok':>9s} {'Pipe tok':>9s}")
    print(f"  {'─'*15} {'─'*14} {'─'*6} {'─'*7} {'─'*10} {'─'*10} {'─'*9} {'─'*9}")

    for c in summary["comparisons"]:
        if c.get("status") == "missing":
            print(f"  {c['id']:<15s} {'MISSING':<14s}")
            continue
        match_str = "YES" if c["exact_match"] else "no"
        print(
            f"  {c['id']:<15s} {c['category']:<14s} "
            f"{match_str:>6s} {c['similarity']:>7.4f} "
            f"{c['baseline_e2e_ms']:>9.1f}ms {c['pipelined_e2e_ms']:>9.1f}ms "
            f"{c['baseline_tokens']:>9d} {c['pipelined_tokens']:>9d}"
        )

    # Show diffs for non-exact matches
    mismatches = [c for c in summary["comparisons"]
                  if not c.get("exact_match") and c.get("status") != "missing"]
    if mismatches:
        print(f"\n  --- Detailed diffs for {len(mismatches)} non-exact matches ---")
        for c in mismatches[:5]:  # Show at most 5
            pid = c["id"]
            b = base_map.get(pid)
            p = pipe_map.get(pid)
            if not b or not p:
                continue
            print(f"\n  [{pid}] (similarity={c['similarity']:.4f})")
            print(f"  Prompt: {b.prompt[:80]}...")

            # Show unified diff
            b_lines = b.response.splitlines(keepends=True)
            p_lines = p.response.splitlines(keepends=True)
            diff = list(difflib.unified_diff(
                b_lines, p_lines,
                fromfile="baseline", tofile="pipelined",
                lineterm="",
            ))
            if diff:
                for line in diff[:20]:
                    print(f"    {line.rstrip()}")
                if len(diff) > 20:
                    print(f"    ... ({len(diff) - 20} more diff lines)")
            else:
                print(f"    (no line-level diff — differences are whitespace/formatting)")

    # Show a few sample outputs for human spot-check
    print(f"\n  --- Sample outputs (first 5 prompts) ---")
    for c in summary["comparisons"][:5]:
        if c.get("status") == "missing":
            continue
        pid = c["id"]
        b = base_map.get(pid)
        p = pipe_map.get(pid)
        if not b or not p:
            continue
        print(f"\n  [{pid}] {b.category}")
        print(f"  Q: {b.prompt[:100]}{'...' if len(b.prompt) > 100 else ''}")
        # Truncate long responses
        b_resp = b.response[:300] + ("..." if len(b.response) > 300 else "")
        p_resp = p.response[:300] + ("..." if len(p.response) > 300 else "")
        print(f"  Baseline:  {b_resp}")
        print(f"  Pipelined: {p_resp}")
        match_icon = "MATCH" if c["exact_match"] else f"DIFF (sim={c['similarity']:.3f})"
        print(f"  -> {match_icon}")


# =========================================================================
# Latency summary
# =========================================================================

def print_latency_summary(
    baseline_results: list[QAResult],
    pipelined_results: list[QAResult],
    label: str,
) -> None:
    """Print latency comparison."""
    def stats(results):
        e2es = [r.e2e_ms for r in results if r.e2e_ms > 0]
        if not e2es:
            return {}
        e2es.sort()
        n = len(e2es)
        return {
            "mean_e2e": sum(e2es) / n,
            "median_e2e": e2es[n // 2],
            "p99_e2e": e2es[int(n * 0.99)],
            "min_e2e": e2es[0],
            "max_e2e": e2es[-1],
            "total_tokens": sum(r.tokens_generated for r in results),
        }

    b_stats = stats(baseline_results)
    p_stats = stats(pipelined_results)

    print(f"\n  Latency Summary ({label})")
    print(f"  {'Metric':<20s} {'Baseline':>12s} {'Pipelined':>12s} {'Delta':>10s}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*10}")

    for key, name in [
        ("mean_e2e", "Mean E2E (ms)"),
        ("median_e2e", "Median E2E (ms)"),
        ("p99_e2e", "P99 E2E (ms)"),
        ("min_e2e", "Min E2E (ms)"),
        ("max_e2e", "Max E2E (ms)"),
    ]:
        bv = b_stats.get(key)
        pv = p_stats.get(key)
        if bv is not None and pv is not None and bv > 0:
            delta = ((pv - bv) / bv) * 100
            print(f"  {name:<20s} {bv:>11.1f}  {pv:>11.1f}  {delta:>+9.1f}%")
        else:
            print(f"  {name:<20s} {'N/A':>12s} {'N/A':>12s}")

    bt = b_stats.get("total_tokens", 0)
    pt = p_stats.get("total_tokens", 0)
    print(f"  {'Total tokens':<20s} {bt:>12d} {pt:>12d}")


# =========================================================================
# Main orchestrator
# =========================================================================

def main() -> None:
    args = parse_args()
    if getattr(args, "strict_pd_emulation", False):
        # Keep the emulation behavior deterministic: disable chunked prefill
        # for the pipelined server regardless of CLI default.
        args.no_chunked_prefill_pipelined = True
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    model_short = args.model.split("/")[-1]

    qa_prompts = QA_PROMPTS
    if getattr(args, "max_prompts", None) is not None:
        n = max(0, int(args.max_prompts))
        qa_prompts = QA_PROMPTS[:n]

    levels = ordered_concurrency_levels(args.concurrency)
    if any(c < 1 for c in levels):
        print("ERROR: --concurrency values must be >= 1", file=sys.stderr)
        sys.exit(2)

    print("=" * 90)
    print("  Real Q&A Correctness Validation for KV Cache Pipelining")
    print("=" * 90)
    print(f"  Model:      {args.model}")
    print(f"  K-window:   {args.k_window}")
    print(f"  Inflate pool: {getattr(args, 'inflate_kv_pool', False)}")
    print(f"  Free original KV (pipelined only): {getattr(args, 'free_original_kv_cache', False)}")
    print(
        "  No chunked prefill (pipelined only): "
        f"{getattr(args, 'no_chunked_prefill_pipelined', False)}"
    )
    if getattr(args, "strict_pd_emulation", False):
        print("  Strict P/D emulation: enabled")
        print(
            "  Warmup pass: "
            f"conc=1, max_tokens={max(0, int(args.prefill_warmup_max_tokens))}"
        )
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Enforce eager (both servers): {getattr(args, 'enforce_eager', True)}")
    print(f"  Prompts:    {len(qa_prompts)}")
    print(f"  Concurrency phases: {levels} (conc=1 runs first when included)")
    if args.skip_multi_request_if_phase1_fails and 1 in levels:
        print("  Stop if Phase 1 fails: yes (skip conc>1)")
    print(f"  Result dir: {args.result_dir}")
    print()

    all_summaries = {}
    all_self = {}

    for concurrency in levels:
        conc_label = phase_label(concurrency)
        print(f"\n{'#' * 90}")
        print(f"# TEST: {conc_label}")
        print(f"{'#' * 90}")

        # --- Baseline ---
        print(f"\n  [1/2] Launching BASELINE server ...")
        _cleanup_port(args.port)
        base_log = str(result_dir / f"server_baseline_conc{concurrency}.log")
        base_proc = launch_server(args, "baseline", base_log)

        try:
            print(f"  Waiting for server (timeout={args.startup_timeout}s) ...")
            if not _wait_for_server(args.port, args.startup_timeout):
                print(f"  [FAIL] Baseline server failed to start. Check: {base_log}")
                continue

            maybe_run_prefill_warmup(
                args=args,
                port=args.port,
                prompts=qa_prompts,
                server_label="baseline",
            )

            print(f"  [OK] Baseline server ready. Sending {len(qa_prompts)} prompts (conc={concurrency}) ...")
            t0 = time.monotonic()
            baseline_results = run_qa_batch(
                args.port, args.model, qa_prompts, args.max_tokens, concurrency
            )
            base_time = time.monotonic() - t0
            print(f"  Baseline done in {base_time:.1f}s "
                  f"({sum(1 for r in baseline_results if r.finish_reason != 'error')}/{len(qa_prompts)} succeeded)")

            # Second identical baseline pass to measure vanilla vLLM's own
            # run-to-run nondeterminism at this concurrency (the reference band).
            baseline_self_summary = None
            if (
                getattr(args, "baseline_self_consistency", True)
                and concurrency > 1
            ):
                print(f"  [self-consistency] Re-running baseline (conc={concurrency}) ...")
                baseline_results_b = run_qa_batch(
                    args.port, args.model, qa_prompts, args.max_tokens, concurrency
                )
                baseline_self_summary = compare_results(
                    baseline_results, baseline_results_b,
                    f"{conc_label} [baseline self-consistency]",
                )
                print(
                    "  [self-consistency] baseline vs baseline: "
                    f"exact={baseline_self_summary['exact_matches']}/"
                    f"{baseline_self_summary['total_prompts']}, "
                    f"avg_sim={baseline_self_summary['avg_similarity']:.4f}"
                )
        finally:
            _kill_server(base_proc)
            time.sleep(15)

        # --- Pipelined ---
        _inf = " + inflated pool" if args.inflate_kv_pool else ""
        print(f"\n  [2/2] Launching PIPELINED server (K={args.k_window}{_inf}) ...")
        _cleanup_port(args.port)
        pipe_log = str(result_dir / f"server_pipelined_conc{concurrency}.log")
        pipe_proc = launch_server(args, "pipelined", pipe_log)

        try:
            print(f"  Waiting for server (timeout={args.startup_timeout}s) ...")
            if not _wait_for_server(args.port, args.startup_timeout):
                print(f"  [FAIL] Pipelined server failed to start. Check: {pipe_log}")
                continue

            maybe_run_prefill_warmup(
                args=args,
                port=args.port,
                prompts=qa_prompts,
                server_label="pipelined",
            )

            print(f"  [OK] Pipelined server ready. Sending {len(qa_prompts)} prompts (conc={concurrency}) ...")
            t0 = time.monotonic()
            pipelined_results = run_qa_batch(
                args.port, args.model, qa_prompts, args.max_tokens, concurrency
            )
            pipe_time = time.monotonic() - t0
            print(f"  Pipelined done in {pipe_time:.1f}s "
                  f"({sum(1 for r in pipelined_results if r.finish_reason != 'error')}/{len(qa_prompts)} succeeded)")
        finally:
            _kill_server(pipe_proc)
            time.sleep(15)

        # --- Compare ---
        summary = compare_results(baseline_results, pipelined_results, conc_label)
        summary["concurrency"] = concurrency
        if baseline_self_summary is not None:
            summary["baseline_self_consistency"] = {
                k: v for k, v in baseline_self_summary.items() if k != "comparisons"
            }
        print_comparison_report(summary, baseline_results, pipelined_results)
        print_latency_summary(baseline_results, pipelined_results, conc_label)

        all_summaries[conc_label] = summary
        all_self[conc_label] = baseline_self_summary

        # Save per-concurrency results
        out = {
            "concurrency": concurrency,
            "baseline": [
                {
                    "id": r.prompt_id, "category": r.category,
                    "prompt": r.prompt, "response": r.response,
                    "e2e_ms": r.e2e_ms, "tokens": r.tokens_generated,
                    "finish_reason": r.finish_reason,
                }
                for r in baseline_results
            ],
            "pipelined": [
                {
                    "id": r.prompt_id, "category": r.category,
                    "prompt": r.prompt, "response": r.response,
                    "e2e_ms": r.e2e_ms, "tokens": r.tokens_generated,
                    "finish_reason": r.finish_reason,
                }
                for r in pipelined_results
            ],
            "comparison": summary,
        }
        out_path = result_dir / f"qa_results_conc{concurrency}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Results saved: {out_path}")

        if (
            args.skip_multi_request_if_phase1_fails
            and concurrency == 1
            and len(levels) > 1
        ):
            if not similarity_threshold_pass(summary):
                print("\n" + "=" * 90)
                print(
                    "  Phase 1 failed similarity thresholds — "
                    "skipping multi-request phases (--skip-multi-request-if-phase1-fails)"
                )
                print("=" * 90)
                break

    # =========================================================================
    # Final verdict
    # =========================================================================
    print(f"\n{'=' * 90}")
    print("  FINAL VERDICT")
    print(f"{'=' * 90}")

    # If every run failed (e.g. server did not start), do not report PASS.
    all_pass = bool(all_summaries)
    if not all_summaries:
        print(
            "\n  [FAIL] No validation runs completed successfully "
            "(check server logs — gated models need `huggingface-cli login` "
            "or `HF_TOKEN`, and the model must be present in the HF cache)."
        )

    for label, summary in all_summaries.items():
        n = summary["total_prompts"]
        exact = summary["exact_matches"]
        avg_sim = summary["avg_similarity"]
        min_sim = summary["min_similarity"]
        conc = summary.get("concurrency", 1)
        baseline_self = all_self.get(label)

        passed = band_pass(summary, baseline_self, conc)

        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        print(f"\n  [{status}] {label}")
        print(f"    Exact matches:      {exact}/{n}")
        if conc > 1 and baseline_self:
            self_n = baseline_self["total_prompts"]
            self_avg = baseline_self["avg_similarity"]
            print(
                f"    Criterion: baseline self-consistency band "
                f"(baseline-vs-baseline: exact={baseline_self['exact_matches']}/{self_n}, "
                f"avg_sim={self_avg:.4f})"
            )
            print(
                f"    Average similarity: {avg_sim:.4f} "
                f"(band floor {self_avg - SELF_BAND_AVG_MARGIN:.4f}; "
                f"corruption floor {CORRUPTION_FLOOR_AVG:.2f})"
            )
        else:
            print(f"    Criterion: strict (conc=1, baseline deterministic)")
            print(f"    Average similarity: {avg_sim:.4f} {'(OK)' if avg_sim > 0.90 else '(BELOW THRESHOLD 0.90)'}")
            print(f"    Min similarity:     {min_sim:.4f} {'(OK)' if min_sim > 0.50 else '(BELOW THRESHOLD 0.50)'}")

    print(f"\n  {'=' * 50}")
    if all_pass:
        print("  OVERALL: PASS — KV pipelining produces correct outputs")
    else:
        print("  OVERALL: FAIL — KV pipelining may have correctness issues")
    print(f"  {'=' * 50}")

    # Save final summary
    final_path = result_dir / "validation_summary.json"
    with open(final_path, "w") as f:
        json.dump({
            "model": args.model,
            "k_window": args.k_window,
            "max_tokens": args.max_tokens,
            "strict_pd_emulation": getattr(args, "strict_pd_emulation", False),
            "prefill_warmup_max_tokens": max(
                0, int(getattr(args, "prefill_warmup_max_tokens", 1))
            ),
            "num_prompts": len(qa_prompts),
            "concurrency_levels": levels,
            "skip_multi_request_if_phase1_fails": (
                args.skip_multi_request_if_phase1_fails
            ),
            "overall_pass": all_pass,
            "summaries": {
                k: {kk: vv for kk, vv in v.items() if kk != "comparisons"}
                for k, v in all_summaries.items()
            },
        }, f, indent=2)
    print(f"\n  Validation summary: {final_path}")
    print(f"  Full results:       {result_dir}/")


if __name__ == "__main__":
    main()
