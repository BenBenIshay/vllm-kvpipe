#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Item 3 correctness gate: output must be token-identical ACROSS a demotion.

This is the acceptance criterion that matters. A request starts on the HBM
tier, decodes for a while, is demoted mid-flight to the host tier, and must
continue producing exactly the tokens vanilla vLLM would.

The mechanism under test: flipping the tier makes the runner seed the host pool
from the request's CURRENT resident blocks, so the host copy must contain the
full history at the moment of the flip. If seeding raced the scheduler's block
cap, or seeded a partial history, output would diverge here.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROMPTS = [
    "The history of the Roman empire be",
    "In quantum mechanics, the wave fun",
    "A recipe for sourdough bread start",
]
NTOK = 48
DEMOTE_AFTER = 12          # steps on HBM before forcing the demotion


def gen(demote: bool):
    from vllm import LLM, SamplingParams

    if demote:
        os.environ["VLLM_KVPIPE_HYBRID_RUNGD"] = "1"
        os.environ["VLLM_KVPIPE_HYBRID_RUNGD_FREE"] = "1"
        os.environ["VLLM_KVPIPE_TIER_POLICY"] = "scored"
    else:
        for k in ("VLLM_KVPIPE_HYBRID_RUNGD", "VLLM_KVPIPE_HYBRID_RUNGD_FREE"):
            os.environ.pop(k, None)
        os.environ["VLLM_KVPIPE_TIER_POLICY"] = "arrival"

    # in-process engine so we can reach the scheduler (no MP client)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="float16",
              enable_prefix_caching=False, gpu_memory_utilization=0.35,
              max_model_len=512, max_num_seqs=8, enforce_eager=True,
              disable_log_stats=True)

    if demote:
        # Force a mid-flight demotion of every running request after N steps by
        # driving the controller's own decision path -- not by bypassing it.
        core = llm.llm_engine.engine_core
        sched = getattr(getattr(core, "engine_core", core), "scheduler", None)
        if sched is None:
            raise RuntimeError(
                "could not reach the scheduler; set "
                "VLLM_ENABLE_V1_MULTIPROCESSING=0")
        ctl = sched.transfer_aware_admission
        ctl._tier_policy = "scored"
        ctl._migrate_min_remaining = 0
        ctl._migrate_cooldown_steps = 0
        ctl._max_migrations_per_step = 8
        orig = sched._apply_tier_migrations
        state = {"n": 0}

        def patched():
            state["n"] += 1
            if state["n"] == DEMOTE_AFTER:
                for r in list(sched.running):
                    if ctl.get_req_tier(r.request_id) != "cpu":
                        ctl.note_migrated(r.request_id, "cpu")
                        print(f"    [demote] {r.request_id} hbm -> cpu "
                              f"at step {state['n']}")
            else:
                orig()
        sched._apply_tier_migrations = patched

    sp = SamplingParams(temperature=0.0, max_tokens=NTOK, ignore_eos=True)
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)
    return [tuple(o.outputs[0].token_ids) for o in outs]


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("vanilla", "both"):
        print("=== VANILLA (no tiers) ===")
        v = gen(False)
        if mode == "vanilla":
            print("TOKENS:" + repr(v)); return 0
    if mode in ("demote", "both"):
        print("=== DEMOTED MID-FLIGHT ===")
        d = gen(True)
        if mode == "demote":
            print("TOKENS:" + repr(d)); return 0
    ok = all(a == b for a, b in zip(v, d))
    print("=" * 64)
    for i, (a, b) in enumerate(zip(v, d)):
        same = a == b
        print(f"[{'OK ' if same else 'DIFF'}] prompt {i}")
        if not same:
            first = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y),
                         min(len(a), len(b)))
            print(f"      first divergence at token {first}")
    print("=" * 64)
    if ok:
        print("MIGRATION CORRECTNESS PASS: output is token-identical across a "
              "mid-flight HBM->host demotion.")
        return 0
    print("MIGRATION CORRECTNESS FAIL: demotion changed the output.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
