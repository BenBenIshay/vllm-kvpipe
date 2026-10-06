#!/usr/bin/env python3
"""The three rider-attend modes do what the M19 experiment's arithmetic assumes.

WHY THIS TEST AND NOT A FLAG-MATRIX ENTRY. `VLLM_KVPIPE_RUNGD_ATTEND_MODE` is
read in `flash_attn.py`, not in the pool constructor, so `pool_construct_test`
cannot reach it. What can go wrong here is control flow, and it is the kind of
mistake that is invisible in the result: if `nokernel` also skipped the scatter,
or `noattend` left the attention call in, the two arms would still produce
plausible numbers and the subtraction `noattend - nokernel` would silently mean
something else. The whole go/no-go on days of attention-backend work rests on
that subtraction, so the wiring is asserted rather than eyeballed.

WHAT THE EXPERIMENT NEEDS TO BE TRUE:

    mode       prepare_layer  write_token  attend   rider rows of `out`
    full            yes           yes       YES     overwritten (real result)
    noattend        yes           yes       no      UNTOUCHED
    nokernel        yes           yes       no      overwritten (stale buffer)

`noattend` minus `nokernel` is therefore exactly the gather + scatter, and
`nokernel`'s own saving is the kernel and its launch. If `noattend` and
`nokernel` ever agree on the last column, the arms are measuring the same thing
and the difference is noise dressed as a decomposition.

Also asserted: the `pool_cpu_override` phase timer is recorded in ALL three
modes. An early return that skipped it would zero the one Class-A number that
cross-checks the fit, and the loss would look like a real saving.

CPU only, no engine, no model. Run:
    ./.venv/bin/python benchmarks/kv_pipelining_bench/attend_ablation_test.py
"""
import json
import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


class RecordingPT:
    """Stands in for the phase timer, so we can see what was timed."""

    def __init__(s):
        s.phases, s.counters = [], []

    def now(s):
        return 0.0

    def add(s, name, _t):
        s.phases.append(name)

    def count(s, name):
        s.counters.append(name)


class Backend:
    """Records which mechanism calls the override actually made."""

    def __init__(s, m, heads, dim):
        s.calls = []
        s._m, s._heads, s._dim = m, heads, dim

    def prepare_layer(s, lidx):
        s.calls.append("prepare_layer")
        return (torch.zeros(1), torch.zeros(1))

    def write_token(s, *a, **k):
        s.calls.append("write_token")

    def attend(s, lidx, handle, q_sub, ctx, out_sub):
        s.calls.append("attend")
        # A value no stale buffer could hold, so "did the scatter land a REAL
        # attention result" is distinguishable from "it landed something".
        out_sub.fill_(7.0)


class Layer:
    layer_name = "L0"

    def __init__(s):
        s._q_scale = torch.ones(1)
        s._k_scale = torch.ones(1)
        s._v_scale = torch.ones(1)


class Split:
    def __init__(s, pool, cpu_idx, m):
        s.rungd_pool = pool
        s.cpu_idx = cpu_idx
        s.rolling_block_table = torch.zeros((m, 4), dtype=torch.int32)
        s.rolling_seqused_k = torch.ones(m, dtype=torch.int32)
        s.rungd_layer_index = {"L0": 0}
        s.tier_backend = None
        # Types taken from hybrid_attn_ctx.HybridDecodeSplit: these three are
        # plain lists (the override tests `cpu_cur_compact` for truthiness, so a
        # multi-element tensor would raise), while the two *_pos are tensors.
        s.cpu_cur_compact = list(range(m))
        s.cpu_cur_offset = [0] * m
        s.cpu_cur_hostblock = [0] * m
        s.cpu_slot_pos = torch.zeros(m, dtype=torch.int64)
        s.cpu_host_pos = torch.zeros(m, dtype=torch.int64)
        s.rolling_max_seqlen_k = 4
        s.cpu_cu_seqlens_q = None


class Pool:
    # No `writeback_layer` attribute at all, so the swap emulation stays out of
    # the way -- that is its shipped state (the flag is off by default).
    def __init__(s, n_layers):
        s.layers = list(range(n_layers))


def run_mode(mode, n_rows=6, m=2, heads=2, dim=8):
    """Call the real `_pool_cpu_override` once, in THIS process, under `mode`.

    Called only in the child. Reloading the module in-process does not work:
    `from vllm.v1.attention.backends import flash_attn` resolves through the
    parent package's cached attribute, so deleting the entry from `sys.modules`
    silently hands back the stale module with the mode already resolved. One
    process per mode is also what the real sbatch does -- every arm is a fresh
    python -- so this matches the thing being measured.
    """
    from vllm.v1.attention.backends import flash_attn as fa

    pt = RecordingPT()
    fa.PT = pt

    class Stub:
        # Bind the real method; never call the real __init__.
        _pool_cpu_override = fa.FlashAttentionImpl._pool_cpu_override
        scale = 1.0
        num_kv_heads = heads
        logits_soft_cap = None
        vllm_flash_attn_version = 2
        alibi_slopes = None
        sinks = None

    cpu_idx = torch.tensor([1, 3], dtype=torch.int64)
    pool = Pool(4)
    split = Split(pool, cpu_idx, m)
    backend = Backend(m, heads, dim)
    split.tier_backend = backend

    q = torch.zeros((n_rows, heads, dim))
    key = torch.zeros((n_rows, heads, dim))
    val = torch.zeros((n_rows, heads, dim))
    out = torch.zeros((n_rows, heads, dim))

    Stub()._pool_cpu_override(split, Layer(), q, key, val, out, n_rows, None)
    rider_rows = out.index_select(0, cpu_idx)
    return dict(calls=backend.calls, phases=pt.phases, counters=pt.counters,
                rider_max=float(rider_rows.abs().max()),
                rider_is_real=bool(torch.allclose(
                    rider_rows, torch.full_like(rider_rows, 7.0))))


def child(mode):
    """Run one mode and print its result as JSON for the parent."""
    print("___RESULT___" + json.dumps(run_mode(mode)))
    return 0


def spawn(mode):
    """One fresh process per mode, with the env the real sbatch would set."""
    env = dict(os.environ)
    env.pop("VLLM_KVPIPE_RUNGD_ATTEND_MODE", None)
    env.pop("VLLM_KVPIPE_RUNGD_ATTEND_MODE_UNSAFE", None)
    if mode != "full":
        env["VLLM_KVPIPE_RUNGD_ATTEND_MODE"] = mode
        env["VLLM_KVPIPE_RUNGD_ATTEND_MODE_UNSAFE"] = "1"
    p = subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--child", mode],
                       capture_output=True, text=True, env=env, timeout=300)
    for line in p.stdout.splitlines():
        if line.startswith("___RESULT___"):
            return json.loads(line[len("___RESULT___"):])
    raise RuntimeError("child for mode %s produced no result:\n%s\n%s"
                       % (mode, p.stdout[-2000:], p.stderr[-2000:]))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        return child(sys.argv[2])
    ok = True
    res = {}
    for mode in ("full", "noattend", "nokernel"):
        res[mode] = spawn(mode)

    def check(label, cond):
        nonlocal ok
        ok &= bool(cond)
        print("  [%s] %s" % ("ok " if cond else "BAD", label))

    print("THE THREE RIDER-ATTEND MODES")
    print()
    for mode in ("full", "noattend", "nokernel"):
        r = res[mode]
        print("  %-9s calls=%-42s counters=%s"
              % (mode, ",".join(r["calls"]) or "-",
                 ",".join(r["counters"]) or "-"))
    print()

    # The mechanism calls that must survive EVERY ablation: the H2D refresh and
    # the mirror write are not what fusion touches, so ablating them would
    # measure F instead of the fusion bound.
    for mode in ("full", "noattend", "nokernel"):
        check("%s keeps prepare_layer and write_token" % mode,
              "prepare_layer" in res[mode]["calls"]
              and "write_token" in res[mode]["calls"])

    check("full CALLS attend", "attend" in res["full"]["calls"])
    check("noattend does NOT call attend",
          "attend" not in res["noattend"]["calls"])
    check("nokernel does NOT call attend",
          "attend" not in res["nokernel"]["calls"])

    # The column the subtraction depends on.
    check("full lands a REAL attention result in the rider rows",
          res["full"]["rider_is_real"])
    check("noattend leaves the rider rows UNTOUCHED",
          res["noattend"]["rider_max"] == 0.0)
    check("nokernel still SCATTERS (gather+scatter retained, kernel gone)",
          "rider_scatter" in res["nokernel"]["phases"]
          and "rider_gather_alloc" in res["nokernel"]["phases"])
    check("noattend skips gather AND scatter",
          "rider_scatter" not in res["noattend"]["phases"]
          and "rider_gather_alloc" not in res["noattend"]["phases"])
    check("so noattend and nokernel are NOT the same ablation",
          set(res["noattend"]["phases"]) != set(res["nokernel"]["phases"]))

    # The timer that must never be lost to an early return.
    for mode in ("full", "noattend", "nokernel"):
        check("%s still records the pool_cpu_override phase" % mode,
              "pool_cpu_override" in res[mode]["phases"])

    # Each ablation announces itself in the counters, so a run's JSONL proves
    # which arm it was rather than relying on the filename.
    check("noattend sets its counter",
          "rider_attend_ablated" in res["noattend"]["counters"])
    check("nokernel sets its counter",
          "rider_kernel_ablated" in res["nokernel"]["counters"])
    check("full sets NEITHER ablation counter",
          not {"rider_attend_ablated", "rider_kernel_ablated"}
          & set(res["full"]["counters"]))

    n = 17
    print()
    print("ATTEND-ABLATION %s (%d checks)" % ("PASS" if ok else "FAIL", n))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
