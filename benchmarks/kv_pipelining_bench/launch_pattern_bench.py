#!/usr/bin/env python3
"""Price the two remaining overhead items BEFORE building either one.

T3 was built, measured correct, and turned out to be a 6x regression, because
the mechanism it relied on (fewer launches) was not the binding cost -- the
binding cost was contiguity, and the strided view it introduced broke it. The
warning was in a comment in the file being edited. This benchmark exists so the
same class of mistake cannot happen twice: it replays the EXACT copy patterns
the pool issues, engine-free, and prices each candidate change as an upper bound
before any of it is implemented.

Two questions, both measured here:

  Q1  THE D2H SIDE OF F. Measured post-T2: write_token spends 2.477 ms per step
      in `write_token_d2h_async` and it is FLAT in riders (+2% from 1 to 8), so
      it is fixed cost -- the largest single remaining item in F. It is 2*L
      small D2H launches per step, one pair per layer. Candidate T4 replaces
      them with 2*L device-to-device copies into a device staging tensor plus
      ONE pair of D2H launches for the whole step. That is only a win if a D2D
      launch is materially cheaper than a D2H launch at this size; if it is not,
      T4 trades 2L launches for 2L launches and buys nothing.

  Q2  THE H2D SIDE OF c. The per-rider launch term is 2*L*r contiguous copies,
      one per run per tensor. Adjacent extents would merge them into ONE
      contiguous copy per tensor per layer -- the allocator change the pool's
      own comment names and declines to attempt. That change is substantial
      (requests grow, so tight extents need compaction), so the upper bound on
      what it could buy is worth knowing first. Measured as r contiguous copies
      against 1 contiguous copy of the same total bytes.

WHAT IS REPORTED is the HOST time to issue the pattern, which is the quantity F
and c are made of -- exposed host work on the critical path -- plus the device
time, to show the bytes are actually moving. A variant that cuts host time while
raising device time past the step budget is not a win, which is precisely how T3
failed.

    ./.venv/bin/python benchmarks/kv_pipelining_bench/launch_pattern_bench.py
"""
import argparse
import json
import statistics
import sys
import time

import torch

# The real shapes. bs/nkv/hd are Qwen-32B's and the block row is 32 KB at fp16.
BS, NKV, HD = 16, 8, 128
DTYPE = torch.float16


def _sync():
    torch.cuda.synchronize()


def time_pattern(fn, reps=30, warmup=10):
    """(host_ms, device_ms) medians.

    Host time is measured around the ISSUE only, with no synchronize inside the
    timed region -- putting a synchronize inside the timing wrapper is the
    documented mistake that made async double-buffering look worthless in this
    repo once already.
    """
    for _ in range(warmup):
        fn()
    _sync()
    host, dev = [], []
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(reps):
        _sync()
        ev0.record()
        t = time.perf_counter()
        fn()
        host.append((time.perf_counter() - t) * 1e3)
        ev1.record()
        _sync()
        dev.append(ev0.elapsed_time(ev1))
    return statistics.median(host), statistics.median(dev)


# --------------------------------------------------------------------------
# Q1: the D2H side of F
# --------------------------------------------------------------------------
def q1(L, n, reps):
    """2L D2H launches (as built) vs 2L D2D + 2 D2H (T4) vs 2 D2H (the floor)."""
    src_k = [torch.randn((n, NKV, HD), dtype=DTYPE, device="cuda")
             for _ in range(L)]
    src_v = [torch.randn((n, NKV, HD), dtype=DTYPE, device="cuda")
             for _ in range(L)]
    host_k = torch.empty((L, n, NKV, HD), dtype=DTYPE, device="cpu",
                         pin_memory=True)
    host_v = torch.empty((L, n, NKV, HD), dtype=DTYPE, device="cpu",
                         pin_memory=True)
    dev_k = torch.empty((L, n, NKV, HD), dtype=DTYPE, device="cuda")
    dev_v = torch.empty((L, n, NKV, HD), dtype=DTYPE, device="cuda")
    # unified device staging, K and V adjacent so ONE D2H moves both
    dev_kv = torch.empty((2, L, n, NKV, HD), dtype=DTYPE, device="cuda")
    host_kv = torch.empty((2, L, n, NKV, HD), dtype=DTYPE, device="cpu",
                          pin_memory=True)

    def as_built():                      # 2L D2H
        for li in range(L):
            host_k[li].copy_(src_k[li], non_blocking=True)
            host_v[li].copy_(src_v[li], non_blocking=True)

    def t4_split():                      # 2L D2D + 2 D2H
        for li in range(L):
            dev_k[li].copy_(src_k[li], non_blocking=True)
            dev_v[li].copy_(src_v[li], non_blocking=True)
        host_k.copy_(dev_k, non_blocking=True)
        host_v.copy_(dev_v, non_blocking=True)

    def t4_unified():                    # 2L D2D + 1 D2H
        for li in range(L):
            dev_kv[0, li].copy_(src_k[li], non_blocking=True)
            dev_kv[1, li].copy_(src_v[li], non_blocking=True)
        host_kv.copy_(dev_kv, non_blocking=True)

    def floor_only():                    # 2 D2H, ignoring how data got staged
        host_k.copy_(dev_k, non_blocking=True)
        host_v.copy_(dev_v, non_blocking=True)

    rows = []
    for lab, fn, launches in (
            ("as built: 2L D2H", as_built, 2 * L),
            ("T4: 2L D2D + 2 D2H", t4_split, 2 * L + 2),
            ("T4u: 2L D2D + 1 D2H", t4_unified, 2 * L + 1),
            ("floor: 2 D2H only", floor_only, 2)):
        h, d = time_pattern(fn, reps)
        rows.append({"variant": lab, "host_ms": h, "dev_ms": d,
                     "launches": launches})
    return rows


# --------------------------------------------------------------------------
# Q2: the H2D side of c
# --------------------------------------------------------------------------
def q2(L, riders, blocks_per_rider, stride, reps):
    """r contiguous copies per tensor (as built) vs 1 (adjacent extents).

    `stride` is the spacing between extent bases. With stride > blocks the runs
    cannot merge, which is the situation today. The merged variant copies the
    same total bytes as ONE contiguous range, which is the upper bound the
    allocator change is worth.
    """
    w = riders * blocks_per_rider
    h_max = max(riders * stride + blocks_per_rider, w) + 8
    hk = torch.empty((h_max, BS, NKV, HD), dtype=DTYPE, device="cpu",
                     pin_memory=True)
    hv = torch.empty((h_max, BS, NKV, HD), dtype=DTYPE, device="cpu",
                     pin_memory=True)
    hk.normal_()
    hv.normal_()
    dk = torch.empty((w, BS, NKV, HD), dtype=DTYPE, device="cuda")
    dv = torch.empty((w, BS, NKV, HD), dtype=DTYPE, device="cuda")
    nb = blocks_per_rider
    per_tensor_bytes = w * BS * NKV * HD * 2

    def per_run():                       # 2*r copies, each contiguous: AS BUILT
        for i in range(riders):
            hs, cs = i * stride, i * nb
            dk.narrow(0, cs, nb).copy_(hk.narrow(0, hs, nb), non_blocking=True)
            dv.narrow(0, cs, nb).copy_(hv.narrow(0, hs, nb), non_blocking=True)

    def merged():                        # 2 copies: ADJACENT EXTENTS
        dk.copy_(hk.narrow(0, 0, w), non_blocking=True)
        dv.copy_(hv.narrow(0, 0, w), non_blocking=True)

    def strided():                       # what T3 did, kept as the control
        sk = hk.as_strided((riders, nb) + hk.shape[1:],
                           (stride * hk.stride(0), hk.stride(0))
                           + hk.stride()[1:], 0)
        sv = hv.as_strided((riders, nb) + hv.shape[1:],
                           (stride * hv.stride(0), hv.stride(0))
                           + hv.stride()[1:], 0)
        dk.view((riders, nb) + dk.shape[1:]).copy_(sk, non_blocking=True)
        dv.view((riders, nb) + dv.shape[1:]).copy_(sv, non_blocking=True)

    # THE SLACK VARIANT, which is the only form of adjacency a real allocator
    # can deliver. Merging runs needs the USED blocks adjacent, but a decode
    # request grows one block per `bs` tokens, so an extent sized for its final
    # length carries unused slack at its tail and the next extent's base sits
    # past it. The gap defeats merging however tightly the extents are packed.
    #
    # The way out is to stop copying the used blocks and copy whole EXTENTS
    # instead: with extents adjacent, all r of them are one contiguous range, so
    # it is one launch -- at the price of moving the slack as well. That trade
    # is only worth it if the launch saving beats the extra bytes, which is a
    # measurement, not an argument. Slack here is the tail of an extent sized
    # for prompt + max_tokens while the request is mid-generation.
    def slack_variant(slack_blocks):
        tot = riders * (nb + slack_blocks)
        hk2 = torch.empty((tot + 8, BS, NKV, HD), dtype=DTYPE, device="cpu",
                          pin_memory=True)
        hk2.normal_()
        dk2 = torch.empty((tot, BS, NKV, HD), dtype=DTYPE, device="cuda")
        dv2 = torch.empty((tot, BS, NKV, HD), dtype=DTYPE, device="cuda")

        def fn():
            dk2.copy_(hk2.narrow(0, 0, tot), non_blocking=True)
            dv2.copy_(hk2.narrow(0, 0, tot), non_blocking=True)
        return fn, 2 * tot * BS * NKV * HD * 2

    rows = []
    cases = [("as built: 2r contiguous", per_run, 2 * riders,
              2 * per_tensor_bytes),
             ("adjacent, no slack: 2", merged, 2, 2 * per_tensor_bytes),
             ("T3 strided (reverted)", strided, 2, 2 * per_tensor_bytes)]
    for sl in (4, 8, 16):
        fn, by = slack_variant(sl)
        cases.append(("adjacent, %d-blk slack: 2" % sl, fn, 2, by))
    for lab, fn, launches, nbytes in cases:
        h, d = time_pattern(fn, reps)
        rows.append({"variant": lab, "host_ms": h, "dev_ms": d,
                     "launches": launches,
                     "bytes_x": nbytes / (2 * per_tensor_bytes),
                     "gbps": nbytes / (d / 1e3) / 1e9 if d else 0})
    return rows


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", type=int, default=64)
    p.add_argument("--rows", type=int, default=24,
                   help="tokens written per layer per step (B_res + r)")
    p.add_argument("--riders", type=int, default=8)
    p.add_argument("--blocks-per-rider", type=int, default=33,
                   help="ceil(S/bs); 33 is ctx 512 at block_size 16")
    p.add_argument("--stride", type=int, default=141,
                   help="extent spacing; 141 is max_model_len 2256 at bs 16")
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--out", default="")
    a = p.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA -- this benchmark needs a GPU")
        return 2
    print("device: %s" % torch.cuda.get_device_name(0))
    print("shapes: block row %d KB, L=%d, rows/layer=%d, riders=%d, "
          "blocks/rider=%d, stride=%d"
          % (BS * NKV * HD * 2 // 1024, a.layers, a.rows, a.riders,
             a.blocks_per_rider, a.stride))
    print()

    print("=" * 76)
    print(" Q1  THE D2H SIDE OF F  -- is batching the per-layer D2H a win?")
    print("=" * 76)
    print(" %-24s %10s %10s %10s"
          % ("variant", "launches", "host ms", "dev ms"))
    r1 = q1(a.layers, a.rows, a.reps)
    for r in r1:
        print(" %-24s %10d %10.3f %10.3f"
              % (r["variant"], r["launches"], r["host_ms"], r["dev_ms"]))
    base = r1[0]["host_ms"]
    print()
    for r in r1[1:]:
        print("   %-24s host %+6.1f%%  (%.3f -> %.3f ms)"
              % (r["variant"], (r["host_ms"] - base) / base * 100, base,
                 r["host_ms"]))
    print()

    print("=" * 76)
    print(" Q2  THE H2D SIDE OF c  -- what would adjacent extents buy?")
    print("=" * 76)
    print(" %-26s %9s %7s %8s %8s %8s"
          % ("variant", "launches", "bytes", "host ms", "dev ms", "GB/s"))
    r2 = q2(a.layers, a.riders, a.blocks_per_rider, a.stride, a.reps)
    for r in r2:
        print(" %-26s %9d %6.2fx %8.3f %8.3f %8.1f"
              % (r["variant"], r["launches"], r["bytes_x"], r["host_ms"],
                 r["dev_ms"], r["gbps"]))
    base2 = r2[0]["host_ms"]
    print()
    for r in r2[1:]:
        tot0 = r2[0]["host_ms"] + r2[0]["dev_ms"]
        tot = r["host_ms"] + r["dev_ms"]
        print("   %-26s host %+7.1f%%, dev %+7.1f%%, host+dev %+7.1f%%"
              % (r["variant"], (r["host_ms"] - base2) / base2 * 100,
                 (r["dev_ms"] - r2[0]["dev_ms"]) / r2[0]["dev_ms"] * 100,
                 (tot - tot0) / tot0 * 100))
    print()
    print(" The T3 row is the CONTROL: it must show the regression that was")
    print(" measured in the engine (device time up, bandwidth collapsed). If it")
    print(" does not, this benchmark is not reproducing the real behaviour and")
    print(" neither Q1 nor Q2 should be acted on.")
    print()

    if a.out:
        json.dump({"device": torch.cuda.get_device_name(0),
                   "args": vars(a), "q1": r1, "q2": r2},
                  open(a.out, "w"), indent=1)
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
