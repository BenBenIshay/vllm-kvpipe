"""Sweep of steady_state.run() over workloads, KV precision, output length and
model, to find where HeteroKV's margin over Pie's mechanism is largest and why.
Fixed contexts are reported as the mean over a range of contexts, so no single
capacity boundary decides the result. Output: one JSON line per case."""
import json
import sys
from multiprocessing import Pool

from steady_state import GH200, M, fixed, lognorm, run

Q72_16 = M("Qwen2.5-72B fp8w/fp16kv", 80, 76.0, 327680)
Q72_8 = M("Qwen2.5-72B fp8w/fp8kv", 80, 76.0, 163840)
Q32_16 = M("Qwen2.5-32B bf16w/fp16kv", 64, 65.5, 262144)
MODELS = {"q72_16": Q72_16, "q72_8": Q72_8, "q32_16": Q32_16}


def max_ctx(m, O):
    return int((GH200.cap - m.w - 4.0) * 1e9 / m.b_tok) - O - 16


def case(args):
    mk, wl, O, ov, st = args
    m = MODELS[mk]
    hi = max_ctx(m, O)
    if wl.startswith("fixed"):
        lo_, hi_ = [int(x) for x in wl.split(":")[1:]]
        ctxs = [c for c in range(lo_, min(hi_, hi) + 1, 1000)]
        gh = gp = 0.0
        for c in ctxs:
            o = run(GH200, m, fixed(c, O), n_req=60, overheads=ov, stage=st)
            gh += o["hetero"] / o["hbm"]; gp += o["pie"] / o["hbm"]
        gh, gp = gh / len(ctxs), gp / len(ctxs)
    else:
        _, med, sig = wl.split(":")
        o = run(GH200, m, lognorm(int(med), float(sig), O, hi=hi), n_req=300,
                overheads=ov, stage=st)
        gh, gp = o["hetero"] / o["hbm"], o["pie"] / o["hbm"]
    return dict(model=mk, wl=wl, O=O, overheads=ov, stage=st, g_het=round(gh, 4),
                g_pie=round(gp, 4), margin=round((gh / gp - 1) * 100, 2))


if __name__ == "__main__":
    grid = []
    for mk in MODELS:
        for wl in ("fixed:4000:16000", "fixed:16000:44000", "logn:6000:0.6",
                   "logn:6000:1.0", "logn:12000:0.6", "logn:12000:1.0"):
            for O in (128, 512, 2048):
                for ov, st in ((True, 2), (False, 2), (True, 0)):
                    grid.append((mk, wl, O, ov, st))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 16) as p:
        for r in p.imap_unordered(case, grid):
            print(json.dumps(r), flush=True)
