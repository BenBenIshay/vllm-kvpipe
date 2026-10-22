"""Background link traffic (Ben, 2026-10-22): the 36 workloads of steady_split.py
with a share of one link direction taken by other work. d2h loads the direction
Pie's swap-out needs and riders barely use; h2d is the matched control that
loads the direction both designs read through. Output: one JSON line per case."""
import json
import sys
from multiprocessing import Pool

from steady_state import GH200, fixed, lognorm, run
from steady_sweep import MODELS, max_ctx

D = ("hbm", "hetero", "split", "pie")


def case(args):
    mk, wl, O, ov, dirn, beta = args
    m = MODELS[mk]
    hi = max_ctx(m, O)
    kw = dict(overheads=ov, designs=D, **{"bg_" + dirn: beta})
    if wl.startswith("fixed"):
        lo_, hi_ = [int(x) for x in wl.split(":")[1:]]
        cs = list(range(lo_, min(hi_, hi) + 1, 1000))
        acc = {d: 0.0 for d in D}
        for c in cs:
            o = run(GH200, m, fixed(c, O), n_req=60, **kw)
            for d in D:
                acc[d] += o[d] / o["hbm"] / len(cs)
    else:
        _, med, sig = wl.split(":")
        o = run(GH200, m, lognorm(int(med), float(sig), O, hi=hi), n_req=300, **kw)
        acc = {d: o[d] / o["hbm"] for d in D}
    return dict(model=mk, wl=wl, O=O, ov=ov, dir=dirn, beta=beta,
                **{d: round(acc[d], 4) for d in D[1:]})


if __name__ == "__main__":
    grid = [(mk, wl, O, True, dirn, beta) for mk in MODELS
            for wl in ("fixed:4000:16000", "fixed:16000:44000", "logn:6000:1.0",
                       "logn:12000:1.0")
            for O in (128, 512, 2048)
            for dirn, beta in (("d2h", 0.0), ("d2h", 0.1), ("d2h", 0.25), ("d2h", 0.5),
                               ("h2d", 0.1), ("h2d", 0.25), ("h2d", 0.5))]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 32) as p:
        for r in p.imap_unordered(case, grid):
            print(json.dumps(r), flush=True)
