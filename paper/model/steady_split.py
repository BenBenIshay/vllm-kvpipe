import json, sys
from multiprocessing import Pool
sys.path.insert(0, __import__('os').path.dirname(__file__))
from steady_state import GH200, fixed, lognorm, run
from steady_sweep import MODELS, max_ctx
D = ("hbm", "hetero", "split", "pie")
def case(args):
    mk, wl, O, ov = args
    m = MODELS[mk]; hi = max_ctx(m, O)
    if wl.startswith("fixed"):
        lo_, hi_ = [int(x) for x in wl.split(":")[1:]]
        acc = {d: 0.0 for d in D}; cs = list(range(lo_, min(hi_, hi) + 1, 1000))
        for c in cs:
            o = run(GH200, m, fixed(c, O), n_req=60, overheads=ov, designs=D)
            for d in D: acc[d] += o[d] / o["hbm"] / len(cs)
    else:
        _, med, sig = wl.split(":")
        o = run(GH200, m, lognorm(int(med), float(sig), O, hi=hi), n_req=300, overheads=ov, designs=D)
        acc = {d: o[d] / o["hbm"] for d in D}
    return dict(model=mk, wl=wl, O=O, ov=ov, **{d: round(acc[d], 4) for d in D[1:]})
grid = [(mk, wl, O, ov) for mk in MODELS for wl in ("fixed:4000:16000", "fixed:16000:44000",
        "logn:6000:1.0", "logn:12000:1.0") for O in (128, 512, 2048) for ov in (True, False)]
with Pool(32) as p:
    for r in p.imap_unordered(case, grid):
        print(json.dumps(r), flush=True)
