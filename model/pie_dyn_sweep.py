"""Workload-shift grid for pie_dynamic.run_dyn(): how much Pie's controller
loses to its adaptation interval when the context mix alternates, and where it
stands against HeteroKV's per-step admission. One JSON line per case."""
import json
import sys
from multiprocessing import Pool

from pie_dynamic import queue_of, run_dyn
from steady_state import GH200
from steady_sweep import MODELS

SHIFTS = {
    "short/long": [(4000, 0.5, 512, 1000, 40000), (28000, 0.3, 512, 1000, 40000)],
    "mid/long": [(10000, 0.5, 512, 1000, 40000), (28000, 0.3, 512, 1000, 40000)],
    "steady": [(12000, 1.0, 512, 1000, 40000)],
}


def case(args):
    mk, sh, phase, A = args
    q = queue_of(SHIFTS[sh], phase, 300, 0)
    o, st = run_dyn(GH200, MODELS[mk], q, adapt=A)
    return dict(model=mk, shift=sh, phase=phase, adapt=A,
                **{d: round(v / o["hbm"], 4) for d, v in o.items() if d != "hbm"},
                stalls=st["pie_ctl"])


if __name__ == "__main__":
    grid = [(mk, sh, ph, A) for mk in ("q72_16", "q72_8") for sh in SHIFTS
            for ph in ((300,) if sh == "steady" else (20, 60, 200))
            for A in (1, 8, 64, 256)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as p:
        for r in p.imap_unordered(case, grid):
            print(json.dumps(r), flush=True)
