#!/usr/bin/env python3
"""Extract the plotted series of section 7.1 from the campaign's run records.

Every CSV beside this script is written by it and by nothing else; the graphs
read only those CSVs. Sources are the harness artifacts archived verbatim in
paper/data/overhead/ (see its MANIFEST.json), plus the per-step phase timers of
the T1 arms, which the manifest keeps only in paper_runs (12 MB):

    .venv/bin/python -I paper/v2/figs/data/extract.py

Rules, the same as the analysis scripts in paper/model/ that the report rests on:
  * a row is used only if the purity gate passed it (`usable`);
  * dT is the harness's own per-row value (step time minus the r=0 step time);
  * a below-knee fit uses only rows with link_ms / T0 <= 1/3 (clean_cf.py);
  * the T1 timer drops the first 3 steps of each cohort size (first-touch) and
    discards an arm whose run record has any row rejected (t1_effect.py).
"""
import csv
import glob
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DATA = os.path.join(REPO, "paper", "data", "overhead")
RUNS = os.environ.get(
    "T1_PHASES", "/mnt/beegfs/benishay/paper_runs/20260930/overhead/08_opt")
MAX_LOAD = 1.0 / 3.0
PRE_T1_MS_PER_BLOCK = 0.00765     # overhead_model.COEFF["d_per_block"]


def rec(rel):
    return json.load(open(os.path.join(DATA, rel)))


def t0_of(d):
    return next(r["step_ms"] for r in d["rows"] if r["riders"] == 0)


def sweep(rel):
    """[(r, dT, dT_min, dT_max, load)] over usable rider rows."""
    d = rec(rel)
    t0 = t0_of(d)
    out = []
    for r in d["rows"]:
        if not r.get("usable") or not r["riders"] or r.get("dT_ms") is None:
            continue
        reps = [x["step_ms"] - t0 for x in r.get("reps", []) if x.get("step_ms")]
        out.append((r["riders"], r["dT_ms"],
                    min(reps) if reps else r["dT_ms"],
                    max(reps) if reps else r["dT_ms"],
                    (r.get("link_ms") or 0.0) / t0, r.get("link_ms") or 0.0))
    return out


def lsq(pts):
    n = len(pts)
    sx = sum(x for x, _ in pts); sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts); sxy = sum(x * y for x, y in pts)
    c = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - c * sx) / n, c


def write(name, header, rows):
    with open(os.path.join(HERE, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


FITS = []   # (arm, rel, F, c, npts, below_knee_only)


def emit_sweep(name, rel, below_knee=True):
    s = sweep(rel)
    write(name + ".csv", ["r", "dT", "lo", "hi", "load", "link"],
          [("%d" % r, "%.4f" % y, "%.4f" % lo, "%.4f" % hi, "%.4f" % ld,
            "%.4f" % lk) for r, y, lo, hi, ld, lk in s])
    pts = [(r, y) for r, y, _, _, ld, _ in s if (ld <= MAX_LOAD or not below_knee)]
    F, c = lsq(pts)
    FITS.append((name, rel, F, c, len(pts), below_knee))
    return F, c


def main():
    # (a) overlap: same job 2181632, ctx 512, 16 residents, L=64. The sweep runs
    # to the knee (load 1.0), so these are NOT below-knee coefficients; the
    # graph shows the whole sweep against the link time it would expose.
    emit_sweep("abl_overlap_on", "13_t6/T6_nofix.json", below_knee=False)
    emit_sweep("abl_overlap_off", "12_asyncbug/ASYNC_off.json", below_knee=False)
    # (b) bytes removed: ctx 128, L=64, every point below the knee.
    emit_sweep("abl_real_a", "15_clean_cf/CF_c128_a.json")
    emit_sweep("abl_real_b", "15_clean_cf/CF_c128_b.json")
    emit_sweep("abl_minbytes", "15_clean_cf/MB_c128.json")
    # (c) deferred write-back: same job 2181935 (M14), ctx 128, L=64.
    emit_sweep("abl_wb_deferred", "16_f_floor/FF_base_L64.json")
    emit_sweep("abl_wb_blocking", "16_f_floor/FF_syncd2h_L64.json")

    # (d) the second attention call (M19, one job 2182025, ctx 128, L=64). The
    # two ablation arms skip work attention needs, so their tokens are wrong by
    # construction (numerically_valid false): they are timing arms only.
    fb = []
    for tag, lab in (("FB_base_a", "full"), ("FB_base_b", "full"),
                     ("FB_base_c", "full"), ("FB_nokernel", "nokernel"),
                     ("FB_noattend", "noattend")):
        F, c = emit_sweep("abl_" + tag, "21_fusion_bound/%s.json" % tag)
        os.remove(os.path.join(HERE, "abl_%s.csv" % tag))   # fit only
        fb.append((tag, lab, "%.4f" % F, "%.4f" % c))
    write("abl_fusion.csv", ["arm", "mode", "F", "c"], fb)

    # c_host vs context, bytes removed (M21, one job 2182796, r = 1..6).
    rows = []
    for ctx in (128, 512, 2048, 4096):
        cs = []
        for p in sorted(glob.glob(os.path.join(DATA, "23_c_nobus_ladder",
                                               "NB_c%d_*.json" % ctx))):
            d = json.load(open(p))
            by = {r["riders"]: r["dT_ms"] for r in d["rows"]
                  if r.get("usable") and r["riders"] and r.get("dT_ms") is not None}
            pts = [(r, by[r]) for r in range(1, 7) if r in by]
            if len(pts) >= 3:
                cs.append(lsq(pts)[1])
        m = statistics.mean(cs)
        sd = statistics.stdev(cs) if len(cs) > 1 else 0.0
        # the per-block table the step-invariant addressing replaced, priced at
        # its measured coefficient on this arm's blocks per rider (ctx + 48
        # generated tokens, block size 16), added to the ctx-128 measurement
        blocks = (ctx + 48 + 15) // 16
        rows.append((ctx, "%.4f" % m, "%.4f" % sd, len(cs), blocks))
    base_m, base_blocks = float(rows[0][1]), rows[0][4]
    write("abl_chost_ctx.csv", ["ctx", "c", "sd", "n", "blocks", "perblock"],
          [r + ("%.4f" % (base_m + PRE_T1_MS_PER_BLOCK * (r[4] - base_blocks)),)
           for r in rows])

    # The block-table build itself: Class A host timer, ctx 512 (M8, T1 arms).
    out = []
    for tag in ("T1_c512_bs16", "T1_c512_bs64"):
        d = rec("08_opt/%s.json" % tag)
        if d.get("n_rows_usable", 0) < d.get("n_rows", 0):
            continue                      # purity gate rejected a row: discard
        groups = {}
        for line in open(os.path.join(RUNS, tag + "_phases.jsonl")):
            try:
                x = json.loads(line)
            except ValueError:
                continue
            ph = x.get("phases", {}).get("build_step_bt_python")
            if not ph:
                continue
            c = x.get("ctx") or {}
            groups.setdefault(c.get("n_active"), []).append(
                (x.get("step") or 0, ph["host_ms"], c.get("w_blocks")))
        for n, recs in sorted(groups.items()):
            recs.sort()
            recs = recs[3:]
            if len(recs) < 5:
                continue
            ms = statistics.median(r[1] for r in recs)
            w = statistics.median(r[2] for r in recs if r[2])
            out.append((tag, n, int(w), "%.5f" % ms,
                         "%.5f" % (PRE_T1_MS_PER_BLOCK * w)))
    write("abl_table_build.csv", ["arm", "riders", "blocks", "ms", "pre_ms"], out)

    with open(os.path.join(HERE, "fits.txt"), "w") as fh:
        fh.write("arm  source  F_ms  c_ms_per_rider  npts  below_knee_only\n")
        for f in FITS:
            fh.write("%s  %s  %.3f  %.3f  %d  %s\n" % f)
    print(open(os.path.join(HERE, "fits.txt")).read())
    for n in sorted(os.listdir(HERE)):
        if n.endswith(".csv"):
            print("wrote", n)


if __name__ == "__main__":
    main()
