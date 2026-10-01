#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fit the host-path overhead law from the rider-cost sweeps.

WHAT IS BEING REPLACED. The model charges the host path a single constant
scaled by layer depth:

    # paper/model/killer_v2.py:33-38
    F_REF_MS, F_REF_LAYERS = 12.3, 64
    def f_fixed_ms(m):
        return F_REF_MS * m.L / F_REF_LAYERS

There is no rider-count, byte, block, context or k-window term anywhere in it,
and the only rider-dependent charge (`delta_ms`) is exactly proportional with a
zero intercept -- so the overhead is affine in riders BY CONSTRUCTION, while
section 6.5 of the paper reports that the measured cost is not affine and that
the fixed/slope split "is not a meaningful decomposition".

THE CANDIDATE LAW. dT(r) = step(r riders) - step(0 riders), in ms:

    A * L * 1[r>0]      switch-on: the ~11*L per-layer ops that appear the
                        moment ANY rider exists. Link-INVARIANT.
    B * L * r           per-rider per-layer launch: the 2*L*R memcpys.
                        Link-INVARIANT in count -- and it grows with exactly
                        what a faster link lets you admit more of.
    C * link_ms         the transfer itself, link_ms = nominal_bytes / X.
                        The ONLY link-bound term.
    D * blocks          per-block host work, blocks = r * ceil(S/bs).
                        Link-invariant; largely a removable artifact.
    E * L * r^2         a superlinear term, included as a CANDIDATE only. If
                        the data does not support it, it must not be fitted:
                        forcing a shape on three points is the mistake that
                        produced F = -23.4 ms from curve_a100_final.json.

Note dT(0) = 0 by construction, so there is no free intercept: every term
carries a factor that vanishes at r = 0. A model that needs a constant offset
is telling you the baseline row is wrong, not that the host path has a
rider-free cost.

WHY MODEL SELECTION AND NOT JUST A FIT. With five candidate terms on correlated
axes, a full fit will always "work" and will hand back coefficients whose signs
mean nothing. So every subset is fitted, ranked by BIC, and the selected model
is the one the data supports -- reported alongside the full model so the
difference is visible. Coefficients come with bootstrap CIs, and a held-out
split reports prediction error on points the fit never saw.

CROSS-CHECK THAT DOES NOT GO THROUGH THE FIT. The xfer-mode arms measure the
same split directly:

    real - minbytes   = the byte cost      (compare with C * link_ms)
    minbytes - nobus  = the launch cost    (compare with B * L * r)
    nobus             = everything else    (compare with A * L + D * blocks)

If the fitted coefficients and the arm differences disagree, the arms win: they
are three measurements, not one regression.

Usage:
    python overhead_fit.py runs/*.json --link-gbps 25.1
    python overhead_fit.py --selftest
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import sys

import numpy as np

# Candidate terms: name -> (builder, human-readable meaning, link behaviour)
TERMS = {
    "A_switch_per_layer": (
        lambda p: p["layers"] * (1.0 if p["r"] > 0 else 0.0),
        "ms per layer, charged once any rider exists", "invariant"),
    "B_per_rider_layer": (
        lambda p: p["layers"] * p["r"],
        "ms per rider per layer (the 2*L*R launches)", "invariant"),
    "C_link": (
        lambda p: p["link_ms"],
        "fraction of the nominal link time that lands on the step", "bound"),
    "D_per_block": (
        lambda p: p["blocks"],
        "ms per rider block (host bookkeeping)", "invariant"),
    "E_superlinear": (
        lambda p: p["layers"] * p["r"] * p["r"],
        "ms per layer per rider^2 (queueing)", "unknown"),
}


def load_points(paths: list[str], link_gbps: float | None,
                require_valid: bool = True,
                model_filter: str = "") -> list[dict]:
    """Flatten sweep JSONs into fittable points, dropping what is not usable."""
    pts: list[dict] = []
    dropped: dict[str, int] = {}

    def drop(why: str) -> None:
        dropped[why] = dropped.get(why, 0) + 1

    for path in paths:
        with open(path) as fh:
            d = json.load(fh)

        # ONE MODEL AT A TIME, unless explicitly pooled. `A` -- the per-layer
        # switch-on cost -- is NOT model-independent: measured 0.419 ms/layer on
        # Qwen-0.5B, 0.246-0.263 on Llama-8B, 0.183 on Qwen-32B, because it is
        # host work that partly hides behind compute and the models differ 2x in
        # compute per layer. Pooling them forces one compromise `A` and the
        # other terms absorb the error -- it produced a NEGATIVE `C_link`, i.e.
        # "more bytes makes the step faster", which is the model telling you it
        # is misspecified rather than a noisy estimate.
        if model_filter and model_filter.lower() not in str(
                d.get("model", "")).lower():
            drop(f"model does not match filter {model_filter!r}")
            continue

        # GROUND RULE 1: only the P/D-emulated decode-node rig is a measurement
        # of this feature. A co-located run cannot even create more than two
        # riders, and it understates dT at r=2 by ~7.5%.
        if not d.get("connector", False):
            drop("co-located rig (no --connector)")
            continue
        if require_valid and not d.get("numerically_valid", True):
            # Not a defect -- the xfer-mode and layer-limit arms are meant to be
            # here; they just belong in the arm comparison, not the law's fit.
            drop("numerically-invalid arm (xfer_mode/layer knob)")
            continue

        rows = d.get("rows", [])
        base = next((x for x in rows if x.get("riders") == 0), None)
        if base is None:
            drop("no r=0 baseline row")
            continue
        if not base.get("usable", True):
            drop("baseline row failed the purity gate")
            continue

        X = (link_gbps if link_gbps else d.get("link_gbps")) or 0.0
        S = int(d.get("ctx", 0))
        bs = int(d.get("block_size", 16) or 16)
        layers = int(d.get("layers", 0) or _layers_of(d.get("model", "")))
        if not layers:
            drop("layer count unknown for model")
            continue
        # The layer knob makes the EFFECTIVE depth the swept quantity.
        eff_layers = int(d.get("rungd_layers") or 0) or layers

        for row in rows:
            r = int(row.get("riders", 0))
            if r == 0:
                continue
            if not row.get("usable", True):
                drop("row failed the purity gate")
                continue
            nominal = float(row.get("nominal_gb_per_step",
                                    row.get("stream_gb_per_step", 0.0)) or 0.0)
            if nominal <= 0:
                drop("no rider traffic recorded (rider axis not real)")
                continue

            # KNEE FILTER. The law dT = F + c*r describes the regime where a
            # rider's transfer still hides inside the step. Above the knee
            # r* = T0*X/u the step IS the transfer and every extra rider costs
            # its full link time, which is a different regime with genuine
            # curvature -- measured: the marginal rider goes 8.67 -> 23.40 ms
            # across the knee at ctx 2048.
            #
            # Feeding both regimes to one fit is how M1's automated run selected
            # `A + C_link + E_superlinear` over the affine law: a single
            # above-knee point (r=16 at ctx 256, where r* = 13.9) supplied
            # curvature and the r^2 term absorbed it. Held-out error was 4.2%,
            # so the accuracy gate PASSED on a structurally wrong model -- the
            # gate cannot catch this, only the filter can.
            t0_ms = float(base["step_ms"])
            u_bytes = (nominal / r) * 1e9 if r else 0.0
            knee = (t0_ms / 1e3) * (X * 1e9) / u_bytes if (X and u_bytes) else 0.0
            if knee and r >= knee:
                drop(f"above the knee (r={r} >= r*={knee:.1f}): transfer-bound, "
                     f"a different regime")
                continue
            pts.append({
                "src": os.path.basename(path),
                # Job identity, for the per-job offset (see fit_with_jobs).
                # Runs before 2026-10-01 do not carry slurm_job_id, so fall
                # back to clustering by file mtime: arms of one job are written
                # minutes apart, jobs are separated by much more.
                "job": str(d.get("slurm_job_id") or "")
                or ("mtime%d" % int(os.path.getmtime(path) // 1800)),
                "r": r, "S": S, "bs": bs,
                "layers": eff_layers, "layers_model": layers,
                "b_res": int(d.get("batch", 0)),
                "xfer_mode": d.get("xfer_mode", "real"),
                "dT": float(row["step_ms"]) - float(base["step_ms"]),
                "link_ms": nominal * 1e9 / (X * 1e9) * 1e3 if X else 0.0,
                "blocks": r * math.ceil(S / bs) if S and bs else 0.0,
                "launches": float(row.get("h2d_launches_per_step", 0.0) or 0.0),
            })

    if dropped:
        print("dropped:")
        for k, v in sorted(dropped.items(), key=lambda kv: -kv[1]):
            print(f"  {v:4d}  {k}")
    return pts


def _layers_of(model: str) -> int:
    m = model.lower()
    if "32b" in m:
        return 64
    if "8b" in m:
        return 32
    if "0.5b" in m:
        return 24
    if "70b" in m:
        return 80
    return 0


def design(pts: list[dict], names: list[str]) -> np.ndarray:
    return np.array([[TERMS[n][0](p) for n in names] for p in pts], float)


def fit(pts: list[dict], names: list[str]) -> dict:
    A = design(pts, names)
    y = np.array([p["dT"] for p in pts], float)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    n, k = len(y), len(names)
    rss = float(resid @ resid)
    # BIC on a Gaussian likelihood. Guard n <= k, where RSS is meaninglessly 0.
    if n > k and rss > 0:
        bic = n * math.log(rss / n) + k * math.log(n)
    else:
        bic = float("inf")
    tss = float(((y - y.mean()) @ (y - y.mean())))
    return {
        "names": names, "coef": coef, "rss": rss, "bic": bic, "n": n,
        "r2": (1 - rss / tss) if tss > 0 else float("nan"),
        "rmse": math.sqrt(rss / n) if n else float("nan"),
        "max_abs_resid": float(np.max(np.abs(resid))) if n else float("nan"),
    }


def fit_with_jobs(pts: list[dict], names: list[str]) -> dict:
    """Fit the law with a PER-JOB additive offset alongside the coefficients.

    WHY. `F` carries a between-job offset of 1.78 ms (14%), bigger than its
    within-job spread of 0.58 ms (5%): one job read ~1.2 ms high on every arm it
    contained. A pooled fit cannot see which job a point came from, so that
    variance ends up in the residual -- the 66-point fit's held-out max error was
    11.0%, and 11% of a 15 ms `dT` is ~1.5 ms, essentially the between-job
    spread. The law was not wrong; the fit was absorbing an unmodelled nuisance.

    Adding one indicator per job (the first dropped for identifiability) lets the
    offsets take it, so the reported coefficients are within-job quantities. The
    offsets are reported, not hidden: their spread IS the reproducibility of the
    apparatus and belongs in the projection's envelope.
    """
    jobs = sorted({p["job"] for p in pts})
    if len(jobs) < 2:
        f = fit(pts, names)
        f["job_offsets"] = {}
        return f
    extra = jobs[1:]                      # drop the first as the reference
    A = np.hstack([
        design(pts, names),
        np.array([[1.0 if p["job"] == j else 0.0 for j in extra]
                  for p in pts], float),
    ])
    y = np.array([p["dT"] for p in pts], float)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    n, k = len(y), A.shape[1]
    rss = float(resid @ resid)
    tss = float((y - y.mean()) @ (y - y.mean()))
    return {
        "names": names,
        "coef": coef[:len(names)],
        "job_offsets": dict(zip(extra, coef[len(names):])),
        "job_ref": jobs[0],
        "rss": rss, "n": n, "k": k,
        "bic": (n * math.log(rss / n) + k * math.log(n)
                if n > k and rss > 0 else float("inf")),
        "r2": (1 - rss / tss) if tss > 0 else float("nan"),
        "rmse": math.sqrt(rss / n) if n else float("nan"),
        "max_abs_resid": float(np.max(np.abs(resid))) if n else float("nan"),
    }


def select(pts: list[dict], candidates: list[str]) -> list[dict]:
    """Every non-empty subset, ranked by BIC. Cheap: 2^k - 1 least squares."""
    out = []
    for size in range(1, len(candidates) + 1):
        for combo in itertools.combinations(candidates, size):
            f = fit(pts, list(combo))
            if math.isfinite(f["bic"]):
                out.append(f)
    return sorted(out, key=lambda f: f["bic"])


def bootstrap(pts: list[dict], names: list[str], iters: int = 2000,
              seed: int = 1234) -> dict:
    rng = np.random.default_rng(seed)
    n = len(pts)
    draws = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        sample = [pts[i] for i in idx]
        try:
            draws.append(fit(sample, names)["coef"])
        except np.linalg.LinAlgError:
            continue
    if not draws:
        return {}
    D = np.array(draws)
    return {nm: (float(np.percentile(D[:, j], 2.5)),
                 float(np.percentile(D[:, j], 97.5)))
            for j, nm in enumerate(names)}


def holdout(pts: list[dict], names: list[str], frac: float = 0.2,
            seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    n = len(pts)
    idx = rng.permutation(n)
    ntest = max(1, int(round(frac * n)))
    test = [pts[i] for i in idx[:ntest]]
    train = [pts[i] for i in idx[ntest:]]
    if len(train) <= len(names):
        return {"skipped": "not enough training points"}
    f = fit(train, names)
    A = design(test, names)
    y = np.array([p["dT"] for p in test], float)
    pred = A @ f["coef"]
    err = np.abs(pred - y)
    rel = err / np.maximum(np.abs(y), 1e-9)
    return {"n_train": len(train), "n_test": len(test),
            "mean_abs_err_ms": float(err.mean()),
            "max_abs_err_ms": float(err.max()),
            "mean_rel_err": float(rel.mean()),
            "max_rel_err": float(rel.max())}


def arm_check(pts_all: list[dict]) -> None:
    """The decomposition measured by the xfer arms, independent of any fit."""
    by: dict[tuple, dict] = {}
    for p in pts_all:
        key = (p["src"].split("_xfer")[0], p["r"], p["S"], p["bs"],
               p["layers"], p["b_res"])
        by.setdefault(key, {})[p["xfer_mode"]] = p["dT"]
    rows = [(k, v) for k, v in by.items() if len(v) >= 2]
    if not rows:
        print("\nno matched xfer-mode arms yet -- run --xfer-mode "
              "real/minbytes/nobus at identical settings to get the direct "
              "decomposition (it does not go through the fit)")
        return
    print("\nDIRECT DECOMPOSITION from the xfer arms (no regression):")
    print(f"{'r':>3} {'S':>6} {'L':>4} {'real':>8} {'minby':>8} {'nobus':>8} "
          f"{'bytes':>8} {'launch':>8} {'rest':>8}")
    for k, v in sorted(rows):
        real, mb, nb = v.get("real"), v.get("minbytes"), v.get("nobus")
        byte_cost = (real - mb) if (real is not None and mb is not None) else None
        lch = (mb - nb) if (mb is not None and nb is not None) else None
        def s(x):
            return f"{x:8.2f}" if x is not None else "       -"
        print(f"{k[1]:>3} {k[2]:>6} {k[4]:>4} {s(real)} {s(mb)} {s(nb)} "
              f"{s(byte_cost)} {s(lch)} {s(nb)}")


def report(pts: list[dict], pts_all: list[dict], top: int = 6) -> dict:
    cands = list(TERMS)
    print(f"\n{len(pts)} usable points "
          f"(r={sorted({p['r'] for p in pts})}, "
          f"S={sorted({p['S'] for p in pts})}, "
          f"L={sorted({p['layers'] for p in pts})}, "
          f"bs={sorted({p['bs'] for p in pts})})")

    if len(pts) <= 2:
        print("\nNOT ENOUGH POINTS TO FIT ANYTHING. This is the state the "
              "archived data is in: appendix H asks for ten rider counts or "
              "more and the usable data has one or two.")
        return {}

    ranked = select(pts, cands)
    print(f"\nMODEL SELECTION (BIC, best first; {len(ranked)} subsets fitted)")
    print(f"{'BIC':>9} {'RMSE':>7} {'R2':>7}  terms")
    for f in ranked[:top]:
        print(f"{f['bic']:>9.2f} {f['rmse']:>7.3f} {f['r2']:>7.4f}  "
              f"{'+'.join(f['names'])}")

    best = ranked[0]
    print(f"\nSELECTED: {' + '.join(best['names'])}")
    ci = bootstrap(pts, best["names"])
    print(f"{'term':>22} {'coef':>12} {'95% CI':>26}  link")
    for j, nm in enumerate(best["names"]):
        lo, hi = ci.get(nm, (float('nan'), float('nan')))
        spans0 = "  <-- CI SPANS ZERO" if lo <= 0 <= hi else ""
        print(f"{nm:>22} {best['coef'][j]:>12.5f} "
              f"[{lo:>11.5f},{hi:>11.5f}]  {TERMS[nm][2]}{spans0}")
    print("  meaning:")
    for nm in best["names"]:
        print(f"    {nm}: {TERMS[nm][1]}")

    ho = holdout(pts, best["names"])
    print(f"\nHELD OUT: {ho}")

    # The same model with a per-job offset. If RMSE drops sharply, the pooled
    # residual was job offset rather than model error -- which is the finding,
    # not a fix applied to flatter the numbers.
    wj = fit_with_jobs(pts, best["names"])
    if wj.get("job_offsets"):
        print(f"\nWITH PER-JOB OFFSETS ({len(wj['job_offsets']) + 1} jobs, "
              f"reference {wj['job_ref']}):")
        print(f"  RMSE {best['rmse']:.3f} -> {wj['rmse']:.3f} "
              f"({(1 - wj['rmse'] / best['rmse']) * 100:+.0f}%), "
              f"R2 {best['r2']:.4f} -> {wj['r2']:.4f}")
        print(f"{'term':>22} {'pooled':>12} {'within-job':>12} {'shift':>8}")
        for j, nm in enumerate(best["names"]):
            a, b = best["coef"][j], wj["coef"][j]
            print(f"{nm:>22} {a:>12.5f} {b:>12.5f} "
                  f"{(b - a) / a * 100 if a else float('nan'):>7.1f}%")
        offs = sorted(wj["job_offsets"].items(), key=lambda kv: kv[1])
        print("  job offsets on dT (ms), relative to the reference job:")
        for j, v in offs:
            print(f"    {j:>14} {v:+8.3f}")
        span = max([0.0] + [v for _, v in offs]) - min([0.0] + [v for _, v in offs])
        print(f"  spread {span:.2f} ms -- this IS the apparatus's "
              f"reproducibility, and belongs in the projection envelope")

    full = fit(pts, cands) if len(pts) > len(cands) else None
    if full is not None:
        print(f"\nFULL MODEL for comparison (all {len(cands)} terms): "
              f"RMSE {full['rmse']:.3f}, R2 {full['r2']:.4f}")
        for j, nm in enumerate(cands):
            print(f"{nm:>22} {full['coef'][j]:>12.5f}")

    arm_check(pts_all)
    return {"selected": best["names"],
            "coef": {n: float(c) for n, c in zip(best["names"], best["coef"])},
            "ci": ci, "holdout": ho, "n": len(pts)}


# ---------------------------------------------------------------- selftest
def _selftest() -> int:
    """Recover known coefficients from synthetic data, then show that the
    archived-data situation is unfittable."""
    print("SELFTEST 1: recover planted coefficients")
    A_true, B_true, C_true, D_true = 0.15, 0.010, 0.55, 0.004
    rng = np.random.default_rng(0)
    pts = []
    for L in (16, 32, 64):
        for S in (256, 512, 1024):
            for bs in (16, 32):
                for r in range(1, 13):
                    link_ms = r * S * 1.3e-4 * 1e3 / 25.1
                    blocks = r * math.ceil(S / bs)
                    dT = (A_true * L + B_true * L * r + C_true * link_ms
                          + D_true * blocks)
                    pts.append({
                        "src": "synth", "r": r, "S": S, "bs": bs,
                        "layers": L, "layers_model": L, "b_res": 16,
                        "xfer_mode": "real",
                        "dT": dT + rng.normal(0, 0.05),
                        "link_ms": link_ms, "blocks": blocks, "launches": 2 * L * r,
                    })
    ranked = select(pts, list(TERMS))
    best = ranked[0]
    got = dict(zip(best["names"], best["coef"]))
    print(f"  selected: {'+'.join(best['names'])}")
    ok = True
    for nm, truth in (("A_switch_per_layer", A_true),
                      ("B_per_rider_layer", B_true),
                      ("C_link", C_true), ("D_per_block", D_true)):
        g = got.get(nm)
        good = g is not None and abs(g - truth) < 0.1 * abs(truth) + 1e-4
        ok &= good
        print(f"  {nm:>22}: truth {truth:<10.5f} got "
              f"{'None' if g is None else f'{g:.5f}'}  "
              f"{'OK' if good else 'MISS'}")
    if "E_superlinear" in best["names"]:
        print("  NOTE: superlinear term selected on linear data -- "
              "over-selection")
        ok = False

    print("\nSELFTEST 2: a truly superlinear dataset must select E")
    pts2 = []
    for r in range(1, 17):
        link_ms = r * 512 * 1.3e-4 * 1e3 / 25.1
        dT = 0.15 * 64 + 0.01 * 64 * r + 0.002 * 64 * r * r + 0.55 * link_ms
        pts2.append({"src": "synth2", "r": r, "S": 512, "bs": 16, "layers": 64,
                     "layers_model": 64, "b_res": 16, "xfer_mode": "real",
                     "dT": dT + rng.normal(0, 0.05), "link_ms": link_ms,
                     "blocks": r * 32, "launches": 128 * r})
    b2 = select(pts2, list(TERMS))[0]
    has_e = "E_superlinear" in b2["names"]
    print(f"  selected: {'+'.join(b2['names'])}  "
          f"{'OK' if has_e else 'MISS (failed to see superlinearity)'}")
    ok &= has_e

    print("\nSELFTEST 3: the archived situation -- 2 usable points")
    r3 = report(pts[:2], pts[:2])
    ok &= (r3 == {})
    print(f"  refused to fit: {'OK' if r3 == {} else 'MISS'}")

    print(f"\nSELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", help="sweep JSONs (globs ok)")
    p.add_argument("--link-gbps", type=float, default=0.0,
                   help="override the link speed recorded in the JSONs")
    p.add_argument("--model-filter", default="",
                   help="substring of the model name; keep only matching runs. "
                        "Use it: `A` is model-specific (host work that hides "
                        "behind compute, and models differ in compute per "
                        "layer), so pooling models misspecifies the fit.")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--out", default="")
    a = p.parse_args()

    if a.selftest:
        return _selftest()

    paths: list[str] = []
    for x in a.paths:
        paths.extend(sorted(glob.glob(x)) or [x])
    if not paths:
        p.error("no input JSONs")

    pts_all = load_points(paths, a.link_gbps or None, require_valid=False,
                          model_filter=a.model_filter)
    pts = [q for q in pts_all if q["xfer_mode"] == "real"]
    res = report(pts, pts_all)
    if a.out and res:
        with open(a.out, "w") as fh:
            json.dump({k: (v if not isinstance(v, dict) else
                           {kk: list(vv) if isinstance(vv, tuple) else vv
                            for kk, vv in v.items()})
                       for k, v in res.items()}, fh, indent=2, default=float)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
