#!/usr/bin/env python3
"""Regenerate paper/data/overhead/MANIFEST.json from the archived run records.

This lived in a scratch directory while generating COMMITTED provenance, which
is exactly backwards: the file that says where every number came from was itself
unversioned. Run from the worktree root:

    ./.venv/bin/python paper/model/make_manifest.py
"""
import datetime
import glob
import hashlib
import json
import os
root = "paper/data/overhead"
rows = []
for f in sorted(glob.glob(root + "/*/*")):
    if not f.endswith(".json"):
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not isinstance(d, dict):
        rows.append({"path": os.path.relpath(f, "paper"),
                     "bytes": os.path.getsize(f),
                     "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest()[:16],
                     "shape": "json list, not a run record"})
        continue
    pg = d.get("purity_gate")
    rows.append({
        "path": os.path.relpath(f, "paper"),
        "bytes": os.path.getsize(f),
        "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest()[:16],
        "model": d.get("model"), "ctx": d.get("ctx"), "batch": d.get("batch"),
        "block_size": d.get("block_size"), "k_window": d.get("k_window"),
        "xfer_mode": d.get("xfer_mode"), "skip_d2h": d.get("skip_d2h"),
        # Which write_token path the arm ran. Without these the archive cannot
        # tell a T2-on arm from a T2-off one, which is the whole point of the
        # bracketed comparison. Absent on arms predating the flag.
        "d2h_probe": d.get("d2h_probe"),
        "d2h_async": d.get("d2h_async"),
        "rungd_layers": d.get("rungd_layers"),
        "connector": d.get("connector"), "force_riders": d.get("force_riders"),
        "numerically_valid": d.get("numerically_valid"),
        "purity_gate": pg.get("verdict") if isinstance(pg, dict) else pg,
        "git_commit": d.get("git_commit"), "git_dirty": d.get("git_dirty"),
        "n_rows": d.get("n_rows"), "n_rows_usable": d.get("n_rows_usable"),
    })
man = {
    "generated": datetime.date.today().isoformat(),
    "campaign": "overhead characterisation M0-M6",
    "branch": "overhead-characterisation",
    "report": "paper/OVERHEAD_CHARACTERISATION.md",
    "note": ("Harness artifacts copied verbatim from "
             "/mnt/beegfs/benishay/paper_runs/20260930/overhead/, NOT "
             "transcriptions. The per-step *_phases.jsonl files (12 MB) are "
             "kept only in paper_runs: the phase timers were abandoned as a "
             "cost-attribution tool (they accounted for 214-304% of dT) and "
             "appear in the report only as a negative result."),
    "regenerate": [
        "paper/model/c_scaling.py  -- report sections 3.1 and 4",
        "paper/model/overhead_model.py -- sections 0, 3.3, 5, 7",
    ],
    "files": rows,
}
open(root + "/MANIFEST.json", "w").write(json.dumps(man, indent=2) + "\n")
print("manifest: %d json artifacts" % len(rows))
print("numerically_invalid (layer-subset timing-only arms): %d"
      % len([r for r in rows if r.get("numerically_valid") is False]))
print("recorded from a dirty tree: %d" % len([r for r in rows if r.get("git_dirty")]))
noconn = [r for r in rows if r.get("connector") is False]
print("ran WITHOUT the connector (not P/D-emulated): %d" % len(noconn))
for r in noconn[:8]:
    print("   ", r["path"])
