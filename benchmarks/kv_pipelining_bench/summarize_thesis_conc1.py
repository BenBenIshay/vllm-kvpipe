#!/usr/bin/env python3
"""Summarize prove_conc1_thesis.sh (R0 / R1 / R3 orchestration proof)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, median, pstdev

CONFIGS = [
    ("R0_baseline", "baseline", "R0"),
    ("R1_pipelined", "pipelined", "R1"),
    ("R3_fake_transfer", "pipelined", "R3"),
]


def _steady_from_block(block: dict) -> tuple[float | None, float | None]:
    lat = block.get("latency") or {}
    iso = lat.get("mean_itl_steady_s_mean")
    thr = None
    for row in block.get("throughput") or []:
        if row.get("concurrency") == 1:
            thr = row.get("concurrent_mean_itl_steady_s_mean")
            break
    iso_ms = float(iso) * 1000.0 if iso is not None else None
    thr_ms = float(thr) * 1000.0 if thr is not None else None
    return iso_ms, thr_ms


def _load_block(run_dir: Path, mode: str) -> dict | None:
    p = run_dir / "summary_all.json"
    if p.is_file():
        all_summary = json.loads(p.read_text())
        block = all_summary.get(mode)
        if block and block.get("latency") and not block.get("error"):
            return block
    for fname in (f"results_{mode}.json", "results_baseline.json"):
        rp = run_dir / fname
        if rp.is_file():
            data = json.loads(rp.read_text())
            if data.get("latency") and not data.get("error"):
                return data
    return None


def _trial_dirs(root: Path, config_name: str) -> list[Path]:
    base = root / config_name
    if not base.is_dir():
        return []
    trials = sorted(base.glob("trial_*"))
    return trials if trials else [base]


def _collect(root: Path, config_name: str, mode: str) -> list[float]:
    values: list[float] = []
    for d in _trial_dirs(root, config_name):
        block = _load_block(d, mode)
        if block is None:
            continue
        iso_ms, _ = _steady_from_block(block)
        if iso_ms is not None:
            values.append(iso_ms)
    return values


def _per_trial_iso(root: Path, trial_num: int) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for dirname, mode, short in CONFIGS:
        d = root / dirname / f"trial_{trial_num}"
        if not d.is_dir():
            out[short] = None
            continue
        block = _load_block(d, mode)
        if block is None:
            out[short] = None
            continue
        iso_ms, _ = _steady_from_block(block)
        out[short] = iso_ms
    return out


def _outliers(vals: list[float], threshold_sigma: float = 2.0) -> list[int]:
    if len(vals) < 3:
        return []
    m = mean(vals)
    sd = pstdev(vals)
    if sd < 1e-6:
        return []
    return [i for i, v in enumerate(vals) if abs(v - m) > threshold_sigma * sd]


def _stats_line(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    med = median(vals)
    if len(vals) == 1:
        return f"{vals[0]:.2f} ms (n=1)"
    sd = pstdev(vals)
    return f"mean={mean(vals):.2f} med={med:.2f} ±{sd:.2f} ms (n={len(vals)})"


def _parse_profile_json(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for log_path in (run_dir / "run.log", run_dir / "server_pipelined.log"):
        if not log_path.is_file():
            continue
        for line in log_path.read_text(errors="replace").splitlines():
            if "KVPIPE_JSON" not in line:
                continue
            idx = line.find("{")
            if idx < 0:
                continue
            try:
                rows.append(json.loads(line[idx:]))
            except json.JSONDecodeError:
                continue
    return rows


def _profile_medians(rows: list[dict]) -> dict[str, float]:
    keys = (
        "gpu_stall_sum_ms",
        "start_load_host_ms",
        "save_host_ms",
        "save_prefetch_ms",
        "wait_host_ms",
    )
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r]
        if vals:
            out[k] = median(vals)
    return out


def _write_results_md(
    root: Path,
    itl: dict[str, list[float]],
    paired: list[dict],
    profile: dict[str, dict[str, float]],
) -> None:
    r0 = itl.get("R0_baseline") or []
    r1 = itl.get("R1_pipelined") or []
    r3 = itl.get("R3_fake_transfer") or []
    med0 = median(r0) if r0 else None
    med1 = median(r1) if r1 else None
    med3 = median(r3) if r3 else None

    lines = [
        "# Orchestration-only thesis results (conc=1)",
        "",
        f"Results: `{root}`",
        "",
        "## Claim",
        "",
        "Layer-wise KV movement via the vLLM connector stack adds decode ITL overhead",
        "from CPU/GPU synchronization and orchestration, not primarily from DDR↔HBM bytes.",
        "",
        "## Configs",
        "",
        "| ID | Config |",
        "|----|--------|",
        "| R0 | `DecodeBenchConnector` — GPU KV, no-op hooks |",
        "| R1 | `PipelinedDecodeBenchConnector` — B1+B2 production |",
        "| R3 | R1 + `VLLM_KVPIPE_FAKE_TRANSFER=1` — same CUDA DAG, zero memcpy |",
        "",
        "## Isolated steady ITL (ms)",
        "",
        "| Config | Stats |",
        "|--------|-------|",
    ]
    for dirname, _, short in CONFIGS:
        lines.append(f"| {short} | {_stats_line(itl.get(dirname) or [])} |")

    lines.extend(["", "## Per-trial matrix (isolated steady ms)", ""])
    lines.append("| trial | R0 | R1 | R3 | R1−R3 | R1−R0 | R3−R0 |")
    lines.append("|-------|-----|-----|-----|-------|-------|-------|")
    for row in paired:
        t = row["trial"]
        v0, v1, v3 = row.get("R0"), row.get("R1"), row.get("R3")

        def _c(v: float | None) -> str:
            return f"{v:.2f}" if v is not None else "—"

        d13 = (v1 - v3) if v1 is not None and v3 is not None else None
        d10 = (v1 - v0) if v1 is not None and v0 is not None else None
        d30 = (v3 - v0) if v3 is not None and v0 is not None else None
        lines.append(
            f"| {t} | {_c(v0)} | {_c(v1)} | {_c(v3)} | "
            f"{_c(d13)} | {_c(d10)} | {_c(d30)} |"
        )

    if med0 is not None and med1 is not None and med3 is not None:
        gap = med1 - med0
        lines.extend(
            [
                "",
                "## Decomposition (medians)",
                "",
                f"- **R1 − R0** = {med1 - med0:.2f} ms (total pipelined gap)",
                f"- **R3 − R0** = {med3 - med0:.2f} ms "
                f"({100 * (med3 - med0) / gap:.0f}% of R1−R0 if gap>0)",
                f"- **R1 − R3** = {med1 - med3:.2f} ms (real memcpy; expect small if overlapped)",
                "",
                "**Interpretation:** If R3 ≈ R1, orchestration/sync dominates. "
                "If R1 − R3 is large, bytes matter on the critical path.",
            ]
        )

    if profile:
        lines.extend(["", "## Profiler (median ms per forward)", ""])
        for name, med in profile.items():
            lines.append(f"### {name}")
            lines.append(f"```json\n{json.dumps(med, indent=2)}\n```")

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "TRIALS=5 GPU_UTIL=0.60 COOLDOWN_S=30 bash "
            "benchmarks/kv_pipelining_bench/prove_conc1_thesis.sh /tmp/thesis_orchestration_proof",
            "python benchmarks/kv_pipelining_bench/summarize_thesis_conc1.py /tmp/thesis_orchestration_proof",
            "```",
        ]
    )
    (root / "THESIS_RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/thesis_orchestration_proof")
    itl: dict[str, list[float]] = {}

    print(f"Results root: {root}\n")
    for dirname, mode, short in CONFIGS:
        vals = _collect(root, dirname, mode)
        itl[dirname] = vals
        oi = _outliers(vals)
        flag = f"  outliers trial idx {oi}" if oi else ""
        print(f"{short} ({dirname}): {_stats_line(vals)}{flag}")

    # Per-trial paired table
    trial_nums: set[int] = set()
    for dirname, _, _ in CONFIGS:
        for d in _trial_dirs(root, dirname):
            if d.name.startswith("trial_"):
                trial_nums.add(int(d.name.split("_")[1]))
    paired: list[dict] = []
    print("\n--- Per-trial isolated steady ITL (ms) ---")
    print(f"{'trial':>5}  {'R0':>8}  {'R1':>8}  {'R3':>8}  {'R1-R3':>8}  {'R1-R0':>8}")
    for t in sorted(trial_nums):
        row = _per_trial_iso(root, t)
        row["trial"] = t
        paired.append(row)
        v0, v1, v3 = row.get("R0"), row.get("R1"), row.get("R3")
        s0 = f"{v0:8.2f}" if v0 is not None else "       —"
        s1 = f"{v1:8.2f}" if v1 is not None else "       —"
        s3 = f"{v3:8.2f}" if v3 is not None else "       —"
        d13 = (v1 - v3) if v1 is not None and v3 is not None else None
        d10 = (v1 - v0) if v1 is not None and v0 is not None else None
        s13 = f"{d13:8.2f}" if d13 is not None else "       —"
        s10 = f"{d10:8.2f}" if d10 is not None else "       —"
        print(f"{t:5d}  {s0}  {s1}  {s3}  {s13}  {s10}")

    r0 = itl.get("R0_baseline") or []
    r1 = itl.get("R1_pipelined") or []
    r3 = itl.get("R3_fake_transfer") or []
    if r0 and r1 and r3:
        med0, med1, med3 = median(r0), median(r1), median(r3)
        gap = med1 - med0
        print("\n--- Decomposition (medians) ---")
        print(f"R1 - R0: {med1 - med0:.2f} ms ({100 * (med1 - med0) / med0:.1f}% vs baseline)")
        if gap > 0:
            print(
                f"R3 - R0: {med3 - med0:.2f} ms "
                f"({100 * (med3 - med0) / gap:.0f}% of R1-R0) — orchestration floor"
            )
            print(
                f"R1 - R3: {med1 - med3:.2f} ms "
                f"({100 * (med1 - med3) / gap:.0f}% of R1-R0) — real memcpy"
            )

    profile: dict[str, dict[str, float]] = {}
    for prof_dir, label in (
        ("profile_R1", "R1_profile"),
        ("profile_R3", "R3_profile"),
    ):
        base = root / prof_dir / "trial_1"
        if not base.is_dir():
            base = root / prof_dir
        rows = _parse_profile_json(base)
        med = _profile_medians(rows[-32:] if len(rows) > 32 else rows)
        if med:
            profile[label] = med
            print(f"\n{label} profiler: {med}")

    _write_results_md(root, itl, paired, profile)
    print(f"\nWrote {root / 'THESIS_RESULTS.md'}")

    # Simple ASCII bar chart (medians)
    if r0 and r1 and r3:
        med0, med1, med3 = median(r0), median(r1), median(r3)
        mx = max(med0, med1, med3)
        print("\n--- Median ITL bar (ASCII) ---")
        for label, v in [("R0", med0), ("R1", med1), ("R3", med3)]:
            bar = "#" * max(1, int(40 * v / mx))
            print(f"{label} {v:5.2f} ms |{bar}|")


if __name__ == "__main__":
    main()
