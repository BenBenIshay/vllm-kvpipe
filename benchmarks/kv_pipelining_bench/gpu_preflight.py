#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU memory preflight for KV pipelining benchmarks.

Refuses to start a run when the selected device does not have enough free
memory (default: >= 85% of total VRAM unused by other processes).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    used_mib: int
    command: str


@dataclass(frozen=True)
class GpuMemorySnapshot:
    device_index: int
    total_mib: int
    used_mib: int
    free_mib: int
    processes: tuple[GpuProcess, ...]

    @property
    def free_fraction(self) -> float:
        if self.total_mib <= 0:
            return 0.0
        return self.free_mib / self.total_mib


def _visible_device_index() -> int:
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not env:
        return 0
    first = env.split(",")[0].strip()
    if first.isdigit():
        return int(first)
    return 0


def _run_nvidia_smi(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi not found; cannot check GPU memory") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"nvidia-smi failed: {exc.stderr.strip() or exc}"
        ) from exc
    return proc.stdout.strip()


def get_gpu_memory_snapshot(device_index: int | None = None) -> GpuMemorySnapshot:
    """Return memory stats for one GPU (default: first visible device)."""
    if device_index is None:
        device_index = _visible_device_index()

    line = _run_nvidia_smi(
        [
            f"--id={device_index}",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        raise RuntimeError(f"unexpected nvidia-smi gpu line: {line!r}")
    total_mib, used_mib, free_mib = (int(x) for x in parts)

    procs: list[GpuProcess] = []
    apps_out = _run_nvidia_smi(
        [
            f"--id={device_index}",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if apps_out:
        for row in apps_out.splitlines():
            row = row.strip()
            if not row:
                continue
            pid_s, mem_s = [x.strip() for x in row.split(",")]
            pid = int(pid_s)
            used = int(mem_s)
            try:
                ps = subprocess.run(
                    ["ps", "-o", "user=,cmd=", "-p", str(pid)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                command = ps.stdout.strip() or "?"
            except subprocess.CalledProcessError:
                command = "?"
            procs.append(GpuProcess(pid=pid, used_mib=used, command=command))

    return GpuMemorySnapshot(
        device_index=device_index,
        total_mib=total_mib,
        used_mib=used_mib,
        free_mib=free_mib,
        processes=tuple(procs),
    )


def format_snapshot(snap: GpuMemorySnapshot) -> str:
    lines = [
        f"GPU {snap.device_index}: "
        f"free {snap.free_mib} / {snap.total_mib} MiB "
        f"({snap.free_fraction * 100:.1f}% free, "
        f"{snap.used_mib} MiB used)",
    ]
    if snap.processes:
        lines.append("  Compute processes on device:")
        for p in snap.processes:
            lines.append(f"    pid={p.pid}  {p.used_mib} MiB  {p.command[:120]}")
    else:
        lines.append("  No compute processes on device.")
    return "\n".join(lines)


def check_gpu_free_fraction(
    min_free_fraction: float = 0.85,
    device_index: int | None = None,
    gpu_memory_utilization: float | None = None,
) -> GpuMemorySnapshot:
    """Raise SystemExit(2) if free VRAM fraction is below *min_free_fraction*.

    If *gpu_memory_utilization* is set (e.g. 0.85 for vLLM), also require that
  free fraction is at least that value so the server can honor --gpu-memory-utilization.
    """
    if not 0.0 < min_free_fraction <= 1.0:
        raise ValueError("min_free_fraction must be in (0, 1]")

    snap = get_gpu_memory_snapshot(device_index)
    required = min_free_fraction
    if gpu_memory_utilization is not None:
        required = max(required, gpu_memory_utilization)

    if snap.free_fraction < required:
        msg = (
            f"GPU preflight failed: need >={required * 100:.0f}% free VRAM, "
            f"have {snap.free_fraction * 100:.1f}% free.\n"
            f"{format_snapshot(snap)}\n"
            "Stop other GPU jobs or lower --gpu-memory-utilization."
        )
        raise SystemExit(msg)

    return snap


def main() -> None:
    p = argparse.ArgumentParser(description="Check GPU is free enough for benchmarks")
    p.add_argument(
        "--min-free-fraction",
        type=float,
        default=0.85,
        help="Minimum fraction of total VRAM that must be free (default: 0.85)",
    )
    p.add_argument("--gpu-index", type=int, default=None)
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Also require free fraction >= this vLLM utilization target",
    )
    args = p.parse_args()

    snap = check_gpu_free_fraction(
        min_free_fraction=args.min_free_fraction,
        device_index=args.gpu_index,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print("GPU preflight OK")
    print(format_snapshot(snap))


if __name__ == "__main__":
    main()
