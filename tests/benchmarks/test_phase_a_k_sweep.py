# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Phase A K-sweep driver (no GPU required)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWEEP_PATH = _REPO_ROOT / "benchmarks/kv_pipelining_bench/phase_a_k_sweep.py"


def _load_phase_a():
    spec = importlib.util.spec_from_file_location("phase_a_k_sweep", _SWEEP_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Required so dataclasses can resolve cls.__module__ during import.
    sys.modules["phase_a_k_sweep"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def phase_a():
    return _load_phase_a()


def test_default_result_root_under_data_ben(phase_a):
    """HF cache + sweep outputs must default to /data/ben to avoid home quota."""
    assert str(phase_a.DEFAULT_RESULT_ROOT).startswith("/data/ben")


def test_effective_gpu_guard_frac_raises_with_concurrency(phase_a):
    """High max concurrency → higher parent HBM cap (vLLM KV pool)."""
    assert phase_a.effective_gpu_guard_frac(0.985, 1) == 0.985
    assert phase_a.effective_gpu_guard_frac(0.985, 16) >= 0.99
    assert phase_a.effective_gpu_guard_frac(0.985, 64) >= 0.995
    assert phase_a.effective_gpu_guard_frac(0.985, 256) >= 0.9995


def test_is_under_data_ben(phase_a):
    assert phase_a._is_under_data_ben(Path("/data/ben/kv_phase_a_internal"))
    assert phase_a._is_under_data_ben(Path("/data/ben"))
    assert not phase_a._is_under_data_ben(Path("/tmp/foo"))


def test_phase_a_sweep_dry_run_accepts_tmp_with_flag():
    """CLI must allow non-/data/ben roots when explicitly opted in (e.g. CI)."""
    cmd = [
        sys.executable,
        str(_SWEEP_PATH),
        "--wave",
        "A",
        "--dry-run",
        "--result-root",
        "/tmp/phase_a_sweep_test",
        "--allow-result-root-outside-data-ben",
    ]
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_phase_a_sweep_rejects_tmp_without_flag():
    cmd = [
        sys.executable,
        str(_SWEEP_PATH),
        "--wave",
        "A",
        "--dry-run",
        "--result-root",
        "/tmp/phase_a_sweep_test_forbidden",
    ]
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 2
    assert "result-root" in proc.stderr.lower() or "/data/ben" in proc.stderr
