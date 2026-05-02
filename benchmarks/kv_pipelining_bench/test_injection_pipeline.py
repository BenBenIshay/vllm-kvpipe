"""Unit + optional GPU smoke tests for injection / KV sizing."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent
_WORKER = _ROOT / "injection_stress_worker.py"
_PY = os.environ.get("VLLM_TEST_PYTHON", "/home/ben/vllm/.venv/bin/python")


def test_kv_sizing_numel():
    sys.path.insert(0, str(_ROOT))
    from kv_sizing import align_numel, numel_from_kv_block_one_layer

    n = numel_from_kv_block_one_layer(16, 8, 128)
    assert n == 2 * 16 * 8 * 128
    assert align_numel(100) % 256 == 0


def test_kv_chunk_cli():
    r = subprocess.run(
        [_PY, str(_ROOT / "kv_chunk_bytes.py"), "--preset", "llama-3.1-8b", "--json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr
    assert "numel" in r.stdout


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="no CUDA")
def test_worker_h2d_async_smoke():
    r = subprocess.run(
        [_PY, str(_WORKER), "h2d-async", "--seconds", "0.4", "--target-gbps", "0", "--chunk-mib", "1", "--concurrent", "2"],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert "mode=h2d_async" in (r.stdout + r.stderr).lower()


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="no CUDA")
def test_worker_d2h_and_mix_smoke():
    for mode, extra in [
        ("d2h-async", ["--target-gbps", "0", "--chunk-mib", "1", "--concurrent", "1"]),
        ("h2d-d2h-mix", ["--h2d-gbps", "1", "--d2h-gbps", "1", "--chunk-mib", "1"]),
    ]:
        r = subprocess.run(
            [_PY, str(_WORKER), mode, "--seconds", "0.3"] + extra,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert r.returncode == 0, (mode, r.stderr, r.stdout)


def test_gold_path_stub():
    sys.path.insert(0, str(_ROOT))
    from kv_replay_gold_path import gold_path_not_implemented

    with pytest.raises(NotImplementedError):
        gold_path_not_implemented()
