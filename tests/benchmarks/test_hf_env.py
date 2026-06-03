# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCH = _REPO_ROOT / "benchmarks/kv_pipelining_bench"


@pytest.fixture(scope="module")
def hf_env_mod():
    if str(_BENCH) not in sys.path:
        sys.path.insert(0, str(_BENCH))
    import hf_env  # noqa: PLC0415

    return hf_env


def test_apply_hf_token_from_file(tmp_path: Path, hf_env_mod, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("VLLM_HF_TOKEN_FILE", str(tmp_path / "tok"))
    p = tmp_path / "tok"
    p.write_text("secret-token\n", encoding="utf-8")
    env: dict[str, str] = {}
    assert hf_env_mod.apply_hf_token_from_user_file(env) is True
    assert env["HF_TOKEN"] == "secret-token"


def test_hf_token_env_wins(hf_env_mod, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("HF_TOKEN", "from-env")
    monkeypatch.setenv("VLLM_HF_TOKEN_FILE", str(tmp_path / "ignored"))
    (tmp_path / "ignored").write_text("file", encoding="utf-8")
    env = os.environ.copy()
    assert hf_env_mod.apply_hf_token_from_user_file(env) is False
    assert env["HF_TOKEN"] == "from-env"
