# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hugging Face auth for kv_pipelining bench scripts.

If ``HF_TOKEN`` is not already set, load it from a local file so gated models
(e.g. Llama) work without exporting secrets in the shell.

Default path: ``~/vllm/hf_token`` (one line, the token only).

Override the path with env ``VLLM_HF_TOKEN_FILE``.
"""
from __future__ import annotations

import os
from pathlib import Path


def hf_token_file_path() -> Path:
    raw = os.environ.get("VLLM_HF_TOKEN_FILE")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / "vllm" / "hf_token").resolve()


def apply_hf_token_from_user_file(env: dict[str, str] | None = None) -> bool:
    """If ``HF_TOKEN`` is missing, set it from the token file when present.

    Mutates ``env`` if passed (e.g. sweep subprocess env); otherwise ``os.environ``.

    Returns True if a token was applied from the file.
    """
    target = env if env is not None else os.environ
    if target.get("HF_TOKEN"):
        return False
    path = hf_token_file_path()
    if not path.is_file():
        return False
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    if not token:
        return False
    target["HF_TOKEN"] = token
    return True
