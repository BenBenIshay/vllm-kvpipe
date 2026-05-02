# SPDX-License-Identifier: Apache-2.0
"""Gold-standard KV staging integration stub."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))
from kv_sizing import numel_from_kv_block_one_layer  # noqa: E402

__all__ = [
    "numel_from_kv_block_one_layer",
    "tensor_replay_stress",
    "gold_path_not_implemented",
]


def tensor_replay_stress(
    numel: int,
    device: torch.device,
    stream: torch.cuda.Stream,
    seconds: float,
) -> None:
    t0 = time.time()
    a = torch.randn(numel, device=device, dtype=torch.float16)
    b = torch.randn(numel, device=device, dtype=torch.float16) * 0.01
    with torch.cuda.stream(stream):
        while time.time() - t0 < seconds:
            torch.add(a, b, out=a)


def gold_path_not_implemented(*_a: Any, **_k: Any) -> None:
    raise NotImplementedError(
        "Wire real CPU→GPU KV copy into vLLM; this module is a stub."
    )
