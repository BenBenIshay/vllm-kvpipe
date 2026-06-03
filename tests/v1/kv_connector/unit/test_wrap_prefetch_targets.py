# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.kv_pipelining_step import build_prefetch_target_indices


def test_prefetch_targets_open_chain():
    # L=24, K=4: last K layers have no target.
    targets = build_prefetch_target_indices(24, 4, wrap=False)
    assert targets[0] == 4
    assert targets[19] == 23
    assert targets[20] is None
    assert targets[23] is None


def test_prefetch_targets_wrap_around():
    targets = build_prefetch_target_indices(24, 4, wrap=True)
    assert targets == [(i + 4) % 24 for i in range(24)]
    assert targets[0] == 4
    assert targets[19] == 23
    assert targets[20] == 0
    assert targets[21] == 1
    assert targets[22] == 2
    assert targets[23] == 3


def test_prefetch_targets_k_zero():
    assert build_prefetch_target_indices(24, 0, wrap=True) == [None] * 24
