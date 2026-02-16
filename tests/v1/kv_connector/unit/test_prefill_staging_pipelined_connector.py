# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for PrefillStagingPipelinedConnector.

Tests scheduler-side metadata building (prefill vs decode detection)
and the connector lifecycle for real-prefill + pipelined-decode mode.
"""

import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.prefill_staging_pipelined_connector import (
    PrefillStagingPipelinedConnector,
    PrefillStagingPipelinedConnectorMetadata,
)
from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)


# =========================================================================
# Helpers
# =========================================================================

def _make_kv_cache_config(
    block_size: int,
    num_blocks: int = 64,
) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            )
        ],
    )


def _make_scheduler_output(
    req_id: str,
    num_computed_tokens: int,
    num_scheduled_tokens: int,
    has_prefill: bool,
) -> SchedulerOutput:
    if has_prefill:
        cached_reqs = CachedRequestData.make_empty()
        scheduled_new_reqs = [
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=[1],
                mm_features=[],
                sampling_params=None,
                pooling_params=None,
                block_ids=([],),
                num_computed_tokens=0,
                lora_request=None,
            )
        ]
    else:
        cached_reqs = CachedRequestData(
            req_ids=[req_id],
            resumed_req_ids=set(),
            new_token_ids=[[]],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[num_computed_tokens],
            num_output_tokens=[1],
        )
        scheduled_new_reqs = []
    return SchedulerOutput(
        scheduled_new_reqs=scheduled_new_reqs,
        scheduled_cached_reqs=cached_reqs,
        num_scheduled_tokens={req_id: num_scheduled_tokens},
        total_num_scheduled_tokens=num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _make_mixed_scheduler_output(
    prefill_req_id: str,
    decode_req_id: str,
    decode_num_computed: int,
) -> SchedulerOutput:
    """Create a scheduler output with both a prefill and a decode request."""
    cached_reqs = CachedRequestData(
        req_ids=[decode_req_id],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[decode_num_computed],
        num_output_tokens=[1],
    )
    scheduled_new_reqs = [
        NewRequestData(
            req_id=prefill_req_id,
            prompt_token_ids=[1],
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=([],),
            num_computed_tokens=0,
            lora_request=None,
        )
    ]
    return SchedulerOutput(
        scheduled_new_reqs=scheduled_new_reqs,
        scheduled_cached_reqs=cached_reqs,
        num_scheduled_tokens={
            prefill_req_id: 1,
            decode_req_id: 1,
        },
        total_num_scheduled_tokens=2,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _make_connector(
    k_window: int = 2,
    kv_cache_config: KVCacheConfig | None = None,
) -> PrefillStagingPipelinedConnector:
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_connector = (
        "PrefillStagingPipelinedConnector"
    )
    vllm_config.cache_config.enable_kv_pipelining = True
    vllm_config.cache_config.kv_k_window = k_window
    if kv_cache_config is None:
        kv_cache_config = _make_kv_cache_config(
            vllm_config.cache_config.block_size,
        )
    return PrefillStagingPipelinedConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=kv_cache_config,
    )


# =========================================================================
# Scheduler-side tests
# =========================================================================


def test_get_num_new_matched_tokens_returns_zero():
    """The connector should not claim any external tokens — let prefill
    happen normally through the model."""
    connector = _make_connector()

    class FakeRequest:
        request_id = "r1"
        num_tokens = 100

    num_matched, is_async = connector.get_num_new_matched_tokens(
        FakeRequest(), num_computed_tokens=0
    )
    assert num_matched == 0
    assert is_async is False


def test_build_connector_meta_pure_prefill():
    """A pure prefill step should have has_prefill_work=True and
    has_decode_work=False."""
    connector = _make_connector()

    scheduler_output = _make_scheduler_output(
        req_id="r1",
        num_computed_tokens=0,
        num_scheduled_tokens=10,
        has_prefill=True,
    )
    metadata = connector.build_connector_meta(scheduler_output)
    assert isinstance(
        metadata, PrefillStagingPipelinedConnectorMetadata
    )
    assert metadata.has_prefill_work is True
    assert metadata.has_decode_work is False
    assert metadata.has_work is True
    # No layers to prefetch during prefill
    assert len(metadata.layers_to_prefetch) == 0


def test_build_connector_meta_pure_decode():
    """A pure decode step should have has_decode_work=True and
    has_prefill_work=False."""
    connector = _make_connector(k_window=2)

    scheduler_output = _make_scheduler_output(
        req_id="r1",
        num_computed_tokens=50,
        num_scheduled_tokens=1,
        has_prefill=False,
    )
    metadata = connector.build_connector_meta(scheduler_output)
    assert isinstance(
        metadata, PrefillStagingPipelinedConnectorMetadata
    )
    assert metadata.has_prefill_work is False
    assert metadata.has_decode_work is True
    assert metadata.has_work is True
    assert metadata.decode_max_seq_len == 51
    # Should have layers_to_prefetch for decode
    assert len(metadata.layers_to_prefetch) <= 2


def test_build_connector_meta_mixed_step():
    """A mixed step (prefill + decode) should have both flags, but
    has_decode_work should be False (decode pipelining only on pure
    decode steps)."""
    connector = _make_connector()

    scheduler_output = _make_mixed_scheduler_output(
        prefill_req_id="r1",
        decode_req_id="r2",
        decode_num_computed=50,
    )
    metadata = connector.build_connector_meta(scheduler_output)
    assert isinstance(
        metadata, PrefillStagingPipelinedConnectorMetadata
    )
    assert metadata.has_prefill_work is True
    # In mixed steps, has_decode_work is False (we don't pipeline)
    assert metadata.has_decode_work is False
    # No prefetch during mixed steps
    assert len(metadata.layers_to_prefetch) == 0


def test_build_connector_meta_decode_max_seq_len():
    """decode_max_seq_len should reflect the computed + scheduled tokens."""
    connector = _make_connector()

    scheduler_output = _make_scheduler_output(
        req_id="r1",
        num_computed_tokens=100,
        num_scheduled_tokens=1,
        has_prefill=False,
    )
    metadata = connector.build_connector_meta(scheduler_output)
    assert metadata.decode_max_seq_len == 101


def test_request_finished_is_noop():
    """request_finished should return (False, None) — no special handling."""
    connector = _make_connector()

    class FakeRequest:
        request_id = "r1"

    result, params = connector.request_finished(FakeRequest(), block_ids=[])
    assert result is False
    assert params is None


def test_build_connector_meta_no_work():
    """When there's no scheduled work, metadata should reflect that."""
    connector = _make_connector()

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    metadata = connector.build_connector_meta(scheduler_output)
    assert metadata.has_work is False
    assert metadata.has_prefill_work is False
    assert metadata.has_decode_work is False
