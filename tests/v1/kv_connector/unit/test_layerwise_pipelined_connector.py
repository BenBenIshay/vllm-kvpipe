# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.layerwise_pipelined_connector import (
    LayerwisePipelinedConnector,
    LayerwisePipelinedConnectorMetadata,
)
from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec


def _make_kv_cache_config(block_size: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=16,
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


def test_layerwise_pipelined_connector_metadata_decode_only():
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_connector = "LayerwisePipelinedConnector"
    vllm_config.cache_config.enable_kv_pipelining = True
    vllm_config.cache_config.kv_k_window = 2

    kv_cache_config = _make_kv_cache_config(vllm_config.cache_config.block_size)
    connector = LayerwisePipelinedConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=kv_cache_config,
    )

    scheduler_output = _make_scheduler_output(
        req_id="r1",
        num_computed_tokens=5,
        num_scheduled_tokens=1,
        has_prefill=False,
    )
    metadata = connector.build_connector_meta(scheduler_output)
    assert isinstance(metadata, LayerwisePipelinedConnectorMetadata)
    assert metadata.has_decode_work is True
    assert metadata.decode_max_seq_len == 6


def test_layerwise_pipelined_connector_metadata_prefill_gates_decode():
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_connector = "LayerwisePipelinedConnector"
    vllm_config.cache_config.enable_kv_pipelining = True
    vllm_config.cache_config.kv_k_window = 2

    kv_cache_config = _make_kv_cache_config(vllm_config.cache_config.block_size)
    connector = LayerwisePipelinedConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=kv_cache_config,
    )

    scheduler_output = _make_scheduler_output(
        req_id="r1",
        num_computed_tokens=0,
        num_scheduled_tokens=1,
        has_prefill=True,
    )
    metadata = connector.build_connector_meta(scheduler_output)
    assert isinstance(metadata, LayerwisePipelinedConnectorMetadata)
    assert metadata.has_decode_work is False
