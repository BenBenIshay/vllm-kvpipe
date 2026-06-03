# SPDX-License-Identifier: Apache-2.0
"""Round-trip tests for LayerwisePipeliningState (CPU staging + prefetch).

If these pass but end-to-end generation fails, the bug is likely in connector
scheduling / layer ordering / slice bounds — not in basic copy_ math.
"""

import pytest
import torch

from vllm.v1.kv_cache_pipelining import LayerwisePipeliningState


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage_to_cpu_prefetch_roundtrip_full_tensor():
    """Full tensor copy GPU→CPU→GPU should preserve values."""
    torch.cuda.set_device(0)
    num_blocks, block_size, heads, dim = 4, 16, 2, 8
    kv = torch.randn(
        2, num_blocks, block_size, heads, dim, device="cuda", dtype=torch.float16
    )
    state = LayerwisePipeliningState()
    state.stage_to_cpu("L0", kv)
    state.wait_for_saves()

    out = torch.zeros_like(kv)
    state.prefetch_to_gpu("L0", out)
    state.wait_for_layer("L0")

    torch.cuda.synchronize()
    assert torch.allclose(kv, out, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage_delta_scatter_matches_reference():
    """Delta scatter into CPU buffer should match reference indexing."""
    torch.cuda.set_device(0)
    num_blocks, block_size, heads, dim = 8, 16, 2, 8
    kv = torch.randn(
        2, num_blocks, block_size, heads, dim, device="cuda", dtype=torch.float16
    )
    # Pick two physical slots in flattened [num_blocks * block_size]
    flat_slots = torch.tensor([3, 41], device="cuda", dtype=torch.long)

    state = LayerwisePipeliningState()
    state.stage_to_cpu("L0", kv.clone())
    state.wait_for_saves()

    # Overwrite those slots with a known delta
    delta_pattern = torch.randn(
        2, flat_slots.numel(), heads, dim, device="cuda", dtype=torch.float16
    )
    kv2 = kv.clone()
    nb, ps = kv2.shape[1], kv2.shape[2]
    view = kv2.reshape(2, nb * ps, heads, dim)
    view[:, flat_slots, ...] = delta_pattern

    state.stage_delta_to_cpu("L0", kv2, flat_slots, num_actual_tokens=2)
    state.wait_for_saves()

    cpu_buf = state._layers["L0"].cpu_buffer
    ref = cpu_buf.clone()
    nb, ps = ref.shape[1], ref.shape[2]
    refv = ref.reshape(2, nb * ps, heads, dim)
    refv[:, flat_slots, ...] = delta_pattern.cpu()

    assert torch.allclose(cpu_buf, ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage_delta_scatter_blocks_first_layout():
    """Triton-style (num_blocks, 2, block_size, ...) delta scatter matches reference."""
    torch.cuda.set_device(0)
    num_blocks, block_size, heads, dim = 8, 16, 2, 8
    kv_blocks_first = torch.randn(
        num_blocks, 2, block_size, heads, dim, device="cuda", dtype=torch.float16
    )
    flat_slots = torch.tensor([3, 41], device="cuda", dtype=torch.long)
    delta_pattern = torch.randn(
        2, flat_slots.numel(), heads, dim, device="cuda", dtype=torch.float16
    )

    state = LayerwisePipeliningState()
    state.stage_to_cpu("L0", kv_blocks_first.clone())
    state.wait_for_saves()

    kv2 = kv_blocks_first.clone()
    nb, ps = kv2.shape[0], kv2.shape[2]
    view = kv2.permute(1, 0, 2, 3, 4).reshape(2, nb * ps, heads, dim)
    view[:, flat_slots, ...] = delta_pattern

    state.stage_delta_to_cpu("L0", kv2, flat_slots, num_actual_tokens=2)
    state.wait_for_saves()

    ref_cpu = kv_blocks_first.cpu().clone()
    nb_r, ps_r = ref_cpu.shape[0], ref_cpu.shape[2]
    view_cpu = ref_cpu.permute(1, 0, 2, 3, 4).reshape(2, nb_r * ps_r, heads, dim)
    view_cpu[:, flat_slots.cpu(), ...] = delta_pattern.cpu()

    assert torch.allclose(state._layers["L0"].cpu_buffer, ref_cpu, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage_delta_scatter_ignores_invalid_slot_sentinels():
    """Negative sentinel slots must be ignored, not wrapped as valid indices."""
    torch.cuda.set_device(0)
    num_blocks, block_size, heads, dim = 8, 16, 2, 8
    kv = torch.randn(
        2, num_blocks, block_size, heads, dim, device="cuda", dtype=torch.float16
    )
    flat_slots = torch.tensor([3, -1, 41, -1], device="cuda", dtype=torch.long)
    valid_slots = torch.tensor([3, 41], device="cuda", dtype=torch.long)

    state = LayerwisePipeliningState()
    state.stage_to_cpu("L0", kv.clone())
    state.wait_for_saves()
    base_cpu = state._layers["L0"].cpu_buffer.clone()

    delta_pattern = torch.randn(
        2, valid_slots.numel(), heads, dim, device="cuda", dtype=torch.float16
    )
    kv2 = kv.clone()
    view = kv2.reshape(2, num_blocks * block_size, heads, dim)
    view[:, valid_slots, ...] = delta_pattern

    state.stage_delta_to_cpu("L0", kv2, flat_slots, num_actual_tokens=4)
    state.wait_for_saves()

    ref = base_cpu.clone()
    ref_view = ref.reshape(2, num_blocks * block_size, heads, dim)
    ref_view[:, valid_slots.cpu(), ...] = delta_pattern.cpu()
    assert torch.allclose(state._layers["L0"].cpu_buffer, ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage_to_cpu_prefetch_roundtrip_blocks_first_layout():
    """Full tensor GPU→CPU→GPU for Triton (num_blocks, 2, block_size, …)."""
    torch.cuda.set_device(0)
    num_blocks, block_size, heads, dim = 4, 16, 2, 8
    kv = torch.randn(
        num_blocks, 2, block_size, heads, dim, device="cuda", dtype=torch.float16
    )
    state = LayerwisePipeliningState()
    state.stage_to_cpu("L0", kv)
    state.wait_for_saves()

    out = torch.zeros_like(kv)
    state.prefetch_to_gpu("L0", out)
    state.wait_for_layer("L0")

    torch.cuda.synchronize()
    assert torch.allclose(kv, out, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage_to_cpu_block_span_preserves_row_offsets():
    """Block-span staging must write into matching CPU rows (not row 0)."""
    torch.cuda.set_device(0)
    nb, block_size, heads, dim = 16, 8, 2, 4
    kv = torch.zeros(
        2, nb, block_size, heads, dim, device="cuda", dtype=torch.float16
    )
    lo, hi = 10, 12
    kv[:, lo : hi + 1].normal_()

    state = LayerwisePipeliningState()
    state.ensure_cpu_buffer("L", kv)
    state.stage_to_cpu_block_span("L", kv, lo, hi)
    state.wait_for_saves()

    cpu_buf = state._layers["L"].cpu_buffer
    assert torch.allclose(
        cpu_buf[:, lo : hi + 1], kv[:, lo : hi + 1].cpu(), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage_to_cpu_block_runs_preserves_disjoint_rows():
    """Multi-run staging must copy each physical block span to matching CPU rows."""
    torch.cuda.set_device(0)
    nb, block_size, heads, dim = 32, 8, 2, 4
    kv = torch.zeros(
        2, nb, block_size, heads, dim, device="cuda", dtype=torch.float16
    )
    runs = [(2, 4), (20, 22)]
    for lo, hi in runs:
        kv[:, lo : hi + 1].normal_()

    state = LayerwisePipeliningState()
    state.ensure_cpu_buffer("L", kv)
    state.stage_to_cpu_block_runs("L", kv, runs)
    state.wait_for_saves()

    cpu_buf = state._layers["L"].cpu_buffer
    for lo, hi in runs:
        assert torch.allclose(
            cpu_buf[:, lo : hi + 1],
            kv[:, lo : hi + 1].cpu(),
            rtol=0,
            atol=0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wait_for_saves_deferred_waits_at_prefetch():
    """Decode-style deferred barrier: CPU delta must be visible before H2D."""
    torch.cuda.set_device(0)
    num_blocks, block_size, heads, dim = 8, 16, 2, 8
    kv = torch.randn(
        2, num_blocks, block_size, heads, dim, device="cuda", dtype=torch.float16
    )
    flat_slots = torch.tensor([3, 41], device="cuda", dtype=torch.long)
    delta_pattern = torch.randn(
        2, flat_slots.numel(), heads, dim, device="cuda", dtype=torch.float16
    )

    state = LayerwisePipeliningState()
    state.stage_to_cpu("L0", kv.clone())
    state.wait_for_saves()

    kv2 = kv.clone()
    view = kv2.reshape(2, num_blocks * block_size, heads, dim)
    view[:, flat_slots, ...] = delta_pattern
    state.stage_delta_to_cpu("L0", kv2, flat_slots, num_actual_tokens=2)
    # Do not join — same as decode-only wait_for_saves(deferred=True).
    state.wait_for_saves(deferred=True)

    out = torch.zeros_like(kv)
    state.prefetch_to_gpu("L0", out)
    state.wait_for_layer("L0")
    torch.cuda.synchronize()

    ref = state._layers["L0"].cpu_buffer.clone()
    ref_view = ref.reshape(2, num_blocks * block_size, heads, dim)
    ref_view[:, flat_slots.cpu(), ...] = delta_pattern.cpu()
    assert torch.allclose(out.cpu(), ref, rtol=0, atol=0), "deferred barrier + prefetch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefetch_block_runs_roundtrip_disjoint_rows():
    """GPU→CPU (runs)→GPU should restore only the copied spans."""
    torch.cuda.set_device(0)
    nb, block_size, heads, dim = 32, 8, 2, 4
    kv = torch.zeros(
        2, nb, block_size, heads, dim, device="cuda", dtype=torch.float16
    )
    runs = [(1, 2), (18, 19)]
    for lo, hi in runs:
        kv[:, lo : hi + 1].normal_()

    state = LayerwisePipeliningState()
    state.ensure_cpu_buffer("L", kv)
    state.stage_to_cpu_block_runs("L", kv, runs)
    state.wait_for_saves()

    out = torch.zeros_like(kv)
    state.prefetch_to_gpu_block_runs("L", out, runs)
    state.wait_for_layer("L")

    torch.cuda.synchronize()
    for lo, hi in runs:
        assert torch.allclose(
            out[:, lo : hi + 1], kv[:, lo : hi + 1], rtol=0, atol=0
        )
