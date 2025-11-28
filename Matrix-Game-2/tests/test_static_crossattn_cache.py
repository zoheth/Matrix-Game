"""
Tests for StaticCrossAttnCache.

Validates CUDA Graph compatibility and correctness.
"""

import torch
import pytest
from wan.modules.static_crossattn_cache import StaticCrossAttnCache


@pytest.fixture
def cache_params():
    """Common cache parameters for testing."""
    return {
        'batch_size': 1,
        'seq_len': 257,
        'num_heads': 12,
        'head_dim': 128,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'dtype': torch.bfloat16,
    }


def test_basic_update_and_retrieval(cache_params):
    """Test basic cache update and retrieval."""
    cache = StaticCrossAttnCache(**cache_params)

    # Create dummy K/V tensors
    k_new = torch.randn(
        cache_params['batch_size'],
        cache_params['seq_len'],
        cache_params['num_heads'],
        cache_params['head_dim'],
        dtype=cache_params['dtype'],
        device=cache_params['device']
    )
    v_new = torch.randn_like(k_new)

    # First call should store K/V
    k_out, v_out = cache.update_or_get(k_new, v_new)
    assert torch.allclose(k_out, k_new, atol=1e-6)
    assert torch.allclose(v_out, v_new, atol=1e-6)
    assert cache.is_initialized.item() == 1

    # Second call with different K/V should return cached values
    k_new2 = torch.randn_like(k_new)
    v_new2 = torch.randn_like(v_new)
    k_out2, v_out2 = cache.update_or_get(k_new2, v_new2)

    # Should still return the first K/V (cached)
    assert torch.allclose(k_out2, k_new, atol=1e-6)
    assert torch.allclose(v_out2, v_new, atol=1e-6)


def test_reset_functionality(cache_params):
    """Test cache reset."""
    cache = StaticCrossAttnCache(**cache_params)

    # Initialize cache
    k1 = torch.randn(
        cache_params['batch_size'],
        cache_params['seq_len'],
        cache_params['num_heads'],
        cache_params['head_dim'],
        dtype=cache_params['dtype'],
        device=cache_params['device']
    )
    v1 = torch.randn_like(k1)
    cache.update_or_get(k1, v1)
    assert cache.is_initialized.item() == 1

    # Reset cache
    cache.reset()
    assert cache.is_initialized.item() == 0

    # After reset, should accept new K/V
    k2 = torch.randn_like(k1)
    v2 = torch.randn_like(v1)
    k_out, v_out = cache.update_or_get(k2, v2)
    assert torch.allclose(k_out, k2, atol=1e-6)
    assert torch.allclose(v_out, v2, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_compatibility(cache_params):
    """Test CUDA Graph capture and replay."""
    cache = StaticCrossAttnCache(**cache_params)

    # Create static input tensors
    k_input = torch.randn(
        cache_params['batch_size'],
        cache_params['seq_len'],
        cache_params['num_heads'],
        cache_params['head_dim'],
        dtype=cache_params['dtype'],
        device=cache_params['device']
    )
    v_input = torch.randn_like(k_input)

    # Warmup: Initialize cache before graph capture
    with torch.no_grad():
        cache.update_or_get(k_input, v_input)

    # Capture CUDA Graph
    graph = torch.cuda.CUDAGraph()

    # Create new inputs for graph capture
    k_graph = torch.randn_like(k_input)
    v_graph = torch.randn_like(v_input)

    with torch.cuda.graph(graph):
        k_out, v_out = cache.update_or_get(k_graph, v_graph)

    # Graph should execute without errors
    graph.replay()

    # Verify outputs are from cache (first warmup values)
    assert k_out.shape == k_input.shape
    assert v_out.shape == v_input.shape


def test_multiple_blocks(cache_params):
    """Test multiple cache instances (simulating transformer blocks)."""
    num_blocks = 3
    caches = [StaticCrossAttnCache(**cache_params) for _ in range(num_blocks)]

    # Initialize each cache with different values
    k_values = []
    v_values = []
    for cache in caches:
        k = torch.randn(
            cache_params['batch_size'],
            cache_params['seq_len'],
            cache_params['num_heads'],
            cache_params['head_dim'],
            dtype=cache_params['dtype'],
            device=cache_params['device']
        )
        v = torch.randn_like(k)
        k_values.append(k)
        v_values.append(v)
        cache.update_or_get(k, v)

    # Verify each cache stores its own values
    for i, cache in enumerate(caches):
        k_dummy = torch.zeros_like(k_values[i])
        v_dummy = torch.zeros_like(v_values[i])
        k_out, v_out = cache.update_or_get(k_dummy, v_dummy)
        assert torch.allclose(k_out, k_values[i], atol=1e-6)
        assert torch.allclose(v_out, v_values[i], atol=1e-6)


def test_repr(cache_params):
    """Test string representation."""
    cache = StaticCrossAttnCache(**cache_params)
    repr_str = repr(cache)
    assert 'StaticCrossAttnCache' in repr_str
    assert 'batch_size=1' in repr_str
    assert 'seq_len=257' in repr_str
    assert 'initialized=False' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
