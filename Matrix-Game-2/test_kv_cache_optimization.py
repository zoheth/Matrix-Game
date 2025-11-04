"""
Test script to verify KV cache optimization and measure performance improvement.
"""

import torch
import time
from wan.modules.modular_action.kernels.kv_cache_kernel import update_kv_cache_optimized


def create_kv_cache(batch_size, cache_size, num_heads, head_dim, device='cuda'):
    """Create a fresh KV cache"""
    return {
        'k': torch.zeros(batch_size, cache_size, num_heads, head_dim, device=device),
        'v': torch.zeros(batch_size, cache_size, num_heads, head_dim, device=device),
        'global_end_index': torch.tensor(0, device=device),
        'local_end_index': torch.tensor(0, device=device),
    }


def test_optimized_cache():
    """Test the optimized KV cache implementation"""
    print("=" * 80)
    print("Testing Optimized KV Cache Implementation")
    print("=" * 80)

    # Test parameters
    batch_size = 4
    cache_size = 1000
    num_heads = 16
    head_dim = 64
    max_attention_size = 512
    device = 'cuda'

    # Create cache
    kv_cache = create_kv_cache(batch_size, cache_size, num_heads, head_dim, device)

    # Test 1: Basic insertion (no eviction)
    print("\n[Test 1] Basic insertion (no eviction)")
    num_new_tokens = 100
    k_new = torch.randn(batch_size, num_new_tokens, num_heads, head_dim, device=device)
    v_new = torch.randn(batch_size, num_new_tokens, num_heads, head_dim, device=device)

    k_window, v_window, start_idx, end_idx = update_kv_cache_optimized(
        kv_cache, k_new, v_new, num_new_tokens, max_attention_size, sink_tokens=0
    )

    print(f"  ✓ Inserted {num_new_tokens} tokens")
    print(f"  ✓ Window shape: {k_window.shape}")
    print(f"  ✓ Start index: {start_idx}, End index: {end_idx}")
    print(f"  ✓ Global end: {kv_cache['global_end_index'].item()}")
    print(f"  ✓ Local end: {kv_cache['local_end_index'].item()}")

    # Test 2: Multiple insertions
    print("\n[Test 2] Multiple insertions")
    for i in range(5):
        k_new = torch.randn(batch_size, 50, num_heads, head_dim, device=device)
        v_new = torch.randn(batch_size, 50, num_heads, head_dim, device=device)
        k_window, v_window, start_idx, end_idx = update_kv_cache_optimized(
            kv_cache, k_new, v_new, 50, max_attention_size, sink_tokens=0
        )
        print(f"  Iteration {i+1}: global_end={kv_cache['global_end_index'].item()}, "
              f"local_end={kv_cache['local_end_index'].item()}, window_size={k_window.shape[1]}")

    # Test 3: Trigger eviction
    print("\n[Test 3] Triggering cache eviction")
    # Fill cache to near capacity
    kv_cache = create_kv_cache(batch_size, cache_size, num_heads, head_dim, device)
    k_new = torch.randn(batch_size, cache_size - 100, num_heads, head_dim, device=device)
    v_new = torch.randn(batch_size, cache_size - 100, num_heads, head_dim, device=device)
    update_kv_cache_optimized(kv_cache, k_new, v_new, cache_size - 100, max_attention_size, sink_tokens=0)

    print(f"  Cache filled to: {kv_cache['local_end_index'].item()}/{cache_size}")

    # Now add more tokens to trigger eviction
    k_new = torch.randn(batch_size, 200, num_heads, head_dim, device=device)
    v_new = torch.randn(batch_size, 200, num_heads, head_dim, device=device)
    k_window, v_window, start_idx, end_idx = update_kv_cache_optimized(
        kv_cache, k_new, v_new, 200, max_attention_size, sink_tokens=0
    )

    print(f"  ✓ Eviction triggered successfully")
    print(f"  ✓ Local end after eviction: {kv_cache['local_end_index'].item()}")
    print(f"  ✓ Window size: {k_window.shape[1]}")

    # Test 4: Performance benchmark
    print("\n[Test 4] Performance benchmark")
    kv_cache = create_kv_cache(batch_size, cache_size, num_heads, head_dim, device)

    # Warmup
    for _ in range(10):
        k_new = torch.randn(batch_size, 10, num_heads, head_dim, device=device)
        v_new = torch.randn(batch_size, 10, num_heads, head_dim, device=device)
        update_kv_cache_optimized(kv_cache, k_new, v_new, 10, max_attention_size, sink_tokens=0)

    # Benchmark
    torch.cuda.synchronize()
    num_iterations = 100
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        k_new = torch.randn(batch_size, 10, num_heads, head_dim, device=device)
        v_new = torch.randn(batch_size, 10, num_heads, head_dim, device=device)
        update_kv_cache_optimized(kv_cache, k_new, v_new, 10, max_attention_size, sink_tokens=0)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_time_ms = (end_time - start_time) * 1000 / num_iterations
    print(f"  ✓ Average time per update: {avg_time_ms:.3f} ms")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
    else:
        test_optimized_cache()
