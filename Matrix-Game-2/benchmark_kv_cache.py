"""
Benchmark script to compare original vs optimized KV cache implementation.
"""

import torch
import time
import numpy as np
from wan.modules.modular_action.kernels.kv_cache_kernel import update_kv_cache_optimized


def create_kv_cache(batch_size, cache_size, num_heads, head_dim, device='cuda'):
    """Create a fresh KV cache"""
    return {
        'k': torch.zeros(batch_size, cache_size, num_heads, head_dim, device=device),
        'v': torch.zeros(batch_size, cache_size, num_heads, head_dim, device=device),
        'global_end_index': torch.tensor(0, device=device),
        'local_end_index': torch.tensor(0, device=device),
    }


def original_update_cache(
    kv_cache, k, v, num_new_tokens, max_attention_size, sink_tokens
):
    """
    Original implementation with 8 .item() calls and 2 .clone() calls.
    """
    current_start = kv_cache["global_end_index"].item()  # D2H #1
    current_end = current_start + num_new_tokens

    kv_cache_size = kv_cache["k"].shape[1]

    # Check if we need to evict tokens
    if (current_end > kv_cache["global_end_index"].item()) and \
       (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):  # D2H #2, #3
        # Calculate eviction
        num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size  # D2H #4
        num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens  # D2H #5

        # Roll the cache: move recent tokens to make space
        kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()  # Clone #1
        kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()  # Clone #2

        local_end_index = kv_cache["local_end_index"].item() + current_end - \
            kv_cache["global_end_index"].item() - num_evicted_tokens  # D2H #6, #7
        local_start_index = local_end_index - num_new_tokens
    else:
        # No eviction needed
        local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()  # D2H #8
        local_start_index = local_end_index - num_new_tokens

    # Insert new keys/values
    kv_cache["k"][:, local_start_index:local_end_index] = k
    kv_cache["v"][:, local_start_index:local_end_index] = v

    # Update global indices
    kv_cache["global_end_index"].fill_(current_end)
    kv_cache["local_end_index"].fill_(local_end_index)

    # Extract attention window
    window_start = max(0, local_end_index - max_attention_size)
    k_window = kv_cache["k"][:, window_start:local_end_index]
    v_window = kv_cache["v"][:, window_start:local_end_index]

    return k_window, v_window, local_start_index, local_end_index


def benchmark_implementation(impl_name, update_func, num_iterations=100):
    """Benchmark a KV cache implementation"""
    batch_size = 4
    cache_size = 1000
    num_heads = 16
    head_dim = 64
    max_attention_size = 512
    device = 'cuda'

    # Create cache
    kv_cache = create_kv_cache(batch_size, cache_size, num_heads, head_dim, device)

    # Warmup
    for _ in range(10):
        k_new = torch.randn(batch_size, 10, num_heads, head_dim, device=device)
        v_new = torch.randn(batch_size, 10, num_heads, head_dim, device=device)
        update_func(kv_cache, k_new, v_new, 10, max_attention_size, 0)

    # Reset cache
    kv_cache = create_kv_cache(batch_size, cache_size, num_heads, head_dim, device)

    # Benchmark
    torch.cuda.synchronize()
    times = []

    for _ in range(num_iterations):
        k_new = torch.randn(batch_size, 10, num_heads, head_dim, device=device)
        v_new = torch.randn(batch_size, 10, num_heads, head_dim, device=device)

        start = time.perf_counter()
        update_func(kv_cache, k_new, v_new, 10, max_attention_size, 0)
        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # Convert to ms

    return {
        'name': impl_name,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
    }


def main():
    print("=" * 80)
    print("KV Cache Implementation Benchmark")
    print("=" * 80)
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")
    print()

    num_iterations = 200

    print(f"Running {num_iterations} iterations for each implementation...\n")

    # Benchmark original
    print("[1/2] Benchmarking original implementation...")
    original_results = benchmark_implementation(
        "Original (8x .item() + 2x .clone())",
        original_update_cache,
        num_iterations
    )

    # Benchmark optimized
    print("[2/2] Benchmarking optimized implementation...")
    optimized_results = benchmark_implementation(
        "Optimized (2x .item() + 0x .clone())",
        update_kv_cache_optimized,
        num_iterations
    )

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for results in [original_results, optimized_results]:
        print(f"\n{results['name']}:")
        print(f"  Mean:    {results['mean']:.4f} ms")
        print(f"  Std Dev: {results['std']:.4f} ms")
        print(f"  Median:  {results['median']:.4f} ms")
        print(f"  Min:     {results['min']:.4f} ms")
        print(f"  Max:     {results['max']:.4f} ms")

    # Calculate speedup
    speedup = original_results['mean'] / optimized_results['mean']
    time_saved = original_results['mean'] - optimized_results['mean']
    reduction_pct = (time_saved / original_results['mean']) * 100

    print("\n" + "=" * 80)
    print("PERFORMANCE IMPROVEMENT")
    print("=" * 80)
    print(f"Speedup:        {speedup:.2f}x")
    print(f"Time saved:     {time_saved:.4f} ms per update")
    print(f"Reduction:      {reduction_pct:.1f}%")
    print()
    print("Key optimizations:")
    print("  ✓ Reduced .item() calls from 8 to 2 (75% reduction in D2H transfers)")
    print("  ✓ Replaced .clone() with .contiguous() (eliminates unnecessary copies)")
    print("  ✓ Better memory access patterns")
    print("=" * 80)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
    else:
        main()
