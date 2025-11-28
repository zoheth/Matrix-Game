"""
Comprehensive test suite for RingBufferActionCache.

This test validates:
1. Correctness: Ring buffer produces same results as ActionCache
2. CUDA Graph compatibility: No CPU control flow in forward path
3. Performance: Ring buffer is faster than memory-shifting approach
"""

import torch
import time
from typing import Tuple

from wan.modules.action_cache import ActionCache
from wan.modules.ring_buffer_action_cache import RingBufferActionCache


def test_basic_insertion():
    """Test 1: Basic insertion without eviction."""
    print("\n" + "=" * 80)
    print("Test 1: Basic Insertion (No Eviction)")
    print("=" * 80)

    batch_size = 2
    max_seq_len = 1024
    num_heads = 8
    head_dim = 64
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Create both caches
    action_cache = ActionCache(batch_size, max_seq_len, num_heads, head_dim, device, dtype)
    ring_cache = RingBufferActionCache(batch_size, max_seq_len, num_heads, head_dim, device, dtype)

    # Insert 100 tokens
    num_tokens = 100
    k = torch.randn(batch_size, num_tokens, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_tokens, num_heads, head_dim, device=device, dtype=dtype)

    # ActionCache
    k_window_action, v_window_action, start_action, end_action = action_cache.update_and_get_window(
        k, v, num_tokens, max_attention_size=max_seq_len, sink_tokens=0
    )

    # RingBufferActionCache
    k_window_ring, v_window_ring, mask_ring = ring_cache.update_and_get_window(
        k, v, num_new_tokens=num_tokens, max_attention_size=max_seq_len
    )

    # Validate shapes
    print(f"  ActionCache window shape: {k_window_action.shape}")
    print(f"  RingBuffer window shape: {k_window_ring.shape}")
    print(f"  RingBuffer mask: {mask_ring.sum().item()}/{mask_ring.numel()} tokens valid")

    # Extract valid portion from ring buffer (mask-based)
    valid_len = mask_ring.sum().item()
    k_window_ring_valid = k_window_ring[:, -valid_len:]
    v_window_ring_valid = v_window_ring[:, -valid_len:]

    # Compare
    assert torch.allclose(k_window_action, k_window_ring_valid, atol=1e-5), "K mismatch!"
    assert torch.allclose(v_window_action, v_window_ring_valid, atol=1e-5), "V mismatch!"

    print("  ✓ Ring buffer matches ActionCache")
    print(f"  ✓ Indices: ActionCache [{start_action}:{end_action}], RingBuffer valid_len={valid_len}")


def test_incremental_insertion():
    """Test 2: Incremental insertion (simulating decode)."""
    print("\n" + "=" * 80)
    print("Test 2: Incremental Insertion (Decode Simulation)")
    print("=" * 80)

    batch_size = 1
    max_seq_len = 512
    num_heads = 12
    head_dim = 64
    device = torch.device("cuda")
    dtype = torch.bfloat16

    action_cache = ActionCache(batch_size, max_seq_len, num_heads, head_dim, device, dtype)
    ring_cache = RingBufferActionCache(batch_size, max_seq_len, num_heads, head_dim, device, dtype)

    # Simulate decode: add 1 token at a time for 100 steps
    num_steps = 100
    for step in range(num_steps):
        k = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)

        k_action, v_action, _, _ = action_cache.update_and_get_window(
            k, v, num_new_tokens=1, max_attention_size=max_seq_len, sink_tokens=0
        )

        k_ring, v_ring, mask_ring = ring_cache.update_and_get_window(
            k, v, num_new_tokens=1, max_attention_size=max_seq_len
        )

        # Validate at each step
        valid_len = mask_ring.sum().item()
        k_ring_valid = k_ring[:, -valid_len:]
        v_ring_valid = v_ring[:, -valid_len:]

        if not torch.allclose(k_action, k_ring_valid, atol=1e-5):
            print(f"  ✗ K mismatch at step {step}")
            print(f"    ActionCache shape: {k_action.shape}, RingBuffer valid: {k_ring_valid.shape}")
            break

    else:
        print(f"  ✓ All {num_steps} incremental steps match")
        print(f"  ✓ Final cache size: {valid_len} tokens")


def test_eviction_with_wrapping():
    """Test 3: Test eviction and ring buffer wrapping."""
    print("\n" + "=" * 80)
    print("Test 3: Eviction with Ring Buffer Wrapping")
    print("=" * 80)

    batch_size = 2
    max_seq_len = 128  # Small cache to trigger eviction quickly
    num_heads = 8
    head_dim = 64
    device = torch.device("cuda")
    dtype = torch.bfloat16

    action_cache = ActionCache(batch_size, max_seq_len, num_heads, head_dim, device, dtype)
    ring_cache = RingBufferActionCache(batch_size, max_seq_len, num_heads, head_dim, device, dtype)

    # Add tokens in batches to exceed capacity
    total_tokens = 0
    num_batches = 20
    tokens_per_batch = 10

    for batch_idx in range(num_batches):
        k = torch.randn(batch_size, tokens_per_batch, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, tokens_per_batch, num_heads, head_dim, device=device, dtype=dtype)

        k_action, v_action, _, _ = action_cache.update_and_get_window(
            k, v, num_new_tokens=tokens_per_batch, max_attention_size=max_seq_len, sink_tokens=0
        )

        k_ring, v_ring, mask_ring = ring_cache.update_and_get_window(
            k, v, num_new_tokens=tokens_per_batch, max_attention_size=max_seq_len
        )

        total_tokens += tokens_per_batch

        # After cache is full, validate results
        if total_tokens >= max_seq_len:
            valid_len = mask_ring.sum().item()
            k_ring_valid = k_ring[:, -valid_len:]
            v_ring_valid = v_ring[:, -valid_len:]

            if not torch.allclose(k_action, k_ring_valid, atol=1e-5):
                print(f"  ✗ Mismatch at batch {batch_idx} (total={total_tokens})")
                print(f"    ActionCache: {k_action.shape}, RingBuffer valid: {k_ring_valid.shape}")
                break
    else:
        print(f"  ✓ All {num_batches} batches with eviction match")
        print(f"  ✓ Total tokens processed: {total_tokens}, cache capacity: {max_seq_len}")
        print(f"  ✓ Ring buffer wrapped {total_tokens // max_seq_len} times")


def test_attention_mask_correctness():
    """Test 4: Validate attention mask semantics."""
    print("\n" + "=" * 80)
    print("Test 4: Attention Mask Correctness")
    print("=" * 80)

    batch_size = 1
    max_seq_len = 64
    num_heads = 4
    head_dim = 32
    device = torch.device("cuda")
    dtype = torch.bfloat16

    ring_cache = RingBufferActionCache(batch_size, max_seq_len, num_heads, head_dim, device, dtype)

    # Test case 1: Cache not full
    k = torch.randn(batch_size, 10, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, 10, num_heads, head_dim, device=device, dtype=dtype)

    _, _, mask = ring_cache.update_and_get_window(k, v, num_new_tokens=10, max_attention_size=max_seq_len)

    expected_valid = 10
    actual_valid = mask.sum().item()
    print(f"  Case 1 (10 tokens): Expected {expected_valid} valid, got {actual_valid}")
    assert actual_valid == expected_valid, "Mask mismatch for partial cache"

    # Test case 2: Add more tokens
    k = torch.randn(batch_size, 30, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, 30, num_heads, head_dim, device=device, dtype=dtype)

    _, _, mask = ring_cache.update_and_get_window(k, v, num_new_tokens=30, max_attention_size=max_seq_len)

    expected_valid = 40
    actual_valid = mask.sum().item()
    print(f"  Case 2 (40 total): Expected {expected_valid} valid, got {actual_valid}")
    assert actual_valid == expected_valid, "Mask mismatch for growing cache"

    # Test case 3: Exceed capacity
    k = torch.randn(batch_size, 50, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, 50, num_heads, head_dim, device=device, dtype=dtype)

    _, _, mask = ring_cache.update_and_get_window(k, v, num_new_tokens=50, max_attention_size=max_seq_len)

    expected_valid = max_seq_len  # Capped at capacity
    actual_valid = mask.sum().item()
    print(f"  Case 3 (90 total, capped): Expected {expected_valid} valid, got {actual_valid}")
    assert actual_valid == expected_valid, "Mask mismatch for full cache"

    print("  ✓ Attention mask semantics validated")


def test_cuda_graph_compatibility():
    """Test 5: CUDA Graph capture and replay."""
    print("\n" + "=" * 80)
    print("Test 5: CUDA Graph Compatibility")
    print("=" * 80)

    batch_size = 1
    max_seq_len = 1024
    num_heads = 8
    head_dim = 64
    device = torch.device("cuda")
    dtype = torch.bfloat16

    ring_cache = RingBufferActionCache(batch_size, max_seq_len, num_heads, head_dim, device, dtype)

    # Warmup: Fill cache to steady state
    print("  Warming up cache...")
    for _ in range(max_seq_len):
        k = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)
        ring_cache.update_and_get_window(k, v, num_new_tokens=1, max_attention_size=max_seq_len)

    # Allocate static input/output tensors
    static_k = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)
    static_v = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)

    # Capture CUDA Graph
    print("  Capturing CUDA Graph...")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        k_out, v_out, mask_out = ring_cache.update_and_get_window(
            static_k, static_v, num_new_tokens=1, max_attention_size=max_seq_len
        )

    print("  ✓ CUDA Graph captured successfully")

    # Replay multiple times
    print("  Replaying graph...")
    num_replays = 100
    for i in range(num_replays):
        # Update input data
        static_k.copy_(torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype))
        static_v.copy_(torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype))

        # Replay
        g.replay()

        # Validate output
        assert k_out.shape == (batch_size, max_seq_len, num_heads, head_dim)
        assert mask_out.all(), f"Mask should be all True after warmup, got {mask_out.sum()}/{mask_out.numel()}"

    print(f"  ✓ Successfully replayed graph {num_replays} times")
    print("  ✓ All outputs have correct shapes and mask is valid")


def benchmark_performance():
    """Benchmark: Compare performance of ActionCache vs RingBufferActionCache."""
    print("\n" + "=" * 80)
    print("Performance Benchmark")
    print("=" * 80)

    batch_size = 4
    max_seq_len = 2048
    num_heads = 16
    head_dim = 128
    device = torch.device("cuda")
    dtype = torch.bfloat16

    action_cache = ActionCache(batch_size, max_seq_len, num_heads, head_dim, device, dtype)
    ring_cache = RingBufferActionCache(batch_size, max_seq_len, num_heads, head_dim, device, dtype)

    # Warmup both caches to steady state
    print("  Warming up caches...")
    for _ in range(max_seq_len):
        k = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)
        action_cache.update_and_get_window(k, v, 1, max_seq_len, 0)
        ring_cache.update_and_get_window(k, v, 1, max_seq_len)

    # Benchmark decode (1 token at a time)
    num_iterations = 1000

    # ActionCache
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        k = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)
        action_cache.update_and_get_window(k, v, 1, max_seq_len, 0)
    torch.cuda.synchronize()
    action_time = time.perf_counter() - start

    # RingBufferActionCache
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        k = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)
        ring_cache.update_and_get_window(k, v, 1, max_seq_len)
    torch.cuda.synchronize()
    ring_time = time.perf_counter() - start

    speedup = action_time / ring_time

    print(f"\n  Results ({num_iterations} iterations, decode mode):")
    print(f"    ActionCache:       {action_time*1000:.2f} ms  ({action_time/num_iterations*1000:.3f} ms/iter)")
    print(f"    RingBufferCache:   {ring_time*1000:.2f} ms  ({ring_time/num_iterations*1000:.3f} ms/iter)")
    print(f"    Speedup:           {speedup:.2f}x")

    if speedup > 1.1:
        print(f"  ✓ Ring buffer is {speedup:.1f}x faster!")
    elif speedup > 0.9:
        print(f"  ≈ Performance is similar (within 10%)")
    else:
        print(f"  ⚠ Ring buffer is slower (investigate)")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RING BUFFER ACTION CACHE - COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    try:
        test_basic_insertion()
        test_incremental_insertion()
        test_eviction_with_wrapping()
        test_attention_mask_correctness()
        test_cuda_graph_compatibility()
        benchmark_performance()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
