"""
Test script for PagedCache integration with FlashInfer.

This script tests the PagedCache and FlashInferCausalSelfAttention integration
without requiring the full model weights.
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_paged_cache_basic():
    """Test basic PagedCache operations."""
    print("=" * 60)
    print("Test 1: Basic PagedCache Operations")
    print("=" * 60)

    from wan.modules.paged_cache import PagedCache

    # Create a small cache for testing
    cache = PagedCache(
        max_total_tokens=1024,
        page_size=16,
        num_heads=12,
        head_dim=128,
        sink_size=0,
        dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Created cache: {cache}")

    # Test append
    seq_len = 880  # One frame
    k = torch.randn(seq_len, 12, 128, dtype=torch.bfloat16, device=cache.device)
    v = torch.randn(seq_len, 12, 128, dtype=torch.bfloat16, device=cache.device)

    cache.append(k, v)
    print(f"After append: {cache}")

    # Test get_kv_for_attention
    k_out, v_out = cache.get_kv_for_attention()
    assert k_out.shape == (seq_len, 12, 128), f"Expected shape {(seq_len, 12, 128)}, got {k_out.shape}"
    print(f"get_kv_for_attention: k_out.shape={k_out.shape}, v_out.shape={v_out.shape}")

    # Test get_flashinfer_meta
    indices, indptr, last_page_len = cache.get_flashinfer_meta(cache.device)
    print(f"FlashInfer meta: indices={indices}, indptr={indptr}, last_page_len={last_page_len}")

    # Test reset
    cache.reset()
    print(f"After reset: {cache}")
    assert cache.seq_len == 0, "seq_len should be 0 after reset"

    print("✓ Basic PagedCache operations passed!")
    return True


def test_paged_cache_eviction():
    """Test PagedCache eviction behavior with page recycling."""
    print("\n" + "=" * 60)
    print("Test 2: PagedCache Eviction with Page Recycling")
    print("=" * 60)

    from wan.modules.paged_cache import PagedCache

    cache = PagedCache(
        max_total_tokens=256,
        page_size=16,
        num_heads=4,
        head_dim=64,
        sink_size=0,
        dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Fill cache with multiple appends, evicting before each to stay within limits
    max_allowed = 128  # Keep only 128 tokens
    for i in range(20):  # More iterations to test recycling
        # Evict BEFORE append to ensure space
        evicted = cache.evict(max_allowed)
        if evicted > 0:
            print(f"  Evicted {evicted} tokens, free_pool size: {len(cache.free_page_pool)}")

        k = torch.randn(32, 4, 64, dtype=torch.bfloat16, device=cache.device)
        v = torch.randn(32, 4, 64, dtype=torch.bfloat16, device=cache.device)
        cache.append(k, v)
        print(f"After append {i+1}: seq_len={cache.seq_len}, pages={len(cache.active_page_indices)}, free_pool={len(cache.free_page_pool)}")

    # Final eviction
    evicted = cache.evict(64)
    print(f"Final eviction: {evicted} tokens, now seq_len={cache.seq_len}")

    # Verify we can still append after eviction (using recycled pages)
    k = torch.randn(32, 4, 64, dtype=torch.bfloat16, device=cache.device)
    v = torch.randn(32, 4, 64, dtype=torch.bfloat16, device=cache.device)
    cache.append(k, v)
    print(f"After final append: seq_len={cache.seq_len}, pages={len(cache.active_page_indices)}")

    print("✓ PagedCache eviction with recycling test passed!")
    return True


def test_flashinfer_attention_with_paged_cache():
    """Test FlashInferCausalSelfAttention with PagedCache."""
    print("\n" + "=" * 60)
    print("Test 3: FlashInferCausalSelfAttention with PagedCache")
    print("=" * 60)

    try:
        import flashinfer
        print(f"FlashInfer available: version={flashinfer.__version__}")
    except ImportError:
        print("FlashInfer not available, skipping this test")
        return True

    from wan.modules.flashinfer_attention import FlashInferCausalSelfAttention
    from wan.modules.paged_cache import PagedCache

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Create attention module
    attn = FlashInferCausalSelfAttention(
        dim=1536,
        num_heads=12,
        local_attn_size=15,
        sink_size=0,
        qk_norm=True,
        eps=1e-6,
        height=22,
        width=40,
        page_size=16,
    ).to(device=device, dtype=dtype)

    # Create PagedCache
    cache = PagedCache(
        max_total_tokens=15 * 22 * 40,  # 15 frames * H * W
        page_size=16,
        num_heads=12,
        head_dim=128,
        sink_size=0,
        dtype=dtype,
        device=device,
    )

    # Create mock freqs (RoPE frequencies)
    freqs = torch.randn(1024, 64, dtype=torch.complex128, device=device)

    # Test forward pass with PagedCache
    batch_size = 1
    seq_len = 22 * 40  # One frame
    x = torch.randn(batch_size, seq_len, 1536, dtype=dtype, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)
    grid_sizes = torch.tensor([1, 22, 40], dtype=torch.long, device=device)

    print(f"Input shape: {x.shape}")
    print(f"PagedCache before forward: {cache}")

    # Forward pass
    output = attn(
        x=x,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=freqs,
        block_mask=None,
        kv_cache=cache,
        current_start=0,
    )

    print(f"Output shape: {output.shape}")
    print(f"PagedCache after forward: {cache}")

    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} != {x.shape}"

    print("✓ FlashInferCausalSelfAttention with PagedCache test passed!")
    return True


def test_paged_cache_denoising_simulation():
    """
    Test PagedCache behavior during simulated denoising loop.

    This test simulates what happens during actual inference:
    - Multiple forward passes with the same current_start (denoising steps)
    - Each step should OVERWRITE, not APPEND

    This is the ROOT CAUSE test for the bug where paged cache produces
    wrong results in full inference.
    """
    print("\n" + "=" * 60)
    print("Test 5a: PagedCache Denoising Simulation")
    print("=" * 60)

    try:
        import flashinfer
        print(f"FlashInfer available: version={flashinfer.__version__}")
    except ImportError:
        print("FlashInfer not available, skipping this test")
        return True

    from wan.modules.flashinfer_attention import FlashInferCausalSelfAttention
    from wan.modules.paged_cache import PagedCache
    from wan.modules.model import rope_params

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Dimensions
    dim = 1536
    num_heads = 12
    head_dim = dim // num_heads
    height, width = 22, 40
    frame_seq_len = height * width  # 880

    # Create attention module
    torch.manual_seed(42)
    attn = FlashInferCausalSelfAttention(
        dim=dim,
        num_heads=num_heads,
        local_attn_size=15,
        sink_size=0,
        qk_norm=True,
        eps=1e-6,
        height=height,
        width=width,
        page_size=16,
    ).to(device=device, dtype=dtype)

    # Create RoPE frequencies
    freqs = rope_params(1024, head_dim).to(device)

    # Create PagedCache
    paged_cache = PagedCache(
        max_total_tokens=15 * frame_seq_len,
        page_size=16,
        num_heads=num_heads,
        head_dim=head_dim,
        sink_size=0,
        dtype=dtype,
        device=device,
    )

    # Simulate denoising: multiple passes for the SAME frame
    current_start = 0  # Same position for all denoising steps
    num_denoising_steps = 5  # Simulate 5 denoising steps

    print(f"\nSimulating {num_denoising_steps} denoising steps for frame at position {current_start}...")

    for step in range(num_denoising_steps):
        torch.manual_seed(200 + step)  # Different input each step
        x = torch.randn(1, frame_seq_len, dim, dtype=dtype, device=device)
        seq_lens = torch.tensor([frame_seq_len], dtype=torch.long, device=device)
        grid_sizes = torch.tensor([1, height, width], dtype=torch.long, device=device)

        output = attn(
            x=x,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            block_mask=None,
            kv_cache=paged_cache,
            current_start=current_start,
        )

        print(f"  Step {step}: cache seq_len={paged_cache.seq_len}, pages={len(paged_cache.active_page_indices)}")

    # Check: after N denoising steps for the same frame,
    # cache should only have 1 frame worth of tokens, not N frames!
    expected_seq_len = frame_seq_len
    actual_seq_len = paged_cache.seq_len

    print(f"\nExpected cache seq_len: {expected_seq_len}")
    print(f"Actual cache seq_len: {actual_seq_len}")

    if actual_seq_len == expected_seq_len:
        print("✓ PagedCache correctly overwrites during denoising!")
        return True
    else:
        print(f"✗ BUG DETECTED: Cache has {actual_seq_len} tokens instead of {expected_seq_len}")
        print(f"  This means PagedCache is appending instead of overwriting during denoising!")
        return False


def test_paged_cache_correctness():
    """
    Test numerical correctness of PagedCache vs dict-based cache.

    This is the CRITICAL test: compares outputs between:
    1. FlashInferCausalSelfAttention with PagedCache
    2. FlashInferCausalSelfAttention with dict-based cache (legacy mode)

    Both should produce identical (or very close) outputs.
    """
    print("\n" + "=" * 60)
    print("Test 5b: PagedCache vs Dict Cache Correctness")
    print("=" * 60)

    try:
        import flashinfer
        print(f"FlashInfer available: version={flashinfer.__version__}")
    except ImportError:
        print("FlashInfer not available, skipping this test")
        return True

    from wan.modules.flashinfer_attention import FlashInferCausalSelfAttention
    from wan.modules.paged_cache import PagedCache
    from wan.modules.model import rope_params

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Dimensions
    dim = 1536
    num_heads = 12
    head_dim = dim // num_heads
    height, width = 22, 40
    frame_seq_len = height * width  # 880
    local_attn_size = 15
    max_cache_tokens = local_attn_size * frame_seq_len

    # Create two identical attention modules
    torch.manual_seed(42)
    attn_paged = FlashInferCausalSelfAttention(
        dim=dim,
        num_heads=num_heads,
        local_attn_size=local_attn_size,
        sink_size=0,
        qk_norm=True,
        eps=1e-6,
        height=height,
        width=width,
        page_size=16,
    ).to(device=device, dtype=dtype)

    # Clone for dict-based cache
    attn_dict = FlashInferCausalSelfAttention(
        dim=dim,
        num_heads=num_heads,
        local_attn_size=local_attn_size,
        sink_size=0,
        qk_norm=True,
        eps=1e-6,
        height=height,
        width=width,
        page_size=16,
    ).to(device=device, dtype=dtype)
    attn_dict.load_state_dict(attn_paged.state_dict())

    # Create RoPE frequencies (proper format)
    freqs = rope_params(1024, head_dim).to(device)

    # Create PagedCache
    paged_cache = PagedCache(
        max_total_tokens=max_cache_tokens,
        page_size=16,
        num_heads=num_heads,
        head_dim=head_dim,
        sink_size=0,
        dtype=dtype,
        device=device,
    )

    # Create dict-based cache
    dict_cache = {
        "k": torch.zeros(1, max_cache_tokens, num_heads, head_dim, dtype=dtype, device=device),
        "v": torch.zeros(1, max_cache_tokens, num_heads, head_dim, dtype=dtype, device=device),
        "local_start_index": torch.tensor([0], device=device),
        "local_end_index": torch.tensor([0], device=device),
        "global_end_index": torch.tensor([0], device=device),
    }

    # Process multiple frames and compare outputs
    num_test_frames = 5
    max_diff_overall = 0.0
    all_close = True

    print(f"\nProcessing {num_test_frames} frames...")

    for frame_idx in range(num_test_frames):
        current_start = frame_idx * frame_seq_len

        # Same input for both
        torch.manual_seed(100 + frame_idx)
        x = torch.randn(1, frame_seq_len, dim, dtype=dtype, device=device)
        seq_lens = torch.tensor([frame_seq_len], dtype=torch.long, device=device)
        grid_sizes = torch.tensor([1, height, width], dtype=torch.long, device=device)

        # Forward with PagedCache
        output_paged = attn_paged(
            x=x,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            block_mask=None,
            kv_cache=paged_cache,
            current_start=current_start,
        )

        # Forward with dict cache
        output_dict = attn_dict(
            x=x,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            block_mask=None,
            kv_cache=dict_cache,
            current_start=current_start,
        )

        # Compare outputs
        diff = (output_paged - output_dict).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        max_diff_overall = max(max_diff_overall, max_diff)

        # Check if outputs are close (tolerance for bf16)
        rtol, atol = 1e-2, 1e-2
        is_close = torch.allclose(output_paged, output_dict, rtol=rtol, atol=atol)

        print(f"  Frame {frame_idx}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, close={is_close}")

        if not is_close:
            all_close = False
            # Print more details for debugging
            print(f"    PagedCache seq_len: {paged_cache.seq_len}")
            print(f"    Dict cache local_end_index: {dict_cache['local_end_index'].item()}")
            print(f"    Output paged stats: min={output_paged.min():.4f}, max={output_paged.max():.4f}, mean={output_paged.mean():.4f}")
            print(f"    Output dict stats: min={output_dict.min():.4f}, max={output_dict.max():.4f}, mean={output_dict.mean():.4f}")

    print(f"\nOverall max diff: {max_diff_overall:.6f}")

    if all_close:
        print("✓ PagedCache correctness test PASSED!")
        return True
    else:
        print("✗ PagedCache correctness test FAILED!")
        print("  Outputs differ significantly between PagedCache and dict-based cache")
        return False


def test_cache_manager_paged_mode():
    """Test CacheManager in paged cache mode."""
    print("\n" + "=" * 60)
    print("Test 4: CacheManager Paged Mode")
    print("=" * 60)

    from pipeline.cache_manager import CacheManager, PAGED_CACHE_AVAILABLE
    from pipeline.config import ModelConfig, CacheConfig

    if not PAGED_CACHE_AVAILABLE:
        print("PagedCache not available, skipping this test")
        return True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    model_config = ModelConfig(
        num_transformer_blocks=30,
        frame_seq_length=880,
        num_attention_heads=12,
        head_dim=128,
    )

    cache_config = CacheConfig(local_attn_size=15)

    # Test with paged cache enabled
    manager = CacheManager(
        model_config=model_config,
        cache_config=cache_config,
        device=device,
        dtype=dtype,
        use_paged_cache=True,
        page_size=16,
    )

    manager.initialize_all_caches(batch_size=1)

    visual_cache, mouse_cache, keyboard_cache, crossattn_cache = manager.get_caches()

    print(f"Visual cache type: {type(visual_cache[0])}")
    print(f"Number of visual cache layers: {len(visual_cache)}")
    print(f"First visual cache: {visual_cache[0]}")

    # Test reset
    manager.reset_all_caches()
    print("Reset successful!")

    print("✓ CacheManager paged mode test passed!")
    return True


def main():
    """Run all tests."""
    print("Starting PagedCache Integration Tests")
    print("=" * 60)

    results = []

    # Test 1: Basic operations
    try:
        results.append(("Basic PagedCache Operations", test_paged_cache_basic()))
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Basic PagedCache Operations", False))

    # Test 2: Eviction
    try:
        results.append(("PagedCache Eviction", test_paged_cache_eviction()))
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("PagedCache Eviction", False))

    # Test 3: FlashInfer attention (requires CUDA)
    if torch.cuda.is_available():
        try:
            results.append(("FlashInfer Attention", test_flashinfer_attention_with_paged_cache()))
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append(("FlashInfer Attention", False))
    else:
        print("\nSkipping FlashInfer attention test (CUDA not available)")

    # Test 4: CacheManager
    try:
        results.append(("CacheManager Paged Mode", test_cache_manager_paged_mode()))
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("CacheManager Paged Mode", False))

    # Test 5a: Denoising simulation (ROOT CAUSE TEST)
    if torch.cuda.is_available():
        try:
            results.append(("PagedCache Denoising Simulation", test_paged_cache_denoising_simulation()))
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append(("PagedCache Denoising Simulation", False))
    else:
        print("\nSkipping PagedCache denoising test (CUDA not available)")

    # Test 5b: Correctness verification
    if torch.cuda.is_available():
        try:
            results.append(("PagedCache Correctness", test_paged_cache_correctness()))
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append(("PagedCache Correctness", False))
    else:
        print("\nSkipping PagedCache correctness test (CUDA not available)")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
