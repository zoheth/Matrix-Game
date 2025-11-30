"""
Test to verify that the Triton kernel replacement in KeyboardInjector
produces identical results to the original implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import copy

# Import the components we need to test
from wan.modules.modular_action.injectors import KeyboardInjector
from wan.modules.modular_action.action_config import ActionConfig
from wan.modules.modular_action.kernels.before_attn_kernel import update_kv_cache_triton
from wan.modules.modular_action.kernels.kv_cache_kernel import update_kv_cache_optimized


def create_test_kv_cache(batch_size: int, cache_size: int, num_heads: int, head_dim: int, device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """Create a test KV cache with proper structure."""
    return {
        "k": torch.zeros([batch_size, cache_size, num_heads, head_dim], dtype=torch.float16, device=device),
        "v": torch.zeros([batch_size, cache_size, num_heads, head_dim], dtype=torch.float16, device=device),
        "global_end_index": torch.tensor([0], dtype=torch.int64, device=device),
        "local_end_index": torch.tensor([0], dtype=torch.int64, device=device),
    }


def test_triton_vs_original_single_step():
    """
    Test that the Triton kernel produces identical results to the original
    implementation for a single KV cache update step.
    """
    print("=" * 80)
    print("Test 1: Single Step KV Cache Update - Triton vs Original")
    print("=" * 80)

    # Test parameters
    B = 1  # Batch size
    S = 16  # Spatial dimension (H*W)
    T_k = 2  # Number of new tokens
    num_heads = 16
    head_dim = 64
    cache_size = 100
    max_attention_size = 50
    sink_tokens = 0

    device = 'cuda'
    dtype = torch.float16

    # Create identical input data
    torch.manual_seed(42)
    k_new = torch.randn(B, T_k, num_heads, head_dim, dtype=dtype, device=device)
    v_new = torch.randn(B, T_k, num_heads, head_dim, dtype=dtype, device=device)

    # Expand to spatial dimension for Triton kernel
    k_expanded = k_new.unsqueeze(1).expand(-1, S, -1, -1, -1)
    v_expanded = v_new.unsqueeze(1).expand(-1, S, -1, -1, -1)
    k_for_triton = k_expanded[0]  # [S, T_k, H, D]
    v_for_triton = v_expanded[0]  # [S, T_k, H, D]

    # Create two identical KV caches
    kv_cache_original = create_test_kv_cache(B * S, cache_size, num_heads, head_dim, device)
    kv_cache_triton = create_test_kv_cache(B, cache_size, num_heads, head_dim, device)

    # --- Original implementation ---
    print("\n1. Running original implementation...")
    k_rope_expanded_orig = k_new.unsqueeze(1).expand(-1, S, -1, -1, -1).reshape(B * S, -1, num_heads, head_dim)
    v_expanded_orig = v_new.unsqueeze(1).expand(-1, S, -1, -1, -1).reshape(B * S, -1, num_heads, head_dim)

    # Take mean of first S positions
    k_cache_update_orig = k_rope_expanded_orig[:S].mean(dim=0, keepdim=True)
    v_cache_update_orig = v_expanded_orig[:S].mean(dim=0, keepdim=True)

    print(f"   Original K cache update shape: {k_cache_update_orig.shape}")
    print(f"   Original V cache update shape: {v_cache_update_orig.shape}")

    k_window_orig, v_window_orig, local_start_orig, local_end_orig = update_kv_cache_optimized(
        kv_cache=kv_cache_original,
        k=k_cache_update_orig,
        v=v_cache_update_orig,
        num_new_tokens=T_k,
        max_attention_size=max_attention_size,
        sink_tokens=sink_tokens,
    )

    print(f"   Original K window shape: {k_window_orig.shape}")
    print(f"   Original local_start: {local_start_orig}, local_end: {local_end_orig}")

    # --- Triton implementation ---
    print("\n2. Running Triton implementation...")
    print(f"   Triton K input shape: {k_for_triton.shape}")
    print(f"   Triton V input shape: {v_for_triton.shape}")

    k_window_triton, v_window_triton, local_start_triton, local_end_triton = update_kv_cache_triton(
        kv_cache=kv_cache_triton,
        k_new_source=k_for_triton,  # [S, T_k, H, D]
        v_new_source=v_for_triton,  # [S, T_k, H, D]
        max_attention_size=max_attention_size,
        sink_tokens=sink_tokens,
    )

    print(f"   Triton K window shape: {k_window_triton.shape}")
    print(f"   Triton local_start: {local_start_triton}, local_end: {local_end_triton}")

    # --- Compare results ---
    print("\n3. Comparing results...")

    # Compare indices
    indices_match = (local_start_orig == local_start_triton) and (local_end_orig == local_end_triton)
    print(f"   Indices match: {indices_match}")

    # Compare K window
    k_window_orig_squeezed = k_window_orig.squeeze(0)  # Remove batch dimension for comparison
    k_window_triton_squeezed = k_window_triton.squeeze(0)

    k_max_diff = (k_window_orig_squeezed - k_window_triton_squeezed).abs().max().item()
    k_mean_diff = (k_window_orig_squeezed - k_window_triton_squeezed).abs().mean().item()

    print(f"   K window max difference: {k_max_diff:.6e}")
    print(f"   K window mean difference: {k_mean_diff:.6e}")

    # Compare V window
    v_window_orig_squeezed = v_window_orig.squeeze(0)
    v_window_triton_squeezed = v_window_triton.squeeze(0)

    v_max_diff = (v_window_orig_squeezed - v_window_triton_squeezed).abs().max().item()
    v_mean_diff = (v_window_orig_squeezed - v_window_triton_squeezed).abs().mean().item()

    print(f"   V window max difference: {v_max_diff:.6e}")
    print(f"   V window mean difference: {v_mean_diff:.6e}")

    # Check if results are close (within floating point tolerance)
    tolerance = 1e-3  # Tolerance for FP16
    k_close = k_max_diff < tolerance
    v_close = v_max_diff < tolerance

    print(f"\n   K windows close (tolerance={tolerance}): {k_close}")
    print(f"   V windows close (tolerance={tolerance}): {v_close}")

    success = indices_match and k_close and v_close
    print(f"\n{'✓' if success else '✗'} Test 1: {'PASSED' if success else 'FAILED'}")

    return success


def test_triton_vs_original_multiple_steps():
    """
    Test multiple sequential KV cache updates to verify the Triton kernel
    maintains correct state across multiple steps.
    """
    print("\n" + "=" * 80)
    print("Test 2: Multiple Step KV Cache Update - Triton vs Original")
    print("=" * 80)

    # Test parameters
    B = 1
    S = 16
    num_steps = 5
    tokens_per_step = 2
    num_heads = 16
    head_dim = 64
    cache_size = 100
    max_attention_size = 50
    sink_tokens = 0

    device = 'cuda'
    dtype = torch.float16

    # Create two identical KV caches
    kv_cache_original = create_test_kv_cache(B * S, cache_size, num_heads, head_dim, device)
    kv_cache_triton = create_test_kv_cache(B, cache_size, num_heads, head_dim, device)

    print(f"\nRunning {num_steps} sequential updates with {tokens_per_step} tokens each...")

    all_success = True

    for step in range(num_steps):
        print(f"\n--- Step {step + 1}/{num_steps} ---")

        # Create random input for this step
        torch.manual_seed(42 + step)
        k_new = torch.randn(B, tokens_per_step, num_heads, head_dim, dtype=dtype, device=device)
        v_new = torch.randn(B, tokens_per_step, num_heads, head_dim, dtype=dtype, device=device)

        # Prepare for Triton
        k_expanded = k_new.unsqueeze(1).expand(-1, S, -1, -1, -1)
        v_expanded = v_new.unsqueeze(1).expand(-1, S, -1, -1, -1)
        k_for_triton = k_expanded[0]
        v_for_triton = v_expanded[0]

        # Original implementation
        k_rope_expanded_orig = k_new.unsqueeze(1).expand(-1, S, -1, -1, -1).reshape(B * S, -1, num_heads, head_dim)
        v_expanded_orig = v_new.unsqueeze(1).expand(-1, S, -1, -1, -1).reshape(B * S, -1, num_heads, head_dim)

        k_cache_update_orig = k_rope_expanded_orig[:S].mean(dim=0, keepdim=True)
        v_cache_update_orig = v_expanded_orig[:S].mean(dim=0, keepdim=True)

        k_window_orig, v_window_orig, local_start_orig, local_end_orig = update_kv_cache_optimized(
            kv_cache=kv_cache_original,
            k=k_cache_update_orig,
            v=v_cache_update_orig,
            num_new_tokens=tokens_per_step,
            max_attention_size=max_attention_size,
            sink_tokens=sink_tokens,
        )

        # Triton implementation
        k_window_triton, v_window_triton, local_start_triton, local_end_triton = update_kv_cache_triton(
            kv_cache=kv_cache_triton,
            k_new_source=k_for_triton,
            v_new_source=v_for_triton,
            max_attention_size=max_attention_size,
            sink_tokens=sink_tokens,
        )

        # Compare
        indices_match = (local_start_orig == local_start_triton) and (local_end_orig == local_end_triton)

        k_window_orig_squeezed = k_window_orig.squeeze(0)
        k_window_triton_squeezed = k_window_triton.squeeze(0)
        k_max_diff = (k_window_orig_squeezed - k_window_triton_squeezed).abs().max().item()

        v_window_orig_squeezed = v_window_orig.squeeze(0)
        v_window_triton_squeezed = v_window_triton.squeeze(0)
        v_max_diff = (v_window_orig_squeezed - v_window_triton_squeezed).abs().max().item()

        tolerance = 1e-3
        step_success = indices_match and (k_max_diff < tolerance) and (v_max_diff < tolerance)

        print(f"   local_end: orig={local_end_orig}, triton={local_end_triton}")
        print(f"   K max diff: {k_max_diff:.6e}, V max diff: {v_max_diff:.6e}")
        print(f"   Step result: {'✓ PASS' if step_success else '✗ FAIL'}")

        all_success = all_success and step_success

    print(f"\n{'✓' if all_success else '✗'} Test 2: {'PASSED' if all_success else 'FAILED'}")

    return all_success


def test_keyboard_injector_end_to_end():
    """
    End-to-end test of KeyboardInjector with the Triton kernel replacement.
    This ensures the full forward pass works correctly.
    """
    print("\n" + "=" * 80)
    print("Test 3: KeyboardInjector End-to-End Test")
    print("=" * 80)

    device = 'cuda'
    dtype = torch.float16

    # Create a minimal ActionConfig
    # Note: mouse_head_dim and keyboard_head_dim are computed properties
    config = ActionConfig(
        img_hidden_size=1024,  # 1024 / 16 heads = 64 head_dim
        mouse_hidden_dim=1024,  # 1024 / 16 heads = 64 mouse_head_dim
        keyboard_hidden_dim=1024,  # 1024 / 16 heads = 64 keyboard_head_dim
        mouse_dim_in=2,
        keyboard_dim_in=4,
        hidden_size=128,
        vae_time_compression_ratio=8,
        windows_size=2,
        heads_num=16,
        local_attn_size=100,
        qkv_bias=True,
        qk_norm=True,
    )

    # Create KeyboardInjector
    print("\nCreating KeyboardInjector...")
    injector = KeyboardInjector(config).to(device).to(dtype)
    injector.eval()

    # Create test inputs
    B = 1
    T = 2
    H, W = 4, 4
    S = H * W

    print(f"Test configuration: B={B}, T={T}, H={H}, W={W}, S={S}")

    x = torch.randn(B, T * S, config.img_hidden_size, dtype=dtype, device=device)
    keyboard_condition = torch.randn(B, T * config.vae_time_compression_ratio, config.keyboard_dim_in, dtype=dtype, device=device)

    # Create dummy freqs_cis (not used when RoPE is applied internally)
    freqs_cis = (
        torch.ones(1, 1, dtype=dtype, device=device),
        torch.zeros(1, 1, dtype=dtype, device=device)
    )

    # Create KV cache
    cache_size = config.local_attn_size
    kv_cache = create_test_kv_cache(B, cache_size, config.heads_num, config.keyboard_head_dim, device)

    print("\nRunning forward pass with causal=True and KV cache...")

    try:
        with torch.no_grad():
            output = injector(
                x=x,
                condition=keyboard_condition,
                freqs_cis=freqs_cis,
                spatial_shape=(H, W),
                temporal_shape=T,
                is_causal=True,
                kv_cache=kv_cache,
                start_frame=0,
                num_frame_per_block=T,
                block_mask=None,
            )

        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        print(f"   Output device: {output.device}")

        # Check output properties
        shape_correct = output.shape == x.shape
        dtype_correct = output.dtype == dtype
        device_correct = output.device == x.device
        no_nans = not torch.isnan(output).any()
        no_infs = not torch.isinf(output).any()

        print(f"\n   Shape correct: {shape_correct}")
        print(f"   Dtype correct: {dtype_correct}")
        print(f"   Device correct: {device_correct}")
        print(f"   No NaNs: {no_nans}")
        print(f"   No Infs: {no_infs}")

        # Check that cache was updated
        cache_updated = kv_cache["local_end_index"].item() > 0
        print(f"   Cache updated: {cache_updated} (local_end_index={kv_cache['local_end_index'].item()})")

        success = shape_correct and dtype_correct and device_correct and no_nans and no_infs and cache_updated

        print(f"\n{'✓' if success else '✗'} Test 3: {'PASSED' if success else 'FAILED'}")

        return success

    except Exception as e:
        print(f"\n✗ Test 3: FAILED with exception:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_eviction():
    """
    Test that cache eviction works correctly when cache fills up.
    """
    print("\n" + "=" * 80)
    print("Test 4: Cache Eviction Test")
    print("=" * 80)

    # Test parameters - small cache to trigger eviction
    B = 1
    S = 16
    num_heads = 16
    head_dim = 64
    cache_size = 10  # Small cache
    max_attention_size = 8
    sink_tokens = 2

    device = 'cuda'
    dtype = torch.float16

    print(f"\nConfiguration: cache_size={cache_size}, max_attention_size={max_attention_size}, sink_tokens={sink_tokens}")

    # Create KV caches
    kv_cache_original = create_test_kv_cache(B * S, cache_size, num_heads, head_dim, device)
    kv_cache_triton = create_test_kv_cache(B, cache_size, num_heads, head_dim, device)

    # Add tokens until we trigger eviction
    num_steps = 8
    tokens_per_step = 2

    print(f"Adding {num_steps} steps × {tokens_per_step} tokens = {num_steps * tokens_per_step} total tokens")
    print(f"This will exceed cache_size={cache_size} and trigger eviction\n")

    all_success = True

    for step in range(num_steps):
        torch.manual_seed(100 + step)
        k_new = torch.randn(B, tokens_per_step, num_heads, head_dim, dtype=dtype, device=device)
        v_new = torch.randn(B, tokens_per_step, num_heads, head_dim, dtype=dtype, device=device)

        # Prepare inputs
        k_expanded = k_new.unsqueeze(1).expand(-1, S, -1, -1, -1)
        v_expanded = v_new.unsqueeze(1).expand(-1, S, -1, -1, -1)
        k_for_triton = k_expanded[0]
        v_for_triton = v_expanded[0]

        # Original
        k_rope_expanded_orig = k_new.unsqueeze(1).expand(-1, S, -1, -1, -1).reshape(B * S, -1, num_heads, head_dim)
        v_expanded_orig = v_new.unsqueeze(1).expand(-1, S, -1, -1, -1).reshape(B * S, -1, num_heads, head_dim)
        k_cache_update_orig = k_rope_expanded_orig[:S].mean(dim=0, keepdim=True)
        v_cache_update_orig = v_expanded_orig[:S].mean(dim=0, keepdim=True)

        k_window_orig, v_window_orig, local_start_orig, local_end_orig = update_kv_cache_optimized(
            kv_cache=kv_cache_original,
            k=k_cache_update_orig,
            v=v_cache_update_orig,
            num_new_tokens=tokens_per_step,
            max_attention_size=max_attention_size,
            sink_tokens=sink_tokens,
        )

        # Triton
        k_window_triton, v_window_triton, local_start_triton, local_end_triton = update_kv_cache_triton(
            kv_cache=kv_cache_triton,
            k_new_source=k_for_triton,
            v_new_source=v_for_triton,
            max_attention_size=max_attention_size,
            sink_tokens=sink_tokens,
        )

        # Compare
        indices_match = (local_start_orig == local_start_triton) and (local_end_orig == local_end_triton)

        k_max_diff = (k_window_orig.squeeze(0) - k_window_triton.squeeze(0)).abs().max().item()
        v_max_diff = (v_window_orig.squeeze(0) - v_window_triton.squeeze(0)).abs().max().item()

        tolerance = 1e-3
        step_success = indices_match and (k_max_diff < tolerance) and (v_max_diff < tolerance)

        evicted = local_end_orig >= cache_size

        print(f"Step {step + 1}: local_end={local_end_orig}, window_size={k_window_orig.size(1)}, " +
              f"evicted={evicted}, K_diff={k_max_diff:.6e}, {'✓' if step_success else '✗'}")

        all_success = all_success and step_success

    print(f"\n{'✓' if all_success else '✗'} Test 4: {'PASSED' if all_success else 'FAILED'}")

    return all_success


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TESTING KEYBOARD INJECTOR TRITON KERNEL REPLACEMENT")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("\n✗ CUDA is not available. Tests require GPU.")
        return False

    print(f"\nCUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch Version: {torch.__version__}")

    results = []

    # Run all tests
    try:
        results.append(("Single Step Update", test_triton_vs_original_single_step()))
    except Exception as e:
        print(f"\n✗ Test 1 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Single Step Update", False))

    try:
        results.append(("Multiple Step Update", test_triton_vs_original_multiple_steps()))
    except Exception as e:
        print(f"\n✗ Test 2 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Multiple Step Update", False))

    try:
        results.append(("End-to-End Forward", test_keyboard_injector_end_to_end()))
    except Exception as e:
        print(f"\n✗ Test 3 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("End-to-End Forward", False))

    try:
        results.append(("Cache Eviction", test_cache_eviction()))
    except Exception as e:
        print(f"\n✗ Test 4 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Cache Eviction", False))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status:12} - {name}")

    all_passed = all(success for _, success in results)

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
