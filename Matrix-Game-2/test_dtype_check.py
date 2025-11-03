"""Check for dtype-related issues in mouse_preprocessor_triton"""
import torch
from wan.modules.modular_action.injectors import MousePreprocessor
from wan.modules.modular_action.kernels.preprocessor_kernel import mouse_preprocessor_triton

def test_dtype_consistency():
    """Test with different dtypes"""
    V, W = 4, 3
    B, N_frames, C_mouse = 1, 48, 2
    BS, T_q, C_hidden = 880, 3, 1536

    dtypes = [torch.float32, torch.float16]

    for dtype in dtypes:
        print(f"\n{'='*60}")
        print(f"Testing with dtype: {dtype}")
        print(f"{'='*60}")

        # Create test data
        hidden_states = torch.randn(BS, T_q, C_hidden, device='cuda', dtype=dtype)
        mouse_condition = torch.randn(B, N_frames, C_mouse, device='cuda', dtype=dtype)

        # PyTorch reference
        preprocessor = MousePreprocessor(V, W)
        expected = preprocessor.forward(hidden_states, mouse_condition, is_causal=True, num_frame_per_block=3)

        # Triton implementation
        actual = mouse_preprocessor_triton(hidden_states, mouse_condition, V, W, is_causal=True, num_frame_per_block=3)

        # Check dtypes
        print(f"Input dtype: {hidden_states.dtype}")
        print(f"Expected output dtype: {expected.dtype}")
        print(f"Actual output dtype: {actual.dtype}")

        # Check for NaN/Inf
        has_nan_expected = torch.isnan(expected).any()
        has_nan_actual = torch.isnan(actual).any()
        has_inf_expected = torch.isinf(expected).any()
        has_inf_actual = torch.isinf(actual).any()

        print(f"\nNaN/Inf check:")
        print(f"  Expected has NaN: {has_nan_expected}")
        print(f"  Actual has NaN: {has_nan_actual}")
        print(f"  Expected has Inf: {has_inf_expected}")
        print(f"  Actual has Inf: {has_inf_actual}")

        # Compare with appropriate tolerance
        rtol, atol = (1e-3, 1e-3) if dtype == torch.float16 else (1e-5, 1e-5)

        hidden_match = torch.allclose(expected[:, :, :C_hidden], actual[:, :, :C_hidden], rtol=rtol, atol=atol)
        mouse_match = torch.allclose(expected[:, :, C_hidden:], actual[:, :, C_hidden:], rtol=rtol, atol=atol)

        print(f"\nComparison (rtol={rtol}, atol={atol}):")
        print(f"  Hidden part match: {hidden_match}")
        print(f"  Mouse part match: {mouse_match}")

        if not mouse_match:
            diff = (expected[:, :, C_hidden:] - actual[:, :, C_hidden:]).abs()
            print(f"  Max diff: {diff.max().item()}")
            print(f"  Mean diff: {diff.mean().item()}")
            print(f"  Median diff: {diff.median().item()}")

            # Show distribution of differences
            print(f"\n  Diff percentiles:")
            for p in [50, 90, 95, 99, 99.9, 100]:
                val = torch.quantile(diff.float(), p/100)
                print(f"    {p}th percentile: {val.item():.6e}")

def test_edge_cases():
    """Test edge cases that might expose dtype issues"""
    print(f"\n{'='*60}")
    print("Testing edge cases")
    print(f"{'='*60}")

    V, W = 4, 3
    dtype = torch.float16

    # Test case 1: Very small batch
    print("\nCase 1: Small batch")
    BS, T_q, C_hidden = 1, 3, 1536
    B, N_frames, C_mouse = 1, 48, 2

    hidden_states = torch.randn(BS, T_q, C_hidden, device='cuda', dtype=dtype)
    mouse_condition = torch.randn(B, N_frames, C_mouse, device='cuda', dtype=dtype)

    preprocessor = MousePreprocessor(V, W)
    expected = preprocessor.forward(hidden_states, mouse_condition, is_causal=True, num_frame_per_block=3)
    actual = mouse_preprocessor_triton(hidden_states, mouse_condition, V, W, is_causal=True, num_frame_per_block=3)

    match = torch.allclose(expected, actual, rtol=1e-3, atol=1e-3)
    print(f"  Match: {match}")
    if not match:
        print(f"  Max diff: {(expected - actual).abs().max().item()}")

    # Test case 2: Minimum frames
    print("\nCase 2: Minimum frames")
    N_frames = 12  # Minimum for causal with num_frame_per_block=3
    BS, T_q, C_hidden = 880, 3, 1536

    hidden_states = torch.randn(BS, T_q, C_hidden, device='cuda', dtype=dtype)
    mouse_condition = torch.randn(B, N_frames, C_mouse, device='cuda', dtype=dtype)

    expected = preprocessor.forward(hidden_states, mouse_condition, is_causal=True, num_frame_per_block=3)
    actual = mouse_preprocessor_triton(hidden_states, mouse_condition, V, W, is_causal=True, num_frame_per_block=3)

    match = torch.allclose(expected, actual, rtol=1e-3, atol=1e-3)
    print(f"  Match: {match}")
    if not match:
        print(f"  Max diff: {(expected - actual).abs().max().item()}")

def test_full_injector():
    """Test the full MouseInjector with Triton kernel"""
    print(f"\n{'='*60}")
    print("Testing full MouseInjector")
    print(f"{'='*60}")

    from wan.modules.modular_action.injectors import MouseInjector
    from wan.modules.modular_action.action_config import ActionConfig

    config = ActionConfig(
        img_hidden_size=1536,
        mouse_dim_in=2,
        keyboard_dim_in=4,
        mouse_hidden_dim=1024,
        keyboard_hidden_dim=1024,
        heads_num=16,
        vae_time_compression_ratio=4,
        windows_size=3,
        local_attn_size=6,
        qkv_bias=False,
        qk_norm=True,
    )

    for dtype in [torch.float32, torch.float16]:
        print(f"\nTesting with dtype: {dtype}")

        injector = MouseInjector(config).to('cuda')
        if dtype == torch.float16:
            injector = injector.half()
        injector.eval()

        B, T, H, W = 1, 3, 22, 40
        S = H * W
        x = torch.randn(B, T * S, config.img_hidden_size, device='cuda', dtype=dtype)
        condition = torch.randn(B, 48, config.mouse_dim_in, device='cuda', dtype=dtype)

        head_dim = config.mouse_head_dim
        freqs_cis = (
            torch.randn(100, head_dim, device='cuda', dtype=dtype),
            torch.randn(100, head_dim, device='cuda', dtype=dtype),
        )

        try:
            with torch.no_grad():
                output = injector(
                    x, condition, freqs_cis,
                    spatial_shape=(H, W),
                    temporal_shape=T,
                    is_causal=True,
                    num_frame_per_block=3,
                )

            print(f"  ✓ Success! Output shape: {output.shape}, dtype: {output.dtype}")

            # Check for NaN/Inf
            if torch.isnan(output).any():
                print(f"  ✗ WARNING: Output contains NaN")
            if torch.isinf(output).any():
                print(f"  ✗ WARNING: Output contains Inf")

        except Exception as e:
            print(f"  ✗ Failed with error: {e}")

if __name__ == "__main__":
    test_dtype_consistency()
    test_edge_cases()
    test_full_injector()
