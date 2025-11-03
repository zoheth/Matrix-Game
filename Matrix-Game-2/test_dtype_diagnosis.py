"""Diagnose dtype-related issues in mouse_preprocessor_triton"""
import torch
import sys

print("=== Environment Check ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Try to import the modules
try:
    from wan.modules.modular_action.injectors import MousePreprocessor
    from wan.modules.modular_action.kernels.preprocessor_kernel import mouse_preprocessor_triton
    print("\n✓ Modules imported successfully")
except ImportError as e:
    print(f"\n✗ Import failed: {e}")
    sys.exit(1)

def test_dtype_float16():
    """Test with float16 (actual inference dtype)"""
    print("\n=== Testing with float16 (inference dtype) ===")

    B, N_frames, C_mouse = 1, 48, 2
    BS, T_q, C_hidden = 880, 3, 1536  # Real inference shapes
    V, W = 4, 3
    num_frame_per_block = 3

    # Create test data in float16
    hidden_states = torch.randn(BS, T_q, C_hidden, device='cuda', dtype=torch.float16)
    mouse_condition = torch.randn(B, N_frames, C_mouse, device='cuda', dtype=torch.float16)

    # PyTorch reference
    preprocessor = MousePreprocessor(V, W)
    expected = preprocessor.forward(hidden_states, mouse_condition, is_causal=True, num_frame_per_block=num_frame_per_block)

    # Triton implementation
    actual = mouse_preprocessor_triton(hidden_states, mouse_condition, V, W, is_causal=True, num_frame_per_block=num_frame_per_block)

    print(f"Expected dtype: {expected.dtype}")
    print(f"Actual dtype: {actual.dtype}")
    print(f"Expected shape: {expected.shape}")
    print(f"Actual shape: {actual.shape}")

    # Compare hidden states (should be identical)
    hidden_match = torch.allclose(expected[:, :, :C_hidden], actual[:, :, :C_hidden], rtol=1e-3, atol=1e-3)
    print(f"\nHidden states match: {hidden_match}")

    # Compare mouse parts
    mouse_expected = expected[:, :, C_hidden:]
    mouse_actual = actual[:, :, C_hidden:]

    diff = (mouse_expected - mouse_actual).abs()
    print(f"\nMouse part comparison:")
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")
    print(f"  90th percentile diff: {torch.quantile(diff, 0.9).item():.6f}")

    mouse_match = torch.allclose(mouse_expected, mouse_actual, rtol=1e-3, atol=1e-3)
    print(f"  Match (rtol=1e-3): {mouse_match}")

    if not mouse_match:
        # Find where differences occur
        mismatch_mask = diff > 1e-3
        num_mismatches = mismatch_mask.sum().item()
        total_elements = mouse_expected.numel()
        print(f"\n  Mismatches: {num_mismatches} / {total_elements} ({100*num_mismatches/total_elements:.2f}%)")

        # Show first few mismatches
        mismatch_indices = torch.nonzero(mismatch_mask)[:5]
        for idx in mismatch_indices:
            bs, t, c = idx.tolist()
            print(f"    [{bs}, {t}, {c}]: expected={mouse_expected[bs, t, c].item():.6f}, actual={mouse_actual[bs, t, c].item():.6f}")

def test_dtype_float32():
    """Test with float32"""
    print("\n=== Testing with float32 ===")

    B, N_frames, C_mouse = 1, 48, 2
    BS, T_q, C_hidden = 880, 3, 1536
    V, W = 4, 3
    num_frame_per_block = 3

    # Create test data in float32
    hidden_states = torch.randn(BS, T_q, C_hidden, device='cuda', dtype=torch.float32)
    mouse_condition = torch.randn(B, N_frames, C_mouse, device='cuda', dtype=torch.float32)

    # PyTorch reference
    preprocessor = MousePreprocessor(V, W)
    expected = preprocessor.forward(hidden_states, mouse_condition, is_causal=True, num_frame_per_block=num_frame_per_block)

    # Triton implementation
    actual = mouse_preprocessor_triton(hidden_states, mouse_condition, V, W, is_causal=True, num_frame_per_block=num_frame_per_block)

    print(f"Expected dtype: {expected.dtype}")
    print(f"Actual dtype: {actual.dtype}")

    # Compare
    hidden_match = torch.allclose(expected[:, :, :C_hidden], actual[:, :, :C_hidden])
    mouse_match = torch.allclose(expected[:, :, C_hidden:], actual[:, :, C_hidden:])

    print(f"Hidden states match: {hidden_match}")
    print(f"Mouse part match: {mouse_match}")

    if not mouse_match:
        diff = (expected[:, :, C_hidden:] - actual[:, :, C_hidden:]).abs()
        print(f"  Max diff: {diff.max().item():.6e}")
        print(f"  Mean diff: {diff.mean().item():.6e}")

def test_indexing_logic():
    """Verify the indexing logic is correct"""
    print("\n=== Testing Indexing Logic ===")

    V, W = 4, 3
    pad_t = V * W  # 12
    N_frames = 48
    num_frame_per_block = 3

    N_feats = (N_frames - 1) // V + 1  # (48-1)//4 + 1 = 12
    t_offset = N_feats - num_frame_per_block  # 12 - 3 = 9

    print(f"V={V}, W={W}, pad_t={pad_t}")
    print(f"N_frames={N_frames}, N_feats={N_feats}")
    print(f"num_frame_per_block={num_frame_per_block}, t_offset={t_offset}")

    print("\nFor each timestep, checking index ranges:")
    for t_local in range(num_frame_per_block):
        t_global = t_local + t_offset  # 9, 10, 11

        # PyTorch formula
        start_padded_torch = V * (t_global - W) + pad_t
        end_padded_torch = t_global * V + pad_t

        print(f"\nTimestep t_local={t_local}, t_global={t_global}:")
        print(f"  PyTorch padded range: [{start_padded_torch}, {end_padded_torch})")
        print(f"  Original indices: [{start_padded_torch - pad_t}, {end_padded_torch - pad_t})")

        # Triton formula (for each k in range(pad_t))
        print(f"  Triton indices (k=0 to k={pad_t-1}):")
        for k in [0, 1, pad_t//2, pad_t-1]:
            padded_idx = V * (t_global - W) + pad_t + k
            original_idx = padded_idx - pad_t
            safe_idx = max(0, min(original_idx, N_frames - 1))
            print(f"    k={k}: padded_idx={padded_idx}, original_idx={original_idx}, safe_idx={safe_idx}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        sys.exit(1)

    test_indexing_logic()
    test_dtype_float32()
    test_dtype_float16()
