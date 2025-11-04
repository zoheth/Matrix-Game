"""Debug script to verify mouse_preprocessor_triton matches MousePreprocessor"""
import torch
from wan.modules.modular_action.injectors import MousePreprocessor
from wan.modules.modular_action.kernels.preprocessor_kernel import mouse_preprocessor_triton

def test_indexing():
    # Small test case
    B, N_frames, C_mouse = 2, 100, 4
    T_q, S = 10, 2  # T=10, S=2 (spatial dimension)
    C_hidden = 8
    V, W = 8, 2

    # Create test data with new layout
    # Old API: hidden_states [BS, T, C] where BS = B*S
    # New API: hidden_states [B, T*S, C]
    hidden_states_new = torch.randn(B, T_q * S, C_hidden, device='cuda')
    mouse_condition = torch.randn(B, N_frames, C_mouse, device='cuda')

    # For reference, convert to old layout
    from einops import rearrange
    hidden_states_old = rearrange(hidden_states_new, "B (T S) C -> (B S) T C", T=T_q, S=S)

    # PyTorch reference (old API)
    preprocessor = MousePreprocessor(V, W)
    expected = preprocessor.forward(hidden_states_old, mouse_condition, is_causal=False, num_frame_per_block=-1)

    # Triton implementation (new API)
    actual = mouse_preprocessor_triton(hidden_states_new, mouse_condition, T_q, V, W, is_causal=False)

    print(f"Expected shape: {expected.shape}")  # [BS, T, C_fused]
    print(f"Actual shape: {actual.shape}")      # [B, S, T, C_fused]

    # Reshape actual to match expected for comparison
    actual_reshaped = actual.reshape(B * S, T_q, -1)

    # Memory layout check
    print(f"\nMemory layout:")
    print(f"Expected is_contiguous: {expected.is_contiguous()}")
    print(f"Actual is_contiguous: {actual.is_contiguous()}")
    print(f"Expected stride: {expected.stride()}")
    print(f"Actual stride: {actual.stride()}")

    # Compare hidden_states part (should be identical)
    hidden_part_match = torch.allclose(expected[:, :, :C_hidden], actual_reshaped[:, :, :C_hidden])
    print(f"\nHidden states match: {hidden_part_match}")

    # Compare mouse part
    mouse_part_expected = expected[:, :, C_hidden:]
    mouse_part_actual = actual_reshaped[:, :, C_hidden:]

    print(f"\nMouse part comparison:")
    print(f"Max diff: {(mouse_part_expected - mouse_part_actual).abs().max().item()}")
    print(f"Mean diff: {(mouse_part_expected - mouse_part_actual).abs().mean().item()}")

    mouse_part_match = torch.allclose(mouse_part_expected, mouse_part_actual, rtol=1e-5, atol=1e-5)
    print(f"Mouse parts match: {mouse_part_match}")

    if not mouse_part_match:
        # Debug: print first few values
        print(f"\nFirst timestep, first spatial location:")
        print(f"Expected mouse[0, 0, :20]: {mouse_part_expected[0, 0, :20]}")
        print(f"Actual mouse[0, 0, :20]: {mouse_part_actual[0, 0, :20]}")

        # Check which timesteps differ
        for t in range(T_q):
            diff = (mouse_part_expected[:, t, :] - mouse_part_actual[:, t, :]).abs().max()
            if diff > 1e-4:
                print(f"Timestep {t} differs: max diff = {diff.item()}")

    return expected, actual

def test_causal_mode():
    # Small test case for causal mode
    B, N_frames, C_mouse = 2, 100, 4
    V, W = 8, 2
    num_frame_per_block = 10
    S = 2  # Spatial dimension
    T_q = num_frame_per_block
    C_hidden = 8

    # Create test data with new layout
    hidden_states_new = torch.randn(B, T_q * S, C_hidden, device='cuda')
    mouse_condition = torch.randn(B, N_frames, C_mouse, device='cuda')

    # For reference, convert to old layout
    from einops import rearrange
    hidden_states_old = rearrange(hidden_states_new, "B (T S) C -> (B S) T C", T=T_q, S=S)

    # PyTorch reference (old API)
    preprocessor = MousePreprocessor(V, W)
    expected = preprocessor.forward(hidden_states_old, mouse_condition, is_causal=True, num_frame_per_block=num_frame_per_block)

    # Triton implementation (new API)
    actual = mouse_preprocessor_triton(hidden_states_new, mouse_condition, T_q, V, W, is_causal=True, num_frame_per_block=num_frame_per_block)

    print(f"\n=== Causal Mode Test ===")
    print(f"Expected shape: {expected.shape}")  # [BS, T, C_fused]
    print(f"Actual shape: {actual.shape}")      # [B, S, T, C_fused]

    # Reshape actual to match expected for comparison
    actual_reshaped = actual.reshape(B * S, T_q, -1)
    BS = B * S  # For loop iterations

    # Memory layout check
    print(f"\nMemory layout:")
    print(f"Expected is_contiguous: {expected.is_contiguous()}")
    print(f"Actual is_contiguous: {actual.is_contiguous()}")
    print(f"Expected stride: {expected.stride()}")
    print(f"Actual stride: {actual.stride()}")

    # Compare hidden_states part (should be identical)
    hidden_part_match = torch.allclose(expected[:, :, :C_hidden], actual_reshaped[:, :, :C_hidden])
    print(f"\nHidden states match: {hidden_part_match}")

    # Compare mouse part
    mouse_part_expected = expected[:, :, C_hidden:]
    mouse_part_actual = actual_reshaped[:, :, C_hidden:]

    print(f"\nMouse part comparison:")
    print(f"Max diff: {(mouse_part_expected - mouse_part_actual).abs().max().item()}")
    print(f"Mean diff: {(mouse_part_expected - mouse_part_actual).abs().mean().item()}")

    mouse_part_match = torch.allclose(mouse_part_expected, mouse_part_actual, rtol=1e-5, atol=1e-5)
    print(f"Mouse parts match: {mouse_part_match}")

    if not mouse_part_match:
        print(f"\nFirst timestep, first spatial location:")
        print(f"Expected mouse[0, 0, :20]: {mouse_part_expected[0, 0, :20]}")
        print(f"Actual mouse[0, 0, :20]: {mouse_part_actual[0, 0, :20]}")

        # Detailed comparison for debugging
        for bs in range(min(2, BS)):
            for t in range(min(3, T_q)):
                diff = (mouse_part_expected[bs, t] - mouse_part_actual[bs, t]).abs().max()
                if diff > 1e-4:
                    print(f"\nMismatch at BS={bs}, T={t}, max_diff={diff.item():.6f}")
                    print(f"Expected[{bs},{t},:10]: {mouse_part_expected[bs, t, :10]}")
                    print(f"Actual[{bs},{t},:10]: {mouse_part_actual[bs, t, :10]}")

    # Verify the t_offset calculation
    N_feats = (N_frames - 1) // V + 1
    t_offset = N_feats - num_frame_per_block
    print(f"\nCausal mode parameters:")
    print(f"N_frames: {N_frames}, V: {V}, N_feats: {N_feats}")
    print(f"num_frame_per_block: {num_frame_per_block}, t_offset: {t_offset}")
    print(f"Expected global indices: {t_offset} to {N_feats}")

    return expected, actual

def test_manual_verification():
    """Manually verify the indexing logic with a simple case"""
    print("\n=== Manual Verification ===")
    B, N_frames, C_mouse = 1, 20, 2
    T_q, S = 3, 2  # T=3, S=2
    C_hidden = 4
    V, W = 4, 2

    # Create simple test data with known values
    hidden_states_new = torch.ones(B, T_q * S, C_hidden, device='cuda')
    # Set mouse_condition to have frame index as value for easy verification
    mouse_condition = torch.zeros(B, N_frames, C_mouse, device='cuda')
    for i in range(N_frames):
        mouse_condition[0, i, :] = i  # Frame i has value i

    # For reference, convert to old layout
    from einops import rearrange
    hidden_states_old = rearrange(hidden_states_new, "B (T S) C -> (B S) T C", T=T_q, S=S)

    # PyTorch reference (old API)
    preprocessor = MousePreprocessor(V, W)
    expected = preprocessor.forward(hidden_states_old, mouse_condition, is_causal=False, num_frame_per_block=-1)

    # Triton implementation (new API)
    actual = mouse_preprocessor_triton(hidden_states_new, mouse_condition, T_q, V, W, is_causal=False)
    actual_reshaped = actual.reshape(B * S, T_q, -1)

    pad_t = V * W  # 8

    print(f"V={V}, W={W}, pad_t={pad_t}")
    print(f"Mouse condition frame values: 0, 1, 2, ..., {N_frames-1}")

    # For each timestep, print what frames should be accessed
    for t in range(T_q):
        # PyTorch formula: mouse_condition_padded[:, V*(t-W)+pad_t : t*V+pad_t, :]
        start_padded = V * (t - W) + pad_t
        end_padded = t * V + pad_t

        print(f"\nTimestep t={t}:")
        print(f"  Padded slice range: [{start_padded}, {end_padded})")
        print(f"  Original indices: [{start_padded - pad_t}, {end_padded - pad_t})")

        # Extract the actual values from both implementations
        mouse_slice_expected = expected[0, t, C_hidden:C_hidden + C_mouse]
        mouse_slice_actual = actual_reshaped[0, t, C_hidden:C_hidden + C_mouse]

        print(f"  Expected first frame value: {mouse_slice_expected}")
        print(f"  Actual first frame value: {mouse_slice_actual}")

        # Compare
        match = torch.allclose(mouse_slice_expected, mouse_slice_actual)
        if not match:
            print(f"  ❌ MISMATCH!")
        else:
            print(f"  ✓ Match")

def test_real_inference_shapes():
    """Test with actual inference parameters from inference_universal.yaml"""
    print("\n=== Real Inference Shapes Test ===")

    # From config.json
    V = 4  # vae_time_compression_ratio
    W = 3  # windows_size
    C_mouse = 2  # mouse_dim_in
    C_hidden = 1536  # img_hidden_size

    # From inference_universal.yaml
    # image_or_video_shape: [1, 16, 15, 44, 80] -> [B, T, C, H, W]
    # After VAE encoding and patching: H = 44 // 2 = 22, W = 80 // 2 = 40
    B = 1
    H_latent, W_latent = 22, 40  # After patching
    S = H_latent * W_latent  # Spatial dimension
    num_frame_per_block = 3  # From config

    # For causal inference, T_q = num_frame_per_block
    T_q = num_frame_per_block

    # Mouse condition: assume we have some frames (e.g., 12 output frames)
    # For 12 output frames at V=4 compression: N_frames = 12 * 4 = 48
    num_output_frames = 12
    N_frames = num_output_frames * V

    print(f"Configuration:")
    print(f"  V={V}, W={W}, pad_t={V*W}")
    print(f"  B={B}, H={H_latent}, W={W_latent}, S={S}")
    print(f"  T_q={T_q}")
    print(f"  C_hidden={C_hidden}, C_mouse={C_mouse}")
    print(f"  N_frames={N_frames}, num_frame_per_block={num_frame_per_block}")

    # Create test data with new layout
    hidden_states_new = torch.randn(B, T_q * S, C_hidden, device='cuda', dtype=torch.float16)
    mouse_condition = torch.randn(B, N_frames, C_mouse, device='cuda', dtype=torch.float16)

    # For reference, convert to old layout
    from einops import rearrange
    hidden_states_old = rearrange(hidden_states_new, "B (T S) C -> (B S) T C", T=T_q, S=S)

    # PyTorch reference (old API)
    preprocessor = MousePreprocessor(V, W)
    expected = preprocessor.forward(hidden_states_old, mouse_condition, is_causal=True, num_frame_per_block=num_frame_per_block)

    # Triton implementation (new API)
    actual = mouse_preprocessor_triton(hidden_states_new, mouse_condition, T_q, V, W, is_causal=True, num_frame_per_block=num_frame_per_block)

    print(f"\nOutput shapes:")
    print(f"  Expected: {expected.shape}")  # [BS, T, C_fused]
    print(f"  Actual: {actual.shape}")      # [B, S, T, C_fused]

    # Reshape actual to match expected for comparison
    actual_reshaped = actual.reshape(B * S, T_q, -1)

    # Memory layout
    print(f"\nMemory layout:")
    print(f"  Expected is_contiguous: {expected.is_contiguous()}")
    print(f"  Actual is_contiguous: {actual.is_contiguous()}")
    print(f"  Expected stride: {expected.stride()}")
    print(f"  Actual stride: {actual.stride()}")
    print(f"  Expected dtype: {expected.dtype}")
    print(f"  Actual dtype: {actual.dtype}")

    # Compare
    hidden_match = torch.allclose(expected[:, :, :C_hidden], actual_reshaped[:, :, :C_hidden], rtol=1e-3, atol=1e-3)
    mouse_match = torch.allclose(expected[:, :, C_hidden:], actual_reshaped[:, :, C_hidden:], rtol=1e-3, atol=1e-3)

    print(f"\nComparison (fp16 tolerance):")
    print(f"  Hidden part match: {hidden_match}")
    print(f"  Mouse part match: {mouse_match}")

    if not mouse_match:
        mouse_expected = expected[:, :, C_hidden:]
        mouse_actual = actual_reshaped[:, :, C_hidden:]
        diff = (mouse_expected - mouse_actual).abs()
        print(f"  Max diff: {diff.max().item()}")
        print(f"  Mean diff: {diff.mean().item()}")

        # Find where the mismatch is
        max_diff_idx = diff.view(-1).argmax()
        bs_idx = max_diff_idx // (T_q * (V * W * C_mouse))
        remainder = max_diff_idx % (T_q * (V * W * C_mouse))
        t_idx = remainder // (V * W * C_mouse)
        c_idx = remainder % (V * W * C_mouse)

        print(f"  Max diff at: BS={bs_idx}, T={t_idx}, C={c_idx}")
        print(f"  Expected value: {mouse_expected.view(-1)[max_diff_idx].item()}")
        print(f"  Actual value: {mouse_actual.view(-1)[max_diff_idx].item()}")

    # Verify t_offset calculation
    N_feats = (N_frames - 1) // V + 1
    t_offset = N_feats - num_frame_per_block
    print(f"\nCausal indexing:")
    print(f"  N_feats={N_feats}, t_offset={t_offset}")
    print(f"  Global timestep range: [{t_offset}, {N_feats})")

    return expected, actual

if __name__ == "__main__":
    print("=== Non-Causal Mode Test ===")
    expected, actual = test_indexing()
    test_causal_mode()
    test_manual_verification()
    test_real_inference_shapes()
