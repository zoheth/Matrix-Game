"""Test with deterministic mouse trajectory to expose indexing errors"""
import torch
from wan.modules.modular_action.injectors import MousePreprocessor
from wan.modules.modular_action.kernels.preprocessor_kernel import mouse_preprocessor_triton

def test_deterministic_trajectory():
    """Test with a known, deterministic mouse trajectory"""
    print("=== Testing with Deterministic Trajectory ===\n")

    # Use real inference parameters
    V = 4
    W = 3
    B = 1
    N_frames = 48
    C_mouse = 2
    num_frame_per_block = 3

    # Small spatial size for easier debugging
    H, W_spatial = 2, 2
    S = H * W_spatial
    BS = B * S
    T_q = num_frame_per_block
    C_hidden = 16

    print(f"Parameters:")
    print(f"  V={V}, W={W}, pad_t={V*W}")
    print(f"  B={B}, N_frames={N_frames}, C_mouse={C_mouse}")
    print(f"  S={S}, BS={BS}, T_q={T_q}, C_hidden={C_hidden}")
    print()

    # Create DETERMINISTIC mouse trajectory (linearly increasing)
    hidden_states = torch.randn(BS, T_q, C_hidden, device='cuda', dtype=torch.float32)
    mouse_condition = torch.zeros(B, N_frames, C_mouse, device='cuda', dtype=torch.float32)

    # Set mouse_condition to have frame index as value
    # This makes it easy to see which frames are being read
    for i in range(N_frames):
        mouse_condition[0, i, 0] = i * 10.0  # x coordinate
        mouse_condition[0, i, 1] = i * 20.0  # y coordinate

    print("Mouse trajectory (first 10 frames):")
    for i in range(min(10, N_frames)):
        print(f"  Frame {i}: x={mouse_condition[0, i, 0].item():.1f}, y={mouse_condition[0, i, 1].item():.1f}")
    print()

    # PyTorch reference
    preprocessor = MousePreprocessor(V, W)
    expected = preprocessor.forward(hidden_states, mouse_condition, is_causal=True, num_frame_per_block=num_frame_per_block)

    # Triton implementation
    actual = mouse_preprocessor_triton(hidden_states, mouse_condition, V, W, is_causal=True, num_frame_per_block=num_frame_per_block)

    print(f"Output shapes:")
    print(f"  Expected: {expected.shape}")
    print(f"  Actual: {actual.shape}")
    print()

    # Extract mouse parts
    mouse_expected = expected[:, :, C_hidden:]
    mouse_actual = actual[:, :, C_hidden:]

    # Compare
    match = torch.allclose(mouse_expected, mouse_actual, rtol=1e-5, atol=1e-5)
    print(f"Mouse parts match: {match}")

    if not match:
        print("\n❌ MISMATCH DETECTED!\n")
        diff = (mouse_expected - mouse_actual).abs()
        print(f"Max diff: {diff.max().item():.6f}")
        print(f"Mean diff: {diff.mean().item():.6f}")

        # Show detailed comparison for first spatial location
        print("\n" + "="*70)
        print("Detailed comparison for first spatial location (bs=0):")
        print("="*70)

        for t in range(T_q):
            print(f"\nTimestep t={t}:")

            # Calculate which frames should be in the window
            N_feats = (N_frames - 1) // V + 1
            t_global = t + (N_feats - num_frame_per_block)
            start_frame = V * (t_global - W)
            end_frame = t_global * V

            print(f"  Global t={t_global}, should read frames [{start_frame}:{end_frame}]")

            # Extract the first frame's coordinates from each output
            # The output is flattened: [frame_0_x, frame_0_y, frame_1_x, frame_1_y, ...]
            expected_first_x = mouse_expected[0, t, 0].item()
            actual_first_x = mouse_actual[0, t, 0].item()

            expected_first_y = mouse_expected[0, t, 1].item()
            actual_first_y = mouse_actual[0, t, 1].item()

            print(f"  Expected first frame: x={expected_first_x:.1f}, y={expected_first_y:.1f}")
            print(f"  Actual first frame:   x={actual_first_x:.1f}, y={actual_first_y:.1f}")

            # The first frame should correspond to start_frame
            if start_frame >= 0:
                correct_x = mouse_condition[0, start_frame, 0].item()
                correct_y = mouse_condition[0, start_frame, 1].item()
                print(f"  Correct (frame {start_frame}):  x={correct_x:.1f}, y={correct_y:.1f}")

            if abs(expected_first_x - actual_first_x) > 0.01:
                print(f"  ❌ X coordinate mismatch!")
                # Guess which frame Triton is reading
                for f in range(max(0, start_frame-5), min(N_frames, end_frame+5)):
                    if abs(actual_first_x - mouse_condition[0, f, 0].item()) < 0.01:
                        offset = f - start_frame
                        print(f"     Triton appears to be reading frame {f} (offset: {offset:+d})")
                        break
            else:
                print(f"  ✓ Match!")

    else:
        print("\n✓ Perfect match! The implementations are aligned.\n")

        # Show what each timestep is reading
        print("="*70)
        print("Verification: Which frames are being read")
        print("="*70)

        for t in range(T_q):
            N_feats = (N_frames - 1) // V + 1
            t_global = t + (N_feats - num_frame_per_block)
            start_frame = V * (t_global - W)
            end_frame = t_global * V

            # Extract first frame from output
            first_x = mouse_actual[0, t, 0].item()

            # Verify it matches the expected frame
            if start_frame >= 0:
                correct_x = mouse_condition[0, start_frame, 0].item()
                status = "✓" if abs(first_x - correct_x) < 0.01 else "❌"
                print(f"t={t} (global={t_global}): reads frames [{start_frame}:{end_frame}], first_x={first_x:.1f} {status}")

def test_with_float16():
    """Test with float16 (actual inference dtype)"""
    print("\n\n" + "="*70)
    print("=== Testing with Float16 ===")
    print("="*70 + "\n")

    V = 4
    W = 3
    B = 1
    N_frames = 48
    C_mouse = 2
    num_frame_per_block = 3
    S = 880  # Real spatial size
    BS = B * S
    T_q = num_frame_per_block
    C_hidden = 1536  # Real hidden size

    print(f"Parameters (real inference):")
    print(f"  V={V}, W={W}, B={B}, S={S}, BS={BS}")
    print(f"  T_q={T_q}, C_hidden={C_hidden}, C_mouse={C_mouse}")
    print()

    # Deterministic trajectory
    hidden_states = torch.randn(BS, T_q, C_hidden, device='cuda', dtype=torch.float16)
    mouse_condition = torch.zeros(B, N_frames, C_mouse, device='cuda', dtype=torch.float16)

    for i in range(N_frames):
        mouse_condition[0, i, 0] = float(i * 10)
        mouse_condition[0, i, 1] = float(i * 20)

    # PyTorch reference
    preprocessor = MousePreprocessor(V, W)
    expected = preprocessor.forward(hidden_states, mouse_condition, is_causal=True, num_frame_per_block=num_frame_per_block)

    # Triton implementation
    actual = mouse_preprocessor_triton(hidden_states, mouse_condition, V, W, is_causal=True, num_frame_per_block=num_frame_per_block)

    # Compare
    mouse_expected = expected[:, :, C_hidden:]
    mouse_actual = actual[:, :, C_hidden:]

    match = torch.allclose(mouse_expected, mouse_actual, rtol=1e-3, atol=1e-3)
    print(f"Mouse parts match (float16): {match}")

    if not match:
        diff = (mouse_expected - mouse_actual).abs()
        print(f"Max diff: {diff.max().item():.6f}")
        print(f"Mean diff: {diff.mean().item():.6f}")

        # Check first spatial location
        print("\nFirst spatial location:")
        for t in range(T_q):
            first_x_expected = mouse_expected[0, t, 0].item()
            first_x_actual = mouse_actual[0, t, 0].item()
            print(f"  t={t}: expected_x={first_x_expected:.1f}, actual_x={first_x_actual:.1f}")
    else:
        print("✓ Float16 test passed!")

def test_non_contiguous_input():
    """Test with non-contiguous mouse_condition to verify the fix"""
    print("\n\n" + "="*70)
    print("=== Testing with Non-Contiguous Input ===")
    print("="*70 + "\n")

    V = 4
    W = 3
    B = 1
    N_frames = 48
    C_mouse = 2
    num_frame_per_block = 3
    S = 4
    BS = B * S
    T_q = num_frame_per_block
    C_hidden = 16

    # Create deterministic trajectory
    hidden_states = torch.randn(BS, T_q, C_hidden, device='cuda', dtype=torch.float32)
    mouse_condition_full = torch.zeros(B, N_frames * 2, C_mouse, device='cuda', dtype=torch.float32)

    for i in range(N_frames * 2):
        mouse_condition_full[0, i, 0] = i * 10.0
        mouse_condition_full[0, i, 1] = i * 20.0

    # Create non-contiguous view (every other frame)
    mouse_condition = mouse_condition_full[:, ::2, :]  # This is NOT contiguous!

    print(f"mouse_condition.is_contiguous(): {mouse_condition.is_contiguous()}")
    print(f"mouse_condition.shape: {mouse_condition.shape}")
    print(f"mouse_condition.stride(): {mouse_condition.stride()}")
    print()

    # PyTorch reference
    preprocessor = MousePreprocessor(V, W)
    expected = preprocessor.forward(hidden_states, mouse_condition, is_causal=True, num_frame_per_block=num_frame_per_block)

    # Triton implementation (should handle non-contiguous input now)
    actual = mouse_preprocessor_triton(hidden_states, mouse_condition, V, W, is_causal=True, num_frame_per_block=num_frame_per_block)

    # Compare
    mouse_expected = expected[:, :, C_hidden:]
    mouse_actual = actual[:, :, C_hidden:]

    match = torch.allclose(mouse_expected, mouse_actual, rtol=1e-5, atol=1e-5)
    print(f"Match with non-contiguous input: {match}")

    if match:
        print("✓ Non-contiguous input test passed! The .contiguous() fix works.")
    else:
        print("❌ Still failing with non-contiguous input!")
        diff = (mouse_expected - mouse_actual).abs()
        print(f"Max diff: {diff.max().item():.6f}")

if __name__ == "__main__":
    test_deterministic_trajectory()
    test_with_float16()
    test_non_contiguous_input()
