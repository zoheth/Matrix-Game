"""Check if there's a stride-related issue in Triton kernel"""

def analyze_stride_calculation():
    """Analyze potential stride issues"""
    print("=== Analyzing Stride Calculation ===\n")

    # Scenario 1: Small test case (from test_preprocessor_debug.py)
    print("Scenario 1: Small test case")
    BS, T_q, C_hidden = 4, 10, 8
    B, N_frames, C_mouse = 2, 100, 4

    S = BS // B
    print(f"BS={BS}, T_q={T_q}, C_hidden={C_hidden}")
    print(f"B={B}, N_frames={N_frames}, C_mouse={C_mouse}")
    print(f"S={S}\n")

    # Scenario 2: Real inference case
    print("Scenario 2: Real inference (from test_real_inference_shapes)")
    BS, T_q, C_hidden = 880, 3, 1536
    B, N_frames, C_mouse = 1, 48, 2

    S = BS // B
    print(f"BS={BS}, T_q={T_q}, C_hidden={C_hidden}")
    print(f"B={B}, N_frames={N_frames}, C_mouse={C_mouse}")
    print(f"S={S}\n")

    # Check b_idx calculation in kernel
    print("In Triton kernel (line 86):")
    print("  b_idx = pid_bs // S")
    print("\nFor different pid_bs values:")
    for pid_bs in [0, 1, 2, S-1, S, S+1, BS-1]:
        b_idx = pid_bs // S
        if pid_bs < BS:
            print(f"  pid_bs={pid_bs:4d} -> b_idx={b_idx}")

def analyze_potential_issues():
    """Analyze potential issues"""
    print("\n\n=== Potential Issues ===\n")

    issues = [
        {
            "title": "1. Integer Division in Triton",
            "description": """
In the Triton kernel (line 95):
  padded_idx = V * (pid_t_global - W) + PAD_T + k

If any of these variables have unexpected types or overflow,
the calculation could be wrong. Triton uses int32 by default."""
        },
        {
            "title": "2. safe_idx Clamping Logic",
            "description": """
In lines 101-104:
  original_idx = padded_idx - PAD_T
  safe_idx = tl.where(original_idx < 0, 0, original_idx)
  safe_idx = tl.where(safe_idx >= N_frames, N_frames - 1, safe_idx)

The clamping is correct, but if there's an edge case where
safe_idx points to the wrong frame, it would cause misalignment."""
        },
        {
            "title": "3. Stride Calculation",
            "description": """
In line 109:
  mouse_in_ptr = (
      MOUSE_CONDITION_PTR
      + b_idx * stride_mouse_b
      + safe_idx * stride_mouse_t
      + c_mouse_offsets
  )

If b_idx or safe_idx is wrong, we'd be reading from the wrong memory location.
This could happen if:
  - b_idx calculation is wrong (pid_bs // S)
  - safe_idx calculation is wrong (clamping logic)"""
        },
        {
            "title": "4. Float16 Precision",
            "description": """
When using float16, numerical errors can accumulate. However,
since this is just data movement (not computation), float16
shouldn't cause issues unless there's type conversion happening."""
        },
        {
            "title": "5. Memory Ordering (CRITICAL)",
            "description": """
⚠️  CRITICAL: Check if the memory is being read in the correct order!

In PyTorch (injectors.py line 78-83):
  group_mouse = torch.stack(group_mouse, dim=1)  # [B, T_q, pad_t, C_mouse]
  group_mouse = group_mouse.unsqueeze(2).expand(B, -1, S, pad_t, C_mouse)
  group_mouse = rearrange(group_mouse, "B T S W C -> (B S) T (W C)")

The final layout is: [BS, T_q, pad_t * C_mouse]
Where pad_t * C_mouse means the mouse features are FLATTENED.

In Triton (line 117):
  fused_out_mouse_ptr = (
      FUSED_PTR
      + pid_bs * stride_fused_bs
      + pid_t_local * stride_fused_t
      + (C_hidden + k * C_mouse)  <-- This is the FLATTENING
      + c_mouse_offsets
  )

The formula `C_hidden + k * C_mouse + c_mouse_offsets` matches
the PyTorch flattening: for frame k, channel c, the index is
C_hidden + k * C_mouse + c.

This looks correct!"""
        },
        {
            "title": "6. Loop Unrolling (tl.static_range)",
            "description": """
The Triton kernel uses `tl.static_range(PAD_T)` which fully unrolls
the loop. For PAD_T=12, this creates 12 separate load/store operations.

If PAD_T is large (e.g., V=8, W=4 -> PAD_T=32), this could:
  - Cause register pressure
  - Affect performance
  - Potentially cause subtle bugs in Triton's compiler

However, for PAD_T=12 (V=4, W=3), this should be fine."""
        }
    ]

    for issue in issues:
        print(issue["title"])
        print(issue["description"])
        print()

def suggest_debugging_steps():
    """Suggest debugging steps"""
    print("\n=== Debugging Steps ===\n")

    steps = [
        "1. Add print statements in Triton kernel to verify indices",
        "2. Compare intermediate values (padded_idx, safe_idx) with PyTorch",
        "3. Check if b_idx is correct for all spatial locations",
        "4. Verify that k loop processes frames in the correct order",
        "5. Test with smaller PAD_T (e.g., V=2, W=2) to simplify debugging",
        "6. Use torch.testing.assert_close with specific atol/rtol for float16",
        "7. Check if the issue appears only in causal mode or also in non-causal",
        "8. Profile both implementations to see if there's a performance difference"
    ]

    for step in steps:
        print(f"  {step}")

if __name__ == "__main__":
    analyze_stride_calculation()
    analyze_potential_issues()
    suggest_debugging_steps()
