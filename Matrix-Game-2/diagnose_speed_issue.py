"""Diagnose why Triton appears to "move faster" than PyTorch"""
import sys

print("=== Critical Analysis: Why Triton 'Moves Faster' ===\n")

print("""
HYPOTHESIS: The issue is NOT with the test passing, but with how the
mouse condition is INTERPRETED during actual inference.

If Triton is "moving faster", it suggests that the mouse coordinates
are being read from a DIFFERENT temporal position than expected.

Possible causes:
""")

causes = [
    {
        "num": 1,
        "title": "Integer Type Overflow/Truncation",
        "description": """
In Triton kernel line 95:
  padded_idx = V * (pid_t_global - W) + PAD_T + k

All of these should be int32, but let's verify:
  - V: int (from Python)
  - pid_t_global: int32 (from tl.program_id + T_OFFSET)
  - W: int (from Python)
  - PAD_T: int (constexpr)
  - k: int (from tl.static_range)

If any of these is computed incorrectly, safe_idx could point to
the wrong frame, causing the mouse to appear at a different position.
""",
        "likelihood": "LOW (Triton handles int32 correctly)",
    },
    {
        "num": 2,
        "title": "safe_idx Clamping Creates Off-by-One Error",
        "description": """
In lines 101-104:
  original_idx = padded_idx - PAD_T
  safe_idx = tl.where(original_idx < 0, 0, original_idx)
  safe_idx = tl.where(safe_idx >= N_frames, N_frames - 1, safe_idx)

This clamping logic could cause issues if:
  - original_idx is supposed to be negative (padding region)
  - original_idx is >= N_frames (beyond the end)

However, for correctly calculated indices, this should match PyTorch's
padding behavior exactly.
""",
        "likelihood": "LOW (Logic matches PyTorch padding)",
    },
    {
        "num": 3,
        "title": "Stride Calculation Error (CRITICAL)",
        "description": """
⚠️  THIS IS THE MOST LIKELY CAUSE!

In line 109:
  mouse_in_ptr = (
      MOUSE_CONDITION_PTR
      + b_idx * stride_mouse_b
      + safe_idx * stride_mouse_t  <-- This line!
      + c_mouse_offsets
  )

If mouse_condition is not contiguous, or if the stride is wrong,
we could be reading from the wrong memory location.

Let's check: mouse_condition shape is [B, N_frames, C_mouse]
  - stride_mouse_b should be N_frames * C_mouse
  - stride_mouse_t should be C_mouse
  - stride_mouse_c should be 1

If mouse_condition is passed as a view or a non-contiguous tensor,
the strides might not match these expected values!
""",
        "likelihood": "HIGH (Stride mismatch is a common Triton issue)",
    },
    {
        "num": 4,
        "title": "safe_idx is int64 instead of int32",
        "description": """
Triton works best with int32. If safe_idx is somehow int64,
the pointer arithmetic could be wrong.

This is unlikely because tl.where returns the same type as its inputs.
""",
        "likelihood": "VERY LOW",
    },
    {
        "num": 5,
        "title": "Race Condition (Multiple Program Instances)",
        "description": """
Triton launches multiple program instances in parallel. Each instance
handles one (pid_bs, pid_t_local) pair.

If there's a race condition in how indices are calculated, different
program instances could interfere with each other.

However, since each instance writes to a different output location,
and reads from non-overlapping input locations, this shouldn't happen.
""",
        "likelihood": "VERY LOW (No shared state)",
    },
    {
        "num": 6,
        "title": "Triton Compiler Bug / Optimization Issue",
        "description": """
Triton's compiler might be reordering operations or optimizing in a way
that changes the behavior.

This is possible but hard to debug without looking at the generated PTX.
""",
        "likelihood": "MEDIUM (Compiler bugs do happen)",
    },
    {
        "num": 7,
        "title": "Test Passes But Uses Different Data",
        "description": """
⚠️  CRITICAL: The test might be passing because it's using RANDOM data!

With random data, small differences in indexing might not be noticeable.
But with REAL mouse trajectories (e.g., linear motion), an off-by-one
error would make the mouse appear to move faster or slower.

To test this: Use a KNOWN mouse trajectory (e.g., linearly increasing
coordinates) and check if Triton reads the same values as PyTorch.
""",
        "likelihood": "VERY HIGH (This is the most likely explanation)",
    },
    {
        "num": 8,
        "title": "t_offset Calculation Error in Causal Mode",
        "description": """
In preprocessor_kernel.py line 153:
  t_offset = N_feats - num_frame_per_block

If N_feats is calculated incorrectly, t_offset will be wrong, and
we'll be reading from the wrong temporal position.

N_feats = (N_frames - 1) // V + 1

This formula is correct and matches PyTorch. However, if N_frames is
not what we expect (e.g., due to padding or batching), this could fail.
""",
        "likelihood": "MEDIUM (Check N_frames value)",
    },
]

for cause in causes:
    print(f"\n{'='*70}")
    print(f"Cause #{cause['num']}: {cause['title']}")
    print(f"Likelihood: {cause['likelihood']}")
    print(f"{'='*70}")
    print(cause['description'])

print("\n" + "="*70)
print("RECOMMENDED DEBUGGING STEPS")
print("="*70)

steps = [
    """
1. Create a deterministic mouse trajectory:
   Instead of torch.randn(), use:
     mouse_condition[0, i, :] = torch.tensor([i * 10, i * 20])
   This creates a linearly increasing trajectory that's easy to verify.
""",
    """
2. Print intermediate values in the Triton kernel:
   Unfortunately, Triton doesn't support print statements inside kernels.
   Instead, create a debug tensor to store intermediate values:
     debug_tensor[pid_bs, pid_t_local, 0] = pid_t_global
     debug_tensor[pid_bs, pid_t_local, 1] = padded_idx
     debug_tensor[pid_bs, pid_t_local, 2] = safe_idx
""",
    """
3. Check mouse_condition.stride():
   Before passing to Triton, verify:
     print(f"mouse_condition.stride() = {mouse_condition.stride()}")
     print(f"Expected: ({N_frames * C_mouse}, {C_mouse}, 1)")

   If they don't match, call mouse_condition.contiguous() first.
""",
    """
4. Test with minimal parameters:
   Use V=2, W=2 (so PAD_T=4) to make debugging easier.
   Use B=1, S=1 (so BS=1) to eliminate spatial complexity.
""",
    """
5. Compare outputs element by element:
   For the first few spatial locations and timesteps, print both
   PyTorch and Triton outputs side by side to see where they diverge.
""",
    """
6. Test non-causal mode first:
   If the issue only appears in causal mode, it's related to t_offset.
   If it appears in both modes, it's related to the core indexing logic.
""",
    """
7. Check if the issue is consistent:
   Run the test multiple times. If the error magnitude varies, it might
   be a numerical precision issue. If it's always the same, it's likely
   a systematic indexing error.
""",
]

for i, step in enumerate(steps, 1):
    print(f"\nStep {i}:{step}")

print("\n" + "="*70)
print("MOST LIKELY ROOT CAUSE")
print("="*70)
print("""
Based on the symptom "Triton moves faster", the most likely causes are:

1. **Cause #7**: The test uses random data, hiding a systematic error
   that becomes visible with real mouse trajectories.

2. **Cause #3**: mouse_condition is not contiguous, causing stride
   mismatch and reading from wrong memory locations.

3. **Cause #8**: t_offset is calculated incorrectly, or N_frames
   doesn't match expectations.

NEXT STEP: Run the test with a deterministic mouse trajectory and
check if the outputs match. Also verify mouse_condition.is_contiguous().
""")

print("\n" + "="*70)
print("IMPLEMENTATION FIX")
print("="*70)
print("""
If the issue is stride-related, add this to mouse_preprocessor_triton():

    # Ensure mouse_condition is contiguous
    if not mouse_condition.is_contiguous():
        mouse_condition = mouse_condition.contiguous()

If the issue is related to deterministic trajectories, the test needs
to be updated to use non-random data that can expose indexing errors.
""")
