"""Analyze the indexing difference between PyTorch and Triton implementations"""

def analyze_pytorch_indexing():
    """Analyze PyTorch MousePreprocessor indexing"""
    print("=== PyTorch MousePreprocessor Indexing ===\n")

    V = 4  # vae_time_compression_ratio
    W = 3  # windows_size
    pad_t = V * W  # 12
    N_frames = 48
    num_frame_per_block = 3

    N_feats = (N_frames - 1) // V + 1  # 12
    t_offset = N_feats - num_frame_per_block  # 9

    print(f"Parameters: V={V}, W={W}, pad_t={pad_t}")
    print(f"N_frames={N_frames}, N_feats={N_feats}, t_offset={t_offset}\n")

    # PyTorch code (from injectors.py lines 56-66):
    # group_mouse = [
    #     mouse_condition_padded[
    #         :, self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i * self.vae_time_compression_ratio + pad_t, :
    #     ]
    #     for i in range(start_global_idx, end_global_idx)
    # ]

    start_global_idx = t_offset  # 9
    end_global_idx = N_feats  # 12

    print("PyTorch loop: for i in range(start_global_idx, end_global_idx)")
    print(f"  range({start_global_idx}, {end_global_idx}) = {list(range(start_global_idx, end_global_idx))}\n")

    for i in range(start_global_idx, end_global_idx):
        start_idx = V * (i - W) + pad_t
        end_idx = i * V + pad_t

        # Convert to original indices (before padding)
        start_orig = start_idx - pad_t
        end_orig = end_idx - pad_t

        print(f"i={i}:")
        print(f"  Padded slice: [{start_idx}:{end_idx}]")
        print(f"  Original indices: [{start_orig}:{end_orig}]")
        print(f"  Window size: {end_idx - start_idx} frames")
        print()

def analyze_triton_indexing():
    """Analyze Triton kernel indexing"""
    print("\n=== Triton Kernel Indexing ===\n")

    V = 4
    W = 3
    PAD_T = V * W  # 12
    N_frames = 48
    num_frame_per_block = 3

    N_feats = (N_frames - 1) // V + 1  # 12
    t_offset = N_feats - num_frame_per_block  # 9

    print(f"Parameters: V={V}, W={W}, PAD_T={PAD_T}")
    print(f"N_frames={N_frames}, N_feats={N_feats}, t_offset={t_offset}\n")

    # Triton code (from preprocessor_kernel.py lines 42-95):
    # pid_t_local = tl.program_id(axis=1)  # 0, 1, 2 (for num_frame_per_block=3)
    # pid_t_global = pid_t_local + T_OFFSET
    #
    # for k in tl.static_range(PAD_T):
    #     padded_idx = V * (pid_t_global - W) + PAD_T + k

    print("Triton loop: for pid_t_local in range(num_frame_per_block)")
    print(f"  range(0, {num_frame_per_block}) = {list(range(num_frame_per_block))}\n")

    for pid_t_local in range(num_frame_per_block):
        pid_t_global = pid_t_local + t_offset

        print(f"pid_t_local={pid_t_local}, pid_t_global={pid_t_global}:")

        # Show first and last k
        for k in [0, 1, 2, PAD_T-2, PAD_T-1]:
            padded_idx = V * (pid_t_global - W) + PAD_T + k
            original_idx = padded_idx - PAD_T
            safe_idx = max(0, min(original_idx, N_frames - 1))

            print(f"  k={k:2d}: padded_idx={padded_idx:2d}, original_idx={original_idx:2d}, safe_idx={safe_idx:2d}")

        # Calculate the range
        k_start = 0
        k_end = PAD_T - 1
        padded_start = V * (pid_t_global - W) + PAD_T + k_start
        padded_end = V * (pid_t_global - W) + PAD_T + k_end

        print(f"  Range: padded[{padded_start}:{padded_end+1}] -> original[{padded_start-PAD_T}:{padded_end-PAD_T+1}]")
        print()

def compare_implementations():
    """Compare the two implementations side by side"""
    print("\n=== Side-by-Side Comparison ===\n")

    V = 4
    W = 3
    pad_t = V * W  # 12
    N_frames = 48
    num_frame_per_block = 3

    N_feats = (N_frames - 1) // V + 1  # 12
    t_offset = N_feats - num_frame_per_block  # 9

    print(f"For each timestep, comparing PyTorch vs Triton:\n")

    for t_local in range(num_frame_per_block):
        t_global = t_local + t_offset

        # PyTorch
        pytorch_start = V * (t_global - W) + pad_t
        pytorch_end = t_global * V + pad_t

        # Triton
        triton_start = V * (t_global - W) + pad_t + 0  # k=0
        triton_end = V * (t_global - W) + pad_t + (pad_t - 1)  # k=pad_t-1

        print(f"t_local={t_local}, t_global={t_global}:")
        print(f"  PyTorch: padded[{pytorch_start}:{pytorch_end}]")
        print(f"  Triton:  padded[{triton_start}:{triton_end+1}]")

        if pytorch_start == triton_start and pytorch_end == triton_end + 1:
            print(f"  ✓ MATCH (Python slice excludes end, Triton includes it)")
        else:
            print(f"  ✗ MISMATCH!")

        # Original indices
        pytorch_orig_start = pytorch_start - pad_t
        pytorch_orig_end = pytorch_end - pad_t
        triton_orig_start = triton_start - pad_t
        triton_orig_end = (triton_end + 1) - pad_t

        print(f"  PyTorch original: [{pytorch_orig_start}:{pytorch_orig_end}]")
        print(f"  Triton original:  [{triton_orig_start}:{triton_orig_end}]")
        print()

if __name__ == "__main__":
    analyze_pytorch_indexing()
    analyze_triton_indexing()
    compare_implementations()
