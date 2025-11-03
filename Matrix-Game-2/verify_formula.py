"""Verify the indexing formula is exactly the same"""

def verify_formula():
    V = 4
    W = 3
    pad_t = V * W  # 12

    # Test for different global timesteps
    for i in [9, 10, 11]:
        print(f"\n=== Global timestep i={i} ===")

        # PyTorch slice
        pytorch_start = V * (i - W) + pad_t
        pytorch_end = i * V + pad_t

        print(f"PyTorch slice: [{pytorch_start}:{pytorch_end}]")
        print(f"  Length: {pytorch_end - pytorch_start}")
        print(f"  Indices: {list(range(pytorch_start, pytorch_end))}")

        # Triton individual indices
        print(f"\nTriton indices (using k):")
        triton_indices = []
        for k in range(pad_t):
            padded_idx = V * (i - W) + pad_t + k
            triton_indices.append(padded_idx)

        print(f"  Length: {len(triton_indices)}")
        print(f"  Indices: {triton_indices}")

        # Compare
        pytorch_indices = list(range(pytorch_start, pytorch_end))
        if pytorch_indices == triton_indices:
            print("\n✓ MATCH!")
        else:
            print("\n✗ MISMATCH!")
            print(f"  PyTorch: {pytorch_indices}")
            print(f"  Triton:  {triton_indices}")

        # Verify the formula
        print(f"\nFormula verification:")
        print(f"  PyTorch start: V*(i-W) + pad_t = {V}*({i}-{W}) + {pad_t} = {pytorch_start}")
        print(f"  Triton k=0:    V*(i-W) + pad_t + 0 = {V}*({i}-{W}) + {pad_t} + 0 = {triton_indices[0]}")
        print(f"  PyTorch end:   i*V + pad_t = {i}*{V} + {pad_t} = {pytorch_end}")
        print(f"  Triton k={pad_t-1}:    V*(i-W) + pad_t + {pad_t-1} = {V}*({i}-{W}) + {pad_t} + {pad_t-1} = {triton_indices[-1]}")

        # The key insight:
        # PyTorch: V*(i-W) + pad_t to i*V + pad_t (exclusive end)
        # Triton:  V*(i-W) + pad_t + k for k in range(pad_t)
        #
        # When k=0: V*(i-W) + pad_t
        # When k=pad_t-1: V*(i-W) + pad_t + pad_t - 1
        #                = V*(i-W) + 2*pad_t - 1
        #
        # But PyTorch end is: i*V + pad_t
        # We need to verify: V*(i-W) + 2*pad_t = i*V + pad_t
        #                    V*i - V*W + 2*pad_t = i*V + pad_t
        #                    - V*W + 2*pad_t = pad_t
        #                    - V*W + V*W = 0  (since pad_t = V*W)
        #                    This doesn't work!

        print(f"\n  Check: Does V*(i-W) + 2*pad_t = i*V + pad_t?")
        lhs = V * (i - W) + 2 * pad_t
        rhs = i * V + pad_t
        print(f"    LHS: {V}*({i}-{W}) + 2*{pad_t} = {lhs}")
        print(f"    RHS: {i}*{V} + {pad_t} = {rhs}")
        print(f"    Equal: {lhs == rhs}")

        # Actually, let me recalculate
        print(f"\n  Correct check: Does V*(i-W) + pad_t + (pad_t-1) + 1 = i*V + pad_t?")
        lhs2 = V * (i - W) + pad_t + (pad_t - 1) + 1
        print(f"    LHS: {V}*({i}-{W}) + {pad_t} + {pad_t-1} + 1 = {lhs2}")
        print(f"    RHS: {i}*{V} + {pad_t} = {rhs}")
        print(f"    Equal: {lhs2 == rhs}")

        # Simplify: V*i - V*W + pad_t + pad_t - 1 + 1 = i*V + pad_t
        #           V*i - V*W + 2*pad_t = i*V + pad_t
        #           Since pad_t = V*W:
        #           V*i - V*W + 2*V*W = i*V + V*W
        #           V*i + V*W = i*V + V*W  ✓

if __name__ == "__main__":
    verify_formula()
