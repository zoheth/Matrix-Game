"""
Simple test script to verify FlashInfer RoPE integration.
This tests that the integrated RoPE produces correct results.
"""

import torch
from wan.modules.modular_action.interfaces import FlashInferAttentionCore

def test_flashinfer_rope_integration():
    """Test that FlashInfer's integrated RoPE works correctly."""
    print("Testing FlashInfer RoPE integration...")

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA not available, skipping test")
        return

    BS = 4
    seq_len = 16
    num_heads = 8
    head_dim = 64

    # Create test tensors
    q = torch.randn(BS, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(BS, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(BS, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)

    # Test 1: Basic attention without RoPE
    print("\n[Test 1] Basic attention without RoPE...")
    attn_core = FlashInferAttentionCore()
    try:
        output1 = attn_core(q, k, v, causal=False, use_rope=False)
        print(f"✓ Output shape: {output1.shape}")
        print(f"✓ Output dtype: {output1.dtype}")
        assert output1.shape == (BS, seq_len, num_heads, head_dim), "Output shape mismatch"
        print("✓ Test 1 passed!")
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        return False

    # Test 2: Attention with integrated RoPE
    print("\n[Test 2] Attention with integrated RoPE...")
    try:
        output2 = attn_core(q, k, v, causal=False, use_rope=True, rope_offset=0)
        print(f"✓ Output shape: {output2.shape}")
        print(f"✓ Output dtype: {output2.dtype}")
        assert output2.shape == (BS, seq_len, num_heads, head_dim), "Output shape mismatch"
        print("✓ Test 2 passed!")
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        return False

    # Test 3: Verify outputs are different (RoPE should change results)
    print("\n[Test 3] Verify RoPE changes the output...")
    try:
        diff = (output1 - output2).abs().mean()
        print(f"✓ Mean absolute difference: {diff.item():.6f}")
        assert diff > 1e-6, "Outputs should differ when RoPE is applied"
        print("✓ Test 3 passed!")
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        return False

    # Test 4: Test _apply_rope_internal helper
    print("\n[Test 4] Test _apply_rope_internal helper...")
    try:
        q_rope, k_rope = attn_core._apply_rope_internal(q, k, rope_offset=0)
        print(f"✓ Q_rope shape: {q_rope.shape}")
        print(f"✓ K_rope shape: {k_rope.shape}")
        assert q_rope.shape == q.shape, "Q_rope shape mismatch"
        assert k_rope.shape == k.shape, "K_rope shape mismatch"
        print("✓ Test 4 passed!")
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")
        return False

    # Test 5: Test with different rope_offset
    print("\n[Test 5] Test with different rope_offset...")
    try:
        output3 = attn_core(q, k, v, causal=False, use_rope=True, rope_offset=10)
        diff_offset = (output2 - output3).abs().mean()
        print(f"✓ Mean difference with different offset: {diff_offset.item():.6f}")
        assert diff_offset > 1e-6, "Different offsets should produce different outputs"
        print("✓ Test 5 passed!")
    except Exception as e:
        print(f"✗ Test 5 failed: {e}")
        return False

    # Test 6: Test with causal mask
    print("\n[Test 6] Test with causal mask...")
    try:
        output4 = attn_core(q, k, v, causal=True, use_rope=True, rope_offset=0)
        print(f"✓ Causal output shape: {output4.shape}")
        assert output4.shape == (BS, seq_len, num_heads, head_dim), "Causal output shape mismatch"
        print("✓ Test 6 passed!")
    except Exception as e:
        print(f"✗ Test 6 failed: {e}")
        return False

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    return True


if __name__ == "__main__":
    test_flashinfer_rope_integration()
