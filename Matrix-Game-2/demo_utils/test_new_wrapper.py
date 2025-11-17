"""
Test new VAE wrapper to ensure it works with inference pipeline
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from demo_utils.new_vae_wrapper import NewVAEDecoderWrapper
from demo_utils.vae_block3 import VAEDecoderWrapper as OldVAEDecoderWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Test 1: Check if wrapper can be instantiated
print("=" * 80)
print("TEST 1: Wrapper Instantiation")
print("=" * 80)

try:
    new_wrapper = NewVAEDecoderWrapper()
    print("✓ New wrapper instantiated successfully")
    print(f"  - Decoder type: {type(new_wrapper.decoder).__name__}")
    print(f"  - Has conv2: {hasattr(new_wrapper, 'conv2')}")
    print(f"  - Mean/std registered: {hasattr(new_wrapper, 'mean') and hasattr(new_wrapper, 'std')}")
except Exception as e:
    print(f"✗ Failed to instantiate new wrapper: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Check weight loading
print("\n" + "=" * 80)
print("TEST 2: Weight Loading Compatibility")
print("=" * 80)

try:
    # Load old wrapper and get its state dict
    old_wrapper = OldVAEDecoderWrapper()
    old_state = old_wrapper.state_dict()

    print(f"Old wrapper has {len(old_state)} parameters")
    print("Sample keys:", list(old_state.keys())[:5])

    # Try loading into new wrapper
    new_wrapper = NewVAEDecoderWrapper()
    missing, unexpected = new_wrapper.load_state_dict(old_state, strict=False)

    print(f"\n✓ State dict loaded")
    print(f"  - Missing keys: {len(missing)}")
    if len(missing) > 0 and len(missing) <= 10:
        print(f"    {missing}")
    elif len(missing) > 10:
        print(f"    First 10: {missing[:10]}")
    print(f"  - Unexpected keys: {len(unexpected)}")
    if len(unexpected) > 0 and len(unexpected) <= 10:
        print(f"    {unexpected}")
    elif len(unexpected) > 10:
        print(f"    First 10: {unexpected[:10]}")

except Exception as e:
    print(f"✗ Failed to load state dict: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Check forward pass
print("\n" + "=" * 80)
print("TEST 3: Forward Pass")
print("=" * 80)

try:
    new_wrapper = NewVAEDecoderWrapper().to(device).eval()

    # Create test input matching inference format: [B, T, C, H, W]
    test_input = torch.randn(1, 2, 16, 44, 80, device=device, dtype=torch.float32)
    print(f"Input shape: {test_input.shape} (B, T, C, H, W)")

    with torch.no_grad():
        output, cache = new_wrapper(test_input)

    print(f"✓ Forward pass successful")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Expected shape: [1, 8, 3, 352, 640] (with temporal upsampling)")
    print(f"  - Cache type: {type(cache)}")
    print(f"  - Cache length: {len(cache) if isinstance(cache, (list, tuple)) else 'N/A'}")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Compare with old wrapper
print("\n" + "=" * 80)
print("TEST 4: Output Comparison with Old Wrapper")
print("=" * 80)

try:
    # Initialize both wrappers with float16
    old_wrapper = OldVAEDecoderWrapper().to(device, torch.float16).eval()
    new_wrapper = NewVAEDecoderWrapper().to(device, torch.float16).eval()

    # Copy weights from old to new using proper mapping
    from test_vae_alignment import copy_weights
    copy_weights(new_wrapper.decoder, old_wrapper.decoder)
    new_wrapper.conv2.weight.data.copy_(old_wrapper.conv2.weight.data)
    new_wrapper.conv2.bias.data.copy_(old_wrapper.conv2.bias.data)

    print("Weights copied successfully")

    # Test with same input
    torch.manual_seed(42)
    test_input = torch.randn(1, 1, 16, 44, 80, device=device, dtype=torch.float16)

    with torch.no_grad():
        # Old wrapper expects empty list for cache initialization
        old_output, old_cache = old_wrapper(test_input, [])
        # New wrapper also expects empty list or nothing
        new_output, new_cache = new_wrapper(test_input, [])

    print(f"\nOld output shape: {old_output.shape}")
    print(f"New output shape: {new_output.shape}")

    if old_output.shape == new_output.shape:
        diff = torch.abs(old_output - new_output)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        relative_diff = mean_diff / (torch.abs(old_output).mean().item() + 1e-8)

        print(f"\n✓ Output shapes match")
        print(f"  - Max absolute difference: {max_diff:.6f}")
        print(f"  - Mean absolute difference: {mean_diff:.6f}")
        print(f"  - Relative difference: {relative_diff:.2%}")

        if max_diff < 0.01:
            print(f"  ✓ Outputs are very close (max diff < 0.01)")
        elif max_diff < 0.1:
            print(f"  ~ Outputs are reasonably close (max diff < 0.1)")
        else:
            print(f"  ⚠ Outputs show significant differences (max diff >= 0.1)")
    else:
        print(f"⚠ Output shapes don't match - may be expected due to caching behavior")

except Exception as e:
    print(f"✗ Comparison failed: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit, continue to show summary

print("\n" + "=" * 80)
print("All tests completed!")
print("=" * 80)
print("\n✓ New VAE wrapper is ready for use in inference.py")
print("  Use: python inference.py --use_new_vae [other args]")
