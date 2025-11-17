"""
End-to-end test mimicking actual inference pipeline
Tests the full VAE decoder workflow with realistic data
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import torch
import numpy as np
from demo_utils.new_vae_wrapper import NewVAEDecoderWrapper
from demo_utils.vae_block3 import VAEDecoderWrapper as OldVAEDecoderWrapper

def test_with_actual_weights():
    """Test with actual pretrained weights"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    pretrained_path = "/home/xx/models/Matrix-Game-2.0"
    vae_weights_path = os.path.join(pretrained_path, "Wan2.1_VAE.pth")

    if not os.path.exists(vae_weights_path):
        print(f"⚠ Warning: Pretrained weights not found at {vae_weights_path}")
        print("Skipping test with actual weights")
        return False

    print("=" * 80)
    print("TEST: End-to-End with Actual Pretrained Weights")
    print("=" * 80)

    # Load weights
    print("\nLoading pretrained weights...")
    vae_state_dict = torch.load(vae_weights_path, map_location="cpu")
    decoder_state_dict = {}
    for key, value in vae_state_dict.items():
        if 'decoder.' in key or 'conv2' in key:
            decoder_state_dict[key] = value

    print(f"Loaded {len(decoder_state_dict)} decoder parameters")

    # Initialize both wrappers
    print("\nInitializing models...")
    old_wrapper = OldVAEDecoderWrapper()
    old_wrapper.load_state_dict(decoder_state_dict)
    old_wrapper.to(device, torch.float16)
    old_wrapper.eval()
    print("✓ Old wrapper loaded")

    new_wrapper = NewVAEDecoderWrapper()
    new_wrapper.load_state_dict(decoder_state_dict)
    new_wrapper.to(device, torch.float16)
    new_wrapper.eval()
    print("✓ New wrapper loaded")

    #
    # 插入此验证步骤 - 使用 copy_weights 函数验证
    #
    print("\n" + "=" * 80)
    print("VERIFYING WEIGHTS AFTER LOAD_STATE_DICT")
    print("=" * 80)

    # Create a reference new wrapper and manually copy weights using copy_weights
    # This gives us a reference to compare against
    from demo_utils.test_vae_alignment import copy_weights
    reference_wrapper = NewVAEDecoderWrapper().to(device, torch.float16)
    reference_wrapper.eval()

    # Copy weights from old to reference using the tested copy_weights function
    copy_weights(reference_wrapper.decoder, old_wrapper.decoder)
    reference_wrapper.conv2.weight.data.copy_(old_wrapper.conv2.weight.data)
    reference_wrapper.conv2.bias.data.copy_(old_wrapper.conv2.bias.data)

    print("   Created reference wrapper with copy_weights")

    # Now compare new_wrapper (loaded via load_state_dict) with reference_wrapper
    weights_match = True
    for (n_name, n_param), (r_name, r_param) in zip(
        new_wrapper.named_parameters(),
        reference_wrapper.named_parameters()
    ):
        if n_param.shape != r_param.shape:
            print(f"   ✗ Shape mismatch: {n_name} (New: {n_param.shape}, Ref: {r_param.shape})")
            weights_match = False
            continue

        if not torch.allclose(n_param.data, r_param.data, atol=1e-6):
            max_diff = (n_param.data - r_param.data).abs().max().item()
            print(f"   ✗ Weights differ: {n_name}")
            print(f"     Max diff: {max_diff}")
            weights_match = False

    if weights_match:
        print("   ✓ All weights match after load_state_dict")
    else:
        print("   ✗ CRITICAL: Weights DO NOT match after load_state_dict")
        # 如果权重不匹配，测试几乎肯定会失败
    print("=" * 80 + "\n")

    if not weights_match:
        return False # 提前退出

    # Test with realistic latent input
    print("\n" + "-" * 80)
    print("Testing frame-by-frame decoding (realistic scenario)")
    print("-" * 80)

    # Simulate realistic latent: [B, T, C, H, W] where T=1 (frame-by-frame)
    torch.manual_seed(42)

    # Realistic latent statistics (normalized)
    latent_frames = []
    num_frames = 3

    for i in range(num_frames):
        # Each frame: [1, 1, 16, 44, 80]
        latent = torch.randn(1, 1, 16, 44, 80, device=device, dtype=torch.float16)
        # Apply realistic statistics (latents are usually normalized)
        latent = latent * 0.5
        latent_frames.append(latent)

    print(f"\nGenerated {num_frames} latent frames of shape [1, 1, 16, 44, 80]")

    # Decode with old wrapper
    print("\n1. Decoding with OLD wrapper:")
    old_outputs = []
    old_cache = [None] * 32  # Old wrapper needs 32 cache slots

    with torch.no_grad():
        for i, latent in enumerate(latent_frames):
            output, old_cache = old_wrapper(latent, *old_cache)
            old_outputs.append(output)
            print(f"   Frame {i+1}: output shape {output.shape}, "
                  f"mean={output.mean().item():.4f}, "
                  f"std={output.std().item():.4f}, "
                  f"min={output.min().item():.4f}, "
                  f"max={output.max().item():.4f}")

    old_video = torch.cat(old_outputs, dim=1)
    print(f"   Final video shape: {old_video.shape}")

    # Decode with new wrapper
    print("\n2. Decoding with NEW wrapper:")
    new_outputs = []
    new_cache = []  # New wrapper initializes cache internally

    with torch.no_grad():
        for i, latent in enumerate(latent_frames):
            output, new_cache = new_wrapper(latent, *new_cache if len(new_cache) > 0 else [])
            new_outputs.append(output)
            print(f"   Frame {i+1}: output shape {output.shape}, "
                  f"mean={output.mean().item():.4f}, "
                  f"std={output.std().item():.4f}, "
                  f"min={output.min().item():.4f}, "
                  f"max={output.max().item():.4f}")

    new_video = torch.cat(new_outputs, dim=1)
    print(f"   Final video shape: {new_video.shape}")

    # Compare outputs
    print("\n" + "-" * 80)
    print("Comparison Results")
    print("-" * 80)

    if old_video.shape != new_video.shape:
        print(f"✗ SHAPE MISMATCH!")
        print(f"   Old: {old_video.shape}")
        print(f"   New: {new_video.shape}")
        return False

    # Frame-by-frame comparison
    print("\nFrame-by-frame analysis:")
    all_good = True

    for i in range(len(old_outputs)):
        old_frame = old_outputs[i]
        new_frame = new_outputs[i]

        diff = torch.abs(old_frame - new_frame)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Check if values are in valid range [-1, 1]
        old_in_range = (old_frame >= -1.1).all() and (old_frame <= 1.1).all()
        new_in_range = (new_frame >= -1.1).all() and (new_frame <= 1.1).all()

        status = "✓" if max_diff < 0.1 else "✗"
        range_status = "✓" if (old_in_range and new_in_range) else "✗"

        print(f"   Frame {i+1}: {status} max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} | "
              f"Range check: {range_status}")

        if max_diff >= 0.1:
            all_good = False
            print(f"      Old range: [{old_frame.min().item():.4f}, {old_frame.max().item():.4f}]")
            print(f"      New range: [{new_frame.min().item():.4f}, {new_frame.max().item():.4f}]")

    # Overall statistics
    total_diff = torch.abs(old_video - new_video)
    max_total = total_diff.max().item()
    mean_total = total_diff.mean().item()
    relative = mean_total / (torch.abs(old_video).mean().item() + 1e-8)

    print(f"\nOverall Statistics:")
    print(f"   Max absolute difference: {max_total:.6f}")
    print(f"   Mean absolute difference: {mean_total:.6f}")
    print(f"   Relative difference: {relative:.2%}")

    # Visual check: convert to uint8 and compare
    print(f"\nVisual Quality Check:")
    old_uint8 = ((old_video.float() + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
    new_uint8 = ((new_video.float() + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)

    uint8_diff = np.abs(old_uint8.astype(np.int16) - new_uint8.astype(np.int16))
    max_uint8_diff = uint8_diff.max()
    mean_uint8_diff = uint8_diff.mean()

    print(f"   Max uint8 difference: {max_uint8_diff} (out of 255)")
    print(f"   Mean uint8 difference: {mean_uint8_diff:.2f}")

    if max_uint8_diff > 20:
        print(f"   ⚠ WARNING: Large visual differences detected!")
        all_good = False
    elif max_uint8_diff > 5:
        print(f"   ~ Minor visual differences (may be acceptable)")
    else:
        print(f"   ✓ Visually very similar")

    # Check for common issues
    print(f"\nDiagnostics:")

    # Check if outputs are all zeros
    if torch.allclose(new_video, torch.zeros_like(new_video), atol=0.01):
        print("   ✗ ERROR: New wrapper output is all zeros!")
        all_good = False
    elif torch.allclose(old_video, torch.zeros_like(old_video), atol=0.01):
        print("   ✗ ERROR: Old wrapper output is all zeros!")
        all_good = False
    else:
        print("   ✓ Outputs are non-zero")

    # Check if outputs are constant
    if new_video.std() < 0.01:
        print("   ✗ ERROR: New wrapper output is nearly constant!")
        all_good = False
    else:
        print(f"   ✓ New wrapper output has variation (std={new_video.std().item():.4f})")

    # Check for NaN or Inf
    if torch.isnan(new_video).any() or torch.isinf(new_video).any():
        print("   ✗ ERROR: New wrapper output contains NaN or Inf!")
        all_good = False
    else:
        print("   ✓ No NaN or Inf values")

    print("\n" + "=" * 80)
    if all_good:
        print("✓ END-TO-END TEST PASSED")
    else:
        print("✗ END-TO-END TEST FAILED")
    print("=" * 80)

    return all_good


def test_individual_components():
    """Test individual components to isolate issues"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 80)
    print("TEST: Individual Component Analysis")
    print("=" * 80)

    pretrained_path = "/home/xx/models/Matrix-Game-2.0"
    vae_weights_path = os.path.join(pretrained_path, "Wan2.1_VAE.pth")

    if not os.path.exists(vae_weights_path):
        print(f"⚠ Skipping - weights not found")
        return

    # Load weights
    vae_state_dict = torch.load(vae_weights_path, map_location="cpu")
    decoder_state_dict = {}
    for key, value in vae_state_dict.items():
        if 'decoder.' in key or 'conv2' in key:
            decoder_state_dict[key] = value

    # Test conv2 separately
    print("\n1. Testing conv2 (input preprocessing):")

    from demo_utils.new_vae import CausalConv3d as NewCausalConv3d
    from wan.modules.vae import CausalConv3d as OldCausalConv3d

    old_conv2 = OldCausalConv3d(16, 16, 1)
    old_conv2.load_state_dict({
        'weight': decoder_state_dict['conv2.weight'],
        'bias': decoder_state_dict['conv2.bias']
    })
    old_conv2.to(device, torch.float16).eval()

    new_conv2 = NewCausalConv3d(16, 16, 1)
    new_conv2.weight.data.copy_(decoder_state_dict['conv2.weight'])
    new_conv2.bias.data.copy_(decoder_state_dict['conv2.bias'])
    new_conv2.to(device, torch.float16).eval()

    test_input = torch.randn(1, 16, 1, 44, 80, device=device, dtype=torch.float16)

    with torch.no_grad():
        old_out = old_conv2(test_input, None)
        new_out = new_conv2(test_input, None)

    conv2_diff = torch.abs(old_out - new_out).max().item()
    print(f"   Conv2 max diff: {conv2_diff:.8f}")

    if conv2_diff < 1e-5:
        print("   ✓ Conv2 outputs match perfectly")
    else:
        print(f"   ✗ Conv2 has differences!")

    # Test decoder first layer
    print("\n2. Testing decoder.conv1:")

    from demo_utils.new_vae import VaeDecoder3d as NewDecoder
    from demo_utils.vae_block3 import VAEDecoder3d as OldDecoder

    old_decoder = OldDecoder()
    new_decoder = NewDecoder()

    # Load only conv1 weights
    old_decoder.conv1.load_state_dict({
        'weight': decoder_state_dict['decoder.conv1.weight'],
        'bias': decoder_state_dict['decoder.conv1.bias']
    })
    new_decoder.conv1.load_state_dict({
        'weight': decoder_state_dict['decoder.conv1.weight'],
        'bias': decoder_state_dict['decoder.conv1.bias']
    })

    old_decoder.to(device, torch.float16).eval()
    new_decoder.to(device, torch.float16).eval()

    with torch.no_grad():
        old_h = old_decoder.conv1(new_out, None)
        new_h = new_decoder.conv1(new_out, None)

    conv1_diff = torch.abs(old_h - new_h).max().item()
    print(f"   Decoder.conv1 max diff: {conv1_diff:.8f}")

    if conv1_diff < 1e-5:
        print("   ✓ Decoder.conv1 outputs match perfectly")
    else:
        print(f"   ✗ Decoder.conv1 has differences!")
        print(f"      Old mean: {old_h.mean().item():.6f}, std: {old_h.std().item():.6f}")
        print(f"      New mean: {new_h.mean().item():.6f}, std: {new_h.std().item():.6f}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("VAE DECODER END-TO-END TESTING")
    print("=" * 80)

    # Test with actual weights
    success = test_with_actual_weights()

    # If test failed, run component analysis
    if not success:
        test_individual_components()

    print("\n")
