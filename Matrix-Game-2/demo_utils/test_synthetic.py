"""
Synthetic test without requiring pretrained weights
Uses identical random initialization to test implementation correctness
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from demo_utils.new_vae_wrapper import NewVAEDecoderWrapper
from demo_utils.vae_block3 import VAEDecoderWrapper as OldVAEDecoderWrapper
from demo_utils.test_vae_alignment import copy_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

print("=" * 80)
print("SYNTHETIC END-TO-END TEST")
print("=" * 80)
print("Testing with random but consistent weights")
print("=" * 80)

# Initialize both wrappers
print("\n1. Initializing models with random weights...")
torch.manual_seed(42)
old_wrapper = OldVAEDecoderWrapper().to(device, torch.float16)
old_wrapper.eval()

torch.manual_seed(42)  # Same seed for reproducibility
new_wrapper = NewVAEDecoderWrapper().to(device, torch.float16)
new_wrapper.eval()

# Copy weights from old to new to ensure they're identical
print("\n2. Copying weights from old to new...")
copy_weights(new_wrapper.decoder, old_wrapper.decoder)
new_wrapper.conv2.weight.data.copy_(old_wrapper.conv2.weight.data)
new_wrapper.conv2.bias.data.copy_(old_wrapper.conv2.bias.data)
print("   ✓ Weights copied")

# Verify weight equality
print("\n3. Verifying weight equality...")
weights_match = True
for (n_name, n_param), (o_name, o_param) in zip(
    new_wrapper.named_parameters(),
    old_wrapper.named_parameters()
):
    if n_param.shape == o_param.shape:
        if not torch.allclose(n_param.data, o_param.data, atol=1e-6):
            print(f"   ✗ Weights differ: {n_name}")
            weights_match = False

if weights_match:
    print("   ✓ All weights match")

# Test frame-by-frame decoding (realistic scenario)
print("\n" + "-" * 80)
print("4. Testing Frame-by-Frame Decoding")
print("-" * 80)

torch.manual_seed(123)  # Different seed for input data
num_frames = 4

latent_frames = []
for i in range(num_frames):
    latent = torch.randn(1, 1, 16, 44, 80, device=device, dtype=torch.float16) * 0.5
    latent_frames.append(latent)

print(f"\nProcessing {num_frames} frames...")

# Decode with old wrapper
print("\nOLD wrapper:")
old_outputs = []
# Old wrapper needs None list with specific size (32 slots)
old_cache = [None] * 32

with torch.no_grad():
    for i, latent in enumerate(latent_frames):
        output, old_cache = old_wrapper(latent, *old_cache)
        old_outputs.append(output)

        stats = {
            'mean': output.mean().item(),
            'std': output.std().item(),
            'min': output.min().item(),
            'max': output.max().item()
        }
        print(f"  Frame {i+1}: shape={output.shape} mean={stats['mean']:.4f} "
              f"std={stats['std']:.4f} range=[{stats['min']:.4f}, {stats['max']:.4f}]")

# Reset for new wrapper
torch.manual_seed(123)  # Same seed to generate identical latents
latent_frames = []
for i in range(num_frames):
    latent = torch.randn(1, 1, 16, 44, 80, device=device, dtype=torch.float16) * 0.5
    latent_frames.append(latent)

# Decode with new wrapper
print("\nNEW wrapper:")
new_outputs = []
# New wrapper initializes cache internally if passed empty list
new_cache = []

with torch.no_grad():
    for i, latent in enumerate(latent_frames):
        output, new_cache = new_wrapper(latent, *new_cache if len(new_cache) > 0 else [])
        new_outputs.append(output)

        stats = {
            'mean': output.mean().item(),
            'std': output.std().item(),
            'min': output.min().item(),
            'max': output.max().item()
        }
        print(f"  Frame {i+1}: shape={output.shape} mean={stats['mean']:.4f} "
              f"std={stats['std']:.4f} range=[{stats['min']:.4f}, {stats['max']:.4f}]")

# Compare outputs
print("\n" + "-" * 80)
print("5. Detailed Comparison")
print("-" * 80)

all_good = True

for i in range(num_frames):
    old_frame = old_outputs[i]
    new_frame = new_outputs[i]

    if old_frame.shape != new_frame.shape:
        print(f"\n✗ Frame {i+1} SHAPE MISMATCH:")
        print(f"  Old: {old_frame.shape}")
        print(f"  New: {new_frame.shape}")
        all_good = False
        continue

    diff = torch.abs(old_frame - new_frame)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    median_diff = diff.median().item()

    # Per-channel analysis
    old_r = old_frame[:, :, 0, :, :].mean().item()
    old_g = old_frame[:, :, 1, :, :].mean().item()
    old_b = old_frame[:, :, 2, :, :].mean().item()

    new_r = new_frame[:, :, 0, :, :].mean().item()
    new_g = new_frame[:, :, 1, :, :].mean().item()
    new_b = new_frame[:, :, 2, :, :].mean().item()

    # Visual (uint8) comparison
    old_uint8 = ((old_frame.float() + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
    new_uint8 = ((new_frame.float() + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
    uint8_diff = np.abs(old_uint8.astype(np.int16) - new_uint8.astype(np.int16))

    print(f"\nFrame {i+1}:")
    print(f"  Floating point:")
    print(f"    Max diff: {max_diff:.6f}")
    print(f"    Mean diff: {mean_diff:.6f}")
    print(f"    Median diff: {median_diff:.6f}")

    print(f"  Per-channel means:")
    print(f"    Old RGB: ({old_r:.4f}, {old_g:.4f}, {old_b:.4f})")
    print(f"    New RGB: ({new_r:.4f}, {new_g:.4f}, {new_b:.4f})")
    print(f"    Diff RGB: ({abs(old_r-new_r):.4f}, {abs(old_g-new_g):.4f}, {abs(old_b-new_b):.4f})")

    print(f"  Visual (uint8):")
    print(f"    Max pixel diff: {uint8_diff.max()}/255")
    print(f"    Mean pixel diff: {uint8_diff.mean():.2f}/255")

    # Determine status
    if max_diff < 0.001:
        status = "✓ EXCELLENT"
    elif max_diff < 0.01:
        status = "✓ GOOD"
    elif max_diff < 0.1:
        status = "~ ACCEPTABLE"
    else:
        status = "✗ POOR"
        all_good = False

    print(f"  Status: {status}")

    # Check for problematic patterns
    if torch.allclose(new_frame, torch.zeros_like(new_frame), atol=0.01):
        print(f"  ✗ ERROR: Output is all zeros!")
        all_good = False

    if new_frame.std() < 0.01:
        print(f"  ✗ ERROR: Output is nearly constant!")
        all_good = False

    if torch.isnan(new_frame).any() or torch.isinf(new_frame).any():
        print(f"  ✗ ERROR: Contains NaN or Inf!")
        all_good = False

# Concatenate all frames
old_video = torch.cat(old_outputs, dim=1)
new_video = torch.cat(new_outputs, dim=1)

print("\n" + "-" * 80)
print("6. Overall Video Statistics")
print("-" * 80)

total_diff = torch.abs(old_video - new_video)
print(f"Total frames: {old_video.shape[1]}")
print(f"Max difference: {total_diff.max().item():.6f}")
print(f"Mean difference: {total_diff.mean().item():.6f}")
print(f"Std difference: {total_diff.std().item():.6f}")
print(f"Relative error: {(total_diff.mean() / (torch.abs(old_video).mean() + 1e-8)).item():.2%}")

# Histogram of differences
diff_flat = total_diff.cpu().numpy().flatten()
percentiles = [50, 90, 95, 99, 99.9]
print(f"\nDifference percentiles:")
for p in percentiles:
    val = np.percentile(diff_flat, p)
    print(f"  {p:5.1f}%: {val:.6f}")

print("\n" + "=" * 80)
if all_good:
    print("✓✓✓ TEST PASSED - Implementations match! ✓✓✓")
else:
    print("✗✗✗ TEST FAILED - Significant differences detected ✗✗✗")
print("=" * 80)

# Additional diagnostics if test failed
if not all_good:
    print("\n" + "-" * 80)
    print("DIAGNOSTIC INFORMATION")
    print("-" * 80)

    # Check cache structure
    print(f"\nCache structures:")
    print(f"  Old cache type: {type(old_cache)}, length: {len(old_cache) if isinstance(old_cache, list) else 'N/A'}")
    print(f"  New cache type: {type(new_cache)}, length: {len(new_cache) if isinstance(new_cache, list) else 'N/A'}")

    # Sample a few pixel locations
    print(f"\nSample pixel values at [0, 0, :, 100, 200]:")
    print(f"  Old: {old_video[0, 0, :, 100, 200].cpu().numpy()}")
    print(f"  New: {new_video[0, 0, :, 100, 200].cpu().numpy()}")
    print(f"  Diff: {(old_video[0, 0, :, 100, 200] - new_video[0, 0, :, 100, 200]).cpu().numpy()}")
