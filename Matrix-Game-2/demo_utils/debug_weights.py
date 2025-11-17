"""
Debug script to find where outputs diverge
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from demo_utils.new_vae_wrapper import NewVAEDecoderWrapper
from demo_utils.vae_block3 import VAEDecoderWrapper as OldVAEDecoderWrapper

device = torch.device('cuda')

# Load real weights
vae_state_dict = torch.load("/home/xx/models/Matrix-Game-2.0/Wan2.1_VAE.pth", map_location="cpu")
decoder_state_dict = {k: v for k, v in vae_state_dict.items()
                      if 'decoder.' in k or 'conv2' in k}

# Initialize wrappers
old_wrapper = OldVAEDecoderWrapper()
old_wrapper.load_state_dict(decoder_state_dict)
old_wrapper.to(device, torch.float16).eval()

new_wrapper = NewVAEDecoderWrapper()
new_wrapper.load_state_dict(decoder_state_dict)
new_wrapper.to(device, torch.float16).eval()

print("Wrappers loaded successfully\n")

# Test with a single frame
torch.manual_seed(42)
latent = torch.randn(1, 1, 16, 44, 80, device=device, dtype=torch.float16) * 0.5

print("Input latent:")
print(f"  Shape: {latent.shape}")
print(f"  Mean: {latent.mean().item():.6f}, Std: {latent.std().item():.6f}")
print(f"  Min: {latent.min().item():.6f}, Max: {latent.max().item():.6f}\n")

# Process through preprocessing (up to decoder input)
print("="*80)
print("PREPROCESSING (up to decoder input)")
print("="*80)

with torch.no_grad():
    # OLD: preprocessing
    z_old = latent.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
    scale = [old_wrapper.mean.to(device=device, dtype=torch.float16),
             1.0 / old_wrapper.std.to(device=device, dtype=torch.float16)]
    z_old = z_old / scale[1].view(1, 16, 1, 1, 1) + scale[0].view(1, 16, 1, 1, 1)
    x_old = old_wrapper.conv2(z_old)

    print("\nOLD after preprocessing:")
    print(f"  Shape: {x_old.shape}")
    print(f"  Mean: {x_old.mean().item():.6f}, Std: {x_old.std().item():.6f}")
    print(f"  Min: {x_old.min().item():.6f}, Max: {x_old.max().item():.6f}")

    # NEW: preprocessing
    z_new = latent.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
    scale_new = [new_wrapper.mean.to(device=device, dtype=torch.float16),
                 1.0 / new_wrapper.std.to(device=device, dtype=torch.float16)]
    z_new = z_new / scale_new[1].view(1, 16, 1, 1, 1) + scale_new[0].view(1, 16, 1, 1, 1)
    x_new = new_wrapper.conv2(z_new, None)

    print("\nNEW after preprocessing:")
    print(f"  Shape: {x_new.shape}")
    print(f"  Mean: {x_new.mean().item():.6f}, Std: {x_new.std().item():.6f}")
    print(f"  Min: {x_new.min().item():.6f}, Max: {x_new.max().item():.6f}")

    diff_preproc = torch.abs(x_old - x_new).max().item()
    print(f"\nPreprocessing diff: {diff_preproc:.8f}")

# Process through decoder layers step by step
print("\n" + "="*80)
print("DECODER LAYERS")
print("="*80)

old_cache = [None] * 32
new_cache = []
feat_idx = [0]  # Use a list so it can be modified by reference

with torch.no_grad():
    # OLD decoder.conv1
    cache_x_old = x_old[:, :, -2:, :, :].clone()
    h_old = old_wrapper.decoder.conv1(x_old, old_cache[feat_idx[0]])
    old_cache[feat_idx[0]] = cache_x_old
    feat_idx[0] += 1

    print("\nOLD after decoder.conv1:")
    print(f"  Mean: {h_old.mean().item():.6f}, Std: {h_old.std().item():.6f}")
    print(f"  Min: {h_old.min().item():.6f}, Max: {h_old.max().item():.6f}")

    # NEW decoder.conv1
    from demo_utils.new_vae import CacheState
    if len(new_cache) == 0:
        new_cache = [CacheState(size=50) for _ in range(4)]
        # Reset indices only when initializing
        for cs in new_cache:
            cs.reset_index()

    h_new = new_wrapper.decoder.conv1(x_new, new_cache[0])

    print("\nNEW after decoder.conv1:")
    print(f"  Mean: {h_new.mean().item():.6f}, Std: {h_new.std().item():.6f}")
    print(f"  Min: {h_new.min().item():.6f}, Max: {h_new.max().item():.6f}")

    diff_conv1 = torch.abs(h_old - h_new).max().item()
    print(f"\nConv1 diff: {diff_conv1:.8f}")

    # Process through middle blocks
    for i, (old_layer, new_layer) in enumerate(zip(old_wrapper.decoder.middle, new_wrapper.decoder.middle)):
        layer_type = type(old_layer).__name__

        if hasattr(old_layer, 'forward') and 'feat_cache' in old_layer.forward.__code__.co_varnames:
            h_old = old_layer(h_old, old_cache, feat_idx)
        else:
            h_old = old_layer(h_old)

        if hasattr(new_layer, 'forward') and 'cache_state' in new_layer.forward.__code__.co_varnames:
            # Don't reset index! Let it accumulate across layers
            h_new = new_layer(h_new, new_cache[1])
        else:
            h_new = new_layer(h_new)

        diff_layer = torch.abs(h_old - h_new).max().item()
        print(f"\nAfter middle[{i}] ({layer_type}):")
        print(f"  OLD: mean={h_old.mean().item():.6f}, max={h_old.max().item():.6f}")
        print(f"  NEW: mean={h_new.mean().item():.6f}, max={h_new.max().item():.6f}")
        print(f"  Diff: {diff_layer:.8f}")

        if diff_layer > 0.01:
            print(f"  ⚠ WARNING: Large difference detected!")

    # Process through upsamples
    print("\n" + "="*80)
    print("UPSAMPLE LAYERS")
    print("="*80)

    for i, (old_layer, new_layer) in enumerate(zip(old_wrapper.decoder.upsamples, new_wrapper.decoder.upsamples)):
        layer_type = type(old_layer).__name__

        if hasattr(old_layer, 'forward'):
            if 'feat_cache' in old_layer.forward.__code__.co_varnames:
                h_old = old_layer(h_old, old_cache, feat_idx)
            else:
                h_old = old_layer(h_old)

        if hasattr(new_layer, 'forward'):
            if 'cache_state' in new_layer.forward.__code__.co_varnames:
                for cs in new_cache:
                    cs.reset_index()
                h_new = new_layer(h_new, new_cache[2])
            else:
                h_new = new_layer(h_new)

        diff_layer = torch.abs(h_old - h_new).max().item()
        mean_diff = torch.abs(h_old - h_new).mean().item()

        print(f"\nAfter upsamples[{i}] ({layer_type}):")
        print(f"  OLD: shape={h_old.shape}, mean={h_old.mean().item():.6f}, max={h_old.max().item():.6f}")
        print(f"  NEW: shape={h_new.shape}, mean={h_new.mean().item():.6f}, max={h_new.max().item():.6f}")
        print(f"  Max diff: {diff_layer:.8f}, Mean diff: {mean_diff:.8f}")

        if diff_layer > 0.1:
            print(f"  ✗✗✗ LARGE DIFFERENCE DETECTED! ✗✗✗")
            print(f"\n  Detailed analysis:")
            print(f"    OLD range: [{h_old.min().item():.6f}, {h_old.max().item():.6f}]")
            print(f"    NEW range: [{h_new.min().item():.6f}, {h_new.max().item():.6f}]")
            print(f"    OLD std: {h_old.std().item():.6f}")
            print(f"    NEW std: {h_new.std().item():.6f}")
            break

        if i >= 5:  # Check first few layers to narrow down
            print("\n  ... (continuing with remaining layers)")
            break

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
