"""Debug first ResidualBlock in upsamples"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import os
from demo_utils.new_vae_wrapper import NewVAEDecoderWrapper
from demo_utils.vae_block3 import VAEDecoderWrapper as OldVAEDecoderWrapper

device = torch.device('cuda')

# Load weights
pretrained_path = "/home/xx/models/Matrix-Game-2.0"
vae_weights_path = os.path.join(pretrained_path, "Wan2.1_VAE.pth")
state_dict = torch.load(vae_weights_path, map_location="cpu")

# Get decoder weights only
decoder_state = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}

# Initialize wrappers
old_wrapper = OldVAEDecoderWrapper().to(device, torch.float16).eval()
old_wrapper.load_state_dict(state_dict)

new_wrapper = NewVAEDecoderWrapper().to(device, torch.float16).eval()
new_wrapper.load_state_dict(state_dict)

# Create test input
torch.manual_seed(42)
latent = torch.randn(1, 1, 16, 44, 80, device=device, dtype=torch.float16)

# Process through conv2
z = latent.permute(0, 2, 1, 3, 4)
scale = [new_wrapper.mean.to(device=device, dtype=torch.float16),
         1.0 / new_wrapper.std.to(device=device, dtype=torch.float16)]
z = z / scale[1].view(1, 16, 1, 1, 1) + scale[0].view(1, 16, 1, 1, 1)
x_input = new_wrapper.conv2(z, None)

print("Input to decoder (after conv2):")
print(f"  Shape: {x_input.shape}")
print(f"  Mean: {x_input.mean().item():.6f}, Std: {x_input.std().item():.6f}")

# Process through conv1
old_cache = [None] * 32
new_cache = []

# Initialize cache
from demo_utils.new_vae import CacheState
new_cache = [CacheState(size=50) for _ in range(4)]

with torch.no_grad():
    # Old decoder conv1
    old_x = old_wrapper.decoder.conv1(x_input, old_cache[0])
    print(f"\nAfter conv1 (OLD): mean={old_x.mean().item():.6f}")
    
    # New decoder conv1
    new_x = new_wrapper.decoder.conv1(x_input, new_cache[0])
    print(f"After conv1 (NEW): mean={new_x.mean().item():.6f}")
    print(f"Conv1 diff: {torch.abs(old_x - new_x).max().item():.8f}")
    
    # Process through middle blocks
    old_feat_idx = [1]  # conv1 used slot 0
    for i, layer in enumerate(old_wrapper.decoder.middle):
        layer_type = type(layer).__name__
        if hasattr(layer, 'forward'):
            if 'feat_cache' in layer.forward.__code__.co_varnames:
                old_x = layer(old_x, old_cache, old_feat_idx)
            else:
                old_x = layer(old_x)
        print(f"After middle[{i}] (OLD {layer_type}): mean={old_x.mean().item():.6f}")
    
    new_cache[1].reset_index()
    for i, layer in enumerate(new_wrapper.decoder.middle):
        layer_type = type(layer).__name__
        if hasattr(layer, 'forward'):
            if 'cache_state' in layer.forward.__code__.co_varnames:
                new_x = layer(new_x, new_cache[1])
            else:
                new_x = layer(new_x)
        print(f"After middle[{i}] (NEW {layer_type}): mean={new_x.mean().item():.6f}")
    
    print(f"\nAfter all middle blocks diff: {torch.abs(old_x - new_x).max().item():.8f}")
    
    # Now test first upsample ResidualBlock step by step
    print("\n" + "="*80)
    print("FIRST UPSAMPLE RESIDUALBLOCK - STEP BY STEP")
    print("="*80)
    
    old_res = old_wrapper.decoder.upsamples[0]
    new_res = new_wrapper.decoder.upsamples[0]
    
    # Shortcut
    old_h = old_res.shortcut(old_x)
    new_h = new_res.shortcut(new_x)
    print(f"After shortcut: diff={torch.abs(old_h - new_h).max().item():.8f}")
    
    # Process through residual path (old)
    old_y = old_x
    print("\nOLD residual path:")
    for i, layer in enumerate(old_res.residual):
        layer_type = type(layer).__name__
        if layer_type == 'CausalConv3d':
            idx = old_feat_idx[0]
            print(f"  Layer {i} ({layer_type}): using cache slot {idx}")
            cache_x = old_y[:, :, -2:, :, :].clone()
            if cache_x.shape[2] < 2 and old_cache[idx] is not None:
                cache_x = torch.cat([old_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            old_y = layer(old_y, old_cache[idx])
            old_cache[idx] = cache_x
            old_feat_idx[0] += 1
            print(f"    After: mean={old_y.mean().item():.6f}, cache slot {idx} updated")
        else:
            old_y = layer(old_y)
            print(f"  Layer {i} ({layer_type}): mean={old_y.mean().item():.6f}")
    
    # Process through residual path (new)
    new_cache[2].reset_index()
    new_y = new_x
    print("\nNEW residual path:")
    print(f"  norm1: mean={new_y.mean().item():.6f}")
    new_y = new_res.norm1(new_y)
    print(f"    After: mean={new_y.mean().item():.6f}")
    
    new_y = new_res.silu1(new_y)
    print(f"  silu1: mean={new_y.mean().item():.6f}")
    
    idx_before = new_cache[2].idx
    new_y = new_res.conv1(new_y, new_cache[2])
    idx_after = new_cache[2].idx
    print(f"  conv1: used cache slot {idx_before}, now at {idx_after}, mean={new_y.mean().item():.6f}")
    
    new_y = new_res.norm2(new_y)
    print(f"  norm2: mean={new_y.mean().item():.6f}")
    
    new_y = new_res.silu2(new_y)
    print(f"  silu2: mean={new_y.mean().item():.6f}")
    
    new_y = new_res.dropout(new_y)
    print(f"  dropout: mean={new_y.mean().item():.6f}")
    
    idx_before = new_cache[2].idx
    new_y = new_res.conv2(new_y, new_cache[2])
    idx_after = new_cache[2].idx
    print(f"  conv2: used cache slot {idx_before}, now at {idx_after}, mean={new_y.mean().item():.6f}")
    
    print(f"\nResidual path diff: {torch.abs(old_y - new_y).max().item():.8f}")
    
    # Final output
    old_out = old_h + old_y
    new_out = new_h + new_y
    
    print(f"\nFinal output diff: {torch.abs(old_out - new_out).max().item():.8f}")
    print(f"OLD output: mean={old_out.mean().item():.6f}, std={old_out.std().item():.6f}")
    print(f"NEW output: mean={new_out.mean().item():.6f}, std={new_out.std().item():.6f}")
