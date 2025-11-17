"""Test layer-by-layer to find where differences occur"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from demo_utils.new_vae import VaeDecoder3d as NewVaeDecoder3d
from demo_utils.vae_block3 import VAEDecoder3d as OldVAEDecoder3d
from demo_utils.test_vae_alignment import copy_weights, initialize_old_cache

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
config = {
    'dim': 96,
    'z_dim': 16,
    'dim_mult': [1, 2, 4, 4],
    'num_res_blocks': 2,
    'attn_scales': [],
    'dropout': 0.0
}

new_model = NewVaeDecoder3d(
    dim=config['dim'],
    z_dim=config['z_dim'],
    dim_mult=config['dim_mult'],
    num_res_blocks=config['num_res_blocks'],
    attn_scales=config['attn_scales'],
    temporal_upsample=[True, True, False],
    dropout=config['dropout']
).to(device).eval()

old_model = OldVAEDecoder3d(
    dim=config['dim'],
    z_dim=config['z_dim'],
    dim_mult=config['dim_mult'],
    num_res_blocks=config['num_res_blocks'],
    attn_scales=config['attn_scales'],
    temperal_upsample=[True, True, False],
    dropout=config['dropout']
).to(device).eval()

copy_weights(new_model, old_model)

# Test input
torch.manual_seed(42)
x = torch.randn(1, 16, 1, 32, 32, device=device)

print("Input shape:", x.shape)
print("Input mean:", x.mean().item(), "std:", x.std().item())

with torch.no_grad():
    # Test conv1
    new_x = new_model.conv1(x, None)
    old_feat_cache = initialize_old_cache(old_model, 1, device)
    old_feat_idx = [0]

    cache_x = x[:, :, -2:, :, :].clone()
    old_x = old_model.conv1(x, old_feat_cache[0])
    old_feat_cache[0] = cache_x
    old_feat_idx[0] += 1

    print("\n=== After conv1 ===")
    print("New shape:", new_x.shape, "mean:", new_x.mean().item(), "std:", new_x.std().item())
    print("Old shape:", old_x.shape, "mean:", old_x.mean().item(), "std:", old_x.std().item())
    diff = torch.abs(new_x - old_x)
    print("Max diff:", diff.max().item(), "Mean diff:", diff.mean().item())

    # Test middle blocks
    for i, (new_layer, old_layer) in enumerate(zip(new_model.middle, old_model.middle)):
        layer_type = type(new_layer).__name__
        if layer_type == 'ResidualBlock':
            new_x = new_layer(new_x, None)
            old_x = old_layer(old_x, old_feat_cache, old_feat_idx)
        else:
            new_x = new_layer(new_x)
            old_x = old_layer(old_x)

        print(f"\n=== After middle[{i}] ({layer_type}) ===")
        print("New shape:", new_x.shape, "mean:", new_x.mean().item(), "std:", new_x.std().item())
        print("Old shape:", old_x.shape, "mean:", old_x.mean().item(), "std:", old_x.std().item())
        diff = torch.abs(new_x - old_x)
        print("Max diff:", diff.max().item(), "Mean diff:", diff.mean().item())

    # Test first few upsamples
    for i in range(min(3, len(new_model.upsamples))):
        new_layer = new_model.upsamples[i]
        old_layer = old_model.upsamples[i]
        layer_type = type(new_layer).__name__

        if layer_type == 'ResidualBlock':
            new_x = new_layer(new_x, None)
            old_x = old_layer(old_x, old_feat_cache, old_feat_idx)
        elif layer_type == 'Resample':
            new_x = new_layer(new_x, None)
            old_x = old_layer(old_x, old_feat_cache, old_feat_idx)
        else:
            new_x = new_layer(new_x)
            old_x = old_layer(old_x, old_feat_cache, old_feat_idx)

        print(f"\n=== After upsamples[{i}] ({layer_type}) ===")
        print("New shape:", new_x.shape, "mean:", new_x.mean().item(), "std:", new_x.std().item())
        print("Old shape:", old_x.shape, "mean:", old_x.mean().item(), "std:", old_x.std().item())
        diff = torch.abs(new_x - old_x)
        print("Max diff:", diff.max().item(), "Mean diff:", diff.mean().item())
