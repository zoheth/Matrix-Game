"""Test RmsNorm differences"""
import torch
import torch.nn.functional as F
from new_vae import RmsNorm as NewRms
from wan.modules.vae import RMS_norm as OldRms

# Create simple test input
torch.manual_seed(42)
x = torch.randn(1, 4, 2, 8, 8)  # (B, C, T, H, W)

print("Input shape:", x.shape)
print("Input sample:", x[0, 0, 0, 0, :5])

# Create both RmsNorm variants
new_rms = NewRms(4, channel_first=True)
old_rms = OldRms(4, channel_first=True, images=False)

# Initialize with same gamma values
gamma_val = torch.ones(4)
new_rms.gamma.data.copy_(gamma_val)
old_rms.gamma.data.copy_(gamma_val.view(4, 1, 1, 1))

print("\nNew RmsNorm gamma shape:", new_rms.gamma.shape)
print("Old RmsNorm gamma shape:", old_rms.gamma.shape)

# Test forward pass
with torch.no_grad():
    new_out = new_rms(x)
    old_out = old_rms(x)

print("\nNew output shape:", new_out.shape)
print("Old output shape:", old_out.shape)

print("\nNew output sample:", new_out[0, 0, 0, 0, :5])
print("Old output sample:", old_out[0, 0, 0, 0, :5])

diff = torch.abs(new_out - old_out)
print("\nMax diff:", diff.max().item())
print("Mean diff:", diff.mean().item())
print("Outputs match:", torch.allclose(new_out, old_out, atol=1e-5))
