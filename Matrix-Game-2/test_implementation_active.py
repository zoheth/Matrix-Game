"""Verify that Triton implementation is actually being used"""
import torch
from wan.modules.modular_action.injectors import MouseInjector
from wan.modules.modular_action.action_config import ActionConfig

# Monkey-patch to detect which implementation is being called
triton_called = False
pytorch_called = False

from wan.modules.modular_action import kernels
original_triton = kernels.preprocessor_kernel.mouse_preprocessor_triton

def tracked_triton(*args, **kwargs):
    global triton_called
    triton_called = True
    print("✓ Triton implementation called!")
    return original_triton(*args, **kwargs)

kernels.preprocessor_kernel.mouse_preprocessor_triton = tracked_triton

# Also monkey-patch the injectors module since it imports at the top
import wan.modules.modular_action.injectors as injectors_module
injectors_module.mouse_preprocessor_triton = tracked_triton

from wan.modules.modular_action.injectors import MousePreprocessor
original_pytorch = MousePreprocessor.forward

def tracked_pytorch(self, *args, **kwargs):
    global pytorch_called
    pytorch_called = True
    print("✗ PyTorch implementation called!")
    return original_pytorch(self, *args, **kwargs)

MousePreprocessor.forward = tracked_pytorch

# Now test
print("Creating MouseInjector...")
config = ActionConfig(
    img_hidden_size=1536,
    mouse_dim_in=2,
    keyboard_dim_in=4,
    mouse_hidden_dim=1024,
    keyboard_hidden_dim=1024,
    heads_num=16,
    vae_time_compression_ratio=4,
    windows_size=3,
    local_attn_size=6,
    qkv_bias=False,
    qk_norm=True,
)

injector = MouseInjector(config).to('cuda').half().eval()  # Convert to float16

print("\nRunning forward pass...")
B, T, H, W = 1, 3, 22, 40
S = H * W
x = torch.randn(B, T * S, config.img_hidden_size, device='cuda', dtype=torch.float16)
condition = torch.randn(B, 48, config.mouse_dim_in, device='cuda', dtype=torch.float16)

head_dim = config.mouse_head_dim
freqs_cis = (
    torch.randn(100, head_dim, device='cuda', dtype=torch.float16),
    torch.randn(100, head_dim, device='cuda', dtype=torch.float16),
)

with torch.no_grad():
    output = injector(
        x, condition, freqs_cis,
        spatial_shape=(H, W),
        temporal_shape=T,
        is_causal=True,
        num_frame_per_block=3,
    )

print(f"\n{'='*60}")
print("Implementation Usage Summary:")
print(f"{'='*60}")
print(f"Triton called: {triton_called}")
print(f"PyTorch called: {pytorch_called}")

if triton_called and not pytorch_called:
    print("\n✓ SUCCESS: Using Triton implementation")
elif pytorch_called and not triton_called:
    print("\n✗ FAIL: Using PyTorch implementation")
else:
    print("\n? UNCLEAR: Both or neither called")
