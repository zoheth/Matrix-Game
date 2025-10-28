#!/usr/bin/env python
"""最小化测试 - 快速验证基本功能"""

import sys
import torch
sys.path.insert(0, '/home/ykx/dev/Matrix-Game/Matrix-Game-2')

print("导入模块...")
from wan.modules.action_module import ActionModule
from wan.modules.modular_action.injectors import MouseInjector
from wan.modules.modular_action.action_config import ActionConfig

print("✓ 导入成功\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"设备: {device}\n")

# 使用实际的推理配置 (来自 inference_universal.yaml)
# image_or_video_shape: [1, 16, 15, 44, 80]
B = 1
latent_frames = 15  # VAE 编码后的帧数
vae_ratio = 4
num_feats = latent_frames  # DiT 的时间 tokens (没有时间 patching)
num_frames = (latent_frames - 1) * vae_ratio + 1  # 57 - action 需要的原始帧数
H, W = 44, 80
S = H * W  # 3520
img_hidden_size = 1536

print(f"配置: num_feats={num_feats}, num_frames={num_frames}, H={H}, W={W}, S={S}")

print("\n创建输入...")
dtype = torch.bfloat16  # Flash Attention 需要 fp16 或 bf16
x = torch.randn(B, num_feats * S, img_hidden_size, device=device, dtype=dtype)
mouse_cond = torch.randn(B, num_frames, 2, device=device, dtype=dtype)
kb_cond = torch.randn(B, num_frames, 6, device=device, dtype=dtype)
print(f"✓ x: {x.shape}, dtype: {x.dtype}, mouse: {mouse_cond.shape}, keyboard: {kb_cond.shape}\n")

print("创建 ActionModule...")
try:
    action_module = ActionModule(
        mouse_dim_in=2,
        keyboard_dim_in=6,
        hidden_size=128,
        img_hidden_size=img_hidden_size,
        enable_keyboard=False,
        enable_mouse=True,
    ).to(device=device, dtype=dtype).eval()
    print("✓ ActionModule 创建成功\n")
except Exception as e:
    print(f"✗ 失败: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("运行 ActionModule...")
try:
    with torch.no_grad():
        out_orig = action_module(
            x, tt=num_feats, th=H, tw=W,
            mouse_condition=mouse_cond,
            keyboard_condition=kb_cond,  # 必须提供
            is_causal=False,
            num_frame_per_block=num_feats  # 非因果模式下处理所有 frames
        )
    print(f"✓ 输出: {out_orig.shape}, norm: {out_orig.norm().item():.4f}\n")
except Exception as e:
    print(f"✗ 失败: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("✅ 所有测试通过！")
