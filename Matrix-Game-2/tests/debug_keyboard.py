#!/usr/bin/env python
"""Debug keyboard injector shape issues"""

import sys
import torch
sys.path.insert(0, '/home/ykx/dev/Matrix-Game/Matrix-Game-2')

from wan.modules.action_module import ActionModule
from wan.modules.modular_action.injectors import KeyboardInjector
from wan.modules.modular_action.action_config import ActionConfig
from wan.modules.posemb_layers import get_nd_rotary_pos_embed

B, H, W = 1, 22, 40
S = H * W
num_feats = 3
num_frames = 9
dtype = torch.bfloat16
device = 'cuda'

print('测试 Keyboard Injector...\n')

x = torch.randn(B, num_feats * S, 1536, device=device, dtype=dtype)
kb = torch.randn(B, num_frames, 6, device=device, dtype=dtype)

# 创建模块
orig = ActionModule(mouse_dim_in=2, keyboard_dim_in=6, hidden_size=128, img_hidden_size=1536,
                    enable_keyboard=True, enable_mouse=False).to(device=device, dtype=dtype).eval()

config = ActionConfig(mouse_dim_in=2, keyboard_dim_in=6, hidden_size=128, img_hidden_size=1536)
kb_inj = KeyboardInjector(config).to(device=device, dtype=dtype).eval()

# 复制权重
kb_inj.preprocessor.keyboard_embed[0].weight.data = orig.keyboard_embed[0].weight.data.clone()
kb_inj.preprocessor.keyboard_embed[0].bias.data = orig.keyboard_embed[0].bias.data.clone()
kb_inj.preprocessor.keyboard_embed[2].weight.data = orig.keyboard_embed[2].weight.data.clone()
kb_inj.preprocessor.keyboard_embed[2].bias.data = orig.keyboard_embed[2].bias.data.clone()
kb_inj.mouse_attn_q.weight.data = orig.mouse_attn_q.weight.data.clone()
kb_inj.keyboard_attn_kv.weight.data = orig.keyboard_attn_kv.weight.data.clone()
kb_inj.q_norm.weight.data = orig.key_attn_q_norm.weight.data.clone()
kb_inj.k_norm.weight.data = orig.key_attn_k_norm.weight.data.clone()
kb_inj.proj_keyboard.weight.data = orig.proj_keyboard.weight.data.clone()

# 获取RoPE
rope_sizes = [num_feats, H // 2, W // 2]
freqs_cos, freqs_sin = get_nd_rotary_pos_embed([8, 28, 28], rope_sizes, theta=256, use_real=True, theta_rescale_factor=1)
total_positions = num_feats * rope_sizes[1] * rope_sizes[2]
freqs_cis = (freqs_cos[-total_positions:], freqs_sin[-total_positions:])

print(f'输入: x={x.shape}, kb={kb.shape}\n')

# 测试非因果模式
print('=== 测试非因果模式 ===')
try:
    with torch.no_grad():
        out_orig = orig(x, tt=num_feats, th=H, tw=W, mouse_condition=None, keyboard_condition=kb,
                       is_causal=False, num_frame_per_block=num_feats)
    print(f'✓ 原始实现: {out_orig.shape}')
except Exception as e:
    print(f'✗ 原始实现失败: {e}')
    import traceback
    traceback.print_exc()

try:
    with torch.no_grad():
        out_new = kb_inj(x, condition=kb, freqs_cis=freqs_cis,
                        spatial_shape=(H, W), temporal_shape=num_feats, is_causal=False)
    print(f'✓ 模块化实现: {out_new.shape}')
    diff = (out_orig - out_new).abs().max().item()
    rel_err = diff / out_orig.abs().max().item()
    print(f'  差异: max={diff:.3e}, rel={rel_err:.2%}')
except Exception as e:
    print(f'✗ 模块化实现失败: {e}')
    import traceback
    traceback.print_exc()

# 测试因果模式
print('\n=== 测试因果模式（带 KV cache）===')

def init_cache(B, S, H, D, size, device, dtype):
    return {
        'k': torch.zeros(B*S, size, H, D, device=device, dtype=dtype),
        'v': torch.zeros(B*S, size, H, D, device=device, dtype=dtype),
        'global_end_index': torch.tensor([0], device=device),
        'local_end_index': torch.tensor([0], device=device),
    }

cache_orig = init_cache(B, S, 16, 64, 20, device, dtype)
cache_new = init_cache(B, S, 16, 64, 20, device, dtype)

try:
    with torch.no_grad():
        out_orig = orig(x, tt=num_feats, th=H, tw=W, mouse_condition=None, keyboard_condition=kb,
                       is_causal=True, kv_cache_keyboard=cache_orig,
                       start_frame=0, num_frame_per_block=num_feats)
    print(f'✓ 原始实现: {out_orig.shape}')
    print(f'  Cache: global_idx={cache_orig["global_end_index"].item()}, local_idx={cache_orig["local_end_index"].item()}')
except Exception as e:
    print(f'✗ 原始实现失败: {e}')
    import traceback
    traceback.print_exc()

try:
    with torch.no_grad():
        out_new = kb_inj(x, condition=kb, freqs_cis=freqs_cis,
                        spatial_shape=(H, W), temporal_shape=num_feats,
                        is_causal=True, kv_cache=cache_new,
                        start_frame=0, num_frame_per_block=num_feats)
    print(f'✓ 模块化实现: {out_new.shape}')
    print(f'  Cache: global_idx={cache_new["global_end_index"].item()}, local_idx={cache_new["local_end_index"].item()}')
    diff = (out_orig - out_new).abs().max().item()
    rel_err = diff / out_orig.abs().max().item()
    print(f'  差异: max={diff:.3e}, rel={rel_err:.2%}')
except Exception as e:
    print(f'✗ 模块化实现失败: {e}')
    import traceback
    traceback.print_exc()
