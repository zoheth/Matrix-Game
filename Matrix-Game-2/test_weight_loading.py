#!/usr/bin/env python
"""
测试模块化 ActionModule 是否能正确加载原始 checkpoint 的权重
"""
import sys
import os
import torch

# 确保使用模块化版本
os.environ["USE_MODULAR_ACTION"] = "1"

sys.path.insert(0, '/home/xx/dev/Matrix-Game/Matrix-Game-2')

from safetensors.torch import load_file
from wan.modules.modular_action import ActionModule
from wan.modules.modular_action.action_config import ActionConfig

def test_weight_loading():
    print("=" * 70)
    print("测试模块化 ActionModule 权重加载")
    print("=" * 70)

    # 1. 加载 checkpoint
    print("\n1. 加载 checkpoint...")
    checkpoint_path = "/home/xx/models/Matrix-Game-2.0/base_distilled_model/base_distill.safetensors"
    state_dict = load_file(checkpoint_path)
    print(f"   ✓ 加载了 {len(state_dict)} 个权重")

    # 2. 提取 block 0 的 action_model 权重
    print("\n2. 提取 block 0 的 action_model 权重...")
    action_state_dict = {}
    prefix = "model.blocks.0.action_model."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            # 移除前缀
            new_key = key[len(prefix):]
            action_state_dict[new_key] = value

    print(f"   ✓ 提取了 {len(action_state_dict)} 个 action_model 权重")
    print("   原始键名示例:")
    for i, key in enumerate(sorted(action_state_dict.keys())[:5]):
        print(f"     - {key}")

    # 3. 创建模块化 ActionModule
    print("\n3. 创建模块化 ActionModule...")
    config = ActionConfig(
        mouse_dim_in=2,
        keyboard_dim_in=4,
        hidden_size=128,
        img_hidden_size=1536,
        keyboard_hidden_dim=1024,
        mouse_hidden_dim=1024,
        vae_time_compression_ratio=4,
        windows_size=3,
        heads_num=16,
        qk_norm=True,
        qkv_bias=False,
        enable_mouse=True,
        enable_keyboard=True,
    )
    module = ActionModule(action_config=config)
    print("   ✓ 模块创建成功")

    # 4. 加载权重
    print("\n4. 加载权重（测试 _load_from_state_dict 钩子）...")
    try:
        result = module.load_state_dict(action_state_dict, strict=True)
        print(f"   ✓ 权重加载成功！")

        if result.missing_keys:
            print(f"   ⚠ Missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"   ⚠ Unexpected keys: {result.unexpected_keys}")

        if not result.missing_keys and not result.unexpected_keys:
            print("   ✓ 所有权重完美匹配！")

    except Exception as e:
        print(f"   ✗ 权重加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. 验证权重形状
    print("\n5. 验证权重形状...")
    shape_correct = True
    for name, param in module.named_parameters():
        expected_shape = param.shape
        print(f"   ✓ {name}: {expected_shape}")

    print("\n" + "=" * 70)
    print("✓ 权重加载测试通过！")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_weight_loading()
    sys.exit(0 if success else 1)
