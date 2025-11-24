"""
FlashInfer Integration Module - 渐进式替换入口点

提供三种模式：
1. DISABLED: 完全使用原始实现（默认）
2. VALIDATE: 同时运行两种实现，比较输出差异
3. ENABLED: 完全使用 FlashInfer 实现

使用方法：
    import os
    os.environ["FLASHINFER_MODE"] = "VALIDATE"  # 或 "ENABLED"

    from wan.modules.flashinfer_integration import maybe_use_flashinfer_attention
    model = maybe_use_flashinfer_attention(model)
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Tuple
from enum import Enum


class FlashInferMode(Enum):
    DISABLED = "DISABLED"   # 使用原始实现
    VALIDATE = "VALIDATE"   # 双路径验证
    ENABLED = "ENABLED"     # 使用 FlashInfer


def get_flashinfer_mode() -> FlashInferMode:
    """获取当前 FlashInfer 模式"""
    mode_str = os.environ.get("FLASHINFER_MODE", "DISABLED").upper()
    try:
        return FlashInferMode(mode_str)
    except ValueError:
        print(f"[WARNING] Unknown FLASHINFER_MODE '{mode_str}', defaulting to DISABLED")
        return FlashInferMode.DISABLED


def check_flashinfer_available() -> Tuple[bool, str]:
    """检查 FlashInfer 是否可用"""
    try:
        import flashinfer
        version = getattr(flashinfer, "__version__", "unknown")
        return True, version
    except ImportError as e:
        return False, str(e)


def validate_rope_consistency(
    original_output: torch.Tensor,
    flashinfer_output: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Tuple[bool, float]:
    """
    验证两种实现的输出一致性

    Returns:
        (is_close, max_diff)
    """
    max_diff = (original_output - flashinfer_output).abs().max().item()
    is_close = torch.allclose(original_output, flashinfer_output, rtol=rtol, atol=atol)
    return is_close, max_diff


class ValidatingAttentionWrapper(nn.Module):
    """
    包装器：同时运行原始和 FlashInfer 实现，验证输出一致性
    """

    def __init__(self, original_module, flashinfer_module, layer_name: str = ""):
        super().__init__()
        self.original = original_module
        self.flashinfer = flashinfer_module
        self.layer_name = layer_name
        self.validation_count = 0
        self.max_diff_seen = 0.0
        self.validation_failures = 0

    def forward(self, *args, **kwargs):
        # 运行原始实现
        with torch.no_grad():
            original_output = self.original(*args, **kwargs)

        # 运行 FlashInfer 实现
        flashinfer_output = self.flashinfer(*args, **kwargs)

        # 验证
        self.validation_count += 1
        is_close, max_diff = validate_rope_consistency(
            original_output.detach(),
            flashinfer_output.detach()
        )

        self.max_diff_seen = max(self.max_diff_seen, max_diff)

        if not is_close:
            self.validation_failures += 1
            if self.validation_failures <= 5:  # 只打印前5次
                print(f"[WARNING] {self.layer_name} validation failed: max_diff={max_diff:.6f}")

        # 每100次打印统计
        if self.validation_count % 100 == 0:
            print(f"[INFO] {self.layer_name}: validated {self.validation_count}x, "
                  f"max_diff={self.max_diff_seen:.6f}, failures={self.validation_failures}")

        # 返回原始输出（保证行为不变）
        return original_output


def maybe_use_flashinfer_attention(model: nn.Module) -> nn.Module:
    """
    根据环境变量决定是否替换 attention

    Args:
        model: CausalWanModel 实例

    Returns:
        可能被修改的模型
    """
    mode = get_flashinfer_mode()

    if mode == FlashInferMode.DISABLED:
        print("[INFO] FlashInfer mode: DISABLED (using original implementation)")
        return model

    # 检查 FlashInfer 可用性
    available, info = check_flashinfer_available()
    if not available:
        print(f"[WARNING] FlashInfer not available: {info}")
        print("[INFO] Falling back to original implementation")
        return model

    print(f"[INFO] FlashInfer mode: {mode.value} (version: {info})")

    if mode == FlashInferMode.ENABLED:
        from wan.modules.flashinfer_attention import replace_attention_with_flashinfer
        return replace_attention_with_flashinfer(model)

    elif mode == FlashInferMode.VALIDATE:
        return _wrap_with_validation(model)

    return model


def _wrap_with_validation(model: nn.Module) -> nn.Module:
    """用验证包装器替换 attention 模块"""
    from wan.modules.causal_model import CausalWanSelfAttention
    from wan.modules.flashinfer_attention import FlashInferCausalSelfAttention

    replaced_count = 0

    for name, module in list(model.named_modules()):
        if isinstance(module, CausalWanSelfAttention):
            # 获取父模块
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr_name = parts[-1]

            # 创建 FlashInfer 版本
            flashinfer_module = FlashInferCausalSelfAttention(
                dim=module.dim,
                num_heads=module.num_heads,
                local_attn_size=module.local_attn_size,
                sink_size=module.sink_size,
                qk_norm=module.qk_norm,
                eps=module.eps,
            )

            # 复制权重
            flashinfer_module.q.load_state_dict(module.q.state_dict())
            flashinfer_module.k.load_state_dict(module.k.state_dict())
            flashinfer_module.v.load_state_dict(module.v.state_dict())
            flashinfer_module.o.load_state_dict(module.o.state_dict())
            flashinfer_module.norm_q.load_state_dict(module.norm_q.state_dict())
            flashinfer_module.norm_k.load_state_dict(module.norm_k.state_dict())

            # 移动到相同设备
            device = next(module.parameters()).device
            dtype = next(module.parameters()).dtype
            flashinfer_module = flashinfer_module.to(device=device, dtype=dtype)

            # 创建验证包装器
            wrapper = ValidatingAttentionWrapper(
                module, flashinfer_module, layer_name=name
            )

            setattr(parent, attr_name, wrapper)
            replaced_count += 1

    print(f"[INFO] Wrapped {replaced_count} attention modules for validation")
    return model


def print_validation_summary(model: nn.Module):
    """打印验证摘要"""
    print("\n" + "=" * 60)
    print("FlashInfer Validation Summary")
    print("=" * 60)

    for name, module in model.named_modules():
        if isinstance(module, ValidatingAttentionWrapper):
            status = "✓ PASS" if module.validation_failures == 0 else "✗ FAIL"
            print(f"{name}:")
            print(f"  - Validations: {module.validation_count}")
            print(f"  - Max diff: {module.max_diff_seen:.6f}")
            print(f"  - Failures: {module.validation_failures}")
            print(f"  - Status: {status}")

    print("=" * 60)
