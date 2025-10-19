"""
Export individual components of CausalWanModel to ONNX for structure visualization.
This script exports components separately to avoid FlexAttention compatibility issues.
"""

import os
import argparse
import torch
from safetensors.torch import load_file
from omegaconf import OmegaConf
from utils.wan_wrapper import WanDiffusionWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Export CausalWanModel components to ONNX")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/inference_yaml/inference_universal.yaml",
        help="Path to the config file"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/onnx_components",
        help="Output directory for ONNX files"
    )
    parser.add_argument(
        "--component",
        type=str,
        default="all",
        choices=["all", "embedding", "block", "head", "attention", "ffn", "cross_attn"],
        help="Which component to export"
    )
    parser.add_argument(
        "--block_idx",
        type=int,
        default=0,
        help="Which transformer block to export (0-29)"
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=17,
        help="ONNX opset version"
    )
    return parser.parse_args()


def export_patch_embedding(model, output_path, opset_version):
    """Export patch embedding layer."""
    print("\n[Exporting Patch Embedding]")

    # Create dummy input: [B, 36, F, H, W]
    dummy_input = torch.randn(1, 36, 5, 176, 320, dtype=torch.bfloat16).cuda()

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output path: {output_path}")

    try:
        torch.onnx.export(
            model.patch_embedding,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            do_constant_folding=True,
        )
        print(f"  ✓ Export successful!")
        print(f"  File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
        return True
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        return False


def export_time_embedding(model, output_path, opset_version):
    """Export time embedding layers."""
    print("\n[Exporting Time Embedding]")

    # Create dummy timestep embedding: [B, freq_dim]
    dummy_input = torch.randn(1, 256, dtype=torch.bfloat16).cuda()

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output path: {output_path}")

    try:
        torch.onnx.export(
            model.time_embedding,
            dummy_input,
            output_path,
            input_names=["timestep_embedding"],
            output_names=["time_features"],
            opset_version=opset_version,
            do_constant_folding=True,
        )
        print(f"  ✓ Export successful!")
        print(f"  File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
        return True
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        return False


def export_ffn(block, output_path, opset_version):
    """Export FFN (MLP) from a transformer block."""
    print("\n[Exporting FFN]")

    # Create dummy input: [B, L, dim]
    dummy_input = torch.randn(1, 880, 1536, dtype=torch.bfloat16).cuda()

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output path: {output_path}")

    try:
        torch.onnx.export(
            block.ffn,
            dummy_input,
            output_path,
            input_names=["hidden_states"],
            output_names=["ffn_output"],
            opset_version=opset_version,
            do_constant_folding=True,
        )
        print(f"  ✓ Export successful!")
        print(f"  File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
        return True
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        return False


def export_cross_attention(block, output_path, opset_version):
    """Export cross-attention module."""
    print("\n[Exporting Cross Attention]")

    # Create dummy inputs
    x = torch.randn(1, 880, 1536, dtype=torch.bfloat16).cuda()
    context = torch.randn(1, 257, 1536, dtype=torch.bfloat16).cuda()

    print(f"  Input x shape: {x.shape}")
    print(f"  Context shape: {context.shape}")
    print(f"  Output path: {output_path}")

    class CrossAttnWrapper(torch.nn.Module):
        def __init__(self, cross_attn):
            super().__init__()
            self.cross_attn = cross_attn

        def forward(self, x, context):
            return self.cross_attn(x, context, crossattn_cache=None)

    wrapper = CrossAttnWrapper(block.cross_attn)

    try:
        torch.onnx.export(
            wrapper,
            (x, context),
            output_path,
            input_names=["hidden_states", "context"],
            output_names=["output"],
            opset_version=opset_version,
            do_constant_folding=True,
        )
        print(f"  ✓ Export successful!")
        print(f"  File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
        return True
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        return False


def export_normalization_layers(block, output_path, opset_version):
    """Export LayerNorm layers."""
    print("\n[Exporting Normalization Layers]")

    dummy_input = torch.randn(1, 880, 1536, dtype=torch.bfloat16).cuda()

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output path: {output_path}")

    try:
        torch.onnx.export(
            block.norm1,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["normalized"],
            opset_version=opset_version,
            do_constant_folding=True,
        )
        print(f"  ✓ Export successful!")
        print(f"  File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
        return True
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        return False


def export_qkv_projection(block, output_path, opset_version):
    """Export Q, K, V projection layers from self-attention."""
    print("\n[Exporting QKV Projections]")

    dummy_input = torch.randn(1, 880, 1536, dtype=torch.bfloat16).cuda()

    class QKVWrapper(torch.nn.Module):
        def __init__(self, self_attn):
            super().__init__()
            self.q = self_attn.q
            self.k = self_attn.k
            self.v = self_attn.v
            self.norm_q = self_attn.norm_q
            self.norm_k = self_attn.norm_k

        def forward(self, x):
            q = self.norm_q(self.q(x))
            k = self.norm_k(self.k(x))
            v = self.v(x)
            return q, k, v

    wrapper = QKVWrapper(block.self_attn)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output path: {output_path}")

    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            input_names=["hidden_states"],
            output_names=["query", "key", "value"],
            opset_version=opset_version,
            do_constant_folding=True,
        )
        print(f"  ✓ Export successful!")
        print(f"  File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
        return True
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        return False


def export_head(model, output_path, opset_version):
    """Export the output head."""
    print("\n[Exporting Head]")

    # Create dummy inputs
    x = torch.randn(1, 880, 1536, dtype=torch.bfloat16).cuda()
    e = torch.randn(1, 1, 1, 1536, dtype=torch.bfloat16).cuda()

    print(f"  Input x shape: {x.shape}")
    print(f"  Input e shape: {e.shape}")
    print(f"  Output path: {output_path}")

    try:
        torch.onnx.export(
            model.head,
            (x, e),
            output_path,
            input_names=["hidden_states", "conditioning"],
            output_names=["output"],
            opset_version=opset_version,
            do_constant_folding=True,
        )
        print(f"  ✓ Export successful!")
        print(f"  File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
        return True
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        return False


def main():
    args = parse_args()

    print("="*60)
    print("CausalWanModel Component Export")
    print("="*60)

    # Load model
    print("\nLoading model...")
    config = OmegaConf.load(args.config_path)
    generator = WanDiffusionWrapper(
        **getattr(config, "model_kwargs", {}),
        is_causal=True
    )

    # Load checkpoint
    if args.checkpoint_path:
        print(f"Loading checkpoint: {args.checkpoint_path}")
        state_dict = load_file(args.checkpoint_path)
        if all(key.startswith('model.') for key in list(state_dict.keys())[:10]):
            state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
        generator.model.load_state_dict(state_dict, strict=False)

    generator = generator.to(device="cuda", dtype=torch.bfloat16)
    generator.eval()

    print(f"✓ Model loaded")
    print(f"  Type: {generator.model.model_type}")
    print(f"  Blocks: {len(generator.model.blocks)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Export components
    results = {}

    if args.component in ["all", "embedding"]:
        path = os.path.join(args.output_dir, "patch_embedding.onnx")
        results["patch_embedding"] = export_patch_embedding(generator.model, path, args.opset_version)

        path = os.path.join(args.output_dir, "time_embedding.onnx")
        results["time_embedding"] = export_time_embedding(generator.model, path, args.opset_version)

    if args.component in ["all", "block", "ffn", "attention", "cross_attn"]:
        block = generator.model.blocks[args.block_idx]
        print(f"\nUsing transformer block {args.block_idx}")

        if args.component in ["all", "block", "ffn"]:
            path = os.path.join(args.output_dir, f"ffn_block{args.block_idx}.onnx")
            results["ffn"] = export_ffn(block, path, args.opset_version)

        if args.component in ["all", "block", "cross_attn"]:
            path = os.path.join(args.output_dir, f"cross_attention_block{args.block_idx}.onnx")
            results["cross_attention"] = export_cross_attention(block, path, args.opset_version)

        if args.component in ["all", "block", "attention"]:
            path = os.path.join(args.output_dir, f"qkv_projection_block{args.block_idx}.onnx")
            results["qkv_projection"] = export_qkv_projection(block, path, args.opset_version)

            path = os.path.join(args.output_dir, f"norm_block{args.block_idx}.onnx")
            results["normalization"] = export_normalization_layers(block, path, args.opset_version)

    if args.component in ["all", "head"]:
        path = os.path.join(args.output_dir, "head.onnx")
        results["head"] = export_head(generator.model, path, args.opset_version)

    # Summary
    print("\n" + "="*60)
    print("Export Summary")
    print("="*60)
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Successful: {successful}/{total}")
    print(f"Output directory: {args.output_dir}")
    print("\nExported components:")
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    print("="*60)

    print("\n💡 Tip: View these ONNX files at https://netron.app/")
    print("   Each file shows a specific component of the model architecture.")


if __name__ == "__main__":
    main()
