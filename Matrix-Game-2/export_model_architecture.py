"""
Export CausalWanModel architecture visualization.
Creates simple, exportable modules for each major component.
"""

import os
import argparse
import torch
from safetensors.torch import load_file
from omegaconf import OmegaConf
from utils.wan_wrapper import WanDiffusionWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Export model architecture components")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/inference_yaml/inference_universal.yaml",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/architecture",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=17,
    )
    return parser.parse_args()


def export_component(component, name, dummy_inputs, input_names, output_names, output_dir, opset_version):
    """Helper function to export a single component."""
    output_path = os.path.join(output_dir, f"{name}.onnx")

    print(f"\n[{name}]")
    print(f"  Output: {output_path}")

    # Test forward
    with torch.no_grad():
        try:
            if isinstance(dummy_inputs, tuple):
                outputs = component(*dummy_inputs)
            else:
                outputs = component(dummy_inputs)

            if isinstance(outputs, tuple):
                print(f"  Outputs: {len(outputs)} tensors")
                for i, out in enumerate(outputs):
                    if isinstance(out, torch.Tensor):
                        print(f"    [{i}] {out.shape}")
            else:
                print(f"  Output shape: {outputs.shape}")
        except Exception as e:
            print(f"  ✗ Forward failed: {e}")
            return False

    # Export
    try:
        torch.onnx.export(
            component,
            dummy_inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            do_constant_folding=True,
        )

        file_size = os.path.getsize(output_path) / (1024**2)
        print(f"  ✓ Exported ({file_size:.2f} MB)")
        return True

    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        return False


def main():
    args = parse_args()

    print("="*70)
    print("CausalWanModel Architecture Export")
    print("="*70)
    print("\nThis exports individual components for architecture visualization.")
    print("Components are simplified to avoid ONNX compatibility issues.\n")

    # Load model
    print("Loading model...")
    config = OmegaConf.load(args.config_path)
    generator = WanDiffusionWrapper(
        **getattr(config, "model_kwargs", {}),
        is_causal=True
    )

    if args.checkpoint_path:
        state_dict = load_file(args.checkpoint_path)
        if all(key.startswith('model.') for key in list(state_dict.keys())[:10]):
            state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
        generator.model.load_state_dict(state_dict, strict=False)

    model = generator.model.to(device="cuda", dtype=torch.bfloat16).eval()

    print(f"✓ Model loaded ({len(model.blocks)} blocks)")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    # ========================================
    # 1. Patch Embedding
    # ========================================
    print("\n" + "="*70)
    print("PART 1: Input Embeddings")
    print("="*70)

    x = torch.randn(1, 36, 5, 176, 320, dtype=torch.bfloat16).cuda()
    results['patch_embedding'] = export_component(
        model.patch_embedding,
        "1_patch_embedding",
        x,
        ["input_concat"],
        ["patch_features"],
        args.output_dir,
        args.opset_version
    )

    # ========================================
    # 2. Time Embedding
    # ========================================
    from wan.modules.model import sinusoidal_embedding_1d

    class TimeEmbeddingWrapper(torch.nn.Module):
        def __init__(self, time_emb, time_proj):
            super().__init__()
            self.time_embedding = time_emb
            self.time_projection = time_proj

        def forward(self, t):
            # t: [B] timestep values
            t_emb = sinusoidal_embedding_1d(256, t)
            e = self.time_embedding(t_emb)
            e0 = self.time_projection(e)
            return e0

    time_wrapper = TimeEmbeddingWrapper(model.time_embedding, model.time_projection).cuda()
    t = torch.tensor([500], dtype=torch.float32).cuda()

    results['time_embedding'] = export_component(
        time_wrapper,
        "2_time_embedding",
        t,
        ["timestep"],
        ["conditioning"],
        args.output_dir,
        args.opset_version
    )

    # ========================================
    # 3. Image/Visual Context Embedding
    # ========================================
    x_visual = torch.randn(1, 257, 1280, dtype=torch.bfloat16).cuda()
    results['img_embedding'] = export_component(
        model.img_emb,
        "3_visual_context_embedding",
        x_visual,
        ["clip_features"],
        ["context"],
        args.output_dir,
        args.opset_version
    )

    # ========================================
    # 4. Transformer Block Components
    # ========================================
    print("\n" + "="*70)
    print("PART 2: Transformer Block Components (Block 0)")
    print("="*70)

    block = model.blocks[0]

    # 4a. LayerNorm
    x = torch.randn(1, 4400, 1536, dtype=torch.bfloat16).cuda()
    results['layernorm'] = export_component(
        block.norm1,
        "4a_layernorm",
        x,
        ["input"],
        ["normalized"],
        args.output_dir,
        args.opset_version
    )

    # 4b. Q/K/V Projections
    class QKVProjection(torch.nn.Module):
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

    qkv_wrapper = QKVProjection(block.self_attn).cuda()
    x = torch.randn(1, 4400, 1536, dtype=torch.bfloat16).cuda()

    results['qkv_projection'] = export_component(
        qkv_wrapper,
        "4b_qkv_projection",
        x,
        ["hidden_states"],
        ["query", "key", "value"],
        args.output_dir,
        args.opset_version
    )

    # 4c. Self-Attention Output Projection
    x = torch.randn(1, 4400, 1536, dtype=torch.bfloat16).cuda()
    results['attn_output'] = export_component(
        block.self_attn.o,
        "4c_attention_output_projection",
        x,
        ["attention_output"],
        ["projected_output"],
        args.output_dir,
        args.opset_version
    )

    # 4d. Cross-Attention (simplified)
    class CrossAttentionWrapper(torch.nn.Module):
        def __init__(self, cross_attn):
            super().__init__()
            self.cross_attn = cross_attn

        def forward(self, x, context):
            return self.cross_attn(x, context, crossattn_cache=None)

    cross_attn_wrapper = CrossAttentionWrapper(block.cross_attn).cuda()
    x = torch.randn(1, 4400, 1536, dtype=torch.bfloat16).cuda()
    context = torch.randn(1, 257, 1536, dtype=torch.bfloat16).cuda()

    results['cross_attention'] = export_component(
        cross_attn_wrapper,
        "4d_cross_attention",
        (x, context),
        ["hidden_states", "context"],
        ["output"],
        args.output_dir,
        args.opset_version
    )

    # 4e. FFN/MLP
    x = torch.randn(1, 4400, 1536, dtype=torch.bfloat16).cuda()
    results['ffn'] = export_component(
        block.ffn,
        "4e_ffn_mlp",
        x,
        ["input"],
        ["output"],
        args.output_dir,
        args.opset_version
    )

    # ========================================
    # 5. Output Head
    # ========================================
    print("\n" + "="*70)
    print("PART 3: Output Head")
    print("="*70)

    class HeadWrapper(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head

        def forward(self, x, e):
            return self.head(x, e)

    head_wrapper = HeadWrapper(model.head).cuda()
    x = torch.randn(1, 880, 1536, dtype=torch.bfloat16).cuda()
    e = torch.randn(1, 1, 1, 1536, dtype=torch.bfloat16).cuda()

    results['head'] = export_component(
        head_wrapper,
        "5_output_head",
        (x, e),
        ["features", "conditioning"],
        ["output"],
        args.output_dir,
        args.opset_version
    )

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("EXPORT SUMMARY")
    print("="*70)

    successful = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nSuccessfully exported: {successful}/{total} components")
    print(f"Output directory: {args.output_dir}\n")

    print("Components:")
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    print("\n" + "="*70)
    print("MODEL ARCHITECTURE OVERVIEW")
    print("="*70)
    print("""
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT PROCESSING                         │
├─────────────────────────────────────────────────────────────────┤
│ 1. Patch Embedding      : Conv3d(36→1536, kernel=(1,2,2))      │
│ 2. Time Embedding       : Sinusoidal + MLP                      │
│ 3. Visual Context Embed : MLP(1280→1536)                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   TRANSFORMER BLOCK (×30)                        │
├─────────────────────────────────────────────────────────────────┤
│ 4a. LayerNorm           : Normalize hidden states               │
│ 4b. Q/K/V Projection    : Linear(1536→1536) + RMSNorm          │
│                                                                   │
│     ┌─────────────────────────────────────────┐                │
│     │  FlexAttention (NOT EXPORTED)           │                │
│     │  - Block-wise causal mask                │                │
│     │  - Local attention window                │                │
│     └─────────────────────────────────────────┘                │
│                                                                   │
│ 4c. Attention Output    : Linear(1536→1536)                     │
│ 4d. Cross-Attention     : Attend to visual context              │
│ 4e. FFN/MLP            : Linear(1536→8960→1536) + GELU         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT HEAD                                │
├─────────────────────────────────────────────────────────────────┤
│ 5. Output Head          : LayerNorm + Linear(1536→128)          │
│    (Unpatchify)         : Reshape to [B, 16, F, H, W]           │
└─────────────────────────────────────────────────────────────────┘
    """)

    print("="*70)
    print("💡 Next Steps:")
    print("  1. View each .onnx file at https://netron.app/")
    print("  2. Components are numbered in execution order")
    print("  3. FlexAttention is not exported (ONNX incompatible)")
    print("  4. All other major components are included")
    print("="*70)


if __name__ == "__main__":
    main()
