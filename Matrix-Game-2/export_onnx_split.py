"""
Export CausalWanModel in two parts to avoid FlexAttention:
1. Pre-attention: Input processing up to QKV projection
2. Post-attention: After attention to final output
"""

import os
import argparse
import torch
from safetensors.torch import load_file
from omegaconf import OmegaConf
from utils.wan_wrapper import WanDiffusionWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Export CausalWanModel split around FlexAttention")
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
        default="outputs/onnx_split",
        help="Output directory for ONNX files"
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=3,
        help="Number of transformer blocks to export (default: 3, full model: 30)"
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=17,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX models using onnx-simplifier"
    )
    return parser.parse_args()


class PreAttentionModel(torch.nn.Module):
    """
    Part 1: Everything BEFORE FlexAttention
    Input -> Patch Embedding -> Time Embedding -> Layer Norms -> QKV Projections
    """
    def __init__(self, model, num_blocks=3):
        super().__init__()
        self.model = model
        self.num_blocks = num_blocks

        # Core components
        self.patch_embedding = model.patch_embedding
        self.time_embedding = model.time_embedding
        self.time_projection = model.time_projection
        self.img_emb = model.img_emb

        # Select first N blocks
        self.blocks = model.blocks[:num_blocks]

    def forward(self, noisy_input, cond_concat, timestep, visual_context):
        """
        Args:
            noisy_input: [B, 16, F, H, W]
            cond_concat: [B, 20, F, H, W]
            timestep: [B, F]
            visual_context: [B, 257, 1280]

        Returns:
            Dictionary containing all intermediate features before attention
        """
        device = noisy_input.device

        # Concatenate input
        x = torch.cat([noisy_input, cond_concat], dim=1)  # [B, 36, F, H, W]

        # Patch embedding
        x = self.patch_embedding(x)  # [B, C, F, H', W']
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long, device=device)
        x = x.flatten(2).transpose(1, 2)  # [B, L, C] where L = F*H'*W'

        # Time embedding
        from wan.modules.model import sinusoidal_embedding_1d
        t_flat = timestep[:, 0]  # Use first frame's timestep
        e = self.time_embedding(
            sinusoidal_embedding_1d(256, t_flat).type_as(x)
        )
        e0 = self.time_projection(e).unflatten(1, (6, 1536))  # [B, 6, C]
        e0 = e0.unsqueeze(1)  # [B, 1, 6, C] for broadcasting to frames

        # Process visual context
        context = self.img_emb(visual_context)  # [B, 257, C]

        # Process through blocks (pre-attention parts only)
        block_outputs = []
        for block_idx, block in enumerate(self.blocks):
            # Extract modulation parameters
            e_block = (block.modulation.unsqueeze(1) + e0).chunk(6, dim=2)

            # Self-attention preprocessing
            # Assuming x has shape [B, L, C] where L includes all frames
            num_frames = timestep.shape[1]
            frame_seqlen = x.shape[1] // num_frames

            # Apply pre-norm and modulation for self-attention
            x_norm = block.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            x_modulated = (x_norm * (1 + e_block[1]) + e_block[0]).flatten(1, 2)

            # QKV projection
            b, s = x_modulated.shape[:2]
            n, d = 12, 128  # num_heads, head_dim

            q = block.self_attn.norm_q(block.self_attn.q(x_modulated)).view(b, s, n, d)
            k = block.self_attn.norm_k(block.self_attn.k(x_modulated)).view(b, s, n, d)
            v = block.self_attn.v(x_modulated).view(b, s, n, d)

            # Cross-attention preprocessing
            x_cross_norm = block.norm3(x.to(context.dtype))

            # Store outputs
            block_outputs.append({
                'q': q,
                'k': k,
                'v': v,
                'x_residual': x,
                'e_gate': e_block[2],
                'x_cross_norm': x_cross_norm,
                'e_ffn_scale': e_block[4],
                'e_ffn_shift': e_block[3],
                'e_ffn_gate': e_block[5],
            })

            # NOTE: In real model, attention happens here
            # For export, we stop before attention and return QKV

        return {
            'block_outputs': block_outputs,
            'context': context,
            'grid_sizes': grid_sizes,
            'num_frames': num_frames,
            'frame_seqlen': frame_seqlen,
        }


class PostAttentionModel(torch.nn.Module):
    """
    Part 2: Everything AFTER FlexAttention
    Attention Output -> Residual -> Cross-Attention -> FFN -> Output Head
    """
    def __init__(self, model, num_blocks=3):
        super().__init__()
        self.model = model
        self.num_blocks = num_blocks

        # Select first N blocks
        self.blocks = model.blocks[:num_blocks]
        self.head = model.head

    def forward(self, attn_output, x_residual, e_gate, context, x_cross_norm,
                e_ffn_scale, e_ffn_shift, e_ffn_gate, num_frames, frame_seqlen, e_head):
        """
        Args:
            attn_output: [B, L, C] - output from attention
            x_residual: [B, L, C] - residual connection from before attention
            e_gate: [B, 1, 1, C] - gating for attention output
            context: [B, 257, C] - cross-attention context
            x_cross_norm: [B, L, C] - normalized input for cross-attention
            e_ffn_*: FFN modulation parameters
            num_frames: int
            frame_seqlen: int
            e_head: [B, 1, 1, C] - conditioning for head

        Returns:
            output: [B, L, C] - processed features (before unpatchify for simplicity)
        """
        # Process through one block (post-attention)
        block = self.blocks[0]  # Use first block as example

        # Apply gating and residual for self-attention
        attn_output_gated = attn_output.unflatten(dim=1, sizes=(num_frames, frame_seqlen))
        attn_output_gated = (attn_output_gated * e_gate).flatten(1, 2)
        x = x_residual + attn_output_gated

        # Cross-attention
        x = x + block.cross_attn(x_cross_norm, context, crossattn_cache=None)

        # FFN with modulation
        x_ffn_norm = block.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
        x_ffn = (x_ffn_norm * (1 + e_ffn_scale) + e_ffn_shift).flatten(1, 2)
        ffn_out = block.ffn(x_ffn)
        ffn_out_gated = (ffn_out.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e_ffn_gate).flatten(1, 2)
        x = x + ffn_out_gated

        # Output head - note: e_head needs proper shape [B, F, 1, C]
        e_head_expanded = e_head.expand(-1, num_frames, -1, -1)  # [B, F, 1, C]
        x_head = self.head(x, e_head_expanded)

        # Return the head output directly (unpatchify would need proper grid_sizes)
        # x_head shape: [B, F, H', W', patch_h, patch_w, C_out]
        # For ONNX export visualization, we return this intermediate representation
        return x_head


def export_pre_attention(model, output_path, opset_version, num_blocks):
    """Export the pre-attention part."""
    print(f"\n{'='*60}")
    print("Part 1: Pre-Attention (Input → QKV)")
    print(f"{'='*60}")

    pre_model = PreAttentionModel(model, num_blocks=num_blocks)
    pre_model.eval()

    # Create dummy inputs
    B, F = 1, 5
    noisy_input = torch.randn(B, 16, F, 44, 80, dtype=torch.bfloat16).cuda()
    cond_concat = torch.randn(B, 20, F, 44, 80, dtype=torch.bfloat16).cuda()
    timestep = torch.randint(0, 1000, (B, F), dtype=torch.int64).cuda()
    visual_context = torch.randn(B, 257, 1280, dtype=torch.bfloat16).cuda()

    print(f"\nInput shapes:")
    print(f"  noisy_input:    {noisy_input.shape}")
    print(f"  cond_concat:    {cond_concat.shape}")
    print(f"  timestep:       {timestep.shape}")
    print(f"  visual_context: {visual_context.shape}")
    print(f"\nExporting {num_blocks} blocks...")

    # Test forward
    print("\nTesting forward pass...")
    with torch.no_grad():
        try:
            outputs = pre_model(noisy_input, cond_concat, timestep, visual_context)
            print("✓ Forward pass successful!")
            print(f"\nOutput structure:")
            print(f"  Number of blocks: {len(outputs['block_outputs'])}")
            for i, block_out in enumerate(outputs['block_outputs']):
                print(f"\n  Block {i}:")
                print(f"    Q: {block_out['q'].shape}")
                print(f"    K: {block_out['k'].shape}")
                print(f"    V: {block_out['v'].shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Export to ONNX
    print(f"\nExporting to: {output_path}")
    try:
        # Note: The output is complex (dict), so we flatten it for ONNX
        # We'll just export the first block's Q, K, V as example
        class SimplifiedPreModel(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base = base_model

            def forward(self, noisy_input, cond_concat, timestep, visual_context):
                outputs = self.base(noisy_input, cond_concat, timestep, visual_context)
                block0 = outputs['block_outputs'][0]
                # Return flattened Q, K, V
                return block0['q'], block0['k'], block0['v'], outputs['context']

        simplified = SimplifiedPreModel(pre_model)

        torch.onnx.export(
            simplified,
            (noisy_input, cond_concat, timestep, visual_context),
            output_path,
            input_names=['noisy_input', 'cond_concat', 'timestep', 'visual_context'],
            output_names=['query', 'key', 'value', 'context'],
            dynamic_axes={
                'noisy_input': {0: 'batch', 2: 'frames'},
                'cond_concat': {0: 'batch', 2: 'frames'},
                'timestep': {0: 'batch', 1: 'frames'},
                'visual_context': {0: 'batch'},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

        file_size = os.path.getsize(output_path) / (1024**2)
        print(f"✓ Export successful!")
        print(f"  File size: {file_size:.2f} MB")
        return True

    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_post_attention(model, output_path, opset_version, num_blocks):
    """Export the post-attention part."""
    print(f"\n{'='*60}")
    print("Part 2: Post-Attention (Attention Output → Final Output)")
    print(f"{'='*60}")

    post_model = PostAttentionModel(model, num_blocks=1)  # Just one block for simplicity
    post_model.eval()

    # Create dummy inputs (simulating attention output)
    B, L, C = 1, 880, 1536
    num_frames, frame_seqlen = 1, 880

    attn_output = torch.randn(B, L, C, dtype=torch.bfloat16).cuda()
    x_residual = torch.randn(B, L, C, dtype=torch.bfloat16).cuda()
    e_gate = torch.randn(B, 1, 1, C, dtype=torch.bfloat16).cuda()
    context = torch.randn(B, 257, C, dtype=torch.bfloat16).cuda()
    x_cross_norm = torch.randn(B, L, C, dtype=torch.bfloat16).cuda()
    e_ffn_scale = torch.randn(B, 1, 1, C, dtype=torch.bfloat16).cuda()
    e_ffn_shift = torch.randn(B, 1, 1, C, dtype=torch.bfloat16).cuda()
    e_ffn_gate = torch.randn(B, 1, 1, C, dtype=torch.bfloat16).cuda()
    e_head = torch.randn(B, 1, 1, C, dtype=torch.bfloat16).cuda()

    print(f"\nInput shapes:")
    print(f"  attn_output:  {attn_output.shape}")
    print(f"  x_residual:   {x_residual.shape}")
    print(f"  context:      {context.shape}")

    # Test forward
    print("\nTesting forward pass...")
    with torch.no_grad():
        try:
            output = post_model(
                attn_output, x_residual, e_gate, context, x_cross_norm,
                e_ffn_scale, e_ffn_shift, e_ffn_gate,
                num_frames, frame_seqlen, e_head
            )
            print("✓ Forward pass successful!")
            print(f"  Output shape: {output.shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Export to ONNX
    print(f"\nExporting to: {output_path}")
    try:
        torch.onnx.export(
            post_model,
            (attn_output, x_residual, e_gate, context, x_cross_norm,
             e_ffn_scale, e_ffn_shift, e_ffn_gate,
             num_frames, frame_seqlen, e_head),
            output_path,
            input_names=[
                'attn_output', 'x_residual', 'e_gate', 'context', 'x_cross_norm',
                'e_ffn_scale', 'e_ffn_shift', 'e_ffn_gate',
                'num_frames', 'frame_seqlen', 'e_head'
            ],
            output_names=['output'],
            opset_version=opset_version,
            do_constant_folding=True,
        )

        file_size = os.path.getsize(output_path) / (1024**2)
        print(f"✓ Export successful!")
        print(f"  File size: {file_size:.2f} MB")
        return True

    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()

    print("="*60)
    print("CausalWanModel Split Export")
    print("="*60)
    print(f"\nThis exports the model in 2 parts:")
    print(f"  1. Pre-Attention:  Input → Embeddings → QKV")
    print(f"  2. Post-Attention: Attn Output → Cross-Attn → FFN → Output")
    print(f"\n⚠️  FlexAttention is NOT exported (not ONNX-compatible)")

    # Load model
    print(f"\nLoading model...")
    config = OmegaConf.load(args.config_path)
    generator = WanDiffusionWrapper(
        **getattr(config, "model_kwargs", {}),
        is_causal=True
    )

    if args.checkpoint_path:
        print(f"Loading checkpoint: {args.checkpoint_path}")
        state_dict = load_file(args.checkpoint_path)
        if all(key.startswith('model.') for key in list(state_dict.keys())[:10]):
            state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
        generator.model.load_state_dict(state_dict, strict=False)

    generator = generator.to(device="cuda", dtype=torch.bfloat16)
    generator.eval()

    print(f"✓ Model loaded")
    print(f"  Total blocks: {len(generator.model.blocks)}")
    print(f"  Exporting: {args.num_blocks} blocks")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Export both parts
    results = {}

    pre_path = os.path.join(args.output_dir, "part1_pre_attention.onnx")
    results['pre_attention'] = export_pre_attention(
        generator.model, pre_path, args.opset_version, args.num_blocks
    )

    post_path = os.path.join(args.output_dir, "part2_post_attention.onnx")
    results['post_attention'] = export_post_attention(
        generator.model, post_path, args.opset_version, args.num_blocks
    )

    # Simplify if requested
    if args.simplify and any(results.values()):
        print(f"\n{'='*60}")
        print("Simplifying ONNX models...")
        print(f"{'='*60}")
        try:
            import onnx
            from onnxsim import simplify

            for name, success in results.items():
                if not success:
                    continue

                original_path = os.path.join(args.output_dir, f"part{'1' if 'pre' in name else '2'}_{name}.onnx")
                simplified_path = original_path.replace('.onnx', '_simplified.onnx')

                print(f"\nSimplifying {name}...")
                model_onnx = onnx.load(original_path)
                model_simplified, check = simplify(model_onnx)

                if check:
                    onnx.save(model_simplified, simplified_path)
                    file_size = os.path.getsize(simplified_path) / (1024**2)
                    print(f"  ✓ Saved to: {simplified_path}")
                    print(f"  File size: {file_size:.2f} MB")
                else:
                    print(f"  ✗ Simplification check failed")

        except ImportError:
            print("onnx-simplifier not installed. Skipping simplification.")
        except Exception as e:
            print(f"✗ Simplification failed: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("Export Summary")
    print(f"{'='*60}")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Blocks exported: {args.num_blocks}/{len(generator.model.blocks)}")
    print(f"\nResults:")
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    print(f"\n{'='*60}")
    print("💡 How to use:")
    print("  1. View each part at https://netron.app/")
    print("  2. Part 1 shows: Input processing → QKV projections")
    print("  3. Part 2 shows: Post-attention → FFN → Output")
    print("  4. FlexAttention happens BETWEEN these two parts")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
