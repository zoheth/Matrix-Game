"""
Export CausalWanModel to ONNX format.
Input shapes are aligned with inference.py for consistency.
"""

import os
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from utils.wan_wrapper import WanDiffusionWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Export CausalWanModel to ONNX")
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
        "--output_path",
        type=str,
        default="outputs/causal_wan_model.onnx",
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="universal",
        choices=["universal", "gta_drive", "templerun"],
        help="Mode for the model export"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for ONNX export"
    )
    parser.add_argument(
        "--num_output_frames",
        type=int,
        default=150,
        help="Number of output latent frames (must match inference.py)"
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
        help="Simplify ONNX model using onnx-simplifier"
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=None,
        help="Number of transformer blocks to export (default: all 30 blocks). Use smaller number (e.g., 2-3) to reduce memory and view model structure."
    )
    parser.add_argument(
        "--num_frames_export",
        type=int,
        default=None,
        help="Number of frames to use for ONNX export (default: same as num_output_frames). Use smaller number (e.g., 5-10) to reduce memory."
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="Skip forward pass test before export (saves memory, but doesn't verify the model works)"
    )
    return parser.parse_args()


def create_dummy_inputs(args, device, dtype):
    """
    Create dummy inputs matching the shapes used in inference.py.

    Based on inference.py:
    - x: noisy latent [B, 16, F, 44, 80]  (16 channels after concatenation with cond_concat)
    - x is concatenated from noise [B, 16, F, 44, 80] + cond_concat [B, 20, F, 44, 80]
    - Actually, x input to model is [B, 36, F, 44, 80] (in_dim=36 from config)
    - t: timestep [B, F] or [B] depending on uniform_timestep
    - visual_context: [B, 257, 1280] (from CLIP encoding)
    - cond_concat: [B, 20, F, 44, 80] (mask_cond[:, :4] + img_cond)
    - mouse_cond: [B, num_frames_pixel, 2] where num_frames_pixel = 1 + 4 * (F - 1)
    - keyboard_cond: [B, num_frames_pixel, 4] for universal mode
    """
    batch_size = args.batch_size
    # Use smaller frame count for export if specified
    num_latent_frames = args.num_frames_export if args.num_frames_export else args.num_output_frames
    num_pixel_frames = 1 + 4 * (num_latent_frames - 1)  # 597 pixel frames

    # Noisy input: [B, 16, F, H, W]
    noisy_input = torch.randn(
        batch_size, 16, num_latent_frames, 44, 80,
        device=device, dtype=dtype
    )

    # Timestep: [B, F] for causal mode
    timestep = torch.randint(
        0, 1000, (batch_size, num_latent_frames),
        device=device, dtype=torch.int64
    )

    # Visual context from CLIP: [B, 257, 1280]
    visual_context = torch.randn(
        batch_size, 257, 1280,
        device=device, dtype=dtype
    )

    # Conditioning concat: [B, 20, F, H, W]
    # This contains mask (4 channels) + encoded image (16 channels)
    cond_concat = torch.randn(
        batch_size, 20, num_latent_frames, 44, 80,
        device=device, dtype=dtype
    )

    # Create conditional dict based on mode
    conditional_dict = {
        "visual_context": visual_context,
        "cond_concat": cond_concat,
    }

    if args.mode == "universal":
        # Mouse condition: [B, num_frames_pixel, 2]
        mouse_cond = torch.randn(
            batch_size, num_pixel_frames, 2,
            device=device, dtype=dtype
        )
        # Keyboard condition: [B, num_frames_pixel, 4]
        keyboard_cond = torch.randn(
            batch_size, num_pixel_frames, 4,
            device=device, dtype=dtype
        )
        conditional_dict["mouse_cond"] = mouse_cond
        conditional_dict["keyboard_cond"] = keyboard_cond

    elif args.mode == "gta_drive":
        # Mouse condition: [B, num_frames_pixel, 2]
        mouse_cond = torch.randn(
            batch_size, num_pixel_frames, 2,
            device=device, dtype=dtype
        )
        # Keyboard condition: [B, num_frames_pixel, 2]
        keyboard_cond = torch.randn(
            batch_size, num_pixel_frames, 2,
            device=device, dtype=dtype
        )
        conditional_dict["mouse_cond"] = mouse_cond
        conditional_dict["keyboard_cond"] = keyboard_cond

    elif args.mode == "templerun":
        # Keyboard condition only: [B, num_frames_pixel, 7]
        keyboard_cond = torch.randn(
            batch_size, num_pixel_frames, 7,
            device=device, dtype=dtype
        )
        conditional_dict["keyboard_cond"] = keyboard_cond

    return noisy_input, timestep, conditional_dict


def export_to_onnx(args):
    """Export CausalWanModel to ONNX format."""

    # Setup device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Load config
    print(f"Loading config from: {args.config_path}")
    config = OmegaConf.load(args.config_path)

    # Initialize model
    print("Initializing WanDiffusionWrapper...")
    generator = WanDiffusionWrapper(
        **getattr(config, "model_kwargs", {}),
        is_causal=True
    )

    # Load checkpoint
    if args.checkpoint_path:
        print(f"Loading checkpoint from: {args.checkpoint_path}")
        from safetensors.torch import load_file
        state_dict = load_file(args.checkpoint_path)

        # Handle 'model.' prefix in checkpoint keys (from WanDiffusionWrapper)
        if all(key.startswith('model.') for key in list(state_dict.keys())[:10]):
            print("Detected 'model.' prefix in checkpoint, removing it...")
            state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}

        # Load with strict=False to handle potential mismatches
        missing_keys, unexpected_keys = generator.model.load_state_dict(state_dict, strict=False)

        # Calculate loaded keys
        total_model_keys = len(generator.model.state_dict())
        loaded_keys = total_model_keys - len(missing_keys)

        print(f"✓ Loaded {loaded_keys}/{total_model_keys} parameters ({100*loaded_keys/total_model_keys:.1f}%)")

        if missing_keys:
            print(f"⚠️  Missing {len(missing_keys)} keys (first 5): {missing_keys[:5]}")
        if unexpected_keys:
            print(f"⚠️  Unexpected {len(unexpected_keys)} keys (first 5): {unexpected_keys[:5]}")

        # Warn if most weights are missing
        if loaded_keys < total_model_keys * 0.5:
            print("🚨 WARNING: Less than 50% of weights loaded! Check checkpoint compatibility.")
            raise RuntimeError("Checkpoint incompatible with model architecture")
        elif loaded_keys == total_model_keys:
            print("✅ All weights loaded successfully!")
        else:
            print("✓ Partial weight loading OK (some layers may be randomly initialized)")

    # Reduce number of blocks if specified (for viewing structure with less memory)
    if args.num_blocks is not None:
        original_num_blocks = len(generator.model.blocks)
        if args.num_blocks < original_num_blocks:
            print(f"\n⚠️  Reducing model from {original_num_blocks} blocks to {args.num_blocks} blocks")
            print(f"   (For structure visualization only - not a complete model!)")
            generator.model.blocks = generator.model.blocks[:args.num_blocks]
            generator.model.num_layers = args.num_blocks
        else:
            print(f"\n✓ Using all {original_num_blocks} blocks (num_blocks >= actual blocks)")

    # Move to device and set to eval mode
    generator = generator.to(device=device, dtype=dtype)
    generator.eval()

    print(f"\nModel loaded successfully.")
    print(f"  Model type: {generator.model.model_type}")
    print(f"  Number of blocks: {len(generator.model.blocks)}")
    print(f"  Local attention size: {generator.model.local_attn_size}")

    # Create dummy inputs
    print("Creating dummy inputs...")
    noisy_input, timestep, conditional_dict = create_dummy_inputs(args, device, dtype)

    print(f"Input shapes:")
    print(f"  - noisy_input: {noisy_input.shape}")
    print(f"  - timestep: {timestep.shape}")
    print(f"  - visual_context: {conditional_dict['visual_context'].shape}")
    print(f"  - cond_concat: {conditional_dict['cond_concat'].shape}")
    if "mouse_cond" in conditional_dict:
        print(f"  - mouse_cond: {conditional_dict['mouse_cond'].shape}")
    if "keyboard_cond" in conditional_dict:
        print(f"  - keyboard_cond: {conditional_dict['keyboard_cond'].shape}")

    # Test forward pass (optional)
    if not args.skip_test:
        print("\nTesting forward pass...")
        with torch.no_grad():
            try:
                flow_pred, pred_x0 = generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=None,
                    kv_cache_mouse=None,
                    kv_cache_keyboard=None,
                    crossattn_cache=None,
                    current_start=None,
                    cache_start=None
                )
                print(f"Forward pass successful!")
                print(f"  - flow_pred shape: {flow_pred.shape}")
                print(f"  - pred_x0 shape: {pred_x0.shape}")
            except Exception as e:
                print(f"⚠️  Forward pass failed: {e}")
                print(f"   Continuing with export anyway (use --skip_test to skip this check)")
                import traceback
                traceback.print_exc()
    else:
        print("\n⚠️  Skipping forward pass test (--skip_test enabled)")

    # Prepare input/output names and dynamic axes
    input_names = ["noisy_input", "timestep", "visual_context", "cond_concat"]
    output_names = ["flow_pred", "pred_x0"]

    # Create input tuple for ONNX export
    # We need to flatten the conditional_dict for ONNX export
    if args.mode == "universal" or args.mode == "gta_drive":
        input_names.extend(["mouse_cond", "keyboard_cond"])
        inputs_tuple = (
            noisy_input,
            timestep,
            conditional_dict["visual_context"],
            conditional_dict["cond_concat"],
            conditional_dict["mouse_cond"],
            conditional_dict["keyboard_cond"],
        )
    else:  # templerun
        input_names.append("keyboard_cond")
        inputs_tuple = (
            noisy_input,
            timestep,
            conditional_dict["visual_context"],
            conditional_dict["cond_concat"],
            conditional_dict["keyboard_cond"],
        )

    # Define dynamic axes
    dynamic_axes = {
        "noisy_input": {0: "batch_size", 2: "num_frames"},
        "timestep": {0: "batch_size", 1: "num_frames"},
        "visual_context": {0: "batch_size"},
        "cond_concat": {0: "batch_size", 2: "num_frames"},
        "flow_pred": {0: "batch_size", 2: "num_frames"},
        "pred_x0": {0: "batch_size", 2: "num_frames"},
    }

    if "mouse_cond" in conditional_dict:
        dynamic_axes["mouse_cond"] = {0: "batch_size", 1: "num_pixel_frames"}
    if "keyboard_cond" in conditional_dict:
        dynamic_axes["keyboard_cond"] = {0: "batch_size", 1: "num_pixel_frames"}

    # Create a wrapper module for ONNX export
    class WanDiffusionONNXWrapper(torch.nn.Module):
        def __init__(self, generator, mode):
            super().__init__()
            self.generator = generator
            self.mode = mode

        def forward(self, noisy_input, timestep, visual_context, cond_concat,
                   mouse_cond=None, keyboard_cond=None):
            conditional_dict = {
                "visual_context": visual_context,
                "cond_concat": cond_concat,
            }
            if mouse_cond is not None:
                conditional_dict["mouse_cond"] = mouse_cond
            if keyboard_cond is not None:
                conditional_dict["keyboard_cond"] = keyboard_cond

            flow_pred, pred_x0 = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=None,
                kv_cache_mouse=None,
                kv_cache_keyboard=None,
                crossattn_cache=None,
                current_start=None,
                cache_start=None
            )
            return flow_pred, pred_x0

    wrapped_model = WanDiffusionONNXWrapper(generator, args.mode)
    wrapped_model.eval()

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # Export to ONNX
    print(f"\nExporting to ONNX: {args.output_path}")
    print(f"Opset version: {args.opset_version}")

    try:
        torch.onnx.export(
            wrapped_model,
            inputs_tuple,
            args.output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset_version,
            do_constant_folding=True,
            verbose=False,
        )
        print(f"✓ ONNX export successful!")

        # Get file size
        file_size = os.path.getsize(args.output_path) / (1024 ** 3)  # GB
        print(f"  File size: {file_size:.2f} GB")

    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        raise

    # Optionally simplify
    if args.simplify:
        print("\nSimplifying ONNX model...")
        try:
            import onnx
            from onnxsim import simplify

            model_onnx = onnx.load(args.output_path)
            model_simplified, check = simplify(model_onnx)

            if check:
                simplified_path = args.output_path.replace(".onnx", "_simplified.onnx")
                onnx.save(model_simplified, simplified_path)
                print(f"✓ Simplified model saved to: {simplified_path}")

                file_size = os.path.getsize(simplified_path) / (1024 ** 3)  # GB
                print(f"  File size: {file_size:.2f} GB")
            else:
                print("✗ Simplification check failed")
        except ImportError:
            print("onnx-simplifier not installed. Skipping simplification.")
            print("Install with: pip install onnx-simplifier")
        except Exception as e:
            print(f"✗ Simplification failed: {e}")

    # Print summary
    print("\n" + "="*60)
    print("Export completed successfully!")
    print("="*60)
    print("\nExport Summary:")
    print(f"  Output file: {args.output_path}")
    if args.num_blocks is not None:
        print(f"  ⚠️  Blocks exported: {args.num_blocks}/30 (partial model for visualization)")
    else:
        print(f"  Blocks exported: 30/30 (complete model)")

    num_frames_used = args.num_frames_export if args.num_frames_export else args.num_output_frames
    if args.num_frames_export is not None:
        print(f"  ⚠️  Frames for export: {num_frames_used} (reduced for memory)")
    else:
        print(f"  Frames for export: {num_frames_used}")

    print(f"  Mode: {args.mode}")
    print(f"  Batch size: {args.batch_size}")
    print("="*60)


def main():
    args = parse_args()
    export_to_onnx(args)


if __name__ == "__main__":
    main()
