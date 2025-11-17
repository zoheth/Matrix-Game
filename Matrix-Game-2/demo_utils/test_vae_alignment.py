"""
Test to verify that VaeDecoder3d (new_vae.py) produces the same results as VAEDecoder3d (vae_block3.py)
"""
import torch
import torch.nn as nn
from typing import List, Optional
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_utils.new_vae import VaeDecoder3d as NewVaeDecoder3d, CacheState
from demo_utils.vae_block3 import VAEDecoder3d as OldVAEDecoder3d


def copy_rms_norm_weights(new_norm, old_norm):
    """Copy RmsNorm weights handling shape differences"""
    # Old: gamma shape [C, 1, 1] or [C, 1, 1, 1]
    # New: gamma shape [C]
    old_gamma = old_norm.gamma.data
    if old_gamma.dim() > 1:
        old_gamma = old_gamma.squeeze()
    new_norm.gamma.data.copy_(old_gamma)


def copy_residual_block_weights(new_block, old_block):
    """Copy ResidualBlock weights handling structure differences"""
    # Old structure: residual.0 (RMS_norm), residual.2 (Conv), residual.3 (RMS_norm), residual.6 (Conv)
    # New structure: norm1, conv1, norm2, conv2

    old_state = old_block.state_dict()

    # Copy norm1 from residual.0
    copy_rms_norm_weights(new_block.norm1, old_block.residual[0])

    # Copy conv1 from residual.2
    new_block.conv1.weight.data.copy_(old_state['residual.2.weight'])
    new_block.conv1.bias.data.copy_(old_state['residual.2.bias'])

    # Copy norm2 from residual.3
    copy_rms_norm_weights(new_block.norm2, old_block.residual[3])

    # Copy conv2 from residual.6
    new_block.conv2.weight.data.copy_(old_state['residual.6.weight'])
    new_block.conv2.bias.data.copy_(old_state['residual.6.bias'])

    # Copy shortcut if present
    if 'shortcut.weight' in old_state and hasattr(new_block, 'shortcut'):
        if hasattr(new_block.shortcut, 'weight'):
            new_block.shortcut.weight.data.copy_(old_state['shortcut.weight'])
            if 'shortcut.bias' in old_state:
                new_block.shortcut.bias.data.copy_(old_state['shortcut.bias'])


def copy_attention_block_weights(new_attn, old_attn):
    """Copy SpatialAttentionBlock weights handling naming differences"""
    # Old: norm.gamma [C, 1, 1], proj
    # New: norm.gamma [C], proj_out

    old_state = old_attn.state_dict()

    # Copy norm
    copy_rms_norm_weights(new_attn.norm, old_attn.norm)

    # Copy to_qkv
    new_attn.to_qkv.weight.data.copy_(old_state['to_qkv.weight'])
    new_attn.to_qkv.bias.data.copy_(old_state['to_qkv.bias'])

    # Copy proj -> proj_out
    new_attn.proj_out.weight.data.copy_(old_state['proj.weight'])
    new_attn.proj_out.bias.data.copy_(old_state['proj.bias'])


def copy_weights(new_model: NewVaeDecoder3d, old_model: OldVAEDecoder3d):
    """Copy weights from old model to new model to ensure they have identical parameters"""
    print("Copying weights from old model to new model...")

    # Copy conv1
    new_model.conv1.weight.data.copy_(old_model.conv1.weight.data)
    new_model.conv1.bias.data.copy_(old_model.conv1.bias.data)
    print("  Copied conv1")

    # Copy middle blocks
    for i, (new_layer, old_layer) in enumerate(zip(new_model.middle, old_model.middle)):
        if type(new_layer).__name__ == 'ResidualBlock':
            copy_residual_block_weights(new_layer, old_layer)
            print(f"  Copied middle ResidualBlock {i}")
        elif type(new_layer).__name__ == 'SpatialAttentionBlock':
            copy_attention_block_weights(new_layer, old_layer)
            print(f"  Copied middle SpatialAttentionBlock {i}")

    # Copy upsamples
    new_idx = 0
    old_idx = 0

    while new_idx < len(new_model.upsamples) and old_idx < len(old_model.upsamples):
        new_layer = new_model.upsamples[new_idx]
        old_layer = old_model.upsamples[old_idx]

        # Match layer types
        if type(new_layer).__name__ == type(old_layer).__name__:
            if type(new_layer).__name__ == 'ResidualBlock':
                copy_residual_block_weights(new_layer, old_layer)
                print(f"  Copied upsamples ResidualBlock at index {new_idx}/{old_idx}")
            elif type(new_layer).__name__ == 'SpatialAttentionBlock':
                copy_attention_block_weights(new_layer, old_layer)
                print(f"  Copied upsamples SpatialAttentionBlock at index {new_idx}/{old_idx}")
            elif type(new_layer).__name__ == 'Resample':
                # Copy resample layers
                if hasattr(new_layer, 'resample') and hasattr(old_layer, 'resample'):
                    # Copy spatial resample (Conv2d)
                    for new_sublayer, old_sublayer in zip(new_layer.resample, old_layer.resample):
                        if hasattr(new_sublayer, 'state_dict') and hasattr(old_sublayer, 'state_dict'):
                            new_sublayer.load_state_dict(old_sublayer.state_dict())

                # Copy time_conv if exists
                if hasattr(new_layer, 'time_conv') and hasattr(old_layer, 'time_conv'):
                    new_layer.time_conv.weight.data.copy_(old_layer.time_conv.weight.data)
                    new_layer.time_conv.bias.data.copy_(old_layer.time_conv.bias.data)

                print(f"  Copied upsamples Resample at index {new_idx}/{old_idx}")
            new_idx += 1
            old_idx += 1
        else:
            print(f"  Layer mismatch at new_idx={new_idx}, old_idx={old_idx}: {type(new_layer).__name__} vs {type(old_layer).__name__}")
            new_idx += 1

    # Copy head
    copy_rms_norm_weights(new_model.head_norm, old_model.head[0])
    new_model.head_conv.weight.data.copy_(old_model.head[2].weight.data)
    new_model.head_conv.bias.data.copy_(old_model.head[2].bias.data)
    print("  Copied head")

    print("Weight copying completed!\n")


def initialize_old_cache(old_model: OldVAEDecoder3d, batch_size: int, device: torch.device) -> List[Optional[torch.Tensor]]:
    """Initialize cache for old model with correct number of slots"""
    # Old model uses 32 cache slots (decoder_conv_num = 32)
    feat_cache = [None] * old_model.decoder_conv_num
    return feat_cache


def initialize_new_cache(new_model: NewVaeDecoder3d) -> List[CacheState]:
    """Initialize cache states for new model"""
    # cache_states[0]: conv1
    # cache_states[1]: middle blocks
    # cache_states[2]: upsamples blocks
    # cache_states[3]: head_conv

    # Estimate size needed for each cache state
    cache_size = 50  # generous size to handle all layers

    print(f"Initializing 4 cache states for new model")
    cache_states = [CacheState(size=cache_size) for _ in range(4)]

    return cache_states


def test_single_frame_inference():
    """Test with single frame input (no caching)"""
    print("=" * 80)
    print("TEST 1: Single Frame Inference (No Caching)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Initialize models with same config
    config = {
        'dim': 96,
        'z_dim': 16,
        'dim_mult': [1, 2, 4, 4],
        'num_res_blocks': 2,
        'attn_scales': [],
        'dropout': 0.0
    }

    new_model = NewVaeDecoder3d(
        dim=config['dim'],
        z_dim=config['z_dim'],
        dim_mult=config['dim_mult'],
        num_res_blocks=config['num_res_blocks'],
        attn_scales=config['attn_scales'],
        temporal_upsample=[True, True, False],
        dropout=config['dropout']
    ).to(device).eval()

    old_model = OldVAEDecoder3d(
        dim=config['dim'],
        z_dim=config['z_dim'],
        dim_mult=config['dim_mult'],
        num_res_blocks=config['num_res_blocks'],
        attn_scales=config['attn_scales'],
        temperal_upsample=[True, True, False],
        dropout=config['dropout']
    ).to(device).eval()

    # Copy weights to ensure identical parameters
    copy_weights(new_model, old_model)

    # Create test input: [batch, channels, time, height, width]
    batch_size = 1
    z_dim = 16
    t = 1  # Single frame
    h, w = 32, 32

    x = torch.randn(batch_size, z_dim, t, h, w, device=device)

    print(f"Input shape: {x.shape}")

    # Run both models without caching
    with torch.no_grad():
        # New model without cache
        new_output = new_model(x, cache_states=None)

        # Old model without cache
        old_feat_cache = initialize_old_cache(old_model, batch_size, device)
        old_output, _ = old_model(x, feat_cache=old_feat_cache)

    print(f"New model output shape: {new_output.shape}")
    print(f"Old model output shape: {old_output.shape}")

    # Compare outputs
    max_diff = torch.max(torch.abs(new_output - old_output)).item()
    mean_diff = torch.mean(torch.abs(new_output - old_output)).item()
    relative_diff = mean_diff / (torch.mean(torch.abs(old_output)).item() + 1e-8)

    print(f"\nComparison Results:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Relative difference: {relative_diff:.6%}")

    tolerance = 1e-4
    if max_diff < tolerance:
        print(f"✓ PASSED: Outputs match within tolerance ({tolerance})")
        return True
    else:
        print(f"✗ FAILED: Outputs differ by more than tolerance ({tolerance})")
        return False


def test_streaming_inference():
    """Test with streaming inference (multiple single frames with caching)"""
    print("\n" + "=" * 80)
    print("TEST 2: Streaming Inference (Frame-by-Frame with Caching)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Initialize models
    config = {
        'dim': 96,
        'z_dim': 16,
        'dim_mult': [1, 2, 4, 4],
        'num_res_blocks': 2,
        'attn_scales': [],
        'dropout': 0.0
    }

    new_model = NewVaeDecoder3d(
        dim=config['dim'],
        z_dim=config['z_dim'],
        dim_mult=config['dim_mult'],
        num_res_blocks=config['num_res_blocks'],
        attn_scales=config['attn_scales'],
        temporal_upsample=[True, True, False],
        dropout=config['dropout']
    ).to(device).eval()

    old_model = OldVAEDecoder3d(
        dim=config['dim'],
        z_dim=config['z_dim'],
        dim_mult=config['dim_mult'],
        num_res_blocks=config['num_res_blocks'],
        attn_scales=config['attn_scales'],
        temperal_upsample=[True, True, False],
        dropout=config['dropout']
    ).to(device).eval()

    copy_weights(new_model, old_model)

    # Create test sequence
    batch_size = 1
    z_dim = 16
    num_frames = 5
    h, w = 32, 32

    frames = [torch.randn(batch_size, z_dim, 1, h, w, device=device) for _ in range(num_frames)]

    print(f"Testing streaming with {num_frames} frames, each of shape {frames[0].shape}")

    # Initialize caches
    new_cache_states = initialize_new_cache(new_model)
    old_feat_cache = initialize_old_cache(old_model, batch_size, device)

    new_outputs = []
    old_outputs = []

    with torch.no_grad():
        for i, frame in enumerate(frames):
            print(f"\nProcessing frame {i+1}/{num_frames}...")

            # Reset cache state indices for new model
            for cache_state in new_cache_states:
                cache_state.reset_index()

            # New model
            new_out = new_model(frame, cache_states=new_cache_states)
            new_outputs.append(new_out)

            # Old model
            old_out, old_feat_cache = old_model(frame, feat_cache=old_feat_cache)
            old_outputs.append(old_out)

            # Compare this frame
            max_diff = torch.max(torch.abs(new_out - old_out)).item()
            mean_diff = torch.mean(torch.abs(new_out - old_out)).item()
            print(f"  Frame {i+1} - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    # Final comparison
    print(f"\n{'=' * 40}")
    print("Final Results:")
    print(f"{'=' * 40}")

    all_passed = True
    tolerance = 1e-4

    for i, (new_out, old_out) in enumerate(zip(new_outputs, old_outputs)):
        max_diff = torch.max(torch.abs(new_out - old_out)).item()
        mean_diff = torch.mean(torch.abs(new_out - old_out)).item()
        relative_diff = mean_diff / (torch.mean(torch.abs(old_out)).item() + 1e-8)

        passed = max_diff < tolerance
        all_passed = all_passed and passed

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"Frame {i+1}: {status} - Max: {max_diff:.6f}, Mean: {mean_diff:.6f}, Relative: {relative_diff:.6%}")

    if all_passed:
        print(f"\n✓ ALL FRAMES PASSED: Outputs match within tolerance ({tolerance})")
        return True
    else:
        print(f"\n✗ SOME FRAMES FAILED: Outputs differ by more than tolerance ({tolerance})")
        return False


def test_batch_inference():
    """Test with multiple frames in batch (full sequence)"""
    print("\n" + "=" * 80)
    print("TEST 3: Batch Inference (Full Sequence)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Initialize models
    config = {
        'dim': 96,
        'z_dim': 16,
        'dim_mult': [1, 2, 4, 4],
        'num_res_blocks': 2,
        'attn_scales': [],
        'dropout': 0.0
    }

    new_model = NewVaeDecoder3d(
        dim=config['dim'],
        z_dim=config['z_dim'],
        dim_mult=config['dim_mult'],
        num_res_blocks=config['num_res_blocks'],
        attn_scales=config['attn_scales'],
        temporal_upsample=[True, True, False],
        dropout=config['dropout']
    ).to(device).eval()

    old_model = OldVAEDecoder3d(
        dim=config['dim'],
        z_dim=config['z_dim'],
        dim_mult=config['dim_mult'],
        num_res_blocks=config['num_res_blocks'],
        attn_scales=config['attn_scales'],
        temperal_upsample=[True, True, False],
        dropout=config['dropout']
    ).to(device).eval()

    copy_weights(new_model, old_model)

    # Create test input with multiple frames
    batch_size = 1
    z_dim = 16
    t = 4  # Multiple frames
    h, w = 32, 32

    x = torch.randn(batch_size, z_dim, t, h, w, device=device)

    print(f"Input shape: {x.shape}")

    # Note: Batch processing may differ due to caching strategy differences
    # This test is mainly to check if the models can handle batch inputs

    with torch.no_grad():
        # New model - may need to process frame by frame for caching
        print("\nProcessing with new model (no cache for batch mode)...")
        new_output = new_model(x, cache_states=None)

        # Old model - processes frame by frame internally
        print("Processing with old model (frame-by-frame with cache)...")
        old_feat_cache = initialize_old_cache(old_model, batch_size, device)

        # Old model expects frame-by-frame processing in wrapper
        old_outputs = []
        for i in range(t):
            old_out, old_feat_cache = old_model(x[:, :, i:i+1, :, :], feat_cache=old_feat_cache)
            old_outputs.append(old_out)
        old_output = torch.cat(old_outputs, dim=2)

    print(f"\nNew model output shape: {new_output.shape}")
    print(f"Old model output shape: {old_output.shape}")

    # Compare outputs
    max_diff = torch.max(torch.abs(new_output - old_output)).item()
    mean_diff = torch.mean(torch.abs(new_output - old_output)).item()
    relative_diff = mean_diff / (torch.mean(torch.abs(old_output)).item() + 1e-8)

    print(f"\nComparison Results:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Relative difference: {relative_diff:.6%}")

    tolerance = 1e-3  # Slightly higher tolerance for batch processing
    if max_diff < tolerance:
        print(f"✓ PASSED: Outputs match within tolerance ({tolerance})")
        return True
    else:
        print(f"✗ FAILED: Outputs differ by more than tolerance ({tolerance})")
        print(f"\nNote: Batch processing may show differences due to caching strategy variations.")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("VAE DECODER ALIGNMENT TEST SUITE")
    print("=" * 80)
    print("Comparing VaeDecoder3d (new_vae.py) vs VAEDecoder3d (vae_block3.py)\n")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    results = []

    # Run tests
    try:
        results.append(("Single Frame Inference", test_single_frame_inference()))
    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Single Frame Inference", False))

    try:
        results.append(("Streaming Inference", test_streaming_inference()))
    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Streaming Inference", False))

    try:
        results.append(("Batch Inference", test_batch_inference()))
    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Batch Inference", False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed! The implementations are aligned.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Please review the differences.")
        return 1


if __name__ == "__main__":
    exit(main())
