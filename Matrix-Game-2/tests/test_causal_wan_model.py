"""
Unit tests for CausalWanModel

This test suite validates:
1. Model output correctness (numerical equivalence)
2. KV cache behavior
3. Performance benchmarking

Usage:
    # Run correctness test only
    pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_forward_correctness -v

    # Run with KV cache
    pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_forward_with_kv_cache -v

    # Run performance benchmark
    pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_performance_benchmark -v

    # Run all tests
    pytest tests/test_causal_wan_model.py -v
"""

import torch
import pytest
import time
from typing import Dict, List, Optional
from omegaconf import OmegaConf
import numpy as np


class TestCausalWanModel:
    """Test suite for CausalWanModel and its optimized variants"""

    @pytest.fixture(scope="class")
    def device(self):
        """Test device"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture(scope="class")
    def dtype(self):
        """Test dtype"""
        return torch.bfloat16

    @pytest.fixture(scope="class")
    def model_config(self):
        """Model configuration from inference config"""
        config = OmegaConf.load("configs/inference_yaml/inference_universal.yaml")
        return config

    @pytest.fixture(scope="class")
    def original_model(self, device, dtype):
        """Load original CausalWanModel

        NOTE: This requires downloading the pretrained weights first:
            huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir Matrix-Game-2.0

        The checkpoint path should be specified via pytest argument or environment variable.
        If not provided, model will be initialized with random weights (for testing structure only).
        """
        from wan.modules.causal_model import CausalWanModel
        from safetensors.torch import load_file
        import os

        # Load model architecture from config
        model = CausalWanModel.from_config("configs/distilled_model/universal")
        model.eval()

        # Try to load pretrained weights
        # Priority: 1. env var, 2. default path, 3. skip (random weights)
        checkpoint_path = os.environ.get(
            "MODEL_CHECKPOINT_PATH",
            os.path.expanduser("~/models/Matrix-Game-2.0/base_distilled_model/base_distill.safetensors")
        )

        if os.path.exists(checkpoint_path):
            print(f"\nLoading pretrained weights from: {checkpoint_path}")
            state_dict = load_file(checkpoint_path)

            # Handle 'model.' prefix in checkpoint keys (from WanDiffusionWrapper)
            if all(key.startswith('model.') for key in list(state_dict.keys())[:10]):
                print("Detected 'model.' prefix in checkpoint, removing it...")
                state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}

            # Load with strict=False to handle architecture mismatches
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            # Calculate how many keys actually loaded
            total_model_keys = len(model.state_dict())
            loaded_keys = total_model_keys - len(missing_keys)

            print(f"✓ Loaded {loaded_keys}/{total_model_keys} parameters ({100*loaded_keys/total_model_keys:.1f}%)")

            if missing_keys:
                print(f"⚠️  Missing {len(missing_keys)} keys: {missing_keys[:3]}...")
            if unexpected_keys:
                print(f"⚠️  Unexpected {len(unexpected_keys)} keys: {unexpected_keys[:3]}...")

            # Warn if most weights are missing
            if loaded_keys < total_model_keys * 0.5:
                print("🚨 WARNING: Less than 50% of weights loaded! Check checkpoint compatibility.")
                print("   Model is mostly using random initialization - results will be meaningless!")
            elif loaded_keys == total_model_keys:
                print("✅ All weights loaded successfully!")
            else:
                print("✓ Partial weight loading OK (some layers randomly initialized)")
        else:
            print(f"\n⚠️  WARNING: Checkpoint not found at {checkpoint_path}")
            print("Model will use random weights. Tests will verify structure only.")
            print("To use pretrained weights, run:")
            print("  huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir Matrix-Game-2.0")

        model.to(device=device, dtype=dtype)
        model.requires_grad_(False)

        # Set num_frame_per_block to match inference config (3 frames)
        model.num_frame_per_block = 3

        return model

    @pytest.fixture(scope="class")
    def test_inputs(self, device, dtype):
        """Create standard test inputs"""
        batch_size = 1
        num_channels = 16
        num_frames = 3  # num_frame_per_block
        height = 44
        width = 80

        # Main input
        x = torch.randn(batch_size, num_channels, num_frames, height, width,
                       device=device, dtype=dtype)

        # Timestep
        t = torch.tensor([500], device=device, dtype=torch.long).repeat(batch_size, num_frames)

        # Conditional inputs
        cond_concat = torch.randn(batch_size, 20, num_frames, height, width,
                                  device=device, dtype=dtype)
        visual_context = torch.randn(batch_size, 257, 1280, device=device, dtype=dtype)

        # Action conditions
        mouse_cond = torch.randn(batch_size, 1 + 4 * (num_frames - 1), 2,
                                device=device, dtype=dtype)
        keyboard_cond = torch.randn(batch_size, 1 + 4 * (num_frames - 1), 4,
                                   device=device, dtype=dtype)

        return {
            "x": x,
            "t": t,
            "cond_concat": cond_concat,
            "visual_context": visual_context,
            "mouse_cond": mouse_cond,
            "keyboard_cond": keyboard_cond,
        }

    @pytest.fixture(scope="class")
    def kv_cache_structure(self, device, dtype):
        """Create KV cache structure for testing"""
        num_blocks = 30
        batch_size = 1
        local_attn_size = 15
        frame_seq_length = 880

        kv_cache_size = local_attn_size * frame_seq_length

        kv_cache = []
        for _ in range(num_blocks):
            kv_cache.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        kv_cache_keyboard = []
        kv_cache_mouse = []
        for _ in range(num_blocks):
            kv_cache_keyboard.append({
                "k": torch.zeros([batch_size, local_attn_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, local_attn_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_mouse.append({
                "k": torch.zeros([batch_size * frame_seq_length, local_attn_size, 16, 64],
                                dtype=dtype, device=device),
                "v": torch.zeros([batch_size * frame_seq_length, local_attn_size, 16, 64],
                                dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        crossattn_cache = []
        for _ in range(num_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })

        return {
            "kv_cache": kv_cache,
            "kv_cache_keyboard": kv_cache_keyboard,
            "kv_cache_mouse": kv_cache_mouse,
            "crossattn_cache": crossattn_cache,
        }

    def test_forward_correctness(self, original_model, test_inputs, device):
        """Test basic forward pass without KV cache"""
        print("\n" + "="*80)
        print("TEST: Forward Pass Correctness (Training Mode)")
        print("="*80)

        with torch.no_grad():
            output = original_model(
                x=test_inputs["x"],
                t=test_inputs["t"],
                visual_context=test_inputs["visual_context"],
                cond_concat=test_inputs["cond_concat"],
                mouse_cond=test_inputs["mouse_cond"],
                keyboard_cond=test_inputs["keyboard_cond"],
            )

        # Validate output shape
        expected_shape = (1, 16, 3, 44, 80)  # [B, C, F, H, W]
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

        # Check for NaN/Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"✓ Output mean: {output.mean().item():.4f}")
        print(f"✓ Output std: {output.std().item():.4f}")
        print("✓ No NaN/Inf detected")
        print("\nTest PASSED ✓")

    def test_forward_with_kv_cache(self, original_model, test_inputs, kv_cache_structure, device):
        """Test forward pass with KV cache (inference mode)"""
        print("\n" + "="*80)
        print("TEST: Forward Pass with KV Cache (Inference Mode)")
        print("="*80)

        current_start = 0

        with torch.no_grad():
            output = original_model(
                x=test_inputs["x"],
                t=test_inputs["t"],
                visual_context=test_inputs["visual_context"],
                cond_concat=test_inputs["cond_concat"],
                mouse_cond=test_inputs["mouse_cond"],
                keyboard_cond=test_inputs["keyboard_cond"],
                kv_cache=kv_cache_structure["kv_cache"],
                kv_cache_keyboard=kv_cache_structure["kv_cache_keyboard"],
                kv_cache_mouse=kv_cache_structure["kv_cache_mouse"],
                crossattn_cache=kv_cache_structure["crossattn_cache"],
                current_start=current_start,
            )

        # Validate output
        expected_shape = (1, 16, 3, 44, 80)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        # Validate KV cache was updated
        assert kv_cache_structure["kv_cache"][0]["global_end_index"].item() > 0, \
            "KV cache was not updated"

        print(f"✓ Output shape: {output.shape}")
        print(f"✓ KV cache updated: global_end_index = {kv_cache_structure['kv_cache'][0]['global_end_index'].item()}")
        print(f"✓ KV cache updated: local_end_index = {kv_cache_structure['kv_cache'][0]['local_end_index'].item()}")
        print("✓ Cross-attention cache initialized:", kv_cache_structure["crossattn_cache"][0]["is_init"])
        print("\nTest PASSED ✓")

    @pytest.mark.parametrize("num_warmup,num_iterations", [(3, 10)])
    def test_performance_benchmark(self, original_model, test_inputs, kv_cache_structure,
                                   device, num_warmup, num_iterations):
        """Benchmark model performance"""
        print("\n" + "="*80)
        print("TEST: Performance Benchmark")
        print("="*80)

        # Warmup
        print(f"Warming up ({num_warmup} iterations)...")
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = original_model(
                    x=test_inputs["x"],
                    t=test_inputs["t"],
                    visual_context=test_inputs["visual_context"],
                    cond_concat=test_inputs["cond_concat"],
                    mouse_cond=test_inputs["mouse_cond"],
                    keyboard_cond=test_inputs["keyboard_cond"],
                    kv_cache=kv_cache_structure["kv_cache"],
                    kv_cache_keyboard=kv_cache_structure["kv_cache_keyboard"],
                    kv_cache_mouse=kv_cache_structure["kv_cache_mouse"],
                    crossattn_cache=kv_cache_structure["crossattn_cache"],
                    current_start=0,
                )
                torch.cuda.synchronize()

        # Benchmark
        print(f"Benchmarking ({num_iterations} iterations)...")
        latencies = []

        with torch.no_grad():
            for _ in range(num_iterations):
                # Reset cache for fair comparison
                for cache in kv_cache_structure["kv_cache"]:
                    cache["global_end_index"].fill_(0)
                    cache["local_end_index"].fill_(0)
                for cache in kv_cache_structure["crossattn_cache"]:
                    cache["is_init"] = False

                start = time.perf_counter()
                _ = original_model(
                    x=test_inputs["x"],
                    t=test_inputs["t"],
                    visual_context=test_inputs["visual_context"],
                    cond_concat=test_inputs["cond_concat"],
                    mouse_cond=test_inputs["mouse_cond"],
                    keyboard_cond=test_inputs["keyboard_cond"],
                    kv_cache=kv_cache_structure["kv_cache"],
                    kv_cache_keyboard=kv_cache_structure["kv_cache_keyboard"],
                    kv_cache_mouse=kv_cache_structure["kv_cache_mouse"],
                    crossattn_cache=kv_cache_structure["crossattn_cache"],
                    current_start=0,
                )
                torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms

        latencies = np.array(latencies)

        print("\nPerformance Results:")
        print("-" * 80)
        print(f"Mean latency:   {latencies.mean():.2f} ms")
        print(f"Median latency: {np.median(latencies):.2f} ms")
        print(f"Min latency:    {latencies.min():.2f} ms")
        print(f"Max latency:    {latencies.max():.2f} ms")
        print(f"Std deviation:  {latencies.std():.2f} ms")
        print(f"P95 latency:    {np.percentile(latencies, 95):.2f} ms")
        print(f"P99 latency:    {np.percentile(latencies, 99):.2f} ms")
        print("-" * 80)

        # Store baseline for comparison (saved to file for future reference)
        baseline_file = "tests/baseline_performance.txt"
        with open(baseline_file, "w") as f:
            f.write(f"Original Model Baseline\n")
            f.write(f"Mean: {latencies.mean():.2f} ms\n")
            f.write(f"Median: {np.median(latencies):.2f} ms\n")
            f.write(f"P95: {np.percentile(latencies, 95):.2f} ms\n")

        print(f"✓ Baseline saved to {baseline_file}")
        print("\nTest PASSED ✓")

    def test_model_comparison(self, original_model, test_inputs, kv_cache_structure, device):
        """
        Template test for comparing original model with optimized version.

        To use this with your optimized model:
        1. Import your optimized model class
        2. Load it similar to original_model fixture
        3. Run both models with same inputs
        4. Compare outputs and performance
        """
        print("\n" + "="*80)
        print("TEST: Model Comparison (Original vs Optimized)")
        print("="*80)
        print("NOTE: This is a template test. Implement your optimized model to enable comparison.")
        print("="*80)

        # TODO: Replace with your optimized model
        # from your_module import OptimizedCausalWanModel
        # optimized_model = OptimizedCausalWanModel.from_config(...)
        # optimized_model.eval()
        # optimized_model.to(device=device, dtype=dtype)

        # Example comparison code (uncomment when you have optimized model):
        """
        with torch.no_grad():
            original_output = original_model(
                x=test_inputs["x"],
                t=test_inputs["t"],
                visual_context=test_inputs["visual_context"],
                cond_concat=test_inputs["cond_concat"],
                mouse_cond=test_inputs["mouse_cond"],
                keyboard_cond=test_inputs["keyboard_cond"],
                kv_cache=kv_cache_structure["kv_cache"],
                kv_cache_keyboard=kv_cache_structure["kv_cache_keyboard"],
                kv_cache_mouse=kv_cache_structure["kv_cache_mouse"],
                crossattn_cache=kv_cache_structure["crossattn_cache"],
                current_start=0,
            )

            optimized_output = optimized_model(
                x=test_inputs["x"],
                t=test_inputs["t"],
                visual_context=test_inputs["visual_context"],
                cond_concat=test_inputs["cond_concat"],
                mouse_cond=test_inputs["mouse_cond"],
                keyboard_cond=test_inputs["keyboard_cond"],
                kv_cache=kv_cache_structure["kv_cache"],
                kv_cache_keyboard=kv_cache_structure["kv_cache_keyboard"],
                kv_cache_mouse=kv_cache_structure["kv_cache_mouse"],
                crossattn_cache=kv_cache_structure["crossattn_cache"],
                current_start=0,
            )

        # Check numerical equivalence
        max_diff = (original_output - optimized_output).abs().max().item()
        mean_diff = (original_output - optimized_output).abs().mean().item()

        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")

        # Assert similarity (adjust tolerance as needed)
        assert max_diff < 1e-2, f"Outputs differ too much: max_diff={max_diff}"

        print("✓ Outputs are numerically equivalent")
        """

        pytest.skip("Optimized model not implemented yet")


if __name__ == "__main__":
    # Allow running directly for quick testing
    pytest.main([__file__, "-v", "-s"])
