"""
Tests for FlashInfer-based KV Cache and Attention modules.

Run with:
    pytest tests/test_flashinfer_kv_cache.py -v
"""

import pytest
import torch
import math

# Skip all tests if FlashInfer is not available
try:
    import flashinfer
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FLASHINFER_AVAILABLE,
    reason="FlashInfer not installed"
)


class TestPrecomputedRoPE3D:
    """Tests for PrecomputedRoPE3DCache."""

    def test_initialization(self):
        """Test that PrecomputedRoPE3DCache initializes correctly."""
        from wan.modules.flashinfer_attention import PrecomputedRoPE3DCache

        # Create mock frequencies (similar to rope_params output)
        head_dim = 128
        freqs = torch.randn(1024, head_dim // 2)

        cache = PrecomputedRoPE3DCache(
            freqs=freqs,
            max_frames=64,
            height=22,
            width=40,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # Check grid shape
        assert cache.freq_grid.shape == (64, 22, 40, head_dim // 2)

    def test_get_freqs_for_frame_range(self):
        """Test frequency extraction for frame ranges."""
        from wan.modules.flashinfer_attention import PrecomputedRoPE3DCache

        head_dim = 128
        freqs = torch.randn(1024, head_dim // 2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cache = PrecomputedRoPE3DCache(
            freqs=freqs,
            max_frames=64,
            height=22,
            width=40,
            device=device,
        )

        # Get frequencies for first 3 frames
        extracted = cache.get_freqs_for_frame_range(0, 3)

        # Expected shape: [3 * 22 * 40, 1, head_dim // 2]
        expected_seq_len = 3 * 22 * 40
        assert extracted.shape == (expected_seq_len, 1, head_dim // 2)

        # Get frequencies for frames 5-8
        extracted2 = cache.get_freqs_for_frame_range(5, 3)
        assert extracted2.shape == (expected_seq_len, 1, head_dim // 2)


class TestApplyRoPE3DPrecomputed:
    """Tests for apply_rope_3d_precomputed function."""

    def test_apply_rope(self):
        """Test that RoPE application produces correct output shape."""
        from wan.modules.flashinfer_attention import (
            PrecomputedRoPE3DCache,
            apply_rope_3d_precomputed,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 2
        seq_len = 22 * 40  # One frame
        num_heads = 12
        head_dim = 128

        # Create input tensor
        x = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Create precomputed frequencies
        freqs = torch.randn(1024, head_dim // 2, device=device)
        cache = PrecomputedRoPE3DCache(
            freqs=freqs,
            max_frames=64,
            height=22,
            width=40,
            device=device,
        )

        # Get frequencies for one frame
        precomputed_freqs = cache.get_freqs_for_frame_range(0, 1)

        # Apply RoPE
        output = apply_rope_3d_precomputed(x, precomputed_freqs)

        # Check output shape matches input
        assert output.shape == x.shape

        # Check output is not identical to input (rotation was applied)
        assert not torch.allclose(output, x)


class TestFlashInferCausalSelfAttention:
    """Tests for FlashInferCausalSelfAttention module."""

    @pytest.fixture
    def attention_module(self):
        """Create a test attention module."""
        from wan.modules.flashinfer_attention import FlashInferCausalSelfAttention

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = FlashInferCausalSelfAttention(
            dim=1536,
            num_heads=12,
            local_attn_size=5,
            sink_size=0,
            qk_norm=True,
            eps=1e-6,
            max_frames=64,
            height=22,
            width=40,
        ).to(device)

        return module

    def test_forward_without_cache(self, attention_module):
        """Test forward pass without KV cache."""
        device = next(attention_module.parameters()).device
        batch_size = 1
        seq_len = 22 * 40  # One frame
        dim = 1536

        # Create inputs
        x = torch.randn(batch_size, seq_len, dim, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.long)
        grid_sizes = torch.tensor([1, 22, 40], dtype=torch.long)
        freqs = torch.randn(1024, 64, device=device)  # head_dim // 2

        # Create mock block mask (not used in this test)
        from torch.nn.attention.flex_attention import create_block_mask

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask,
            B=None, H=None,
            Q_LEN=math.ceil(seq_len / 128) * 128,
            KV_LEN=math.ceil(seq_len / 128) * 128,
            _compile=False,
            device=device
        )

        # Forward pass
        output = attention_module(
            x=x,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            block_mask=block_mask,
            kv_cache=None,
        )

        assert output.shape == x.shape

    def test_forward_with_cache(self, attention_module):
        """Test forward pass with KV cache."""
        device = next(attention_module.parameters()).device
        dtype = torch.bfloat16  # FlashInfer requires float16/bfloat16
        batch_size = 1
        frame_seq_len = 22 * 40
        dim = 1536
        cache_size = 5 * frame_seq_len

        # Convert attention module to bfloat16
        attention_module = attention_module.to(dtype)

        # Create inputs for incremental forward (FlashInfer requires float16/bfloat16)
        x = torch.randn(batch_size, frame_seq_len, dim, device=device, dtype=dtype)
        seq_lens = torch.tensor([frame_seq_len], dtype=torch.long)
        grid_sizes = torch.tensor([1, 22, 40], dtype=torch.long)
        freqs = torch.randn(1024, 64, device=device, dtype=dtype)

        # Create KV cache (must match input dtype)
        kv_cache = {
            "k": torch.zeros(batch_size, cache_size, 12, 128, device=device, dtype=dtype),
            "v": torch.zeros(batch_size, cache_size, 12, 128, device=device, dtype=dtype),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        }

        # Forward pass with cache
        output = attention_module(
            x=x,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            block_mask=None,
            kv_cache=kv_cache,
            current_start=0,
        )

        assert output.shape == x.shape

        # Check cache was updated
        assert kv_cache["global_end_index"].item() == frame_seq_len
        assert kv_cache["local_end_index"].item() == frame_seq_len


class TestFlashInferKVCacheWrapper:
    """Tests for FlashInferKVCacheWrapper."""

    def test_initialization(self):
        """Test wrapper initialization."""
        from pipeline.flashinfer_inference import FlashInferKVCacheWrapper

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        wrapper = FlashInferKVCacheWrapper(
            batch_size=1,
            num_layers=30,
            num_heads=12,
            head_dim=128,
            max_seq_len=15 * 880,
            local_attn_size=5,
            dtype=torch.bfloat16,
            device=device,
        )

        # Check number of layers
        assert len(wrapper) == 30

        # Check cache shape
        cache = wrapper[0]
        assert cache["k"].shape == (1, 5 * 880, 12, 128)
        assert cache["v"].shape == (1, 5 * 880, 12, 128)

    def test_reset(self):
        """Test cache reset."""
        from pipeline.flashinfer_inference import FlashInferKVCacheWrapper

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        wrapper = FlashInferKVCacheWrapper(
            batch_size=1,
            num_layers=5,
            num_heads=12,
            head_dim=128,
            max_seq_len=15 * 880,
            local_attn_size=5,
            dtype=torch.bfloat16,
            device=device,
        )

        # Modify indices
        wrapper[0]["global_end_index"].fill_(100)
        wrapper[0]["local_end_index"].fill_(50)

        # Reset
        wrapper.reset()

        # Check indices are reset
        assert wrapper[0]["global_end_index"].item() == 0
        assert wrapper[0]["local_end_index"].item() == 0


class TestCUDAGraphInfrastructure:
    """Tests for CUDA Graph infrastructure."""

    def test_static_tensor_pool(self):
        """Test StaticTensorPool allocation and access."""
        from wan.modules.cuda_graph_inference import StaticTensorPool

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use float32 dtype to match randn default
        pool = StaticTensorPool(device, dtype=torch.float32)

        # Allocate tensor with explicit dtype
        tensor = pool.allocate("test", (2, 3, 4), dtype=torch.float32)
        assert tensor.shape == (2, 3, 4)
        assert tensor.device.type == device.type

        # Get same tensor
        tensor2 = pool.get("test")
        assert tensor is tensor2

        # Copy data (ensure same dtype)
        data = torch.randn(2, 3, 4, device=device, dtype=torch.float32)
        pool.copy_to("test", data)
        assert torch.allclose(pool.get("test"), data)

    def test_cuda_graph_config(self):
        """Test CUDAGraphConfig defaults."""
        from wan.modules.cuda_graph_inference import CUDAGraphConfig

        config = CUDAGraphConfig()
        assert config.batch_size == 1
        assert config.num_frames_per_block == 1
        assert config.warmup_iterations == 3


class TestIntegration:
    """Integration tests for the full pipeline."""

    @pytest.mark.slow
    def test_replace_attention_with_flashinfer(self):
        """Test that attention replacement works correctly."""
        from wan.modules.flashinfer_attention import replace_attention_with_flashinfer
        from wan.modules.causal_model import CausalWanSelfAttention

        # Create a simple model with CausalWanSelfAttention
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = CausalWanSelfAttention(
                    dim=1536,
                    num_heads=12,
                    local_attn_size=5,
                )

        model = SimpleModel()

        # Check original type
        assert isinstance(model.attn, CausalWanSelfAttention)

        # Replace with FlashInfer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model = replace_attention_with_flashinfer(model)

        # Check new type
        from wan.modules.flashinfer_attention import FlashInferCausalSelfAttention
        assert isinstance(model.attn, FlashInferCausalSelfAttention)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
