"""
Tests for the refactored pipeline.

This ensures the refactored code maintains the same functionality as the legacy code.
"""

import torch
import pytest
from pipeline import (
    PipelineConfig,
    ActionDict,
    PrerecordedActionStrategy,
    CacheManager,
    ConditionProcessor,
)


class TestActionStrategies:
    """Test action input strategies."""

    def test_action_dict_validation(self):
        """Test ActionDict validation."""
        # Valid universal action
        action = ActionDict(mouse=torch.tensor([0.1, 0.0]), keyboard=torch.tensor([1, 0, 0, 0]))
        action.validate('universal')  # Should not raise

        # Invalid universal action (missing mouse)
        action_invalid = ActionDict(keyboard=torch.tensor([1, 0, 0, 0]))
        with pytest.raises(ValueError):
            action_invalid.validate('universal')

        # Valid templerun action (no mouse needed)
        action_templerun = ActionDict(keyboard=torch.tensor([0, 1, 0, 0, 0, 0, 0]))
        action_templerun.validate('templerun')  # Should not raise

    def test_prerecorded_strategy(self):
        """Test PrerecordedActionStrategy."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        actions = [
            ActionDict(mouse=torch.tensor([0.0, 0.1]), keyboard=torch.tensor([1, 0, 0, 0])),
            ActionDict(mouse=torch.tensor([0.1, 0.0]), keyboard=torch.tensor([0, 0, 0, 1])),
        ]

        strategy = PrerecordedActionStrategy('universal', actions, device=device)

        # Get first action
        action0 = strategy.get_action(0)
        assert torch.allclose(action0['mouse'], torch.tensor([0.0, 0.1], device=device))
        assert torch.allclose(action0['keyboard'], torch.tensor([1, 0, 0, 0], device=device))

        # Get second action
        action1 = strategy.get_action(1)
        assert torch.allclose(action1['mouse'], torch.tensor([0.1, 0.0], device=device))

        # Out of bounds should raise
        with pytest.raises(IndexError):
            strategy.get_action(2)


class TestPipelineConfig:
    """Test pipeline configuration."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = PipelineConfig(mode='universal')

        assert config.mode == 'universal'
        assert config.model.num_transformer_blocks == 30
        assert config.model.frame_seq_length == 880
        assert config.cache.local_attn_size == 15
        assert config.vae.temporal_compression == 4
        assert config.vae.latent_channels == 16

    def test_invalid_mode(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError):
            PipelineConfig(mode='invalid_mode')

    def test_vae_temporal_calculations(self):
        """Test VAE temporal calculation utilities."""
        vae_config = PipelineConfig().vae

        # Test latent frame count calculation
        assert vae_config.get_latent_frame_count(1) == 1
        assert vae_config.get_latent_frame_count(5) == 2  # 1 + (5-1)//4 = 1 + 1 = 2
        assert vae_config.get_latent_frame_count(9) == 3  # 1 + (9-1)//4 = 1 + 2 = 3

        # Test video frame count calculation
        assert vae_config.get_video_frame_count(1) == 1
        assert vae_config.get_video_frame_count(2) == 5   # 1 + (2-1)*4 = 5
        assert vae_config.get_video_frame_count(3) == 9   # 1 + (3-1)*4 = 9

        # Test action condition length
        assert vae_config.get_action_condition_length(1) == 1
        assert vae_config.get_action_condition_length(2) == 5  # 1 + 4*(2-1) = 5
        assert vae_config.get_action_condition_length(3) == 9  # 1 + 4*(3-1) = 9


class TestCacheManager:
    """Test KV cache management."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        config = PipelineConfig()
        cache_manager = CacheManager(
            model_config=config.model,
            cache_config=config.cache,
            device=torch.device('cpu'),
            dtype=torch.float32
        )

        assert not cache_manager.is_initialized()

        cache_manager.initialize_all_caches(batch_size=1)
        assert cache_manager.is_initialized()

        # Check cache structure
        visual_cache, mouse_cache, keyboard_cache, crossattn_cache = cache_manager.get_caches()

        # Should have one cache dict per transformer block
        assert len(visual_cache) == config.model.num_transformer_blocks
        assert len(mouse_cache) == config.model.num_transformer_blocks
        assert len(keyboard_cache) == config.model.num_transformer_blocks
        assert len(crossattn_cache) == config.model.num_transformer_blocks

        # Check visual cache shape
        visual_kv_size = config.cache.local_attn_size * config.model.frame_seq_length
        assert visual_cache[0]['k'].shape == (
            1, visual_kv_size, config.model.num_attention_heads, config.model.head_dim
        )

    def test_cache_reset(self):
        """Test cache reset functionality."""
        config = PipelineConfig()
        cache_manager = CacheManager(
            model_config=config.model,
            cache_config=config.cache,
            device=torch.device('cpu'),
            dtype=torch.float32
        )

        cache_manager.initialize_all_caches(batch_size=1)
        visual_cache, _, _, crossattn_cache = cache_manager.get_caches()

        # Modify cache indices
        visual_cache[0]['global_end_index'] = torch.tensor([100])
        crossattn_cache[0]['is_init'] = True

        # Reset
        cache_manager.reset_all_caches()

        # Check indices are reset
        assert visual_cache[0]['global_end_index'].item() == 0
        assert crossattn_cache[0]['is_init'] == False


class TestConditionProcessor:
    """Test condition processing."""

    def test_action_sequence_length(self):
        """Test action sequence length calculation."""
        config = PipelineConfig()
        processor = ConditionProcessor(config.vae, mode='universal')

        # Test sequence length calculation
        assert processor.get_action_sequence_length(1) == 1
        assert processor.get_action_sequence_length(2) == 5
        assert processor.get_action_sequence_length(10) == 37  # 1 + 4*9

    def test_slice_block_conditions(self):
        """Test slicing conditions for a block."""
        config = PipelineConfig()
        processor = ConditionProcessor(config.vae, mode='universal')

        # Create dummy conditions
        batch_size = 1
        num_frames = 5
        conditional_dict = {
            'cond_concat': torch.randn(batch_size, 20, num_frames, 44, 80),
            'visual_context': torch.randn(batch_size, 257, 1536),
            'mouse_cond': torch.randn(batch_size, 50, 2),
            'keyboard_cond': torch.randn(batch_size, 50, 4)
        }

        # Slice first block
        sliced, _ = processor.slice_block_conditions(
            conditional_dict,
            current_start_frame=0,
            num_frames=2
        )

        # Check shapes
        assert sliced['cond_concat'].shape[2] == 2  # 2 frames
        assert sliced['visual_context'].shape == conditional_dict['visual_context'].shape
        assert sliced['mouse_cond'].shape[1] == 5  # 1 + 4*1 = 5
        assert sliced['keyboard_cond'].shape[1] == 5

    def test_action_replacement(self):
        """Test action replacement in conditions."""
        config = PipelineConfig()
        processor = ConditionProcessor(config.vae, mode='universal')

        # Create dummy conditions
        batch_size = 1
        num_frames = 5
        conditional_dict = {
            'cond_concat': torch.randn(batch_size, 20, num_frames, 44, 80),
            'visual_context': torch.randn(batch_size, 257, 1536),
            'mouse_cond': torch.zeros(batch_size, 50, 2),
            'keyboard_cond': torch.zeros(batch_size, 50, 4)
        }

        # Replace action
        new_action = ActionDict(
            mouse=torch.tensor([0.5, 0.5]),
            keyboard=torch.tensor([1.0, 0.0, 0.0, 0.0])
        )

        sliced, updated = processor.slice_block_conditions(
            conditional_dict,
            current_start_frame=0,
            num_frames=1,
            replace_action=new_action
        )

        # Check that actions were replaced
        assert torch.allclose(updated['mouse_cond'][0, 0], torch.tensor([0.5, 0.5]))
        assert torch.allclose(updated['keyboard_cond'][0, 0, 0], torch.tensor(1.0))


def test_backward_compatibility():
    """Test that legacy imports still work."""
    from pipeline import CausalInferencePipeline, CausalInferenceStreamingPipeline

    # These should be the legacy classes
    from pipeline.causal_inference import (
        CausalInferencePipeline as LegacyBatch,
        CausalInferenceStreamingPipeline as LegacyStreaming
    )

    assert CausalInferencePipeline is LegacyBatch
    assert CausalInferenceStreamingPipeline is LegacyStreaming


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
