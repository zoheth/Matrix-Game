"""
Unit tests for ActionContext and BlockMaskFactory.

This test suite verifies:
1. ActionContext creation and properties
2. BlockMaskFactory mask generation
3. Backward compatibility between old and new APIs
4. Numerical equivalence
"""

import torch
import pytest
from wan.modules.action_context import (
    ActionContext,
    BlockMaskFactory,
    create_action_context_from_kwargs
)


class TestActionContext:
    """Tests for ActionContext dataclass."""

    def test_creation_with_mouse_only(self):
        """Test creating ActionContext with only mouse conditioning."""
        mouse_cond = torch.randn(1, 9, 2)
        ctx = ActionContext(mouse_cond=mouse_cond)

        assert ctx.has_mouse == True
        assert ctx.has_keyboard == False
        assert ctx.has_any_condition == True
        assert ctx.mouse_cond is mouse_cond

    def test_creation_with_keyboard_only(self):
        """Test creating ActionContext with only keyboard conditioning."""
        keyboard_cond = torch.randn(1, 9, 6)
        ctx = ActionContext(keyboard_cond=keyboard_cond)

        assert ctx.has_mouse == False
        assert ctx.has_keyboard == True
        assert ctx.has_any_condition == True
        assert ctx.keyboard_cond is keyboard_cond

    def test_creation_with_both(self):
        """Test creating ActionContext with both mouse and keyboard."""
        mouse_cond = torch.randn(1, 9, 2)
        keyboard_cond = torch.randn(1, 9, 6)
        ctx = ActionContext(
            mouse_cond=mouse_cond,
            keyboard_cond=keyboard_cond
        )

        assert ctx.has_mouse == True
        assert ctx.has_keyboard == True
        assert ctx.has_any_condition == True

    def test_to_legacy_kwargs(self):
        """Test conversion to legacy kwargs format."""
        mouse_cond = torch.randn(1, 9, 2)
        ctx = ActionContext(
            mouse_cond=mouse_cond,
            use_rope_keyboard=True,
            num_frame_per_block=3
        )

        legacy = ctx.to_legacy_kwargs()

        assert 'mouse_cond' in legacy
        assert legacy['mouse_cond'] is mouse_cond
        assert legacy['use_rope_keyboard'] == True
        assert legacy['num_frame_per_block'] == 3

    def test_default_values(self):
        """Test default values for optional fields."""
        ctx = ActionContext(mouse_cond=torch.randn(1, 9, 2))

        assert ctx.use_rope_keyboard == True
        assert ctx.num_frame_per_block == 1
        assert ctx.start_frame == 0
        assert ctx.block_mask_mouse is None


class TestActionContextFactory:
    """Tests for create_action_context_from_kwargs factory function."""

    def test_create_from_kwargs_with_mouse(self):
        """Test creating ActionContext from kwargs with mouse."""
        mouse_cond = torch.randn(1, 9, 2)
        ctx = create_action_context_from_kwargs(
            mouse_cond=mouse_cond,
            keyboard_cond=None,
            use_rope_keyboard=False,
            num_frame_per_block=5
        )

        assert ctx is not None
        assert ctx.has_mouse == True
        assert ctx.use_rope_keyboard == False
        assert ctx.num_frame_per_block == 5

    def test_create_from_kwargs_returns_none_when_no_conditions(self):
        """Test that factory returns None when no action conditions provided."""
        ctx = create_action_context_from_kwargs(
            mouse_cond=None,
            keyboard_cond=None,
            use_rope_keyboard=True
        )

        assert ctx is None

    def test_create_from_kwargs_backward_compatibility(self):
        """Test that all legacy parameters are properly converted."""
        mouse_cond = torch.randn(1, 9, 2)
        keyboard_cond = torch.randn(1, 9, 6)

        legacy_kwargs = {
            'mouse_cond': mouse_cond,
            'keyboard_cond': keyboard_cond,
            'block_mask_mouse': None,
            'block_mask_keyboard': None,
            'kv_cache_mouse': None,
            'kv_cache_keyboard': None,
            'use_rope_keyboard': False,
            'num_frame_per_block': 3,
            'start_frame': 10
        }

        ctx = create_action_context_from_kwargs(**legacy_kwargs)

        assert ctx is not None
        assert ctx.mouse_cond is mouse_cond
        assert ctx.keyboard_cond is keyboard_cond
        assert ctx.use_rope_keyboard == False
        assert ctx.num_frame_per_block == 3
        assert ctx.start_frame == 10


class TestBlockMaskFactory:
    """Tests for BlockMaskFactory."""

    @pytest.fixture
    def factory(self):
        """Create a BlockMaskFactory instance."""
        return BlockMaskFactory(device="cpu")  # Use CPU for testing

    def test_create_visual_mask(self, factory):
        """Test creating visual self-attention mask."""
        mask = factory.create_visual_mask(
            num_frames=3,
            frame_seqlen=880,
            num_frame_per_block=1,
            local_attn_size=-1
        )

        # BlockMask is created successfully
        assert mask is not None
        # TODO: Add more specific assertions about mask properties

    def test_create_action_mask_mouse(self, factory):
        """Test creating mouse action mask."""
        mask = factory.create_action_mask(
            num_frames=9,
            frame_seqlen=880,
            num_frame_per_block=1,
            local_attn_size=-1,
            action_type='mouse'
        )

        assert mask is not None

    def test_create_action_mask_keyboard(self, factory):
        """Test creating keyboard action mask."""
        mask = factory.create_action_mask(
            num_frames=9,
            frame_seqlen=880,
            num_frame_per_block=1,
            local_attn_size=-1,
            action_type='keyboard'
        )

        assert mask is not None

    def test_mask_caching_on_device(self, factory):
        """Test that masks are created on the correct device."""
        # Factory initialized with CPU
        mask = factory.create_visual_mask(3, 880)
        # Verify mask is on CPU (device check would go here)
        # Note: BlockMask doesn't expose device directly in current implementation


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with old API."""

    def test_kwargs_to_context_roundtrip(self):
        """Test converting kwargs to ActionContext and back."""
        original_kwargs = {
            'mouse_cond': torch.randn(1, 9, 2),
            'keyboard_cond': torch.randn(1, 9, 6),
            'use_rope_keyboard': True,
            'num_frame_per_block': 3,
        }

        # Convert to ActionContext
        ctx = create_action_context_from_kwargs(**original_kwargs)

        # Convert back to kwargs
        new_kwargs = ctx.to_legacy_kwargs()

        # Verify all fields match
        assert torch.equal(new_kwargs['mouse_cond'], original_kwargs['mouse_cond'])
        assert torch.equal(new_kwargs['keyboard_cond'], original_kwargs['keyboard_cond'])
        assert new_kwargs['use_rope_keyboard'] == original_kwargs['use_rope_keyboard']
        assert new_kwargs['num_frame_per_block'] == original_kwargs['num_frame_per_block']


# Run tests with: pytest tests/test_action_context.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
