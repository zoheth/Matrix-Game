"""
Pipeline module for causal video generation.

This module provides both refactored and legacy interfaces for inference.
"""

# Legacy imports (for backward compatibility)
from pipeline.causal_inference import (
    CausalInferencePipeline as LegacyCausalInferencePipeline,
    CausalInferenceStreamingPipeline as LegacyCausalInferenceStreamingPipeline
)

# New refactored imports
from pipeline.batch_pipeline import BatchCausalInferencePipeline
from pipeline.streaming_pipeline import StreamingCausalInferencePipeline, default_cli_continuation
from pipeline.config import PipelineConfig, ModelConfig, CacheConfig, VAEConfig, InferenceConfig
from pipeline.action_strategies import (
    ActionStrategy,
    ActionDict,
    CLIActionStrategy,
    PrerecordedActionStrategy,
    CallbackActionStrategy
)
from pipeline.cache_manager import CacheManager
from pipeline.condition_processor import ConditionProcessor


# Backward compatibility aliases
# These allow old code to continue working without modification
CausalInferencePipeline = LegacyCausalInferencePipeline
CausalInferenceStreamingPipeline = LegacyCausalInferenceStreamingPipeline


__all__ = [
    # Legacy (backward compatible)
    'CausalInferencePipeline',
    'CausalInferenceStreamingPipeline',
    'LegacyCausalInferencePipeline',
    'LegacyCausalInferenceStreamingPipeline',

    # New refactored
    'BatchCausalInferencePipeline',
    'StreamingCausalInferencePipeline',
    'default_cli_continuation',

    # Configuration
    'PipelineConfig',
    'ModelConfig',
    'CacheConfig',
    'VAEConfig',
    'InferenceConfig',

    # Action strategies
    'ActionStrategy',
    'ActionDict',
    'CLIActionStrategy',
    'PrerecordedActionStrategy',
    'CallbackActionStrategy',

    # Utilities
    'CacheManager',
    'ConditionProcessor',
]
