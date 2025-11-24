"""
CUDA Graph-compatible Inference Module for CausalWanModel.

This module provides infrastructure for capturing and replaying CUDA Graphs
during inference, eliminating CPU overhead for repeated forward passes.

Key requirements for CUDA Graph compatibility:
1. Static tensor shapes - all tensors must have fixed shapes
2. No dynamic memory allocation - all memory pre-allocated
3. No CPU-GPU synchronization - no .item() calls in the captured region
4. No control flow dependent on tensor values
5. Pre-computed masks and frequencies

Usage:
    ```python
    # Initialize
    graph_runner = CUDAGraphInferenceRunner(model, config)

    # Warmup and capture
    graph_runner.warmup_and_capture(sample_inputs)

    # Inference (uses captured graph)
    output = graph_runner.run(new_inputs)
    ```
"""

from typing import Dict, List, Optional, Tuple, Callable, Any
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class CUDAGraphConfig:
    """Configuration for CUDA Graph capture."""
    # Input shapes (must be static)
    batch_size: int = 1
    num_frames_per_block: int = 1
    height: int = 22  # After patchification
    width: int = 40   # After patchification
    hidden_dim: int = 1536
    num_heads: int = 12
    head_dim: int = 128

    # Memory pool
    max_graph_pool_size: int = 512 * 1024 * 1024  # 512MB

    # Warmup
    warmup_iterations: int = 3

    # Stream
    capture_stream: Optional[torch.cuda.Stream] = None


class StaticTensorPool:
    """
    Pre-allocated pool of static tensors for CUDA Graph capture.

    All tensors used within the captured region must be pre-allocated
    and have static shapes. This class manages the allocation and
    provides tensors on demand.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self._tensors: Dict[str, torch.Tensor] = {}

    def allocate(self, name: str, shape: Tuple[int, ...], dtype: torch.dtype = None) -> torch.Tensor:
        """
        Allocate or retrieve a static tensor.

        Args:
            name: Unique identifier for this tensor
            shape: Tensor shape
            dtype: Data type (default: self.dtype)

        Returns:
            Pre-allocated tensor
        """
        if name not in self._tensors:
            self._tensors[name] = torch.zeros(
                shape,
                dtype=dtype or self.dtype,
                device=self.device
            )
        return self._tensors[name]

    def get(self, name: str) -> torch.Tensor:
        """Get a previously allocated tensor."""
        if name not in self._tensors:
            raise KeyError(f"Tensor '{name}' not allocated. Call allocate() first.")
        return self._tensors[name]

    def copy_to(self, name: str, data: torch.Tensor) -> None:
        """Copy data to a static tensor (in-place)."""
        self._tensors[name].copy_(data)

    def clear(self) -> None:
        """Clear all allocated tensors."""
        self._tensors.clear()


class CUDAGraphInferenceRunner:
    """
    Manages CUDA Graph capture and replay for model inference.

    This class handles:
    1. Pre-allocation of all required tensors
    2. Warmup iterations to stabilize memory layout
    3. Graph capture with proper stream management
    4. Graph replay with input updates
    """

    def __init__(
        self,
        model: nn.Module,
        config: CUDAGraphConfig,
        forward_fn: Optional[Callable] = None,
    ):
        """
        Initialize the CUDA Graph runner.

        Args:
            model: The model to capture
            config: Graph configuration
            forward_fn: Custom forward function (default: model.__call__)
        """
        self.model = model
        self.config = config
        self.forward_fn = forward_fn or model
        self.device = next(model.parameters()).device

        # Static tensor pool
        self.tensor_pool = StaticTensorPool(self.device)

        # Graph state
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.stream: torch.cuda.Stream = config.capture_stream or torch.cuda.Stream()
        self.is_captured = False

        # Static I/O tensors (set during capture)
        self._static_inputs: Dict[str, torch.Tensor] = {}
        self._static_outputs: Dict[str, torch.Tensor] = {}

    def _allocate_static_tensors(self, sample_inputs: Dict[str, Any]) -> None:
        """
        Pre-allocate all static tensors based on sample inputs.

        Args:
            sample_inputs: Sample input dictionary to infer shapes
        """
        for name, value in sample_inputs.items():
            if isinstance(value, torch.Tensor):
                self._static_inputs[name] = self.tensor_pool.allocate(
                    f"input_{name}", value.shape, value.dtype
                )
                # Copy initial values
                self._static_inputs[name].copy_(value)

    def _prepare_static_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare inputs for graph execution by using static tensors.

        Args:
            inputs: Input dictionary

        Returns:
            Modified inputs with static tensors
        """
        result = {}
        for name, value in inputs.items():
            if name in self._static_inputs:
                result[name] = self._static_inputs[name]
            else:
                result[name] = value
        return result

    def warmup(self, sample_inputs: Dict[str, Any]) -> None:
        """
        Run warmup iterations to stabilize memory layout.

        This must be called before capture() to ensure consistent
        memory allocations during graph capture.

        Args:
            sample_inputs: Sample inputs for warmup
        """
        # Allocate static tensors
        self._allocate_static_tensors(sample_inputs)

        # Warmup iterations
        static_inputs = self._prepare_static_inputs(sample_inputs)

        torch.cuda.synchronize()
        with torch.cuda.stream(self.stream):
            for _ in range(self.config.warmup_iterations):
                _ = self.forward_fn(**static_inputs)
        torch.cuda.synchronize()

    def capture(self, sample_inputs: Dict[str, Any]) -> None:
        """
        Capture CUDA Graph for the model.

        Args:
            sample_inputs: Sample inputs (used to determine static tensor shapes)
        """
        if self.is_captured:
            print("[WARNING] Graph already captured, skipping")
            return

        # Prepare static inputs
        static_inputs = self._prepare_static_inputs(sample_inputs)

        # Create graph
        self.graph = torch.cuda.CUDAGraph()

        # Capture
        torch.cuda.synchronize()
        with torch.cuda.stream(self.stream):
            with torch.cuda.graph(self.graph, stream=self.stream):
                outputs = self.forward_fn(**static_inputs)

        # Store output references
        if isinstance(outputs, torch.Tensor):
            self._static_outputs = {"output": outputs}
        elif isinstance(outputs, tuple):
            self._static_outputs = {f"output_{i}": o for i, o in enumerate(outputs)}
        elif isinstance(outputs, dict):
            self._static_outputs = outputs
        else:
            self._static_outputs = {"output": outputs}

        self.is_captured = True
        print("[INFO] CUDA Graph captured successfully")

    def warmup_and_capture(self, sample_inputs: Dict[str, Any]) -> None:
        """
        Convenience method to warmup and capture in one call.

        Args:
            sample_inputs: Sample inputs for the model
        """
        self.warmup(sample_inputs)
        self.capture(sample_inputs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Run inference using the captured graph.

        Args:
            inputs: New input values (copied to static tensors)

        Returns:
            Output dictionary
        """
        if not self.is_captured:
            raise RuntimeError("Graph not captured. Call capture() first.")

        # Update static input tensors
        for name, value in inputs.items():
            if name in self._static_inputs and isinstance(value, torch.Tensor):
                self._static_inputs[name].copy_(value)

        # Replay graph
        self.graph.replay()

        return self._static_outputs

    def run_without_graph(self, inputs: Dict[str, Any]) -> Any:
        """
        Run inference without using the graph (for debugging/comparison).

        Args:
            inputs: Input dictionary

        Returns:
            Model outputs
        """
        return self.forward_fn(**inputs)


class CUDAGraphCausalInference:
    """
    CUDA Graph-enabled causal inference for video generation.

    This class wraps the full inference loop with CUDA Graph support,
    capturing the denoising step for maximum efficiency.
    """

    def __init__(
        self,
        generator: nn.Module,
        num_layers: int = 30,
        frame_seq_len: int = 880,
        local_attn_size: int = -1,
        device: torch.device = None,
    ):
        """
        Initialize CUDA Graph causal inference.

        Args:
            generator: The diffusion generator model
            num_layers: Number of transformer layers
            frame_seq_len: Tokens per frame
            local_attn_size: Sliding window size
            device: Target device
        """
        self.generator = generator
        self.num_layers = num_layers
        self.frame_seq_len = frame_seq_len
        self.local_attn_size = local_attn_size
        self.device = device or torch.device("cuda")

        # Graph runners for different step types
        self._denoising_graph: Optional[CUDAGraphInferenceRunner] = None
        self._context_update_graph: Optional[CUDAGraphInferenceRunner] = None

        # Precomputed resources
        self._static_timesteps: Dict[int, torch.Tensor] = {}

    def _create_denoising_forward(self, kv_cache, kv_cache_mouse, kv_cache_keyboard, crossattn_cache):
        """Create a forward function for the denoising step."""
        def forward(
            noisy_image_or_video: torch.Tensor,
            cond_concat: torch.Tensor,
            visual_context: torch.Tensor,
            timestep: torch.Tensor,
            current_start: int,
            mouse_cond: Optional[torch.Tensor] = None,
            keyboard_cond: Optional[torch.Tensor] = None,
        ):
            conditional_dict = {
                "cond_concat": cond_concat,
                "visual_context": visual_context,
            }
            if mouse_cond is not None:
                conditional_dict["mouse_cond"] = mouse_cond
            if keyboard_cond is not None:
                conditional_dict["keyboard_cond"] = keyboard_cond

            return self.generator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=kv_cache,
                kv_cache_mouse=kv_cache_mouse,
                kv_cache_keyboard=kv_cache_keyboard,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
            )

        return forward

    def prepare_graphs(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        sample_inputs: Dict[str, Any],
        kv_cache: List[Dict],
        kv_cache_mouse: List[Dict],
        kv_cache_keyboard: List[Dict],
        crossattn_cache: List[Dict],
    ) -> None:
        """
        Prepare CUDA Graphs for inference.

        This should be called once before the main inference loop.

        Args:
            batch_size: Batch size
            num_frames: Frames per block
            height: Latent height
            width: Latent width
            dtype: Data type
            sample_inputs: Sample inputs for shape inference
            kv_cache: KV cache
            kv_cache_mouse: Mouse KV cache
            kv_cache_keyboard: Keyboard KV cache
            crossattn_cache: Cross-attention cache
        """
        config = CUDAGraphConfig(
            batch_size=batch_size,
            num_frames_per_block=num_frames,
            height=height // 2,  # After patchification
            width=width // 2,
        )

        # Create forward function with bound caches
        forward_fn = self._create_denoising_forward(
            kv_cache, kv_cache_mouse, kv_cache_keyboard, crossattn_cache
        )

        # Create graph runner
        self._denoising_graph = CUDAGraphInferenceRunner(
            self.generator,
            config,
            forward_fn=forward_fn,
        )

        # Warmup and capture
        self._denoising_graph.warmup_and_capture(sample_inputs)

    def run_denoising_step(self, inputs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run a single denoising step using the captured graph.

        Args:
            inputs: Input dictionary

        Returns:
            Tuple of (noise_pred, denoised_pred)
        """
        if self._denoising_graph is None:
            raise RuntimeError("Graphs not prepared. Call prepare_graphs() first.")

        outputs = self._denoising_graph.run(inputs)
        return outputs.get("output_0"), outputs.get("output_1")


def check_cuda_graph_compatibility(model: nn.Module) -> List[str]:
    """
    Check if a model is compatible with CUDA Graph capture.

    Returns a list of potential issues that may prevent graph capture.

    Args:
        model: The model to check

    Returns:
        List of compatibility issues (empty if compatible)
    """
    issues = []

    # Check for dynamic shapes
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if param.grad.is_sparse:
                issues.append(f"Sparse gradients in {name}")

    # Check for problematic operations in forward hooks
    for name, module in model.named_modules():
        # Check for operations that may cause issues
        if hasattr(module, 'forward'):
            import inspect
            source = inspect.getsource(module.forward) if hasattr(module.forward, '__code__') else ""

            if '.item()' in source:
                issues.append(f"Module {name} may use .item() which causes CPU-GPU sync")

            if 'torch.tensor(' in source and 'device=' not in source:
                issues.append(f"Module {name} may create tensors without explicit device")

    return issues
