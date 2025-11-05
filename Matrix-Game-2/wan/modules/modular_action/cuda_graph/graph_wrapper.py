"""
Base CUDA Graph wrapper with shape-based caching strategy
"""

from typing import Dict, Tuple, Optional, Callable, Any
import torch
from dataclasses import dataclass


@dataclass
class GraphKey:
    """Key for identifying a specific CUDA Graph configuration"""
    batch_size: int
    temporal_shape: int
    spatial_h: int
    spatial_w: int
    is_causal: bool
    num_frame_per_block: int

    def __hash__(self):
        return hash((
            self.batch_size,
            self.temporal_shape,
            self.spatial_h,
            self.spatial_w,
            self.is_causal,
            self.num_frame_per_block,
        ))

    def __eq__(self, other):
        if not isinstance(other, GraphKey):
            return False
        return (
            self.batch_size == other.batch_size
            and self.temporal_shape == other.temporal_shape
            and self.spatial_h == other.spatial_h
            and self.spatial_w == other.spatial_w
            and self.is_causal == other.is_causal
            and self.num_frame_per_block == other.num_frame_per_block
        )


@dataclass
class CachedGraph:
    """Cached CUDA Graph with static input/output tensors"""
    graph: torch.cuda.CUDAGraph
    static_inputs: Dict[str, torch.Tensor]
    static_outputs: Dict[str, torch.Tensor]
    key: GraphKey


class CUDAGraphWrapper:
    """
    Base class for CUDA Graph wrappers with automatic caching strategy.

    Features:
    - Shape-based graph caching
    - Automatic fallback to eager mode for new shapes
    - Optional graph capacity limit (LRU eviction)
    """

    def __init__(
        self,
        max_cached_graphs: int = 10,
        enable_cuda_graph: bool = True,
        num_warmup_iters: int = 3,
    ):
        """
        Args:
            max_cached_graphs: Maximum number of graphs to cache (-1 for unlimited)
            enable_cuda_graph: Global switch for CUDA Graph (False = always eager)
            num_warmup_iters: Number of warmup iterations before capture
        """
        self.max_cached_graphs = max_cached_graphs
        self.enable_cuda_graph = enable_cuda_graph
        self.num_warmup_iters = num_warmup_iters

        # Cache: GraphKey -> CachedGraph
        self.graph_cache: Dict[GraphKey, CachedGraph] = {}

        # Statistics
        self.stats = {
            'graph_hits': 0,
            'graph_misses': 0,
            'eager_executions': 0,
        }

    def _create_graph_key(self, **kwargs) -> GraphKey:
        """Create a GraphKey from forward() arguments. Override in subclass."""
        raise NotImplementedError("Subclass must implement _create_graph_key()")

    def _create_static_inputs(self, key: GraphKey) -> Dict[str, torch.Tensor]:
        """Allocate static input tensors for a given shape. Override in subclass."""
        raise NotImplementedError("Subclass must implement _create_static_inputs()")

    def _create_static_outputs(self, key: GraphKey) -> Dict[str, torch.Tensor]:
        """Allocate static output tensors for a given shape. Override in subclass."""
        raise NotImplementedError("Subclass must implement _create_static_outputs()")

    def _copy_inputs_to_static(
        self,
        static_inputs: Dict[str, torch.Tensor],
        dynamic_inputs: Dict[str, Any],
    ):
        """Copy dynamic inputs to static tensors. Override in subclass."""
        raise NotImplementedError("Subclass must implement _copy_inputs_to_static()")

    def _copy_outputs_from_static(
        self,
        static_outputs: Dict[str, torch.Tensor],
    ) -> Any:
        """Copy static outputs to dynamic tensors (usually clone). Override in subclass."""
        raise NotImplementedError("Subclass must implement _copy_outputs_from_static()")

    def _execute_eager(self, **kwargs) -> Any:
        """Execute the forward pass in eager mode. Override in subclass."""
        raise NotImplementedError("Subclass must implement _execute_eager()")

    def _execute_graph_body(
        self,
        static_inputs: Dict[str, torch.Tensor],
        static_outputs: Dict[str, torch.Tensor],
    ):
        """
        Execute the computation body that will be captured in CUDA Graph.
        This is the core forward logic. Override in subclass.
        """
        raise NotImplementedError("Subclass must implement _execute_graph_body()")

    def _warmup(self, key: GraphKey, static_inputs: Dict[str, torch.Tensor], static_outputs: Dict[str, torch.Tensor]):
        """Warmup iterations to stabilize internal states (e.g., FlashInfer workspace)"""
        for _ in range(self.num_warmup_iters):
            self._execute_graph_body(static_inputs, static_outputs)
        torch.cuda.synchronize()

    def _capture_graph(self, key: GraphKey) -> CachedGraph:
        """Capture a new CUDA Graph for the given shape"""
        # Allocate static tensors
        static_inputs = self._create_static_inputs(key)
        static_outputs = self._create_static_outputs(key)

        # Warmup
        self._warmup(key, static_inputs, static_outputs)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._execute_graph_body(static_inputs, static_outputs)

        torch.cuda.synchronize()

        return CachedGraph(
            graph=graph,
            static_inputs=static_inputs,
            static_outputs=static_outputs,
            key=key,
        )

    def _get_or_create_graph(self, key: GraphKey) -> Optional[CachedGraph]:
        """Get cached graph or create a new one"""
        if key in self.graph_cache:
            self.stats['graph_hits'] += 1
            return self.graph_cache[key]

        self.stats['graph_misses'] += 1

        # Check capacity
        if self.max_cached_graphs > 0 and len(self.graph_cache) >= self.max_cached_graphs:
            # Simple eviction: remove first item (could use LRU in production)
            first_key = next(iter(self.graph_cache))
            del self.graph_cache[first_key]
            print(f"[CUDAGraphWrapper] Evicted graph: {first_key}")

        # Capture new graph
        try:
            cached_graph = self._capture_graph(key)
            self.graph_cache[key] = cached_graph
            print(f"[CUDAGraphWrapper] Captured new graph: {key}")
            return cached_graph
        except Exception as e:
            print(f"[CUDAGraphWrapper] Failed to capture graph for {key}: {e}")
            return None

    def forward_with_graph(self, **kwargs) -> Any:
        """
        Main entry point: execute with CUDA Graph if possible, otherwise fallback to eager.

        Subclass should call this method from their forward() implementation.
        """
        if not self.enable_cuda_graph:
            self.stats['eager_executions'] += 1
            return self._execute_eager(**kwargs)

        # Create graph key
        key = self._create_graph_key(**kwargs)

        # Get or create graph
        cached_graph = self._get_or_create_graph(key)

        if cached_graph is None:
            # Fallback to eager
            self.stats['eager_executions'] += 1
            return self._execute_eager(**kwargs)

        # Copy inputs to static tensors
        self._copy_inputs_to_static(cached_graph.static_inputs, kwargs)

        # Replay graph
        cached_graph.graph.replay()

        # Copy outputs from static tensors
        return self._copy_outputs_from_static(cached_graph.static_outputs)

    def get_stats(self) -> Dict[str, int]:
        """Get execution statistics"""
        return self.stats.copy()

    def clear_cache(self):
        """Clear all cached graphs"""
        self.graph_cache.clear()
        print(f"[CUDAGraphWrapper] Cleared {len(self.graph_cache)} cached graphs")
