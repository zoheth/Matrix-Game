"""
Refactored VAE Decoder Block with improved architecture.

This module provides:
- Proper separation of batch and streaming inference modes
- Encapsulated cache management
- Clear documentation and type hints
- Eliminated code duplication
- Fixed known bugs (mutable defaults, magic values, etc.)
"""

from typing import List, Optional, Tuple, Union
from einops import rearrange
import torch
import torch.nn as nn

from wan.modules.vae import AttentionBlock, CausalConv3d, RMS_norm, ResidualBlock, Upsample


# Constants
CACHE_FRAMES = 2  # Number of frames to cache for causal convolution
FIRST_CHUNK_MARKER = 'FIRST_CHUNK'  # Marker for first chunk in streaming mode


class DecoderCache:
    """
    Encapsulated cache manager for streaming inference.

    This class manages the feature caches for all layers in the decoder,
    eliminating the need for manual index tracking with feat_idx=[0].

    Attributes:
        caches: List of cached tensors for each layer
        _index: Current position in the cache list
    """

    def __init__(self, num_layers: int = 32):
        """
        Initialize cache with given number of layers.

        Args:
            num_layers: Number of layers that need caching
        """
        self.caches: List[Optional[Union[torch.Tensor, str]]] = [None] * num_layers
        self._index = 0

    def get_and_advance(self) -> Tuple[Optional[Union[torch.Tensor, str]], int]:
        """
        Get current cache and advance index.

        Returns:
            Tuple of (cache_value, cache_index)
        """
        cache = self.caches[self._index]
        idx = self._index
        self._index += 1
        return cache, idx

    def set(self, idx: int, value: Union[torch.Tensor, str]) -> None:
        """
        Set cache at given index.

        Args:
            idx: Index to set cache at
            value: Cache value (tensor or marker string)
        """
        self.caches[idx] = value

    def reset_index(self) -> None:
        """Reset index to 0 for next forward pass."""
        self._index = 0

    @classmethod
    def from_list(cls, cache_list: List) -> 'DecoderCache':
        """
        Create DecoderCache from existing list format.

        Args:
            cache_list: List in legacy format

        Returns:
            DecoderCache instance
        """
        cache = cls(len(cache_list))
        cache.caches = cache_list
        return cache

    def to_list(self) -> List:
        """Convert to legacy list format for backward compatibility."""
        return self.caches


def _prepare_cache_input(
    x: torch.Tensor,
    cache: Optional[Union[torch.Tensor, str]],
    cache_frames: int = CACHE_FRAMES
) -> torch.Tensor:
    """
    Prepare cache input by concatenating with current tensor.

    Args:
        x: Current tensor (B, C, T, H, W)
        cache: Cached tensor from previous chunk or marker string
        cache_frames: Number of frames to extract from x for caching

    Returns:
        Tensor to use as cache input for causal convolution
    """
    cache_x = x[:, :, -cache_frames:, :, :].clone()

    if cache_x.shape[2] < cache_frames and cache is not None:
        if isinstance(cache, str):  # First chunk marker
            # Pad with zeros
            padding = torch.zeros_like(cache_x).to(cache_x.device)
            cache_x = torch.cat([padding, cache_x], dim=2)
        else:
            # Use cached frame from previous chunk
            cache_x = torch.cat([
                cache[:, :, -1:, :, :].to(cache_x.device),
                cache_x
            ], dim=2)

    return cache_x


class Resample(nn.Module):
    """
    Resampling module supporting 2D and 3D up/downsampling.

    This module handles spatial and temporal resampling with proper caching
    for streaming inference.

    Args:
        dim: Input channel dimension
        mode: Resampling mode - one of ['none', 'upsample2d', 'upsample3d',
              'downsample2d', 'downsample3d']
    """

    VALID_MODES = ('none', 'upsample2d', 'upsample3d', 'downsample2d', 'downsample3d')

    def __init__(self, dim: int, mode: str):
        super().__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode}")

        self.dim = dim
        self.mode = mode
        self.cache_frames = CACHE_FRAMES

        # Build layers based on mode
        self._build_layers()

    def _build_layers(self) -> None:
        """Build resampling layers based on mode."""
        if self.mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest'),
                nn.Conv2d(self.dim, self.dim // 2, 3, padding=1)
            )

        elif self.mode == 'upsample3d':
            # Spatial upsample
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest'),
                nn.Conv2d(self.dim, self.dim // 2, 3, padding=1)
            )
            # Temporal upsample via causal conv
            # Output dim * 2 to split into 2 time steps
            self.time_conv = CausalConv3d(
                self.dim, self.dim * 2, (3, 1, 1), padding=(1, 0, 0)
            )

        elif self.mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(self.dim, self.dim, 3, stride=(2, 2))
            )

        elif self.mode == 'downsample3d':
            # Spatial downsample
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(self.dim, self.dim, 3, stride=(2, 2))
            )
            # Temporal downsample via causal conv
            self.time_conv = CausalConv3d(
                self.dim, self.dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )

        else:  # 'none'
            self.resample = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[Union[List, DecoderCache]] = None,
        feat_idx: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional caching for streaming inference.

        Args:
            x: Input tensor (B, C, T, H, W)
            feat_cache: Feature cache (list or DecoderCache)
            feat_idx: Feature index (for legacy list-based caching)

        Returns:
            Resampled tensor (B, C', T', H', W')
        """
        b, c, t, h, w = x.size()

        # Handle 3D upsampling with temporal dimension
        if self.mode == 'upsample3d':
            x = self._upsample3d_temporal(x, feat_cache, feat_idx)

        # Apply spatial resampling (handles time dimension by flattening)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        # Handle 3D downsampling with temporal dimension
        if self.mode == 'downsample3d':
            x = self._downsample3d_temporal(x, feat_cache, feat_idx)

        return x

    def _upsample3d_temporal(
        self,
        x: torch.Tensor,
        feat_cache: Optional[Union[List, DecoderCache]],
        feat_idx: Optional[List[int]]
    ) -> torch.Tensor:
        """
        Apply temporal upsampling for upsample3d mode.

        This uses a causal 3D convolution with output channels = input_dim * 2,
        then reshapes to double the temporal dimension.
        """
        if feat_cache is not None:
            idx = feat_idx[0] if feat_idx else 0
            cache = feat_cache[idx] if isinstance(feat_cache, list) else feat_cache.caches[idx]

            if cache is None:
                # First chunk - mark and skip temporal processing
                if isinstance(feat_cache, list):
                    feat_cache[idx] = FIRST_CHUNK_MARKER
                else:
                    feat_cache.set(idx, FIRST_CHUNK_MARKER)

                if feat_idx:
                    feat_idx[0] += 1
                elif isinstance(feat_cache, DecoderCache):
                    feat_cache._index += 1
            else:
                # Prepare cache for causal convolution
                cache_x = _prepare_cache_input(x, cache, CACHE_FRAMES)

                # Apply causal convolution with cache
                if cache == FIRST_CHUNK_MARKER:
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, cache)

                # Update cache
                if isinstance(feat_cache, list):
                    feat_cache[idx] = cache_x
                else:
                    feat_cache.set(idx, cache_x)

                if feat_idx:
                    feat_idx[0] += 1
                elif isinstance(feat_cache, DecoderCache):
                    feat_cache._index += 1

                # Reshape to double temporal dimension
                # Output has dim*2 channels, reshape to split into 2 time steps
                b, c_out, t, h, w = x.shape
                c = c_out // 2
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(b, c, t * 2, h, w)

        return x

    def _downsample3d_temporal(
        self,
        x: torch.Tensor,
        feat_cache: Optional[Union[List, DecoderCache]],
        feat_idx: Optional[List[int]]
    ) -> torch.Tensor:
        """
        Apply temporal downsampling for downsample3d mode.

        This uses a causal 3D convolution with stride=2 in time dimension.
        """
        if feat_cache is not None:
            idx = feat_idx[0] if feat_idx else 0
            cache = feat_cache[idx] if isinstance(feat_cache, list) else feat_cache.caches[idx]

            if cache is None:
                # First chunk - store current output as cache
                cache_x = x.clone()
                if isinstance(feat_cache, list):
                    feat_cache[idx] = cache_x
                else:
                    feat_cache.set(idx, cache_x)

                if feat_idx:
                    feat_idx[0] += 1
                elif isinstance(feat_cache, DecoderCache):
                    feat_cache._index += 1
            else:
                # Cache last frame for next iteration
                cache_x = x[:, :, -1:, :, :].clone()

                # Concatenate with previous cache and apply temporal downsampling
                x = self.time_conv(
                    torch.cat([cache[:, :, -1:, :, :], x], dim=2)
                )

                # Update cache
                if isinstance(feat_cache, list):
                    feat_cache[idx] = cache_x
                else:
                    feat_cache.set(idx, cache_x)

                if feat_idx:
                    feat_idx[0] += 1
                elif isinstance(feat_cache, DecoderCache):
                    feat_cache._index += 1

        return x


class VAEDecoderWrapper(nn.Module):
    """
    Wrapper for VAE decoder that handles preprocessing and mode selection.

    This wrapper:
    - Handles input/output tensor transpositions
    - Applies normalization with learnable mean/std
    - Supports both batch and streaming inference modes

    The forward method automatically detects streaming mode when feat_cache is provided.
    """

    def __init__(self):
        super().__init__()
        self.decoder = VAEDecoder3d()

        # Normalization parameters (from pretrained model)
        # Note: These are not registered as buffers to maintain compatibility
        # with pretrained weights that don't include these parameters
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.z_dim = 16

        # Initial convolution
        self.conv2 = CausalConv3d(self.z_dim, self.z_dim, 1)

    def forward(
        self,
        z: torch.Tensor,
        *feat_cache: List[torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Forward pass supporting both batch and streaming modes.

        Args:
            z: Latent tensor in format (B, T, C, H, W)
            *feat_cache: Optional feature caches for streaming mode

        Returns:
            If feat_cache provided (streaming mode): (decoded_video, updated_cache)
            Otherwise (batch mode): decoded_video

            decoded_video shape: (B, T, 3, H', W')
        """
        with torch.profiler.record_function("VAE_Decode_Preprocessing"):
            # Convert from (B, T, C, H, W) to (B, C, T, H, W)
            z = z.permute(0, 2, 1, 3, 4)

            feat_cache_list = list(feat_cache) if feat_cache else None

            # Apply normalization: z = z * std + mean
            device, dtype = z.device, z.dtype
            mean = self.mean.to(device=device, dtype=dtype)
            std = self.std.to(device=device, dtype=dtype)
            z = z * std.view(1, self.z_dim, 1, 1, 1) + mean.view(1, self.z_dim, 1, 1, 1)

        with torch.profiler.record_function("VAE_Conv2_Input"):
            x = self.conv2(z)

        # Detect mode based on cache presence
        if feat_cache_list is not None:
            # Streaming mode: process frame by frame
            out, feat_cache_list = self._forward_streaming(x, feat_cache_list)
        else:
            # Batch mode: process entire sequence at once
            out = self._forward_batch(x)

        with torch.profiler.record_function("VAE_Decode_Postprocessing"):
            out = out.float().clamp_(-1, 1)
            # Convert from (B, C, T, H, W) to (B, T, C, H, W)
            out = out.permute(0, 2, 1, 3, 4)

        if feat_cache_list is not None:
            return out, feat_cache_list
        else:
            return out

    def _forward_streaming(
        self,
        x: torch.Tensor,
        feat_cache: List
    ) -> Tuple[torch.Tensor, List]:
        """
        Streaming inference: process frame by frame with caching.

        This mode is necessary for real-time/online inference where frames
        arrive sequentially and must be processed immediately.

        Args:
            x: Input tensor (B, C, T, H, W)
            feat_cache: List of cached features from previous chunks

        Returns:
            Tuple of (output_tensor, updated_cache)
        """
        num_frames = x.shape[2]
        outputs = []

        for i in range(num_frames):
            with torch.profiler.record_function(f"VAE_Decode_Frame_{i}"):
                frame = x[:, :, i:i + 1, :, :]
                out_frame, feat_cache = self.decoder(frame, feat_cache=feat_cache)
                outputs.append(out_frame)

        # Concatenate all frames
        out = torch.cat(outputs, dim=2)
        return out, feat_cache

    def _forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Batch inference: process entire sequence at once.

        This mode is more efficient for offline processing where the entire
        sequence is available. It leverages GPU parallelism across the time
        dimension.

        Args:
            x: Input tensor (B, C, T, H, W)

        Returns:
            Output tensor (B, C, T, H', W')
        """
        with torch.profiler.record_function("VAE_Decode_Batch"):
            out, _ = self.decoder(x, feat_cache=None)
        return out


class VAEDecoder3d(nn.Module):
    """
    3D VAE Decoder with support for both batch and streaming inference.

    This decoder uses:
    - Causal 3D convolutions for temporal consistency
    - Residual blocks with attention at specified scales
    - Progressive upsampling with 2D and 3D operations

    Args:
        dim: Base channel dimension
        z_dim: Latent dimension
        dim_mult: Channel multipliers for each level
        num_res_blocks: Number of residual blocks per level
        attn_scales: Scales at which to apply attention
        temporal_upsample: Whether to upsample temporally at each level
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: List[int] = None,
        num_res_blocks: int = 2,
        attn_scales: List[float] = None,
        temporal_upsample: List[bool] = None,
        dropout: float = 0.0
    ):
        super().__init__()

        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temporal_upsample is None:
            temporal_upsample = [True, True, False]

        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_upsample = temporal_upsample
        self.cache_frames = CACHE_FRAMES

        # Calculate dimensions at each level
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # Initial convolution
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # Middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout)
        )

        # Build upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # Adjust input dimension for concatenated features
            if i in (1, 2, 3):
                in_dim = in_dim // 2

            # Add residual blocks with optional attention
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # Add upsample block (except at last level)
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temporal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0

        self.upsamples = nn.Sequential(*upsamples)

        # Output head
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1)
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List] = None
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Forward pass with optional caching.

        Args:
            x: Input tensor (B, C, T, H, W)
            feat_cache: Optional feature cache for streaming mode

        Returns:
            Tuple of (output, updated_cache)
        """
        feat_idx = [0] if feat_cache is not None else None

        # Initial convolution with caching
        with torch.profiler.record_function("VAE_Dec_Conv1"):
            if feat_cache is not None:
                idx = feat_idx[0]
                cache_x = _prepare_cache_input(x, feat_cache[idx], CACHE_FRAMES)
                x = self.conv1(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = self.conv1(x)

        # Middle blocks
        with torch.profiler.record_function("VAE_Dec_Middle_Blocks"):
            for layer in self.middle:
                if isinstance(layer, ResidualBlock) and feat_cache is not None:
                    x = layer(x, feat_cache, feat_idx)
                else:
                    x = layer(x)

        # Upsample blocks
        with torch.profiler.record_function("VAE_Dec_Upsample_Blocks"):
            for layer in self.upsamples:
                x = layer(x, feat_cache, feat_idx)

        # Output head with caching
        with torch.profiler.record_function("VAE_Dec_Head"):
            for layer in self.head:
                if isinstance(layer, CausalConv3d) and feat_cache is not None:
                    idx = feat_idx[0]
                    cache_x = _prepare_cache_input(x, feat_cache[idx], CACHE_FRAMES)
                    x = layer(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                else:
                    x = layer(x)

        return x, feat_cache
