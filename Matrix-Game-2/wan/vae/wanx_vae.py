import os
import torch
import torch.nn as nn
from pathlib import Path
from .wanx_vae_src import WanVAE, CLIPModel


class WanVAEEncoder(nn.Module):
    """WAN VAE Encoder with integrated CLIP visual features.

    Combines VAE encoding/decoding with CLIP visual feature extraction
    for video generation tasks.
    """

    def __init__(self, vae, clip):
        super().__init__()
        self.vae = vae
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.clip = clip
        if clip is not None:
            self.clip.requires_grad_(False)
            self.clip.eval()

    def encode(self, x, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        """Encode input frames to latent space.

        Args:
            x: Input tensor [B, C, T, H, W]
            device: Target device
            tiled: Whether to use tiled encoding
            tile_size: Size of tiles for tiled encoding
            tile_stride: Stride for tiled encoding

        Returns:
            Encoded latent tensor
        """
        return self.vae.encode(x, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

    def extract_clip_features(self, x):
        """Extract CLIP visual features from input.

        Args:
            x: Input tensor

        Returns:
            CLIP feature tensor
        """
        return self.clip(x)

    def decode(self, latents, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        """Decode latents back to pixel space.

        Args:
            latents: Latent tensor
            device: Target device
            tiled: Whether to use tiled decoding
            tile_size: Size of tiles for tiled decoding
            tile_stride: Stride for tiled decoding

        Returns:
            Decoded video tensor
        """
        return self.vae.decode(latents, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

    def to(self, device, dtype):
        """Move encoder and CLIP to specified device and dtype.

        Args:
            device: Target device
            dtype: Target data type

        Returns:
            self
        """
        self.vae = self.vae.to(device, dtype)

        if self.clip is not None:
            self.clip = self.clip.to(device, dtype)

        return self


def create_wan_encoder(model_path, device, dtype):
    """Create and initialize a WAN VAE encoder with CLIP.

    This encoder combines VAE encoding and CLIP visual feature extraction,
    ready for inference with all models loaded and moved to the target device.

    Args:
        model_path: Path to pretrained model directory
        device: Device to place models on (e.g., 'cuda', 'cpu')
        dtype: Data type for model weights (e.g., torch.float16)

    Returns:
        WanVAEEncoder: Fully initialized encoder ready for inference
    """
    vae = WanVAE(pretrained_path=os.path.join(model_path, "Wan2.1_VAE.pth")).to(dtype)
    clip = CLIPModel(
        checkpoint_path=os.path.join(model_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        tokenizer_path=os.path.join(model_path, 'xlm-roberta-large')
    )
    encoder = WanVAEEncoder(vae, clip)
    return encoder.to(device, dtype)
