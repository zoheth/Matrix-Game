"""
Model initialization and management
"""
import os
import torch
from pathlib import Path
from omegaconf import OmegaConf

from pipeline import CausalInferenceStreamingPipeline
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.wan_wrapper import WanDiffusionWrapper
from safetensors.torch import load_file
from server.vae_cache import load_vae_decoder_with_cache


class ModelManager:
    """Manages loading and initialization of all models"""

    def __init__(
        self,
        config_path: str,
        pretrained_model_path: str,
        checkpoint_path: str = "",
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
        vae_compile_mode: str = "auto"
    ):
        """
        Initialize model manager

        Args:
            config_path: Path to inference config YAML
            pretrained_model_path: Path to pretrained models
            checkpoint_path: Optional path to checkpoint
            device: PyTorch device (defaults to CUDA)
            dtype: PyTorch dtype for main model
            vae_compile_mode: VAE compilation mode ("auto", "force", or "none")
        """
        self.config_path = config_path
        self.pretrained_model_path = pretrained_model_path
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device("cuda")
        self.dtype = dtype
        self.vae_compile_mode = vae_compile_mode

        self.config = None
        self.pipeline = None
        self.vae = None

    def load_models(self):
        """Load all models (generator, VAE, decoder)"""
        print("Loading configuration...")
        self.config = OmegaConf.load(self.config_path)

        print("Initializing models...")
        self._load_pipeline()
        self._load_vae()
        print("✅ Models initialized successfully")

    def _load_pipeline(self):
        """Load the main inference pipeline with generator and decoder"""
        # Initialize generator
        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}),
            is_causal=True
        )

        # Initialize VAE decoder with caching support (aligned with inference.py)
        decoder = self._load_vae_decoder()

        # Create pipeline
        self.pipeline = CausalInferenceStreamingPipeline(
            self.config,
            generator=generator,
            vae_decoder=decoder
        )

        # Load checkpoint if provided
        if self.checkpoint_path:
            print(f"Loading checkpoint from {self.checkpoint_path}")
            state_dict = load_file(self.checkpoint_path)
            self.pipeline.generator.load_state_dict(state_dict)

        # Move to device
        self.pipeline = self.pipeline.to(device=self.device, dtype=self.dtype)
        self.pipeline.vae_decoder.to(torch.float16)

    def _load_vae_decoder(self) -> VAEDecoderWrapper:
        """
        Load VAE decoder with optional compilation and caching.
        Uses the same caching logic as inference.py for consistency.
        """
        decoder = load_vae_decoder_with_cache(
            pretrained_model_path=self.pretrained_model_path,
            device=self.device,
            compile_mode=self.vae_compile_mode
        )
        return decoder

    def _load_vae(self):
        """Load VAE encoder"""
        self.vae = get_wanx_vae_wrapper(
            self.pretrained_model_path,
            torch.float16
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae = self.vae.to(self.device, self.dtype)

    def get_pipeline(self) -> CausalInferenceStreamingPipeline:
        """Get the inference pipeline"""
        if self.pipeline is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.pipeline

    def get_vae(self):
        """Get the VAE encoder"""
        if self.vae is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.vae

    def get_config(self):
        """Get the loaded configuration"""
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_models() first.")
        return self.config
