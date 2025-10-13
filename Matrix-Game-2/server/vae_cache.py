"""
VAE Decoder Compilation Cache Management

This module provides utilities for caching compiled VAE decoder models
to avoid expensive recompilation on subsequent runs.
"""
import os
import torch
from typing import Optional

from demo_utils.vae_block3 import VAEDecoderWrapper


class VAECompilationManager:
    """Manages compilation and caching of VAE decoder models"""

    def __init__(
        self,
        pretrained_model_path: str,
        device: torch.device,
        compile_mode: str = "auto"
    ):
        """
        Initialize VAE compilation manager

        Args:
            pretrained_model_path: Path to pretrained models directory
            device: PyTorch device
            compile_mode: Compilation mode - "auto", "force", or "none"
                - auto: Use cached compiled model if available, compile otherwise
                - force: Always recompile and save
                - none: Skip compilation entirely
        """
        self.pretrained_model_path = pretrained_model_path
        self.device = device
        self.compile_mode = compile_mode
        self.compiled_model_path = os.path.join(
            pretrained_model_path,
            "compiled_vae_decoder.pt"
        )
        self.vae_weights_path = os.path.join(
            pretrained_model_path,
            "Wan2.1_VAE.pth"
        )

    def load_vae_decoder(self) -> VAEDecoderWrapper:
        """
        Load VAE decoder with optional compilation based on mode

        Returns:
            VAE decoder model (compiled or not based on mode)
        """
        # Handle compilation based on mode
        if self.compile_mode == "none":
            return self._load_uncompiled_decoder()
        elif self.compile_mode == "force":
            return self._force_compile_decoder()
        elif self.compile_mode == "auto":
            return self._auto_load_or_compile_decoder()
        else:
            raise ValueError(
                f"Invalid compile_mode: {self.compile_mode}. "
                "Must be 'auto', 'force', or 'none'"
            )

    def _load_uncompiled_decoder(self) -> VAEDecoderWrapper:
        """Load VAE decoder without compilation"""
        print("VAE decoder compilation skipped (mode=none)")
        decoder = self._create_base_decoder()
        return decoder

    def _force_compile_decoder(self) -> VAEDecoderWrapper:
        """Force recompilation of VAE decoder"""
        print("Force compiling VAE decoder...")
        decoder = self._create_base_decoder()
        decoder.compile(mode="max-autotune-no-cudagraphs")

        print(f"Saving compiled model to {self.compiled_model_path}...")
        torch.save(decoder, self.compiled_model_path)
        print("Compiled model saved!")

        return decoder

    def _auto_load_or_compile_decoder(self) -> VAEDecoderWrapper:
        """Auto-load cached compiled model or compile if not available"""
        if os.path.exists(self.compiled_model_path):
            print(f"Loading cached compiled model from {self.compiled_model_path}...")
            try:
                decoder = torch.load(
                    self.compiled_model_path,
                    map_location=self.device,
                    weights_only=False
                )
                print("Cached compiled model loaded!")
                return decoder
            except Exception as e:
                print(f"Warning: Failed to load cached model: {e}")
                print("Falling back to recompilation...")
                return self._force_compile_decoder()
        else:
            print(f"No cached compiled model found. Compiling VAE decoder (first run)...")
            decoder = self._create_base_decoder()
            decoder.compile(mode="max-autotune-no-cudagraphs")

            print(f"Saving compiled model to {self.compiled_model_path}...")
            torch.save(decoder, self.compiled_model_path)
            print("Compiled model saved for future use!")

            return decoder

    def _create_base_decoder(self) -> VAEDecoderWrapper:
        """Create base VAE decoder from weights"""
        # Load base model
        decoder = VAEDecoderWrapper()

        # Load VAE state dict and extract decoder parameters
        vae_state_dict = torch.load(
            self.vae_weights_path,
            map_location="cpu"
        )

        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value

        decoder.load_state_dict(decoder_state_dict)
        decoder.to(self.device, torch.float16)
        decoder.requires_grad_(False)
        decoder.eval()

        return decoder

    def clear_cache(self):
        """Remove cached compiled model file"""
        if os.path.exists(self.compiled_model_path):
            os.remove(self.compiled_model_path)
            print(f"Cleared cached compiled model: {self.compiled_model_path}")
        else:
            print("No cached compiled model to clear")


def load_vae_decoder_with_cache(
    pretrained_model_path: str,
    device: torch.device,
    compile_mode: str = "auto"
) -> VAEDecoderWrapper:
    """
    Convenience function to load VAE decoder with compilation cache

    Args:
        pretrained_model_path: Path to pretrained models directory
        device: PyTorch device
        compile_mode: Compilation mode ("auto", "force", or "none")

    Returns:
        VAE decoder model
    """
    manager = VAECompilationManager(
        pretrained_model_path=pretrained_model_path,
        device=device,
        compile_mode=compile_mode
    )
    return manager.load_vae_decoder()
