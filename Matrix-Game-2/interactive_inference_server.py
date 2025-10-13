"""
Interactive HTTP Streaming Server for Matrix Game Inference (Refactored)

This is a refactored version with improved modularity and extensibility:
- Separate configuration management for different game modes
- Modular action parsing and mapping
- Clean separation of model loading, inference, and serving
- Easy to add new game modes or extend functionality
"""
import asyncio
import argparse
import torch
from omegaconf import OmegaConf

from utils.misc import set_seed
from server import (
    GameConfig,
    ModelManager,
    ActionMapper,
    StreamingServer,
    InferenceEngine
)


class InteractiveGameServer:
    """Main application orchestrator"""

    def __init__(self, args):
        """
        Initialize the interactive game server

        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.enable_profile = args.enable_profile
        self.profiler = None

        # Load game mode from config
        inference_config = OmegaConf.load(args.config_path)
        self.mode = inference_config.pop('mode', 'universal')

        # Get game-specific configuration
        self.game_config = GameConfig.get_config(self.mode)

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all server components"""
        print("=" * 60)
        print("🎮 Matrix Game Interactive Streaming Server")
        print("=" * 60)
        print(f"Mode: {self.mode}")
        print(f"Config: {self.args.config_path}")
        print("=" * 60)

        # 1. Model Manager - handles all model loading
        print("\n[1/4] Initializing Model Manager...")
        self.model_manager = ModelManager(
            config_path=self.args.config_path,
            pretrained_model_path=self.args.pretrained_model_path,
            checkpoint_path=self.args.checkpoint_path,
            device=self.device,
            dtype=self.dtype,
            vae_compile_mode=self.args.vae_compile_mode
        )
        self.model_manager.load_models()

        # 2. Action Mapper - handles input parsing
        print("\n[2/4] Initializing Action Mapper...")
        self.action_mapper = ActionMapper(
            config=self.game_config,
            device=self.device,
            dtype=self.dtype
        )

        # 3. Streaming Server - handles HTTP/WebSocket
        print("\n[3/4] Initializing Streaming Server...")
        self.server = StreamingServer(
            config=self.game_config,
            frame_queue_size=30
        )

        # 4. Inference Engine - handles frame generation
        print("\n[4/4] Initializing Inference Engine...")
        # Note: profiler will be set later when starting
        self.inference_engine = InferenceEngine(
            model_manager=self.model_manager,
            action_mapper=self.action_mapper,
            server=self.server,
            max_latent_frames=self.args.max_latent_frames,
            profiler=None,  # Will be set in _start_with_profiling if enabled
            profile_steps=4
        )

        print("\n✅ All components initialized successfully!")

    async def start(self):
        """Start the server and inference loop"""
        print("\n" + "=" * 60)
        print(f"🚀 Starting server on http://{self.args.server_host}:{self.args.server_port}")
        if self.enable_profile:
            print(f"📊 Profiling enabled - output: {self.args.profile_output}")
        print("=" * 60)

        # Start server task
        server_task = asyncio.create_task(
            self.server.start(
                host=self.args.server_host,
                port=self.args.server_port
            )
        )

        # Wait for server to initialize
        await asyncio.sleep(2)

        # Start inference task (with or without profiling)
        if self.enable_profile:
            await self._start_with_profiling(server_task)
        else:
            inference_task = asyncio.create_task(
                self.inference_engine.run(image_path=self.args.img_path)
            )
            try:
                await asyncio.gather(server_task, inference_task)
            except KeyboardInterrupt:
                print("\n\n🛑 Shutting down gracefully...")
            finally:
                print("✅ Cleanup complete")

    async def _start_with_profiling(self, server_task):
        """Start inference with torch profiling enabled"""
        print("\n📊 Starting profiled run...")
        print("Note: Profiling first 4 inference iterations only.")
        print("Profile will be automatically saved after 4 iterations.")

        # Create profiler
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            with_modules=True,
        )

        # Pass profiler to inference engine
        self.inference_engine.profiler = self.profiler

        # Start profiler
        self.profiler.__enter__()

        # Track if we've already exported the profile
        profile_exported = False

        # Start inference task
        inference_task = asyncio.create_task(
            self.inference_engine.run(image_path=self.args.img_path)
        )

        try:
            # Monitor inference count and auto-export after 4 steps
            while not inference_task.done():
                await asyncio.sleep(0.5)

                # Check if we've completed profiling
                if self.inference_engine.inference_count >= 4 and not profile_exported:
                    print("\n📊 4 inference iterations completed - stopping profiler...")
                    # Stop profiler and export
                    self.profiler.__exit__(None, None, None)
                    self._export_profile()
                    profile_exported = True
                    print("✅ Profiling complete! Server continues running...\n")

            await asyncio.gather(server_task, inference_task)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down gracefully...")
        finally:
            # Export profile if not already done
            if self.profiler is not None and not profile_exported:
                self.profiler.__exit__(None, None, None)
                self._export_profile()
            print("✅ Cleanup complete")

    def _export_profile(self):
        """Export profiling trace to file"""
        import os

        print("\n📊 Exporting profile trace...")
        try:
            self.profiler.export_chrome_trace(self.args.profile_output)
            file_size = os.path.getsize(self.args.profile_output) / 1024 / 1024
            print(f"✅ Profile trace saved to: {self.args.profile_output}")
            print(f"   File size: {file_size:.2f} MB")
            print(f"   Open in Chrome: chrome://tracing")
        except Exception as e:
            print(f"❌ Failed to export profile trace: {e}")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Matrix Game Interactive HTTP Streaming Server (Refactored)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and config paths
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/inference_yaml/inference_universal.yaml",
        help="Path to inference configuration YAML"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to model checkpoint (optional)"
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="Matrix-Game-2.0",
        help="Path to pretrained models"
    )

    # Server settings
    parser.add_argument(
        "--server_host",
        type=str,
        default="0.0.0.0",
        help="Server host address"
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=8000,
        help="Server port"
    )

    # Inference settings
    parser.add_argument(
        "--img_path",
        type=str,
        default=None,
        help="Initial image path (prompts if not provided)"
    )
    parser.add_argument(
        "--max_latent_frames",
        type=int,
        default=300,
        help="Maximum latent frames (300 = ~1200 actual frames)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--vae_compile_mode",
        type=str,
        default="auto",
        choices=["auto", "force", "none"],
        help="VAE decoder compile mode: auto (use cache if available), force (recompile), none (no compile)"
    )
    parser.add_argument(
        "--enable_profile",
        action="store_true",
        help="Enable torch profiling for performance analysis"
    )
    parser.add_argument(
        "--profile_output",
        type=str,
        default="profile_trace.json",
        help="Path to save profile trace file"
    )

    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()
    set_seed(args.seed)

    # Create and start server
    game_server = InteractiveGameServer(args)
    await game_server.start()


if __name__ == "__main__":
    asyncio.run(main())
