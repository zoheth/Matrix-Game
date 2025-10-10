"""
Interactive HTTP Streaming Server for Matrix Game Inference
Real-time interaction via WebSocket for keyboard/mouse input
Dynamic frame rate control and on-demand inference
"""

import asyncio
import os
import argparse
import torch
import numpy as np
import cv2
import copy
import time
import json
from typing import Optional, Dict
from pathlib import Path
from collections import deque

from omegaconf import OmegaConf
from torchvision.transforms import v2
from diffusers.utils import load_image
from einops import rearrange

from pipeline import CausalInferenceStreamingPipeline
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.misc import set_seed
from utils.conditions import *
from utils.wan_wrapper import WanDiffusionWrapper
from safetensors.torch import load_file
from demo_utils.constant import ZERO_VAE_CACHE

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn


class InteractiveStreamingServer:
    """Interactive HTTP streaming server with WebSocket for input"""

    def __init__(self, mode='universal'):
        self.app = FastAPI(title="Matrix Game Interactive Streaming")
        self.frame_queue = asyncio.Queue(maxsize=30)
        self.action_queue = asyncio.Queue()  # No limit, buffer all actions
        self.is_streaming = False
        self.mode = mode
        self.fps_stats = deque(maxlen=30)  # Track FPS over last 30 frames
        self.active_connections = set()
        self.last_frame: Optional[np.ndarray] = None 
        self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            return self._get_html_page()

        @self.app.get("/stream")
        async def stream():
            return StreamingResponse(
                self._generate_stream(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_websocket(websocket)

    def _get_html_page(self):
        """Generate interactive HTML page with keyboard/mouse controls"""
        control_instructions = self._get_control_instructions()

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Matrix Game Interactive Stream</title>
            <style>
                body {{
                    margin: 0;
                    background: #000;
                    font-family: Arial, sans-serif;
                    overflow: hidden;
                }}
                .container {{
                    display: flex;
                    height: 100vh;
                }}
                .video-panel {{
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    position: relative;
                }}
                .control-panel {{
                    width: 300px;
                    background: #1a1a1a;
                    padding: 20px;
                    overflow-y: auto;
                    color: #fff;
                }}
                h1 {{
                    color: #fff;
                    margin: 0 0 20px 0;
                    font-size: 24px;
                }}
                img {{
                    max-width: 90%;
                    max-height: 90vh;
                    border: 2px solid #fff;
                    border-radius: 8px;
                }}
                .status {{
                    position: absolute;
                    top: 20px;
                    right: 20px;
                    background: rgba(0, 0, 0, 0.7);
                    padding: 10px 20px;
                    border-radius: 5px;
                    color: #0f0;
                    font-family: monospace;
                }}
                .info {{
                    background: #2a2a2a;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 15px;
                }}
                .info h3 {{
                    margin: 0 0 10px 0;
                    color: #4CAF50;
                }}
                .info p {{
                    margin: 5px 0;
                    font-size: 14px;
                    line-height: 1.6;
                }}
                .current-action {{
                    background: #2a2a2a;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 15px;
                }}
                .action-display {{
                    font-size: 16px;
                    color: #FFA500;
                    font-weight: bold;
                    margin-top: 10px;
                }}
                kbd {{
                    background: #4CAF50;
                    color: white;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: monospace;
                    font-weight: bold;
                }}
                .connected {{
                    color: #0f0;
                }}
                .disconnected {{
                    color: #f00;
                }}
                #videoStream {{
                    cursor: crosshair;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="video-panel">
                    <img id="videoStream" src="/stream" alt="Game Stream" tabindex="0">
                    <div class="status">
                        <div>Status: <span id="wsStatus" class="disconnected">Disconnected</span></div>
                        <div>FPS: <span id="fpsDisplay">0</span></div>
                        <div>Latency: <span id="latencyDisplay">0ms</span></div>
                    </div>
                </div>
                <div class="control-panel">
                    <h1>🎮 Controls</h1>
                    {control_instructions}
                    <div class="current-action">
                        <h3>Current Action</h3>
                        <div class="action-display" id="currentAction">Waiting for input...</div>
                    </div>
                    <div class="info">
                        <h3>ℹ️ Info</h3>
                        <p>• Click on the video to focus</p>
                        <p>• Press keys to control the game</p>
                        <p>• New frames generated on demand</p>
                        <p>• Up to 300 latent frames (~1200 actual)</p>
                        <p>• Auto-resets when limit reached</p>
                    </div>
                </div>
            </div>
            <script>
                let ws = null;
                let mousePos = {{x: 0, y: 0}};
                let lastMousePos = {{x: 0, y: 0}};
                const videoElement = document.getElementById('videoStream');
                const wsStatusElement = document.getElementById('wsStatus');
                const fpsDisplay = document.getElementById('fpsDisplay');
                const latencyDisplay = document.getElementById('latencyDisplay');
                const currentActionDisplay = document.getElementById('currentAction');

                // Connect WebSocket
                function connectWebSocket() {{
                    ws = new WebSocket(`ws://${{window.location.host}}/ws`);

                    ws.onopen = () => {{
                        wsStatusElement.textContent = 'Connected';
                        wsStatusElement.className = 'connected';
                        console.log('WebSocket connected');
                    }};

                    ws.onclose = () => {{
                        wsStatusElement.textContent = 'Disconnected';
                        wsStatusElement.className = 'disconnected';
                        console.log('WebSocket disconnected, reconnecting...');
                        setTimeout(connectWebSocket, 1000);
                    }};

                    ws.onerror = (error) => {{
                        console.error('WebSocket error:', error);
                    }};

                    ws.onmessage = (event) => {{
                        const data = JSON.parse(event.data);
                        if (data.fps !== undefined) {{
                            fpsDisplay.textContent = data.fps.toFixed(1);
                        }}
                        if (data.latency !== undefined) {{
                            latencyDisplay.textContent = data.latency.toFixed(0) + 'ms';
                        }}
                    }};
                }}

                connectWebSocket();

                // Keyboard event handling
                videoElement.addEventListener('keydown', (e) => {{
                    if (ws && ws.readyState === WebSocket.OPEN) {{
                        const action = {{
                            type: 'keyboard',
                            key: e.key.toLowerCase(),
                            timestamp: Date.now()
                        }};
                        ws.send(JSON.stringify(action));
                        updateActionDisplay(action);
                        e.preventDefault();
                    }}
                }});

                function updateActionDisplay(action) {{
                    let displayText = '';
                    if (action.type === 'keyboard') {{
                        displayText = `Keyboard: ${{action.key.toUpperCase()}}`;
                    }} else if (action.type === 'mouse') {{
                        displayText = `Mouse: Δx=${{action.dx.toFixed(3)}}, Δy=${{action.dy.toFixed(3)}}`;
                    }}
                    currentActionDisplay.textContent = displayText;
                }}

                // Auto-focus video element
                videoElement.focus();
            </script>
        </body>
        </html>
        """

    def _get_control_instructions(self):
        """Get control instructions based on mode"""
        if self.mode == 'universal':
            return """
            <div class="info">
                <h3>🎮 Universal Mode</h3>
                <p><strong>Camera:</strong></p>
                <p>• <kbd>I</kbd> = Up</p>
                <p>• <kbd>K</kbd> = Down</p>
                <p>• <kbd>J</kbd> = Left</p>
                <p>• <kbd>L</kbd> = Right</p>
                <p>• <kbd>U</kbd> = No move</p>
                <p><strong>Movement:</strong></p>
                <p>• <kbd>W</kbd> = Forward</p>
                <p>• <kbd>S</kbd> = Back</p>
                <p>• <kbd>A</kbd> = Left</p>
                <p>• <kbd>D</kbd> = Right</p>
                <p>• <kbd>Q</kbd> = No move</p>
            </div>
            """
        elif self.mode == 'gta_drive':
            return """
            <div class="info">
                <h3>🚗 GTA Drive Mode</h3>
                <p><strong>Steering:</strong></p>
                <p>• <kbd>A</kbd> = Left</p>
                <p>• <kbd>D</kbd> = Right</p>
                <p>• <kbd>Q</kbd> = Straight</p>
                <p><strong>Acceleration:</strong></p>
                <p>• <kbd>W</kbd> = Forward</p>
                <p>• <kbd>S</kbd> = Back</p>
                <p>• <kbd>Q</kbd> = Coast</p>
            </div>
            """
        else:  # templerun
            return """
            <div class="info">
                <h3>🏃 Temple Run Mode</h3>
                <p>• <kbd>W</kbd> = Jump</p>
                <p>• <kbd>S</kbd> = Slide</p>
                <p>• <kbd>A</kbd> = Left side</p>
                <p>• <kbd>D</kbd> = Right side</p>
                <p>• <kbd>Z</kbd> = Turn left</p>
                <p>• <kbd>C</kbd> = Turn right</p>
                <p>• <kbd>Q</kbd> = No move</p>
            </div>
            """

    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection for real-time input"""
        await websocket.accept()
        self.active_connections.add(websocket)

        try:
            while True:
                data = await websocket.receive_text()
                action = json.loads(data)  # Parse JSON action safely
                await self.action_queue.put(action)
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
        except Exception as e:
            print(f"WebSocket error: {e}")
            self.active_connections.remove(websocket)

    async def _generate_stream(self):
        """Generate MJPEG stream, sending the last known frame on timeout."""
        while True:
            frame_to_send = None
            try:
                new_frame = await asyncio.wait_for(
                    self.frame_queue.get(),
                    timeout=1.0
                )

                if new_frame is None:
                    break
                
                self.last_frame = new_frame
                frame_to_send = self.last_frame

            except asyncio.TimeoutError:
                if self.last_frame is not None:
                    frame_to_send = self.last_frame

            if frame_to_send is None:
                frame_to_send = np.zeros((352, 640, 3), dtype=np.uint8)

            _, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame_to_send, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    async def add_frame(self, frame: np.ndarray, fps: float = 0, latency: float = 0):
        """Add frame to streaming queue and send stats to clients"""
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self.frame_queue.put(frame)

        # Send FPS and latency stats to all connected clients
        stats = {"fps": fps, "latency": latency}
        for connection in list(self.active_connections):
            try:
                await connection.send_json(stats)
            except:
                self.active_connections.discard(connection)

    async def get_next_action(self, timeout: float = None) -> Optional[Dict]:
        """Get LATEST action from queue, discarding old ones"""
        try:
            # Get first action (blocks if queue is empty)
            if timeout:
                latest_action = await asyncio.wait_for(self.action_queue.get(), timeout=timeout)
            else:
                latest_action = await self.action_queue.get()

            # Drain all remaining actions, keeping only the latest
            discarded_count = 0
            while not self.action_queue.empty():
                try:
                    latest_action = self.action_queue.get_nowait()
                    discarded_count += 1
                except asyncio.QueueEmpty:
                    break

            if discarded_count > 0:
                print(f"⚠️  Discarded {discarded_count} old actions, using latest: {latest_action.get('type')}")

            return latest_action
        except asyncio.TimeoutError:
            return None

    async def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the server"""
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


class InteractiveStreamingInference:
    """Real-time interactive inference engine"""

    def __init__(self, args, server):
        self.args = args
        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16
        self.server = server
        self._init_config()
        self._init_models()
        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Action mapping
        self.CAM_VALUE = 0.1
        self._init_action_maps()

    def _init_config(self):
        self.config = OmegaConf.load(self.args.config_path)

    def _init_models(self):
        print("Initializing models...")
        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}), is_causal=True)
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load(
            os.path.join(self.args.pretrained_model_path, "Wan2.1_VAE.pth"),
            map_location="cpu"
        )
        decoder_state_dict = {k: v for k, v in vae_state_dict.items() if 'decoder.' in k or 'conv2' in k}
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(self.device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")

        pipeline = CausalInferenceStreamingPipeline(self.config, generator=generator, vae_decoder=current_vae_decoder)
        if self.args.checkpoint_path:
            print(f"Loading checkpoint from {self.args.checkpoint_path}")
            state_dict = load_file(self.args.checkpoint_path)
            pipeline.generator.load_state_dict(state_dict)
        self.pipeline = pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline.vae_decoder.to(torch.float16)

        vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.device, self.weight_dtype)
        print("Models initialized successfully")

    def _init_action_maps(self):
        """Initialize action mappings for different modes"""
        mode = self.server.mode

        if mode == 'universal':
            # Camera: mouse movement (continuous)
            self.camera_map = {
                "i": [self.CAM_VALUE, 0],
                "k": [-self.CAM_VALUE, 0],
                "j": [0, -self.CAM_VALUE],
                "l": [0, self.CAM_VALUE],
                "u": [0, 0]
            }
            # Keyboard: 4-dim one-hot [W, S, A, D]
            self.keyboard_map = {
                "w": 0, "s": 1, "a": 2, "d": 3, "q": -1
            }
            self.keyboard_dim = 4

        elif mode == 'gta_drive':
            # Camera: steering (continuous)
            self.camera_map = {
                "a": [0, -self.CAM_VALUE],
                "d": [0, self.CAM_VALUE],
                "q": [0, 0]
            }
            # Keyboard: 2-dim one-hot [W, S]
            self.keyboard_map = {
                "w": 0, "s": 1, "q": -1
            }
            self.keyboard_dim = 2

        else:  # templerun
            self.camera_map = None
            # Keyboard: 7-dim one-hot [None, Jump, Slide, TurnLeft, TurnRight, LeftSide, RightSide]
            self.keyboard_map = {
                "q": 0,  # no action
                "w": 1,  # jump
                "s": 2,  # slide
                "z": 3,  # turn left
                "c": 4,  # turn right
                "a": 5,  # left side
                "d": 6   # right side
            }
            self.keyboard_dim = 7

    def parse_action(self, action_data: Dict) -> Optional[Dict]:
        """Parse action from client input"""
        mode = self.server.mode

        if action_data['type'] == 'keyboard':
            key = action_data['key']

            if mode != 'templerun':
                # Check if it's a camera or keyboard action
                if key in self.camera_map:
                    # Camera action (mouse)
                    keyboard_onehot = torch.zeros(self.keyboard_dim, device=self.device, dtype=self.weight_dtype)
                    return {
                        'mouse': torch.tensor(self.camera_map[key], device=self.device, dtype=self.weight_dtype),
                        'keyboard': keyboard_onehot
                    }
                elif key in self.keyboard_map:
                    # Keyboard action
                    idx = self.keyboard_map[key]
                    keyboard_onehot = torch.zeros(self.keyboard_dim, device=self.device, dtype=self.weight_dtype)
                    if idx >= 0:
                        keyboard_onehot[idx] = 1.0

                    mouse_zero = torch.tensor([0.0, 0.0], device=self.device, dtype=self.weight_dtype)
                    return {
                        'mouse': mouse_zero,
                        'keyboard': keyboard_onehot
                    }
            else:
                # Temple run mode - keyboard only
                if key in self.keyboard_map:
                    idx = self.keyboard_map[key]
                    keyboard_onehot = torch.zeros(self.keyboard_dim, device=self.device, dtype=self.weight_dtype)
                    if idx >= 0:
                        keyboard_onehot[idx] = 1.0
                    return {
                        'keyboard': keyboard_onehot
                    }

        # Disable mouse movement - use keyboard only like inference_streaming.py
        # elif action_data['type'] == 'mouse' and mode != 'templerun':
        #     dx = action_data.get('dx', 0)
        #     dy = action_data.get('dy', 0)
        #     mouse_cond = torch.tensor([dy * self.CAM_VALUE, dx * self.CAM_VALUE],
        #                              device=self.device, dtype=self.weight_dtype)
        #     keyboard_onehot = torch.zeros(self.keyboard_dim, device=self.device, dtype=self.weight_dtype)
        #     return {
        #         'mouse': mouse_cond,
        #         'keyboard': keyboard_onehot
        #     }

        return None

    def _resizecrop(self, image, th, tw):
        w, h = image.size
        if h / w > th / tw:
            new_w, new_h = int(w), int(w * th / tw)
        else:
            new_h, new_w = int(h), int(h * tw / th)
        return image.crop(((w - new_w) / 2, (h - new_h) / 2, (w + new_w) / 2, (h + new_h) / 2))

    async def run_interactive_inference(self, image_path=None):
        """Run interactive inference loop - generates frames on demand"""
        mode = self.server.mode

        # Load initial image
        if image_path is None:
            img_path = input("Please input the image path: ").strip()
            image = load_image(img_path)
        else:
            image, img_path = load_image(image_path), image_path

        image = self._resizecrop(image, 352, 640)
        image_tensor = self.frame_process(image)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)

        # Prepare initial conditions with configurable buffer size
        max_latent_frames = getattr(self.args, 'max_latent_frames', 300)
        num_frames_total = 4 * max_latent_frames + 1

        print(f"Allocating buffer for {max_latent_frames} latent frames ({num_frames_total} actual frames)")
        estimated_memory_mb = (num_frames_total * 2 * 4) / (1024 * 1024)  # Rough estimate
        print(f"Estimated conditional dict memory: ~{estimated_memory_mb:.1f} MB")

        # Encode first frame only initially to save memory
        padding_video = torch.zeros_like(image_tensor).repeat(1, 1, 4 * (max_latent_frames - 1), 1, 1)
        img_cond = torch.concat([image_tensor, padding_video], dim=2)
        tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)

        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
        visual_context = self.vae.clip.encode_video(image_tensor)

        # Initialize conditional dict with reasonable size
        conditional_dict = {
            "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
            "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype)
        }

        # Get keyboard dimension from config
        if mode == 'universal':
            keyboard_size = 4
        elif mode == 'gta_drive':
            keyboard_size = 2
        else:  # templerun
            keyboard_size = 7

        if mode != 'templerun':
            conditional_dict['mouse_cond'] = torch.zeros([1, num_frames_total, 2], device=self.device, dtype=self.weight_dtype)

        conditional_dict['keyboard_cond'] = torch.zeros([1, num_frames_total, keyboard_size], device=self.device, dtype=self.weight_dtype)

        # Store max frames for later checks
        self.max_latent_frames = max_latent_frames

        print("=" * 60)
        print("🎮 Interactive Inference Started!")
        print("=" * 60)
        print("• Waiting for client input via WebSocket...")
        print("• Open browser at http://{}:{}".format(self.args.server_host, self.args.server_port))
        print("• Press keys to generate new frames on demand")
        print("• No frame limit - play infinitely!")
        print("=" * 60)

        await self._inference_loop(conditional_dict, mode)

    async def _inference_loop(self, conditional_dict, mode):
        """Main inference loop - generate frames on demand"""
        batch_size = 1
        num_frame_per_block = self.pipeline.num_frame_per_block  # Use pipeline's setting

        # Clear GPU cache before starting
        torch.cuda.empty_cache()

        # Initialize caches
        self.pipeline.kv_cache1 = None
        self.pipeline.kv_cache_keyboard = None
        self.pipeline.kv_cache_mouse = None
        self.pipeline.crossattn_cache = None

        self.pipeline._initialize_kv_cache(batch_size=batch_size, dtype=self.weight_dtype, device=self.device)
        self.pipeline._initialize_kv_cache_mouse_and_keyboard(batch_size=batch_size, dtype=self.weight_dtype, device=self.device)
        self.pipeline._initialize_crossattn_cache(batch_size=batch_size, dtype=self.weight_dtype, device=self.device)

        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None

        current_start_frame = 0
        frame_times = deque(maxlen=30)

        print("✅ Ready! Waiting for first action...")

        with torch.no_grad():
            while True:
                # Wait for next action from client
                action_data = await self.server.get_next_action()

                if action_data is None:
                    continue

                # Parse action
                current_action = self.parse_action(action_data)
                if current_action is None:
                    continue

                # Check if we've reached the frame limit
                if current_start_frame >= self.max_latent_frames:
                    print(f"⚠️ Reached maximum frame limit ({self.max_latent_frames} latent frames)")
                    print(f"🔄 Resetting to continue playing...")
                    # Reset to beginning but keep KV caches for continuity
                    current_start_frame = 0
                    # Clear VAE cache
                    for j in range(len(vae_cache)):
                        vae_cache[j] = None
                    print("✅ Reset complete, continuing from frame 0")

                print(f"📥 Action received: {action_data['type']} - {action_data.get('key', action_data.get('dx', 0))}")

                inference_start = time.time()

                # Update conditional dict with current action
                self._update_conditional_dict(conditional_dict, current_action, current_start_frame, num_frame_per_block, mode)

                # Generate noise for current block
                noise = torch.randn([batch_size, 16, num_frame_per_block, 44, 80], device=self.device, dtype=self.weight_dtype)

                # Denoising loop
                noisy_input = noise

                # Get current conditions
                curr_cond = self._cond_current(conditional_dict, current_start_frame, num_frame_per_block, mode=mode)

                # Debug: print shapes
                if current_start_frame == 0:
                    print(f"DEBUG: current_start_frame={current_start_frame}")
                    print(f"DEBUG: keyboard_cond shape: {curr_cond['keyboard_cond'].shape}")
                    if 'mouse_cond' in curr_cond:
                        print(f"DEBUG: mouse_cond shape: {curr_cond['mouse_cond'].shape}")
                    print(f"DEBUG: Expected frames: {1 + 4 * (current_start_frame + num_frame_per_block - 1)}")

                for index, current_timestep in enumerate(self.pipeline.denoising_step_list):
                    timestep = torch.ones([batch_size, num_frame_per_block], device=self.device, dtype=torch.int64) * current_timestep
                    _, denoised_pred = self.pipeline.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=curr_cond,
                        timestep=timestep,
                        kv_cache=self.pipeline.kv_cache1,
                        kv_cache_mouse=self.pipeline.kv_cache_mouse,
                        kv_cache_keyboard=self.pipeline.kv_cache_keyboard,
                        crossattn_cache=self.pipeline.crossattn_cache,
                        current_start=current_start_frame * self.pipeline.frame_seq_length
                    )

                    if index < len(self.pipeline.denoising_step_list) - 1:
                        next_timestep = self.pipeline.denoising_step_list[index + 1]
                        noisy_input = self.pipeline.scheduler.add_noise(
                            rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),
                            torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                            next_timestep * torch.ones([batch_size * num_frame_per_block], device=self.device, dtype=torch.long)
                        )
                        noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=batch_size)

                # Update KV cache with clean context
                context_timestep = torch.ones([batch_size, num_frame_per_block], device=self.device, dtype=torch.int64) * self.pipeline.args.context_noise
                self.pipeline.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=self._cond_current(conditional_dict, current_start_frame, num_frame_per_block, mode=mode),
                    timestep=context_timestep,
                    kv_cache=self.pipeline.kv_cache1,
                    kv_cache_mouse=self.pipeline.kv_cache_mouse,
                    kv_cache_keyboard=self.pipeline.kv_cache_keyboard,
                    crossattn_cache=self.pipeline.crossattn_cache,
                    current_start=current_start_frame * self.pipeline.frame_seq_length
                )

                # Decode to video
                denoised_pred = denoised_pred.transpose(1, 2)
                video, vae_cache = self.pipeline.vae_decoder(denoised_pred.half(), *vae_cache)
                video_np = rearrange(video, "B T C H W -> B T H W C")
                video_np = ((video_np.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
                video_np = np.ascontiguousarray(video_np)

                inference_time = time.time() - inference_start
                frame_times.append(inference_time)

                # Calculate FPS and latency
                avg_frame_time = sum(frame_times) / len(frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                latency_ms = avg_frame_time * 1000

                # Send frames to client
                for frame_idx in range(video_np.shape[0]):
                    await self.server.add_frame(video_np[frame_idx], fps=fps, latency=latency_ms)

                current_start_frame += num_frame_per_block

                print(f"✅ Frame {current_start_frame} generated | FPS: {fps:.1f} | Latency: {latency_ms:.0f}ms")

    def _update_conditional_dict(self, conditional_dict, action, current_start_frame, num_frame_per_block, mode):
        """Update conditional dict with new action - matches original logic"""
        # Calculate the frame range to fill (matches cond_current logic)
        if current_start_frame == 0:
            last_frame_num = 1 + 4 * (num_frame_per_block - 1)  # For frame 0, fill first frame
        else:
            last_frame_num = 4 * num_frame_per_block

        final_frame = 1 + 4 * (current_start_frame + num_frame_per_block - 1)
        start_idx = final_frame - last_frame_num
        end_idx = final_frame

        print(f"DEBUG update: start_idx={start_idx}, end_idx={end_idx}, current_start_frame={current_start_frame}")

        if mode != 'templerun' and 'mouse' in action:
            conditional_dict["mouse_cond"][:, start_idx:end_idx] = action['mouse'][None, None, :].repeat(1, last_frame_num, 1)

        if 'keyboard' in action:
            conditional_dict["keyboard_cond"][:, start_idx:end_idx] = action['keyboard'][None, None, :].repeat(1, last_frame_num, 1)

    def _cond_current(self, conditional_dict, current_start_frame, num_frame_per_block, mode='universal'):
        """Get current conditional slice"""
        # For causal inference, need full history up to current point
        end_frame_idx = 1 + 4 * (current_start_frame + num_frame_per_block - 1)

        new_cond = {
            "cond_concat": conditional_dict["cond_concat"][:, :, current_start_frame:current_start_frame + num_frame_per_block],
            "visual_context": conditional_dict["visual_context"]
        }
        if mode != 'templerun':
            new_cond["mouse_cond"] = conditional_dict["mouse_cond"][:, :end_frame_idx]
        new_cond["keyboard_cond"] = conditional_dict["keyboard_cond"][:, :end_frame_idx]
        return new_cond


def parse_args():
    parser = argparse.ArgumentParser(description="Matrix Game Interactive HTTP Streaming")
    parser.add_argument("--config_path", type=str,
                       default="configs/inference_yaml/inference_universal.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained_model_path", type=str,
                       default="Matrix-Game-2.0")
    parser.add_argument("--server_host", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--img_path", type=str, default=None)
    parser.add_argument("--max_latent_frames", type=int, default=300,
                       help="Maximum latent frames (300 = ~1200 actual frames)")
    args = parser.parse_args()
    return args


async def main():
    args = parse_args()
    set_seed(args.seed)

    # Get mode from config
    config = OmegaConf.load(args.config_path)
    mode = config.pop('mode', 'universal')

    # Initialize server
    server = InteractiveStreamingServer(mode=mode)

    # Initialize inference engine
    print("Initializing inference engine...")
    inference = InteractiveStreamingInference(args, server)
    print("Inference engine ready.")

    print("=" * 60)
    print("🎮 Matrix Game Interactive HTTP Streaming Server")
    print("=" * 60)
    print(f"Server starting on: http://{args.server_host}:{args.server_port}")
    print("Open this URL in your browser to play!")
    print("=" * 60)

    # Start server and inference tasks
    server_task = asyncio.create_task(
        server.start(host=args.server_host, port=args.server_port)
    )

    await asyncio.sleep(2)  # Wait for server to start

    inference_task = asyncio.create_task(
        inference.run_interactive_inference(image_path=args.img_path)
    )

    try:
        await asyncio.gather(server_task, inference_task)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        print("Cleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())
