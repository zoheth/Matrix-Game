"""
HTTP Streaming Server with WebSocket for real-time interaction
"""
import asyncio
import json
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Optional, Set
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

from server.config import GameModeConfig
from server.templates import generate_html_page


class StreamingServer:
    """HTTP/WebSocket server for streaming frames and receiving input"""

    def __init__(self, config: GameModeConfig, frame_queue_size: int = 30):
        """
        Initialize streaming server

        Args:
            config: Game mode configuration
            frame_queue_size: Maximum frames to buffer
        """
        self.config = config
        self.app = FastAPI(title="Matrix Game Interactive Streaming")
        self.frame_queue = asyncio.Queue(maxsize=frame_queue_size)
        self.action_queue = asyncio.Queue()  # Unbounded for input buffering
        self.image_queue = asyncio.Queue()  # Queue for client-uploaded images
        self.active_connections: Set[WebSocket] = set()
        self.last_frame: Optional[np.ndarray] = None
        self.fps_stats = deque(maxlen=30)

        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes"""

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

    def _get_html_page(self) -> str:
        """Generate HTML page with mode-specific controls"""
        return generate_html_page(self.config.control_instructions_html)

    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection for real-time input and image upload"""
        await websocket.accept()
        self.active_connections.add(websocket)

        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)

                # Check message type
                if message.get('type') == 'image':
                    # Handle image upload
                    await self._handle_image_upload(message)
                else:
                    # Handle regular action (keyboard/mouse)
                    await self.action_queue.put(message)
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
        except Exception as e:
            print(f"WebSocket error: {e}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def _handle_image_upload(self, message: Dict):
        """
        Handle image upload from client

        Args:
            message: Dictionary containing image data in base64 format
                     Expected format: {"type": "image", "data": "base64_string"}
        """
        try:
            # Extract base64 image data
            image_data = message.get('data', '')

            # Remove data URL prefix if present (e.g., "data:image/png;base64,")
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]

            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_data)

            # Load image using PIL
            image = Image.open(BytesIO(image_bytes))

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Put image in queue
            await self.image_queue.put(image)

            print(f"✅ Received new image from client: {image.size}")

        except Exception as e:
            print(f"❌ Error processing uploaded image: {e}")

    async def _generate_stream(self):
        """Generate MJPEG stream, reusing last frame on timeout"""
        while True:
            frame_to_send = None

            try:
                # Wait for new frame with timeout
                new_frame = await asyncio.wait_for(
                    self.frame_queue.get(),
                    timeout=1.0
                )

                if new_frame is None:  # Sentinel value for shutdown
                    break

                self.last_frame = new_frame
                frame_to_send = self.last_frame

            except asyncio.TimeoutError:
                # Reuse last frame if no new frame available
                if self.last_frame is not None:
                    frame_to_send = self.last_frame

            # Send black frame if no frame available yet
            if frame_to_send is None:
                frame_to_send = np.zeros((352, 640, 3), dtype=np.uint8)

            # Encode as JPEG
            _, jpeg = cv2.imencode(
                '.jpg',
                cv2.cvtColor(frame_to_send, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            frame_bytes = jpeg.tobytes()

            # Send as MJPEG chunk
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    async def add_frame(self, frame: np.ndarray, fps: float = 0, latency: float = 0):
        """
        Add frame to streaming queue and send stats to clients

        Args:
            frame: RGB frame as numpy array
            fps: Current FPS
            latency: Current latency in milliseconds
        """
        # Drop oldest frame if queue is full
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        await self.frame_queue.put(frame)

        # Broadcast stats to all connected clients
        stats = {"fps": fps, "latency": latency}
        await self._broadcast_stats(stats)

    async def _broadcast_stats(self, stats: Dict):
        """Broadcast statistics to all connected WebSocket clients"""
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_json(stats)
            except:
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections.discard(conn)

    async def get_next_action(self, timeout: Optional[float] = None) -> Optional[Dict]:
        """
        Get the latest action from queue, discarding stale ones

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Latest action dict or None if timeout/empty
        """
        try:
            # Get first action (blocks if queue is empty)
            if timeout:
                latest_action = await asyncio.wait_for(
                    self.action_queue.get(),
                    timeout=timeout
                )
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
                print(f"⚠️  Discarded {discarded_count} old actions, using latest")

            return latest_action

        except asyncio.TimeoutError:
            return None

    async def get_next_image(self, timeout: Optional[float] = None) -> Optional[Image.Image]:
        """
        Get next uploaded image from queue

        Args:
            timeout: Optional timeout in seconds

        Returns:
            PIL Image or None if timeout/empty
        """
        try:
            if timeout:
                image = await asyncio.wait_for(
                    self.image_queue.get(),
                    timeout=timeout
                )
            else:
                image = await self.image_queue.get()

            return image

        except asyncio.TimeoutError:
            return None

    def has_pending_image(self) -> bool:
        """Check if there's a pending image in the queue"""
        return not self.image_queue.empty()

    async def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the HTTP/WebSocket server"""
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
