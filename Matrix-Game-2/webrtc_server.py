"""
WebRTC Streaming Server for Real-time Game Inference Visualization

Architecture:
- FastAPI for HTTP and WebSocket signaling
- aiortc for WebRTC peer connection
- Asyncio queue for frame buffering
- Custom VideoStreamTrack for frame delivery
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, Dict
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class InferenceVideoTrack(VideoStreamTrack):
    """
    Custom video track that delivers frames from inference pipeline.

    Uses an asyncio queue for thread-safe frame passing between
    the inference engine and WebRTC streaming.
    """

    def __init__(self):
        super().__init__()
        self.frame_queue = asyncio.Queue(maxsize=30)  # Buffer up to 30 frames
        self._frame_count = 0
        self._timestamp = 0
        self._fps = 30  # Target FPS
        self._frame_duration = 1.0 / self._fps

    async def recv(self):
        """
        Called by WebRTC to get the next frame.
        Returns VideoFrame in the format expected by WebRTC.
        """
        try:
            # Get frame from queue with timeout
            frame_data = await asyncio.wait_for(
                self.frame_queue.get(),
                timeout=5.0
            )

            if frame_data is None:
                # Sentinel value to stop streaming
                raise StopAsyncIteration

            # Convert numpy array to VideoFrame
            frame = self._numpy_to_video_frame(frame_data)

            # Update timing
            pts, time_base = await self.next_timestamp()
            frame.pts = pts
            frame.time_base = time_base

            self._frame_count += 1
            return frame

        except asyncio.TimeoutError:
            logger.warning("Frame timeout - generating black frame")
            # Return a black frame to keep connection alive
            return self._generate_black_frame()

    def _numpy_to_video_frame(self, img: np.ndarray) -> VideoFrame:
        """Convert numpy array (H, W, C) to VideoFrame."""
        # Ensure correct format (RGB, uint8)
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        if len(img.shape) == 2:
            # Grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            # RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif img.shape[2] == 3 and img.shape[2] == 3:
            # Ensure RGB (not BGR)
            pass

        # Create VideoFrame from numpy array
        frame = VideoFrame.from_ndarray(img, format="rgb24")
        return frame

    def _generate_black_frame(self) -> VideoFrame:
        """Generate a black frame as fallback."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        return self._numpy_to_video_frame(img)

    async def add_frame(self, frame: np.ndarray):
        """
        Add a frame to the queue for streaming.
        Called by the inference engine.
        """
        try:
            # Non-blocking put with queue size check
            if self.frame_queue.full():
                # Drop oldest frame if queue is full
                try:
                    self.frame_queue.get_nowait()
                    logger.debug("Dropped frame - queue full")
                except asyncio.QueueEmpty:
                    pass

            await self.frame_queue.put(frame)
        except Exception as e:
            logger.error(f"Error adding frame: {e}")

    async def stop(self):
        """Stop the video track gracefully."""
        await self.frame_queue.put(None)  # Sentinel value


class StreamingServer:
    """
    WebRTC Streaming Server Manager

    Manages multiple peer connections and coordinates between
    inference engine and WebRTC clients.
    """

    def __init__(self):
        self.app = FastAPI(title="Matrix Game WebRTC Streaming")
        self.peers: Dict[str, RTCPeerConnection] = {}
        self.video_tracks: Dict[str, InferenceVideoTrack] = {}
        self.relay = MediaRelay()
        self.active_inference_track: Optional[InferenceVideoTrack] = None

        # Connection state tracking
        self.client_connected = asyncio.Event()
        self.has_active_connection = False

        self._setup_routes()

        # Track template directory
        self.template_dir = Path(__file__).parent / "templates"
        self.template_dir.mkdir(exist_ok=True)

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            """Serve the WebRTC viewer page."""
            template_path = Path(__file__).parent / "templates" / "webrtc_viewer.html"
            if template_path.exists():
                return template_path.read_text()
            return """
            <html>
                <body>
                    <h1>WebRTC Viewer Template Not Found</h1>
                    <p>Please create templates/webrtc_viewer.html</p>
                </body>
            </html>
            """

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "active_peers": len(self.peers),
                "streaming": self.active_inference_track is not None
            }

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for WebRTC signaling."""
            await self._handle_websocket(websocket)

    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebRTC signaling over WebSocket."""
        await websocket.accept()
        peer_id = str(uuid.uuid4())

        # Configure ICE servers for NAT traversal
        from aiortc import RTCConfiguration, RTCIceServer
        config = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun1.l.google.com:19302"])
            ]
        )
        pc = RTCPeerConnection(configuration=config)
        self.peers[peer_id] = pc

        logger.info(f"New peer connected: {peer_id}")

        try:
            # Setup peer connection callbacks
            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"Peer {peer_id} connection state: {pc.connectionState}")
                if pc.connectionState == "connected":
                    self.has_active_connection = True
                    self.client_connected.set()
                    logger.info("🎉 Client connected - ready for streaming!")
                elif pc.connectionState in ["failed", "closed"]:
                    logger.warning(f"Peer {peer_id} connection failed or closed")
                    self.has_active_connection = False
                    # Don't call cleanup here - it will be handled in the finally block

            @pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange():
                logger.info(f"Peer {peer_id} ICE connection state: {pc.iceConnectionState}")
                if pc.iceConnectionState == "failed":
                    logger.error(f"ICE connection failed for peer {peer_id}")

            @pc.on("icegatheringstatechange")
            async def on_icegatheringstatechange():
                logger.info(f"Peer {peer_id} ICE gathering state: {pc.iceGatheringState}")

            # Handle signaling messages
            async for message in websocket.iter_text():
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    logger.info(f"Received signaling message: {msg_type}")

                    if msg_type == "offer":
                        await self._handle_offer(peer_id, pc, data, websocket)
                    elif msg_type == "ice-candidate":
                        await self._handle_ice_candidate(pc, data)
                    elif msg_type == "ping":
                        # Keep-alive ping from client
                        await websocket.send_json({"type": "pong"})
                    else:
                        logger.warning(f"Unknown message type: {msg_type}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)

            # If we exit the message loop, WebSocket was closed by client
            logger.info(f"Client {peer_id} closed WebSocket connection")

        except WebSocketDisconnect as e:
            logger.info(f"Peer {peer_id} WebSocket disconnected: {e}")
        except Exception as e:
            logger.error(f"Error handling peer {peer_id}: {e}", exc_info=True)
        finally:
            logger.info(f"Exiting WebSocket handler for peer {peer_id}")
            await self._cleanup_peer(peer_id)

    async def _handle_offer(self, peer_id: str, pc: RTCPeerConnection,
                           data: dict, websocket: WebSocket):
        """Handle WebRTC offer from client."""
        try:
            # First, set remote description from the client's offer
            offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
            await pc.setRemoteDescription(offer)
            logger.info(f"Set remote description for peer {peer_id}")

            # Create video track if not exists
            if self.active_inference_track is None:
                self.active_inference_track = InferenceVideoTrack()
                logger.info("Created new InferenceVideoTrack")

            # For now, use track directly (relay can cause issues with direction negotiation)
            # TODO: Re-enable relay for multi-client support after fixing direction issues
            video_track = self.active_inference_track
            logger.info(f"Using video track directly for peer {peer_id}")

            # Add track to peer connection
            # The client has a recvonly transceiver, so we add a sendonly track
            sender = pc.addTrack(video_track)
            self.video_tracks[peer_id] = self.active_inference_track
            logger.info(f"Added video track to peer {peer_id}, sender: {sender}")

            # Create and send answer
            answer = await pc.createAnswer()
            logger.info(f"Created answer for peer {peer_id}")

            await pc.setLocalDescription(answer)
            logger.info(f"Set local description for peer {peer_id}")

            await websocket.send_json({
                "type": pc.localDescription.type,
                "sdp": pc.localDescription.sdp
            })

            logger.info(f"Sent answer to peer {peer_id}")
        except Exception as e:
            logger.error(f"Error in _handle_offer: {e}", exc_info=True)
            raise

    async def _handle_ice_candidate(self, pc: RTCPeerConnection, data: dict):
        """Handle ICE candidate from client."""
        try:
            if "candidate" in data and data["candidate"]:
                # Note: aiortc handles ICE candidates automatically in most cases
                # The client sends candidates, but we don't need to manually add them
                logger.debug(f"Received ICE candidate: {data.get('candidate')}")
        except Exception as e:
            logger.error(f"Error handling ICE candidate: {e}")

    async def _cleanup_peer(self, peer_id: str):
        """Clean up peer connection resources."""
        # Check if already cleaned up
        if peer_id not in self.peers and peer_id not in self.video_tracks:
            logger.debug(f"Peer {peer_id} already cleaned up")
            return

        if peer_id in self.peers:
            pc = self.peers[peer_id]
            try:
                await pc.close()
            except Exception as e:
                logger.warning(f"Error closing peer connection: {e}")
            del self.peers[peer_id]

        if peer_id in self.video_tracks:
            del self.video_tracks[peer_id]

        # Update connection state
        if len(self.peers) == 0:
            self.has_active_connection = False
            self.client_connected.clear()

        logger.info(f"Cleaned up peer {peer_id}")

    def get_frame_callback(self):
        """
        Get callback function for inference engine to send frames.

        Returns:
            Async callback that accepts numpy array frames
        """
        async def frame_callback(frame: np.ndarray):
            if self.active_inference_track is not None:
                await self.active_inference_track.add_frame(frame)

        return frame_callback

    async def wait_for_client(self, timeout: Optional[float] = None):
        """
        Wait for a client to connect before starting inference.

        Args:
            timeout: Optional timeout in seconds. If None, waits indefinitely.

        Returns:
            True if client connected, False if timeout
        """
        logger.info("Waiting for client connection...")
        try:
            if timeout:
                await asyncio.wait_for(self.client_connected.wait(), timeout=timeout)
            else:
                await self.client_connected.wait()
            return True
        except asyncio.TimeoutError:
            logger.warning(f"No client connected after {timeout} seconds")
            return False

    async def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the streaming server."""
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def shutdown(self):
        """Shutdown server and cleanup resources."""
        logger.info("Shutting down streaming server...")

        # Stop video track
        if self.active_inference_track:
            await self.active_inference_track.stop()

        # Close all peer connections
        for peer_id in list(self.peers.keys()):
            await self._cleanup_peer(peer_id)

        logger.info("Server shutdown complete")


def create_server() -> StreamingServer:
    """Factory function to create streaming server instance."""
    return StreamingServer()


if __name__ == "__main__":
    """Run server standalone for testing."""
    server = create_server()

    # Example: Simulate frame generation
    async def generate_test_frames():
        """Generate test pattern frames."""
        callback = server.get_frame_callback()
        frame_count = 0

        while True:
            # Generate a test pattern
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add frame counter
            cv2.putText(img, f"Frame {frame_count}", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Add moving square
            x = int((frame_count % 640))
            cv2.rectangle(img, (x, 200), (x + 50, 250), (0, 255, 0), -1)

            await callback(img)
            frame_count += 1
            await asyncio.sleep(1/30)  # 30 FPS

    async def main():
        # Start test frame generation
        asyncio.create_task(generate_test_frames())

        # Start server
        await server.start()

    asyncio.run(main())
