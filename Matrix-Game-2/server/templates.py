"""
HTML templates for the web interface
"""


def generate_html_page(control_instructions: str) -> str:
    """
    Generate the interactive HTML page

    Args:
        control_instructions: HTML snippet with game-specific controls

    Returns:
        Complete HTML page as string
    """
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
                width: 350px;
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
            .upload-section {{
                background: #2a2a2a;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 15px;
            }}
            .upload-section h3 {{
                margin: 0 0 10px 0;
                color: #FFA500;
            }}
            .file-input-wrapper {{
                position: relative;
                overflow: hidden;
                display: inline-block;
                width: 100%;
            }}
            .file-input-wrapper input[type=file] {{
                position: absolute;
                left: -9999px;
            }}
            .file-input-label {{
                display: block;
                padding: 10px 15px;
                background: #4CAF50;
                color: white;
                text-align: center;
                border-radius: 5px;
                cursor: pointer;
                transition: background 0.3s;
            }}
            .file-input-label:hover {{
                background: #45a049;
            }}
            .file-name {{
                margin-top: 10px;
                font-size: 12px;
                color: #aaa;
                word-break: break-all;
            }}
            .upload-btn {{
                width: 100%;
                padding: 10px;
                margin-top: 10px;
                background: #FF9800;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
                font-weight: bold;
                transition: background 0.3s;
            }}
            .upload-btn:hover {{
                background: #F57C00;
            }}
            .upload-btn:disabled {{
                background: #555;
                cursor: not-allowed;
            }}
            .upload-status {{
                margin-top: 10px;
                font-size: 12px;
                color: #4CAF50;
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

                <div class="upload-section">
                    <h3>🖼️ Upload New Image</h3>
                    <div class="file-input-wrapper">
                        <input type="file" id="imageInput" accept="image/*">
                        <label for="imageInput" class="file-input-label">Choose Image</label>
                    </div>
                    <div class="file-name" id="fileName">No file chosen</div>
                    <button id="uploadBtn" class="upload-btn" disabled>Upload & Reset</button>
                    <div class="upload-status" id="uploadStatus"></div>
                </div>

                {control_instructions}
                <div class="current-action">
                    <h3>Current Action</h3>
                    <div class="action-display" id="currentAction">Waiting for input...</div>
                </div>
                <div class="info">
                    <h3>ℹ️ Info</h3>
                    <p>• Upload an image to reset the game</p>
                    <p>• Click on the video to focus</p>
                    <p>• Press keys to control the game</p>
                    <p>• New frames generated on demand</p>
                    <p>• Up to 300 latent frames (~1200 actual)</p>
                    <p>• Auto-resets when limit reached</p>
                </div>
            </div>
        </div>
        <script>
            {_get_javascript()}
        </script>
    </body>
    </html>
    """


def _get_javascript() -> str:
    """Get the JavaScript code for the web interface"""
    return """
            let ws = null;
            let mousePos = {x: 0, y: 0};
            let lastMousePos = {x: 0, y: 0};
            const videoElement = document.getElementById('videoStream');
            const wsStatusElement = document.getElementById('wsStatus');
            const fpsDisplay = document.getElementById('fpsDisplay');
            const latencyDisplay = document.getElementById('latencyDisplay');
            const currentActionDisplay = document.getElementById('currentAction');

            // Connect WebSocket
            function connectWebSocket() {
                ws = new WebSocket(`ws://${window.location.host}/ws`);

                ws.onopen = () => {
                    wsStatusElement.textContent = 'Connected';
                    wsStatusElement.className = 'connected';
                    console.log('WebSocket connected');
                };

                ws.onclose = () => {
                    wsStatusElement.textContent = 'Disconnected';
                    wsStatusElement.className = 'disconnected';
                    console.log('WebSocket disconnected, reconnecting...');
                    setTimeout(connectWebSocket, 1000);
                };

                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };

                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.fps !== undefined) {
                        fpsDisplay.textContent = data.fps.toFixed(1);
                    }
                    if (data.latency !== undefined) {
                        latencyDisplay.textContent = data.latency.toFixed(0) + 'ms';
                    }
                };
            }

            connectWebSocket();

            // Keyboard event handling
            videoElement.addEventListener('keydown', (e) => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    const action = {
                        type: 'keyboard',
                        key: e.key.toLowerCase(),
                        timestamp: Date.now()
                    };
                    ws.send(JSON.stringify(action));
                    updateActionDisplay(action);
                    e.preventDefault();
                }
            });

            function updateActionDisplay(action) {
                let displayText = '';
                if (action.type === 'keyboard') {
                    displayText = `Keyboard: ${action.key.toUpperCase()}`;
                } else if (action.type === 'mouse') {
                    displayText = `Mouse: Δx=${action.dx.toFixed(3)}, Δy=${action.dy.toFixed(3)}`;
                } else if (action.type === 'image') {
                    displayText = 'Image uploaded!';
                }
                currentActionDisplay.textContent = displayText;
            }

            // Image upload handling
            const imageInput = document.getElementById('imageInput');
            const uploadBtn = document.getElementById('uploadBtn');
            const fileName = document.getElementById('fileName');
            const uploadStatus = document.getElementById('uploadStatus');
            let selectedFile = null;

            imageInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) {
                    selectedFile = file;
                    fileName.textContent = file.name;
                    uploadBtn.disabled = false;
                } else {
                    selectedFile = null;
                    fileName.textContent = 'No file chosen';
                    uploadBtn.disabled = true;
                }
            });

            uploadBtn.addEventListener('click', () => {
                if (!selectedFile || !ws || ws.readyState !== WebSocket.OPEN) {
                    uploadStatus.textContent = 'Cannot upload: Not connected';
                    uploadStatus.style.color = '#f00';
                    return;
                }

                uploadStatus.textContent = 'Uploading...';
                uploadStatus.style.color = '#FFA500';

                const reader = new FileReader();
                reader.onload = (event) => {
                    const base64Data = event.target.result;
                    const message = {
                        type: 'image',
                        data: base64Data
                    };

                    ws.send(JSON.stringify(message));
                    uploadStatus.textContent = '✓ Image sent! Resetting...';
                    uploadStatus.style.color = '#4CAF50';
                    updateActionDisplay({type: 'image'});

                    // Reset form
                    setTimeout(() => {
                        imageInput.value = '';
                        selectedFile = null;
                        fileName.textContent = 'No file chosen';
                        uploadBtn.disabled = true;
                        uploadStatus.textContent = '';
                    }, 2000);
                };

                reader.onerror = () => {
                    uploadStatus.textContent = '✗ Failed to read image';
                    uploadStatus.style.color = '#f00';
                };

                reader.readAsDataURL(selectedFile);
            });

            // Auto-focus video element
            videoElement.focus();
    """
