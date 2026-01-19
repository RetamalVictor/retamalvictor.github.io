/**
 * DepthDemo - Real-time depth estimation from webcam
 *
 * Features:
 * - Live webcam feed with depth estimation
 * - Side-by-side RGB/depth visualization
 * - 3D point cloud visualization
 * - Performance metrics (FPS, latency, backend)
 */

import { DepthEngine } from './DepthEngine.js';
import { PointCloudViewer } from './PointCloudViewer.js';

export interface DepthDemoConfig {
    containerId: string;
    modelPath: string;
}

type DemoStatus = 'loading' | 'ready' | 'running' | 'error' | 'no-camera' | 'requesting-camera';
type ViewMode = '2d' | '3d';

interface DemoState {
    status: DemoStatus;
    fps: number;
    latencyMs: number;
    modelSizeMB: number;
    backend: 'webgpu' | 'wasm';
    viewMode: ViewMode;
    errorMessage?: string;
    backendWarning?: string;
}

export class DepthDemo {
    private container: HTMLElement;
    private config: DepthDemoConfig;
    private engine: DepthEngine | null = null;
    private state: DemoState;
    private isDestroyed = false;

    // Video/canvas elements
    private video: HTMLVideoElement | null = null;
    private videoCanvas: HTMLCanvasElement | null = null;
    private videoCtx: CanvasRenderingContext2D | null = null;
    private depthCanvas: HTMLCanvasElement | null = null;
    private depthCtx: CanvasRenderingContext2D | null = null;

    // Animation
    private animationId: number | null = null;
    private frameCount = 0;
    private fpsUpdateTime = 0;

    // Async inference state
    private isInferenceRunning = false;
    private latestDepthMap: Float32Array | null = null;

    // Reusable canvas for rendering (avoid per-frame allocation)
    private tempRenderCanvas: HTMLCanvasElement | null = null;
    private tempRenderCtx: CanvasRenderingContext2D | null = null;

    // Point cloud viewer
    private pointCloudViewer: PointCloudViewer | null = null;
    private depthOutputSize = 256;

    constructor(config: DepthDemoConfig) {
        const container = document.getElementById(config.containerId);
        if (!container) {
            throw new Error(`Container #${config.containerId} not found`);
        }

        this.container = container;
        this.config = config;
        this.state = {
            status: 'loading',
            fps: 0,
            latencyMs: 0,
            modelSizeMB: 0,
            backend: 'wasm',
            viewMode: '2d'
        };

        this.init();
    }

    private async init(): Promise<void> {
        this.render();

        try {
            // Initialize depth engine first
            this.updateStatus('loading', 'Loading depth model...');
            this.engine = await DepthEngine.create(this.config.modelPath, (backend, warning) => {
                this.state.backend = backend;
                if (warning) {
                    this.state.backendWarning = warning;
                }
            });

            // Check if destroyed during model loading
            if (this.isDestroyed) {
                this.engine?.destroy();
                this.engine = null;
                return;
            }

            const stats = this.engine.getStats();
            this.state.modelSizeMB = stats.modelSizeMB;
            this.state.backend = stats.backend;

            // Request camera access
            this.updateStatus('requesting-camera', 'Requesting camera access...');
            await this.setupCamera();

            // Check if destroyed during camera setup
            if (this.isDestroyed) {
                // Clean up camera stream if it was acquired
                if (this.video && this.video.srcObject) {
                    const stream = this.video.srcObject as MediaStream;
                    stream.getTracks().forEach(track => track.stop());
                    this.video.srcObject = null;
                }
                return;
            }

            this.state.status = 'ready';
            this.render();

            // Start inference loop
            this.startLoop();

        } catch (error) {
            // Ignore errors if destroyed during init
            if (this.isDestroyed) return;

            console.error('[DepthDemo] Initialization failed:', error);
            const message = error instanceof Error ? error.message : 'Unknown error';

            if (message.includes('NotAllowedError') || message.includes('Permission')) {
                this.updateStatus('no-camera', 'Camera access denied. Please allow camera access and reload.');
            } else if (message.includes('NotFoundError') || message.includes('No camera')) {
                this.updateStatus('no-camera', 'No camera found. Please connect a camera and reload.');
            } else {
                this.updateStatus('error', message);
            }
        }
    }

    private updateStatus(status: DemoStatus, message?: string): void {
        this.state.status = status;
        if (message) {
            this.state.errorMessage = message;
        }
        this.render();
    }

    private async setupCamera(): Promise<void> {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            }
        });

        this.video = document.createElement('video');
        this.video.srcObject = stream;
        this.video.playsInline = true;
        this.video.muted = true;

        await new Promise<void>((resolve) => {
            this.video!.onloadedmetadata = () => {
                this.video!.play();
                resolve();
            };
        });
    }

    private startLoop(): void {
        if (this.isDestroyed) return;

        this.state.status = 'running';
        this.fpsUpdateTime = performance.now();
        this.runLoop();
    }

    private async runLoop(): Promise<void> {
        if (this.isDestroyed || this.state.status !== 'running') return;

        const now = performance.now();

        // Capture frame from video
        if (this.video && this.videoCanvas && this.videoCtx && this.engine) {
            // Draw video to canvas (always, for smooth video display)
            this.videoCtx.drawImage(
                this.video,
                0, 0,
                this.videoCanvas.width,
                this.videoCanvas.height
            );

            // Start new inference if none is running
            if (!this.isInferenceRunning) {
                this.isInferenceRunning = true;

                // Get image data for inference
                const imageData = this.videoCtx.getImageData(
                    0, 0,
                    this.videoCanvas.width,
                    this.videoCanvas.height
                );

                // Run inference asynchronously (non-blocking)
                this.engine.predict(imageData).then((depthMap) => {
                    this.latestDepthMap = depthMap;

                    // Update metrics
                    const stats = this.engine!.getStats();
                    this.state.latencyMs = stats.lastLatencyMs;

                    this.isInferenceRunning = false;
                }).catch((e) => {
                    console.error('[DepthDemo] Inference error:', e);
                    this.isInferenceRunning = false;
                });
            }

            // Render latest depth map (if available)
            if (this.latestDepthMap) {
                if (this.state.viewMode === '2d') {
                    this.renderDepth(this.latestDepthMap);
                } else if (this.pointCloudViewer) {
                    this.pointCloudViewer.update(this.latestDepthMap, this.depthOutputSize, this.depthOutputSize);
                }
            }
        }

        // Update FPS counter
        this.frameCount++;
        if (now - this.fpsUpdateTime >= 1000) {
            this.state.fps = this.frameCount;
            this.frameCount = 0;
            this.fpsUpdateTime = now;
            this.updateMetricsDisplay();
        }

        // Schedule next frame
        this.animationId = requestAnimationFrame(() => this.runLoop());
    }

    private renderDepth(depthMap: Float32Array): void {
        if (!this.depthCanvas || !this.depthCtx) return;

        // Derive output dimensions from depth array (model outputs 378x378, not 384x384)
        const outputSize = Math.round(Math.sqrt(depthMap.length));
        const width = this.depthCanvas.width;
        const height = this.depthCanvas.height;

        // Initialize reusable temp canvas if needed
        if (!this.tempRenderCanvas || this.tempRenderCanvas.width !== outputSize) {
            this.tempRenderCanvas = document.createElement('canvas');
            this.tempRenderCanvas.width = outputSize;
            this.tempRenderCanvas.height = outputSize;
            this.tempRenderCtx = this.tempRenderCanvas.getContext('2d')!;
        }

        // Normalize depth for visualization
        let minDepth = Infinity;
        let maxDepth = -Infinity;
        for (let i = 0; i < depthMap.length; i++) {
            if (depthMap[i] < minDepth) minDepth = depthMap[i];
            if (depthMap[i] > maxDepth) maxDepth = depthMap[i];
        }
        const range = maxDepth - minDepth || 1;

        // Create output image with grayscale colormap
        const imageData = this.tempRenderCtx!.createImageData(outputSize, outputSize);

        for (let i = 0; i < depthMap.length; i++) {
            // Normalize to 0-1 (model outputs disparity: close = high values = bright)
            const normalized = (depthMap[i] - minDepth) / range;
            const value = Math.floor(normalized * 255);

            imageData.data[i * 4] = value;
            imageData.data[i * 4 + 1] = value;
            imageData.data[i * 4 + 2] = value;
            imageData.data[i * 4 + 3] = 255;
        }

        // Draw to reusable temp canvas, then scale to output
        this.tempRenderCtx!.putImageData(imageData, 0, 0);
        this.depthCtx.drawImage(this.tempRenderCanvas, 0, 0, width, height);
    }

    private updateMetricsDisplay(): void {
        const fpsEl = this.container.querySelector('#depth-fps');
        const latencyEl = this.container.querySelector('#depth-latency');

        if (fpsEl) fpsEl.textContent = `${this.state.fps}`;
        if (latencyEl) latencyEl.textContent = `${this.state.latencyMs.toFixed(1)}`;
    }

    private render(): void {
        const { status, errorMessage, modelSizeMB, backend } = this.state;

        if (status === 'loading' || status === 'requesting-camera') {
            this.container.innerHTML = `
                <div class="flex flex-col items-center justify-center h-full text-center p-4">
                    <div class="animate-pulse mb-4">
                        <svg class="w-12 h-12 text-accent-cyan" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
                        </svg>
                    </div>
                    <p class="text-gray-400 text-sm">${errorMessage || 'Initializing...'}</p>
                </div>
            `;
            return;
        }

        if (status === 'error' || status === 'no-camera') {
            this.container.innerHTML = `
                <div class="flex flex-col items-center justify-center h-full text-center p-4">
                    <div class="mb-4 text-red-400">
                        <svg class="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
                        </svg>
                    </div>
                    <p class="text-red-400 text-sm mb-2">${errorMessage || 'An error occurred'}</p>
                    <button onclick="location.reload()" class="text-xs text-accent-cyan hover:underline">
                        Reload page
                    </button>
                </div>
            `;
            return;
        }

        const { viewMode, backendWarning } = this.state;

        // Ready/running state - show view based on mode
        this.container.innerHTML = `
            <div class="h-full flex flex-col">
                ${backendWarning ? `
                <!-- CPU fallback warning -->
                <div class="mb-2 px-3 py-2 bg-yellow-900/30 border border-yellow-600/50 rounded text-xs text-yellow-400 flex items-center gap-2">
                    <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
                    </svg>
                    <span>${backendWarning}</span>
                </div>
                ` : ''}

                <!-- View toggle -->
                <div class="flex items-center gap-2 mb-2">
                    <button id="view-2d-btn" class="px-2 py-1 text-xs rounded ${viewMode === '2d' ? 'bg-accent-cyan text-dark-bg' : 'bg-dark-bg text-gray-400 hover:text-white'}">
                        2D View
                    </button>
                    <button id="view-3d-btn" class="px-2 py-1 text-xs rounded ${viewMode === '3d' ? 'bg-accent-cyan text-dark-bg' : 'bg-dark-bg text-gray-400 hover:text-white'}">
                        3D Point Cloud
                    </button>
                    ${viewMode === '3d' ? '<span class="text-xs text-gray-500 ml-2">Drag to rotate, scroll to zoom</span>' : ''}
                </div>

                <!-- Content area -->
                ${viewMode === '2d' ? `
                    <div class="flex-1 flex gap-2 min-h-0">
                        <!-- RGB feed -->
                        <div class="flex-1 flex flex-col min-w-0">
                            <div class="text-xs text-gray-500 mb-1 px-1">RGB</div>
                            <div class="flex-1 bg-dark-bg rounded overflow-hidden relative">
                                <canvas id="depth-video-canvas" class="w-full h-full object-contain"></canvas>
                            </div>
                        </div>
                        <!-- Depth feed -->
                        <div class="flex-1 flex flex-col min-w-0">
                            <div class="text-xs text-gray-500 mb-1 px-1">Depth (relative)</div>
                            <div class="flex-1 bg-dark-bg rounded overflow-hidden relative">
                                <canvas id="depth-depth-canvas" class="w-full h-full object-contain"></canvas>
                            </div>
                        </div>
                    </div>
                ` : `
                    <div class="flex-1 flex gap-2 min-h-0">
                        <!-- RGB feed (smaller) -->
                        <div class="w-1/4 flex flex-col min-w-0">
                            <div class="text-xs text-gray-500 mb-1 px-1">RGB</div>
                            <div class="flex-1 bg-dark-bg rounded overflow-hidden relative">
                                <canvas id="depth-video-canvas" class="w-full h-full object-contain"></canvas>
                            </div>
                        </div>
                        <!-- Point cloud -->
                        <div class="flex-1 flex flex-col min-w-0">
                            <div class="text-xs text-gray-500 mb-1 px-1">Point Cloud (relative depth)</div>
                            <div id="pointcloud-container" class="flex-1 bg-dark-bg rounded overflow-hidden relative"></div>
                        </div>
                    </div>
                `}

                <!-- Metrics bar -->
                <div class="mt-2 flex items-center justify-between text-xs bg-dark-bg/50 rounded px-2 py-1">
                    <div class="flex gap-4">
                        <span class="text-gray-500">FPS: <span id="depth-fps" class="text-accent-cyan font-mono">0</span></span>
                        <span class="text-gray-500">Latency: <span id="depth-latency" class="text-accent-cyan font-mono">0</span>ms</span>
                    </div>
                    <div class="flex gap-4">
                        <span class="text-gray-500">Model: <span class="text-gray-400 font-mono">${modelSizeMB.toFixed(1)}MB</span></span>
                        <span class="text-gray-500">Backend: <span class="${backend === 'webgpu' ? 'text-green-400' : 'text-yellow-400'} font-mono">${backend.toUpperCase()}</span></span>
                    </div>
                </div>
            </div>
        `;

        // Setup canvases and point cloud
        this.setupCanvases();
        this.setupViewToggle();

        if (viewMode === '3d') {
            this.setupPointCloud();
        }
    }

    private setupCanvases(): void {
        const inputSize = this.engine?.getInputSize() || 256;
        this.depthOutputSize = inputSize;

        this.videoCanvas = this.container.querySelector('#depth-video-canvas') as HTMLCanvasElement;
        this.depthCanvas = this.container.querySelector('#depth-depth-canvas') as HTMLCanvasElement;

        if (this.videoCanvas) {
            // Use video dimensions or fallback
            const videoWidth = this.video?.videoWidth || 640;
            const videoHeight = this.video?.videoHeight || 480;
            this.videoCanvas.width = videoWidth;
            this.videoCanvas.height = videoHeight;
            this.videoCtx = this.videoCanvas.getContext('2d', { willReadFrequently: true });
        }

        if (this.depthCanvas) {
            this.depthCanvas.width = inputSize;
            this.depthCanvas.height = inputSize;
            this.depthCtx = this.depthCanvas.getContext('2d');
        }
    }

    private setupViewToggle(): void {
        const btn2d = this.container.querySelector('#view-2d-btn');
        const btn3d = this.container.querySelector('#view-3d-btn');

        btn2d?.addEventListener('click', () => {
            if (this.state.viewMode !== '2d') {
                this.state.viewMode = '2d';
                this.destroyPointCloud();
                this.render();
            }
        });

        btn3d?.addEventListener('click', () => {
            if (this.state.viewMode !== '3d') {
                this.state.viewMode = '3d';
                this.render();
            }
        });
    }

    private setupPointCloud(): void {
        const container = this.container.querySelector('#pointcloud-container') as HTMLElement;
        if (!container) return;

        // Destroy existing point cloud if any
        this.destroyPointCloud();

        // Create new point cloud viewer
        this.pointCloudViewer = new PointCloudViewer(container, {
            subsample: 2,
            depthScale: 0.5,
            smoothingFactor: 0.3
        });
    }

    private destroyPointCloud(): void {
        if (this.pointCloudViewer) {
            this.pointCloudViewer.destroy();
            this.pointCloudViewer = null;
        }
    }

    /**
     * Reset the demo
     */
    public reset(): void {
        // Just re-render, loop continues
        this.render();
    }

    /**
     * Cleanup resources
     */
    public destroy(): void {
        this.isDestroyed = true;

        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }

        // Stop video stream
        if (this.video && this.video.srcObject) {
            const stream = this.video.srcObject as MediaStream;
            stream.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
        }

        // Cleanup point cloud
        this.destroyPointCloud();

        // Cleanup engine
        if (this.engine) {
            this.engine.destroy();
            this.engine = null;
        }

        // Clear container
        this.container.innerHTML = '';
    }
}
