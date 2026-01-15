/**
 * DepthDemo - Real-time depth estimation from webcam
 *
 * Features:
 * - Live webcam feed with depth estimation
 * - Side-by-side RGB/depth visualization
 * - Performance metrics (FPS, latency, backend)
 * - Temporal smoothing for point cloud (coming Day 3)
 */

import { DepthEngine } from './DepthEngine.js';

export interface DepthDemoConfig {
    containerId: string;
    modelPath: string;
}

type DemoStatus = 'loading' | 'ready' | 'running' | 'error' | 'no-camera' | 'requesting-camera';

interface DemoState {
    status: DemoStatus;
    fps: number;
    latencyMs: number;
    modelSizeMB: number;
    backend: 'webgpu' | 'wasm';
    errorMessage?: string;
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
            backend: 'wasm'
        };

        this.init();
    }

    private async init(): Promise<void> {
        this.render();

        try {
            // Initialize depth engine first
            this.updateStatus('loading', 'Loading depth model...');
            this.engine = await DepthEngine.create(this.config.modelPath);

            const stats = this.engine.getStats();
            this.state.modelSizeMB = stats.modelSizeMB;
            this.state.backend = stats.backend;

            // Request camera access
            this.updateStatus('requesting-camera', 'Requesting camera access...');
            await this.setupCamera();

            this.state.status = 'ready';
            this.render();

            // Start inference loop
            this.startLoop();

        } catch (error) {
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
            // Draw video to canvas
            this.videoCtx.drawImage(
                this.video,
                0, 0,
                this.videoCanvas.width,
                this.videoCanvas.height
            );

            // Get image data
            const imageData = this.videoCtx.getImageData(
                0, 0,
                this.videoCanvas.width,
                this.videoCanvas.height
            );

            try {
                // Run depth inference
                const depthMap = await this.engine.predict(imageData);

                // Render depth visualization
                this.renderDepth(depthMap);

                // Update metrics
                const stats = this.engine.getStats();
                this.state.latencyMs = stats.lastLatencyMs;

            } catch (e) {
                console.error('[DepthDemo] Inference error:', e);
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

        // Normalize depth for visualization
        let minDepth = Infinity;
        let maxDepth = -Infinity;
        for (let i = 0; i < depthMap.length; i++) {
            if (depthMap[i] < minDepth) minDepth = depthMap[i];
            if (depthMap[i] > maxDepth) maxDepth = depthMap[i];
        }
        const range = maxDepth - minDepth || 1;

        // Create output image with grayscale colormap
        const imageData = this.depthCtx.createImageData(outputSize, outputSize);

        for (let i = 0; i < depthMap.length; i++) {
            // Normalize to 0-1, invert so close = bright
            const normalized = 1 - (depthMap[i] - minDepth) / range;
            const value = Math.floor(normalized * 255);

            imageData.data[i * 4] = value;
            imageData.data[i * 4 + 1] = value;
            imageData.data[i * 4 + 2] = value;
            imageData.data[i * 4 + 3] = 255;
        }

        // Draw at output size first, then scale to canvas
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = outputSize;
        tempCanvas.height = outputSize;
        const tempCtx = tempCanvas.getContext('2d')!;
        tempCtx.putImageData(imageData, 0, 0);

        // Draw scaled to output canvas
        this.depthCtx.drawImage(tempCanvas, 0, 0, width, height);
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

        // Ready/running state - show dual canvas view
        this.container.innerHTML = `
            <div class="h-full flex flex-col">
                <!-- Video feeds -->
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

        // Setup canvases
        this.setupCanvases();
    }

    private setupCanvases(): void {
        const inputSize = this.engine?.getInputSize() || 384;

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

        // Cleanup engine
        if (this.engine) {
            this.engine.destroy();
            this.engine = null;
        }

        // Clear container
        this.container.innerHTML = '';
    }
}
