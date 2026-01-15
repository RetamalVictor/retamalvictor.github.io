/**
 * DepthEngine - ONNX Runtime Web inference for depth estimation
 *
 * Supports WebGPU backend with WASM fallback.
 * Input: 256x256 RGB image
 */

// Import WebGPU build which includes both WebGPU and WASM backends
import * as ort from 'onnxruntime-web/webgpu';

export interface DepthEngineConfig {
    inputSize: number;
    inputName: string;
    outputName: string;
    normalizeMean: number[];
    normalizeStd: number[];
}

export interface DepthEngineStats {
    modelSizeMB: number;
    backend: 'webgpu' | 'wasm';
    lastLatencyMs: number;
}

/** Callback for backend status updates */
export type BackendCallback = (backend: 'webgpu' | 'wasm', warning?: string) => void;

export class DepthEngine {
    private session: ort.InferenceSession | null = null;
    private config: DepthEngineConfig;
    private stats: DepthEngineStats;
    private isReady = false;

    // Reusable buffers for preprocessing
    private inputBuffer: Float32Array | null = null;
    private resizeCanvas: HTMLCanvasElement;
    private resizeCtx: CanvasRenderingContext2D;
    private sourceCanvas: HTMLCanvasElement | null = null;
    private sourceCtx: CanvasRenderingContext2D | null = null;

    private constructor(config: DepthEngineConfig) {
        this.config = config;
        this.stats = {
            modelSizeMB: 0,
            backend: 'wasm',
            lastLatencyMs: 0
        };

        // Create reusable canvas for resizing
        this.resizeCanvas = document.createElement('canvas');
        this.resizeCanvas.width = config.inputSize;
        this.resizeCanvas.height = config.inputSize;
        this.resizeCtx = this.resizeCanvas.getContext('2d', { willReadFrequently: true })!;

        // Pre-allocate input buffer
        this.inputBuffer = new Float32Array(3 * config.inputSize * config.inputSize);
    }

    /**
     * Create and initialize the depth engine
     * @param modelPath - Path to model directory
     * @param onBackend - Callback when backend is determined (for UI warnings)
     */
    static async create(modelPath: string, onBackend?: BackendCallback): Promise<DepthEngine> {
        // Load config
        const configResponse = await fetch(`${modelPath}/config.json`);
        const config: DepthEngineConfig = await configResponse.json();

        const engine = new DepthEngine(config);

        const modelUrl = `${modelPath}/midas_v21_small_256.onnx`;

        // First, get model size
        try {
            const modelResponse = await fetch(modelUrl);
            const modelBlob = await modelResponse.blob();
            engine.stats.modelSizeMB = modelBlob.size / (1024 * 1024);
        } catch (e) {
            console.warn('[DepthEngine] Could not determine model size:', e);
        }

        // Try WebGPU first
        let session: ort.InferenceSession | null = null;
        let backend: 'webgpu' | 'wasm' = 'wasm';

        // Check for WebGPU support
        const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;

        if (hasWebGPU) {
            try {
                // Request adapter to verify WebGPU actually works
                const adapter = await (navigator as any).gpu.requestAdapter();
                if (adapter) {
                    console.log('[DepthEngine] Attempting WebGPU backend...');
                    session = await ort.InferenceSession.create(modelUrl, {
                        executionProviders: ['webgpu'],
                        graphOptimizationLevel: 'all'
                    });
                    backend = 'webgpu';
                    console.log('[DepthEngine] WebGPU backend initialized');
                    onBackend?.('webgpu');
                }
            } catch (e) {
                console.warn('[DepthEngine] WebGPU failed:', e);
                session = null;
            }
        }

        // Fallback to WASM
        if (!session) {
            // Check if multi-threading is available (requires COOP/COEP headers)
            const canUseThreads = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;

            if (canUseThreads) {
                ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
                console.log(`[DepthEngine] Using WASM backend (${ort.env.wasm.numThreads} threads)`);
            } else {
                ort.env.wasm.numThreads = 1;
                console.log('[DepthEngine] Using WASM backend (single-threaded, no COOP/COEP)');
            }

            session = await ort.InferenceSession.create(modelUrl, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });
            backend = 'wasm';

            // Notify UI about fallback
            if (!canUseThreads) {
                const warning = hasWebGPU
                    ? 'WebGPU failed. Using single-threaded CPU (slower).'
                    : 'WebGPU not supported. Using single-threaded CPU (slower).';
                onBackend?.('wasm', warning);
            } else {
                // Multi-threaded WASM, no warning needed but still not as fast as WebGPU
                onBackend?.('wasm');
            }
        }

        engine.session = session;
        engine.stats.backend = backend;
        engine.isReady = true;

        return engine;
    }

    /**
     * Run depth inference on an image
     * @param imageData - RGB image data from canvas
     * @returns Float32Array of depth values (H x W)
     */
    async predict(imageData: ImageData): Promise<Float32Array> {
        if (!this.session || !this.isReady) {
            throw new Error('DepthEngine not initialized');
        }

        const startTime = performance.now();

        // Preprocess: resize and normalize
        const tensor = this.preprocess(imageData);

        // Run inference
        const feeds: Record<string, ort.Tensor> = {};
        feeds[this.config.inputName] = tensor;

        const results = await this.session.run(feeds);

        // Get depth output
        const depthTensor = results[this.config.outputName];
        const depthData = depthTensor.data as Float32Array;

        // Update stats
        this.stats.lastLatencyMs = performance.now() - startTime;

        // Cleanup tensor
        tensor.dispose();

        return depthData;
    }

    /**
     * Preprocess image: resize to input size, normalize, convert to CHW
     */
    private preprocess(imageData: ImageData): ort.Tensor {
        const { inputSize, normalizeMean, normalizeStd } = this.config;

        // Reuse source canvas (resize if needed)
        if (!this.sourceCanvas || this.sourceCanvas.width !== imageData.width || this.sourceCanvas.height !== imageData.height) {
            this.sourceCanvas = document.createElement('canvas');
            this.sourceCanvas.width = imageData.width;
            this.sourceCanvas.height = imageData.height;
            this.sourceCtx = this.sourceCanvas.getContext('2d')!;
        }
        this.sourceCtx!.putImageData(imageData, 0, 0);

        // Draw resized
        this.resizeCtx.drawImage(this.sourceCanvas, 0, 0, inputSize, inputSize);

        // Get resized pixel data
        const resizedData = this.resizeCtx.getImageData(0, 0, inputSize, inputSize);
        const pixels = resizedData.data;

        // Convert to CHW format with normalization
        const buffer = this.inputBuffer!;
        const channelSize = inputSize * inputSize;

        for (let i = 0; i < channelSize; i++) {
            const pixelIdx = i * 4;
            // R channel
            buffer[i] = (pixels[pixelIdx] / 255 - normalizeMean[0]) / normalizeStd[0];
            // G channel
            buffer[channelSize + i] = (pixels[pixelIdx + 1] / 255 - normalizeMean[1]) / normalizeStd[1];
            // B channel
            buffer[2 * channelSize + i] = (pixels[pixelIdx + 2] / 255 - normalizeMean[2]) / normalizeStd[2];
        }

        return new ort.Tensor('float32', buffer, [1, 3, inputSize, inputSize]);
    }

    /**
     * Get engine statistics
     */
    getStats(): DepthEngineStats {
        return { ...this.stats };
    }

    /**
     * Check if engine is ready
     */
    ready(): boolean {
        return this.isReady;
    }

    /**
     * Get input size
     */
    getInputSize(): number {
        return this.config.inputSize;
    }

    /**
     * Cleanup resources
     */
    destroy(): void {
        if (this.session) {
            // Session cleanup - ORT handles this
            this.session = null;
        }
        this.inputBuffer = null;
        this.isReady = false;
    }
}
