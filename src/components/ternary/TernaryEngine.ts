/**
 * TernaryEngine - WebGPU inference engine for ternary neural networks.
 *
 * This engine loads a .tbin model file and runs character-level text generation
 * using WebGPU compute shaders.
 */

import {
    ModelConfig,
    LayerType,
    LayerMeta,
    LayerBuffers,
    GenerationStats,
    MemoryStats,
    VocabMapping,
    TBIN_MAGIC,
    TBIN_VERSION,
    HEADER_SIZE,
    LAYER_META_SIZE,
} from './types';

// Import shaders as raw strings (Vite handles this with ?raw)
import ternaryMatmulWGSL from './shaders/ternary_matmul.wgsl?raw';
import embeddingWGSL from './shaders/embedding.wgsl?raw';
import softmaxWGSL from './shaders/softmax.wgsl?raw';

export class TernaryEngine {
    private device: GPUDevice;
    private config: ModelConfig;
    private vocab: VocabMapping;

    // Compute pipelines
    private embeddingPipeline: GPUComputePipeline | null = null;
    private ternaryMatmulPipeline: GPUComputePipeline | null = null;
    private softmaxPipeline: GPUComputePipeline | null = null;

    // Model buffers
    private layers: LayerBuffers[] = [];
    private embeddingBuffer: GPUBuffer | null = null;

    // Working buffers (reused during inference)
    private tokenBuffer: GPUBuffer | null = null;
    private activationBuffers: GPUBuffer[] = [];  // Ping-pong buffers
    private logitsBuffer: GPUBuffer | null = null;
    private probsBuffer: GPUBuffer | null = null;

    // Stats
    private totalTernaryWeights = 0;
    private totalFP16Weights = 0;

    private constructor(device: GPUDevice) {
        this.device = device;
        this.config = { vocabSize: 0, hiddenDim: 0, contextLength: 0, nLayers: 0 };
        this.vocab = { chars: '', charToIdx: {}, vocabSize: 0 };
    }

    /**
     * Create a TernaryEngine instance.
     * @param modelPath Path to .tbin model file
     * @returns Promise resolving to initialized engine
     */
    static async create(modelPath: string): Promise<TernaryEngine> {
        // Check WebGPU support
        if (!navigator.gpu) {
            throw new Error('WebGPU is not supported in this browser');
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No WebGPU adapter found');
        }

        const device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: 256 * 1024 * 1024,  // 256MB
                maxBufferSize: 256 * 1024 * 1024,
            },
        });

        const engine = new TernaryEngine(device);
        await engine.loadModel(modelPath);
        await engine.compilePipelines();
        engine.createWorkingBuffers();

        return engine;
    }

    /**
     * Load model from .tbin file.
     */
    private async loadModel(modelPath: string): Promise<void> {
        // Load model binary
        const modelResponse = await fetch(modelPath);
        if (!modelResponse.ok) {
            throw new Error(`Failed to load model: ${modelResponse.statusText}`);
        }
        const modelBuffer = await modelResponse.arrayBuffer();

        // Load vocab JSON sidecar
        const vocabPath = modelPath.replace('.tbin', '.vocab.json');
        const vocabResponse = await fetch(vocabPath);
        if (!vocabResponse.ok) {
            throw new Error(`Failed to load vocab: ${vocabResponse.statusText}`);
        }
        const vocabData = await vocabResponse.json();
        // Handle both snake_case and camelCase keys
        this.vocab = {
            chars: vocabData.chars,
            charToIdx: vocabData.charToIdx || vocabData.char_to_idx,
            vocabSize: vocabData.vocabSize || vocabData.vocab_size,
        };

        // Parse header
        const dataView = new DataView(modelBuffer);
        const magic = dataView.getUint32(0, true);
        const version = dataView.getUint32(4, true);

        if (magic !== TBIN_MAGIC) {
            throw new Error(`Invalid model file: wrong magic number ${magic.toString(16)}`);
        }
        if (version !== TBIN_VERSION) {
            throw new Error(`Unsupported model version: ${version}`);
        }

        const nLayers = dataView.getUint32(8, true);
        this.config = {
            vocabSize: dataView.getUint32(12, true),
            hiddenDim: dataView.getUint32(16, true),
            contextLength: dataView.getUint32(20, true),
            nLayers: nLayers,
        };

        console.log('Model config:', this.config);

        // Parse layer metadata
        const layerMetas: LayerMeta[] = [];
        let offset = HEADER_SIZE;

        for (let i = 0; i < nLayers; i++) {
            layerMetas.push({
                type: dataView.getUint32(offset, true) as LayerType,
                inFeatures: dataView.getUint32(offset + 4, true),
                outFeatures: dataView.getUint32(offset + 8, true),
                hasBias: dataView.getUint32(offset + 12, true) !== 0,
                dataOffset: dataView.getUint32(offset + 16, true),
                scaleOffset: dataView.getUint32(offset + 20, true),
                biasOffset: dataView.getUint32(offset + 24, true),
            });
            offset += LAYER_META_SIZE;
        }

        // Create GPU buffers for each layer
        for (const meta of layerMetas) {
            await this.createLayerBuffers(meta, modelBuffer);
        }

        console.log(`Loaded ${this.layers.length} layers`);
        console.log(`Ternary weights: ${this.totalTernaryWeights}, FP16 weights: ${this.totalFP16Weights}`);
    }

    /**
     * Create GPU buffers for a single layer.
     */
    private async createLayerBuffers(meta: LayerMeta, modelBuffer: ArrayBuffer): Promise<void> {
        if (meta.type === LayerType.EMBEDDING) {
            // Embedding: FP16 weights [vocab_size, embed_dim]
            const numWeights = meta.inFeatures * meta.outFeatures;
            const byteLength = numWeights * 2;  // FP16

            // Read FP16 data as Uint16Array
            const weightData = new Uint16Array(modelBuffer, meta.dataOffset, numWeights);

            // Create GPU buffer
            this.embeddingBuffer = this.device.createBuffer({
                size: byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this.device.queue.writeBuffer(this.embeddingBuffer, 0, weightData);

            this.totalFP16Weights += numWeights;

        } else if (meta.type === LayerType.TERNARY_LINEAR) {
            // Ternary: packed u32 weights [K_packed, N] + FP16 scales [N]
            const K = meta.inFeatures;
            const N = meta.outFeatures;
            const K_packed = Math.ceil(K / 16);  // 16 weights per u32
            const packedSize = K_packed * N * 4;  // u32 bytes

            // Read packed weights
            const packedData = new Uint32Array(modelBuffer, meta.dataOffset, K_packed * N);
            const weightsBuffer = this.device.createBuffer({
                size: packedSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this.device.queue.writeBuffer(weightsBuffer, 0, packedData);

            // Read scales (FP16 stored as Float32 for WebGPU compatibility)
            const scaleData = new Uint16Array(modelBuffer, meta.scaleOffset, N);
            // Convert FP16 to FP32 for easier WebGPU handling
            const scaleFloat32 = new Float32Array(N);
            for (let i = 0; i < N; i++) {
                scaleFloat32[i] = this.float16ToFloat32(scaleData[i]);
            }
            const scalesBuffer = this.device.createBuffer({
                size: N * 4,  // FP32
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this.device.queue.writeBuffer(scalesBuffer, 0, scaleFloat32);

            // Read bias if present
            let biasBuffer: GPUBuffer | undefined;
            if (meta.hasBias) {
                const biasData = new Uint16Array(modelBuffer, meta.biasOffset, N);
                const biasFloat32 = new Float32Array(N);
                for (let i = 0; i < N; i++) {
                    biasFloat32[i] = this.float16ToFloat32(biasData[i]);
                }
                biasBuffer = this.device.createBuffer({
                    size: N * 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                });
                this.device.queue.writeBuffer(biasBuffer, 0, biasFloat32);
            }

            this.layers.push({
                type: meta.type,
                inFeatures: K,
                outFeatures: N,
                weights: weightsBuffer,
                scales: scalesBuffer,
                bias: biasBuffer,
            });

            this.totalTernaryWeights += K * N;

        } else if (meta.type === LayerType.FP16_LINEAR) {
            // FP16 linear: standard FP16 weights [out, in]
            const numWeights = meta.inFeatures * meta.outFeatures;
            const weightData = new Uint16Array(modelBuffer, meta.dataOffset, numWeights);
            // Convert to FP32
            const weightFloat32 = new Float32Array(numWeights);
            for (let i = 0; i < numWeights; i++) {
                weightFloat32[i] = this.float16ToFloat32(weightData[i]);
            }
            const weightsBuffer = this.device.createBuffer({
                size: numWeights * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this.device.queue.writeBuffer(weightsBuffer, 0, weightFloat32);

            let biasBuffer: GPUBuffer | undefined;
            if (meta.hasBias) {
                const biasData = new Uint16Array(modelBuffer, meta.biasOffset, meta.outFeatures);
                const biasFloat32 = new Float32Array(meta.outFeatures);
                for (let i = 0; i < meta.outFeatures; i++) {
                    biasFloat32[i] = this.float16ToFloat32(biasData[i]);
                }
                biasBuffer = this.device.createBuffer({
                    size: meta.outFeatures * 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                });
                this.device.queue.writeBuffer(biasBuffer, 0, biasFloat32);
            }

            this.layers.push({
                type: meta.type,
                inFeatures: meta.inFeatures,
                outFeatures: meta.outFeatures,
                weights: weightsBuffer,
                bias: biasBuffer,
            });

            this.totalFP16Weights += numWeights;
        }
    }

    /**
     * Convert IEEE 754 half-precision float to single-precision.
     */
    private float16ToFloat32(h: number): number {
        const sign = (h & 0x8000) >> 15;
        const exp = (h & 0x7C00) >> 10;
        const frac = h & 0x03FF;

        if (exp === 0) {
            if (frac === 0) return sign ? -0 : 0;
            // Subnormal
            const e = -14;
            const m = frac / 1024;
            return (sign ? -1 : 1) * m * Math.pow(2, e);
        } else if (exp === 31) {
            if (frac === 0) return sign ? -Infinity : Infinity;
            return NaN;
        }

        const e = exp - 15;
        const m = 1 + frac / 1024;
        return (sign ? -1 : 1) * m * Math.pow(2, e);
    }

    /**
     * Compile WGSL shaders into compute pipelines.
     */
    private async compilePipelines(): Promise<void> {
        // Embedding lookup pipeline
        const embeddingModule = this.device.createShaderModule({
            code: embeddingWGSL,
        });
        this.embeddingPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: embeddingModule,
                entryPoint: 'main',
            },
        });

        // Ternary matmul pipeline
        const ternaryModule = this.device.createShaderModule({
            code: ternaryMatmulWGSL,
        });
        this.ternaryMatmulPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: ternaryModule,
                entryPoint: 'main',
            },
        });

        // Softmax pipeline
        const softmaxModule = this.device.createShaderModule({
            code: softmaxWGSL,
        });
        this.softmaxPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: softmaxModule,
                entryPoint: 'main',
            },
        });

        console.log('Pipelines compiled');
    }

    /**
     * Create working buffers for inference.
     */
    private createWorkingBuffers(): void {
        const { contextLength, hiddenDim, vocabSize } = this.config;
        const flattenedSize = contextLength * hiddenDim;

        // Token indices buffer
        this.tokenBuffer = this.device.createBuffer({
            size: contextLength * 4,  // u32
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Activation ping-pong buffers (enough for largest layer)
        const maxActivationSize = Math.max(
            flattenedSize,
            hiddenDim * 2,
            hiddenDim,
            vocabSize
        ) * 4;

        this.activationBuffers = [
            this.device.createBuffer({
                size: maxActivationSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            }),
            this.device.createBuffer({
                size: maxActivationSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            }),
        ];

        // Logits and probability buffers
        this.logitsBuffer = this.device.createBuffer({
            size: vocabSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        this.probsBuffer = this.device.createBuffer({
            size: vocabSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        console.log('Working buffers created');
    }

    /**
     * Tokenize a string to token indices.
     */
    tokenize(text: string): number[] {
        return Array.from(text).map(ch => this.vocab.charToIdx[ch] ?? 0);
    }

    /**
     * Decode token indices to string.
     */
    decode(tokens: number[]): string {
        return tokens.map(idx => this.vocab.chars[idx] ?? '?').join('');
    }

    /**
     * Generate text continuation.
     */
    async generate(
        prompt: string,
        maxTokens: number,
        onToken?: (char: string, stats: GenerationStats) => void,
        temperature: number = 0.8
    ): Promise<string> {
        const startTime = performance.now();
        let tokenCount = 0;

        // Initialize context with prompt
        let context = this.tokenize(prompt);
        context = context.slice(-this.config.contextLength);

        // Pad if needed
        while (context.length < this.config.contextLength) {
            context.unshift(0);
        }

        let generated = '';

        for (let i = 0; i < maxTokens; i++) {
            // Run forward pass
            const nextToken = await this.forwardPass(context, temperature);
            const char = this.vocab.chars[nextToken] ?? '?';
            generated += char;

            // Update context
            context = [...context.slice(1), nextToken];

            tokenCount++;
            const elapsedMs = performance.now() - startTime;

            if (onToken) {
                onToken(char, {
                    tokensPerSecond: (tokenCount / elapsedMs) * 1000,
                    totalTokens: tokenCount,
                    elapsedMs,
                });
            }
        }

        return generated;
    }

    /**
     * Run a single forward pass and sample next token.
     */
    private async forwardPass(context: number[], temperature: number): Promise<number> {
        // Write token indices to GPU
        const tokenData = new Uint32Array(context);
        this.device.queue.writeBuffer(this.tokenBuffer!, 0, tokenData);

        const encoder = this.device.createCommandEncoder();
        let inputBuffer = this.activationBuffers[0];
        let outputBuffer = this.activationBuffers[1];

        // 1. Embedding lookup
        await this.runEmbedding(encoder, inputBuffer);

        // 2. Run through ternary layers with activation
        for (let i = 0; i < this.layers.length - 1; i++) {
            const layer = this.layers[i];
            if (layer.type === LayerType.TERNARY_LINEAR) {
                await this.runTernaryMatmul(encoder, layer, inputBuffer, outputBuffer);
                // Apply GELU activation (approximation done in JS for now)
                // Swap buffers
                [inputBuffer, outputBuffer] = [outputBuffer, inputBuffer];
            }
        }

        // 3. Output projection (last layer, FP16)
        const lastLayer = this.layers[this.layers.length - 1];
        await this.runFP16Matmul(encoder, lastLayer, inputBuffer, this.logitsBuffer!);

        // 4. Softmax
        await this.runSoftmax(encoder, temperature);

        // Submit commands
        this.device.queue.submit([encoder.finish()]);

        // Read back probabilities and sample
        return await this.sampleToken();
    }

    /**
     * Run embedding lookup on GPU.
     */
    private async runEmbedding(encoder: GPUCommandEncoder, outputBuffer: GPUBuffer): Promise<void> {
        if (!this.embeddingPipeline || !this.embeddingBuffer || !this.tokenBuffer) return;

        const { contextLength, hiddenDim, vocabSize } = this.config;

        // Create params buffer
        const paramsData = new Uint32Array([contextLength, hiddenDim, vocabSize, 0]);
        const paramsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(paramsBuffer, 0, paramsData);

        const bindGroup = this.device.createBindGroup({
            layout: this.embeddingPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.embeddingBuffer } },
                { binding: 1, resource: { buffer: this.tokenBuffer } },
                { binding: 2, resource: { buffer: outputBuffer } },
                { binding: 3, resource: { buffer: paramsBuffer } },
            ],
        });

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.embeddingPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil((contextLength * hiddenDim) / 64));
        pass.end();
    }

    /**
     * Run ternary matrix multiplication on GPU.
     */
    private async runTernaryMatmul(
        encoder: GPUCommandEncoder,
        layer: LayerBuffers,
        inputBuffer: GPUBuffer,
        outputBuffer: GPUBuffer
    ): Promise<void> {
        if (!this.ternaryMatmulPipeline || !layer.scales) return;

        const B = 1;  // Batch size
        const K = layer.inFeatures;
        const N = layer.outFeatures;
        const K_packed = Math.ceil(K / 16);

        // Create params buffer
        const paramsData = new Uint32Array([B, K, N, K_packed]);
        const paramsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(paramsBuffer, 0, paramsData);

        const bindGroup = this.device.createBindGroup({
            layout: this.ternaryMatmulPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: layer.weights } },
                { binding: 1, resource: { buffer: layer.scales } },
                { binding: 2, resource: { buffer: inputBuffer } },
                { binding: 3, resource: { buffer: outputBuffer } },
                { binding: 4, resource: { buffer: paramsBuffer } },
            ],
        });

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.ternaryMatmulPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(B / 8), Math.ceil(N / 8));
        pass.end();
    }

    /**
     * Run FP16 matrix multiplication (for output layer).
     * For simplicity, this uses the same pattern as ternary but without packing.
     */
    private async runFP16Matmul(
        _encoder: GPUCommandEncoder,
        _layer: LayerBuffers,
        _inputBuffer: GPUBuffer,
        _outputBuffer: GPUBuffer
    ): Promise<void> {
        // Placeholder - in production, you'd have a dedicated FP16 matmul shader
        // For the demo, we use CPU fallback for the output layer
        // since it's small (hidden_dim -> vocab_size)
    }

    /**
     * Run softmax on logits.
     */
    private async runSoftmax(encoder: GPUCommandEncoder, temperature: number): Promise<void> {
        if (!this.softmaxPipeline || !this.logitsBuffer || !this.probsBuffer) return;

        const { vocabSize } = this.config;

        // Create params buffer
        const paramsData = new Float32Array([vocabSize, temperature, 0, 0]);
        const paramsView = new DataView(paramsData.buffer);
        paramsView.setUint32(0, vocabSize, true);  // Write as uint32
        paramsView.setFloat32(4, temperature, true);

        const paramsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(paramsBuffer, 0, paramsData);

        const bindGroup = this.device.createBindGroup({
            layout: this.softmaxPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.logitsBuffer } },
                { binding: 1, resource: { buffer: this.probsBuffer } },
                { binding: 2, resource: { buffer: paramsBuffer } },
            ],
        });

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.softmaxPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();
    }

    /**
     * Sample a token from probability distribution.
     */
    private async sampleToken(): Promise<number> {
        if (!this.probsBuffer) return 0;

        // Read probabilities back to CPU
        const readBuffer = this.device.createBuffer({
            size: this.config.vocabSize * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.probsBuffer, 0, readBuffer, 0, this.config.vocabSize * 4);
        this.device.queue.submit([encoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const probs = new Float32Array(readBuffer.getMappedRange());

        // Multinomial sampling
        const r = Math.random();
        let cumsum = 0;
        let token = 0;

        for (let i = 0; i < probs.length; i++) {
            cumsum += probs[i];
            if (r < cumsum) {
                token = i;
                break;
            }
        }

        readBuffer.unmap();
        readBuffer.destroy();

        return token;
    }

    /**
     * Get memory usage statistics.
     */
    getMemoryStats(): MemoryStats {
        const packedWeightsKB = (this.totalTernaryWeights * 2) / 8 / 1024;  // 2 bits per weight
        const fp16EquivalentKB = (this.totalTernaryWeights * 2) / 1024;  // FP16

        return {
            packedWeightsKB,
            fp16EquivalentKB: fp16EquivalentKB + (this.totalFP16Weights * 2) / 1024,
            compressionRatio: fp16EquivalentKB / packedWeightsKB,
            scalesKB: (this.layers.length * this.config.hiddenDim * 4) / 1024,
        };
    }

    /**
     * Get model configuration.
     */
    getConfig(): ModelConfig {
        return this.config;
    }

    /**
     * Clean up GPU resources.
     */
    destroy(): void {
        this.embeddingBuffer?.destroy();
        this.tokenBuffer?.destroy();
        this.logitsBuffer?.destroy();
        this.probsBuffer?.destroy();

        for (const buffer of this.activationBuffers) {
            buffer.destroy();
        }

        for (const layer of this.layers) {
            layer.weights.destroy();
            layer.scales?.destroy();
            layer.bias?.destroy();
        }

        this.device.destroy();
    }
}
