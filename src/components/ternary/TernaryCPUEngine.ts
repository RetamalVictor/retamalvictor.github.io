/**
 * TernaryCPUEngine - CPU fallback for browsers without WebGPU.
 *
 * Uses pure JavaScript with TypedArrays for inference.
 * Slower than WebGPU but works in all browsers.
 */

import {
    ModelConfig,
    LayerType,
    GenerationStats,
    MemoryStats,
    VocabMapping,
    TBIN_MAGIC,
    TBIN_VERSION,
    HEADER_SIZE,
    LAYER_META_SIZE,
} from './types';

interface CPULayer {
    type: LayerType;
    inFeatures: number;
    outFeatures: number;
    weights: Float32Array;      // Unpacked ternary or FP32 weights
    scales?: Float32Array;      // Per-channel scales
    bias?: Float32Array;
}

export class TernaryCPUEngine {
    private config: ModelConfig;
    private vocab: VocabMapping;
    private embeddingWeights: Float32Array;
    private layers: CPULayer[] = [];

    // Stats
    private totalTernaryWeights = 0;
    private totalFP16Weights = 0;

    private constructor() {
        this.config = { vocabSize: 0, hiddenDim: 0, contextLength: 0, nLayers: 0 };
        this.vocab = { chars: '', charToIdx: {}, vocabSize: 0 };
        this.embeddingWeights = new Float32Array(0);
    }

    /**
     * Create a TernaryCPUEngine instance.
     */
    static async create(modelPath: string): Promise<TernaryCPUEngine> {
        const engine = new TernaryCPUEngine();
        await engine.loadModel(modelPath);
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

        console.log('[CPU] Model config:', this.config);

        // Parse layer metadata and load weights
        let offset = HEADER_SIZE;

        for (let i = 0; i < nLayers; i++) {
            const layerType = dataView.getUint32(offset, true) as LayerType;
            const inFeatures = dataView.getUint32(offset + 4, true);
            const outFeatures = dataView.getUint32(offset + 8, true);
            const hasBias = dataView.getUint32(offset + 12, true) !== 0;
            const dataOffset = dataView.getUint32(offset + 16, true);
            const scaleOffset = dataView.getUint32(offset + 20, true);
            const biasOffset = dataView.getUint32(offset + 24, true);

            offset += LAYER_META_SIZE;

            if (layerType === LayerType.EMBEDDING) {
                this.loadEmbedding(modelBuffer, inFeatures, outFeatures, dataOffset);
            } else if (layerType === LayerType.TERNARY_LINEAR) {
                this.loadTernaryLinear(modelBuffer, inFeatures, outFeatures, hasBias, dataOffset, scaleOffset, biasOffset);
            } else if (layerType === LayerType.FP16_LINEAR) {
                this.loadFP16Linear(modelBuffer, inFeatures, outFeatures, hasBias, dataOffset, biasOffset);
            }
        }

        console.log(`[CPU] Loaded ${this.layers.length} layers`);
    }

    private loadEmbedding(buffer: ArrayBuffer, vocabSize: number, embedDim: number, dataOffset: number): void {
        const numWeights = vocabSize * embedDim;
        const fp16Data = new Uint16Array(buffer, dataOffset, numWeights);
        this.embeddingWeights = new Float32Array(numWeights);

        for (let i = 0; i < numWeights; i++) {
            this.embeddingWeights[i] = this.float16ToFloat32(fp16Data[i]);
        }

        this.totalFP16Weights += numWeights;
    }

    private loadTernaryLinear(
        buffer: ArrayBuffer,
        inFeatures: number,
        outFeatures: number,
        hasBias: boolean,
        dataOffset: number,
        scaleOffset: number,
        biasOffset: number
    ): void {
        const K = inFeatures;
        const N = outFeatures;
        const K_bytes = Math.ceil(K / 4);  // 4 weights per byte

        // Read packed ternary weights as Uint8Array - layout is [K_bytes, N] row-major
        const packedData = new Uint8Array(buffer, dataOffset, K_bytes * N);
        const weights = new Float32Array(K * N);

        // Unpack: layout is [K_bytes, N] stored row-major
        // packed_T[kb, n] contains weights for output n, inputs kb*4 to kb*4+3
        for (let n = 0; n < N; n++) {
            for (let kb = 0; kb < K_bytes; kb++) {
                const byteIdx = kb * N + n;  // Row-major indexing
                const packed_byte = packedData[byteIdx];

                for (let i = 0; i < 4 && kb * 4 + i < K; i++) {
                    const code = (packed_byte >> (i * 2)) & 0x3;
                    // 0 = 0, 1 = +1, 2 = -1
                    const ternary = code === 1 ? 1 : (code === 2 ? -1 : 0);
                    weights[n * K + kb * 4 + i] = ternary;
                }
            }
        }


        // Read scales
        const scaleData = new Uint16Array(buffer, scaleOffset, N);
        const scales = new Float32Array(N);
        for (let i = 0; i < N; i++) {
            scales[i] = this.float16ToFloat32(scaleData[i]);
        }

        // Read bias if present
        let bias: Float32Array | undefined;
        if (hasBias) {
            const biasData = new Uint16Array(buffer, biasOffset, N);
            bias = new Float32Array(N);
            for (let i = 0; i < N; i++) {
                bias[i] = this.float16ToFloat32(biasData[i]);
            }
        }

        this.layers.push({
            type: LayerType.TERNARY_LINEAR,
            inFeatures: K,
            outFeatures: N,
            weights,
            scales,
            bias,
        });

        this.totalTernaryWeights += K * N;
    }

    private loadFP16Linear(
        buffer: ArrayBuffer,
        inFeatures: number,
        outFeatures: number,
        hasBias: boolean,
        dataOffset: number,
        biasOffset: number
    ): void {
        const numWeights = inFeatures * outFeatures;
        const fp16Data = new Uint16Array(buffer, dataOffset, numWeights);
        const weights = new Float32Array(numWeights);

        for (let i = 0; i < numWeights; i++) {
            weights[i] = this.float16ToFloat32(fp16Data[i]);
        }

        let bias: Float32Array | undefined;
        if (hasBias) {
            const biasData = new Uint16Array(buffer, biasOffset, outFeatures);
            bias = new Float32Array(outFeatures);
            for (let i = 0; i < outFeatures; i++) {
                bias[i] = this.float16ToFloat32(biasData[i]);
            }
        }

        this.layers.push({
            type: LayerType.FP16_LINEAR,
            inFeatures,
            outFeatures,
            weights,
            bias,
        });

        this.totalFP16Weights += numWeights;
    }

    private float16ToFloat32(h: number): number {
        const sign = (h & 0x8000) >> 15;
        const exp = (h & 0x7C00) >> 10;
        const frac = h & 0x03FF;

        if (exp === 0) {
            if (frac === 0) return sign ? -0 : 0;
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
            const nextToken = this.forwardPass(context, temperature);
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

            // Yield to UI every few tokens
            if (i % 5 === 0) {
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }

        return generated;
    }

    /**
     * Run a single forward pass and sample next token.
     */
    private forwardPass(context: number[], temperature: number): number {
        const { contextLength, hiddenDim } = this.config;

        // 1. Embedding lookup and flatten
        let activation: Float32Array = new Float32Array(contextLength * hiddenDim);
        for (let t = 0; t < contextLength; t++) {
            const tokenIdx = context[t];
            const embOffset = tokenIdx * hiddenDim;
            for (let d = 0; d < hiddenDim; d++) {
                activation[t * hiddenDim + d] = this.embeddingWeights[embOffset + d];
            }
        }

        // 2. Run through layers
        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];

            if (layer.type === LayerType.TERNARY_LINEAR) {
                activation = this.ternaryMatmul(activation, layer);
                // Apply GELU activation (except for last layer)
                if (i < this.layers.length - 1) {
                    activation = this.gelu(activation);
                }
            } else if (layer.type === LayerType.FP16_LINEAR) {
                activation = this.fp16Matmul(activation, layer);
            }

        }

        // 3. Softmax and sample
        const probs = this.softmax(activation, temperature);
        return this.sample(probs);
    }

    private ternaryMatmul(input: Float32Array, layer: CPULayer): Float32Array {
        const K = layer.inFeatures;
        const N = layer.outFeatures;
        const output = new Float32Array(N);

        for (let n = 0; n < N; n++) {
            let acc = 0;
            const wOffset = n * K;
            for (let k = 0; k < K; k++) {
                acc += input[k] * layer.weights[wOffset + k];
            }
            // Apply scale
            output[n] = acc * (layer.scales?.[n] ?? 1);
            // Add bias
            if (layer.bias) {
                output[n] += layer.bias[n];
            }
        }

        return output;
    }

    private fp16Matmul(input: Float32Array, layer: CPULayer): Float32Array {
        const K = layer.inFeatures;
        const N = layer.outFeatures;
        const output = new Float32Array(N);

        for (let n = 0; n < N; n++) {
            let acc = 0;
            const wOffset = n * K;
            for (let k = 0; k < K; k++) {
                acc += input[k] * layer.weights[wOffset + k];
            }
            output[n] = acc;
            if (layer.bias) {
                output[n] += layer.bias[n];
            }
        }

        return output;
    }

    private gelu(x: Float32Array): Float32Array {
        const result = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) {
            // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            const xi = x[i];
            const cdf = 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (xi + 0.044715 * xi * xi * xi)));
            result[i] = xi * cdf;
        }
        return result;
    }

    private softmax(logits: Float32Array, temperature: number): Float32Array {
        const probs = new Float32Array(logits.length);

        // Find max for numerical stability
        let maxLogit = -Infinity;
        for (let i = 0; i < logits.length; i++) {
            const scaled = logits[i] / temperature;
            if (scaled > maxLogit) maxLogit = scaled;
        }

        // Compute exp and sum
        let sum = 0;
        for (let i = 0; i < logits.length; i++) {
            probs[i] = Math.exp(logits[i] / temperature - maxLogit);
            sum += probs[i];
        }

        // Normalize
        for (let i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }

        return probs;
    }

    private sample(probs: Float32Array): number {
        const r = Math.random();
        let cumsum = 0;

        for (let i = 0; i < probs.length; i++) {
            cumsum += probs[i];
            if (r < cumsum) {
                return i;
            }
        }

        return probs.length - 1;
    }

    /**
     * Get memory usage statistics.
     */
    getMemoryStats(): MemoryStats {
        const packedWeightsKB = (this.totalTernaryWeights * 2) / 8 / 1024;
        const fp16EquivalentKB = (this.totalTernaryWeights * 2) / 1024;

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
     * Clean up resources.
     */
    destroy(): void {
        // Nothing to clean up for CPU engine
    }
}
