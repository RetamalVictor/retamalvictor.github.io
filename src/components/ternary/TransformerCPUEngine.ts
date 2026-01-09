/**
 * TransformerCPUEngine - LLaMA-style transformer inference in JavaScript.
 *
 * Supports ternary quantization (1.58 bits/weight) with per-channel scales.
 * Implements: RMSNorm, RoPE, Multi-head Attention, SwiGLU MLP.
 */

import { SafeTensorsLoader } from './SafeTensorsLoader';
import { BPETokenizer } from './BPETokenizer';
import { GenerationStats } from './types';

interface TransformerConfig {
    vocabSize: number;
    dim: number;
    nLayers: number;
    nHeads: number;
    nKvHeads: number;  // For GQA support (nKvHeads < nHeads)
    maxSeqLen: number;
}

// KV Cache for a single layer
interface LayerKVCache {
    k: Float32Array;  // [nKvHeads, seqLen, headDim]
    v: Float32Array;  // [nKvHeads, seqLen, headDim]
}

interface TernaryLayer {
    weightsPacked: Uint8Array;   // Packed 2-bit weights [outFeatures, inFeatures/4]
    scales: Float32Array;         // Per-output-channel scales [outFeatures]
    outFeatures: number;
    inFeatures: number;
}

interface TransformerBlock {
    norm1Weight: Float32Array;    // Pre-attention RMSNorm [dim]
    qProj: TernaryLayer;          // Query projection
    kvProj: TernaryLayer;         // Key+Value projection (combined)
    proj: TernaryLayer;           // Output projection
    norm2Weight: Float32Array;    // Pre-MLP RMSNorm [dim]
    wGate: TernaryLayer;          // SwiGLU gate
    wUp: TernaryLayer;            // SwiGLU up
    wDown: TernaryLayer;          // SwiGLU down
}

// RMSNorm epsilon
const NORM_EPS = 1e-5;

export class TransformerCPUEngine {
    private config: TransformerConfig;
    private tokenizer: BPETokenizer | null = null;

    // Model weights
    private embedding: Float32Array;       // [vocabSize, dim]
    private blocks: TransformerBlock[] = [];
    private normWeight: Float32Array;      // Final RMSNorm [dim]
    private head: Float32Array;            // Output projection [vocabSize, dim]

    // RoPE cache
    private cosCache: Float32Array;        // [maxSeqLen, headDim/2]
    private sinCache: Float32Array;        // [maxSeqLen, headDim/2]

    // KV Cache for incremental generation
    private kvCache: LayerKVCache[] = [];
    private cacheSeqLen: number = 0;       // Current cached sequence length

    // Generation control
    private stopRequested: boolean = false;

    // Memory tracking
    private ternaryWeights = 0;
    private fp16Weights = 0;

    private constructor() {
        this.config = {
            vocabSize: 0,
            dim: 0,
            nLayers: 0,
            nHeads: 0,
            nKvHeads: 0,
            maxSeqLen: 0,
        };
        this.embedding = new Float32Array(0);
        this.normWeight = new Float32Array(0);
        this.head = new Float32Array(0);
        this.cosCache = new Float32Array(0);
        this.sinCache = new Float32Array(0);
    }

    /**
     * Create engine from model directory URL.
     */
    static async create(modelPath: string): Promise<TransformerCPUEngine> {
        const engine = new TransformerCPUEngine();
        await engine.loadModel(modelPath);
        return engine;
    }

    /**
     * Load model from SafeTensors format.
     */
    private async loadModel(basePath: string): Promise<void> {
        console.log('[Transformer] Loading model from', basePath);

        // Load config
        const configResp = await fetch(`${basePath}/config.json`);
        if (!configResp.ok) throw new Error('Failed to load config.json');
        const config = await configResp.json();

        // n_kv_heads defaults to n_heads (MHA) if not specified (GQA has n_kv_heads < n_heads)
        const nKvHeads = config.n_kv_heads ?? config.n_heads;

        this.config = {
            vocabSize: config.vocab_size,
            dim: config.dim,
            nLayers: config.n_layers,
            nHeads: config.n_heads,
            nKvHeads: nKvHeads,
            maxSeqLen: config.max_seq_len,
        };

        console.log(`[Transformer] Attention: ${nKvHeads === config.n_heads ? 'MHA' : 'GQA'} (${config.n_heads} Q heads, ${nKvHeads} KV heads)`);

        console.log('[Transformer] Config:', this.config);

        // Load tokenizer
        this.tokenizer = await BPETokenizer.fromUrl(`${basePath}/tokenizer.json`);
        console.log('[Transformer] Tokenizer loaded, vocab size:', this.tokenizer.vocabSize);

        // Load SafeTensors
        const loader = await SafeTensorsLoader.fromUrl(`${basePath}/model.safetensors`);
        console.log('[Transformer] SafeTensors loaded, tensors:', loader.getTensorNames().length);

        // Load embedding
        const tokTensor = loader.getTensorFloat32('tok.weight');
        this.embedding = tokTensor.data;
        this.fp16Weights += this.config.vocabSize * this.config.dim;

        // Load head
        const headTensor = loader.getTensorFloat32('head.weight');
        this.head = headTensor.data;
        this.fp16Weights += this.config.vocabSize * this.config.dim;

        // Load final norm
        const normTensor = loader.getTensorFloat32('norm.weight');
        this.normWeight = normTensor.data;

        // Load transformer blocks
        for (let i = 0; i < this.config.nLayers; i++) {
            const block = this.loadBlock(loader, i);
            this.blocks.push(block);
        }

        // Initialize RoPE cache
        this.initRoPECache();

        console.log(`[Transformer] Loaded ${this.config.nLayers} blocks`);
        console.log(`[Transformer] Ternary weights: ${(this.ternaryWeights / 1e6).toFixed(1)}M`);
        console.log(`[Transformer] FP16 weights: ${(this.fp16Weights / 1e6).toFixed(1)}M`);
    }

    /**
     * Load a single transformer block.
     */
    private loadBlock(loader: SafeTensorsLoader, blockIdx: number): TransformerBlock {
        const prefix = `blocks.${blockIdx}`;

        // RMSNorm weights
        const norm1 = loader.getTensorFloat32(`${prefix}.norm1.weight`);
        const norm2 = loader.getTensorFloat32(`${prefix}.norm2.weight`);

        // Attention projections
        const qProj = this.loadTernaryLayer(loader, `${prefix}.attn.q_proj`);
        const kvProj = this.loadTernaryLayer(loader, `${prefix}.attn.kv_proj`);
        const proj = this.loadTernaryLayer(loader, `${prefix}.attn.proj`);

        // MLP
        const wGate = this.loadTernaryLayer(loader, `${prefix}.mlp.w_gate`);
        const wUp = this.loadTernaryLayer(loader, `${prefix}.mlp.w_up`);
        const wDown = this.loadTernaryLayer(loader, `${prefix}.mlp.w_down`);

        return {
            norm1Weight: norm1.data,
            qProj,
            kvProj,
            proj,
            norm2Weight: norm2.data,
            wGate,
            wUp,
            wDown,
        };
    }

    /**
     * Load a ternary linear layer.
     */
    private loadTernaryLayer(loader: SafeTensorsLoader, prefix: string): TernaryLayer {
        const weightTensor = loader.getTensorBuffer(`${prefix}.weight_packed`);
        const scaleTensor = loader.getTensorFloat32(`${prefix}.scale`);

        // Shape is [outFeatures, inFeatures/4] for packed weights
        const outFeatures = weightTensor.shape[0];
        const inFeatures = weightTensor.shape[1] * 4;

        this.ternaryWeights += outFeatures * inFeatures;

        return {
            weightsPacked: new Uint8Array(weightTensor.data),
            scales: scaleTensor.data,
            outFeatures,
            inFeatures,
        };
    }

    /**
     * Initialize RoPE cos/sin cache.
     */
    private initRoPECache(): void {
        const headDim = this.config.dim / this.config.nHeads;
        const halfDim = headDim / 2;
        const theta = 10000.0;

        this.cosCache = new Float32Array(this.config.maxSeqLen * halfDim);
        this.sinCache = new Float32Array(this.config.maxSeqLen * halfDim);

        for (let pos = 0; pos < this.config.maxSeqLen; pos++) {
            for (let i = 0; i < halfDim; i++) {
                const freq = 1.0 / Math.pow(theta, (2 * i) / headDim);
                const angle = pos * freq;
                this.cosCache[pos * halfDim + i] = Math.cos(angle);
                this.sinCache[pos * halfDim + i] = Math.sin(angle);
            }
        }
    }

    /**
     * Tokenize text.
     */
    tokenize(text: string): number[] {
        if (!this.tokenizer) throw new Error('Tokenizer not loaded');
        return this.tokenizer.encode(text);
    }

    /**
     * Decode tokens to text.
     */
    decode(tokens: number[]): string {
        if (!this.tokenizer) throw new Error('Tokenizer not loaded');
        return this.tokenizer.decode(tokens);
    }

    /**
     * Decode single token.
     */
    decodeToken(token: number): string {
        if (!this.tokenizer) throw new Error('Tokenizer not loaded');
        return this.tokenizer.decodeToken(token);
    }

    /**
     * Stop ongoing generation.
     */
    stop(): void {
        this.stopRequested = true;
    }

    /**
     * Reset KV cache for new generation.
     */
    resetCache(): void {
        this.kvCache = [];
        this.cacheSeqLen = 0;
    }

    /**
     * Initialize KV cache for all layers.
     */
    private initKVCache(): void {
        const { nLayers, nKvHeads, maxSeqLen, dim, nHeads } = this.config;
        const headDim = dim / nHeads;

        this.kvCache = [];
        for (let i = 0; i < nLayers; i++) {
            this.kvCache.push({
                k: new Float32Array(nKvHeads * maxSeqLen * headDim),
                v: new Float32Array(nKvHeads * maxSeqLen * headDim),
            });
        }
        this.cacheSeqLen = 0;
    }

    /**
     * Generate text continuation with KV cache.
     */
    async generate(
        prompt: string,
        maxTokens: number,
        onToken?: (token: string, stats: GenerationStats) => void,
        temperature: number = 0.8
    ): Promise<string> {
        this.stopRequested = false;
        const startTime = performance.now();

        // Tokenize prompt
        const tokens = this.tokenize(prompt);
        console.log(`[Generate] Prompt: ${tokens.length} tokens`);

        // Initialize fresh KV cache
        this.initKVCache();

        // Prefill: process all prompt tokens at once
        let logits = this.forwardPrefill(tokens);
        let generated = '';

        for (let step = 0; step < maxTokens; step++) {
            // Check for stop request
            if (this.stopRequested) {
                console.log('[Generate] Stopped by user');
                break;
            }

            // Sample next token
            const nextToken = this.sample(logits, temperature);

            // Decode the new token
            const decoded = this.decodeToken(nextToken);
            generated += decoded;

            // Report progress
            const elapsedMs = performance.now() - startTime;
            if (onToken) {
                onToken(decoded, {
                    tokensPerSecond: ((step + 1) / elapsedMs) * 1000,
                    totalTokens: step + 1,
                    elapsedMs,
                });
            }

            // Yield to UI every token for responsiveness
            await new Promise(resolve => setTimeout(resolve, 0));

            // Check if we've hit max sequence length
            if (this.cacheSeqLen >= this.config.maxSeqLen - 1) {
                console.log('[Generate] Hit max sequence length');
                break;
            }

            // Decode step: process only the new token using cached KV
            logits = this.forwardDecode(nextToken);
        }

        return generated;
    }

    /**
     * Prefill: process all prompt tokens and populate KV cache.
     * Returns logits for the last token.
     */
    private forwardPrefill(tokens: number[]): Float32Array {
        const { dim, nHeads, nLayers } = this.config;
        const seqLen = tokens.length;
        const headDim = dim / nHeads;

        // Embedding lookup: [seqLen, dim]
        let hidden = new Float32Array(seqLen * dim);
        for (let t = 0; t < seqLen; t++) {
            const tokenId = tokens[t];
            const srcOffset = tokenId * dim;
            const dstOffset = t * dim;
            for (let d = 0; d < dim; d++) {
                hidden[dstOffset + d] = this.embedding[srcOffset + d];
            }
        }

        // Process each block
        for (let blockIdx = 0; blockIdx < nLayers; blockIdx++) {
            const block = this.blocks[blockIdx];
            const cache = this.kvCache[blockIdx];

            // Pre-attention RMSNorm
            const normed1 = this.rmsNorm(hidden, block.norm1Weight, seqLen, dim);

            // Attention with KV cache population
            const attnOut = this.attentionPrefill(normed1, block, cache, seqLen, headDim);

            // Residual
            for (let i = 0; i < hidden.length; i++) {
                hidden[i] += attnOut[i];
            }

            // Pre-MLP RMSNorm
            const normed2 = this.rmsNorm(hidden, block.norm2Weight, seqLen, dim);

            // MLP (SwiGLU)
            const mlpOut = this.swiglu(normed2, block, seqLen);

            // Residual
            for (let i = 0; i < hidden.length; i++) {
                hidden[i] += mlpOut[i];
            }
        }

        // Update cache sequence length
        this.cacheSeqLen = seqLen;

        // Final RMSNorm
        const finalHidden = this.rmsNorm(hidden, this.normWeight, seqLen, dim);

        // Get last token's hidden state
        const lastHidden = new Float32Array(dim);
        const lastOffset = (seqLen - 1) * dim;
        for (let d = 0; d < dim; d++) {
            lastHidden[d] = finalHidden[lastOffset + d];
        }

        // Output projection: [vocabSize]
        return this.matmulFP32(lastHidden, this.head, dim, this.config.vocabSize);
    }

    /**
     * Decode: process single token using cached KV.
     * Returns logits for the new token.
     */
    private forwardDecode(token: number): Float32Array {
        const { dim, nHeads, nLayers } = this.config;
        const headDim = dim / nHeads;
        const pos = this.cacheSeqLen;  // Position of new token

        // Embedding lookup: [1, dim] -> just [dim]
        const hidden = new Float32Array(dim);
        const srcOffset = token * dim;
        for (let d = 0; d < dim; d++) {
            hidden[d] = this.embedding[srcOffset + d];
        }

        // Process each block
        for (let blockIdx = 0; blockIdx < nLayers; blockIdx++) {
            const block = this.blocks[blockIdx];
            const cache = this.kvCache[blockIdx];

            // Pre-attention RMSNorm (single token)
            const normed1 = this.rmsNormSingle(hidden, block.norm1Weight);

            // Attention with KV cache lookup
            const attnOut = this.attentionDecode(normed1, block, cache, pos, headDim);

            // Residual
            for (let d = 0; d < dim; d++) {
                hidden[d] += attnOut[d];
            }

            // Pre-MLP RMSNorm (single token)
            const normed2 = this.rmsNormSingle(hidden, block.norm2Weight);

            // MLP (SwiGLU) - single token
            const mlpOut = this.swigluSingle(normed2, block);

            // Residual
            for (let d = 0; d < dim; d++) {
                hidden[d] += mlpOut[d];
            }
        }

        // Update cache sequence length
        this.cacheSeqLen = pos + 1;

        // Final RMSNorm
        const finalHidden = this.rmsNormSingle(hidden, this.normWeight);

        // Output projection: [vocabSize]
        return this.matmulFP32(finalHidden, this.head, dim, this.config.vocabSize);
    }

    /**
     * RMSNorm: x * w / rms(x)
     */
    private rmsNorm(
        x: Float32Array,
        weight: Float32Array,
        seqLen: number,
        dim: number
    ): Float32Array {
        const result = new Float32Array(x.length);

        for (let t = 0; t < seqLen; t++) {
            const offset = t * dim;

            // Compute RMS
            let sumSq = 0;
            for (let d = 0; d < dim; d++) {
                const val = x[offset + d];
                sumSq += val * val;
            }
            const rms = Math.sqrt(sumSq / dim + NORM_EPS);

            // Normalize and scale
            for (let d = 0; d < dim; d++) {
                result[offset + d] = (x[offset + d] / rms) * weight[d];
            }
        }

        return result;
    }

    /**
     * RMSNorm for single token (decode phase).
     */
    private rmsNormSingle(x: Float32Array, weight: Float32Array): Float32Array {
        const dim = x.length;
        const result = new Float32Array(dim);

        // Compute RMS
        let sumSq = 0;
        for (let d = 0; d < dim; d++) {
            sumSq += x[d] * x[d];
        }
        const rms = Math.sqrt(sumSq / dim + NORM_EPS);

        // Normalize and scale
        for (let d = 0; d < dim; d++) {
            result[d] = (x[d] / rms) * weight[d];
        }

        return result;
    }

    /**
     * Apply RoPE (Rotary Position Embedding).
     */
    private applyRoPE(
        x: Float32Array,
        seqLen: number,
        nHeads: number,
        headDim: number
    ): void {
        const halfDim = headDim / 2;
        const dim = this.config.dim;

        for (let t = 0; t < seqLen; t++) {
            for (let h = 0; h < nHeads; h++) {
                for (let i = 0; i < halfDim; i++) {
                    const idx = t * dim + h * headDim + i;
                    const idx2 = idx + halfDim;

                    const cos = this.cosCache[t * halfDim + i];
                    const sin = this.sinCache[t * halfDim + i];

                    const x0 = x[idx];
                    const x1 = x[idx2];

                    x[idx] = x0 * cos - x1 * sin;
                    x[idx2] = x0 * sin + x1 * cos;
                }
            }
        }
    }

    /**
     * SwiGLU MLP: down(silu(gate(x)) * up(x))
     */
    private swiglu(
        x: Float32Array,
        block: TransformerBlock,
        seqLen: number
    ): Float32Array {
        // Gate and up projections
        const gate = this.ternaryMatmul(x, block.wGate, seqLen);
        const up = this.ternaryMatmul(x, block.wUp, seqLen);

        // SiLU (swish) activation on gate, multiply with up
        for (let i = 0; i < gate.length; i++) {
            const g = gate[i];
            const silu = g / (1 + Math.exp(-g));  // SiLU = x * sigmoid(x)
            gate[i] = silu * up[i];
        }

        // Down projection
        return this.ternaryMatmul(gate, block.wDown, seqLen);
    }

    /**
     * SwiGLU MLP for single token (decode phase).
     */
    private swigluSingle(x: Float32Array, block: TransformerBlock): Float32Array {
        // Gate and up projections (single token)
        const gate = this.ternaryMatmulSingle(x, block.wGate);
        const up = this.ternaryMatmulSingle(x, block.wUp);

        // SiLU activation on gate, multiply with up
        for (let i = 0; i < gate.length; i++) {
            const g = gate[i];
            const silu = g / (1 + Math.exp(-g));
            gate[i] = silu * up[i];
        }

        // Down projection
        return this.ternaryMatmulSingle(gate, block.wDown);
    }

    /**
     * Attention with KV cache population (prefill phase).
     * Processes all tokens and stores K, V in cache.
     */
    private attentionPrefill(
        x: Float32Array,
        block: TransformerBlock,
        cache: LayerKVCache,
        seqLen: number,
        headDim: number
    ): Float32Array {
        const { dim, nHeads, nKvHeads, maxSeqLen } = this.config;
        const kvDim = nKvHeads * headDim;

        // Q projection: [seqLen, dim]
        const q = this.ternaryMatmul(x, block.qProj, seqLen);

        // KV projection: [seqLen, 2 * kvDim] (K and V concatenated)
        const kv = this.ternaryMatmul(x, block.kvProj, seqLen);

        // Extract K and V, and store in cache
        // K and V have shape [seqLen, nKvHeads, headDim]
        for (let t = 0; t < seqLen; t++) {
            const kvOffset = t * 2 * kvDim;
            for (let kh = 0; kh < nKvHeads; kh++) {
                const cacheOffset = kh * maxSeqLen * headDim + t * headDim;
                const kOffset = kvOffset + kh * headDim;
                const vOffset = kvOffset + kvDim + kh * headDim;
                for (let d = 0; d < headDim; d++) {
                    cache.k[cacheOffset + d] = kv[kOffset + d];
                    cache.v[cacheOffset + d] = kv[vOffset + d];
                }
            }
        }

        // Apply RoPE to Q
        this.applyRoPE(q, seqLen, nHeads, headDim);

        // Apply RoPE to K in cache
        this.applyRoPEToCache(cache.k, seqLen, nKvHeads, headDim);

        // Compute attention
        const scale = 1.0 / Math.sqrt(headDim);
        const attnOut = new Float32Array(seqLen * dim);

        // GQA: each KV head serves (nHeads / nKvHeads) Q heads
        const headsPerKv = nHeads / nKvHeads;

        for (let h = 0; h < nHeads; h++) {
            const kvHead = Math.floor(h / headsPerKv);  // Which KV head this Q head uses

            for (let tq = 0; tq < seqLen; tq++) {
                // Compute attention scores
                const scores = new Float32Array(seqLen);
                let maxScore = -Infinity;

                for (let tk = 0; tk <= tq; tk++) {  // Causal mask
                    let score = 0;
                    const qOffset = tq * dim + h * headDim;
                    const kOffset = kvHead * maxSeqLen * headDim + tk * headDim;

                    for (let d = 0; d < headDim; d++) {
                        score += q[qOffset + d] * cache.k[kOffset + d];
                    }
                    score *= scale;
                    scores[tk] = score;
                    if (score > maxScore) maxScore = score;
                }

                // Softmax
                let sumExp = 0;
                for (let tk = 0; tk <= tq; tk++) {
                    scores[tk] = Math.exp(scores[tk] - maxScore);
                    sumExp += scores[tk];
                }
                for (let tk = 0; tk <= tq; tk++) {
                    scores[tk] /= sumExp;
                }

                // Weighted sum of values
                const outOffset = tq * dim + h * headDim;
                for (let d = 0; d < headDim; d++) {
                    let acc = 0;
                    for (let tk = 0; tk <= tq; tk++) {
                        const vOffset = kvHead * maxSeqLen * headDim + tk * headDim;
                        acc += scores[tk] * cache.v[vOffset + d];
                    }
                    attnOut[outOffset + d] = acc;
                }
            }
        }

        // Output projection
        return this.ternaryMatmul(attnOut, block.proj, seqLen);
    }

    /**
     * Attention with KV cache lookup (decode phase).
     * Processes single token using cached K, V.
     */
    private attentionDecode(
        x: Float32Array,
        block: TransformerBlock,
        cache: LayerKVCache,
        pos: number,
        headDim: number
    ): Float32Array {
        const { dim, nHeads, nKvHeads, maxSeqLen } = this.config;
        const kvDim = nKvHeads * headDim;

        // Q projection: [dim]
        const q = this.ternaryMatmulSingle(x, block.qProj);

        // KV projection: [2 * kvDim]
        const kv = this.ternaryMatmulSingle(x, block.kvProj);

        // Extract K and V for this position and store in cache
        for (let kh = 0; kh < nKvHeads; kh++) {
            const cacheOffset = kh * maxSeqLen * headDim + pos * headDim;
            const kOffset = kh * headDim;
            const vOffset = kvDim + kh * headDim;
            for (let d = 0; d < headDim; d++) {
                cache.k[cacheOffset + d] = kv[kOffset + d];
                cache.v[cacheOffset + d] = kv[vOffset + d];
            }
        }

        // Apply RoPE to Q at current position
        this.applyRoPESingle(q, pos, nHeads, headDim);

        // Apply RoPE to K at current position in cache
        this.applyRoPEToSinglePos(cache.k, pos, nKvHeads, headDim);

        // Compute attention
        const scale = 1.0 / Math.sqrt(headDim);
        const attnOut = new Float32Array(dim);

        // GQA: each KV head serves (nHeads / nKvHeads) Q heads
        const headsPerKv = nHeads / nKvHeads;
        const seqLen = pos + 1;  // Attend to all cached positions

        for (let h = 0; h < nHeads; h++) {
            const kvHead = Math.floor(h / headsPerKv);

            // Compute attention scores for all cached positions
            const scores = new Float32Array(seqLen);
            let maxScore = -Infinity;

            for (let tk = 0; tk < seqLen; tk++) {
                let score = 0;
                const qOffset = h * headDim;
                const kOffset = kvHead * maxSeqLen * headDim + tk * headDim;

                for (let d = 0; d < headDim; d++) {
                    score += q[qOffset + d] * cache.k[kOffset + d];
                }
                score *= scale;
                scores[tk] = score;
                if (score > maxScore) maxScore = score;
            }

            // Softmax
            let sumExp = 0;
            for (let tk = 0; tk < seqLen; tk++) {
                scores[tk] = Math.exp(scores[tk] - maxScore);
                sumExp += scores[tk];
            }
            for (let tk = 0; tk < seqLen; tk++) {
                scores[tk] /= sumExp;
            }

            // Weighted sum of values
            const outOffset = h * headDim;
            for (let d = 0; d < headDim; d++) {
                let acc = 0;
                for (let tk = 0; tk < seqLen; tk++) {
                    const vOffset = kvHead * maxSeqLen * headDim + tk * headDim;
                    acc += scores[tk] * cache.v[vOffset + d];
                }
                attnOut[outOffset + d] = acc;
            }
        }

        // Output projection
        return this.ternaryMatmulSingle(attnOut, block.proj);
    }

    /**
     * Apply RoPE to K values in cache.
     */
    private applyRoPEToCache(
        k: Float32Array,
        seqLen: number,
        nKvHeads: number,
        headDim: number
    ): void {
        const halfDim = headDim / 2;
        const { maxSeqLen } = this.config;

        for (let t = 0; t < seqLen; t++) {
            for (let kh = 0; kh < nKvHeads; kh++) {
                const baseIdx = kh * maxSeqLen * headDim + t * headDim;
                for (let i = 0; i < halfDim; i++) {
                    const idx = baseIdx + i;
                    const idx2 = idx + halfDim;

                    const cos = this.cosCache[t * halfDim + i];
                    const sin = this.sinCache[t * halfDim + i];

                    const x0 = k[idx];
                    const x1 = k[idx2];

                    k[idx] = x0 * cos - x1 * sin;
                    k[idx2] = x0 * sin + x1 * cos;
                }
            }
        }
    }

    /**
     * Apply RoPE to single position in cache.
     */
    private applyRoPEToSinglePos(
        k: Float32Array,
        pos: number,
        nKvHeads: number,
        headDim: number
    ): void {
        const halfDim = headDim / 2;
        const { maxSeqLen } = this.config;

        for (let kh = 0; kh < nKvHeads; kh++) {
            const baseIdx = kh * maxSeqLen * headDim + pos * headDim;
            for (let i = 0; i < halfDim; i++) {
                const idx = baseIdx + i;
                const idx2 = idx + halfDim;

                const cos = this.cosCache[pos * halfDim + i];
                const sin = this.sinCache[pos * halfDim + i];

                const x0 = k[idx];
                const x1 = k[idx2];

                k[idx] = x0 * cos - x1 * sin;
                k[idx2] = x0 * sin + x1 * cos;
            }
        }
    }

    /**
     * Apply RoPE to Q for single token.
     */
    private applyRoPESingle(
        q: Float32Array,
        pos: number,
        nHeads: number,
        headDim: number
    ): void {
        const halfDim = headDim / 2;

        for (let h = 0; h < nHeads; h++) {
            for (let i = 0; i < halfDim; i++) {
                const idx = h * headDim + i;
                const idx2 = idx + halfDim;

                const cos = this.cosCache[pos * halfDim + i];
                const sin = this.sinCache[pos * halfDim + i];

                const x0 = q[idx];
                const x1 = q[idx2];

                q[idx] = x0 * cos - x1 * sin;
                q[idx2] = x0 * sin + x1 * cos;
            }
        }
    }

    /**
     * Ternary matrix multiplication with scales.
     * Input: [seqLen, inFeatures]
     * Weights: packed [outFeatures, inFeatures/4]
     * Output: [seqLen, outFeatures]
     */
    private ternaryMatmul(
        input: Float32Array,
        layer: TernaryLayer,
        seqLen: number
    ): Float32Array {
        const { inFeatures, outFeatures, weightsPacked, scales } = layer;
        const output = new Float32Array(seqLen * outFeatures);
        const inBytes = Math.ceil(inFeatures / 4);

        for (let t = 0; t < seqLen; t++) {
            const inputOffset = t * inFeatures;
            const outputOffset = t * outFeatures;

            for (let n = 0; n < outFeatures; n++) {
                let acc = 0;
                const weightOffset = n * inBytes;

                // Process packed weights
                for (let kb = 0; kb < inBytes; kb++) {
                    const packed = weightsPacked[weightOffset + kb];

                    // Unpack 4 ternary values from byte
                    for (let i = 0; i < 4 && kb * 4 + i < inFeatures; i++) {
                        const code = (packed >> (i * 2)) & 0x3;
                        // 0 = 0, 1 = +1, 2 = -1
                        const w = code === 1 ? 1 : (code === 2 ? -1 : 0);
                        const k = kb * 4 + i;
                        acc += input[inputOffset + k] * w;
                    }
                }

                // Apply scale
                output[outputOffset + n] = acc * scales[n];
            }
        }

        return output;
    }

    /**
     * Ternary matrix multiplication for single token (decode phase).
     * Input: [inFeatures]
     * Output: [outFeatures]
     */
    private ternaryMatmulSingle(input: Float32Array, layer: TernaryLayer): Float32Array {
        const { inFeatures, outFeatures, weightsPacked, scales } = layer;
        const output = new Float32Array(outFeatures);
        const inBytes = Math.ceil(inFeatures / 4);

        for (let n = 0; n < outFeatures; n++) {
            let acc = 0;
            const weightOffset = n * inBytes;

            // Process packed weights
            for (let kb = 0; kb < inBytes; kb++) {
                const packed = weightsPacked[weightOffset + kb];

                // Unpack 4 ternary values from byte
                for (let i = 0; i < 4 && kb * 4 + i < inFeatures; i++) {
                    const code = (packed >> (i * 2)) & 0x3;
                    const w = code === 1 ? 1 : (code === 2 ? -1 : 0);
                    const k = kb * 4 + i;
                    acc += input[k] * w;
                }
            }

            output[n] = acc * scales[n];
        }

        return output;
    }

    /**
     * FP32 matrix multiplication (for head).
     * Input: [K]
     * Weights: [N, K]
     * Output: [N]
     */
    private matmulFP32(
        input: Float32Array,
        weights: Float32Array,
        K: number,
        N: number
    ): Float32Array {
        const output = new Float32Array(N);

        for (let n = 0; n < N; n++) {
            let acc = 0;
            const weightOffset = n * K;
            for (let k = 0; k < K; k++) {
                acc += input[k] * weights[weightOffset + k];
            }
            output[n] = acc;
        }

        return output;
    }

    /**
     * Sample from logits with temperature.
     */
    private sample(logits: Float32Array, temperature: number): number {
        // Apply temperature
        const scaled = new Float32Array(logits.length);
        let maxLogit = -Infinity;

        for (let i = 0; i < logits.length; i++) {
            scaled[i] = logits[i] / temperature;
            if (scaled[i] > maxLogit) maxLogit = scaled[i];
        }

        // Softmax
        let sumExp = 0;
        for (let i = 0; i < scaled.length; i++) {
            scaled[i] = Math.exp(scaled[i] - maxLogit);
            sumExp += scaled[i];
        }
        for (let i = 0; i < scaled.length; i++) {
            scaled[i] /= sumExp;
        }

        // Sample
        const r = Math.random();
        let cumsum = 0;
        for (let i = 0; i < scaled.length; i++) {
            cumsum += scaled[i];
            if (r < cumsum) return i;
        }

        return scaled.length - 1;
    }

    /**
     * Get memory statistics.
     */
    getMemoryStats(): {
        packedWeightsKB: number;
        fp16EquivalentKB: number;
        compressionRatio: number;
        scalesKB: number;
    } {
        // Packed ternary: 2 bits per weight = 0.25 bytes
        const packedWeightsKB = (this.ternaryWeights * 0.25) / 1024;

        // If stored as FP16: 2 bytes per weight
        const ternaryAsFP16KB = (this.ternaryWeights * 2) / 1024;
        const actualFP16KB = (this.fp16Weights * 2) / 1024;

        // Scales: 7 ternary layers per block * nLayers
        const numScales = this.blocks.reduce((acc, b) => {
            return acc +
                b.qProj.outFeatures +
                b.kvProj.outFeatures +
                b.proj.outFeatures +
                b.wGate.outFeatures +
                b.wUp.outFeatures +
                b.wDown.outFeatures;
        }, 0);
        const scalesKB = (numScales * 4) / 1024;  // FP32 scales

        return {
            packedWeightsKB: packedWeightsKB + scalesKB,
            fp16EquivalentKB: ternaryAsFP16KB + actualFP16KB,
            compressionRatio: (ternaryAsFP16KB + actualFP16KB) / (packedWeightsKB + scalesKB + actualFP16KB),
            scalesKB,
        };
    }

    /**
     * Get model config for UI.
     */
    getConfig(): {
        vocabSize: number;
        hiddenDim: number;
        contextLength: number;
        nLayers: number;
    } {
        return {
            vocabSize: this.config.vocabSize,
            hiddenDim: this.config.dim,
            contextLength: this.config.maxSeqLen,
            nLayers: this.config.nLayers,
        };
    }

    /**
     * Clean up resources.
     */
    destroy(): void {
        // Nothing to clean up for CPU
    }
}
