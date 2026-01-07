/**
 * Type definitions for the ternary neural network browser demo.
 */

/** Model configuration loaded from .tbin header */
export interface ModelConfig {
    vocabSize: number;
    hiddenDim: number;
    contextLength: number;
    nLayers: number;
}

/** Layer types in the binary format */
export enum LayerType {
    EMBEDDING = 0,
    TERNARY_LINEAR = 1,
    FP16_LINEAR = 2,
    LAYERNORM = 3,
}

/** Metadata for a single layer */
export interface LayerMeta {
    type: LayerType;
    inFeatures: number;
    outFeatures: number;
    hasBias: boolean;
    dataOffset: number;
    scaleOffset: number;
    biasOffset: number;
}

/** GPU buffers for a layer */
export interface LayerBuffers {
    type: LayerType;
    inFeatures: number;
    outFeatures: number;
    weights: GPUBuffer;       // Packed ternary or FP16
    scales?: GPUBuffer;       // Per-channel scales (ternary only)
    bias?: GPUBuffer;         // Optional bias
}

/** Statistics during text generation */
export interface GenerationStats {
    tokensPerSecond: number;
    totalTokens: number;
    elapsedMs: number;
}

/** Memory usage statistics */
export interface MemoryStats {
    packedWeightsKB: number;
    fp16EquivalentKB: number;
    compressionRatio: number;
    scalesKB: number;
}

/** Weight distribution for visualization */
export interface WeightDistribution {
    negativeOne: number;  // Count of -1 weights
    zero: number;         // Count of 0 weights
    positiveOne: number;  // Count of +1 weights
    total: number;
}

/** Demo component configuration */
export interface TernaryLMDemoConfig {
    containerId: string;
    modelPath: string;
    backgroundColor?: number;
    maxTokens?: number;
    defaultPrompt?: string;
}

/** Demo state for UI rendering */
export interface DemoState {
    status: 'loading' | 'ready' | 'generating' | 'error';
    prompt: string;
    output: string;
    stats: GenerationStats | null;
    showUnderTheHood: boolean;
    errorMessage?: string;
}

/** Vocabulary mapping from JSON sidecar */
export interface VocabMapping {
    chars: string;
    charToIdx: Record<string, number>;
    vocabSize: number;
}

/** Binary format constants */
export const TBIN_MAGIC = 0x54424E31;  // "TBN1"
export const TBIN_VERSION = 1;
export const HEADER_SIZE = 64;
export const LAYER_META_SIZE = 32;
