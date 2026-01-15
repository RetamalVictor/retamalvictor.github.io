/**
 * BitTorchAdapter - Wraps @bittorch/js for use with TernaryLMDemo
 *
 * This adapter implements the InferenceEngine interface expected by TernaryLMDemo,
 * delegating all inference work to the bittorch.js library.
 */

import { TernaryTransformer } from '@bittorch/js';

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

/** Model configuration */
export interface ModelConfig {
  vocabSize: number;
  hiddenDim: number;
  contextLength: number;
  nLayers: number;
}

/**
 * Adapter wrapping TernaryTransformer for the blog's InferenceEngine interface.
 *
 * @example
 * ```typescript
 * const engine = await BitTorchAdapter.create('/path/to/model');
 * const output = await engine.generate('Hello', 50, (token, stats) => {
 *   console.log(token, stats.tokensPerSecond);
 * });
 * engine.destroy();
 * ```
 */
export class BitTorchAdapter {
  private model: TernaryTransformer;

  private constructor(model: TernaryTransformer) {
    this.model = model;
  }

  /**
   * Create a new adapter by loading a model.
   *
   * @param modelPath - URL or path to model directory
   * @returns Promise resolving to BitTorchAdapter instance
   */
  static async create(modelPath: string): Promise<BitTorchAdapter> {
    const model = await TernaryTransformer.load(modelPath);
    return new BitTorchAdapter(model);
  }

  /**
   * Generate text continuation.
   *
   * @param prompt - Starting text
   * @param maxTokens - Maximum tokens to generate
   * @param onToken - Callback for each generated token
   * @param temperature - Sampling temperature (default: 0.8)
   * @returns Generated text (excluding prompt)
   */
  async generate(
    prompt: string,
    maxTokens: number,
    onToken?: (char: string, stats: GenerationStats) => void,
    temperature = 0.8
  ): Promise<string> {
    return this.model.generate(prompt, {
      maxTokens,
      temperature,
      onToken,
    });
  }

  /**
   * Get memory usage statistics.
   */
  getMemoryStats(): MemoryStats {
    return this.model.memoryStats;
  }

  /**
   * Get model configuration.
   */
  getConfig(): ModelConfig {
    const c = this.model.config;
    return {
      vocabSize: c.vocabSize,
      hiddenDim: c.dim,
      contextLength: c.maxSeqLen,
      nLayers: c.nLayers,
    };
  }

  /**
   * Check if GPU acceleration is enabled.
   */
  isGPUEnabled(): boolean {
    return this.model.isGPUEnabled;
  }

  /**
   * Stop ongoing generation.
   */
  stop(): void {
    this.model.stop();
  }

  /**
   * Release all resources.
   */
  destroy(): void {
    this.model.destroy();
  }
}
