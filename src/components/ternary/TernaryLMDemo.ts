/**
 * TernaryLMDemo - Interactive browser demo for ternary neural network inference.
 *
 * This component provides a UI for text generation with an "under the hood"
 * panel showing memory savings and performance metrics.
 */

import { TransformerCPUEngine } from './TransformerCPUEngine';

/** Demo component configuration */
interface TernaryLMDemoConfig {
    containerId: string;
    modelPath: string;
    backgroundColor?: number;
    maxTokens?: number;
    defaultPrompt?: string;
}

/** Statistics during text generation */
interface GenerationStats {
    tokensPerSecond: number;
    totalTokens: number;
    elapsedMs: number;
}

/** Demo state for UI rendering */
interface DemoState {
    status: 'loading' | 'ready' | 'generating' | 'error';
    prompt: string;
    output: string;
    stats: GenerationStats | null;
    showUnderTheHood: boolean;
    showHowItWorks: boolean;
    errorMessage?: string;
}

// Common interface for both engines
interface InferenceEngine {
    generate(
        prompt: string,
        maxTokens: number,
        onToken?: (char: string, stats: GenerationStats) => void,
        temperature?: number
    ): Promise<string>;
    getMemoryStats(): { packedWeightsKB: number; fp16EquivalentKB: number; compressionRatio: number; scalesKB: number };
    getConfig(): { vocabSize: number; hiddenDim: number; contextLength: number; nLayers: number };
    destroy(): void;
    stop?(): void;  // Optional - only TransformerCPUEngine supports this
}

export class TernaryLMDemo {
    private container: HTMLElement;
    private config: TernaryLMDemoConfig;
    private engine: InferenceEngine | null = null;
    private usingCPU = false;
    private isTransformer = false;
    private state: DemoState;
    private isDestroyed = false;

    constructor(config: TernaryLMDemoConfig) {
        const container = document.getElementById(config.containerId);
        if (!container) {
            throw new Error(`Container #${config.containerId} not found`);
        }

        this.container = container;
        this.config = config;
        this.state = {
            status: 'loading',
            prompt: config.defaultPrompt || 'ROMEO: ',
            output: '',
            stats: null,
            showUnderTheHood: false,
            showHowItWorks: false,
        };

        this.init();
    }

    /**
     * Initialize the demo.
     */
    private async init(): Promise<void> {
        this.render();

        try {
            console.log('Loading transformer model...');
            this.engine = await TransformerCPUEngine.create(this.config.modelPath);

            // Check actual GPU status from engine
            const engineCPU = this.engine as TransformerCPUEngine;
            this.usingCPU = !engineCPU.isGPUEnabled();
            this.isTransformer = true;

            console.log('Transformer engine initialized, GPU:', engineCPU.isGPUEnabled());
        } catch (error) {
            console.error('Transformer engine failed:', error);
            this.state.status = 'error';
            this.state.errorMessage = error instanceof Error ? error.message : 'Failed to load model';
            this.showFallback();
            return;
        }

        this.state.status = 'ready';
        this.render();
    }

    /**
     * Render the demo UI.
     */
    private render(): void {
        if (this.isDestroyed) return;

        this.container.innerHTML = `
            <div class="bg-[#12121a] rounded-lg border border-gray-700/50 overflow-hidden shadow-xl h-full flex flex-col">
                <!-- Header -->
                <div class="px-4 py-2 border-b border-gray-700/50 flex items-center justify-between bg-[#0a0a0f] flex-shrink-0">
                    <div class="flex items-center gap-2">
                        <span class="text-[#00d4ff] text-sm">●</span>
                        <span class="text-sm font-medium text-gray-300">${this.isTransformer ? 'Ternary Transformer' : 'Ternary LM Demo'}</span>
                        <span class="text-xs text-gray-500 ml-2">1.58 bits/weight</span>
                        <span class="text-xs px-1.5 py-0.5 rounded ${this.usingCPU ? 'bg-yellow-500/20 text-yellow-400' : 'bg-green-500/20 text-green-400'}">
                            ${this.usingCPU ? 'CPU' : 'WebGPU'}
                        </span>
                    </div>
                    <div class="flex items-center gap-3">
                        <button
                            id="toggle-how-it-works"
                            class="text-xs text-gray-500 hover:text-[#00d4ff] transition-colors flex items-center gap-1"
                        >
                            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                            </svg>
                            <span>How it works</span>
                        </button>
                        <button
                            id="toggle-hood"
                            class="text-xs text-gray-500 hover:text-[#a855f7] transition-colors flex items-center gap-1"
                        >
                            ${this.state.showUnderTheHood ? 'Hide Details' : 'Under the Hood'}
                            <span class="text-[10px]">${this.state.showUnderTheHood ? '▲' : '▼'}</span>
                        </button>
                    </div>
                </div>

                <!-- Input Area -->
                <div class="p-3 flex-shrink-0">
                    <div class="flex gap-2 mb-3">
                        <input
                            type="text"
                            id="prompt-input"
                            value="${this.escapeHtml(this.state.prompt)}"
                            class="flex-1 px-3 py-2 bg-[#0a0a0f] border border-gray-700/50 rounded text-white text-sm
                                   focus:border-[#00d4ff] focus:ring-1 focus:ring-[#00d4ff]/30 outline-none
                                   placeholder-gray-600 font-mono"
                            placeholder="Enter prompt..."
                            ${this.state.status === 'generating' ? 'disabled' : ''}
                        />
                        ${this.state.status === 'generating' ? `
                        <button
                            id="stop-btn"
                            class="px-4 py-2 rounded text-sm font-medium transition-all
                                   bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/30"
                        >
                            Stop
                        </button>
                        ` : `
                        <button
                            id="generate-btn"
                            class="px-4 py-2 rounded text-sm font-medium transition-all
                                   ${this.state.status === 'ready'
                                       ? 'bg-[#00d4ff]/20 text-[#00d4ff] hover:bg-[#00d4ff]/30 border border-[#00d4ff]/30'
                                       : 'bg-gray-800 text-gray-500 cursor-not-allowed border border-gray-700'}"
                            ${this.state.status !== 'ready' ? 'disabled' : ''}
                        >
                            ${this.getButtonText()}
                        </button>
                        `}
                    </div>

                    <!-- Output Display -->
                    <div class="font-mono text-xs bg-[#0a0a0f] p-3 rounded border border-gray-700/50
                                min-h-[60px] max-h-[100px] overflow-y-auto">
                        <span class="text-gray-500">${this.escapeHtml(this.state.prompt)}</span><span
                            class="text-[#a855f7]"
                            id="generated-output"
                        >${this.escapeHtml(this.state.output)}</span>${this.state.status === 'generating'
                            ? '<span class="animate-pulse text-[#00d4ff]">▌</span>'
                            : ''}
                    </div>
                </div>

                <!-- Scrollable area for Stats + Under the Hood -->
                <div class="flex-1 overflow-y-auto min-h-0">
                    <!-- Stats Bar -->
                    ${this.state.stats ? this.renderStatsBar() : this.renderLoadingStats()}

                    <!-- Under the Hood Panel -->
                    ${this.state.showUnderTheHood ? this.renderUnderTheHood() : ''}
                </div>
            </div>

            <!-- How It Works Panel (slides in from right) -->
            <div id="how-it-works-panel" class="fixed top-0 right-0 h-full w-96 max-w-[90vw] border-l transform ${this.state.showHowItWorks ? 'translate-x-0' : 'translate-x-full'} transition-transform duration-300 ease-in-out z-50 overflow-y-auto" style="background-color: #0a0a0f; border-color: #1e1e2e;">
                ${this.renderHowItWorksPanel()}
            </div>

            <!-- Overlay when panel is open -->
            <div id="how-it-works-overlay" class="fixed inset-0 bg-black/50 ${this.state.showHowItWorks ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-300 z-40"></div>
        `;

        this.setupEventListeners();
    }

    /**
     * Get button text based on state.
     */
    private getButtonText(): string {
        switch (this.state.status) {
            case 'loading': return 'Loading...';
            case 'generating': return 'Generating...';
            case 'error': return 'Error';
            default: return 'Generate';
        }
    }

    /**
     * Render the statistics bar.
     */
    private renderStatsBar(): string {
        const stats = this.state.stats!;
        const memStats = this.engine?.getMemoryStats();

        return `
            <div class="px-3 py-1.5 bg-[#0a0a0f]/50 border-t border-gray-700/50
                        flex items-center gap-3 text-xs text-gray-400 flex-shrink-0">
                <span>
                    <span class="text-[#00d4ff] font-medium">${stats.tokensPerSecond.toFixed(1)}</span>
                    tok/s
                </span>
                <span>
                    <span class="text-[#a855f7] font-medium">${stats.totalTokens}</span>
                    tokens
                </span>
                ${memStats ? `
                    <span>
                        <span class="text-green-400 font-medium">${memStats.compressionRatio.toFixed(1)}×</span>
                        compression
                    </span>
                ` : ''}
                <span class="text-gray-600">|</span>
                <span class="text-gray-500">${stats.elapsedMs.toFixed(0)}ms</span>
            </div>
        `;
    }

    /**
     * Render loading state for stats bar.
     */
    private renderLoadingStats(): string {
        if (this.state.status === 'loading') {
            return `
                <div class="px-3 py-1.5 bg-[#0a0a0f]/50 border-t border-gray-700/50
                            flex items-center gap-2 text-xs text-gray-500 flex-shrink-0">
                    <div class="w-4 h-4 border-2 border-gray-700 border-t-[#00d4ff] rounded-full animate-spin"></div>
                    <span>Loading model...</span>
                </div>
            `;
        }
        return '';
    }

    /**
     * Render the "Under the Hood" panel.
     */
    private renderUnderTheHood(): string {
        const memStats = this.engine?.getMemoryStats();
        const config = this.engine?.getConfig();

        if (!memStats || !config) {
            return `
                <div class="border-t border-gray-700/50 p-4 text-sm text-gray-500">
                    Loading model information...
                </div>
            `;
        }

        const compressionPercent = (1 / memStats.compressionRatio) * 100;

        return `
            <div class="border-t border-gray-700/50 bg-[#0a0a0f]/30">
                <div class="p-3 grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
                    <!-- Memory Savings -->
                    <div class="bg-[#0a0a0f] rounded-lg p-3 border border-gray-700/30">
                        <div class="text-gray-400 text-xs mb-2 font-medium uppercase tracking-wide">
                            Memory Footprint
                        </div>
                        <div class="space-y-2">
                            <div>
                                <div class="flex justify-between text-xs text-gray-500 mb-1">
                                    <span>FP16 baseline</span>
                                    <span>${memStats.fp16EquivalentKB.toFixed(0)} KB</span>
                                </div>
                                <div class="h-2 bg-gray-700 rounded-full overflow-hidden">
                                    <div class="h-full bg-gray-500 rounded-full" style="width: 100%"></div>
                                </div>
                            </div>
                            <div>
                                <div class="flex justify-between text-xs mb-1">
                                    <span class="text-[#00d4ff]">Ternary (1.58-bit)</span>
                                    <span class="text-[#00d4ff]">${memStats.packedWeightsKB.toFixed(0)} KB</span>
                                </div>
                                <div class="h-2 bg-gray-700 rounded-full overflow-hidden">
                                    <div
                                        class="h-full bg-gradient-to-r from-[#00d4ff] to-[#a855f7] rounded-full"
                                        style="width: ${compressionPercent.toFixed(1)}%"
                                    ></div>
                                </div>
                            </div>
                        </div>
                        <div class="mt-3 text-xs text-gray-500">
                            <span class="text-green-400 font-medium">${memStats.compressionRatio.toFixed(1)}×</span>
                            smaller than FP16
                        </div>
                    </div>

                    <!-- Architecture Info -->
                    <div class="bg-[#0a0a0f] rounded-lg p-3 border border-gray-700/30">
                        <div class="text-gray-400 text-xs mb-2 font-medium uppercase tracking-wide">
                            ${this.isTransformer ? 'Transformer Architecture' : 'Model Architecture'}
                        </div>
                        <div class="space-y-2 text-xs">
                            <div class="flex justify-between">
                                <span class="text-gray-500">Vocabulary</span>
                                <span class="text-gray-300 font-mono">${config.vocabSize.toLocaleString()} ${this.isTransformer ? 'BPE' : 'chars'}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-500">${this.isTransformer ? 'Dim' : 'Hidden dim'}</span>
                                <span class="text-gray-300 font-mono">${config.hiddenDim}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-500">Context</span>
                                <span class="text-gray-300 font-mono">${config.contextLength.toLocaleString()} tokens</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-500">${this.isTransformer ? 'Blocks' : 'Layers'}</span>
                                <span class="text-gray-300 font-mono">${config.nLayers}</span>
                            </div>
                        </div>
                        <div class="mt-3 pt-2 border-t border-gray-700/50">
                            <div class="flex items-center gap-2 text-xs">
                                <span class="text-[#00d4ff]">●</span>
                                <span class="text-gray-400">Weights: {-1, 0, +1}</span>
                            </div>
                            ${this.isTransformer ? `
                            <div class="flex items-center gap-2 text-xs mt-1">
                                <span class="text-[#a855f7]">●</span>
                                <span class="text-gray-400">RMSNorm + RoPE + SwiGLU</span>
                            </div>
                            ` : ''}
                        </div>
                    </div>
                </div>

                <!-- Technical Note -->
                <div class="px-3 pb-3 text-xs text-gray-500">
                    Running entirely in your browser using
                    ${this.usingCPU
                        ? '<span class="text-yellow-400">JavaScript (CPU fallback)</span>'
                        : '<span class="text-[#00d4ff]">WebGPU</span> compute shaders'}.
                    <a
                        href="https://github.com/RetamalVictor/bittorch"
                        target="_blank"
                        class="text-[#a855f7] hover:underline ml-1"
                    >View source →</a>
                </div>
            </div>
        `;
    }

    /**
     * Render the "How It Works" sliding panel.
     */
    private renderHowItWorksPanel(): string {
        // Color constants (matching IBVS panel)
        const cyan = '#00d4ff';
        const purple = '#a855f7';
        const yellow = '#facc15';
        const darkBg = '#0a0a0f';

        return `
            <!-- Panel Header -->
            <div class="sticky top-0 border-b p-4 flex items-center justify-between" style="background-color: ${darkBg}; border-color: #1e1e2e;">
                <h2 class="text-lg font-semibold" style="color: ${cyan};">Ternary Inference</h2>
                <button id="how-it-works-close" class="p-1 rounded text-gray-400 hover:text-white transition-colors">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                </button>
            </div>

            <!-- Panel Content -->
            <div class="p-4 space-y-6">
                <!-- Overview -->
                <div>
                    <h3 class="font-medium mb-2 text-white">Overview</h3>
                    <p class="text-gray-400 leading-relaxed">
                        This demo runs a ternary-quantized transformer entirely in the browser.
                        Model weights are restricted to <strong class="text-white">{-1, 0, +1}</strong> and stored using 2-bit packing,
                        achieving <strong class="text-white">~8x compression vs FP16</strong> (~16x vs FP32), with small per-channel scale overhead.
                    </p>
                    <p class="text-gray-500 text-sm mt-2">
                        Inference-only. Optimized for bandwidth efficiency rather than raw arithmetic throughput.
                    </p>
                </div>

                <!-- Weight Encoding -->
                <div>
                    <h3 class="font-medium mb-2 text-white">Weight Encoding</h3>
                    <p class="text-gray-400 mb-3">
                        Each weight is encoded using 2 bits, packed four per byte:
                    </p>
                    <div class="rounded-lg p-3 font-mono text-sm space-y-1" style="background-color: ${darkBg};">
                        <div class="text-gray-300"><span style="color: ${cyan};">00</span> →  <span class="text-white">0</span></div>
                        <div class="text-gray-300"><span style="color: ${cyan};">01</span> → <span class="text-white">+1</span></div>
                        <div class="text-gray-300"><span style="color: ${cyan};">10</span> → <span class="text-white">-1</span></div>
                    </div>
                    <p class="text-gray-500 text-sm mt-2">
                        Per-output-channel scales (<span style="color: ${yellow};">FP16/FP32</span>) restore magnitude:
                        <strong class="text-gray-400">W = scale * W_ternary</strong>
                    </p>
                </div>

                <!-- On-Demand Unpacking -->
                <div>
                    <h3 class="font-medium mb-2 text-white">On-Demand Unpacking</h3>
                    <p class="text-gray-400 mb-3">
                        Weights are <strong class="text-white">never pre-expanded</strong> to INT8 or FP16. Decoding happens inside the dot product:
                    </p>
                    <div class="rounded-lg p-3 font-mono text-sm overflow-x-auto space-y-1" style="background-color: ${darkBg};">
                        <div class="text-gray-300"><span style="color: ${purple};">let</span> word: <span style="color: ${cyan};">u32</span> = packed[wordIdx];</div>
                        <div class="text-gray-300"><span style="color: ${purple};">let</span> code = (word >> bitOffset) & <span style="color: ${yellow};">0x3u</span>;</div>
                        <div class="text-gray-300"><span style="color: ${purple};">let</span> w = <span style="color: ${cyan};">decode_ternary</span>(code);</div>
                        <div class="text-gray-300">acc += input[k] * w;</div>
                    </div>
                    <div class="mt-3 rounded-lg p-3 text-sm" style="background-color: ${darkBg};">
                        <table class="w-full text-left">
                            <thead class="text-gray-500 border-b" style="border-color: #1e1e2e;">
                                <tr>
                                    <th class="pb-2 font-medium">Strategy</th>
                                    <th class="pb-2 font-medium">Traffic</th>
                                    <th class="pb-2 font-medium">Notes</th>
                                </tr>
                            </thead>
                            <tbody class="text-gray-400">
                                <tr class="border-b" style="border-color: #1e1e2e50;">
                                    <td class="py-2">Pre-expanded</td>
                                    <td class="py-2 text-gray-500">4-8x</td>
                                    <td class="py-2 text-gray-500">Higher memory, cache pressure</td>
                                </tr>
                                <tr>
                                    <td class="py-2" style="color: ${cyan};">On-demand</td>
                                    <td class="py-2" style="color: ${cyan};">1x</td>
                                    <td class="py-2 text-gray-500">Decode cost amortized</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Why This Is Efficient -->
                <div>
                    <h3 class="font-medium mb-2 text-white">Why This Is Efficient</h3>
                    <div class="space-y-3 text-sm text-gray-400">
                        <div>
                            <strong class="text-white">1. Bandwidth dominates.</strong>
                            These kernels are typically limited by memory bandwidth. Smaller weights reduce traffic and improve cache residency.
                        </div>
                        <div>
                            <strong class="text-white">2. Decode cost is low.</strong>
                            Bitwise shifts and masks are inexpensive relative to fetching wider formats.
                        </div>
                        <div>
                            <strong class="text-white">3. Browser-specific wins.</strong>
                            Packed weights reduce download size, parse overhead, peak memory, and GPU upload time.
                        </div>
                    </div>
                    <p class="text-gray-500 text-sm mt-3">
                        Performance benefit comes from <span style="color: ${cyan};">data movement reduction</span>, not from eliminating multiplications.
                    </p>
                </div>

                <!-- Runtime -->
                <div>
                    <h3 class="font-medium mb-2 text-white">${this.usingCPU ? 'CPU' : 'WebGPU'} Execution</h3>
                    ${this.usingCPU ? `
                    <p class="text-gray-400">
                        WebGPU unavailable. Running CPU fallback with same packed format and decode logic.
                        Single-threaded JavaScript—<strong style="color: ${yellow};">significantly slower</strong>.
                    </p>
                    <p class="text-gray-500 text-sm mt-2">
                        WebGPU or WASM SIMD would provide substantial speedups.
                    </p>
                    ` : `
                    <p class="text-gray-400">
                        Running on <span style="color: ${cyan};">WebGPU</span> with compute shaders. Weights remain packed in GPU buffers; decoding happens at execution time.
                    </p>
                    `}
                </div>

                <!-- Data Flow -->
                <div>
                    <h3 class="font-medium mb-2 text-white">Data Flow</h3>
                    <p class="text-gray-500 text-sm mb-3">
                        Weights remain packed throughout the entire pipeline. No intermediate expanded representations.
                    </p>
                    <div class="rounded-lg p-3 font-mono text-sm text-gray-300 space-y-1" style="background-color: ${darkBg};">
                        <div><span style="color: ${cyan};">SafeTensors</span> <span class="text-gray-500">(packed)</span></div>
                        <div class="text-gray-600 pl-4">↓</div>
                        <div><span style="color: ${cyan};">Uint8Array</span></div>
                        <div class="text-gray-600 pl-4">↓</div>
                        <div><span style="color: ${yellow};">GPU Buffer</span></div>
                        <div class="text-gray-600 pl-4">↓ <span style="color: ${purple};">on-the-fly decode</span></div>
                        <div><span style="color: ${yellow};">Compute Shader</span></div>
                        <div class="text-gray-600 pl-4">↓</div>
                        <div><span class="text-white">Output</span> <span class="text-gray-500">(Float32)</span></div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Show fallback UI when WebGPU is unavailable.
     */
    private showFallback(): void {
        const isWebGPUError = this.state.errorMessage?.includes('WebGPU');

        this.container.innerHTML = `
            <div class="bg-[#12121a] rounded-lg border border-gray-700/50 p-8 text-center">
                <div class="w-16 h-16 mx-auto mb-4 rounded-full bg-red-500/10
                            flex items-center justify-center border border-red-500/30">
                    <svg class="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                    </svg>
                </div>
                <h3 class="text-lg font-medium text-white mb-2">
                    ${isWebGPUError ? 'WebGPU Not Available' : 'Demo Unavailable'}
                </h3>
                <p class="text-gray-400 text-sm mb-4 max-w-md mx-auto">
                    ${isWebGPUError
                        ? 'This interactive demo requires WebGPU support. Try using Chrome 113+, Edge 113+, or another browser with WebGPU enabled.'
                        : `Failed to initialize: ${this.state.errorMessage}`}
                </p>
                <div class="flex gap-3 justify-center">
                    <a
                        href="https://github.com/RetamalVictor/bittorch"
                        target="_blank"
                        class="px-4 py-2 bg-[#00d4ff]/20 text-[#00d4ff] rounded text-sm
                               hover:bg-[#00d4ff]/30 transition-colors border border-[#00d4ff]/30"
                    >
                        View Code on GitHub
                    </a>
                    <a
                        href="https://caniuse.com/webgpu"
                        target="_blank"
                        class="px-4 py-2 bg-gray-800 text-gray-300 rounded text-sm
                               hover:bg-gray-700 transition-colors border border-gray-700"
                    >
                        Check Browser Support
                    </a>
                </div>
            </div>
        `;
    }

    /**
     * Set up event listeners.
     */
    private setupEventListeners(): void {
        // Generate button
        const generateBtn = document.getElementById('generate-btn');
        if (generateBtn) {
            generateBtn.addEventListener('click', () => this.handleGenerate());
        }

        // Stop button
        const stopBtn = document.getElementById('stop-btn');
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.handleStop());
        }

        // Prompt input - enter key
        const promptInput = document.getElementById('prompt-input') as HTMLInputElement;
        if (promptInput) {
            promptInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && this.state.status === 'ready') {
                    this.state.prompt = promptInput.value;
                    this.handleGenerate();
                }
            });
            promptInput.addEventListener('input', () => {
                this.state.prompt = promptInput.value;
            });
        }

        // Toggle under the hood
        const toggleBtn = document.getElementById('toggle-hood');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                this.state.showUnderTheHood = !this.state.showUnderTheHood;
                this.render();
            });
        }

        // Toggle how it works panel
        const howItWorksToggle = document.getElementById('toggle-how-it-works');
        if (howItWorksToggle) {
            howItWorksToggle.addEventListener('click', () => {
                this.state.showHowItWorks = true;
                this.render();
            });
        }

        // Close how it works panel
        const howItWorksClose = document.getElementById('how-it-works-close');
        if (howItWorksClose) {
            howItWorksClose.addEventListener('click', () => {
                this.state.showHowItWorks = false;
                this.render();
            });
        }

        // Overlay click closes panel
        const howItWorksOverlay = document.getElementById('how-it-works-overlay');
        if (howItWorksOverlay) {
            howItWorksOverlay.addEventListener('click', () => {
                this.state.showHowItWorks = false;
                this.render();
            });
        }

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.state.showHowItWorks) {
                this.state.showHowItWorks = false;
                this.render();
            }
        });
    }

    /**
     * Handle stop button click.
     */
    private handleStop(): void {
        if (this.engine?.stop) {
            this.engine.stop();
        }
    }

    /**
     * Handle generate button click.
     */
    private async handleGenerate(): Promise<void> {
        if (!this.engine || this.state.status !== 'ready') return;

        this.state.status = 'generating';
        this.state.output = '';
        this.state.stats = null;
        this.render();

        try {
            await this.engine.generate(
                this.state.prompt,
                this.config.maxTokens || 100,
                (char, stats) => {
                    this.state.output += char;
                    this.state.stats = stats;

                    // Update only the output span for performance
                    const outputEl = document.getElementById('generated-output');
                    if (outputEl) {
                        outputEl.textContent = this.state.output;
                    }

                    // Update stats bar if visible
                    if (this.state.stats) {
                        this.updateStatsBar(stats);
                    }
                },
                0.8  // temperature
            );
        } catch (error) {
            console.error('Generation error:', error);
            this.state.errorMessage = error instanceof Error ? error.message : 'Generation failed';
        } finally {
            this.state.status = 'ready';
            this.render();
        }
    }

    /**
     * Update stats bar without full re-render.
     */
    private updateStatsBar(stats: GenerationStats): void {
        // Find stats bar and update inline
        const statsBar = this.container.querySelector('.text-xs.text-gray-400');
        if (statsBar && this.state.stats) {
            const memStats = this.engine?.getMemoryStats();
            statsBar.innerHTML = `
                <span>
                    <span class="text-[#00d4ff] font-medium">${stats.tokensPerSecond.toFixed(1)}</span>
                    tok/s
                </span>
                <span>
                    <span class="text-[#a855f7] font-medium">${stats.totalTokens}</span>
                    tokens
                </span>
                ${memStats ? `
                    <span>
                        <span class="text-green-400 font-medium">${memStats.compressionRatio.toFixed(1)}×</span>
                        compression
                    </span>
                ` : ''}
                <span class="text-gray-600">|</span>
                <span class="text-gray-500">${stats.elapsedMs.toFixed(0)}ms</span>
            `;
        }
    }

    /**
     * Escape HTML special characters.
     */
    private escapeHtml(text: string): string {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Clean up resources.
     */
    public destroy(): void {
        this.isDestroyed = true;
        this.engine?.destroy();
    }
}
