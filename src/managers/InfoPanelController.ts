import type { DemoType } from './DemoManager.js';

/**
 * Info panel content structure
 */
export interface InfoPanelContent {
    title: string;
    content: string;
}

/**
 * Info panel content for each demo type
 */
export const INFO_PANEL_CONTENT: Record<DemoType, InfoPanelContent> = {
    'ibvs': {
        title: 'Visual Servoing Demo',
        content: `
            <!-- Overview -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Overview</h3>
                <p class="text-gray-400 leading-relaxed">
                    This demo shows <strong class="text-white">Image-Based Visual Servoing (IBVS)</strong>
                    controlling a quadrotor to track a target using only camera feedback.
                    The drone tilts to move, demonstrating underactuated dynamics.
                </p>
            </div>

            <!-- IBVS Controller -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">IBVS Controller</h3>
                <p class="text-gray-400 mb-3">
                    The controller minimizes feature error in image space:
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs space-y-2">
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">e</span> = <span class="text-accent-purple">s*</span> - <span class="text-white">s</span>
                        <span class="text-gray-500 ml-2">// feature error</span>
                    </div>
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">v</span> = <span class="text-yellow-400">λ</span> · <span class="text-white">L</span><sup>+</sup> · <span class="text-accent-cyan">e</span>
                        <span class="text-gray-500 ml-2">// control law</span>
                    </div>
                </div>
                <p class="text-gray-500 text-xs mt-2">
                    <strong class="text-gray-400">L</strong> is the interaction matrix relating feature motion to camera velocity.
                    <strong class="text-gray-400">L<sup>+</sup></strong> is its pseudo-inverse.
                </p>
            </div>

            <!-- Interaction Matrix -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Interaction Matrix</h3>
                <p class="text-gray-400 mb-3">
                    For each image point (x, y) at depth Z:
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs overflow-x-auto">
                    <div class="text-gray-300 whitespace-nowrap">
                        L = [<span class="text-accent-cyan">-f/Z</span>, 0, <span class="text-accent-cyan">x/Z</span>, <span class="text-yellow-400">xy/f</span>, <span class="text-yellow-400">-(f+x²/f)</span>, <span class="text-yellow-400">y</span>]
                    </div>
                    <div class="text-gray-300 whitespace-nowrap mt-1">
                        &nbsp;&nbsp;&nbsp;&nbsp;[0, <span class="text-accent-cyan">-f/Z</span>, <span class="text-accent-cyan">y/Z</span>, <span class="text-yellow-400">f+y²/f</span>, <span class="text-yellow-400">-xy/f</span>, <span class="text-yellow-400">-x</span>]
                    </div>
                </div>
                <p class="text-gray-500 text-xs mt-2">
                    <span class="text-accent-cyan">Cyan</span>: translation terms,
                    <span class="text-yellow-400">Yellow</span>: rotation terms
                </p>
            </div>

            <!-- Quadrotor Dynamics -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Quadrotor Dynamics</h3>
                <p class="text-gray-400 mb-3">
                    Underactuated model: lateral motion requires tilting.
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs space-y-2">
                    <div class="text-gray-500">// Velocity to tilt (feedforward + damping)</div>
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">θ</span><sub>des</sub> = <span class="text-yellow-400">k<sub>p</sub></span>·v<sub>des</sub> - <span class="text-yellow-400">k<sub>d</sub></span>·v
                    </div>
                    <div class="text-gray-500 mt-2">// Attitude controller (PD)</div>
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">τ</span> = <span class="text-yellow-400">K<sub>p</sub></span>·(θ<sub>des</sub> - θ) - <span class="text-yellow-400">K<sub>d</sub></span>·ω
                    </div>
                    <div class="text-gray-500 mt-2">// Thrust produces acceleration</div>
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">a</span><sub>x</sub> = -T·sin(θ<sub>roll</sub>)
                    </div>
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">a</span><sub>z</sub> = T·sin(θ<sub>pitch</sub>)
                    </div>
                </div>
            </div>

            <!-- Camera Model -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Pinhole Camera</h3>
                <p class="text-gray-400 mb-3">
                    Projects 3D world points to 2D image coordinates:
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs space-y-2">
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">u</span> = f·(X/Z) + c<sub>x</sub>
                    </div>
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">v</span> = f·(Y/Z) + c<sub>y</sub>
                    </div>
                </div>
                <p class="text-gray-500 text-xs mt-2">
                    <strong class="text-gray-400">f</strong>: focal length,
                    <strong class="text-gray-400">(c<sub>x</sub>, c<sub>y</sub>)</strong>: principal point
                </p>
            </div>

            <!-- Parameters -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Parameters</h3>
                <div class="bg-dark-bg rounded-lg p-3 text-xs space-y-1">
                    <div class="flex justify-between">
                        <span class="text-gray-400">IBVS gain (λ)</span>
                        <span class="text-white font-mono">0.5</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Velocity gain (k<sub>p</sub>)</span>
                        <span class="text-white font-mono">0.5</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Velocity damping (k<sub>d</sub>)</span>
                        <span class="text-white font-mono">0.3</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Focal length</span>
                        <span class="text-white font-mono">50mm</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Desired distance</span>
                        <span class="text-white font-mono">2.0m</span>
                    </div>
                </div>
            </div>

            <!-- References -->
            <div class="info-section border-t border-dark-border pt-4">
                <h3 class="text-accent-purple font-medium mb-2">References</h3>
                <ul class="text-gray-500 text-xs space-y-1">
                    <li>Chaumette & Hutchinson, "Visual Servo Control" (2006)</li>
                    <li>Corke, "Robotics, Vision and Control" (2017)</li>
                </ul>
            </div>
        `
    },
    'ternary': {
        title: 'Ternary Language Model',
        content: `
            <!-- Overview -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Overview</h3>
                <p class="text-gray-400 leading-relaxed">
                    This demo shows a <strong class="text-white">Ternary Quantized Language Model</strong>
                    running entirely in the browser. Weights are compressed to {-1, 0, +1}, enabling
                    efficient inference with minimal memory footprint.
                </p>
            </div>

            <!-- Architecture -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Architecture</h3>
                <p class="text-gray-400 mb-3">
                    Transformer-based model with ternary weights:
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs space-y-2">
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">W</span> ∈ {-1, 0, +1}<sup>d×d</sup>
                    </div>
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">y</span> = W · x <span class="text-gray-500">// No multiplications!</span>
                    </div>
                </div>
                <p class="text-gray-500 text-xs mt-2">
                    Multiplications become additions/subtractions, enabling fast CPU inference.
                </p>
            </div>

            <!-- Quantization -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Quantization</h3>
                <p class="text-gray-400 mb-3">
                    Ternary quantization reduces memory by ~16x:
                </p>
                <div class="bg-dark-bg rounded-lg p-3 text-xs space-y-1">
                    <div class="flex justify-between">
                        <span class="text-gray-400">FP32 weights</span>
                        <span class="text-white font-mono">32 bits/param</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Ternary weights</span>
                        <span class="text-white font-mono">2 bits/param</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Compression</span>
                        <span class="text-accent-cyan font-mono">16x</span>
                    </div>
                </div>
            </div>

            <!-- Model Details -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Model Details</h3>
                <div class="bg-dark-bg rounded-lg p-3 text-xs space-y-1">
                    <div class="flex justify-between">
                        <span class="text-gray-400">Parameters</span>
                        <span class="text-white font-mono">~1M</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Layers</span>
                        <span class="text-white font-mono">4</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Hidden dim</span>
                        <span class="text-white font-mono">256</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Vocab size</span>
                        <span class="text-white font-mono">65</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Training data</span>
                        <span class="text-white font-mono">Shakespeare</span>
                    </div>
                </div>
            </div>

            <!-- References -->
            <div class="info-section border-t border-dark-border pt-4">
                <h3 class="text-accent-purple font-medium mb-2">References</h3>
                <ul class="text-gray-500 text-xs space-y-1">
                    <li>Ma et al., "The Era of 1-bit LLMs" (2024)</li>
                    <li>Karpathy, "nanoGPT" (2023)</li>
                </ul>
            </div>
        `
    },
    'drone-racing': {
        title: 'Drone Racing Demo',
        content: `
            <!-- Overview -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Overview</h3>
                <p class="text-gray-400 leading-relaxed">
                    This demo showcases <strong class="text-white">Model Predictive Control (MPC)</strong> for autonomous drone racing.
                    A quadrotor follows a racing trajectory while an MPC controller computes optimal control inputs in real-time.
                </p>
            </div>

            <!-- MPC Controller -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">MPC Controller</h3>
                <p class="text-gray-400 mb-3">
                    Model Predictive Control optimizes future control inputs by predicting system behavior:
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs space-y-2">
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">min</span> Σ (x - x<sub>ref</sub>)ᵀQ(x - x<sub>ref</sub>) + uᵀRu
                    </div>
                    <div class="text-gray-500 text-xs mt-2">subject to: dynamics, input constraints</div>
                </div>
            </div>

            <!-- Quadrotor Dynamics -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Quadrotor Dynamics</h3>
                <p class="text-gray-400 mb-3">
                    6-DOF rigid body model with rate control:
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs space-y-1">
                    <div class="text-gray-300">
                        <span class="text-accent-purple">Inputs:</span> thrust, roll rate, pitch rate, yaw rate
                    </div>
                    <div class="text-gray-300">
                        <span class="text-accent-purple">State:</span> position (x,y,z), velocity, orientation
                    </div>
                </div>
            </div>

            <!-- Track -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Racing Track</h3>
                <p class="text-gray-400 mb-3">
                    The track features a power loop maneuver with stacked gates:
                </p>
                <div class="bg-dark-bg rounded-lg p-3 text-xs space-y-1">
                    <div class="flex justify-between">
                        <span class="text-gray-400">Track type</span>
                        <span class="text-white font-mono">Power Loop</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Gates</span>
                        <span class="text-white font-mono">4 (2 stacked pairs)</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Target speed</span>
                        <span class="text-white font-mono">18 m/s</span>
                    </div>
                </div>
            </div>

            <!-- Controls -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Controls</h3>
                <div class="bg-dark-bg rounded-lg p-3 text-xs space-y-1">
                    <div class="flex justify-between">
                        <span class="text-gray-400">Mouse drag</span>
                        <span class="text-white">Orbit camera</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Scroll</span>
                        <span class="text-white">Zoom in/out</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Overview/Follow</span>
                        <span class="text-white">Camera modes</span>
                    </div>
                </div>
            </div>

            <!-- References -->
            <div class="info-section border-t border-dark-border pt-4">
                <h3 class="text-accent-purple font-medium mb-2">References</h3>
                <ul class="text-gray-500 text-xs space-y-1">
                    <li>Torrente et al., "Data-Driven MPC for Quadrotors" (2021)</li>
                    <li>Foehn et al., "Time-Optimal Planning for Quadrotor Waypoint Flight" (2021)</li>
                </ul>
            </div>
        `
    }
};

/**
 * Controls the info panel (slide-in panel with demo explanations)
 */
export class InfoPanelController {
    private panel: HTMLElement | null = null;
    private overlay: HTMLElement | null = null;
    private isOpen: boolean = false;
    private boundHandleEscape: (e: KeyboardEvent) => void;

    constructor() {
        this.boundHandleEscape = this.handleEscape.bind(this);
    }

    /**
     * Initialize the panel with DOM elements
     */
    public initialize(): void {
        this.panel = document.getElementById('info-panel');
        this.overlay = document.getElementById('info-panel-overlay');

        const toggleBtn = document.getElementById('info-panel-toggle');
        const closeBtn = document.getElementById('info-panel-close');

        if (!this.panel || !this.overlay || !toggleBtn) {
            console.warn('InfoPanelController: required elements not found');
            return;
        }

        toggleBtn.addEventListener('click', () => this.open());
        closeBtn?.addEventListener('click', () => this.close());
        this.overlay.addEventListener('click', () => this.close());

        document.addEventListener('keydown', this.boundHandleEscape);
    }

    /**
     * Handle escape key to close panel
     */
    private handleEscape(e: KeyboardEvent): void {
        if (e.key === 'Escape' && this.isOpen) {
            this.close();
        }
    }

    /**
     * Open the info panel
     */
    public open(): void {
        if (!this.panel || !this.overlay) return;

        this.panel.classList.remove('translate-x-full');
        this.overlay.classList.remove('opacity-0', 'pointer-events-none');
        this.overlay.classList.add('opacity-100', 'pointer-events-auto');
        this.isOpen = true;
    }

    /**
     * Close the info panel
     */
    public close(): void {
        if (!this.panel || !this.overlay) return;

        this.panel.classList.add('translate-x-full');
        this.overlay.classList.add('opacity-0', 'pointer-events-none');
        this.overlay.classList.remove('opacity-100', 'pointer-events-auto');
        this.isOpen = false;
    }

    /**
     * Toggle the panel open/closed
     */
    public toggle(): void {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }

    /**
     * Update panel content for a specific demo type
     */
    public setContent(demoType: DemoType): void {
        const title = document.getElementById('info-panel-title');
        const content = document.getElementById('info-panel-content');
        if (!title || !content) return;

        const panelData = INFO_PANEL_CONTENT[demoType];
        title.textContent = panelData.title;
        content.innerHTML = panelData.content;
    }

    /**
     * Check if panel is currently open
     */
    public isCurrentlyOpen(): boolean {
        return this.isOpen;
    }

    /**
     * Cleanup event listeners
     */
    public destroy(): void {
        document.removeEventListener('keydown', this.boundHandleEscape);
    }
}
