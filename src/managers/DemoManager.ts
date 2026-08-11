import { VisualServoDemo } from '../components/VisualServoDemo.js';
// Shared with the depth demo, which uses it to pick which model to fetch.
import { isMemoryConstrained } from '../utils/deviceCapability.js';

/**
 * Types for the home page demo tabs
 */
export type DemoType = 'ibvs' | 'ternary' | 'drone-racing' | 'depth' | 'mapf';

/**
 * Interface for demo instances
 */
export interface DemoInstance {
    destroy(): void;
    reset?(): void;
    resetSimulation?(): void;
}

/**
 * Demo hints shown in the UI for each demo type
 */
export const DEMO_HINTS: Record<DemoType, string> = {
    'ibvs': 'Drag the target',
    'ternary': 'Enter a prompt',
    'drone-racing': 'Use mouse to orbit',
    'depth': 'Allow camera access',
    'mapf': 'Drag the sliders',
};

/**
 * Callback type for when demo changes
 */
export type DemoChangeCallback = (demoType: DemoType) => void;

/**
 * Roughly what a demo downloads on a device we consider constrained.
 *
 * Depth is absent deliberately: it now serves a 5.5 MB model to phones and
 * keeps the 66 MB one for everything else, so there is nothing to warn about.
 * The ternary demo has only one model and it is 39 MB.
 */
const DEMO_DOWNLOAD_MB: Partial<Record<DemoType, number>> = {
    'ternary': 39,
};

/** Above this, ask before downloading on a device that may not survive it. */
const CONFIRM_ABOVE_MB = 20;

/**
 * Manages demo lifecycle: switching, reset, destroy
 */
export class DemoManager {
    private containerId: string;
    private activeDemo: DemoType = 'drone-racing';
    private currentInstance: DemoInstance | null = null;
    private onDemoChange: DemoChangeCallback | null = null;
    /** Heavy demos the reader has already accepted the download for. */
    private confirmed = new Set<DemoType>();
    /** The element this manager actually mounted into. */
    private mountedOn: HTMLElement | null = null;

    constructor(containerId: string) {
        this.containerId = containerId;
    }

    /**
     * Initialize the default demo (IBVS) and setup tabs
     */
    public async initialize(): Promise<void> {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`DemoManager: container #${this.containerId} not found`);
            return;
        }

        // Initialize default demo (Drone Racing)
        container.innerHTML = '';
        this.mountedOn = container;
        await this.createDemo(this.activeDemo, container);

        this.setupTabs();
    }

    /**
     * Is this manager still attached to the element currently on the page?
     *
     * Navigating away and back rebuilds the hero, so the element this mounted
     * into is replaced by a fresh one carrying its "Loading 3D scene"
     * placeholder. A manager holding the old node will never paint again, and
     * worse, keeps its demo running against a detached element that nobody can
     * see and no visibility observer will pause.
     */
    public isMounted(): boolean {
        return this.mountedOn !== null
            && this.mountedOn.isConnected
            && this.mountedOn === document.getElementById(this.containerId);
    }

    /**
     * Set callback for when demo changes
     */
    public onDemoChanged(callback: DemoChangeCallback): void {
        this.onDemoChange = callback;
    }

    /**
     * Setup demo tab click handlers
     */
    private setupTabs(): void {
        const tabs = document.querySelectorAll('#demo-tabs .demo-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const demoType = tab.getAttribute('data-demo') as DemoType;
                if (demoType && demoType !== this.activeDemo) {
                    this.switchTo(demoType);
                }
            });
        });
    }

    /**
     * Switch to a different demo type
     */
    public async switchTo(demoType: DemoType): Promise<void> {
        // Destroy current demo
        this.destroyCurrentInstance();

        // Clear container and show loading
        const container = document.getElementById(this.containerId);
        if (!container) return;
        container.innerHTML = '<div class="text-center text-gray-500 p-8"><div class="animate-pulse">Loading demo...</div></div>';

        // Update active tab styling
        this.updateTabStyles(demoType);

        // Update hint text
        this.updateHintText(demoType);

        this.activeDemo = demoType;

        // Notify listeners
        if (this.onDemoChange) {
            this.onDemoChange(demoType);
        }

        // Create new demo
        try {
            await this.createDemo(demoType, container);
        } catch (error) {
            console.error(`Failed to load ${demoType} demo:`, error);
            container.innerHTML = `<div class="text-center text-red-400 p-8">Failed to load demo</div>`;
        }
    }

    /**
     * Create the demo instance based on type
     */
    private async createDemo(demoType: DemoType, container: HTMLElement): Promise<void> {
        container.innerHTML = '';

        const download = DEMO_DOWNLOAD_MB[demoType] ?? 0;
        if (download >= CONFIRM_ABOVE_MB && isMemoryConstrained() && !this.confirmed.has(demoType)) {
            this.renderDownloadGate(demoType, download, container);
            return;
        }

        await this.instantiate(demoType, container);
    }

    /**
     * Ask before pulling a large model onto a phone.
     *
     * Loading it unasked is how the tab dies: the reader taps a tab expecting a
     * demo and gets a reload instead, with nothing to explain it. Given the
     * choice they can still go ahead — it is their data and their battery.
     */
    private renderDownloadGate(demoType: DemoType, megabytes: number, container: HTMLElement): void {
        container.innerHTML = `
            <div class="h-full flex items-center justify-center p-6 text-center">
                <div class="max-w-xs">
                    <div class="text-sm text-gray-300 mb-1">This demo downloads ${megabytes} MB</div>
                    <p class="text-xs text-gray-500 mb-4">
                        It runs a neural network in the page. On a phone that is enough memory
                        to reload the tab.
                    </p>
                    <button id="demo-load-anyway"
                            class="px-4 py-2 rounded-lg text-sm border border-accent-cyan/40 text-accent-cyan
                                   hover:bg-accent-cyan/10 transition-colors">
                        Load it anyway
                    </button>
                </div>
            </div>`;

        container.querySelector('#demo-load-anyway')?.addEventListener('click', () => {
            this.confirmed.add(demoType);
            container.innerHTML = '<div class="h-full flex items-center justify-center text-gray-500 text-sm"><div class="animate-pulse">Loading…</div></div>';
            this.instantiate(demoType, container).catch(error => {
                console.error(`Failed to load ${demoType} demo:`, error);
                container.innerHTML = '<div class="text-center text-red-400 p-8">Failed to load demo</div>';
            });
        });
    }

    private async instantiate(demoType: DemoType, container: HTMLElement): Promise<void> {
        container.innerHTML = '';

        switch (demoType) {
            case 'ibvs':
                this.currentInstance = new VisualServoDemo({
                    containerId: this.containerId,
                    backgroundColor: 0x0a0a0f
                });
                break;

            case 'ternary':
                // Lazy load TernaryLMDemo
                const { TernaryLMDemo } = await import('../components/ternary/TernaryLMDemo.js');
                this.currentInstance = new TernaryLMDemo({
                    containerId: this.containerId,
                    modelPath: '/assets/models/transformer_new',
                    defaultPrompt: 'The theory of relativity is ',
                    maxTokens: 100
                });
                break;

            case 'drone-racing':
                // Lazy load DroneRacingDemo
                const { DroneRacingDemo } = await import('../components/drone-racing/DroneRacingDemo.js');
                this.currentInstance = new DroneRacingDemo(this.containerId);
                break;

            case 'depth':
                // Lazy load DepthDemo
                const { DepthDemo } = await import('../components/depth/DepthDemo.js');
                this.currentInstance = new DepthDemo({
                    containerId: this.containerId,
                    modelPath: '/assets/models/depth'
                });
                break;

            case 'mapf':
                // Lazy load MapfDemo
                const { MapfDemo } = await import('../components/mapf/index.js');
                this.currentInstance = new MapfDemo({
                    containerId: this.containerId,
                    modelPath: '/assets/models/mapf',
                    compact: true
                });
                break;
        }
    }

    /**
     * Update tab styling for active/inactive states
     */
    private updateTabStyles(activeType: DemoType): void {
        const tabs = document.querySelectorAll('#demo-tabs .demo-tab');
        tabs.forEach(tab => {
            const tabDemo = tab.getAttribute('data-demo');
            if (tabDemo === activeType) {
                tab.classList.remove('text-gray-400', 'border-transparent', 'hover:text-gray-300', 'hover:border-gray-600');
                tab.classList.add('text-accent-cyan', 'border-accent-cyan', 'bg-dark-surface/50');
            } else {
                tab.classList.remove('text-accent-cyan', 'border-accent-cyan', 'bg-dark-surface/50');
                tab.classList.add('text-gray-400', 'border-transparent', 'hover:text-gray-300', 'hover:border-gray-600');
            }
        });
    }

    /**
     * Update hint text for current demo
     */
    private updateHintText(demoType: DemoType): void {
        const hint = document.getElementById('demo-hint');
        if (hint) {
            hint.textContent = DEMO_HINTS[demoType];
            // Demos that ship their own labelled controls do not need the
            // floating hint, and it would sit on top of their header.
            const selfExplaining = demoType === 'ternary' || demoType === 'mapf';
            hint.style.display = selfExplaining ? 'none' : 'block';
        }
    }

    /**
     * Reset the current demo to initial state
     */
    public reset(): void {
        if (this.currentInstance) {
            if (typeof this.currentInstance.reset === 'function') {
                this.currentInstance.reset();
            } else if (typeof this.currentInstance.resetSimulation === 'function') {
                this.currentInstance.resetSimulation();
            }
        }
    }

    /**
     * Get the currently active demo type
     */
    public getActiveDemo(): DemoType {
        return this.activeDemo;
    }

    /**
     * Get the current demo instance
     */
    public getCurrentInstance(): DemoInstance | null {
        return this.currentInstance;
    }

    /**
     * Destroy the current demo instance
     */
    private destroyCurrentInstance(): void {
        if (this.currentInstance) {
            if (typeof this.currentInstance.destroy === 'function') {
                this.currentInstance.destroy();
            }
            this.currentInstance = null;
        }
    }

    /**
     * Destroy the manager and cleanup
     */
    public destroy(): void {
        this.destroyCurrentInstance();
        this.mountedOn = null;
    }
}
