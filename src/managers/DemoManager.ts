import { VisualServoDemo } from '../components/VisualServoDemo.js';

/**
 * Types for the home page demo tabs
 */
export type DemoType = 'ibvs' | 'ternary' | 'drone-racing';

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
};

/**
 * Callback type for when demo changes
 */
export type DemoChangeCallback = (demoType: DemoType) => void;

/**
 * Manages demo lifecycle: switching, reset, destroy
 */
export class DemoManager {
    private containerId: string;
    private activeDemo: DemoType = 'ibvs';
    private currentInstance: DemoInstance | null = null;
    private onDemoChange: DemoChangeCallback | null = null;

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

        // Initialize default demo (IBVS)
        this.currentInstance = new VisualServoDemo({
            containerId: this.containerId,
            backgroundColor: 0x0a0a0f
        });

        this.setupTabs();
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
                    modelPath: '/assets/models/transformer',
                    defaultPrompt: 'ROMEO: ',
                    maxTokens: 100
                });
                break;

            case 'drone-racing':
                // Lazy load DroneRacingDemo
                const { DroneRacingDemo } = await import('../components/drone-racing/DroneRacingDemo.js');
                this.currentInstance = new DroneRacingDemo(this.containerId);
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
            // Hide hint for ternary (it has its own UI)
            hint.style.display = demoType === 'ternary' ? 'none' : 'block';
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
    }
}
