import { Navigation } from '../utils/navigation.js';
import { DroneRacingDemo } from '../components/drone-racing/index.js';

/**
 * Demo configuration
 */
interface DemoConfig {
    id: string;
    title: string;
    description: string;
    component: string;
}

const DEMO_CONFIGS: Record<string, DemoConfig> = {
    'drone-racing': {
        id: 'drone-racing',
        title: 'Autonomous Drone Racing Pipeline',
        description: 'Interactive demonstration of a complete autonomous drone racing system. The demo shows gate detection, trajectory generation through racing gates, and MPC-based trajectory tracking - all running in real-time in your browser.',
        component: 'DroneRacingDemo',
    },
};

export class DemoPage {
    private container: HTMLElement;
    private currentDemo: DroneRacingDemo | null = null;

    constructor(container: HTMLElement) {
        this.container = container;
    }

    public async render(demoId: string): Promise<void> {
        const config = DEMO_CONFIGS[demoId];

        if (!config) {
            this.renderNotFound(demoId);
            return;
        }

        document.title = `${config.title} - Victor Retamal`;

        this.container.innerHTML = `
            <div class="min-h-screen bg-dark-bg flex flex-col">
                <!-- Header -->
                <header class="bg-dark-surface border-b border-dark-border flex-shrink-0">
                    <div class="max-w-7xl mx-auto px-6 py-4">
                        <div class="flex justify-between items-center">
                            <div>
                                <h1 class="text-xl font-bold text-white">${config.title}</h1>
                                <p class="text-gray-400 text-sm">${config.description}</p>
                            </div>
                            <div class="flex items-center gap-4">
                                <button id="back-btn" class="text-gray-400 hover:text-accent-cyan transition-colors flex items-center">
                                    <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
                                    </svg>
                                    Back
                                </button>
                            </div>
                        </div>
                    </div>
                </header>

                <!-- Demo Container -->
                <main class="flex-1 relative">
                    <div id="demo-container" class="absolute inset-0">
                        <!-- Demo will be loaded here -->
                    </div>
                </main>
            </div>
        `;

        this.setupEventListeners();
        await this.loadDemo(config);
    }

    private async loadDemo(config: DemoConfig): Promise<void> {
        const container = document.getElementById('demo-container');
        if (!container) return;

        try {
            switch (config.component) {
                case 'DroneRacingDemo':
                    this.currentDemo = new DroneRacingDemo('demo-container');
                    break;
                default:
                    container.innerHTML = `
                        <div class="flex items-center justify-center h-full text-gray-400">
                            <p>Demo component "${config.component}" not found.</p>
                        </div>
                    `;
            }
        } catch (error) {
            console.error('Failed to load demo:', error);
            container.innerHTML = `
                <div class="flex items-center justify-center h-full text-gray-400">
                    <div class="text-center">
                        <p class="text-xl mb-2">Failed to load demo</p>
                        <p class="text-sm">${error instanceof Error ? error.message : 'Unknown error'}</p>
                    </div>
                </div>
            `;
        }
    }

    private renderNotFound(demoId: string): void {
        this.container.innerHTML = `
            <div class="min-h-screen bg-dark-bg flex items-center justify-center">
                <div class="text-center">
                    <h1 class="text-4xl font-bold text-white mb-4">Demo Not Found</h1>
                    <p class="text-gray-400 mb-6">The demo "${demoId}" doesn't exist.</p>
                    <button id="back-btn" class="btn-primary">
                        Back to Demos
                    </button>
                </div>
            </div>
        `;
        this.setupEventListeners();
    }

    private setupEventListeners(): void {
        const backBtn = document.getElementById('back-btn');
        backBtn?.addEventListener('click', () => {
            Navigation.toDemos();
        });
    }

    public destroy(): void {
        if (this.currentDemo) {
            this.currentDemo.destroy();
            this.currentDemo = null;
        }
    }
}
