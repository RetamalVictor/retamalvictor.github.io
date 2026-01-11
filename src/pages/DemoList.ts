import { createElement } from '../utils/dom.js';
import { Navigation } from '../utils/navigation.js';

/**
 * Demo metadata
 */
interface Demo {
    id: string;
    title: string;
    summary: string;
    tags: string[];
    component: string;  // Component name to load
    thumbnail?: string;
}

/**
 * Available demos
 */
const DEMOS: Demo[] = [
    {
        id: 'drone-racing',
        title: 'Autonomous Drone Racing',
        summary: 'Interactive visualization of a full autonomous drone racing pipeline: gate detection, trajectory generation, and MPC control.',
        tags: ['Robotics', 'Control', 'Computer Vision', 'Three.js'],
        component: 'DroneRacingDemo',
    },
    // Future demos can be added here
];

export class DemoListPage {
    private container: HTMLElement;
    private demos: Demo[] = DEMOS;

    constructor(container: HTMLElement) {
        this.container = container;
    }

    public async render(): Promise<void> {
        await this.renderDemoList();
        this.setupEventListeners();
    }

    private async renderDemoList(): Promise<void> {
        document.title = 'Demos - Victor Retamal';

        this.container.innerHTML = `
            <div class="min-h-screen bg-dark-bg">
                <!-- Header -->
                <header class="bg-dark-surface border-b border-dark-border">
                    <div class="max-w-4xl mx-auto px-6 py-8">
                        <div class="flex justify-between items-center">
                            <div>
                                <h1 class="text-3xl font-bold text-white mb-2">Interactive Demos</h1>
                                <p class="text-gray-400">Explore interactive visualizations of robotics and ML concepts</p>
                            </div>
                            <div>
                                <button id="back-btn" class="text-gray-400 hover:text-accent-cyan transition-colors flex items-center">
                                    <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
                                    </svg>
                                    Home
                                </button>
                            </div>
                        </div>
                    </div>
                </header>

                <!-- Demos Grid -->
                <main class="max-w-4xl mx-auto px-6 py-12">
                    <div id="demos-container" class="grid gap-6 md:grid-cols-2">
                        <!-- Demos will be loaded here -->
                    </div>
                </main>
            </div>
        `;

        this.populateDemos();
    }

    private populateDemos(): void {
        const container = document.getElementById('demos-container');
        if (!container) return;

        container.innerHTML = '';

        this.demos.forEach(demo => {
            const demoElement = this.createDemoCard(demo);
            container.appendChild(demoElement);
        });
    }

    private createDemoCard(demo: Demo): HTMLElement {
        const card = createElement('article', 'bg-dark-surface border border-dark-border rounded-lg overflow-hidden hover:border-accent-purple transition-all cursor-pointer group');

        card.innerHTML = `
            <div class="aspect-video bg-gradient-to-br from-accent-purple/20 to-accent-cyan/20 flex items-center justify-center">
                <div class="text-6xl opacity-50 group-hover:opacity-80 transition-opacity">
                    ${this.getDemoIcon(demo.id)}
                </div>
            </div>
            <div class="p-6">
                <h2 class="text-xl font-bold text-white mb-3 group-hover:text-accent-cyan transition-colors">
                    ${demo.title}
                </h2>
                <p class="text-gray-400 mb-4 leading-relaxed text-sm">
                    ${demo.summary}
                </p>
                <div class="flex flex-wrap gap-2 mb-4">
                    ${demo.tags.map(tag => `
                        <span class="px-2 py-1 text-xs rounded bg-dark-border text-gray-400">
                            ${tag}
                        </span>
                    `).join('')}
                </div>
                <button class="launch-demo-btn text-accent-purple font-medium hover:text-white transition-colors flex items-center" data-id="${demo.id}">
                    Launch Demo
                    <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path>
                    </svg>
                </button>
            </div>
        `;

        card.addEventListener('click', () => {
            Navigation.toDemo(demo.id);
        });

        return card;
    }

    private getDemoIcon(demoId: string): string {
        const icons: Record<string, string> = {
            'drone-racing': '🚁',
            'default': '🎮',
        };
        return icons[demoId] || icons['default'];
    }

    private setupEventListeners(): void {
        const backBtn = document.getElementById('back-btn');
        backBtn?.addEventListener('click', () => {
            Navigation.toHome();
        });

        const launchBtns = document.querySelectorAll('.launch-demo-btn');
        launchBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const id = (btn as HTMLElement).dataset.id;
                if (id) {
                    Navigation.toDemo(id);
                }
            });
        });
    }

    public destroy(): void {
        // Cleanup if needed
    }
}
