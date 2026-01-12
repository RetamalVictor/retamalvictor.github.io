import './styles/main.css';
import { Header } from './components/Header.js';
import { VisualServoDemo } from './components/VisualServoDemo.js';
import { Router } from './utils/router.js';
import { Navigation, initializeNavigation } from './utils/navigation.js';
import { ProjectDetailPage } from './pages/ProjectDetail.js';
import { CVPage } from './pages/CV.js';
import { BlogListPage } from './pages/BlogList.js';
import { BlogPostPage } from './pages/BlogPost.js';
import { DemoListPage } from './pages/DemoList.js';
import { DemoPage } from './pages/DemoPage.js';
import { addIntersectionObserver } from './utils/dom.js';
import { config } from './utils/config.js';
import { templateManager } from './utils/template.js';
import type { Project } from './types/index.js';

// Demo types for home page tabs
type HeroDemoType = 'ibvs' | 'ternary' | 'drone-racing';

// Demo hints for each type
const DEMO_HINTS: Record<HeroDemoType, string> = {
    'ibvs': 'Drag the target',
    'ternary': 'Enter a prompt',
    'drone-racing': 'Use mouse to orbit',
};

// Info panel content for each demo type
const INFO_PANEL_CONTENT: Record<HeroDemoType, { title: string; content: string }> = {
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

class Portfolio {
    private header!: Header;
    private projects: Project[] = [];
    private observer: IntersectionObserver | null = null;
    private heroDemo: VisualServoDemo | null = null;
    private router!: Router;
    private projectDetailPage: ProjectDetailPage | null = null;
    private cvPage: CVPage | null = null;
    private blogListPage: BlogListPage | null = null;
    private blogPostPage: BlogPostPage | null = null;
    private demoListPage: DemoListPage | null = null;
    private demoPage: DemoPage | null = null;

    // Demo tab state
    private activeDemo: HeroDemoType = 'ibvs';
    private currentDemoInstance: any = null;
    private isExpanded: boolean = false;

    constructor() {
        this.init();
    }

    private async init(): Promise<void> {
        try {
            // Initialize configuration system first
            await config.initialize();

            // Load data from configuration
            this.loadProjectsData();

            // Setup router with configuration-driven titles
            // But DON'T initialize it yet - we'll do that after checking the current route
            this.setupRouter();
        } catch (error) {
            console.error('Failed to initialize portfolio:', error);
        }
    }

    private setupRouter(): void {
        const siteConfig = config.getSiteConfig();
        const pages = siteConfig.pages;

        // Set base path - empty for custom domain (victor-retamal.com)
        const environment = process.env.NODE_ENV === 'production' ? 'production' : 'development';
        const basePath = '';

        this.router = new Router(basePath);
        this.router.addRoute('/', this.renderHomePage.bind(this), pages.home);
        this.router.addRoute('/project/:id', this.renderProjectPage.bind(this), pages.projects);
        this.router.addRoute('/cv', this.renderCVPage.bind(this), pages.cv);
        this.router.addRoute('/blog', this.renderBlogPage.bind(this), pages.blog);
        this.router.addRoute('/blog/:slug', this.renderBlogPostPage.bind(this), pages.blog_post);
        this.router.addRoute('/demos', this.renderDemosPage.bind(this), 'Demos');
        this.router.addRoute('/demos/:id', this.renderDemoPage.bind(this), 'Demo');

        // Initialize navigation module
        initializeNavigation(this.router, environment as 'development' | 'production');

        // Now that all routes are added, initialize the router to handle the current URL
        this.router.initialize();
    }

    private async renderHomePage(): Promise<void> {
        // Check if we need to restore the layout
        if (!document.getElementById('main-content')) {
            await this.initializeLayout();
            this.setupThreeViewers();
            this.setupIntersectionObserver();
            this.setupScrollAnimations();
        } else {
            // Show all main elements
            const mainContent = document.getElementById('main-content');
            const headerContainer = document.getElementById('header-container');
            const heroContainer = document.getElementById('hero-container');

            if (mainContent) mainContent.style.display = 'block';
            if (headerContainer) headerContainer.style.display = 'block';
            if (heroContainer) heroContainer.style.display = 'block';
        }
    }

    /**
     * Clear main layout for full-page content.
     * Hides the main sections and clears the app container.
     */
    private clearMainLayout(): void {
        const mainContent = document.getElementById('main-content');
        const headerContainer = document.getElementById('header-container');
        const heroContainer = document.getElementById('hero-container');
        const app = document.getElementById('app');

        if (mainContent) mainContent.style.display = 'none';
        if (headerContainer) headerContainer.style.display = 'none';
        if (heroContainer) heroContainer.style.display = 'none';
        if (app) app.innerHTML = '';
    }

    private async renderProjectPage(): Promise<void> {
        const path = window.location.pathname;
        const projectId = this.router.getRouteParams('/project/:id', path).id;

        const app = document.getElementById('app')!;

        // Hide main content
        const mainContent = document.getElementById('main-content');
        if (mainContent) {
            mainContent.style.display = 'none';
        }

        // Create or update project detail page
        if (!this.projectDetailPage) {
            this.projectDetailPage = new ProjectDetailPage(app);
        }

        await this.projectDetailPage.render(projectId, this.projects);
    }

    private async renderCVPage(): Promise<void> {
        const app = document.getElementById('app')!;
        this.clearMainLayout();

        // Create CV page
        this.cvPage = new CVPage(app);
        await this.cvPage.render();
    }

    private async renderBlogPage(): Promise<void> {
        const app = document.getElementById('app')!;
        this.clearMainLayout();

        // Create blog list page
        this.blogListPage = new BlogListPage(app);
        await this.blogListPage.render();
    }

    private async renderBlogPostPage(): Promise<void> {
        const path = window.location.pathname;
        const postSlug = this.router.getRouteParams('/blog/:slug', path).slug;

        const app = document.getElementById('app')!;
        this.clearMainLayout();

        // Create blog post page
        this.blogPostPage = new BlogPostPage(app);
        await this.blogPostPage.render(postSlug);
    }

    private async renderDemosPage(): Promise<void> {
        const app = document.getElementById('app')!;
        this.clearMainLayout();

        // Create demos list page
        this.demoListPage = new DemoListPage(app);
        await this.demoListPage.render();
    }

    private async renderDemoPage(): Promise<void> {
        const path = window.location.pathname;
        const demoId = this.router.getRouteParams('/demos/:id', path).id;

        const app = document.getElementById('app')!;
        this.clearMainLayout();

        // Create demo page
        if (this.demoPage) {
            this.demoPage.destroy();
        }
        this.demoPage = new DemoPage(app);
        await this.demoPage.render(demoId);
    }

    private async initializeLayout(): Promise<void> {
        const app = document.getElementById('app')!;

        // Load the main layout template
        const mainLayoutTemplate = await templateManager.loadConfigurableTemplate('/src/templates/main-layout.html');
        app.innerHTML = mainLayoutTemplate;

        // Load each section template and populate containers
        await this.loadSectionTemplates();

        const headerContainer = document.getElementById('header-container')!;
        this.header = new Header(headerContainer, this.router);

        await this.loadHeroSection();
    }

    private async loadSectionTemplates(): Promise<void> {
        try {
            // Load section templates
            const recentPostsTemplate = await templateManager.loadConfigurableTemplate('/src/templates/recent-posts-section.html');
            const footerTemplate = await templateManager.loadConfigurableTemplate('/src/templates/footer.html');

            // Populate containers
            document.getElementById('recent-posts-container')!.innerHTML = recentPostsTemplate;
            document.getElementById('footer-container')!.innerHTML = footerTemplate;

            // Populate dynamic content
            this.populateRecentPosts();
            this.populateFooterContent();

        } catch (error) {
            console.error('Failed to load section templates:', error);
        }
    }

    private populateRecentPosts(): void {
        const container = document.getElementById('recent-posts-grid');
        if (!container) return;

        try {
            const blogData = config.getBlogPostsData();
            const posts = blogData.posts || [];

            // Get the 3 most recent posts (already sorted by date in YAML)
            const recentPosts = posts.slice(0, 3);

            container.innerHTML = recentPosts.map((post: any) => `
                <a href="/blog/${post.slug}" class="card p-6 group cursor-pointer block">
                    <div class="flex items-center gap-2 text-sm text-gray-500 mb-3">
                        <span>${post.date}</span>
                        <span>·</span>
                        <span>${post.readTime}</span>
                    </div>
                    <h3 class="text-lg font-semibold text-white mb-2 group-hover:text-accent-cyan transition-colors">
                        ${post.title}
                    </h3>
                    <p class="text-gray-400 text-sm line-clamp-2">
                        ${post.summary}
                    </p>
                    <div class="flex flex-wrap gap-2 mt-4">
                        ${post.tags.slice(0, 3).map((tag: string) => `
                            <span class="text-xs px-2 py-1 rounded bg-dark-border text-gray-400">
                                ${tag}
                            </span>
                        `).join('')}
                    </div>
                </a>
            `).join('');

            // Add click handlers for navigation
            container.querySelectorAll('a').forEach(link => {
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    const href = link.getAttribute('href');
                    if (href) {
                        Navigation.to(href);
                    }
                });
            });
        } catch (error) {
            console.error('Failed to load recent posts:', error);
            container.innerHTML = '<p class="text-gray-500">Unable to load recent posts.</p>';
        }
    }

    private populateFooterContent(): void {
        // Footer content is now static in the template
        // Social links are hardcoded with template variables
    }

    private async loadHeroSection(): Promise<void> {
        try {
            const heroContainer = document.getElementById('hero-container')!;
            const heroTemplate = await templateManager.loadConfigurableTemplate('/src/templates/hero.html');
            heroContainer.innerHTML = heroTemplate;
        } catch (error) {
            console.error('Failed to load hero section:', error);
        }
    }

    private loadProjectsData(): void {
        try {
            const projectsData = config.getProjectsData();
            // Type assertion since we know the YAML data conforms to Project interface
            this.projects = projectsData.projects as Project[];
            console.log(`Loaded ${this.projects.length} projects from configuration`);
        } catch (error) {
            console.error('Failed to load projects data:', error);
            // Fallback to empty array
            this.projects = [];
        }
    }

    private setupThreeViewers(): void {
        try {
            // Initialize Visual Servoing demo (default)
            const heroContainer = document.getElementById('hero-three-scene');
            if (heroContainer) {
                this.heroDemo = new VisualServoDemo({
                    containerId: 'hero-three-scene',
                    backgroundColor: 0x0a0a0f
                });
                this.currentDemoInstance = this.heroDemo;

                // Setup reset button
                const resetBtn = document.getElementById('reset-demo-btn');
                if (resetBtn && this.heroDemo) {
                    resetBtn.addEventListener('click', () => {
                        this.resetCurrentDemo();
                    });
                }

                // Setup info panel toggle
                this.setupInfoPanel();

                // Setup demo tabs
                this.setupDemoTabs();

                // Setup expand button
                this.setupExpandButton();
            }
        } catch (error) {
            console.error('Failed to initialize Visual Servo demo:', error);
        }
    }

    private setupDemoTabs(): void {
        const tabs = document.querySelectorAll('#demo-tabs .demo-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const demoType = tab.getAttribute('data-demo') as HeroDemoType;
                if (demoType && demoType !== this.activeDemo) {
                    this.switchDemo(demoType);
                }
            });
        });
    }

    private async switchDemo(demoType: HeroDemoType): Promise<void> {
        // Destroy current demo
        if (this.currentDemoInstance) {
            if (typeof this.currentDemoInstance.destroy === 'function') {
                this.currentDemoInstance.destroy();
            }
            this.currentDemoInstance = null;
            this.heroDemo = null;
        }

        // Clear container
        const container = document.getElementById('hero-three-scene');
        if (!container) return;
        container.innerHTML = '<div class="text-center text-gray-500 p-8"><div class="animate-pulse">Loading demo...</div></div>';

        // Update active tab styling
        const tabs = document.querySelectorAll('#demo-tabs .demo-tab');
        tabs.forEach(tab => {
            const tabDemo = tab.getAttribute('data-demo');
            if (tabDemo === demoType) {
                tab.classList.remove('text-gray-400', 'border-transparent', 'hover:text-gray-300', 'hover:border-gray-600');
                tab.classList.add('text-accent-cyan', 'border-accent-cyan', 'bg-dark-surface/50');
            } else {
                tab.classList.remove('text-accent-cyan', 'border-accent-cyan', 'bg-dark-surface/50');
                tab.classList.add('text-gray-400', 'border-transparent', 'hover:text-gray-300', 'hover:border-gray-600');
            }
        });

        // Update hint text
        const hint = document.getElementById('demo-hint');
        if (hint) {
            hint.textContent = DEMO_HINTS[demoType];
            // Hide hint for ternary (it has its own UI)
            hint.style.display = demoType === 'ternary' ? 'none' : 'block';
        }

        // Show "How it works" button for all demos
        const infoToggle = document.getElementById('info-panel-toggle');
        if (infoToggle) {
            (infoToggle as HTMLElement).style.display = 'flex';
        }

        // Update info panel content for current demo
        this.updateInfoPanelContent(demoType);

        // Show reset button for all demos
        const resetBtn = document.getElementById('reset-demo-btn');
        if (resetBtn) {
            resetBtn.style.display = 'block';
        }

        this.activeDemo = demoType;

        // Create new demo
        try {
            switch (demoType) {
                case 'ibvs':
                    container.innerHTML = '';
                    this.heroDemo = new VisualServoDemo({
                        containerId: 'hero-three-scene',
                        backgroundColor: 0x0a0a0f
                    });
                    this.currentDemoInstance = this.heroDemo;
                    break;

                case 'ternary':
                    // Lazy load TernaryLMDemo
                    const { TernaryLMDemo } = await import('./components/ternary/TernaryLMDemo.js');
                    container.innerHTML = '';
                    this.currentDemoInstance = new TernaryLMDemo({
                        containerId: 'hero-three-scene',
                        modelPath: '/assets/models/transformer',
                        defaultPrompt: 'ROMEO: ',
                        maxTokens: 100
                    });
                    break;

                case 'drone-racing':
                    // Lazy load DroneRacingDemo
                    const { DroneRacingDemo } = await import('./components/drone-racing/DroneRacingDemo.js');
                    container.innerHTML = '';
                    this.currentDemoInstance = new DroneRacingDemo('hero-three-scene');
                    break;
            }
        } catch (error) {
            console.error(`Failed to load ${demoType} demo:`, error);
            container.innerHTML = `<div class="text-center text-red-400 p-8">Failed to load demo</div>`;
        }
    }

    private resetCurrentDemo(): void {
        if (this.currentDemoInstance && typeof this.currentDemoInstance.reset === 'function') {
            this.currentDemoInstance.reset();
        } else if (this.currentDemoInstance && typeof this.currentDemoInstance.resetSimulation === 'function') {
            this.currentDemoInstance.resetSimulation();
        }
    }

    private updateInfoPanelContent(demoType: HeroDemoType): void {
        const title = document.getElementById('info-panel-title');
        const content = document.getElementById('info-panel-content');
        if (!title || !content) return;

        const panelData = INFO_PANEL_CONTENT[demoType];
        title.textContent = panelData.title;
        content.innerHTML = panelData.content;
    }

    private setupExpandButton(): void {
        const expandBtn = document.getElementById('expand-demo-btn');
        if (!expandBtn) return;

        expandBtn.addEventListener('click', () => {
            this.toggleExpand();
        });

        // Close on escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isExpanded) {
                this.toggleExpand();
            }
        });
    }

    private expandOverlay: HTMLElement | null = null;
    private expandedWrapper: HTMLElement | null = null;

    private toggleExpand(): void {
        const demoContainer = document.getElementById('demo-container');
        const expandBtn = document.getElementById('expand-demo-btn');
        if (!demoContainer || !expandBtn) return;

        this.isExpanded = !this.isExpanded;

        if (this.isExpanded) {
            // Create dark overlay
            this.expandOverlay = document.createElement('div');
            this.expandOverlay.id = 'expand-overlay';
            this.expandOverlay.style.cssText = `
                position: fixed;
                inset: 0;
                background: rgba(0, 0, 0, 0.85);
                z-index: 40;
            `;
            this.expandOverlay.addEventListener('click', () => this.toggleExpand());
            document.body.appendChild(this.expandOverlay);

            // Create expanded wrapper
            this.expandedWrapper = document.createElement('div');
            this.expandedWrapper.id = 'expanded-demo-wrapper';
            this.expandedWrapper.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 90vw;
                height: 85vh;
                max-width: 1400px;
                background: #0a0a0f;
                border: 1px solid #1e1e2e;
                border-radius: 12px;
                z-index: 50;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            `;

            // Close button (top-right corner inside modal)
            const closeBtn = document.createElement('button');
            closeBtn.innerHTML = `
                <svg style="width: 24px; height: 24px;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                </svg>
            `;
            closeBtn.style.cssText = `
                position: absolute;
                top: 12px;
                right: 12px;
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 50%;
                color: #ffffff;
                cursor: pointer;
                padding: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
                z-index: 60;
            `;
            closeBtn.addEventListener('mouseenter', () => {
                closeBtn.style.background = 'rgba(255, 0, 0, 0.3)';
                closeBtn.style.borderColor = 'rgba(255, 100, 100, 0.5)';
            });
            closeBtn.addEventListener('mouseleave', () => {
                closeBtn.style.background = 'rgba(255, 255, 255, 0.1)';
                closeBtn.style.borderColor = 'rgba(255, 255, 255, 0.2)';
            });
            closeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleExpand();
            });
            this.expandedWrapper.appendChild(closeBtn);

            // Move demo container into expanded wrapper
            const contentArea = document.createElement('div');
            contentArea.style.cssText = 'flex: 1; overflow: hidden;';

            // Store original parent reference for restoration
            demoContainer.dataset.originalParent = 'demo-section';

            // Move container
            contentArea.appendChild(demoContainer);
            this.expandedWrapper.appendChild(contentArea);
            document.body.appendChild(this.expandedWrapper);

            // Update container styles for expanded view
            demoContainer.classList.remove('relative');
            demoContainer.style.height = '100%';
            const threeContainer = demoContainer.querySelector('.three-container') as HTMLElement;
            if (threeContainer) {
                threeContainer.classList.remove('h-80', 'lg:h-96');
                threeContainer.style.height = '100%';
            }

            // Update button icon
            expandBtn.innerHTML = `
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                </svg>
            `;
            expandBtn.title = 'Close expanded view';

            // Trigger resize after a short delay for layout to settle
            setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
        } else {
            // Restore demo container to original position
            const demoSection = document.getElementById('demo-section');
            if (demoSection && demoContainer) {
                // Find the tab bar and insert after it
                const tabBar = demoSection.querySelector('.flex.items-center.justify-between');
                if (tabBar && tabBar.nextSibling) {
                    demoSection.insertBefore(demoContainer, tabBar.nextSibling);
                } else {
                    demoSection.appendChild(demoContainer);
                }

                // Restore container styles
                demoContainer.classList.add('relative');
                demoContainer.style.height = '';
                const threeContainer = demoContainer.querySelector('.three-container') as HTMLElement;
                if (threeContainer) {
                    threeContainer.classList.add('h-80', 'lg:h-96');
                    threeContainer.style.height = '';
                }
            }

            // Remove overlay and wrapper
            if (this.expandOverlay) {
                this.expandOverlay.remove();
                this.expandOverlay = null;
            }
            if (this.expandedWrapper) {
                this.expandedWrapper.remove();
                this.expandedWrapper = null;
            }

            // Update button icon
            expandBtn.innerHTML = `
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"></path>
                </svg>
            `;
            expandBtn.title = 'Expand demo';

            // Trigger resize
            setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
        }
    }

    private setupInfoPanel(): void {
        const panel = document.getElementById('info-panel');
        const overlay = document.getElementById('info-panel-overlay');
        const toggleBtn = document.getElementById('info-panel-toggle');
        const closeBtn = document.getElementById('info-panel-close');

        if (!panel || !overlay || !toggleBtn) return;

        const openPanel = () => {
            panel.classList.remove('translate-x-full');
            overlay.classList.remove('opacity-0', 'pointer-events-none');
            overlay.classList.add('opacity-100', 'pointer-events-auto');
        };

        const closePanel = () => {
            panel.classList.add('translate-x-full');
            overlay.classList.add('opacity-0', 'pointer-events-none');
            overlay.classList.remove('opacity-100', 'pointer-events-auto');
        };

        toggleBtn.addEventListener('click', openPanel);
        closeBtn?.addEventListener('click', closePanel);
        overlay.addEventListener('click', closePanel);

        // Close on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !panel.classList.contains('translate-x-full')) {
                closePanel();
            }
        });
    }

    private setupIntersectionObserver(): void {
        const sections = document.querySelectorAll('section[id]');

        this.observer = addIntersectionObserver(
            Array.from(sections),
            (entry) => {
                if (entry.isIntersecting) {
                    const sectionId = entry.target.id;
                    this.header.updateActiveLink(sectionId);

                    entry.target.classList.add('fade-in-up');
                }
            },
            { threshold: 0.3 }
        );
    }

    private setupScrollAnimations(): void {
        const animatedElements = document.querySelectorAll('.fade-in-up');

        const animationObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-fade-in-up');
                }
            });
        }, { threshold: 0.1 });

        animatedElements.forEach(el => animationObserver.observe(el));

        // Setup CV button click handler with a delay to ensure DOM is ready
        setTimeout(() => {
            const viewCvBtn = document.getElementById('view-cv-btn');
            if (viewCvBtn) {
                viewCvBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    Navigation.toCV();
                });
            }

        }, 100);
    }

    public destroy(): void {
        this.observer?.disconnect();
        this.heroDemo?.destroy();
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    try {
        new Portfolio();
    } catch (error) {
        console.error('Failed to initialize portfolio:', error);
        // Fallback: show basic content
        document.getElementById('app')!.innerHTML = `
            <div class="min-h-screen bg-dark-bg flex items-center justify-center">
                <div class="text-center">
                    <h1 class="text-2xl font-bold mb-4 text-white">Loading...</h1>
                    <p class="text-gray-500">Please wait</p>
                </div>
            </div>
        `;
    }
});