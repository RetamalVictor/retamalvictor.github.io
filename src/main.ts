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

        // Hide main content and header
        const mainContent = document.getElementById('main-content');
        const headerContainer = document.getElementById('header-container');
        const heroContainer = document.getElementById('hero-container');

        if (mainContent) mainContent.style.display = 'none';
        if (headerContainer) headerContainer.style.display = 'none';
        if (heroContainer) heroContainer.style.display = 'none';

        // Clear app container for CV page
        app.innerHTML = '';

        // Create CV page
        this.cvPage = new CVPage(app);
        await this.cvPage.render();
    }

    private async renderBlogPage(): Promise<void> {
        const app = document.getElementById('app')!;

        // Hide main content and header
        const mainContent = document.getElementById('main-content');
        const headerContainer = document.getElementById('header-container');
        const heroContainer = document.getElementById('hero-container');

        if (mainContent) mainContent.style.display = 'none';
        if (headerContainer) headerContainer.style.display = 'none';
        if (heroContainer) heroContainer.style.display = 'none';

        // Clear app container for blog page
        app.innerHTML = '';

        // Create blog list page
        this.blogListPage = new BlogListPage(app);
        await this.blogListPage.render();
    }

    private async renderBlogPostPage(): Promise<void> {
        const path = window.location.pathname;
        const postSlug = this.router.getRouteParams('/blog/:slug', path).slug;

        const app = document.getElementById('app')!;

        // Hide main content and header
        const mainContent = document.getElementById('main-content');
        const headerContainer = document.getElementById('header-container');
        const heroContainer = document.getElementById('hero-container');

        if (mainContent) mainContent.style.display = 'none';
        if (headerContainer) headerContainer.style.display = 'none';
        if (heroContainer) heroContainer.style.display = 'none';

        // Clear app container for blog post page
        app.innerHTML = '';

        // Create blog post page
        this.blogPostPage = new BlogPostPage(app);
        await this.blogPostPage.render(postSlug);
    }

    private async renderDemosPage(): Promise<void> {
        const app = document.getElementById('app')!;

        // Hide main content and header
        const mainContent = document.getElementById('main-content');
        const headerContainer = document.getElementById('header-container');
        const heroContainer = document.getElementById('hero-container');

        if (mainContent) mainContent.style.display = 'none';
        if (headerContainer) headerContainer.style.display = 'none';
        if (heroContainer) heroContainer.style.display = 'none';

        // Clear app container for demos page
        app.innerHTML = '';

        // Create demos list page
        this.demoListPage = new DemoListPage(app);
        await this.demoListPage.render();
    }

    private async renderDemoPage(): Promise<void> {
        const path = window.location.pathname;
        const demoId = this.router.getRouteParams('/demos/:id', path).id;

        const app = document.getElementById('app')!;

        // Hide main content and header
        const mainContent = document.getElementById('main-content');
        const headerContainer = document.getElementById('header-container');
        const heroContainer = document.getElementById('hero-container');

        if (mainContent) mainContent.style.display = 'none';
        if (headerContainer) headerContainer.style.display = 'none';
        if (heroContainer) heroContainer.style.display = 'none';

        // Clear app container for demo page
        app.innerHTML = '';

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

        // Show/hide external "How it works" button
        // TernaryLMDemo and DroneRacingDemo have their own built-in panels
        const infoToggle = document.getElementById('info-panel-toggle');
        if (infoToggle) {
            (infoToggle as HTMLElement).style.display = demoType === 'ibvs' ? 'flex' : 'none';
        }

        // Show/hide reset button (only IBVS uses external reset)
        const resetBtn = document.getElementById('reset-demo-btn');
        if (resetBtn) {
            resetBtn.style.display = demoType === 'ibvs' ? 'block' : 'none';
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
        if (this.activeDemo === 'ibvs' && this.heroDemo) {
            this.heroDemo.reset();
        }
        // Other demos don't have reset functionality
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