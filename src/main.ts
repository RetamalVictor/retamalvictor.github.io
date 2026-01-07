import './styles/main.css';
import { Header } from './components/Header.js';
import { ProjectCard } from './components/ProjectCard.js';
import { WebGPUParticles } from './components/WebGPUParticles.js';
import { Router } from './utils/router.js';
import { Navigation, initializeNavigation } from './utils/navigation.js';
import { ProjectDetailPage } from './pages/ProjectDetail.js';
import { CVPage } from './pages/CV.js';
import { BlogListPage } from './pages/BlogList.js';
import { BlogPostPage } from './pages/BlogPost.js';
import { addIntersectionObserver } from './utils/dom.js';
import { config } from './utils/config.js';
import { templateManager } from './utils/template.js';
import type { Project } from './types/index.js';

class Portfolio {
    private header!: Header;
    private projects: Project[] = [];
    private observer: IntersectionObserver | null = null;
    private heroParticles: WebGPUParticles | null = null;
    private router!: Router;
    private projectDetailPage: ProjectDetailPage | null = null;
    private cvPage: CVPage | null = null;
    private blogListPage: BlogListPage | null = null;
    private blogPostPage: BlogPostPage | null = null;

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

        // Set base path for production
        const environment = process.env.NODE_ENV === 'production' ? 'production' : 'development';
        const basePath = environment === 'production' ? '/blog' : '';

        this.router = new Router(basePath);
        this.router.addRoute('/', this.renderHomePage.bind(this), pages.home);
        this.router.addRoute('/project/:id', this.renderProjectPage.bind(this), pages.projects);
        this.router.addRoute('/cv', this.renderCVPage.bind(this), pages.cv);
        this.router.addRoute('/blog', this.renderBlogPage.bind(this), pages.blog);
        this.router.addRoute('/blog/:slug', this.renderBlogPostPage.bind(this), pages.blog_post);

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

    private populateResearchAreas(): void {
        const container = document.getElementById('research-areas-container');
        if (container) {
            container.innerHTML = templateManager.createResearchAreasHTML();
            // Setup card expansion interactivity
            this.setupResearchAreaInteractivity();
        }
    }

    private setupResearchAreaInteractivity(): void {
        const cards = document.querySelectorAll('.research-area-card[role="button"]');

        cards.forEach(card => {
            const toggleCard = () => {
                const relatedItems = card.querySelector('.related-items');
                const chevron = card.querySelector('.chevron-icon');
                const isExpanded = card.getAttribute('aria-expanded') === 'true';

                if (relatedItems && chevron) {
                    if (isExpanded) {
                        // Collapse
                        relatedItems.classList.add('hidden');
                        chevron.classList.remove('rotate-180');
                        card.setAttribute('aria-expanded', 'false');
                    } else {
                        // Expand
                        relatedItems.classList.remove('hidden');
                        chevron.classList.add('rotate-180');
                        card.setAttribute('aria-expanded', 'true');
                    }
                }
            };

            // Click handler
            card.addEventListener('click', (e: Event) => {
                // Prevent expansion if clicking on a link
                if ((e.target as HTMLElement).tagName === 'A' ||
                    (e.target as HTMLElement).closest('a')) {
                    return;
                }
                toggleCard();
            });

            // Keyboard accessibility
            card.addEventListener('keydown', (e: Event) => {
                const keyEvent = e as KeyboardEvent;
                if (keyEvent.key === 'Enter' || keyEvent.key === ' ') {
                    keyEvent.preventDefault();
                    toggleCard();
                }
            });
        });
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
            // Initialize WebGPU particle system
            const heroContainer = document.getElementById('hero-three-scene');
            if (heroContainer) {
                this.heroParticles = new WebGPUParticles({
                    containerId: 'hero-three-scene',
                    particleCount: 30000,
                    backgroundColor: 0x0a0a0f
                });

                // Setup reset button
                const resetBtn = document.getElementById('reset-particles-btn');
                if (resetBtn && this.heroParticles) {
                    resetBtn.addEventListener('click', () => {
                        this.heroParticles?.reset();
                    });
                }
            }
        } catch (error) {
            console.error('Failed to initialize WebGPU particles:', error);
        }
    }

    private renderProjects(): void {
        const projectsGrid = document.getElementById('projects-grid')!;
        projectsGrid.innerHTML = '';

        const featuredProjects = this.projects.filter(p => p.featured);

        featuredProjects.forEach(project => {
            new ProjectCard(project, projectsGrid);
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
        this.heroParticles?.destroy();
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