import { createElement } from '../utils/dom.js';
import { Navigation } from '../utils/navigation.js';
import { parseFrontmatter, extractSlugFromPath, type BlogPostMeta } from '../utils/frontmatter.js';
import { seo } from '../utils/seo.js';
import { refreshThemeToggles, themeToggleMarkup } from '../utils/theme.js';

// Auto-import all markdown files from content/markdown/
const markdownModules = import.meta.glob('../content/markdown/*.md', {
    query: '?raw',
    import: 'default',
    eager: true,
}) as Record<string, string>;

type BlogPost = BlogPostMeta;

export class BlogListPage {
    private container: HTMLElement;
    private posts: BlogPost[] = [];

    constructor(container: HTMLElement) {
        this.container = container;
    }

    public async render(): Promise<void> {
        try {
            await this.loadBlogData();
            await this.renderBlogList();
            this.setupEventListeners();
        } catch (error) {
            console.error('Blog List Page: Error during render:', error);
            throw error; // Re-throw to let router handle it
        }
    }

    private async loadBlogData(): Promise<void> {
        try {
            // Parse all markdown files and extract metadata from frontmatter
            this.posts = [];

            for (const [path, rawContent] of Object.entries(markdownModules)) {
                const slug = extractSlugFromPath(path);
                const { meta } = parseFrontmatter(rawContent, slug);
                this.posts.push(meta);
            }

            // Sort by date (newest first)
            this.posts.sort((a, b) => {
                return new Date(b.date).getTime() - new Date(a.date).getTime();
            });
        } catch (error) {
            console.error('Failed to load blog data:', error);
            this.posts = [];
        }
    }

    private async renderBlogList(): Promise<void> {
        seo.blogList();

        if (this.posts.length === 0) {
            this.renderError();
            return;
        }

        this.container.innerHTML = `
            <div class="min-h-screen">
                <!-- Header -->
                <header class="sticky top-0 z-40">
                    <div class="win-bar">
                        <span class="win-title">/blog/index.html</span>
                        <span class="win-controls" aria-hidden="true"><i></i><i></i><i></i></span>
                    </div>
                    <div class="bg-dark-surface border-b border-dark-border">
                        <div class="max-w-4xl mx-auto px-4 sm:px-6 py-4">
                            <div class="flex flex-wrap justify-between items-center gap-3">
                                <div>
                                    <h1 class="heading-retro text-2xl">Blog</h1>
                                    <p class="text-sm text-gray-500 mt-1 pl-[1.3rem]">Thoughts on ML, robotics, and engineering</p>
                                </div>
                                <div class="flex items-center gap-2">
                                    <button id="back-btn" class="btn-mini">&#8592; Home</button>
                                    ${themeToggleMarkup()}
                                </div>
                            </div>
                        </div>
                    </div>
                </header>

                <!-- Blog Posts -->
                <main class="max-w-4xl mx-auto px-4 sm:px-6 py-10">
                    <p class="text-xs font-mono text-gray-500 mb-4">${this.posts.length} ${this.posts.length === 1 ? 'entry' : 'entries'} &middot; newest first</p>
                    <div id="blog-posts-container" class="space-y-5">
                        <!-- Blog posts will be loaded here -->
                    </div>
                </main>
            </div>
        `;

        this.populateBlogPosts();
        refreshThemeToggles();
    }

    private populateBlogPosts(): void {
        const container = document.getElementById('blog-posts-container');
        if (!container) return;

        container.innerHTML = '';

        // Posts are already sorted by date in loadBlogData
        this.posts.forEach(post => {
            const postElement = this.createBlogPostCard(post);
            container.appendChild(postElement);
        });
    }

    private createBlogPostCard(post: BlogPost): HTMLElement {
        const card = createElement('article', 'card group cursor-pointer');

        card.innerHTML = `
            <div class="win-bar">
                <span class="win-title">${post.slug}.md</span>
                <span class="ml-auto font-normal tracking-normal normal-case opacity-80">${post.readTime}</span>
            </div>

            <div class="p-5 md:p-6">
                <p class="text-xs font-mono text-gray-500 mb-2">${this.formatDate(post.date)}</p>

                <h2 class="text-xl font-bold text-white mb-3 group-hover:text-accent-cyan transition-colors">${post.title}</h2>

                <p class="text-gray-400 mb-4 leading-relaxed">${post.summary}</p>

                <div class="flex flex-wrap gap-1.5 mb-5">
                    ${post.tags.map(tag => `<span class="pill">${tag}</span>`).join('')}
                </div>

                <button class="read-more-btn btn-mini" data-slug="${post.slug}">
                    Read post
                    <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="3" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7"></path>
                    </svg>
                </button>
            </div>
        `;

        // Add click handler for the entire card
        card.addEventListener('click', () => {
            Navigation.toBlogPost(post.slug);
        });

        return card;
    }

    private formatDate(date: string): string {
        const options: Intl.DateTimeFormatOptions = {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        };
        return new Date(date).toLocaleDateString('en-US', options);
    }

    private renderError(): void {
        this.container.innerHTML = `
            <div class="min-h-screen flex items-center justify-center p-6">
                <div class="win max-w-md w-full">
                    <div class="win-bar">
                        <span class="win-title">error.html</span>
                        <span class="win-controls" aria-hidden="true"><i></i><i></i><i></i></span>
                    </div>
                    <div class="win-body text-center">
                        <h1 class="text-2xl font-bold text-white mb-3">Blog unavailable</h1>
                        <p class="text-gray-400 mb-6">Failed to load blog posts. Please try again later.</p>
                        <button id="back-btn" class="btn-primary">
                            Back to home
                        </button>
                    </div>
                </div>
            </div>
        `;
        this.setupEventListeners();
    }

    private setupEventListeners(): void {
        const backBtn = document.getElementById('back-btn');
        backBtn?.addEventListener('click', () => {
            Navigation.toHome();
        });

        // Handle read more buttons
        const readMoreBtns = document.querySelectorAll('.read-more-btn');
        readMoreBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent card click
                const slug = (btn as HTMLElement).dataset.slug;
                if (slug) {
                    Navigation.toBlogPost(slug);
                }
            });
        });
    }

    public destroy(): void {
        // Cleanup if needed
    }
}