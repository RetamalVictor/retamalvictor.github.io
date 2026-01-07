import { createElement } from '../utils/dom.js';
import { Navigation } from '../utils/navigation.js';
import * as yaml from 'js-yaml';
import blogDataYaml from '../data/blog-posts.yaml?raw';

interface BlogPost {
    slug: string;
    title: string;
    date: string;
    tags: string[];
    summary: string;
    readTime: string;
    notebook: string;
    featured: boolean;
}

interface BlogData {
    posts: BlogPost[];
}

export class BlogListPage {
    private container: HTMLElement;
    private blogData: BlogData | null = null;

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
            // Load blog data from YAML file
            const parsedData = yaml.load(blogDataYaml) as any;
            this.blogData = {
                posts: parsedData.posts || []
            };
        } catch (error) {
            console.error('Failed to load blog data:', error);
            this.blogData = null;
        }
    }

    private async renderBlogList(): Promise<void> {
        document.title = 'Blog - Victor Retamal';

        if (!this.blogData) {
            this.renderError();
            return;
        }

        this.container.innerHTML = `
            <div class="min-h-screen bg-dark-bg">
                <!-- Header -->
                <header class="bg-dark-surface border-b border-dark-border">
                    <div class="max-w-4xl mx-auto px-6 py-8">
                        <div class="flex justify-between items-center">
                            <div>
                                <h1 class="text-3xl font-bold text-white mb-2">Blog</h1>
                                <p class="text-gray-400">Thoughts on ML, robotics, and engineering</p>
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

                <!-- Blog Posts -->
                <main class="max-w-4xl mx-auto px-6 py-12">
                    <div id="blog-posts-container" class="space-y-6">
                        <!-- Blog posts will be loaded here -->
                    </div>
                </main>
            </div>
        `;

        this.populateBlogPosts();
    }

    private populateBlogPosts(): void {
        if (!this.blogData) return;

        const container = document.getElementById('blog-posts-container');
        if (!container) return;

        container.innerHTML = '';

        // Sort posts by date in descending order (newest first)
        const sortedPosts = [...this.blogData.posts].sort((a, b) => {
            const dateA = new Date(a.date);
            const dateB = new Date(b.date);
            return dateB.getTime() - dateA.getTime(); // Descending order
        });

        sortedPosts.forEach(post => {
            const postElement = this.createBlogPostCard(post);
            container.appendChild(postElement);
        });
    }

    private createBlogPostCard(post: BlogPost): HTMLElement {
        const card = createElement('article', 'bg-dark-surface border border-dark-border rounded-lg p-6 hover:border-accent-cyan transition-all cursor-pointer');

        card.innerHTML = `
            <div class="flex items-center gap-3 mb-3">
                <span class="text-sm text-gray-500">${this.formatDate(post.date)}</span>
                <span class="text-gray-600">·</span>
                <span class="text-sm text-gray-500">${post.readTime} read</span>
            </div>

            <h2 class="text-xl font-bold text-white mb-3 group-hover:text-accent-cyan transition-colors">${post.title}</h2>

            <p class="text-gray-400 mb-4 leading-relaxed">${post.summary}</p>

            <div class="flex flex-wrap gap-2 mb-4">
                ${post.tags.map(tag => `
                    <span class="px-2 py-1 text-xs rounded bg-dark-border text-gray-400">
                        ${tag}
                    </span>
                `).join('')}
            </div>

            <div class="flex items-center justify-between">
                <button class="read-more-btn text-accent-cyan font-medium hover:text-white transition-colors flex items-center" data-slug="${post.slug}">
                    Read Post
                    <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                    </svg>
                </button>
                ${post.notebook ? '<span class="text-sm text-gray-500">📓 Notebook</span>' : '<span class="text-sm text-gray-500">📝 Article</span>'}
            </div>
        `;

        // Add click handler for the entire card
        card.addEventListener('click', () => {
            Navigation.toBlogPost(post.slug);
        });

        return card;
    }

    private getTagColor(tag: string): string {
        const colors: { [key: string]: string } = {
            'machine-learning': 'bg-blue-100 text-blue-800',
            'mathematics': 'bg-green-100 text-green-800',
            'transformers': 'bg-purple-100 text-purple-800',
            'attention': 'bg-yellow-100 text-yellow-800',
            'cnn': 'bg-red-100 text-red-800',
            'computer-vision': 'bg-indigo-100 text-indigo-800',
            'research': 'bg-pink-100 text-pink-800'
        };
        return colors[tag] || 'bg-gray-100 text-gray-800';
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
            <div class="min-h-screen bg-dark-bg flex items-center justify-center">
                <div class="text-center">
                    <h1 class="text-4xl font-bold text-white mb-4">Blog Unavailable</h1>
                    <p class="text-gray-400 mb-6">Failed to load blog posts. Please try again later.</p>
                    <button id="back-btn" class="btn-primary">
                        Back to Home
                    </button>
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