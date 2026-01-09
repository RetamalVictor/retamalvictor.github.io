import { createElement } from '../utils/dom.js';
import { Navigation } from '../utils/navigation.js';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import hljs from 'highlight.js';
import 'highlight.js/styles/github-dark.css';
import { marked } from 'marked';
import { parseFrontmatter, extractSlugFromPath, type BlogPostMeta } from '../utils/frontmatter.js';

// Auto-import all markdown files from content/markdown/
// This eliminates the need for manual imports when adding new posts
const markdownModules = import.meta.glob('../content/markdown/*.md', {
    query: '?raw',
    import: 'default',
    eager: true,
}) as Record<string, string>;

interface BlogPost extends BlogPostMeta {
    content?: string;
}

export class BlogPostPage {
    private container: HTMLElement;
    private blogPost: BlogPost | null = null;
    private allPosts: BlogPost[] = [];
    private prevPost: BlogPost | null = null;
    private nextPost: BlogPost | null = null;
    private ternaryDemos: any[] = [];  // TernaryLMDemo instances

    constructor(container: HTMLElement) {
        this.container = container;
    }

    public async render(slug: string): Promise<void> {
        try {
            console.log('Blog Post Page: Starting render for', slug);
            await this.loadBlogPost(slug);
            console.log('Blog Post Page: Post loaded', this.blogPost?.title);
            await this.renderBlogPost();
            console.log('Blog Post Page: Post rendered');
            this.setupEventListeners();
            console.log('Blog Post Page: Event listeners setup - COMPLETE');
        } catch (error) {
            console.error('Blog Post Page: Error during render:', error);
            throw error; // Re-throw to let router handle it
        }
    }

    private async loadBlogPost(slug: string): Promise<void> {
        try {
            // Parse all markdown files and extract metadata from frontmatter
            const posts: BlogPost[] = [];

            for (const [path, rawContent] of Object.entries(markdownModules)) {
                const fileSlug = extractSlugFromPath(path);
                const { meta, content } = parseFrontmatter(rawContent, fileSlug);
                posts.push({
                    ...meta,
                    content: this.convertMarkdownToHTML(content),
                });
            }

            // Sort posts by date (newest first)
            this.allPosts = posts.sort((a, b) => {
                return new Date(b.date).getTime() - new Date(a.date).getTime();
            });

            // Find the post with matching slug and its index
            const currentIndex = this.allPosts.findIndex(post => post.slug === slug);
            const foundPost = currentIndex >= 0 ? this.allPosts[currentIndex] : null;

            if (foundPost) {
                this.blogPost = foundPost;

                // Get prev/next posts (prev = newer, next = older)
                this.prevPost = currentIndex > 0 ? this.allPosts[currentIndex - 1] : null;
                this.nextPost = currentIndex < this.allPosts.length - 1 ? this.allPosts[currentIndex + 1] : null;
            } else {
                this.blogPost = null;
                this.prevPost = null;
                this.nextPost = null;
            }

            console.log('Blog post loaded successfully:', this.blogPost);
        } catch (error) {
            console.error('Failed to load blog post:', error);
            this.blogPost = null;
        }
    }

    private convertMarkdownToHTML(markdown: string): string {
        // Configure marked options for better rendering
        marked.setOptions({
            breaks: true,  // Enable line breaks
            gfm: true,     // GitHub Flavored Markdown
            pedantic: false
        } as any);  // Cast to any to avoid TypeScript issues with newer marked versions

        // Remove social badges section (shields.io links at the end)
        let cleanedMarkdown = markdown.replace(
            /\*\*Connect with me:\*\*.*$/s,
            ''
        );

        // Also remove standalone shields.io badge lines
        cleanedMarkdown = cleanedMarkdown.replace(
            /\[!\[.*?\]\(https:\/\/img\.shields\.io\/.*?\)\]\(.*?\)/g,
            ''
        );

        // Use marked to convert markdown to HTML
        let html = marked.parse(cleanedMarkdown) as string;

        // Fix image paths for production (add base URL)
        const baseUrl = import.meta.env.BASE_URL || '/';
        html = html.replace(
            /(<img[^>]+src=["'])\/images\//g,
            `$1${baseUrl}images/`
        );

        // Remove inline styles from figure elements only (preserve other inline styles for colored code blocks)
        html = html.replace(/<figure([^>]*)\s+style="[^"]*"([^>]*)>/g, '<figure$1$2>');

        // Wrap in blog-content div
        html = `<div class="blog-content">${html}</div>`;

        return html;
    }

    private async renderBlogPost(): Promise<void> {
        if (!this.blogPost) {
            this.renderNotFound();
            return;
        }

        document.title = `${this.blogPost.title} - Victor Retamal`;

        this.container.innerHTML = `
            <div class="min-h-screen bg-dark-bg">
                <!-- Header -->
                <header class="bg-dark-surface border-b border-dark-border">
                    <div class="max-w-4xl mx-auto px-6 py-8">
                        <div class="flex items-center justify-between mb-6">
                            <button id="back-to-blog-btn" class="text-gray-400 hover:text-accent-cyan transition-colors flex items-center">
                                <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
                                </svg>
                                Blog
                            </button>

                            <button id="back-home-btn" class="text-gray-400 hover:text-accent-cyan transition-colors text-sm">
                                Home
                            </button>
                        </div>

                        <h1 class="text-3xl md:text-4xl font-bold text-white mb-4">${this.blogPost.title}</h1>

                        <div class="flex flex-wrap items-center gap-4 text-gray-400">
                            <span>${this.formatDate(this.blogPost.date)}</span>
                            <span>·</span>
                            <span>${this.blogPost.readTime} read</span>
                            <span>·</span>
                            <span>📝 Article</span>
                        </div>

                        <div class="flex flex-wrap gap-2 mt-4">
                            ${this.blogPost.tags.map(tag => `
                                <span class="px-2 py-1 text-xs rounded bg-dark-border text-gray-400">
                                    ${tag}
                                </span>
                            `).join('')}
                        </div>
                    </div>
                </header>

                <!-- Content -->
                <article class="max-w-4xl mx-auto px-6 py-12">
                    <div class="bg-dark-surface border-l-4 border-accent-cyan p-4 mb-8 rounded-r">
                        <p class="text-gray-300 italic">${this.blogPost.summary}</p>
                    </div>

                    <div id="blog-content" class="blog-content blog-content-dark">
                        ${this.blogPost.content || '<p class="text-gray-400">Content loading...</p>'}
                    </div>

                    <!-- Post Navigation -->
                    <nav class="mt-12 pt-8 border-t border-dark-border">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            ${this.prevPost ? `
                                <a href="/blog/${this.prevPost.slug}" class="prev-post-link group p-4 rounded-lg bg-dark-surface border border-dark-border hover:border-accent-cyan transition-all">
                                    <div class="text-sm text-gray-500 mb-1 flex items-center">
                                        <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path>
                                        </svg>
                                        Previous Post
                                    </div>
                                    <div class="text-white group-hover:text-accent-cyan transition-colors font-medium">${this.prevPost.title}</div>
                                </a>
                            ` : '<div></div>'}
                            ${this.nextPost ? `
                                <a href="/blog/${this.nextPost.slug}" class="next-post-link group p-4 rounded-lg bg-dark-surface border border-dark-border hover:border-accent-cyan transition-all text-right">
                                    <div class="text-sm text-gray-500 mb-1 flex items-center justify-end">
                                        Next Post
                                        <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                                        </svg>
                                    </div>
                                    <div class="text-white group-hover:text-accent-cyan transition-colors font-medium">${this.nextPost.title}</div>
                                </a>
                            ` : '<div></div>'}
                        </div>
                        <div class="mt-6 text-center">
                            <button id="back-to-blog-btn-bottom" class="text-gray-400 hover:text-accent-cyan transition-colors">
                                ← View All Posts
                            </button>
                        </div>
                    </nav>
                </article>
            </div>
        `;

        // Initialize math rendering if content contains math
        this.initializeMathRendering();
        this.initializeSyntaxHighlighting();

        // Initialize embedded demos
        this.initializeEmbeddedDemos();
    }

    private renderNotFound(): void {
        document.title = 'Post Not Found - Victor Retamal';

        this.container.innerHTML = `
            <div class="min-h-screen bg-dark-bg flex items-center justify-center">
                <div class="text-center">
                    <h1 class="text-4xl font-bold text-white mb-4">Post Not Found</h1>
                    <p class="text-gray-400 mb-6">The blog post you're looking for doesn't exist.</p>
                    <div class="space-x-4">
                        <button id="back-to-blog-btn" class="btn-primary">
                            Back to Blog
                        </button>
                        <button id="back-home-btn" class="btn-secondary">
                            Home
                        </button>
                    </div>
                </div>
            </div>
        `;
        this.setupEventListeners();
    }

    private formatDate(date: string): string {
        const options: Intl.DateTimeFormatOptions = {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        };
        return new Date(date).toLocaleDateString('en-US', options);
    }

    private initializeMathRendering(): void {
        try {
            const content = document.getElementById('blog-content');
            if (content) {
                let html = content.innerHTML;

                // Process block math ($$...$$) first - handles multiline
                html = html.replace(/\$\$([\s\S]*?)\$\$/g, (match, math) => {
                    try {
                        // Clean up HTML entities and whitespace
                        const cleanedMath = math
                            .replace(/<[^>]*>/g, '') // Remove HTML tags
                            .replace(/<br\s*\/?>/gi, ' ') // Replace <br> with space
                            .replace(/&lt;/g, '<')
                            .replace(/&gt;/g, '>')
                            .replace(/&amp;/g, '&')
                            .replace(/\\{/g, '\\lbrace ')
                            .replace(/\\}/g, '\\rbrace ')
                            .replace(/\\\\/g, '\\\\ ') // Ensure line breaks work in cases environments
                            .trim();
                        return `<div class="katex-display">${katex.renderToString(cleanedMath, {
                            displayMode: true,
                            throwOnError: false,
                            errorColor: '#cc0000'
                        })}</div>`;
                    } catch (error) {
                        console.warn('KaTeX rendering error for block math:', math, error);
                        return match;
                    }
                });

                // Process inline math ($...$)
                // Match $...$ but not $$...$$ (already processed) and not currency like $20
                html = html.replace(/(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)/g, (match, math) => {
                    try {
                        // Clean up HTML entities
                        const cleanedMath = math
                            .replace(/<[^>]*>/g, '') // Remove HTML tags
                            .replace(/&lt;/g, '<')
                            .replace(/&gt;/g, '>')
                            .replace(/&amp;/g, '&')
                            .replace(/\\{/g, '\\lbrace ')
                            .replace(/\\}/g, '\\rbrace ');
                        return katex.renderToString(cleanedMath, {
                            displayMode: false,
                            throwOnError: false,
                            errorColor: '#cc0000'
                        });
                    } catch (error) {
                        console.warn('KaTeX rendering error for inline math:', math, error);
                        return match;
                    }
                });

                content.innerHTML = html;
            }

            console.log('Math rendering with KaTeX completed');
        } catch (error) {
            console.error('Error initializing math rendering:', error);
        }
    }

    private initializeSyntaxHighlighting(): void {
        try {
            // Configure highlight.js for common languages used in ML/math
            hljs.configure({
                languages: ['python', 'javascript', 'typescript', 'bash', 'json', 'yaml', 'markdown', 'latex']
            });

            // Find all code blocks and apply syntax highlighting
            const codeBlocks = document.querySelectorAll('pre code');
            codeBlocks.forEach(block => {
                // Apply highlighting
                hljs.highlightElement(block as HTMLElement);

                // Add copy button to code blocks
                const pre = block.parentElement;
                if (pre) {
                    this.addCopyButton(pre, block as HTMLElement);
                }
            });

            console.log('Syntax highlighting with highlight.js completed');
        } catch (error) {
            console.error('Error initializing syntax highlighting:', error);
        }
    }

    private initializeEmbeddedDemos(): void {
        // Check for ternary LM demo container
        const ternaryContainer = document.getElementById('ternary-lm-demo');
        if (ternaryContainer) {
            // Lazy load the demo component
            import('../components/ternary').then(({ TernaryLMDemo }) => {
                try {
                    const demo = new TernaryLMDemo({
                        containerId: 'ternary-lm-demo',
                        modelPath: '/assets/models/transformer',
                        maxTokens: 50,  // Reduced for CPU inference speed
                        defaultPrompt: 'The meaning of life is',
                    });
                    this.ternaryDemos.push(demo);
                    console.log('Ternary LM demo initialized');
                } catch (error) {
                    console.error('Failed to initialize ternary demo:', error);
                }
            }).catch(error => {
                console.error('Failed to load ternary demo module:', error);
            });
        }

        // Check for training comparison demo container
        const trainingComparisonContainer = document.getElementById('training-comparison-demo');
        if (trainingComparisonContainer) {
            import('../components/tinylm').then(({ TrainingComparisonDemo }) => {
                try {
                    new TrainingComparisonDemo({
                        containerId: 'training-comparison-demo',
                    });
                    console.log('Training comparison demo initialized');
                } catch (error) {
                    console.error('Failed to initialize training comparison demo:', error);
                }
            }).catch(error => {
                console.error('Failed to load training comparison demo module:', error);
            });
        }
    }

    private addCopyButton(pre: HTMLElement, codeBlock: HTMLElement): void {
        // Create copy button
        const copyBtn = createElement('button',
            'absolute top-2 right-2 bg-gray-800 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity hover:bg-gray-700'
        );
        copyBtn.textContent = 'Copy';
        copyBtn.setAttribute('aria-label', 'Copy code to clipboard');

        // Add relative positioning and group class to pre element
        pre.classList.add('relative', 'group');

        // Add copy functionality
        copyBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            e.stopPropagation();

            try {
                const code = codeBlock.textContent || '';
                await navigator.clipboard.writeText(code);

                // Temporary feedback
                const originalText = copyBtn.textContent;
                copyBtn.textContent = 'Copied!';
                copyBtn.classList.add('bg-green-600');

                setTimeout(() => {
                    copyBtn.textContent = originalText;
                    copyBtn.classList.remove('bg-green-600');
                }, 2000);
            } catch (error) {
                console.error('Failed to copy code:', error);
                copyBtn.textContent = 'Failed';
                setTimeout(() => {
                    copyBtn.textContent = 'Copy';
                }, 2000);
            }
        });

        pre.appendChild(copyBtn);
    }

    private setupEventListeners(): void {
        // Back to blog buttons
        const backToBlogBtns = document.querySelectorAll('#back-to-blog-btn, #back-to-blog-btn-bottom');
        backToBlogBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                Navigation.toBlogList();
            });
        });

        // Back to home buttons
        const backHomeBtns = document.querySelectorAll('#back-home-btn');
        backHomeBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                Navigation.toHome();
            });
        });

        // Previous/Next post links
        const prevLink = document.querySelector('.prev-post-link');
        const nextLink = document.querySelector('.next-post-link');

        if (prevLink) {
            prevLink.addEventListener('click', (e) => {
                e.preventDefault();
                const href = prevLink.getAttribute('href');
                if (href) Navigation.to(href);
            });
        }

        if (nextLink) {
            nextLink.addEventListener('click', (e) => {
                e.preventDefault();
                const href = nextLink.getAttribute('href');
                if (href) Navigation.to(href);
            });
        }
    }

    public destroy(): void {
        // Cleanup demos
        for (const demo of this.ternaryDemos) {
            demo.destroy();
        }
        this.ternaryDemos = [];
    }
}