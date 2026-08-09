import { createElement } from '../utils/dom.js';
import { Navigation } from '../utils/navigation.js';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import hljs from 'highlight.js';
import 'highlight.js/styles/github-dark.css';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { parseFrontmatter, extractSlugFromPath, type BlogPostMeta } from '../utils/frontmatter.js';
import { seo } from '../utils/seo.js';
import { refreshThemeToggles, themeToggleMarkup } from '../utils/theme.js';

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
    private embeddedDemos: Array<{ destroy(): void }> = [];

    constructor(container: HTMLElement) {
        this.container = container;
    }

    public async render(slug: string): Promise<void> {
        try {
            await this.loadBlogPost(slug);
            await this.renderBlogPost();
            this.setupEventListeners();
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

        // Sanitize HTML to prevent XSS attacks while preserving safe content
        html = DOMPurify.sanitize(html, {
            ALLOWED_TAGS: [
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'br', 'hr',
                'ul', 'ol', 'li', 'dl', 'dt', 'dd',
                'strong', 'em', 'b', 'i', 'u', 's', 'strike', 'del', 'ins',
                'a', 'img', 'figure', 'figcaption',
                'blockquote', 'pre', 'code', 'span',
                'table', 'thead', 'tbody', 'tr', 'th', 'td',
                'div', 'section', 'article', 'aside',
                'sup', 'sub', 'mark', 'abbr', 'details', 'summary'
            ],
            ALLOWED_ATTR: [
                'href', 'src', 'alt', 'title', 'class', 'id', 'name',
                'target', 'rel', 'width', 'height', 'loading',
                'colspan', 'rowspan', 'scope', 'style'
            ],
            ALLOW_DATA_ATTR: false,
            ADD_ATTR: ['target'],
            FORBID_TAGS: ['script', 'style', 'iframe', 'form', 'input', 'object', 'embed'],
            FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover']
        });

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

        seo.blogPost({
            title: this.blogPost.title,
            summary: this.blogPost.summary,
            slug: this.blogPost.slug,
            date: this.blogPost.date,
            tags: this.blogPost.tags,
        });

        this.container.innerHTML = `
            <div class="min-h-screen">
                <!-- Header -->
                <header class="sticky top-0 z-40">
                    <div class="win-bar">
                        <span class="win-title">victor-retamal.com/blog</span>
                        <span class="win-controls" aria-hidden="true"><i></i><i></i><i></i></span>
                    </div>
                    <div class="bg-dark-surface border-b border-dark-border">
                        <div class="max-w-4xl mx-auto px-4 sm:px-6 py-3 flex flex-wrap items-center justify-between gap-2">
                            <div class="flex items-center gap-2">
                                <button id="back-to-blog-btn" class="btn-mini">&#8592; Blog</button>
                                <button id="back-home-btn" class="btn-mini">Home</button>
                            </div>
                            ${themeToggleMarkup()}
                        </div>
                    </div>
                </header>

                <!-- Content -->
                <article class="max-w-4xl mx-auto px-4 sm:px-6 py-8">
                    <div class="win">
                        <div class="win-bar">
                            <span class="win-title">${this.blogPost.slug}.md</span>
                            <span class="ml-auto font-normal tracking-normal normal-case opacity-80">${this.blogPost.readTime} read</span>
                        </div>
                        <div class="win-body">
                            <p class="text-xs font-mono text-gray-500 mb-3">${this.formatDate(this.blogPost.date)}</p>

                            <h1 class="text-3xl md:text-4xl font-black text-white leading-tight mb-4">${this.blogPost.title}</h1>

                            <p class="text-gray-400 leading-relaxed border-l-4 border-accent-cyan pl-4">${this.blogPost.summary}</p>

                            <div class="flex flex-wrap gap-1.5 mt-5">
                                ${this.blogPost.tags.map(tag => `<span class="pill">${tag}</span>`).join('')}
                            </div>

                            <hr class="rule-dots my-7">

                            <div id="blog-content" class="blog-content">
                                ${this.blogPost.content || '<p class="text-gray-400">Content loading...</p>'}
                            </div>
                        </div>
                    </div>

                    <!-- Post Navigation -->
                    <nav class="mt-12 pt-8 border-t-2 border-dotted border-dark-border">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            ${this.prevPost ? `
                                <a href="/blog/${this.prevPost.slug}" class="prev-post-link card group p-4">
                                    <div class="text-[0.65rem] font-bold uppercase tracking-widest text-gray-500 mb-1 flex items-center gap-1">
                                        <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="3" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7"></path>
                                        </svg>
                                        Previous
                                    </div>
                                    <div class="text-white group-hover:text-accent-cyan transition-colors font-semibold">${this.prevPost.title}</div>
                                </a>
                            ` : '<div></div>'}
                            ${this.nextPost ? `
                                <a href="/blog/${this.nextPost.slug}" class="next-post-link card group p-4 text-right">
                                    <div class="text-[0.65rem] font-bold uppercase tracking-widest text-gray-500 mb-1 flex items-center justify-end gap-1">
                                        Next
                                        <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="3" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7"></path>
                                        </svg>
                                    </div>
                                    <div class="text-white group-hover:text-accent-cyan transition-colors font-semibold">${this.nextPost.title}</div>
                                </a>
                            ` : '<div></div>'}
                        </div>
                        <div class="mt-6 text-center">
                            <button id="back-to-blog-btn-bottom" class="btn-mini">&#8592; View all posts</button>
                        </div>
                    </nav>
                </article>
            </div>
        `;

        refreshThemeToggles();
        this.removeDuplicateTitle();

        // Initialize math rendering if content contains math
        this.initializeMathRendering();
        this.initializeSyntaxHighlighting();

        // Initialize embedded demos
        this.initializeEmbeddedDemos();
    }

    private renderNotFound(): void {
        seo.blogPost({
            title: 'Post Not Found',
            summary: 'The blog post you are looking for does not exist.',
            slug: 'not-found',
        });

        this.container.innerHTML = `
            <div class="min-h-screen flex items-center justify-center p-6">
                <div class="win max-w-md w-full text-center">
                    <div class="win-bar">
                        <span class="win-title">404.htm</span>
                        <span class="win-controls" aria-hidden="true"><i></i><i></i><i></i></span>
                    </div>
                    <div class="win-body">
                    <h1 class="text-3xl font-black text-white mb-3">Post not found</h1>
                    <p class="text-gray-400 mb-6">The blog post you're looking for doesn't exist.</p>
                    <div class="flex flex-wrap gap-3 justify-center">
                        <button id="back-to-blog-btn" class="btn-primary">
                            Back to Blog
                        </button>
                        <button id="back-home-btn" class="btn-secondary">
                            Home
                        </button>
                    </div>
                    </div>
                </div>
            </div>
        `;
        this.setupEventListeners();
    }

    /**
     * Posts usually repeat their title as the first heading of the markdown.
     * The page header already shows it, so drop the duplicate.
     */
    private removeDuplicateTitle(): void {
        const heading = document.getElementById('blog-content')?.querySelector('h1');
        if (!heading || !this.blogPost) return;

        const normalize = (text: string) => text.trim().replace(/\s+/g, ' ').toLowerCase();
        if (normalize(heading.textContent || '') === normalize(this.blogPost.title)) {
            heading.remove();
        }
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
                        modelPath: '/assets/models/transformer_new',
                        maxTokens: 50,  // Reduced for CPU inference speed
                        defaultPrompt: 'The meaning of life is',
                    });
                    this.ternaryDemos.push(demo);
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
                } catch (error) {
                    console.error('Failed to initialize training comparison demo:', error);
                }
            }).catch(error => {
                console.error('Failed to load training comparison demo module:', error);
            });
        }

        // CT Fracture Segmentation post components
        const stlContainer = document.getElementById('stl-viewer-demo');
        if (stlContainer) {
            import('../components/stl-viewer').then(({ STLViewer }) => {
                try {
                    const viewer = new STLViewer({ containerId: 'stl-viewer-demo' });
                    this.embeddedDemos.push(viewer);
                } catch (error) {
                    console.error('Failed to initialize STL viewer:', error);
                }
            }).catch(error => {
                console.error('Failed to load STL viewer module:', error);
            });
        }

        const youtubeContainer = document.getElementById('youtube-video');
        if (youtubeContainer) {
            import('../components/ct-segmentation').then(({ YouTubeEmbed }) => {
                try {
                    const embed = new YouTubeEmbed({
                        containerId: 'youtube-video',
                        videoId: 'XxzP5Dqwdmc',
                        caption: 'End-to-end segmentation pipeline: CT input → ABBC prediction → fragment instance recovery → 3D mesh generation.',
                    });
                    this.embeddedDemos.push(embed);
                } catch (error) {
                    console.error('Failed to initialize YouTube embed:', error);
                }
            }).catch(error => {
                console.error('Failed to load YouTube embed module:', error);
            });
        }

        const pipelineContainer = document.getElementById('pipeline-diagram');
        if (pipelineContainer) {
            import('../components/ct-segmentation').then(({ PipelineDiagram }) => {
                try {
                    const diagram = new PipelineDiagram({ containerId: 'pipeline-diagram' });
                    this.embeddedDemos.push(diagram);
                } catch (error) {
                    console.error('Failed to initialize pipeline diagram:', error);
                }
            }).catch(error => {
                console.error('Failed to load pipeline diagram module:', error);
            });
        }

        const archContainer = document.getElementById('architecture-diagram');
        if (archContainer) {
            import('../components/ct-segmentation').then(({ ArchDiagram }) => {
                try {
                    const diagram = new ArchDiagram({ containerId: 'architecture-diagram' });
                    this.embeddedDemos.push(diagram);
                } catch (error) {
                    console.error('Failed to initialize architecture diagram:', error);
                }
            }).catch(error => {
                console.error('Failed to load architecture diagram module:', error);
            });
        }

        // Inline visual components
        const scaleContainer = document.getElementById('scale-comparison');
        if (scaleContainer) {
            import('../components/ct-segmentation').then(({ ScaleComparison }) => {
                try {
                    const viz = new ScaleComparison({ containerId: 'scale-comparison' });
                    this.embeddedDemos.push(viz);
                } catch (error) {
                    console.error('Failed to initialize scale comparison:', error);
                }
            }).catch(error => {
                console.error('Failed to load scale comparison module:', error);
            });
        }

        const voxelContainer = document.getElementById('voxel-anisotropy');
        if (voxelContainer) {
            import('../components/ct-segmentation').then(({ VoxelAnisotropy }) => {
                try {
                    const viz = new VoxelAnisotropy({ containerId: 'voxel-anisotropy' });
                    this.embeddedDemos.push(viz);
                } catch (error) {
                    console.error('Failed to initialize voxel anisotropy:', error);
                }
            }).catch(error => {
                console.error('Failed to load voxel anisotropy module:', error);
            });
        }

        const abbcContainer = document.getElementById('abbc-cross-section');
        if (abbcContainer) {
            import('../components/ct-segmentation').then(({ ABBCDiagram }) => {
                try {
                    const viz = new ABBCDiagram({ containerId: 'abbc-cross-section' });
                    this.embeddedDemos.push(viz);
                } catch (error) {
                    console.error('Failed to initialize ABBC diagram:', error);
                }
            }).catch(error => {
                console.error('Failed to load ABBC diagram module:', error);
            });
        }

        const trainingChartContainer = document.getElementById('training-chart-demo');
        if (trainingChartContainer) {
            import('../components/ct-segmentation').then(({ TrainingChart }) => {
                try {
                    const chart = new TrainingChart({ containerId: 'training-chart-demo' });
                    this.embeddedDemos.push(chart);
                } catch (error) {
                    console.error('Failed to initialize training chart:', error);
                }
            }).catch(error => {
                console.error('Failed to load training chart module:', error);
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
        // Cleanup embedded demos (STL viewer, charts, etc.)
        for (const demo of this.embeddedDemos) {
            try { demo.destroy(); } catch (e) { /* ignore */ }
        }
        this.embeddedDemos = [];

        // Cleanup ternary demos
        for (const demo of this.ternaryDemos) {
            try { demo.destroy(); } catch (e) { /* ignore */ }
        }
        this.ternaryDemos = [];
    }
}