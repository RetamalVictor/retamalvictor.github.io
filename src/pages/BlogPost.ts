import { createElement } from '../utils/dom.js';
import { Navigation } from '../utils/navigation.js';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import hljs from 'highlight.js';
import 'highlight.js/styles/github.css';
import * as yaml from 'js-yaml';
import blogDataYaml from '../data/blog-posts.yaml?raw';
import { marked } from 'marked';

// Import markdown files as raw strings
import learningJourneyMd from '../content/markdown/learning-journey-intro.md?raw';
import mlPipelineMd from '../content/markdown/ml_pipeline_fundamentals.md?raw';
import sysEngPart0Md from '../content/markdown/systems-engineering-part-0.md?raw';
import sysEngPart1Md from '../content/markdown/systems-engineering-part-1.md?raw';
import sysEngPart2Md from '../content/markdown/systems-engineering-part-2.md?raw';

interface BlogPost {
    slug: string;
    title: string;
    date: string;
    tags: string[];
    summary: string;
    readTime: string;
    notebook: string;
    featured: boolean;
    content?: string;
}

export class BlogPostPage {
    private container: HTMLElement;
    private blogPost: BlogPost | null = null;

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
            // Load blog data from YAML file
            const parsedData = yaml.load(blogDataYaml) as any;
            const posts = parsedData.posts || [];

            // Find the post with matching slug
            const foundPost = posts.find((post: any) => post.slug === slug);

            if (foundPost) {
                const content = await this.getPostContent(slug);
                this.blogPost = {
                    ...foundPost,
                    content: content
                };
            } else {
                this.blogPost = null;
            }

            console.log('Blog post loaded successfully:', this.blogPost);
        } catch (error) {
            console.error('Failed to load blog post:', error);
            this.blogPost = null;
        }
    }

    private async getPostContent(slug: string): Promise<string> {
        // Map slugs to imported markdown content
        const markdownContent: Record<string, string> = {
            "learning-journey-intro": learningJourneyMd,
            "ml-pipeline-fundamentals": mlPipelineMd,
            "systems-engineering-part-0": sysEngPart0Md,
            "systems-engineering-part-1": sysEngPart1Md,
            "systems-engineering-part-2": sysEngPart2Md
        };

        // Map slugs to notebook HTML files (still need to be fetched)
        const notebookPaths: Record<string, string> = {
            "linear-regression-basics": "/src/content/notebooks/linear_regression_basics.html",
            "numpy-fundamentals": "/src/content/notebooks/numpy_fundamentals.html"
        };

        // Check if it's a markdown post
        if (markdownContent[slug]) {
            return this.convertMarkdownToHTML(markdownContent[slug]);
        }

        // Check if it's a notebook
        const notebookPath = notebookPaths[slug];
        if (!notebookPath) {
            // Return default content for unmapped slugs
            return `
                <div class="blog-content">
                    <p class="lead">This blog post is coming soon! We're working on converting the Jupyter notebook content to a web-friendly format.</p>
                    <p>Check back soon for the full content, or explore our other available posts in the meantime.</p>
                </div>
            `;
        }

        try {
            // Load notebook HTML from file
            const response = await fetch(notebookPath);
            if (!response.ok) {
                throw new Error(`Failed to load content: ${response.statusText}`);
            }

            // Notebook files are already HTML, just return them
            const content = await response.text();
            return content;
        } catch (error) {
            console.error(`Error loading content for ${slug}:`, error);
            return `
                <div class="blog-content">
                    <p class="lead">Sorry, we couldn't load this blog post content.</p>
                    <p>Please try again later or contact us if the problem persists.</p>
                </div>
            `;
        }
    }

    private convertMarkdownToHTML(markdown: string): string {
        // Configure marked options for better rendering
        marked.setOptions({
            breaks: true,  // Enable line breaks
            gfm: true,     // GitHub Flavored Markdown
            pedantic: false
        } as any);  // Cast to any to avoid TypeScript issues with newer marked versions

        // Use marked to convert markdown to HTML
        let html = marked.parse(markdown) as string;

        // Fix image paths for production (add base URL)
        const baseUrl = import.meta.env.BASE_URL || '/';
        html = html.replace(
            /(<img[^>]+src=["'])\/images\//g,
            `$1${baseUrl}images/`
        );

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
                            <span>${this.blogPost.notebook ? '📓 Notebook' : '📝 Article'}</span>
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

                    <!-- Navigation -->
                    <div class="mt-12 pt-8 border-t border-dark-border">
                        <div class="flex justify-between items-center">
                            <button id="back-to-blog-btn-bottom" class="text-accent-cyan hover:text-white font-medium flex items-center">
                                <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
                                </svg>
                                Back to All Posts
                            </button>
                        </div>
                    </div>
                </article>
            </div>
        `;

        // Initialize math rendering if content contains math
        this.initializeMathRendering();
        this.initializeSyntaxHighlighting();
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
            // Render block math ($$...$$)
            const blockMathElements = document.querySelectorAll('.math-block');
            blockMathElements.forEach(element => {
                const mathContent = element.textContent?.trim();
                if (mathContent) {
                    // Remove $$ delimiters if present
                    const cleanedMath = mathContent.replace(/^\$\$/, '').replace(/\$\$$/, '');
                    element.innerHTML = katex.renderToString(cleanedMath, {
                        displayMode: true,
                        throwOnError: false,
                        errorColor: '#cc0000'
                    });
                }
            });

            // Render inline math ($...$)
            const content = document.getElementById('blog-content');
            if (content) {
                let html = content.innerHTML;

                // Process inline math ($...$) but skip currency (like $20, $1, etc)
                // Only match if not followed/preceded by a digit
                html = html.replace(/(?<!\d)\$([^$]+?)\$(?!\d)/g, (match, math) => {
                    // Skip if the content starts with a digit (likely a price)
                    if (/^\d/.test(math)) {
                        return match; // Return original for prices
                    }
                    try {
                        return katex.renderToString(math, {
                            displayMode: false,
                            throwOnError: false,
                            errorColor: '#cc0000'
                        });
                    } catch (error) {
                        console.warn('KaTeX rendering error for inline math:', error);
                        return match; // Return original if rendering fails
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
    }

    public destroy(): void {
        // Cleanup if needed
    }
}