import { loadTemplate, createElement } from '../utils/dom.js';
import type { Project } from '../types/index.js';
import { seo } from '../utils/seo.js';

export class ProjectDetailPage {
    private container: HTMLElement;
    private project: Project | null = null;

    constructor(container: HTMLElement) {
        this.container = container;
    }

    public async render(projectId: string, projects: Project[]): Promise<void> {
        // Scroll to top when navigating to project page
        window.scrollTo(0, 0);

        this.project = projects.find(p => p.id === projectId) || null;

        if (!this.project) {
            await this.renderNotFound();
            return;
        }

        await this.renderProject();
        this.populateContent();
        this.setupEventListeners();
    }

    private async renderProject(): Promise<void> {
        try {
            const template = await loadTemplate('/src/templates/project-detail.html');
            this.container.innerHTML = template;
        } catch (error) {
            console.error('Failed to load project detail template:', error);
            this.renderFallback();
        }
    }

    private async renderNotFound(): Promise<void> {
        try {
            const template = await loadTemplate('/src/templates/project-not-found.html');
            this.container.innerHTML = template;
            this.setupEventListeners();
        } catch (error) {
            console.error('Failed to load not found template:', error);
            this.container.innerHTML = `
                <div class="min-h-screen bg-gray-50 flex items-center justify-center">
                    <div class="text-center">
                        <h1 class="text-4xl font-bold text-gray-900 mb-4">Project Not Found</h1>
                        <button id="back-btn" class="bg-blue-600 text-white px-6 py-3 rounded-lg">Back to Portfolio</button>
                    </div>
                </div>
            `;
            this.setupEventListeners();
        }
    }

    private populateContent(): void {
        if (!this.project) return;

        // Update SEO meta tags
        seo.project({
            title: this.project.title,
            description: this.project.description,
            id: this.project.id,
        });

        // Hero section
        this.updateElement('project-title', this.project.title);
        this.updateElement('project-description', this.project.description);
        this.updateElement('project-year', this.project.year.toString());

        // Category styling
        this.updateCategoryElements();

        // Technologies
        this.populateTechnologies();

        // Links
        this.populateLinks();

        // Detailed content
        this.populateDetailedContent();

        // Update hero gradient
        this.updateHeroGradient();
    }

    private updateElement(id: string, content: string): void {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = content;
        }
    }

    private updateCategoryElements(): void {
        if (!this.project) return;

        const categoryLabel = this.getCategoryLabel();
        const categoryColor = this.getCategoryColor();

        // Update category badge in hero
        const categoryElement = document.getElementById('project-category');
        if (categoryElement) {
            categoryElement.textContent = categoryLabel;
            categoryElement.className = `inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${categoryColor} text-white`;
        }

        // Update category detail section with tags if available
        const categoryDetailElement = document.getElementById('project-category-detail');
        if (categoryDetailElement) {
            if (this.project.tags && this.project.tags.length > 0) {
                // Show tags as multiple badges
                categoryDetailElement.outerHTML = `
                    <div id="project-category-detail" class="flex flex-wrap gap-2">
                        ${this.project.tags.map(tag => `
                            <span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${categoryColor} text-white">
                                ${tag}
                            </span>
                        `).join('')}
                    </div>
                `;
            } else {
                // Fallback to category label
                categoryDetailElement.textContent = categoryLabel;
                categoryDetailElement.className = `inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${categoryColor} text-white`;
            }
        }

        // Update year details
        const yearDetail = document.getElementById('project-year-detail');
        if (yearDetail) {
            yearDetail.textContent = this.project.year.toString();
        }
    }

    private populateTechnologies(): void {
        if (!this.project) return;

        const container = document.getElementById('project-technologies');
        if (!container) return;

        container.innerHTML = '';
        this.project.technologies.forEach(tech => {
            const span = createElement('span', 'bg-gray-100 text-gray-800 px-3 py-1 rounded-full text-sm', tech);
            container.appendChild(span);
        });
    }

    private populateLinks(): void {
        if (!this.project) return;

        const container = document.getElementById('project-links');
        if (!container) return;

        container.innerHTML = '';

        if (this.project.githubUrl) {
            const githubLink = this.createLinkButton(
                this.project.githubUrl,
                'View Code',
                'bg-white text-blue-600 hover:bg-gray-100',
                this.getGithubIcon()
            );
            container.appendChild(githubLink);
        }

        if (this.project.demoUrl) {
            const demoLink = this.createLinkButton(
                this.project.demoUrl,
                'Live Demo',
                'bg-transparent border-2 border-white text-white hover:bg-white hover:text-blue-600',
                this.getExternalIcon()
            );
            container.appendChild(demoLink);
        }
    }

    private createLinkButton(href: string, text: string, classes: string, icon: string): HTMLElement {
        const link = createElement('a', `${classes} px-6 py-3 rounded-lg font-semibold transition-colors flex items-center`);
        link.setAttribute('href', href);
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener');
        link.innerHTML = `${icon}${text}`;
        return link;
    }

    private populateDetailedContent(): void {
        if (!this.project) return;

        const container = document.getElementById('project-content');
        if (!container) return;

        if (this.project.longDescription) {
            // Split by double newlines to create paragraphs
            const paragraphs = this.project.longDescription
                .split('\n\n')
                .filter(p => p.trim())
                .map(p => `<p class="text-justify mb-4">${p.trim()}</p>`)
                .join('');
            container.innerHTML = paragraphs;
        } else {
            // Generate default content
            const categoryLabel = this.getCategoryLabel().toLowerCase();
            const techs = this.project.technologies.slice(0, 2).join(' and ');

            container.innerHTML = `
                <p>${this.project.description}</p>
                <p><strong>Note:</strong> This is a sample project page. You can customize this content by adding a <code>longDescription</code> field to your project data in <code>src/main.ts</code>.</p>
                <h4 class="text-xl font-semibold mt-6 mb-3">Key Features</h4>
                <ul class="list-disc list-inside space-y-2">
                    <li>Advanced ${categoryLabel} implementation</li>
                    <li>Built with modern ${techs} technologies</li>
                    <li>Optimized for performance and scalability</li>
                    <li>Comprehensive testing and documentation</li>
                </ul>
                <h4 class="text-xl font-semibold mt-6 mb-3">Technical Approach</h4>
                <p>This project demonstrates expertise in ${this.project.technologies.join(', ')}, showcasing best practices in software engineering and research methodologies.</p>
            `;
        }
    }

    private updateHeroGradient(): void {
        if (!this.project) return;

        const heroSection = document.getElementById('project-hero');
        if (heroSection) {
            const gradientClass = this.getGradientColors();
            heroSection.className = heroSection.className.replace(/from-\w+-\d+\s+to-\w+-\d+/, gradientClass);
        }
    }

    private setupEventListeners(): void {
        const backBtn = document.getElementById('back-btn');
        backBtn?.addEventListener('click', () => {
            window.history.back();
        });
    }

    private renderFallback(): void {
        this.container.innerHTML = `
            <div class="min-h-screen bg-gray-50">
                <div class="text-center py-20">
                    <h1 class="text-2xl font-bold mb-4">Project Details</h1>
                    <p class="text-gray-600">Failed to load project template.</p>
                    <button id="back-btn" class="mt-4 bg-blue-600 text-white px-6 py-3 rounded-lg">Back</button>
                </div>
            </div>
        `;
        this.setupEventListeners();
    }

    // Helper methods for styling
    private getGradientColors(): string {
        const gradients = {
            'robotics': 'from-blue-500 to-cyan-600',
            'computer-vision': 'from-purple-500 to-pink-600',
            'machine-learning': 'from-green-500 to-teal-600',
            'multi-agent': 'from-orange-500 to-red-600',
            'research': 'from-indigo-500 to-violet-600'
        };
        return gradients[this.project!.category] || 'from-gray-500 to-gray-600';
    }

    private getCategoryColor(): string {
        const colors = {
            'robotics': 'bg-blue-500',
            'computer-vision': 'bg-purple-500',
            'machine-learning': 'bg-green-500',
            'multi-agent': 'bg-orange-500',
            'research': 'bg-indigo-500'
        };
        return colors[this.project!.category] || 'bg-gray-500';
    }

    private getCategoryLabel(): string {
        const labels = {
            'robotics': 'Robotics',
            'computer-vision': 'Computer Vision',
            'machine-learning': 'Machine Learning',
            'multi-agent': 'Multi-Agent Systems',
            'research': 'Research'
        };
        return labels[this.project!.category] || 'Other';
    }

    private getGithubIcon(): string {
        return `<svg class="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 0C4.477 0 0 4.484 0 10.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0110 4.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.203 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.942.359.31.678.921.678 1.856 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0020 10.017C20 4.484 15.522 0 10 0z" clip-rule="evenodd"></path></svg>`;
    }

    private getExternalIcon(): string {
        return `<svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path></svg>`;
    }

    public destroy(): void {
        // Cleanup if needed
    }
}