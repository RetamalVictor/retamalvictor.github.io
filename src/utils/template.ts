import { config } from './config.js';
import { loadTemplate } from './dom.js';

/**
 * Enhanced template manager that handles configuration-driven templates
 */
export class TemplateManager {
    private static instance: TemplateManager;
    private templateCache: Map<string, string> = new Map();

    private constructor() {}

    public static getInstance(): TemplateManager {
        if (!TemplateManager.instance) {
            TemplateManager.instance = new TemplateManager();
        }
        return TemplateManager.instance;
    }

    /**
     * Load a template with configuration variable replacement
     */
    public async loadConfigurableTemplate(templatePath: string): Promise<string> {
        try {
            // Check cache first
            const cacheKey = templatePath;
            if (this.templateCache.has(cacheKey)) {
                return this.templateCache.get(cacheKey)!;
            }

            // Load template
            const rawTemplate = await loadTemplate(templatePath);

            // Replace configuration variables
            const configData = config.getAllConfig();
            const processedTemplate = config.replaceTemplateVars(rawTemplate, configData);

            // Cache the processed template
            this.templateCache.set(cacheKey, processedTemplate);

            return processedTemplate;
        } catch (error) {
            console.error(`Failed to load configurable template ${templatePath}:`, error);
            throw error;
        }
    }

    /**
     * Load and render a section with dynamic content
     */
    public async loadSection(sectionName: string): Promise<string> {
        const templatePath = `/src/templates/${sectionName}-section.html`;
        return this.loadConfigurableTemplate(templatePath);
    }

    /**
     * Create research areas HTML from configuration
     */
    public createResearchAreasHTML(): string {
        const researchData = config.getResearchAreas();

        return researchData.areas.map(area => {
            // Get related items for this area
            const relatedItems = config.getRelatedItems(area.id);
            const hasRelatedItems = relatedItems.projects.length > 0 ||
                                   relatedItems.blogs.length > 0 ||
                                   relatedItems.publications.length > 0;

            // Create related items HTML
            const relatedItemsHTML = hasRelatedItems ? `
                <div class="related-items mt-4 pt-4 border-t border-gray-100 hidden" data-area-id="${area.id}">
                    ${relatedItems.projects.length > 0 ? `
                        <div class="mb-3">
                            <h5 class="text-sm font-semibold text-gray-700 mb-2">Projects</h5>
                            <ul class="space-y-2">
                                ${relatedItems.projects.map(project => `
                                    <li>
                                        <a href="/blog/project/${project.id}" class="text-sm text-blue-600 hover:text-blue-800 hover:underline flex items-start">
                                            <svg class="w-4 h-4 mr-1 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                                            </svg>
                                            <span>${project.title}</span>
                                        </a>
                                    </li>
                                `).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    ${relatedItems.blogs.length > 0 ? `
                        <div class="mb-3">
                            <h5 class="text-sm font-semibold text-gray-700 mb-2">Blog Posts</h5>
                            <ul class="space-y-2">
                                ${relatedItems.blogs.map(blog => `
                                    <li>
                                        <a href="/blog/${blog.slug}" class="text-sm text-green-600 hover:text-green-800 hover:underline flex items-start">
                                            <svg class="w-4 h-4 mr-1 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                                            </svg>
                                            <span>${blog.title}</span>
                                        </a>
                                    </li>
                                `).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    ${relatedItems.publications.length > 0 ? `
                        <div class="mb-3">
                            <h5 class="text-sm font-semibold text-gray-700 mb-2">Publications</h5>
                            <ul class="space-y-2">
                                ${relatedItems.publications.map(pub => {
                                    const authors = Array.isArray(pub.authors) ? pub.authors.join(', ') : pub.authors;
                                    const authorsWithBold = authors
                                        .replace(/Víctor Retamal Guiberteau/g, '<strong>Víctor Retamal Guiberteau</strong>')
                                        .replace(/Victor Retamal/g, '<strong>Victor Retamal</strong>');
                                    const doiUrl = pub.doi ? `https://doi.org/${pub.doi}` : '#';
                                    return `
                                    <li>
                                        <a href="${doiUrl}" target="_blank" rel="noopener" class="text-sm text-purple-600 hover:text-purple-800 hover:underline flex items-start">
                                            <svg class="w-4 h-4 mr-1 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path>
                                            </svg>
                                            <span>${pub.title}</span>
                                        </a>
                                        <p class="text-xs text-gray-500 ml-5">${authorsWithBold}</p>
                                        <p class="text-xs text-gray-500 ml-5">${pub.journal}, ${pub.date}</p>
                                    </li>
                                `}).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            ` : '';

            return `
                <div class="research-area-card bg-white rounded-lg shadow-sm border border-gray-100 transition-all ${hasRelatedItems ? 'cursor-pointer hover:shadow-md' : ''}"
                     data-area-id="${area.id}"
                     ${hasRelatedItems ? 'role="button" tabindex="0" aria-expanded="false"' : ''}>
                    <div class="flex items-center space-x-4 p-4">
                        <div class="w-12 h-12 ${area.color_scheme.background} rounded-lg flex items-center justify-center flex-shrink-0">
                            <svg class="w-6 h-6 ${area.color_scheme.icon}" fill="currentColor" viewBox="0 0 20 20">
                                <path d="${area.icon_path}"></path>
                            </svg>
                        </div>
                        <div class="flex-1">
                            <h4 class="text-lg font-semibold">${area.title}</h4>
                            <p class="text-gray-600 text-sm">${area.description}</p>
                        </div>
                        ${hasRelatedItems ? `
                            <div class="chevron-icon flex-shrink-0 transition-transform duration-200">
                                <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                                </svg>
                            </div>
                        ` : ''}
                    </div>
                    ${relatedItemsHTML}
                </div>
            `;
        }).join('');
    }

    /**
     * Create footer social links HTML
     */
    public createFooterSocialLinksHTML(): string {
        const siteConfig = config.getSiteConfig();

        const socialNetworks = [
            { key: 'linkedin', icon: 'M16.338 16.338H13.67V12.16c0-.995-.017-2.277-1.387-2.277-1.39 0-1.601 1.086-1.601 2.207v4.248H8.014v-8.59h2.559v1.174h.037c.356-.675 1.227-1.387 2.526-1.387 2.703 0 3.203 1.778 3.203 4.092v4.711zM5.005 6.575a1.548 1.548 0 11-.003-3.096 1.548 1.548 0 01.003 3.096zm-1.337 9.763H6.34v-8.59H3.667v8.59zM17.668 1H2.328C1.595 1 1 1.581 1 2.298v15.403C1 18.418 1.595 19 2.328 19h15.34c.734 0 1.332-.582 1.332-1.299V2.298C19 1.581 18.402 1 17.668 1z' },
            { key: 'github', icon: 'M10 0C4.477 0 0 4.484 0 10.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0110 4.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.203 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.942.359.31.678.921.678 1.856 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0020 10.017C20 4.484 15.522 0 10 0z' }
        ];

        return socialNetworks.map(network => {
            const username = siteConfig.social[network.key as keyof typeof siteConfig.social];
            if (!username) return '';

            const url = config.getSocialURL(network.key, username);
            const capitalizedNetwork = network.key.charAt(0).toUpperCase() + network.key.slice(1);

            return `
                <a href="${url}" class="text-gray-300 hover:text-white transition-colors" target="_blank" rel="noopener">
                    <span class="sr-only">${capitalizedNetwork}</span>
                    <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="${network.icon}" clip-rule="evenodd"></path>
                    </svg>
                </a>
            `;
        }).filter(link => link).join('');
    }

    /**
     * Create footer research areas list HTML
     */
    public createFooterResearchAreasHTML(): string {
        const researchAreas = config.getSiteConfig().research_areas;
        return researchAreas.map(area => `<li>${area}</li>`).join('');
    }

    /**
     * Clear template cache (useful for development)
     */
    public clearCache(): void {
        this.templateCache.clear();
    }
}

// Export singleton instance
export const templateManager = TemplateManager.getInstance();