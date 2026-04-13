import * as yaml from 'js-yaml';
import { parseFrontmatter, extractSlugFromPath } from './frontmatter.js';

// Import all configuration files as raw strings
import siteConfigYaml from '../data/site-config.yaml?raw';
import uiTextYaml from '../data/ui-text.yaml?raw';
import themeConfigYaml from '../data/theme-config.yaml?raw';
import researchAreasYaml from '../data/research-areas.yaml?raw';
import blogConfigYaml from '../data/blog-posts.yaml?raw'; // Only for blog_config section

// Auto-import all markdown files for blog posts
const markdownModules = import.meta.glob('../content/markdown/*.md', {
    query: '?raw',
    import: 'default',
    eager: true,
}) as Record<string, string>;

// Configuration interfaces
export interface SiteConfig {
    site: {
        title: string;
        description: string;
        url: string;
        language: string;
        copyright: string;
    };
    personal: {
        name: string;
        title: string;
        subtitle: string;
        label: string;
        location: string;
        email: string;
        phone: string;
        website: string;
    };
    bio: {
        short: string;
        long: string;
    };
    social: {
        linkedin: string;
        github: string;
        twitter: string;
        orcid: string;
        researchgate: string;
    };
    social_urls: {
        [key: string]: string;
    };
    pages: {
        [key: string]: string;
    };
    assets: Record<string, string>;
    research_areas: string[];
}

export interface UIText {
    navigation: any;
    buttons: any;
    sections: any;
    content: any;
    status: any;
    errors: any;
    accessibility: any;
    blog: any;
    projects: any;
    forms: any;
    time: any;
    placeholders: any;
}

export interface ThemeConfig {
    brand: any;
    project_categories: any;
    blog_tags: any;
    technology_tags: any;
    gradients: any;
    buttons: any;
    status: any;
    backgrounds: any;
    text: any;
    layout: any;
    animations: any;
    components: any;
    three_viewer: any;
}

export interface ResearchAreas {
    research_focus: {
        title: string;
        description: string;
    };
    areas: Array<{
        id: string;
        title: string;
        description: string;
        icon_path: string;
        color_scheme: {
            background: string;
            icon: string;
        };
        related_items?: {
            projects: string[];
            blogs: string[];
            publications: number[];
        };
    }>;
    visualization: {
        title: string;
        loading_text: string;
        placeholder_text: string;
    };
}

export interface BlogPost {
    slug: string;
    title: string;
    date: string;
    tags: string[];
    summary: string;
    readTime: string;
    notebook: string;
    featured: boolean;
}

export interface BlogPostsData {
    posts: BlogPost[];
    blog_config: {
        title: string;
        subtitle: string;
        description: string;
        author: string;
        posts_per_page: number;
        enable_tags: boolean;
        enable_search: boolean;
    };
}

/**
 * Configuration manager for loading and accessing all app configuration
 */
export class ConfigManager {
    private static instance: ConfigManager;
    private siteConfig: SiteConfig | null = null;
    private uiText: UIText | null = null;
    private themeConfig: ThemeConfig | null = null;
    private researchAreas: ResearchAreas | null = null;
    private blogPostsData: BlogPostsData | null = null;

    private constructor() {}

    public static getInstance(): ConfigManager {
        if (!ConfigManager.instance) {
            ConfigManager.instance = new ConfigManager();
        }
        return ConfigManager.instance;
    }

    /**
     * Initialize all configuration data
     */
    public async initialize(): Promise<void> {
        try {
            // Load all configurations
            this.siteConfig = yaml.load(siteConfigYaml) as SiteConfig;
            this.uiText = yaml.load(uiTextYaml) as UIText;
            this.themeConfig = yaml.load(themeConfigYaml) as ThemeConfig;
            this.researchAreas = yaml.load(researchAreasYaml) as ResearchAreas;

            // Load blog posts from frontmatter (single source of truth)
            const blogConfig = yaml.load(blogConfigYaml) as any;
            const posts: BlogPost[] = [];

            for (const [path, rawContent] of Object.entries(markdownModules)) {
                const slug = extractSlugFromPath(path);
                const { meta } = parseFrontmatter(rawContent, slug);
                posts.push({
                    slug: meta.slug,
                    title: meta.title,
                    date: meta.date,
                    tags: meta.tags,
                    summary: meta.summary,
                    readTime: meta.readTime,
                    notebook: '',
                    featured: meta.featured,
                });
            }

            // Sort by date (newest first)
            posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

            this.blogPostsData = {
                posts,
                blog_config: blogConfig.blog_config,
            };

            console.log('Configuration loaded successfully');
        } catch (error) {
            console.error('Failed to load configuration:', error);
            throw error;
        }
    }

    /**
     * Get site configuration
     */
    public getSiteConfig(): SiteConfig {
        if (!this.siteConfig) {
            throw new Error('Site configuration not loaded. Call initialize() first.');
        }
        return this.siteConfig;
    }

    /**
     * Get UI text configuration
     */
    public getUIText(): UIText {
        if (!this.uiText) {
            throw new Error('UI text not loaded. Call initialize() first.');
        }
        return this.uiText;
    }

    /**
     * Get theme configuration
     */
    public getThemeConfig(): ThemeConfig {
        if (!this.themeConfig) {
            throw new Error('Theme configuration not loaded. Call initialize() first.');
        }
        return this.themeConfig;
    }

    /**
     * Get research areas data
     */
    public getResearchAreas(): ResearchAreas {
        if (!this.researchAreas) {
            throw new Error('Research areas not loaded. Call initialize() first.');
        }
        return this.researchAreas;
    }

    /**
     * Get blog posts data
     */
    public getBlogPostsData(): BlogPostsData {
        if (!this.blogPostsData) {
            throw new Error('Blog posts data not loaded. Call initialize() first.');
        }
        return this.blogPostsData;
    }

    /**
     * Get social media URL for a given network and username
     */
    public getSocialURL(network: string, username: string): string {
        const siteConfig = this.getSiteConfig();
        const template = siteConfig.social_urls[network.toLowerCase()];
        return template ? template.replace('{username}', username) : '#';
    }

    /**
     * Replace template variables in a string
     */
    public replaceTemplateVars(template: string, variables: { [key: string]: any }): string {
        let result = template;

        // Helper function to get nested object value
        const getNestedValue = (obj: any, path: string): any => {
            return path.split('.').reduce((curr, prop) => curr?.[prop], obj);
        };

        // Replace variables like {{site.title}} or {{personal.name}}
        result = result.replace(/\{\{([^}]+)\}\}/g, (match, path) => {
            const value = getNestedValue(variables, path.trim());
            return value !== undefined ? String(value) : match;
        });

        return result;
    }

    /**
     * Get all configuration data for template replacement
     */
    public getAllConfig(): { [key: string]: any } {
        return {
            site: this.getSiteConfig().site,
            personal: this.getSiteConfig().personal,
            bio: this.getSiteConfig().bio,
            social: this.getSiteConfig().social,
            pages: this.getSiteConfig().pages,
            assets: this.getSiteConfig().assets,
            research_areas: this.getSiteConfig().research_areas,
            navigation: this.getUIText().navigation,
            buttons: this.getUIText().buttons,
            sections: this.getUIText().sections,
            content: this.getUIText().content,
            status: this.getUIText().status,
            errors: this.getUIText().errors,
            accessibility: this.getUIText().accessibility,
            blog: this.getUIText().blog,
            projects: this.getUIText().projects,
            placeholders: this.getUIText().placeholders,
            theme: this.getThemeConfig()
        };
    }

    /**
     * Get related items for a specific research area
     */
    public getRelatedItems(areaId: string) {
        const researchAreas = this.getResearchAreas();
        const area = researchAreas.areas.find(a => a.id === areaId);

        if (!area || !area.related_items) {
            return { projects: [], blogs: [], publications: [] };
        }

        const blogs = this.getBlogPostsBySlug(area.related_items.blogs);
        const publications = this.getPublicationsByIndices(area.related_items.publications);

        return { projects: [], blogs, publications };
    }

    /**
     * Get blog posts by their slugs
     */
    private getBlogPostsBySlug(slugs: string[]) {
        const blogPostsData = this.getBlogPostsData();
        return blogPostsData.posts.filter(p => slugs.includes(p.slug));
    }

    /**
     * Get publications by their indices
     */
    private getPublicationsByIndices(indices: number[]) {
        const researchData = this.getResearchAreas() as any;
        const publications = researchData.publications || [];
        return indices.map((i: number) => publications[i]).filter((p: any) => p !== undefined);
    }
}

// Export singleton instance
export const config = ConfigManager.getInstance();