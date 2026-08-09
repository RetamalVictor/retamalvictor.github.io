/**
 * SEO Manager - Dynamic meta tag management for SPA
 */

const SITE_URL = 'https://victor-retamal.com';
const SITE_NAME = 'Victor Retamal';
const DEFAULT_IMAGE = `${SITE_URL}/images/og-image.png`;
const DEFAULT_IMAGE_WIDTH = '1200';
const DEFAULT_IMAGE_HEIGHT = '630';
const TWITTER_HANDLE = '@vretamal';

interface BlogPostSEO {
    title: string;
    summary: string;
    slug: string;
    date?: string;
    tags?: string[];
}

class SEOManager {
    /**
     * Set or update a meta tag by name attribute
     */
    setMetaTag(name: string, content: string): void {
        let meta = document.querySelector(`meta[name="${name}"]`) as HTMLMetaElement;
        if (!meta) {
            meta = document.createElement('meta');
            meta.name = name;
            document.head.appendChild(meta);
        }
        meta.content = content;
    }

    /**
     * Set or update a meta tag by property attribute (for OG tags)
     */
    setMetaProperty(property: string, content: string): void {
        let meta = document.querySelector(`meta[property="${property}"]`) as HTMLMetaElement;
        if (!meta) {
            meta = document.createElement('meta');
            meta.setAttribute('property', property);
            document.head.appendChild(meta);
        }
        meta.content = content;
    }

    /**
     * Set or update a link tag by rel attribute
     */
    setLinkTag(rel: string, href: string): void {
        let link = document.querySelector(`link[rel="${rel}"]`) as HTMLLinkElement;
        if (!link) {
            link = document.createElement('link');
            link.rel = rel;
            document.head.appendChild(link);
        }
        link.href = href;
    }

    /**
     * Update all SEO tags at once
     */
    private updateAll(options: {
        title: string;
        description: string;
        url: string;
        image?: string;
        type?: string;
        noindex?: boolean;
    }): void {
        const { title, description, url, image = DEFAULT_IMAGE, type = 'website', noindex = false } = options;

        // Document title
        document.title = title;

        // Meta description
        this.setMetaTag('description', description);

        // Indexing - set on every page so it resets when navigating away from
        // an unlisted one
        this.setMetaTag('robots', noindex ? 'noindex, nofollow' : 'index, follow');

        // Canonical URL
        this.setLinkTag('canonical', url);

        // Open Graph
        this.setMetaProperty('og:title', title);
        this.setMetaProperty('og:description', description);
        this.setMetaProperty('og:url', url);
        this.setMetaProperty('og:type', type);
        this.setMetaProperty('og:image', image);
        this.setMetaProperty('og:image:width', DEFAULT_IMAGE_WIDTH);
        this.setMetaProperty('og:image:height', DEFAULT_IMAGE_HEIGHT);
        this.setMetaProperty('og:site_name', SITE_NAME);

        // Twitter Card
        this.setMetaTag('twitter:card', 'summary_large_image');
        this.setMetaTag('twitter:site', TWITTER_HANDLE);
        this.setMetaTag('twitter:creator', TWITTER_HANDLE);
        this.setMetaTag('twitter:title', title);
        this.setMetaTag('twitter:description', description);
        this.setMetaTag('twitter:image', image);
    }

    /**
     * SEO for home page
     */
    home(): void {
        this.updateAll({
            title: 'Victor Retamal - ML & Robotics Engineer',
            description: 'Machine Learning Research Engineer specializing in computer vision, reinforcement learning, and multi-agent systems for robotics applications.',
            url: SITE_URL,
        });
    }

    /**
     * SEO for blog list page
     */
    blogList(): void {
        this.updateAll({
            title: 'Blog - Victor Retamal',
            description: 'Thoughts on machine learning, robotics, and engineering. Technical articles about deep learning, computer vision, and autonomous systems.',
            url: `${SITE_URL}/blog`,
        });
    }

    /**
     * SEO for individual blog post
     */
    blogPost(post: BlogPostSEO): void {
        const title = `${post.title} - Victor Retamal`;
        const url = `${SITE_URL}/blog/${post.slug}`;

        this.updateAll({
            title,
            description: post.summary,
            url,
            type: 'article',
        });

        // Additional article-specific meta
        if (post.date) {
            this.setMetaProperty('article:published_time', post.date);
        }
        if (post.tags && post.tags.length > 0) {
            this.setMetaTag('keywords', post.tags.join(', '));
        }
    }

    /**
     * SEO for services page.
     * The page is currently unlisted: it is reachable by direct link but is
     * not in the navigation, the sitemap, or the pre-render list, so it is
     * marked noindex to keep it out of search results.
     */
    services(): void {
        this.updateAll({
            title: 'Services - Victor Retamal',
            description: 'ML engineering, robotics, and consulting services. Sim-to-real infrastructure, multi-agent RL, medical imaging, and computer vision for edge deployment.',
            url: `${SITE_URL}/services`,
            noindex: true,
        });

        this.setJsonLd('services-jsonld', {
            '@context': 'https://schema.org',
            '@type': 'Service',
            name: 'ML Engineering & Robotics Services',
            description: 'Contract ML engineering: sim-to-real pipelines, multi-agent RL, medical imaging, and computer vision for edge deployment.',
            provider: {
                '@type': 'Person',
                name: 'Victor Retamal',
                url: SITE_URL,
                jobTitle: 'Senior Machine Learning Engineer',
                alumniOf: {
                    '@type': 'EducationalOrganization',
                    name: 'Vrije Universiteit Amsterdam & University of Amsterdam',
                },
            },
            serviceType: 'Engineering Consulting',
            areaServed: 'Online',
            url: `${SITE_URL}/services`,
            hasOfferCatalog: {
                '@type': 'OfferCatalog',
                name: 'Services',
                itemListElement: [
                    'Sim-to-Real Infrastructure',
                    'Multi-Agent RL & Robotics',
                    'Medical Imaging & Surgical AI',
                    'Computer Vision & Edge Deployment',
                ],
            },
        });
    }

    /**
     * Insert or update a JSON-LD script tag
     */
    private setJsonLd(id: string, data: Record<string, unknown>): void {
        let script = document.getElementById(id) as HTMLScriptElement;
        if (!script) {
            script = document.createElement('script');
            script.id = id;
            script.type = 'application/ld+json';
            document.head.appendChild(script);
        }
        script.textContent = JSON.stringify(data);
    }

}

// Export singleton instance
export const seo = new SEOManager();
