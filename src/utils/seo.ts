/**
 * SEO Manager - Dynamic meta tag management for SPA
 */

import { DEMOS } from './demos.js';

const SITE_URL = 'https://victor-retamal.com';
const SITE_NAME = 'Victor Retamal';
const DEFAULT_IMAGE = `${SITE_URL}/images/og-image.png`;
const DEFAULT_IMAGE_WIDTH = '1200';
const DEFAULT_IMAGE_HEIGHT = '630';
const TWITTER_HANDLE = '@Victor_Retamal_';

interface BlogPostSEO {
    title: string;
    summary: string;
    slug: string;
    date?: string;
    tags?: string[];
}

class SEOManager {
    /** Structured data added by the current page, cleared on navigation */
    private jsonLdIds = new Set<string>();

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

        // Structured data belongs to a single page
        this.clearJsonLd();

        // Article metadata must not survive onto a non-article page
        if (type !== 'article') {
            document.querySelectorAll('meta[property^="article:"]').forEach(node => node.remove());
        }

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
            title: 'Victor Retamal - Machine Learning & Robotics Engineer',
            description: 'ML and robotics engineer working on computer vision, reinforcement learning and multi-agent systems for autonomous robots.',
            url: SITE_URL,
        });

        this.setJsonLd('person-jsonld', {
            '@context': 'https://schema.org',
            ...this.person(),
            description: 'Currently teaching machines to think and move. I design intelligent systems that learn from the world and act in it. ML and robotics engineer working on sim-to-real pipelines, multi-agent reinforcement learning, computer vision and medical imaging.',
            knowsAbout: [
                'Multi-Agent Reinforcement Learning',
                'Sim-to-Real Robotics',
                'Model Predictive Control',
                'Neural Network Quantization',
                'Computer Vision',
                '3D Medical Image Segmentation',
            ],
            alumniOf: [
                { '@type': 'EducationalOrganization', name: 'Vrije Universiteit Amsterdam' },
                { '@type': 'EducationalOrganization', name: 'University of Amsterdam' },
            ],
        });

        this.setJsonLd('website-jsonld', {
            '@context': 'https://schema.org',
            '@type': 'WebSite',
            name: SITE_NAME,
            url: SITE_URL,
            inLanguage: 'en',
            author: this.person(),
        });

        // The demos are the substance of the home page, and each one's
        // explanation sits behind a tab and a button. The markup is in the
        // document either way, but this states plainly what all five are, so
        // nothing has to infer the machine learning ones from a slide-in panel.
        this.setJsonLd('demos-jsonld', {
            '@context': 'https://schema.org',
            '@type': 'ItemList',
            name: 'Interactive robotics and machine learning demos',
            description: 'Simulations on the home page. The physics, control and inference all run client-side, in the browser.',
            itemListOrder: 'https://schema.org/ItemListUnordered',
            numberOfItems: DEMOS.length,
            itemListElement: DEMOS.map((demo, index) => ({
                '@type': 'ListItem',
                position: index + 1,
                item: {
                    '@type': 'WebApplication',
                    name: demo.name,
                    description: demo.summary,
                    url: `${SITE_URL}/#demo-section`,
                    applicationCategory: 'Simulation',
                    operatingSystem: 'Any',
                    browserRequirements: 'Requires JavaScript',
                    isAccessibleForFree: true,
                    keywords: demo.keywords.join(', '),
                    author: this.person(),
                },
            })),
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

        this.setJsonLd('blog-jsonld', {
            '@context': 'https://schema.org',
            '@type': 'Blog',
            name: 'Victor Retamal - Blog',
            description: 'Technical writing on machine learning, robotics and engineering.',
            url: `${SITE_URL}/blog`,
            inLanguage: 'en',
            author: this.person(),
        });

        this.setJsonLd('breadcrumb-jsonld', this.breadcrumbs([
            { name: 'Home', url: SITE_URL },
            { name: 'Blog', url: `${SITE_URL}/blog` },
        ]));
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
        this.setMetaProperty('article:author', SITE_NAME);

        this.setJsonLd('blogposting-jsonld', {
            '@context': 'https://schema.org',
            '@type': 'BlogPosting',
            headline: post.title,
            description: post.summary,
            url,
            mainEntityOfPage: { '@type': 'WebPage', '@id': url },
            inLanguage: 'en',
            image: DEFAULT_IMAGE,
            ...(post.date ? { datePublished: post.date, dateModified: post.date } : {}),
            ...(post.tags?.length ? { keywords: post.tags.join(', ') } : {}),
            author: this.person(),
            publisher: this.person(),
            isPartOf: {
                '@type': 'Blog',
                name: 'Victor Retamal - Blog',
                url: `${SITE_URL}/blog`,
            },
        });

        this.setJsonLd('breadcrumb-jsonld', this.breadcrumbs([
            { name: 'Home', url: SITE_URL },
            { name: 'Blog', url: `${SITE_URL}/blog` },
            { name: post.title, url },
        ]));
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
                jobTitle: 'Machine Learning & Robotics Engineer',
                alumniOf: [
                    { '@type': 'EducationalOrganization', name: 'Vrije Universiteit Amsterdam' },
                    { '@type': 'EducationalOrganization', name: 'University of Amsterdam' },
                ],
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
        this.jsonLdIds.add(id);
    }

    /**
     * Drop structured data from the page we are leaving. Without this a
     * BlogPosting would still be describing the home page after navigating
     * back to it.
     */
    private clearJsonLd(): void {
        // Every block goes, not just the ones this instance added: a page
        // served through the SPA fallback can arrive with structured data
        // already baked into its head by the pre-renderer.
        document.querySelectorAll('script[type="application/ld+json"]').forEach(node => node.remove());
        this.jsonLdIds.clear();
    }

    /** The person behind the site, reused as author and publisher. */
    private person(): Record<string, unknown> {
        return {
            '@type': 'Person',
            name: SITE_NAME,
            url: SITE_URL,
            jobTitle: 'Machine Learning & Robotics Engineer',
            sameAs: [
                'https://github.com/RetamalVictor',
                'https://www.linkedin.com/in/victor-retamal/',
                'https://x.com/Victor_Retamal_',
                'https://scholar.google.com/citations?user=rSJjk7EAAAAJ',
            ],
        };
    }

    private breadcrumbs(trail: Array<{ name: string; url: string }>): Record<string, unknown> {
        return {
            '@context': 'https://schema.org',
            '@type': 'BreadcrumbList',
            itemListElement: trail.map((crumb, index) => ({
                '@type': 'ListItem',
                position: index + 1,
                name: crumb.name,
                item: crumb.url,
            })),
        };
    }

}

// Export singleton instance
export const seo = new SEOManager();
