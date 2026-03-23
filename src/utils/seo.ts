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

interface ProjectSEO {
    title: string;
    description: string;
    id: string;
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
    }): void {
        const { title, description, url, image = DEFAULT_IMAGE, type = 'website' } = options;

        // Document title
        document.title = title;

        // Meta description
        this.setMetaTag('description', description);

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
            title: 'Victor Retamal - ML Research Engineer',
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
     * SEO for tutoring page
     */
    tutoring(): void {
        this.updateAll({
            title: 'Private Tutoring - Victor Retamal',
            description: 'Private 1-on-1 tutoring in machine learning, deep learning, mathematics, reinforcement learning, computer vision, and robotics.',
            url: `${SITE_URL}/tutoring`,
        });

        // JSON-LD structured data for tutoring service
        this.setJsonLd('tutoring-jsonld', {
            '@context': 'https://schema.org',
            '@type': 'Service',
            name: 'Private ML & Math Tutoring',
            description: 'Private 1-on-1 tutoring in machine learning, deep learning, mathematics, reinforcement learning, computer vision, Python, and robotics.',
            provider: {
                '@type': 'Person',
                name: 'Victor Retamal',
                url: SITE_URL,
                jobTitle: 'Machine Learning Engineer',
                alumniOf: {
                    '@type': 'EducationalOrganization',
                    name: 'Vrije Universiteit Amsterdam & University of Amsterdam',
                },
            },
            serviceType: 'Private Tutoring',
            areaServed: 'Online',
            url: `${SITE_URL}/tutoring`,
            offers: {
                '@type': 'Offer',
                description: 'Free intro call, then flexible 1-on-1 video sessions',
                availability: 'https://schema.org/InStock',
            },
            hasOfferCatalog: {
                '@type': 'OfferCatalog',
                name: 'Tutoring Topics',
                itemListElement: [
                    'Mathematics for ML',
                    'Machine Learning',
                    'Deep Learning',
                    'Reinforcement Learning',
                    'Computer Vision',
                    'Python for Engineers',
                    'Robotics',
                ],
            },
        });
    }

    /**
     * SEO for Spanish tutoring page
     */
    tutoringEs(): void {
        this.updateAll({
            title: 'Clases Particulares - Victor Retamal',
            description: 'Clases particulares 1 a 1 de machine learning, deep learning, matemáticas, reinforcement learning, visión por computador e ingeniería de software.',
            url: `${SITE_URL}/tutoring/es`,
        });

        this.setJsonLd('tutoring-es-jsonld', {
            '@context': 'https://schema.org',
            '@type': 'Service',
            name: 'Clases Particulares de ML y Matemáticas',
            description: 'Clases particulares 1 a 1 de machine learning, deep learning, matemáticas, reinforcement learning, visión por computador, ingeniería de software y tesis.',
            provider: {
                '@type': 'Person',
                name: 'Victor Retamal',
                url: SITE_URL,
                jobTitle: 'Machine Learning Engineer',
                alumniOf: {
                    '@type': 'EducationalOrganization',
                    name: 'Vrije Universiteit Amsterdam & University of Amsterdam',
                },
            },
            serviceType: 'Clases Particulares',
            areaServed: 'Online',
            url: `${SITE_URL}/tutoring/es`,
            inLanguage: 'es',
            offers: {
                '@type': 'Offer',
                description: 'Llamada introductoria gratis, luego sesiones flexibles 1 a 1 por video',
                availability: 'https://schema.org/InStock',
            },
            hasOfferCatalog: {
                '@type': 'OfferCatalog',
                name: 'Temas de Tutoría',
                itemListElement: [
                    'Matemáticas para ML',
                    'Machine Learning',
                    'Deep Learning',
                    'Reinforcement Learning',
                    'Visión por Computador',
                    'Ingeniería ML y Despliegue',
                    'Ingeniería de Software',
                    'Tesis y Proyectos',
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

    /**
     * SEO for project detail page
     */
    project(project: ProjectSEO): void {
        const title = `${project.title} - Victor Retamal`;
        const url = `${SITE_URL}/project/${project.id}`;

        this.updateAll({
            title,
            description: project.description,
            url,
        });
    }
}

// Export singleton instance
export const seo = new SEOManager();
