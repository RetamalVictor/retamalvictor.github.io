/**
 * Simple frontmatter parser for markdown files.
 *
 * Frontmatter is YAML between --- delimiters at the start of a file:
 *
 * ---
 * title: My Post
 * date: 2025-01-01
 * tags: [tag1, tag2]
 * ---
 *
 * # Content starts here...
 */

import * as yaml from 'js-yaml';

export interface BlogPostMeta {
    slug: string;
    title: string;
    date: string;
    tags: string[];
    summary: string;
    readTime: string;
    featured: boolean;
}

export interface ParsedMarkdown {
    meta: BlogPostMeta;
    content: string;
}

/**
 * Parse frontmatter from a markdown string.
 * Returns the metadata and the content without frontmatter.
 */
export function parseFrontmatter(markdown: string, slug: string): ParsedMarkdown {
    const frontmatterRegex = /^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/;
    const match = markdown.match(frontmatterRegex);

    if (!match) {
        // No frontmatter found, return defaults
        return {
            meta: {
                slug,
                title: slug.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
                date: new Date().toISOString().split('T')[0],
                tags: [],
                summary: '',
                readTime: '5 min',
                featured: false,
            },
            content: markdown,
        };
    }

    const [, frontmatterYaml, content] = match;

    try {
        const parsed = yaml.load(frontmatterYaml) as Record<string, any>;

        const meta: BlogPostMeta = {
            slug,
            title: parsed.title || slug,
            date: parsed.date || new Date().toISOString().split('T')[0],
            tags: Array.isArray(parsed.tags) ? parsed.tags : [],
            summary: parsed.summary || '',
            readTime: parsed.readTime || '5 min',
            featured: parsed.featured === true,
        };

        return { meta, content: content.trim() };
    } catch (error) {
        console.error(`Failed to parse frontmatter for ${slug}:`, error);
        return {
            meta: {
                slug,
                title: slug,
                date: new Date().toISOString().split('T')[0],
                tags: [],
                summary: '',
                readTime: '5 min',
                featured: false,
            },
            content: markdown,
        };
    }
}

/**
 * Extract slug from a file path.
 * /path/to/my-post.md -> my-post
 */
export function extractSlugFromPath(path: string): string {
    const filename = path.split('/').pop() || '';
    return filename.replace(/\.md$/, '');
}
