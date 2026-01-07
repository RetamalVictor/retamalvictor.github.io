import { Router } from './router.js';

/**
 * Navigation utility for consistent routing throughout the application
 * Provides a centralized way to handle navigation that works in both development and production
 */
export class Navigation {
    private static router: Router | null = null;
    private static baseUrl: string = '';

    /**
     * Initialize navigation with router instance and base URL
     */
    public static initialize(router: Router, baseUrl: string = '') {
        Navigation.router = router;
        Navigation.baseUrl = baseUrl;
    }

    /**
     * Navigate to a route using the router if available, otherwise use browser navigation
     */
    public static to(path: string): void {
        if (Navigation.router) {
            // Use client-side routing - router handles base path internally
            Navigation.router.navigate(path);
        } else {
            // Fallback to browser navigation - need full path
            const fullPath = Navigation.getFullPath(path);
            window.location.href = fullPath;
        }
    }

    /**
     * Get the full URL path including base URL for production deployments
     */
    public static getFullPath(path: string): string {
        // Remove leading slash from path to avoid double slashes
        const cleanPath = path.startsWith('/') ? path.substring(1) : path;

        // If base URL is set, prepend it
        if (Navigation.baseUrl) {
            return `${Navigation.baseUrl}/${cleanPath}`;
        }

        return `/${cleanPath}`;
    }

    /**
     * Navigate to blog list page
     */
    public static toBlogList(): void {
        // Router strips base path from routes, so /blog/blog becomes /blog
        // Just use the route as-is
        Navigation.to('/blog');
    }

    /**
     * Navigate to specific blog post
     */
    public static toBlogPost(slug: string): void {
        // Router strips base path from routes, so /blog/blog/:slug becomes /blog/:slug
        // Just use the route as-is
        Navigation.to(`/blog/${slug}`);
    }

    /**
     * Navigate to home page
     */
    public static toHome(): void {
        Navigation.to('/');
    }

    /**
     * Navigate to CV page
     */
    public static toCV(): void {
        Navigation.to('/cv');
    }

    /**
     * Navigate to project detail page
     */
    public static toProject(id: string): void {
        Navigation.to(`/project/${id}`);
    }

    /**
     * Force a full page reload navigation (useful for external links or when router fails)
     */
    public static reload(path: string): void {
        window.location.href = Navigation.getFullPath(path);
    }

    /**
     * Go back in browser history
     */
    public static back(): void {
        if (window.history.length > 1) {
            window.history.back();
        } else {
            // If no history, go to home page
            Navigation.toHome();
        }
    }

    /**
     * Replace current history entry (doesn't add to history stack)
     */
    public static replace(path: string): void {
        const fullPath = Navigation.getFullPath(path);

        if (Navigation.router) {
            // Use router's replace if available (we'd need to add this method to router)
            window.history.replaceState({}, '', fullPath);
            Navigation.router.navigate(fullPath);
        } else {
            window.location.replace(fullPath);
        }
    }

    /**
     * Check if we're currently on a specific route
     */
    public static isCurrentRoute(path: string): boolean {
        const currentPath = window.location.pathname;
        const targetPath = Navigation.getFullPath(path);
        return currentPath === targetPath;
    }

    /**
     * Get current route path
     */
    public static getCurrentPath(): string {
        return window.location.pathname;
    }

    /**
     * External link navigation (opens in new tab by default)
     */
    public static external(url: string, newTab: boolean = true): void {
        if (newTab) {
            window.open(url, '_blank', 'noopener,noreferrer');
        } else {
            window.location.href = url;
        }
    }
}

/**
 * Configuration for different environments
 */
export const NavigationConfig = {
    development: {
        baseUrl: ''
    },
    production: {
        baseUrl: '' // Empty for custom domain (victor-retamal.com)
    }
};

/**
 * Initialize navigation for the current environment
 */
export function initializeNavigation(router: Router, environment: 'development' | 'production' = 'development') {
    const config = NavigationConfig[environment];
    Navigation.initialize(router, config.baseUrl);
}