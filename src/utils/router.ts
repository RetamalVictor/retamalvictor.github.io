export interface Route {
    path: string;
    component: () => Promise<void> | void;
    title?: string;
}

// Allowed characters for route parameters (alphanumeric, hyphens, underscores, dots)
const SAFE_PARAM_PATTERN = /^[a-zA-Z0-9_.-]+$/;

export class Router {
    private routes: Map<string, Route> = new Map();
    private currentRoute: string = '/';
    private initialized: boolean = false;
    private basePath: string = '';

    constructor(basePath: string = '') {
        this.basePath = basePath;
        this.init();
    }

    private init(): void {
        window.addEventListener('popstate', () => {
            this.handleRouteChange();
        });

        // Don't handle initial route here - wait until routes are added
    }

    public initialize(): void {
        // Handle initial route after all routes have been registered
        if (!this.initialized) {
            this.initialized = true;
            this.handleRouteChange();
        }
    }

    public addRoute(path: string, component: () => Promise<void> | void, title?: string): void {
        this.routes.set(path, { path, component, title });
    }

    public navigate(path: string): void {
        if (path === this.currentRoute) return;

        this.currentRoute = path;
        // Add base path when updating browser URL
        const fullPath = this.basePath ? `${this.basePath}${path}` : path;
        window.history.pushState({}, '', fullPath);
        this.handleRouteChange();
    }

    private async handleRouteChange(): Promise<void> {
        let path = window.location.pathname;

        // Strip base path if present
        if (this.basePath && path.startsWith(this.basePath)) {
            path = path.substring(this.basePath.length);
            // Ensure path starts with /
            if (!path.startsWith('/')) {
                path = '/' + path;
            }
        }

        // Handle empty path as root
        if (path === '') {
            path = '/';
        }

        this.currentRoute = path;

        // Check for exact match
        let route = this.routes.get(path);

        // Check for dynamic routes (e.g., /project/:id)
        if (!route) {
            for (const [routePath, routeConfig] of this.routes) {
                if (this.matchDynamicRoute(routePath, path)) {
                    route = routeConfig;
                    break;
                }
            }
        }

        // Fallback to home route
        if (!route) {
            route = this.routes.get('/');
        }

        if (route) {
            // Update page title
            if (route.title) {
                document.title = `${route.title} - Victor Retamal`;
            }

            // Execute route component
            try {
                await route.component();
            } catch (error) {
                console.error('Router: Error loading route:', error);
                // On error, navigate to home
                const homeRoute = this.routes.get('/');
                if (homeRoute && homeRoute !== route) {
                    await homeRoute.component();
                }
            }
        }
    }

    private matchDynamicRoute(routePath: string, currentPath: string): boolean {
        const routeParts = routePath.split('/');
        const currentParts = currentPath.split('/');

        if (routeParts.length !== currentParts.length) {
            return false;
        }

        return routeParts.every((part, index) => {
            return part.startsWith(':') || part === currentParts[index];
        });
    }

    public getRouteParams(routePath: string, currentPath: string): Record<string, string> {
        // Strip base path from currentPath if present
        let cleanPath = currentPath;
        if (this.basePath && cleanPath.startsWith(this.basePath)) {
            cleanPath = cleanPath.substring(this.basePath.length);
            if (!cleanPath.startsWith('/')) {
                cleanPath = '/' + cleanPath;
            }
        }

        const routeParts = routePath.split('/');
        const currentParts = cleanPath.split('/');
        const params: Record<string, string> = {};

        routeParts.forEach((part, index) => {
            if (part.startsWith(':')) {
                const paramName = part.substring(1);
                const paramValue = currentParts[index] || '';

                // Validate parameter against safe pattern to prevent injection attacks
                if (!SAFE_PARAM_PATTERN.test(paramValue)) {
                    console.warn(`Router: Invalid parameter value for ${paramName}: ${paramValue}`);
                    params[paramName] = '';  // Return empty string for invalid params
                } else {
                    params[paramName] = paramValue;
                }
            }
        });

        return params;
    }

    public getCurrentRoute(): string {
        return this.currentRoute;
    }
}