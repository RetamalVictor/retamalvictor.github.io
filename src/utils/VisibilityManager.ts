/**
 * VisibilityManager - Handles visibility-based pausing for demos
 *
 * Tracks both viewport visibility (IntersectionObserver) and page visibility
 * (Page Visibility API) to determine when a component should pause/resume.
 *
 * Usage:
 * ```typescript
 * this.visibilityManager = new VisibilityManager(
 *     this.container,
 *     (paused) => {
 *         if (paused) {
 *             cancelAnimationFrame(this.animationId);
 *         } else {
 *             this.lastFrameTime = performance.now() / 1000;
 *             this.animate();
 *         }
 *     }
 * );
 *
 * // In destroy():
 * this.visibilityManager.destroy();
 * ```
 */
export class VisibilityManager {
    private container: HTMLElement;
    private onPauseChange: (paused: boolean) => void;

    private isPaused: boolean = false;
    private isVisible: boolean = true;
    private isPageVisible: boolean = true;

    private intersectionObserver: IntersectionObserver | null = null;
    private boundHandlePageVisibility: () => void;

    /**
     * Create a new VisibilityManager
     *
     * @param container - The DOM element to observe for viewport visibility
     * @param onPauseChange - Callback invoked when pause state changes
     * @param options - Optional configuration
     */
    constructor(
        container: HTMLElement,
        onPauseChange: (paused: boolean) => void,
        options: { threshold?: number } = {}
    ) {
        this.container = container;
        this.onPauseChange = onPauseChange;
        this.boundHandlePageVisibility = this.handlePageVisibility.bind(this);

        this.setup(options.threshold ?? 0.1);
    }

    /**
     * Setup visibility observers
     */
    private setup(threshold: number): void {
        // IntersectionObserver for viewport visibility
        this.intersectionObserver = new IntersectionObserver(
            (entries) => {
                entries.forEach(entry => {
                    this.isVisible = entry.isIntersecting;
                    this.updatePauseState();
                });
            },
            { threshold }
        );
        this.intersectionObserver.observe(this.container);

        // Page Visibility API for tab switching
        document.addEventListener('visibilitychange', this.boundHandlePageVisibility);
    }

    /**
     * Handle page visibility changes (tab switching)
     */
    private handlePageVisibility(): void {
        this.isPageVisible = document.visibilityState === 'visible';
        this.updatePauseState();
    }

    /**
     * Update pause state and notify callback if changed
     */
    private updatePauseState(): void {
        const shouldPause = !this.isVisible || !this.isPageVisible;

        if (shouldPause !== this.isPaused) {
            this.isPaused = shouldPause;
            this.onPauseChange(this.isPaused);
        }
    }

    /**
     * Check if currently paused
     */
    public getIsPaused(): boolean {
        return this.isPaused;
    }

    /**
     * Manually set paused state (useful for user-triggered pause)
     */
    public setPaused(paused: boolean): void {
        if (paused !== this.isPaused) {
            this.isPaused = paused;
            this.onPauseChange(this.isPaused);
        }
    }

    /**
     * Clean up observers and event listeners
     */
    public destroy(): void {
        if (this.intersectionObserver) {
            this.intersectionObserver.disconnect();
            this.intersectionObserver = null;
        }
        document.removeEventListener('visibilitychange', this.boundHandlePageVisibility);
    }
}
