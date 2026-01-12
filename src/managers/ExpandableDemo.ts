/**
 * Manages the expand/collapse functionality for fullscreen demo view
 */
export class ExpandableDemo {
    private originalParentId: string;
    private isExpanded: boolean = false;
    private overlay: HTMLElement | null = null;
    private wrapper: HTMLElement | null = null;
    private boundHandleEscape: (e: KeyboardEvent) => void;

    constructor(originalParentId: string) {
        this.originalParentId = originalParentId;
        this.boundHandleEscape = this.handleEscape.bind(this);
    }

    /**
     * Initialize the expand button and keyboard shortcuts
     */
    public initialize(): void {
        const expandBtn = document.getElementById('expand-demo-btn');
        if (!expandBtn) return;

        expandBtn.addEventListener('click', () => this.toggle());
        document.addEventListener('keydown', this.boundHandleEscape);
    }

    /**
     * Handle escape key to collapse
     */
    private handleEscape(e: KeyboardEvent): void {
        if (e.key === 'Escape' && this.isExpanded) {
            this.collapse();
        }
    }

    /**
     * Toggle between expanded and collapsed states
     */
    public toggle(): void {
        if (this.isExpanded) {
            this.collapse();
        } else {
            this.expand();
        }
    }

    /**
     * Expand demo to fullscreen modal
     */
    public expand(): void {
        const demoContainer = document.getElementById('demo-container');
        const expandBtn = document.getElementById('expand-demo-btn');
        if (!demoContainer || !expandBtn) return;

        this.isExpanded = true;

        // Create dark overlay
        this.overlay = document.createElement('div');
        this.overlay.id = 'expand-overlay';
        this.overlay.style.cssText = `
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.85);
            z-index: 40;
        `;
        this.overlay.addEventListener('click', () => this.collapse());
        document.body.appendChild(this.overlay);

        // Create expanded wrapper
        this.wrapper = document.createElement('div');
        this.wrapper.id = 'expanded-demo-wrapper';
        this.wrapper.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 90vw;
            height: 85vh;
            max-width: 1400px;
            background: #0a0a0f;
            border: 1px solid #1e1e2e;
            border-radius: 12px;
            z-index: 50;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        `;

        // Create close button
        const closeBtn = this.createCloseButton();
        this.wrapper.appendChild(closeBtn);

        // Move demo container into expanded wrapper
        const contentArea = document.createElement('div');
        contentArea.style.cssText = 'flex: 1; overflow: hidden;';
        demoContainer.dataset.originalParent = this.originalParentId;
        contentArea.appendChild(demoContainer);
        this.wrapper.appendChild(contentArea);
        document.body.appendChild(this.wrapper);

        // Update container styles for expanded view
        demoContainer.classList.remove('relative');
        demoContainer.style.height = '100%';
        const threeContainer = demoContainer.querySelector('.three-container') as HTMLElement;
        if (threeContainer) {
            threeContainer.classList.remove('h-80', 'lg:h-96');
            threeContainer.style.height = '100%';
        }

        // Update expand button icon
        this.updateButtonIcon(expandBtn, true);

        // Trigger resize after layout settles
        setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
    }

    /**
     * Collapse from fullscreen back to inline
     */
    public collapse(): void {
        const demoContainer = document.getElementById('demo-container');
        const expandBtn = document.getElementById('expand-demo-btn');
        if (!demoContainer || !expandBtn) return;

        this.isExpanded = false;

        // Restore demo container to original position
        const demoSection = document.getElementById(this.originalParentId);
        if (demoSection) {
            // Find the tab bar and insert after it
            const tabBar = demoSection.querySelector('.flex.items-center.justify-between');
            if (tabBar && tabBar.nextSibling) {
                demoSection.insertBefore(demoContainer, tabBar.nextSibling);
            } else {
                demoSection.appendChild(demoContainer);
            }

            // Restore container styles
            demoContainer.classList.add('relative');
            demoContainer.style.height = '';
            const threeContainer = demoContainer.querySelector('.three-container') as HTMLElement;
            if (threeContainer) {
                threeContainer.classList.add('h-80', 'lg:h-96');
                threeContainer.style.height = '';
            }
        }

        // Remove overlay and wrapper
        if (this.overlay) {
            this.overlay.remove();
            this.overlay = null;
        }
        if (this.wrapper) {
            this.wrapper.remove();
            this.wrapper = null;
        }

        // Update expand button icon
        this.updateButtonIcon(expandBtn, false);

        // Trigger resize
        setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
    }

    /**
     * Create the close button for expanded view
     */
    private createCloseButton(): HTMLButtonElement {
        const closeBtn = document.createElement('button');
        closeBtn.innerHTML = `
            <svg style="width: 24px; height: 24px;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
        `;
        closeBtn.style.cssText = `
            position: absolute;
            top: 12px;
            right: 12px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            color: #ffffff;
            cursor: pointer;
            padding: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
            z-index: 60;
        `;
        closeBtn.addEventListener('mouseenter', () => {
            closeBtn.style.background = 'rgba(255, 0, 0, 0.3)';
            closeBtn.style.borderColor = 'rgba(255, 100, 100, 0.5)';
        });
        closeBtn.addEventListener('mouseleave', () => {
            closeBtn.style.background = 'rgba(255, 255, 255, 0.1)';
            closeBtn.style.borderColor = 'rgba(255, 255, 255, 0.2)';
        });
        closeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.collapse();
        });
        return closeBtn;
    }

    /**
     * Update expand button icon based on state
     */
    private updateButtonIcon(btn: HTMLElement, expanded: boolean): void {
        if (expanded) {
            btn.innerHTML = `
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                </svg>
            `;
            btn.title = 'Close expanded view';
        } else {
            btn.innerHTML = `
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"></path>
                </svg>
            `;
            btn.title = 'Expand demo';
        }
    }

    /**
     * Check if demo is currently expanded
     */
    public isCurrentlyExpanded(): boolean {
        return this.isExpanded;
    }

    /**
     * Cleanup event listeners
     */
    public destroy(): void {
        document.removeEventListener('keydown', this.boundHandleEscape);
        if (this.isExpanded) {
            this.collapse();
        }
    }
}
