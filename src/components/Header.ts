import { loadTemplate, smoothScrollTo } from '../utils/dom.js';
import { Navigation } from '../utils/navigation.js';
import { Router } from '../utils/router.js';

export class Header {
    private element: HTMLElement | null = null;
    private mobileMenuOpen = false;

    constructor(private container: HTMLElement, _router?: Router) {
        this.init();
    }

    private async init(): Promise<void> {
        try {
            const template = await loadTemplate('/src/templates/header.html');
            this.container.innerHTML = template;
            this.element = this.container.querySelector('header');

            if (this.element) {
                this.setupEventListeners();
                this.setupScrollEffect();
            }
        } catch (error) {
            console.error('Failed to initialize header:', error);
            this.renderFallback();
        }
    }

    private setupEventListeners(): void {
        if (!this.element) return;

        const mobileMenuBtn = this.element.querySelector('#mobile-menu-btn');
        const navLinks = this.element.querySelectorAll('.nav-link, #mobile-menu a');

        mobileMenuBtn?.addEventListener('click', () => {
            this.toggleMobileMenu();
        });

        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                const href = (link as HTMLAnchorElement).getAttribute('href');
                if (href?.startsWith('#')) {
                    e.preventDefault();
                    const targetId = href.substring(1);
                    smoothScrollTo(targetId);
                    if (this.mobileMenuOpen) {
                        this.toggleMobileMenu();
                    }
                } else if (href?.startsWith('/')) {
                    // Handle route navigation (like /blog)
                    e.preventDefault(); // Prevent default browser navigation
                    if (this.mobileMenuOpen) {
                        this.toggleMobileMenu();
                    }
                    // Use Navigation module for consistent routing
                    Navigation.to(href);
                }
            });
        });

        document.addEventListener('click', (e) => {
            if (this.mobileMenuOpen && !this.element?.contains(e.target as Node)) {
                this.toggleMobileMenu();
            }
        });
    }

    private toggleMobileMenu(): void {
        const mobileMenu = this.element?.querySelector('#mobile-menu');
        if (mobileMenu) {
            this.mobileMenuOpen = !this.mobileMenuOpen;
            if (this.mobileMenuOpen) {
                mobileMenu.classList.remove('hidden');
                mobileMenu.classList.add('animate-fade-in-down');
            } else {
                mobileMenu.classList.add('hidden');
                mobileMenu.classList.remove('animate-fade-in-down');
            }
        }
    }

    private setupScrollEffect(): void {
        let lastScrollY = window.scrollY;

        window.addEventListener('scroll', () => {
            const currentScrollY = window.scrollY;

            if (this.element) {
                if (currentScrollY > 100) {
                    this.element.classList.add('bg-opacity-95', 'backdrop-blur-sm');
                } else {
                    this.element.classList.remove('bg-opacity-95', 'backdrop-blur-sm');
                }

                if (currentScrollY > lastScrollY && currentScrollY > 500) {
                    this.element.style.transform = 'translateY(-100%)';
                } else {
                    this.element.style.transform = 'translateY(0)';
                }
            }

            lastScrollY = currentScrollY;
        });
    }

    private renderFallback(): void {
        this.container.innerHTML = `
            <header class="bg-gray-900 text-white sticky top-0 z-50">
                <nav class="max-w-6xl mx-auto px-6 py-4">
                    <div class="flex justify-between items-center">
                        <h1 class="text-2xl font-bold">Victor Retamal</h1>
                        <div class="space-x-6">
                            <a href="#about" class="hover:text-blue-400">About</a>
                            <a href="/blog" class="hover:text-blue-400">Blog</a>
                            <a href="/blog" class="hover:text-blue-400">Blog</a>
                        </div>
                    </div>
                </nav>
            </header>
        `;
    }

    public updateActiveLink(sectionId: string): void {
        if (!this.element) return;

        const links = this.element.querySelectorAll('.nav-link');
        links.forEach(link => {
            const href = (link as HTMLAnchorElement).getAttribute('href');
            if (href === `#${sectionId}`) {
                link.classList.add('text-blue-400');
            } else {
                link.classList.remove('text-blue-400');
            }
        });
    }
}