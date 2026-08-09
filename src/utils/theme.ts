/**
 * Theme switching.
 *
 * The whole palette lives in CSS variables (see styles/main.css), so flipping
 * one attribute on <html> repaints the site. The choice is remembered in
 * localStorage and applied by a tiny inline script in index.html to avoid a
 * flash of the wrong theme before this module loads.
 */

export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'theme';
const DEFAULT_THEME: Theme = 'light';

export function getTheme(): Theme {
    const current = document.documentElement.getAttribute('data-theme');
    if (current === 'dark' || current === 'light') return current;

    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored === 'dark' || stored === 'light') return stored;
    } catch {
        // Storage can be blocked (private mode, embedded contexts)
    }

    return DEFAULT_THEME;
}

export function setTheme(theme: Theme): void {
    document.documentElement.setAttribute('data-theme', theme);

    try {
        localStorage.setItem(STORAGE_KEY, theme);
    } catch {
        // Non-fatal: the theme still applies for this session
    }

    syncToggles(theme);
    window.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
}

export function toggleTheme(): Theme {
    const next: Theme = getTheme() === 'dark' ? 'light' : 'dark';
    setTheme(next);
    return next;
}

/**
 * Keep every toggle button on the page in sync. Buttons are found by attribute
 * rather than id because pages render their own headers. Visible labels and
 * icons are handled in CSS so late-rendered markup is always correct.
 */
function syncToggles(theme: Theme): void {
    document.querySelectorAll<HTMLElement>('[data-theme-toggle]').forEach(btn => {
        btn.setAttribute('aria-pressed', String(theme === 'dark'));
        btn.setAttribute('title', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
    });
}

/** Call after rendering markup that contains a toggle. */
export function refreshThemeToggles(): void {
    syncToggles(getTheme());
}

/**
 * Markup for a toggle button, for pages that render their own header.
 * Clicks are picked up by the delegated listener in initTheme().
 */
export function themeToggleMarkup(className = 'btn-mini'): string {
    return `
        <button data-theme-toggle type="button" aria-pressed="false" title="Switch theme" class="${className}">
            <svg class="theme-icon-moon w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"></path>
            </svg>
            <svg class="theme-icon-sun w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"></path>
            </svg>
            <span class="theme-label-dark">Dark</span>
            <span class="theme-label-light">Light</span>
        </button>`;
}

let initialized = false;

export function initTheme(): void {
    if (initialized) return;
    initialized = true;

    setTheme(getTheme());

    // Delegated so toggles inside dynamically rendered pages keep working
    document.addEventListener('click', (event) => {
        const target = event.target as HTMLElement | null;
        const toggle = target?.closest('[data-theme-toggle]');
        if (toggle) {
            event.preventDefault();
            toggleTheme();
        }
    });

    // Toggles are rendered after this runs, so refresh their state on the next frame
    requestAnimationFrame(() => syncToggles(getTheme()));
}
