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
