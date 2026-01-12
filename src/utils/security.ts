/**
 * Security utilities for preventing XSS and other injection attacks.
 */

/**
 * Escape HTML special characters to prevent XSS attacks.
 *
 * Converts characters that have special meaning in HTML to their
 * entity equivalents, making them safe to insert into innerHTML.
 *
 * @param str - The string to escape
 * @returns The escaped string safe for HTML insertion
 *
 * @example
 * ```typescript
 * const userInput = '<script>alert("xss")</script>';
 * element.innerHTML = `<p>${escapeHtml(userInput)}</p>`;
 * // Results in: <p>&lt;script&gt;alert("xss")&lt;/script&gt;</p>
 * ```
 */
export function escapeHtml(str: string): string {
    if (typeof str !== 'string') {
        return '';
    }

    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

/**
 * Escape HTML using the DOM (alternative implementation).
 *
 * Uses the browser's built-in text content escaping for a more
 * comprehensive escape that handles edge cases.
 *
 * @param str - The string to escape
 * @returns The escaped string safe for HTML insertion
 */
export function escapeHtmlDOM(str: string): string {
    if (typeof str !== 'string') {
        return '';
    }

    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/**
 * Escape a string for use in a URL parameter.
 *
 * @param str - The string to escape
 * @returns The URL-encoded string
 */
export function escapeUrlParam(str: string): string {
    if (typeof str !== 'string') {
        return '';
    }

    return encodeURIComponent(str);
}

/**
 * Validate that a string contains only safe URL characters.
 *
 * Useful for validating usernames, slugs, and other URL path components.
 *
 * @param str - The string to validate
 * @returns True if the string is safe for URL paths
 */
export function isSafeUrlPath(str: string): boolean {
    if (typeof str !== 'string') {
        return false;
    }

    // Allow alphanumeric, hyphens, underscores, and dots
    return /^[a-zA-Z0-9_.-]+$/.test(str);
}
