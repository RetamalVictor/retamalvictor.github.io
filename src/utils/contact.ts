/**
 * Contact address handling.
 *
 * The address is stored encoded and only assembled when someone actually
 * clicks, so it never appears as plain text in the served HTML, in a mailto:
 * attribute, or in the pre-rendered snapshots that crawlers read. Anyone
 * determined can still decode it - this just keeps it off the easy harvest.
 */

import { config } from './config.js';

let cached: string | null = null;

export function contactAddress(): string {
    if (cached) return cached;

    const encoded = config.getSiteConfig()?.personal?.email_encoded || '';
    cached = encoded ? atob(encoded) : '';
    return cached;
}

/** Masked form, safe to render. */
export function maskedAddress(): string {
    const address = contactAddress();
    const [user, domain] = address.split('@');
    if (!user || !domain) return '';
    return `${user.slice(0, 2)}${'•'.repeat(Math.max(user.length - 2, 3))}@${domain}`;
}

export function openMail(subject?: string): void {
    const address = contactAddress();
    if (!address) return;

    const query = subject ? `?subject=${encodeURIComponent(subject)}` : '';
    window.location.href = `mailto:${address}${query}`;
}

export async function copyAddress(): Promise<boolean> {
    const address = contactAddress();
    if (!address) return false;

    try {
        await navigator.clipboard.writeText(address);
        return true;
    } catch {
        // Clipboard needs a secure context; caller decides what to show
        return false;
    }
}
