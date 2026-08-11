/**
 * How much can we ask of this device?
 *
 * The demos on this site download neural networks and run them in the page.
 * That is fine on a laptop and can lose the tab on a phone, so a couple of
 * places need to agree on what counts as a small device: the hero's download
 * gate, and the depth demo when it picks which model to fetch.
 */

/**
 * Is this a device where a large model is a real risk?
 *
 * Either signal is enough, and deliberately so. `deviceMemory` is the direct
 * answer, but Safari does not implement it — which is exactly the browser that
 * needs this most — and phones that do implement it happily report 8 GB while
 * still killing a tab that allocates a few hundred megabytes. So a phone counts
 * as constrained on its shape alone: a coarse pointer with a short side under
 * 500 CSS pixels, which covers phones and leaves tablets and laptops alone.
 */
export function isMemoryConstrained(): boolean {
    if (typeof navigator === 'undefined' || typeof window === 'undefined') return false;

    const memory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
    if (typeof memory === 'number' && memory <= 4) return true;

    if (typeof window.matchMedia !== 'function') return false;
    return window.matchMedia('(pointer: coarse)').matches
        && Math.min(window.screen.width, window.screen.height) <= 500;
}
