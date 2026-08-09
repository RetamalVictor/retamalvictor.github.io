/**
 * Palette bridge for the WebGL demos.
 *
 * Three.js needs plain numbers, so the CSS variables are resolved here and
 * materials are tagged with the token they came from. When the theme flips,
 * applySceneTheme walks the scene and repaints anything tagged - no rebuild,
 * no interrupted simulation.
 */

import * as THREE from 'three';

/** Resolve a palette token ("--c-accent") to 0xRRGGBB. */
export function themeColor(token: string, fallback = 0x808080): number {
    const raw = getComputedStyle(document.documentElement).getPropertyValue(token).trim();
    if (!raw) return fallback;

    const [r, g, b] = raw.split(/[\s,/]+/).map(Number);
    if ([r, g, b].some(channel => !Number.isFinite(channel))) return fallback;

    return (r << 16) | (g << 8) | b;
}

type Colored = THREE.Material & { color?: THREE.Color };

/** Mark a material as following a palette token, and paint it now. */
export function themed<T extends Colored>(material: T, token: string): T {
    material.userData.themeToken = token;
    material.color?.setHex(themeColor(token));
    return material;
}

/** Repaint every tagged material below this object. */
export function applySceneTheme(root: THREE.Object3D | null): void {
    if (!root) return;

    root.traverse(object => {
        const material = (object as THREE.Mesh).material;
        if (!material) return;

        const materials = Array.isArray(material) ? material : [material];
        for (const entry of materials as Colored[]) {
            const token = entry.userData?.themeToken as string | undefined;
            if (token) entry.color?.setHex(themeColor(token));
        }
    });
}

/** Build a GridHelper in the current theme. Grids bake their colours into the
 *  geometry, so switching theme means replacing the helper. */
export function themedGrid(size: number, divisions: number): THREE.GridHelper {
    return new THREE.GridHelper(
        size,
        divisions,
        themeColor('--c-scene-grid-center'),
        themeColor('--c-scene-grid'),
    );
}

/** Subscribe to theme changes. Returns the unsubscribe function. */
export function onThemeChange(handler: () => void): () => void {
    window.addEventListener('themechange', handler);
    return () => window.removeEventListener('themechange', handler);
}
