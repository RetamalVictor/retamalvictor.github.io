/**
 * Palette bridge for the WebGL demos.
 *
 * Three.js needs plain numbers, so the CSS variables are resolved here and
 * materials are tagged with the token they came from. When the theme flips,
 * applySceneTheme walks the scene and repaints anything tagged - no rebuild,
 * no interrupted simulation.
 */

import * as THREE from 'three';

/**
 * Last-resort values, used only when a token cannot be read - which happens
 * if a module evaluates before the stylesheet exists. They mirror the dark
 * palette, so a demo that loses the race still looks like it always did.
 */
export const FALLBACK_COLORS: Record<string, number> = {
    '--c-accent': 0x00d4ff,
    '--c-accent-2': 0xa855f7,
    '--c-ink': 0xffffff,
    '--c-red': 0xff4444,
    '--c-green': 0x22c55e,
    '--c-yellow': 0xfacc15,
    '--c-scene-bg': 0x0a0a0f,
    '--c-scene-grid': 0x1e1e2e,
    '--c-scene-grid-center': 0x2a2a3e,
    '--c-scene-floor': 0x1a1a2e,
};

/**
 * Reading a custom property forces style resolution, so results are cached.
 * The cache is keyed on the theme itself rather than invalidated by an event
 * listener, so it cannot go stale by running after a component's repaint.
 * Values that fell back are never cached - the stylesheet may simply not
 * have arrived yet.
 */
const cache = new Map<string, number>();
let cachedTheme: string | null = null;

/** Resolve a palette token ("--c-accent") to 0xRRGGBB. */
export function themeColor(token: string): number {
    const theme = document.documentElement.getAttribute('data-theme');
    if (theme !== cachedTheme) {
        cache.clear();
        cachedTheme = theme;
    }

    const cached = cache.get(token);
    if (cached !== undefined) return cached;

    const raw = getComputedStyle(document.documentElement).getPropertyValue(token).trim();
    const [r, g, b] = raw.split(/[\s,/]+/).map(Number);

    if (!raw || [r, g, b].some(channel => !Number.isFinite(channel))) {
        return FALLBACK_COLORS[token] ?? 0x808080;
    }

    const color = (r << 16) | (g << 8) | b;
    cache.set(token, color);
    return color;
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

/** Build a GridHelper in the current theme. */
export function themedGrid(size: number, divisions: number): THREE.GridHelper {
    return new THREE.GridHelper(
        size,
        divisions,
        themeColor('--c-scene-grid-center'),
        themeColor('--c-scene-grid'),
    );
}

/**
 * Replace a grid with one in the current theme, keeping its position.
 * Grids bake their colours into geometry, so they cannot be recoloured
 * in place the way materials can.
 */
export function swapGrid(
    scene: THREE.Scene,
    current: THREE.GridHelper | null,
    size: number,
    divisions: number,
): THREE.GridHelper {
    const position = current?.position.clone();

    if (current) {
        scene.remove(current);
        current.geometry.dispose();
        const material = current.material;
        (Array.isArray(material) ? material : [material]).forEach(entry => entry.dispose());
    }

    const grid = themedGrid(size, divisions);
    if (position) grid.position.copy(position);
    scene.add(grid);

    return grid;
}

/** A component that repaints itself when the palette changes. */
export interface ThemeAware {
    applyTheme(): void;
}

/** Subscribe to theme changes. Returns the unsubscribe function. */
export function onThemeChange(handler: () => void): () => void {
    window.addEventListener('themechange', handler);
    return () => window.removeEventListener('themechange', handler);
}

/** Wire a component's applyTheme to the theme event. Returns unsubscribe. */
export function bindTheme(target: ThemeAware): () => void {
    return onThemeChange(() => target.applyTheme());
}
