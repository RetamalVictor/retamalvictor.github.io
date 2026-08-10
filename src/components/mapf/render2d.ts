/**
 * Canvas2D renderer for one panel.
 *
 * Colour encodes robot *state*, not identity. Two hundred arbitrary hues would
 * be decoration; moving / arrived / stuck is the thing the demo is about, and it
 * is what makes a deadlock legible at a glance — a screen of amber is a fleet
 * that has stopped, whatever the step counter says.
 *
 * Communication edges are drawn from the adjacency matrix the policy actually
 * consumed, so what you see is the graph the network saw.
 */

import type { MapfEnv } from './env';

interface Palette {
    background: string;
    grid: string;
    obstacle: string;
    goal: string;
    edge: string;
    moving: string;
    arrived: string;
    stuck: string;
}

export class GridRenderer {
    private readonly canvas: HTMLCanvasElement;
    private readonly ctx: CanvasRenderingContext2D | null;
    private width = 0;
    private height = 0;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.resize();
    }

    /** Match the backing store to the CSS box and the device pixel ratio. */
    resize(): void {
        const rect = this.canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;

        const dpr = window.devicePixelRatio || 1;
        this.width = rect.width;
        this.height = rect.height;
        this.canvas.width = Math.round(rect.width * dpr);
        this.canvas.height = Math.round(rect.height * dpr);
        this.ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    /**
     * Canvas cannot read CSS variables, so resolve the tokens per frame. The
     * lookup is cheap next to drawing a few hundred robots, and it means the
     * demo follows a theme change without being told about it.
     */
    private palette(): Palette {
        const styles = getComputedStyle(document.documentElement);
        const token = (name: string, alpha = 1) => {
            const channels = styles.getPropertyValue(name).trim();
            if (!channels) return '#888888';
            return alpha === 1 ? `rgb(${channels})` : `rgb(${channels} / ${alpha})`;
        };

        return {
            background: token('--c-bg'),
            grid: token('--c-border', 0.55),
            obstacle: token('--c-gray-400', 0.55),
            goal: token('--c-accent-2', 0.35),
            edge: token('--c-accent', 0.45),
            moving: token('--c-accent'),
            arrived: token('--c-green'),
            stuck: token('--c-amber'),
        };
    }

    /**
     * @param alpha how far between the previous and current cell to draw,
     *   in [0, 1]. Robots move on a grid but sliding them between cells reads
     *   as motion rather than as a slideshow.
     * @param showEdges draw the communication graph
     */
    draw(env: MapfEnv, alpha: number, showEdges: boolean): void {
        const ctx = this.ctx;
        if (!ctx || this.width === 0) return;

        const colors = this.palette();
        const board = env.board;
        const size = Math.min(this.width, this.height);
        const cell = size / board;
        const offsetX = (this.width - size) / 2;
        const offsetY = (this.height - size) / 2;

        // Board y grows upward; canvas y grows downward.
        const px = (x: number) => offsetX + (x + 0.5) * cell;
        const py = (y: number) => offsetY + (board - 0.5 - y) * cell;

        ctx.clearRect(0, 0, this.width, this.height);
        ctx.fillStyle = colors.background;
        ctx.fillRect(offsetX, offsetY, size, size);

        // Below about six pixels a cell, grid lines turn the board into a solid
        // block of ink and hide the robots.
        if (cell >= 6) {
            ctx.strokeStyle = colors.grid;
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            for (let i = 0; i <= board; i++) {
                const at = offsetX + i * cell;
                ctx.moveTo(at, offsetY);
                ctx.lineTo(at, offsetY + size);
                ctx.moveTo(offsetX, offsetY + i * cell);
                ctx.lineTo(offsetX + size, offsetY + i * cell);
            }
            ctx.stroke();
        }

        ctx.fillStyle = colors.obstacle;
        for (let y = 0; y < board; y++) {
            for (let x = 0; x < board; x++) {
                if (!env.obstacles[y * board + x]) continue;
                ctx.fillRect(offsetX + x * cell, offsetY + (board - 1 - y) * cell, cell, cell);
            }
        }

        const n = env.numAgents;
        const radius = Math.max(1.1, cell * 0.34);

        ctx.fillStyle = colors.goal;
        for (let i = 0; i < n; i++) {
            if (env.atGoal(i)) continue;
            const r = Math.max(0.8, cell * 0.16);
            ctx.beginPath();
            ctx.arc(px(env.goalX[i]), py(env.goalY[i]), r, 0, Math.PI * 2);
            ctx.fill();
        }

        // Interpolated positions are reused by the edges, so compute once.
        const xs = new Float32Array(n);
        const ys = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            xs[i] = px(env.prevX[i] + (env.posX[i] - env.prevX[i]) * alpha);
            ys[i] = py(env.prevY[i] + (env.posY[i] - env.prevY[i]) * alpha);
        }

        if (showEdges) {
            ctx.strokeStyle = colors.edge;
            ctx.lineWidth = Math.max(0.4, cell * 0.06);
            ctx.beginPath();
            for (let i = 0; i < n; i++) {
                const row = i * n;
                for (let j = i + 1; j < n; j++) {
                    // The graph is directed; draw a link if either end sees the
                    // other, so an edge is never half-missing.
                    if (env.adjacency[row + j] === 0 && env.adjacency[j * n + i] === 0) continue;
                    ctx.moveTo(xs[i], ys[i]);
                    ctx.lineTo(xs[j], ys[j]);
                }
            }
            ctx.stroke();
        }

        for (let i = 0; i < n; i++) {
            const arrived = env.atGoal(i);
            const stalled = !arrived && env.posX[i] === env.prevX[i] && env.posY[i] === env.prevY[i];
            ctx.fillStyle = arrived ? colors.arrived : stalled ? colors.stuck : colors.moving;
            ctx.beginPath();
            ctx.arc(xs[i], ys[i], radius, 0, Math.PI * 2);
            ctx.fill();
        }
    }
}
