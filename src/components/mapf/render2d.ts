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
 *
 * Almost nothing here is recomputed per frame. The simulation advances about 12
 * times a second while the display runs at 60, so the board, the obstacles, the
 * goals, the palette and the edge list are all cached and rebuilt only when
 * something they depend on actually changes. What is left per frame is one
 * `drawImage` and one pass over the robots.
 */

import type { MapfEnv } from './env';

/**
 * Consecutive stationary steps before a robot is drawn as stuck. Robots idle by
 * choice all the time; three in a row is a jam, and colouring the first one
 * would make a healthy fleet flicker amber.
 */
export const STUCK_STEPS = 3;

interface Palette {
    background: string;
    grid: string;
    /** Outline around the playing surface, so it reads as a distinct object. */
    frame: string;
    /** Behind the board, where the canvas is not square. */
    surround: string;
    obstacle: string;
    goal: string;
    edge: string;
    edgeFocus: string;
    radio: string;
    moving: string;
    arrived: string;
    stuck: string;
}

export class GridRenderer {
    private readonly canvas: HTMLCanvasElement;
    private readonly ctx: CanvasRenderingContext2D | null;
    private width = 0;
    private height = 0;
    private dpr = 1;

    /** Background, grid, obstacles and goals — everything fixed for an episode. */
    private readonly backdrop: HTMLCanvasElement;
    private backdropKey = '';

    private palette: Palette | null = null;
    private paletteTheme: string | null = null;

    /** Flat pairs of robot indices, rebuilt when the environment steps. */
    private edges = new Int32Array(0);
    private edgeCount = 0;
    private edgeVersion = -1;

    private xs = new Float32Array(0);
    private ys = new Float32Array(0);

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d', { alpha: false });
        this.backdrop = document.createElement('canvas');
        this.resize();
    }

    /** Match the backing store to the CSS box and the device pixel ratio. */
    resize(): void {
        const rect = this.canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;

        this.dpr = window.devicePixelRatio || 1;
        this.width = rect.width;
        this.height = rect.height;
        this.canvas.width = Math.round(rect.width * this.dpr);
        this.canvas.height = Math.round(rect.height * this.dpr);
        this.ctx?.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
        this.backdropKey = '';
    }

    /** Force the backdrop to be rebuilt — obstacles or goals have changed. */
    invalidate(): void {
        this.backdropKey = '';
        this.edgeVersion = -1;
    }

    /**
     * Nothing to release: Canvas2D holds no GPU objects of its own.
     *
     * Deliberately does not zero the backdrop. A renderer can be disposed while
     * the animation frame is mid-flight, and a stray draw against a 0x0 source
     * throws where it should simply be ignored.
     */
    dispose(): void {
        this.backdropKey = '';
    }

    /**
     * Canvas cannot read CSS variables, so the tokens are resolved here. Reading
     * a custom property forces style resolution, which is far too expensive to
     * do per frame, so the result is cached and keyed on the theme itself rather
     * than invalidated by an event listener — it cannot go stale.
     */
    private colors(): Palette {
        const theme = document.documentElement.getAttribute('data-theme');
        if (this.palette && theme === this.paletteTheme) return this.palette;

        const styles = getComputedStyle(document.documentElement);
        const token = (name: string, alpha = 1) => {
            const channels = styles.getPropertyValue(name).trim();
            if (!channels) return '#888888';
            return alpha === 1 ? `rgb(${channels})` : `rgb(${channels} / ${alpha})`;
        };

        this.palette = {
            // The board is a surface in its own right, not more page. It takes
            // the page background while the panel around it is the surface
            // colour, so the two read as different things, and the grid is drawn
            // in the accent rather than the border colour — a border-coloured
            // grid on a border-coloured edge is just more background.
            background: token('--c-bg'),
            grid: token('--c-accent', 0.16),
            frame: token('--c-border'),
            surround: token('--c-surface'),
            obstacle: token('--c-gray-400', 0.55),
            goal: token('--c-accent-2', 0.55),
            edge: token('--c-accent', 0.45),
            edgeFocus: token('--c-accent'),
            radio: token('--c-accent', 0.5),
            moving: token('--c-accent'),
            arrived: token('--c-green'),
            stuck: token('--c-amber'),
        };
        this.paletteTheme = theme;
        return this.palette;
    }

    /**
     * @param alpha how far between the previous and current cell to draw,
     *   in [0, 1]. Robots move on a grid but sliding them between cells reads
     *   as motion rather than as a slideshow.
     * @param focus which robot's links to draw; negative draws all of them.
     */
    draw(env: MapfEnv, alpha: number, focus: number): void {
        const ctx = this.ctx;
        if (!ctx || this.width === 0) return;

        const colors = this.colors();
        const board = env.board;
        const size = Math.min(this.width, this.height);
        const cell = size / board;
        const offsetX = (this.width - size) / 2;
        const offsetY = (this.height - size) / 2;

        this.buildBackdrop(env, colors, size, cell, offsetX, offsetY);
        ctx.drawImage(this.backdrop, 0, 0, this.width, this.height);

        const n = env.numAgents;
        if (this.xs.length < n) {
            this.xs = new Float32Array(n);
            this.ys = new Float32Array(n);
        }

        // Board y grows upward; canvas y grows downward.
        const baseX = offsetX + 0.5 * cell;
        const baseY = offsetY + (board - 0.5) * cell;
        for (let i = 0; i < n; i++) {
            this.xs[i] = baseX + (env.prevX[i] + (env.posX[i] - env.prevX[i]) * alpha) * cell;
            this.ys[i] = baseY - (env.prevY[i] + (env.posY[i] - env.prevY[i]) * alpha) * cell;
        }

        const single = focus >= 0 && focus < n;
        if (single) {
            // The radio disc, so the link rule is visible rather than implied:
            // an edge to every robot inside this circle.
            ctx.strokeStyle = colors.radio;
            ctx.setLineDash([3, 3]);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(this.xs[focus], this.ys[focus], env.sensingRange * cell, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        this.buildEdges(env);
        ctx.strokeStyle = single ? colors.edgeFocus : colors.edge;
        ctx.lineWidth = Math.max(single ? 1 : 0.4, cell * (single ? 0.1 : 0.06));
        ctx.beginPath();
        for (let e = 0; e < this.edgeCount; e++) {
            const i = this.edges[e * 2];
            const j = this.edges[e * 2 + 1];
            if (single && i !== focus && j !== focus) continue;
            ctx.moveTo(this.xs[i], this.ys[i]);
            ctx.lineTo(this.xs[j], this.ys[j]);
        }
        ctx.stroke();

        // Three fills rather than 3N, and drawn arrived-first so the robots that
        // still matter sit on top.
        //
        // Success is all-or-nothing: an episode is won only when *every* robot
        // is home, so with 96 of 100 arrived the outcome rests on the four that
        // have not. Those four are impossible to pick out of a hundred identical
        // dots, so arrived robots are drawn small and faded and the rest stay
        // full strength. What is left to do is then what you see.
        // Robots are the subject, so they get the strongest mark on the board:
        // a floor of two pixels so a large fleet does not dissolve into specks,
        // and a ring in the board colour so neighbouring robots stay countable
        // where they cluster.
        const radius = Math.max(2, cell * 0.44);
        const twoPi = Math.PI * 2;
        ctx.lineWidth = Math.max(0.5, cell * 0.09);
        ctx.strokeStyle = colors.background;

        for (let pass = 0; pass < 3; pass++) {
            const done = pass === 0;
            ctx.fillStyle = done ? colors.arrived : pass === 1 ? colors.moving : colors.stuck;
            ctx.globalAlpha = done ? 0.4 : 1;
            const r = done ? radius * 0.5 : radius;
            ctx.beginPath();
            for (let i = 0; i < n; i++) {
                const arrived = env.posX[i] === env.goalX[i] && env.posY[i] === env.goalY[i];
                const group = arrived ? 0 : env.stalled[i] >= STUCK_STEPS ? 2 : 1;
                if (group !== pass) continue;
                ctx.moveTo(this.xs[i] + r, this.ys[i]);
                ctx.arc(this.xs[i], this.ys[i], r, 0, twoPi);
            }
            ctx.fill();
            // Only the active robots are outlined; the arrived ones are meant
            // to recede.
            if (!done && radius > 2.5) ctx.stroke();
        }
        ctx.globalAlpha = 1;
    }

    /**
     * The board, its obstacles and the goals, drawn once per episode.
     *
     * Redrawing a few thousand rectangles and a grid at 60 Hz is most of what a
     * naive version of this spends its time on, and none of it changes between
     * steps.
     */
    private buildBackdrop(
        env: MapfEnv,
        colors: Palette,
        size: number,
        cell: number,
        offsetX: number,
        offsetY: number
    ): void {
        const key = `${env.board}|${this.width}x${this.height}|${this.dpr}|${this.paletteTheme}|${env.instanceId}`;
        if (key === this.backdropKey) return;
        this.backdropKey = key;

        this.backdrop.width = Math.max(1, Math.round(this.width * this.dpr));
        this.backdrop.height = Math.max(1, Math.round(this.height * this.dpr));
        const ctx = this.backdrop.getContext('2d');
        if (!ctx) return;
        ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

        const board = env.board;
        // The canvas is rarely square, so paint the letterbox in the panel
        // colour first and the board on top of it. Leaving it unpainted on an
        // opaque context shows through as black.
        ctx.fillStyle = colors.surround;
        ctx.fillRect(0, 0, this.width, this.height);
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

        ctx.strokeStyle = colors.frame;
        ctx.lineWidth = 1;
        ctx.strokeRect(offsetX + 0.5, offsetY + 0.5, size - 1, size - 1);

        // Inset and faded. Filling the whole cell makes obstacles the heaviest
        // mark on the board — at a hundred robots they were three times the
        // visual weight of the robots, which are the actual subject.
        ctx.fillStyle = colors.obstacle;
        const inset = cell * 0.16;
        const block = cell - inset * 2;
        for (let y = 0; y < board; y++) {
            const top = offsetY + (board - 1 - y) * cell + inset;
            for (let x = 0; x < board; x++) {
                if (!env.obstacles[y * board + x]) continue;
                ctx.fillRect(offsetX + x * cell + inset, top, block, block);
            }
        }

        // Every goal, including those already reached: a robot's circle is twice
        // the marker's radius, so an occupied goal is hidden underneath it.
        // Rings, not dots: a goal at robot size would be mistaken for a robot,
        // and at the old size it was invisible.
        const marker = Math.max(1.6, cell * 0.34);
        const twoPi = Math.PI * 2;
        ctx.strokeStyle = colors.goal;
        ctx.lineWidth = Math.max(0.8, cell * 0.1);
        ctx.beginPath();
        for (let i = 0; i < env.numAgents; i++) {
            const cx = offsetX + (env.goalX[i] + 0.5) * cell;
            const cy = offsetY + (board - 0.5 - env.goalY[i]) * cell;
            ctx.moveTo(cx + marker, cy);
            ctx.arc(cx, cy, marker, 0, twoPi);
        }
        ctx.stroke();
    }

    /**
     * Collect the undirected edge list once per simulation step.
     *
     * Scanning the N by N adjacency every frame is the one part of drawing that
     * grows quadratically; at 200 robots it is 40,000 reads that would otherwise
     * happen sixty times a second for a graph that changes twelve.
     */
    private buildEdges(env: MapfEnv): void {
        if (env.version === this.edgeVersion) return;
        this.edgeVersion = env.version;

        const n = env.numAgents;
        let count = 0;
        for (let i = 0; i < n; i++) {
            const row = i * n;
            for (let j = i + 1; j < n; j++) {
                // The graph is directed; draw a link if either end sees the
                // other, so an edge is never half-missing.
                if (env.adjacency[row + j] === 0 && env.adjacency[j * n + i] === 0) continue;
                count++;
            }
        }

        if (this.edges.length < count * 2) this.edges = new Int32Array(count * 2 + 64);

        let at = 0;
        for (let i = 0; i < n; i++) {
            const row = i * n;
            for (let j = i + 1; j < n; j++) {
                if (env.adjacency[row + j] === 0 && env.adjacency[j * n + i] === 0) continue;
                this.edges[at++] = i;
                this.edges[at++] = j;
            }
        }
        this.edgeCount = count;
    }
}
