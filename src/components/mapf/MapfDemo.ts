/**
 * MapfDemo — a hundred robots, no plan, five cells of vision each.
 *
 * Every robot sees a 5×5 patch and a bearing to its goal, exchanges one feature
 * vector with every robot inside its radio range, and picks a move. The weights
 * are shared and the fleet size appears nowhere in them, which is why ten
 * robots' worth of training drives two hundred.
 *
 * This was a side-by-side against a non-communicating policy. That comparison
 * turned out to be mostly measuring how little the baseline had been trained
 * on: retrained on the same data it goes from 0.21 to 0.79 at a hundred robots,
 * against 0.94 here. A race that flattering to one side is not worth running,
 * and one board at twice the size shows the mesh far better than two did.
 *
 * The temperature slider is the part worth playing with. At zero the policy
 * takes its argmax, and since the environment is deterministic, two robots that
 * want the same cell collide, revert, and do it again forever — the fleet
 * freezes and no amount of waiting helps. A little noise breaks the tie. Too
 * much drowns out the policy. Both failure modes are one drag apart.
 */

import { VisibilityManager } from '../../utils/VisibilityManager';
import { MapfEnv, boardForAgents, sampleInstance } from './env';
import { Policy } from './policy';
import { FleetRng, Rng } from './random';
import { GridRenderer, STUCK_STEPS } from './render2d';
import { loadAssets } from './weights';
import type { BoardRenderer, DemoManifest, Instance, Model } from './types';

export interface MapfDemoConfig {
    containerId: string;
    modelPath: string;
    defaultAgents?: number;
    defaultTemperature?: number;
    /**
     * Fill the container's height instead of sizing the boards by their aspect
     * ratio, and drop the chrome that duplicates what the host already shows.
     * For the home page hero, which is a fixed-height box with its own
     * "how it works" panel.
     */
    compact?: boolean;
}

/** The article's fleet sizes, plus headroom past where the story is told. */
const FLEET_SIZES = [10, 20, 40, 60, 100, 150, 200];

/**
 * Policy steps per second. Robots slide between cells at the display's rate, so
 * this is the simulation rate, not the frame rate. Fast enough that a 40-robot
 * episode resolves in a few seconds, slow enough to watch a jam form.
 */
const STEPS_PER_SECOND = 12;

/**
 * Floor for the adaptive rate. A big fleet is allowed to think more slowly, but
 * not so slowly that it stops looking like motion.
 */
const MIN_STEPS_PER_SECOND = 5;

/**
 * How much of the wall clock the simulation may consume.
 *
 * A fleet of 200 costs about 26 ms per panel per step; running both at the full
 * rate spends two thirds of every second thinking, and what is left does not
 * draw a smooth frame. Backing the rate off keeps the animation fluid, which is
 * what a viewer actually perceives — nobody counts simulation steps.
 */
const STEP_BUDGET = 0.4;

/** How long to hold a finished episode before starting the next. */
const EPISODE_PAUSE_MS = 900;

/**
 * Stats are read by a human, not an instrument. Rewriting them every frame is
 * a DOM parse 120 times a second for text that nobody can follow that fast.
 */
const STATS_INTERVAL_MS = 120;

/** Episode outcomes kept in the win/loss strip under each board. */
const HISTORY_LENGTH = 18;

/**
 * Stop on its own after this long without a click or a drag.
 *
 * This runs two neural networks in a loop, and on the home page it starts by
 * itself for anyone who lands there. Left alone it would keep a phone's CPU busy
 * for as long as the tab is open. Two minutes is a dozen episodes — long past
 * the point where anyone is still watching — and the Resume button is right
 * there.
 */
const IDLE_LIMIT_MS = 120_000;

interface Panel {
    key: string;
    title: string;
    subtitle: string;
    model: Model;
    canvas: HTMLCanvasElement;
    renderer: BoardRenderer;
    stats: HTMLElement;
    score: HTMLElement;
    history: HTMLElement;
    /** One entry per finished episode: did every robot get home? */
    outcomes: boolean[];
    drawnOutcomes: number;
    env: MapfEnv;
    policy: Policy;
    rng: FleetRng;
    done: boolean;
    solved: number;
    episodes: number;
}

export class MapfDemo {
    private readonly container: HTMLElement;
    private readonly config: MapfDemoConfig;
    private readonly compact: boolean;

    private manifest: DemoManifest | null = null;
    private panels: Panel[] = [];
    private instance: Instance | null = null;
    private horizon = 0;
    private agents: number;
    private temperature: number;
    private obstacleFraction = 0.02;
    private sensingRange = 4;
    /** The third dimension is decoration; the simulation is flat either way. */
    private view: '2d' | '3d' = '2d';
    private switchingView = false;
    /** Robot whose links are drawn; negative shows every link. */
    private focus = -1;
    private running = true;
    /** Scrolled into view and the tab in the foreground. */
    private visible = true;
    private lastInteraction = 0;
    private destroyed = false;

    private frame = 0;
    private lastFrameTime = 0;
    private accumulator = 0;
    private finishedAt = 0;
    private statsAt = 0;
    private stepCostMs = 0;
    private pending: Panel[] = [];
    private episodeSeed = 1;
    private instanceRng = new Rng(20240607);

    private visibility: VisibilityManager | null = null;
    private resizeObserver: ResizeObserver | null = null;

    constructor(config: MapfDemoConfig) {
        const container = document.getElementById(config.containerId);
        if (!container) throw new Error(`Container #${config.containerId} not found`);

        this.container = container;
        this.config = config;
        this.compact = config.compact ?? false;
        this.agents = config.defaultAgents ?? 40;
        this.temperature = config.defaultTemperature ?? 3;

        this.renderLoading();
        void this.init();
    }

    private async init(): Promise<void> {
        let assets;
        try {
            assets = await loadAssets(this.config.modelPath);
        } catch (error) {
            this.renderError(error instanceof Error ? error.message : String(error));
            return;
        }
        if (this.destroyed) return;

        this.manifest = assets.manifest;
        this.agents = this.config.defaultAgents ?? assets.manifest.defaults.agents;
        this.temperature = this.config.defaultTemperature ?? assets.manifest.defaults.temperature;
        this.obstacleFraction = assets.manifest.defaults.obstacle_fraction;
        this.sensingRange = assets.manifest.sensing_range;

        this.renderShell(assets.manifest, assets.models);
        this.buildPanels(assets.models);
        this.newEpisode();
        this.bindControls();

        this.resizeObserver = new ResizeObserver(() => {
            for (const panel of this.panels) panel.renderer.resize();
        });
        this.resizeObserver.observe(this.container);

        // Scrolled out of view or the tab in the background: stop entirely.
        this.visibility = new VisibilityManager(this.container, paused => {
            this.visible = !paused;
            // Coming back after a while should not immediately trip the idle
            // timeout on a demo the reader has just scrolled to.
            if (!paused) this.lastInteraction = performance.now();
            this.syncLoop();
        });

        // Someone who has asked for less animation should not be handed two
        // fleets of robots moving on arrival.
        const calm = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
        this.running = !calm;
        if (calm) {
            const play = this.container.querySelector<HTMLButtonElement>('#mapf-play');
            if (play) play.textContent = 'Play';
            this.render(1);
        }

        this.lastInteraction = performance.now();
        this.syncLoop();
    }

    // ── setup ────────────────────────────────────────────────────────────

    private buildPanels(models: Record<string, Model>): void {
        // One board. The demo used to race a non-communicating policy alongside
        // this one, but that comparison was largely measuring how little the
        // baseline had been trained on: retrained on the same data it goes from
        // 0.21 to 0.79 at a hundred robots, against 0.94 here. A side-by-side
        // that flattering to one side is not worth showing, and a single board
        // twice the size shows the thing itself far better.
        const descriptions: Array<[string, string, string]> = [
            ['comm', 'Communication', 'one message to every robot in range'],
        ];

        this.panels = descriptions
            .filter(([key]) => models[key])
            .map(([key, title, subtitle]) => {
                const canvas = this.query<HTMLCanvasElement>(`#mapf-canvas-${key}`);
                return {
                    key,
                    title,
                    subtitle,
                    model: models[key],
                    canvas,
                    renderer: new GridRenderer(canvas),
                    stats: this.query<HTMLElement>(`#mapf-stats-${key}`),
                    score: this.query<HTMLElement>(`#mapf-score-${key}`),
                    history: this.query<HTMLElement>(`#mapf-history-${key}`),
                    outcomes: [],
                    drawnOutcomes: -1,
                    // Replaced immediately by newEpisode; typed non-null here.
                    env: null as unknown as MapfEnv,
                    policy: null as unknown as Policy,
                    rng: null as unknown as FleetRng,
                    done: false,
                    solved: 0,
                    episodes: 0,
                };
            });
    }

    /**
     * Start a fresh instance on every panel.
     *
     * Both panels get the same obstacles, starts, goals *and* the same per-robot
     * random streams, so a difference between them is the policy and nothing
     * else.
     */
    private newEpisode(): void {
        const manifest = this.manifest;
        if (!manifest) return;

        const board = boardForAgents(this.agents);
        this.instance = sampleInstance(
            this.instanceRng,
            this.agents,
            board,
            this.obstacleFraction
        );
        this.horizon = manifest.defaults.horizon_factor * board;
        this.episodeSeed = this.instanceRng.nextUint32();

        const envConfig = {
            board,
            sensingRange: this.sensingRange,
            pad: manifest.pad,
            maxNeighbours: manifest.max_neighbours,
        };

        for (const panel of this.panels) {
            // Reuse everything when only the instance changed. The environment
            // and the policy own several N-by-N buffers between them; rebuilding
            // them every few seconds is a collection you can see.
            const reusable = panel.env
                && panel.env.board === board
                && panel.env.numAgents === this.agents;

            if (reusable) {
                panel.env.sensingRange = this.sensingRange;
                panel.env.load(this.instance);
                panel.rng.reseed(this.episodeSeed);
            } else {
                panel.env = new MapfEnv(envConfig, this.instance);
                panel.policy = new Policy(panel.model, this.agents, manifest.fov[0]);
                panel.rng = new FleetRng(this.agents, this.episodeSeed);
                panel.renderer.resize();
            }
            panel.done = false;
        }
        this.finishedAt = 0;
        this.accumulator = 0;
        this.pending.length = 0;
        if (this.focus >= 0) this.focus = this.pickFocus();
        // A fleet-size change makes the old measurement meaningless.
        this.stepCostMs = 0;
    }

    /** Rebuild after a fleet-size change; the tallies no longer apply. */
    private resetStatistics(): void {
        for (const panel of this.panels) {
            panel.solved = 0;
            panel.episodes = 0;
            panel.outcomes.length = 0;
            panel.drawnOutcomes = -1;
        }
    }

    // ── loop ─────────────────────────────────────────────────────────────

    /**
     * Start or stop the animation frame to match intent and visibility.
     *
     * A paused demo cancels the loop outright rather than idling inside it. The
     * naive version still wakes sixty times a second to redraw a frame that has
     * not changed, which costs a phone exactly as much as running does.
     */
    private syncLoop(): void {
        const shouldRun = this.running && this.visible && !this.destroyed;
        if (shouldRun && this.frame === 0) {
            this.lastFrameTime = performance.now();
            this.frame = requestAnimationFrame(this.loop);
        } else if (!shouldRun && this.frame !== 0) {
            cancelAnimationFrame(this.frame);
            this.frame = 0;
        }
    }

    /**
     * Which robot to follow when showing a single neighbourhood.
     *
     * The best-connected one near the middle of the board: a robot on the rim
     * with one link demonstrates nothing, and one in the thick of it shows the
     * rule — an edge to everything inside the radio disc.
     */
    private pickFocus(): number {
        const panel = this.panels[0];
        if (!panel) return -1;
        const env = panel.env;
        const n = env.numAgents;
        const middle = env.board / 2;

        let best = -1;
        let bestScore = -Infinity;
        for (let i = 0; i < n; i++) {
            let degree = 0;
            const row = i * n;
            for (let j = 0; j < n; j++) degree += env.adjacency[row + j];
            const dx = env.posX[i] - middle;
            const dy = env.posY[i] - middle;
            const score = degree - Math.sqrt(dx * dx + dy * dy) / env.board;
            if (score > bestScore) { bestScore = score; best = i; }
        }
        return best;
    }

    /** Record that someone is still here, deferring the idle stop. */
    private touch(): void {
        this.lastInteraction = performance.now();
    }

    /**
     * Swap the 2D and 3D views.
     *
     * A canvas can only ever hold one kind of context, so the element itself has
     * to be replaced — asking for WebGL on a canvas that has handed out a 2D
     * context returns null. The simulation is untouched: the same environment
     * and the same policy keep running, and only the thing drawing them changes.
     */
    private async setView(view: '2d' | '3d'): Promise<void> {
        if (this.switchingView || view === this.view || this.destroyed) return;
        this.switchingView = true;

        const button = this.container.querySelector<HTMLButtonElement>('#mapf-view');
        if (button) button.textContent = view === '3d' ? '…' : '2D';

        try {
            for (const panel of this.panels) {
                const fresh = panel.canvas.cloneNode(false) as HTMLCanvasElement;
                panel.canvas.replaceWith(fresh);
                panel.renderer.dispose();
                panel.canvas = fresh;
                panel.renderer = view === '3d'
                    ? await (await import('./render3d')).createRenderer3D(fresh)
                    : new GridRenderer(fresh);
                panel.renderer.resize();
            }
            this.view = view;
        } catch (error) {
            // three.js failed to load, or WebGL is unavailable. Stay where we are.
            console.error('MAPF demo: could not switch view', error);
        }

        if (button) button.textContent = this.view === '3d' ? '2D' : '3D';
        this.switchingView = false;
        this.render(1);
    }

    private setRunning(running: boolean, interacted = true): void {
        if (interacted) this.lastInteraction = performance.now();
        if (this.running === running) return;
        this.running = running;

        const play = this.container.querySelector<HTMLButtonElement>('#mapf-play');
        if (play) play.textContent = running ? 'Pause' : 'Resume';
        this.syncLoop();
        if (!running) this.render(1);
    }

    private render(alpha: number): void {
        // Swapping views replaces the canvas and its renderer across an await;
        // drawing into the half-swapped pair is a guaranteed error.
        if (this.switchingView) return;
        for (const panel of this.panels) {
            panel.renderer.draw(panel.env, alpha, this.focus);
        }
        this.updateStats();
    }

    private loop = (): void => {
        if (this.destroyed || !this.running || !this.visible) {
            this.frame = 0;
            return;
        }
        this.frame = requestAnimationFrame(this.loop);

        const now = performance.now();
        const elapsed = Math.min(now - this.lastFrameTime, 250);
        this.lastFrameTime = now;

        if (now - this.lastInteraction >= IDLE_LIMIT_MS) {
            this.setRunning(false, false);
            return;
        }

        const interval = this.stepInterval();
        this.accumulator += elapsed;
        if (this.accumulator >= interval) {
            this.accumulator -= interval;
            if (this.accumulator > interval) this.accumulator = interval;
            this.beginTick(now);
        }
        // At most one fleet per frame. Stepping both in the same frame is a
        // visible stutter once they are large, and at twelve steps a second
        // there are several frames to spread the work over.
        const next = this.pending.pop();
        if (next) this.stepPanel(next);

        if (!this.switchingView) {
            for (const panel of this.panels) {
                panel.renderer.draw(panel.env, Math.min(1, this.accumulator / interval), this.focus);
            }
        }

        if (now - this.statsAt >= STATS_INTERVAL_MS) {
            this.statsAt = now;
            this.updateStats();
        }
    };

    private beginTick(now: number): void {
        // Still working through the last tick: drop this one rather than let
        // the queue grow. The simulation slows down; it does not spiral.
        if (this.pending.length > 0) return;

        if (this.panels.every(panel => panel.done)) {
            // Hold the finished boards briefly so the outcome is readable.
            if (this.finishedAt === 0) this.finishedAt = now;
            if (now - this.finishedAt >= EPISODE_PAUSE_MS) this.newEpisode();
            return;
        }

        for (const panel of this.panels) {
            if (!panel.done) this.pending.push(panel);
        }
    }

    /**
     * How long to wait between simulation steps, given what a step costs here.
     *
     * Measured rather than assumed: the same fleet size is a different amount of
     * work on a laptop and on a phone.
     */
    private stepInterval(): number {
        const fastest = 1000 / STEPS_PER_SECOND;
        const slowest = 1000 / MIN_STEPS_PER_SECOND;
        if (this.stepCostMs <= 0) return fastest;
        const affordable = (this.stepCostMs * this.panels.length) / STEP_BUDGET;
        return Math.min(slowest, Math.max(fastest, affordable));
    }

    private stepPanel(panel: Panel): void {
        if (panel.done) return;

        const started = performance.now();
        panel.policy.forward(panel.env.fov, panel.env.adjacency);
        const actions = panel.policy.selectActions(this.temperature, panel.rng);
        panel.env.step(actions);
        // Smoothed, so one slow frame does not swing the rate.
        const cost = performance.now() - started;
        this.stepCostMs = this.stepCostMs === 0 ? cost : this.stepCostMs * 0.8 + cost * 0.2;

        if (panel.env.allAtGoal()) {
            panel.done = true;
            panel.solved++;
            panel.episodes++;
            panel.outcomes.push(true);
        } else if (panel.env.time >= this.horizon) {
            panel.done = true;
            panel.episodes++;
            panel.outcomes.push(false);
        }
    }

    private updateStats(): void {
        for (const panel of this.panels) {
            const env = panel.env;
            const home = env.numAtGoal();
            const stuck = env.numStuck(STUCK_STEPS);

            panel.score.textContent = panel.episodes > 0
                ? (panel.solved / panel.episodes).toFixed(2)
                : '—';

            // A row of green and red squares says which policy is winning
            // faster than any number does, and each square is one real episode.
            if (panel.drawnOutcomes !== panel.outcomes.length) {
                panel.drawnOutcomes = panel.outcomes.length;
                const recent = panel.outcomes.slice(-HISTORY_LENGTH);
                panel.history.innerHTML = recent
                    .map(won => `<span class="inline-block w-[6px] h-[10px] rounded-[1px]" style="background:rgb(var(--c-${won ? 'green' : 'red'}))"></span>`)
                    .join('');
            }

            // One line: in compact mode the host's reset button floats over this
            // corner, and a wrapped stats line runs into it.
            const parts = [
                `<span class="${home === env.numAgents ? 'text-[rgb(var(--c-green))]' : ''}">${home}/${env.numAgents} home</span>`,
            ];
            if (stuck > 0) {
                parts.push(`<span class="text-[rgb(var(--c-amber))]">${stuck} stuck</span>`);
            }
            parts.push(`${env.time}/${this.horizon}`);
            if (panel.key === 'comm' && !this.compact) {
                parts.push(`${env.meanNeighbours().toFixed(1)} links`);
            }
            panel.stats.innerHTML = parts.join('<span class="opacity-40"> · </span>');
        }
    }

    // ── controls ─────────────────────────────────────────────────────────

    private bindControls(): void {
        const fleet = this.query<HTMLInputElement>('#mapf-fleet');
        const fleetValue = this.query<HTMLElement>('#mapf-fleet-value');
        fleet.value = String(Math.max(0, FLEET_SIZES.indexOf(this.agents)));
        fleet.addEventListener('input', () => {
            this.agents = FLEET_SIZES[Number(fleet.value)];
            fleetValue.textContent = `${this.agents} robots · ${boardForAgents(this.agents)}²`;
            this.query<HTMLElement>('#mapf-obstacles-value').textContent = this.describeObstacles();
            this.touch();
            this.resetStatistics();
            this.newEpisode();
        });

        const obstacles = this.query<HTMLInputElement>('#mapf-obstacles');
        const obstaclesValue = this.query<HTMLElement>('#mapf-obstacles-value');
        obstacles.value = String(Math.round(this.obstacleFraction * 100));
        obstacles.addEventListener('input', () => {
            this.obstacleFraction = Number(obstacles.value) / 100;
            obstaclesValue.textContent = this.describeObstacles();
            this.touch();
            this.resetStatistics();
            this.newEpisode();
        });

        const sensing = this.query<HTMLInputElement>('#mapf-sensing');
        const sensingValue = this.query<HTMLElement>('#mapf-sensing-value');
        sensing.value = String(this.sensingRange);
        sensing.addEventListener('input', () => {
            this.sensingRange = Number(sensing.value);
            sensingValue.textContent = this.describeSensing();
            this.touch();
            this.resetStatistics();
            this.newEpisode();
        });

        const temperature = this.query<HTMLInputElement>('#mapf-temperature');
        const temperatureValue = this.query<HTMLElement>('#mapf-temperature-value');
        temperature.value = String(this.temperature);
        temperature.addEventListener('input', () => {
            this.temperature = Number(temperature.value);
            temperatureValue.textContent = this.describeTemperature();
            this.touch();
            this.resetStatistics();
        });

        this.query<HTMLButtonElement>('#mapf-play').addEventListener('click', () => {
            this.setRunning(!this.running);
        });

        this.query<HTMLButtonElement>('#mapf-new').addEventListener('click', () => {
            this.newEpisode();
            this.setRunning(true);
        });

        const focusButton = this.query<HTMLButtonElement>('#mapf-focus');
        focusButton.addEventListener('click', () => {
            this.touch();
            this.focus = this.focus >= 0 ? -1 : this.pickFocus();
            focusButton.textContent = this.focus >= 0 ? 'All links' : '1 robot';
            this.render(1);
        });

        this.query<HTMLButtonElement>('#mapf-view').addEventListener('click', () => {
            this.touch();
            void this.setView(this.view === '2d' ? '3d' : '2d');
        });

        // Absent in compact mode, where the host page has its own explainer.
        const hood = this.container.querySelector<HTMLButtonElement>('#mapf-hood');
        const hoodPanel = this.container.querySelector<HTMLElement>('#mapf-hood-panel');
        if (hood && hoodPanel) {
            hood.addEventListener('click', () => {
                const hidden = hoodPanel.classList.toggle('hidden');
                hood.textContent = hidden ? 'Under the hood ▾' : 'Hide details ▴';
            });
        }
    }

    /**
     * Say honestly whether this density is one the policy has seen.
     *
     * The range comes from the manifest, derived at export from the dataset
     * directories the model was actually trained on. It used to be hardcoded to
     * "2% or it is out of distribution", which was true of the previous model
     * and became a lie the moment the curriculum grew to span 2-15%.
     */
    private describeObstacles(): string {
        const board = boardForAgents(this.agents);
        const count = Math.round(board * board * this.obstacleFraction);
        const percent = Math.round(this.obstacleFraction * 100);
        const range = this.manifest?.defaults.trained_obstacles;

        let note: string;
        if (percent === 0) note = 'open board';
        else if (!range) note = `${count} cells`;
        else if (this.obstacleFraction < range[0]) note = 'sparser than trained';
        else if (this.obstacleFraction > range[1]) note = 'denser than trained';
        else note = `within training ${Math.round(range[0] * 100)}–${Math.round(range[1] * 100)}%`;

        return `${percent}% · ${count} cells — ${note}`;
    }

    private describeSensing(): string {
        const r = this.sensingRange;
        const trained = this.manifest?.sensing_range ?? 8;
        const note = r <= 2 ? 'barely connected'
            : r < trained - 1 ? 'fewer neighbours than trained'
            : r <= trained + 1 ? 'as trained'
            : 'wider than trained';
        return `${r} cells — ${note}`;
    }

    private describeTemperature(): string {
        const t = this.temperature;
        if (t <= 0) return '0 — argmax, deadlocks forever';
        if (t < 1.5) return `${t.toFixed(2)} — barely breaks ties`;
        if (t <= 3.5) return `${t.toFixed(2)} — works`;
        return `${t.toFixed(2)} — noise drowns the policy`;
    }

    // ── markup ───────────────────────────────────────────────────────────

    private renderLoading(): void {
        this.container.innerHTML = `
            <div class="rounded-lg border border-[rgb(var(--c-border))] bg-[rgb(var(--c-surface))] p-8 text-center">
                <div class="text-sm text-[rgb(var(--c-gray-400))]">Loading policy…</div>
            </div>`;
    }

    private renderError(message: string): void {
        this.container.innerHTML = `
            <div class="rounded-lg border border-[rgb(var(--c-border))] bg-[rgb(var(--c-surface))] p-6">
                <div class="text-sm font-medium text-[rgb(var(--c-red))]">The demo could not start</div>
                <div class="mt-1 text-xs text-[rgb(var(--c-gray-400))] font-mono">${escapeHtml(message)}</div>
            </div>`;
    }

    private renderShell(manifest: DemoManifest, models: Record<string, Model>): void {
        const comm = models.comm;
        const board = boardForAgents(this.agents);

        // In compact mode the boards take whatever height is left rather than
        // setting it, so the whole thing fits a fixed-height host.
        const c = this.compact;

        const boardCanvas = (key: string) => `
            <canvas id="mapf-canvas-${key}" class="w-full block ${c ? 'h-full' : 'aspect-square max-h-[62vh] mx-auto'}"></canvas>`;

        const readout = (key: string) => `
            <div class="flex items-center justify-between gap-3 ${c ? '' : 'mt-3'}">
                <div class="flex items-baseline gap-1.5 flex-shrink-0">
                    <span id="mapf-score-${key}" class="${c ? 'text-lg' : 'text-2xl'} font-bold leading-none tabular-nums">—</span>
                    <span class="text-[9px] uppercase tracking-wide text-[rgb(var(--c-gray-500))] truncate">${c ? 'all home' : 'every robot home'}</span>
                </div>
                <div id="mapf-history-${key}" class="flex gap-[2px] flex-wrap justify-end"></div>
            </div>
            <div id="mapf-stats-${key}" class="mt-0.5 ${c ? 'text-[9px] leading-tight' : 'text-[11px]'} font-mono text-[rgb(var(--c-gray-400))]"></div>`;

        // One fragment, two arrangements. The hero is a short, wide box: with a
        // single board, stacking four sliders above it leaves the board 114
        // pixels tall. Beside it they cost nothing.
        const hood = c ? '' : `
            <button id="mapf-hood" class="text-xs text-[rgb(var(--c-gray-500))] hover:text-[rgb(var(--c-accent))] transition-colors">
                Under the hood ▾
            </button>`;

        const hoodPanel = c ? '' : `
            <div id="mapf-hood-panel" class="hidden px-4 py-3 border-b border-[rgb(var(--c-border))] bg-[rgb(var(--c-bg))] text-xs text-[rgb(var(--c-gray-400))] space-y-1.5">
                <p>
                    Each robot sees a ${manifest.fov[0]}×${manifest.fov[1]} patch of the board and a marker
                    projected onto the patch edge pointing at its goal — never the whole map, never another
                    robot's plan. A three-layer CNN turns that into 64 numbers.
                </p>
                <p>
                    It then aggregates over the communication graph — an edge to <em>every</em> robot within
                    ${manifest.sensing_range} cells, ${comm.filterNumber} hops of
                    <span class="font-mono">D<sup>-½</sup>(A+I)D<sup>-½</sup></span>, and a linear head onto five
                    actions. The weights are shared by every robot and the fleet size appears nowhere in them,
                    which is the entire reason a policy trained on ${comm.trainedAgents} runs on ${FLEET_SIZES[FLEET_SIZES.length - 1]}.
                </p>
                <p>
                    ${comm.parameters.toLocaleString()} parameters, ${Math.round(comm.parameters * 4 / 1024)} KB,
                    running in this tab. Behaviour cloned from an optimal centralised planner that cannot itself
                    get past about 20 robots.
                </p>
            </div>`;

        // Without a key the board is a few hundred identical dots. Everything
        // the demo is saying is encoded in colour, so the key is not optional.
        const swatch = (style: string, label: string) =>
            `<span class="flex items-center gap-1 whitespace-nowrap">${style}${label}</span>`;
        const dot = (token: string, opacity = 1, size = 7) =>
            `<i class="inline-block rounded-full" style="width:${size}px;height:${size}px;background:rgb(var(--${token}));opacity:${opacity}"></i>`;

        const slider = (id: string, label: string, attrs: string, value: string) => `
            <label class="flex flex-col ${c ? 'gap-0.5' : 'gap-1'} ${c ? '' : 'min-w-[9rem] flex-1'}">
                <span class="text-[10px] uppercase tracking-wide leading-none text-[rgb(var(--c-gray-500))]">${label}</span>
                <input id="mapf-${id}" type="range" ${attrs} class="w-full accent-[rgb(var(--c-accent))]">
                <span id="mapf-${id}-value" class="text-[10px] leading-tight font-mono text-[rgb(var(--c-gray-400))]">${value}</span>
            </label>`;

        const controls = `
            ${slider('fleet', 'Fleet size', `min="0" max="${FLEET_SIZES.length - 1}" step="1"`,
                     `${this.agents} robots · ${board}²`)}
            ${slider('obstacles', 'Obstacles', 'min="0" max="15" step="1"', this.describeObstacles())}
            ${slider('sensing', 'Radio range', 'min="2" max="14" step="1"', this.describeSensing())}
            ${slider('temperature', 'Temperature', 'min="0" max="5" step="0.25"', this.describeTemperature())}
            <div class="flex items-center flex-wrap gap-1.5">
                <button id="mapf-play" class="px-2.5 py-1 text-xs rounded border border-[rgb(var(--c-border))] hover:border-[rgb(var(--c-accent))] hover:text-[rgb(var(--c-accent))] transition-colors">Pause</button>
                <button id="mapf-new" class="px-2.5 py-1 text-xs rounded border border-[rgb(var(--c-border))] hover:border-[rgb(var(--c-accent))] hover:text-[rgb(var(--c-accent))] transition-colors">New</button>
                <button id="mapf-view" class="px-2.5 py-1 text-xs rounded border border-[rgb(var(--c-border))] hover:border-[rgb(var(--c-accent))] hover:text-[rgb(var(--c-accent))] transition-colors">3D</button>
                <button id="mapf-focus" class="px-2.5 py-1 text-xs rounded border border-[rgb(var(--c-border))] hover:border-[rgb(var(--c-accent))] hover:text-[rgb(var(--c-accent))] transition-colors">1 robot</button>
            </div>`;

        const legend = `
            <div class="${c ? 'px-3 py-1' : 'px-4 py-2'} border-t border-[rgb(var(--c-border))] bg-[rgb(var(--c-bg))]
                        flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-[rgb(var(--c-gray-500))]">
                ${swatch(dot('c-accent'), 'moving')}
                ${swatch(dot('c-green', 0.45, 5), 'home')}
                ${swatch(dot('c-amber'), `stuck ${STUCK_STEPS}+ steps`)}
                ${swatch(`<i class="inline-block w-[7px] h-[7px]" style="background:rgb(var(--c-gray-400)/0.55)"></i>`, 'obstacle')}
                ${swatch(`<i class="inline-block w-[10px] h-[1px]" style="background:rgb(var(--c-accent)/0.7)"></i>`, 'message link')}
                ${swatch(dot('c-accent-2', 0.4, 5), 'goal')}
            </div>`;

        const footer = c ? '' : `
            <div class="px-4 py-2 border-t border-[rgb(var(--c-border))] bg-[rgb(var(--c-bg))] text-[11px] text-[rgb(var(--c-gray-500))]">
                No robot can see more than five cells around itself, and none of them has a plan. An episode
                counts only when <em>every</em> robot is home, so the last few stragglers decide it — which is
                why arrived robots fade out. Drag the temperature to zero to watch the fleet freeze solid.
            </div>`;

        this.container.innerHTML = `
        <div class="rounded-lg border border-[rgb(var(--c-border))] bg-[rgb(var(--c-surface))] overflow-hidden ${c ? 'h-full w-full flex flex-col' : 'shadow-xl'}">

            <div class="${c ? 'px-3 py-1.5' : 'px-4 py-2'} border-b border-[rgb(var(--c-border))] bg-[rgb(var(--c-bg))] flex items-center justify-between gap-3 flex-wrap flex-shrink-0">
                <div class="flex items-center gap-2 min-w-0">
                    <span class="text-[rgb(var(--c-accent))] text-sm">●</span>
                    <span class="text-sm font-medium truncate">Multi-robot path finding</span>
                    <span class="hidden sm:inline text-xs text-[rgb(var(--c-gray-500))] truncate">
                        trained on ${comm.trainedAgents} robots · runs on any number
                    </span>
                </div>
                ${hood}
            </div>
            ${hoodPanel}

            ${c ? `
            <div class="flex-1 min-h-0 min-w-0 flex">
                <!-- Beside the board, not above it. The host is only ~256px tall on
                     a phone, so stacking left the board 66px; a narrower column
                     keeps it square and usable. -->
                <div class="w-[6.75rem] sm:w-[11.5rem] flex-shrink-0 border-r border-[rgb(var(--c-border))]
                            flex flex-col min-h-0">
                    <!-- The score is the one number worth watching, so it sits
                         outside the scroll area and cannot be the thing that gets cut. -->
                    <div class="px-2 sm:px-3 pt-1.5 pb-1 border-b border-[rgb(var(--c-border))]">${readout('comm')}</div>
                    <div class="px-2 sm:px-3 py-1.5 flex flex-col gap-1 overflow-y-auto overflow-x-hidden min-h-0">${controls}</div>
                </div>
                <div class="flex-1 min-w-0 min-h-0 p-1.5 sm:p-2">${boardCanvas('comm')}</div>
            </div>
            ${legend}` : `
            <div class="px-4 py-3 gap-x-6 gap-y-3 border-b border-[rgb(var(--c-border))] flex flex-wrap items-end">
                ${controls}
            </div>
            <div class="p-3">
                ${boardCanvas('comm')}
                ${readout('comm')}
            </div>
            ${legend}
            ${footer}`}
        </div>`;
    }

    private query<T extends HTMLElement>(selector: string): T {
        const element = this.container.querySelector<T>(selector);
        if (!element) throw new Error(`MapfDemo: ${selector} missing from the shell`);
        return element;
    }

    // ── teardown ─────────────────────────────────────────────────────────

    destroy(): void {
        this.destroyed = true;
        cancelAnimationFrame(this.frame);
        this.frame = 0;
        this.visibility?.destroy();
        this.visibility = null;
        this.resizeObserver?.disconnect();
        this.resizeObserver = null;
        this.panels = [];
        this.container.innerHTML = '';
    }
}

function escapeHtml(value: string): string {
    const div = document.createElement('div');
    div.textContent = value;
    return div.innerHTML;
}
