/**
 * MapfDemo — two fleets, one instance, one difference.
 *
 * Both panels get identical starts, goals, obstacles and random streams. The
 * only thing that differs is whether the policy is allowed to exchange a feature
 * vector with its neighbours. Everything the demo claims is therefore visible
 * rather than asserted: at twenty robots the panels are indistinguishable, and
 * somewhere past a hundred the left one falls apart.
 *
 * The temperature slider is the other half. At zero the policy takes its argmax,
 * and since the environment is deterministic, two robots that want the same cell
 * will collide, revert, and do it again forever — the fleet freezes and no
 * amount of waiting helps. A little noise breaks the tie. Too much drowns out
 * the policy and the fleet wanders. Both failure modes are one drag apart.
 */

import { VisibilityManager } from '../../utils/VisibilityManager';
import { MapfEnv, boardForAgents, sampleInstance } from './env';
import { Policy } from './policy';
import { FleetRng, Rng } from './random';
import { GridRenderer, STUCK_STEPS } from './render2d';
import { loadAssets } from './weights';
import type { DemoManifest, Instance, Model } from './types';

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
    renderer: GridRenderer;
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
        const descriptions: Array<[string, string, string]> = [
            ['nocomm', 'No communication', 'each robot decides alone'],
            ['comm', 'Communication', 'one message to the four nearest'],
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

    /** Record that someone is still here, deferring the idle stop. */
    private touch(): void {
        this.lastInteraction = performance.now();
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
        for (const panel of this.panels) {
            panel.renderer.draw(panel.env, alpha, panel.key === 'comm');
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

        for (const panel of this.panels) {
            panel.renderer.draw(panel.env, Math.min(1, this.accumulator / interval), panel.key === 'comm');
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
     * The policy was cloned from demonstrations on boards with 2% obstacles, so
     * anything above that is out-of-distribution and says so.
     */
    private describeObstacles(): string {
        const board = boardForAgents(this.agents);
        const count = Math.round(board * board * this.obstacleFraction);
        const percent = Math.round(this.obstacleFraction * 100);
        const note = percent === 0 ? ' — open board'
            : percent <= 2 ? ' — as trained'
            : ' — beyond training';
        return `${percent}% · ${count} cells${note}`;
    }

    /**
     * Labels come from a sweep in the source repository (results/sensing_sweep.json),
     * at 100 robots: alpha is 0.54 at range 2, flat at 0.60 from 3 to 6, then
     * 0.54 at 8, 0.39 at 10 and 0.34 at 14 — worse than at range 2.
     *
     * Widening does not add neighbours, because only the four nearest are kept.
     * It swaps near ones for far ones, and a message from a robot too distant to
     * matter is worse than no message.
     */
    private describeSensing(): string {
        const r = this.sensingRange;
        const note = r <= 2 ? 'barely connected'
            : r === 4 ? 'as trained'
            : r <= 6 ? 'still fine'
            : r <= 9 ? 'links reaching too far'
            : 'far links crowd out near ones';
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
        const total = Object.values(models).reduce((sum, model) => sum + model.parameters, 0);

        // In compact mode the boards take whatever height is left rather than
        // setting it, so the whole thing fits a fixed-height host.
        const c = this.compact;

        const panel = (key: string, title: string, subtitle: string) => `
            <div class="bg-[rgb(var(--c-surface))] ${c ? 'p-2 flex flex-col min-h-0' : 'p-3'}">
                <div class="flex items-baseline justify-between gap-2 ${c ? 'mb-1' : 'mb-2'}">
                    <span class="text-[11px] font-semibold uppercase tracking-wide">${title}</span>
                    ${c ? '' : `<span class="text-[10px] text-[rgb(var(--c-gray-500))]">${subtitle}</span>`}
                </div>
                <canvas id="mapf-canvas-${key}" class="w-full block ${c ? 'flex-1 min-h-0' : 'aspect-square'}"></canvas>
                <div class="flex items-center justify-between gap-2 ${c ? 'mt-1' : 'mt-2'}">
                    <div class="flex items-baseline gap-1.5 flex-shrink-0">
                        <span id="mapf-score-${key}" class="${c ? 'text-base' : 'text-xl'} font-bold leading-none tabular-nums">—</span>
                        <span class="text-[9px] uppercase tracking-wide text-[rgb(var(--c-gray-500))]">all home</span>
                    </div>
                    <!-- The host floats a reset button over this corner in compact mode. -->
                    <div id="mapf-history-${key}" class="flex gap-[2px] flex-wrap justify-end ${c ? 'pr-8' : ''}"></div>
                </div>
                <div id="mapf-stats-${key}" class="${c ? 'mt-0.5 text-[10px]' : 'mt-1 text-[11px]'} font-mono text-[rgb(var(--c-gray-400))]"></div>
            </div>`;

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
                    The right-hand policy then aggregates over the communication graph: an edge to each of the
                    four nearest robots within ${manifest.sensing_range} cells, ${comm.filterNumber} hops of
                    <span class="font-mono">D<sup>-½</sup>(A+I)D<sup>-½</sup></span>, and a linear head onto five
                    actions. The weights are shared by every robot and the fleet size appears nowhere in them,
                    which is the entire reason a policy trained on ${comm.trainedAgents} runs on ${FLEET_SIZES[FLEET_SIZES.length - 1]}.
                </p>
                <p>
                    ${total.toLocaleString()} parameters across both policies, ${Math.round(total * 4 / 1024)} KB,
                    running in this tab. Behaviour cloned from an optimal centralised planner that cannot itself
                    get past about 20 robots.
                </p>
            </div>`;

        // Without a key the boards are a hundred identical dots. Everything the
        // demo is saying is encoded in colour, so the key is not optional.
        const swatch = (style: string, label: string) =>
            `<span class="flex items-center gap-1 whitespace-nowrap">${style}${label}</span>`;
        const dot = (token: string, opacity = 1, size = 7) =>
            `<i class="inline-block rounded-full" style="width:${size}px;height:${size}px;background:rgb(var(--${token}));opacity:${opacity}"></i>`;

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
                An episode counts only when <em>every</em> robot is home, so the last few stragglers decide it —
                which is why arrived robots fade out. Same obstacles, starts, goals and random draws on both
                sides; the only difference is the graph. Try 20 robots (identical), then 100.
            </div>`;

        this.container.innerHTML = `
        <div class="rounded-lg border border-[rgb(var(--c-border))] bg-[rgb(var(--c-surface))] overflow-hidden ${c ? 'h-full w-full flex flex-col' : 'shadow-xl'}">

            <div class="${c ? 'px-3 py-1.5' : 'px-4 py-2'} border-b border-[rgb(var(--c-border))] bg-[rgb(var(--c-bg))] flex items-center justify-between gap-3 flex-wrap flex-shrink-0">
                <div class="flex items-center gap-2 min-w-0">
                    <span class="text-[rgb(var(--c-accent))] text-sm">●</span>
                    <span class="text-sm font-medium">Multi-robot path finding</span>
                    <span class="text-xs text-[rgb(var(--c-gray-500))]">
                        trained on ${comm.trainedAgents} robots · runs on any number
                    </span>
                </div>
                ${hood}
            </div>
            ${hoodPanel}

            <div class="${c ? 'px-3 py-2 gap-x-4 gap-y-2' : 'px-4 py-3 gap-x-6 gap-y-3'} border-b border-[rgb(var(--c-border))] flex flex-wrap items-end flex-shrink-0">
                <label class="flex flex-col gap-1 min-w-[9rem] flex-1">
                    <span class="text-[10px] uppercase tracking-wide text-[rgb(var(--c-gray-500))]">Fleet size</span>
                    <input id="mapf-fleet" type="range" min="0" max="${FLEET_SIZES.length - 1}" step="1" class="w-full accent-[rgb(var(--c-accent))]">
                    <span id="mapf-fleet-value" class="text-[10px] font-mono text-[rgb(var(--c-gray-400))]">${this.agents} robots · ${board}²</span>
                </label>

                <label class="flex flex-col gap-1 min-w-[9rem] flex-1">
                    <span class="text-[10px] uppercase tracking-wide text-[rgb(var(--c-gray-500))]">Obstacles</span>
                    <input id="mapf-obstacles" type="range" min="0" max="15" step="1" class="w-full accent-[rgb(var(--c-accent))]">
                    <span id="mapf-obstacles-value" class="text-[10px] font-mono text-[rgb(var(--c-gray-400))]">${this.describeObstacles()}</span>
                </label>

                <label class="flex flex-col gap-1 min-w-[9rem] flex-1">
                    <span class="text-[10px] uppercase tracking-wide text-[rgb(var(--c-gray-500))]">Radio range</span>
                    <input id="mapf-sensing" type="range" min="2" max="14" step="1" class="w-full accent-[rgb(var(--c-accent))]">
                    <span id="mapf-sensing-value" class="text-[10px] font-mono text-[rgb(var(--c-gray-400))]">${this.describeSensing()}</span>
                </label>

                <label class="flex flex-col gap-1 min-w-[9rem] flex-1">
                    <span class="text-[10px] uppercase tracking-wide text-[rgb(var(--c-gray-500))]">Temperature</span>
                    <input id="mapf-temperature" type="range" min="0" max="5" step="0.25" class="w-full accent-[rgb(var(--c-accent))]">
                    <span id="mapf-temperature-value" class="text-[10px] font-mono text-[rgb(var(--c-gray-400))]">${this.describeTemperature()}</span>
                </label>

                <div class="flex items-center gap-2">
                    <button id="mapf-play" class="px-2.5 py-1 text-xs rounded border border-[rgb(var(--c-border))] hover:border-[rgb(var(--c-accent))] hover:text-[rgb(var(--c-accent))] transition-colors">Pause</button>
                    <button id="mapf-new" class="px-2.5 py-1 text-xs rounded border border-[rgb(var(--c-border))] hover:border-[rgb(var(--c-accent))] hover:text-[rgb(var(--c-accent))] transition-colors">New</button>
                </div>
            </div>

            <div class="grid ${c ? 'grid-cols-2 flex-1 min-h-0' : 'grid-cols-1 sm:grid-cols-2'} gap-px bg-[rgb(var(--c-border))]">
                ${panel('nocomm', 'No communication', 'each robot decides alone')}
                ${panel('comm', 'Communication', 'one message to the four nearest')}
            </div>
            ${legend}
            ${footer}
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
