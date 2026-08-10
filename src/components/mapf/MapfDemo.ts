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
import { GridRenderer } from './render2d';
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

/** How long to hold a finished episode before starting the next. */
const EPISODE_PAUSE_MS = 900;

interface Panel {
    key: string;
    title: string;
    subtitle: string;
    model: Model;
    canvas: HTMLCanvasElement;
    renderer: GridRenderer;
    stats: HTMLElement;
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
    private running = true;
    private destroyed = false;

    private frame = 0;
    private lastFrameTime = 0;
    private accumulator = 0;
    private finishedAt = 0;
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

        this.renderShell(assets.manifest, assets.models);
        this.buildPanels(assets.models);
        this.newEpisode();
        this.bindControls();

        this.resizeObserver = new ResizeObserver(() => {
            for (const panel of this.panels) panel.renderer.resize();
        });
        this.resizeObserver.observe(this.container);

        this.visibility = new VisibilityManager(this.container, paused => {
            if (paused) {
                cancelAnimationFrame(this.frame);
                this.frame = 0;
            } else if (this.running) {
                this.lastFrameTime = performance.now();
                this.loop();
            }
        });

        this.lastFrameTime = performance.now();
        this.loop();
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
            manifest.defaults.obstacle_fraction
        );
        this.horizon = manifest.defaults.horizon_factor * board;
        this.episodeSeed = this.instanceRng.nextUint32();

        const envConfig = {
            board,
            sensingRange: manifest.sensing_range,
            pad: manifest.pad,
            maxNeighbours: manifest.max_neighbours,
        };

        for (const panel of this.panels) {
            panel.env = new MapfEnv(envConfig, this.instance);
            panel.policy = new Policy(panel.model, this.agents, manifest.fov[0]);
            panel.rng = new FleetRng(this.agents, this.episodeSeed);
            panel.done = false;
            panel.renderer.resize();
        }
        this.finishedAt = 0;
        this.accumulator = 0;
    }

    /** Rebuild after a fleet-size change; the tallies no longer apply. */
    private resetStatistics(): void {
        for (const panel of this.panels) {
            panel.solved = 0;
            panel.episodes = 0;
        }
    }

    // ── loop ─────────────────────────────────────────────────────────────

    private loop = (): void => {
        if (this.destroyed) return;
        this.frame = requestAnimationFrame(this.loop);

        const now = performance.now();
        const elapsed = Math.min(now - this.lastFrameTime, 250);
        this.lastFrameTime = now;

        const interval = 1000 / STEPS_PER_SECOND;
        if (this.running) {
            this.accumulator += elapsed;
            // One step per frame at most. At large fleets a step costs more than
            // a frame, so catching up would spiral instead of drawing anything.
            if (this.accumulator >= interval) {
                this.accumulator -= interval;
                if (this.accumulator > interval) this.accumulator = interval;
                this.advance(now);
            }
        }

        const alpha = this.running ? Math.min(1, this.accumulator / interval) : 1;
        for (const panel of this.panels) {
            panel.renderer.draw(panel.env, alpha, panel.key === 'comm');
        }
        this.updateStats();
    };

    private advance(now: number): void {
        const everyoneDone = this.panels.every(panel => panel.done);
        if (everyoneDone) {
            // Hold the finished boards briefly so the outcome is readable.
            if (this.finishedAt === 0) this.finishedAt = now;
            if (now - this.finishedAt >= EPISODE_PAUSE_MS) this.newEpisode();
            return;
        }

        for (const panel of this.panels) {
            if (panel.done) continue;

            panel.policy.forward(panel.env.fov, panel.env.adjacency);
            const actions = panel.policy.selectActions(this.temperature, panel.rng);
            panel.env.step(actions);

            if (panel.env.allAtGoal()) {
                panel.done = true;
                panel.solved++;
                panel.episodes++;
            } else if (panel.env.time >= this.horizon) {
                panel.done = true;
                panel.episodes++;
            }
        }
    }

    private updateStats(): void {
        for (const panel of this.panels) {
            const env = panel.env;
            const home = env.numAtGoal();
            const alpha = panel.episodes > 0 ? panel.solved / panel.episodes : 0;
            const rate = panel.episodes > 0
                ? `${alpha.toFixed(2)} over ${panel.episodes}`
                : '—';

            // Kept to one line in compact mode: the host's reset button floats
            // over the bottom-right corner, and a wrapped stats line runs into it.
            const parts = [
                `<span class="${home === env.numAgents ? 'text-[rgb(var(--c-green))]' : ''}">${home}/${env.numAgents} home</span>`,
                this.compact ? `${env.time}/${this.horizon}` : `step ${env.time}/${this.horizon}`,
                this.compact ? `α ${rate.replace(' over ', '/')}` : `all home: ${rate}`,
            ];
            if (panel.key === 'comm' && !this.compact) {
                parts.push(`${env.meanNeighbours().toFixed(1)} neighbours`);
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
            this.resetStatistics();
            this.newEpisode();
        });

        const temperature = this.query<HTMLInputElement>('#mapf-temperature');
        const temperatureValue = this.query<HTMLElement>('#mapf-temperature-value');
        temperature.value = String(this.temperature);
        temperature.addEventListener('input', () => {
            this.temperature = Number(temperature.value);
            temperatureValue.textContent = this.describeTemperature();
            this.resetStatistics();
        });

        const play = this.query<HTMLButtonElement>('#mapf-play');
        play.addEventListener('click', () => {
            this.running = !this.running;
            play.textContent = this.running ? 'Pause' : 'Play';
        });

        this.query<HTMLButtonElement>('#mapf-new').addEventListener('click', () => {
            this.newEpisode();
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
                <div id="mapf-stats-${key}" class="${c ? 'mt-1 text-[10px]' : 'mt-2 text-[11px]'} font-mono text-[rgb(var(--c-gray-400))]"></div>
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

        const footer = c ? '' : `
            <div class="px-4 py-2 border-t border-[rgb(var(--c-border))] bg-[rgb(var(--c-bg))] text-[11px] text-[rgb(var(--c-gray-500))]">
                Same obstacles, same starts, same goals, same random draws — the only difference is the graph.
                Try 20 robots (identical), then 100. Then drag the temperature to zero and watch both freeze.
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

                <label class="flex flex-col gap-1 min-w-[10rem] flex-1">
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
