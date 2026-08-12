/**
 * ChaseHero - the trained agent chasing a goal you move with the cursor,
 * with the world model's forecast of the next couple of seconds below it.
 *
 * The main panel is the real environment. The white ball is steered by the
 * policy from runs/ac/follow, which receives those pixels and nothing else,
 * never the simulator's state.
 *
 * The strip underneath is not a replay and not a side-by-side. From the
 * state the filter is in right now, the prior is rolled forward with the
 * actor in the loop and the decoder paints each imagined frame, so the
 * strip is what the model expects to happen next. There is deliberately no
 * ground truth beside it: how fast that forecast drifts is measured later
 * in the post, with numbers, which is the honest place for it. A hero is
 * for showing what the thing does.
 */

import { AgentModel, type AgentState } from './AgentModel.js';
import {
    GOAL_PARAMS,
    goalRender,
    goalReset,
    goalReward,
    goalStep,
    makeGoalRng,
    type GoalState,
} from './GoalEnv.js';

export interface ChaseHeroConfig {
    containerId: string;
    modelPath?: string;
}

const SCALE = 9;            // main panel: 32 -> 288px
const TAPE_SCALE = 2;       // forecast frames: 32 -> 64px
const TAPE_LEN = 8;
const TAPE_STRIDE = 2;      // show every other imagined step
const FPS = 12;
const REFRESH_EVERY = 8;    // steps between forecast refreshes

export class ChaseHero {
    private container: HTMLElement;
    private model: AgentModel | null = null;
    private params = { ...GOAL_PARAMS };
    private env!: GoalState;
    private agent!: AgentState;
    private pointer: { gx: number; gy: number } | null = null;
    private timer: number | null = null;
    private sinceRefresh = 1e9;
    private rewardTrace: number[] = [];

    private canvas!: HTMLCanvasElement;
    private ctx!: CanvasRenderingContext2D;
    private tape: HTMLCanvasElement[] = [];
    private status!: HTMLElement;
    private rewardEl!: HTMLElement;

    constructor(config: ChaseHeroConfig) {
        const container = document.getElementById(config.containerId);
        if (!container) throw new Error(`Container #${config.containerId} not found`);
        this.container = container;
        this.build();
        void this.start(config.modelPath ?? '/models/world-models/follow');
    }

    private build(): void {
        const wrap = document.createElement('div');
        wrap.style.cssText =
            'border:1px solid rgb(var(--c-border));background:rgb(var(--c-surface));' +
            'border-radius:3px;padding:20px;';

        const top = document.createElement('div');
        top.style.cssText = 'display:flex;flex-wrap:wrap;gap:22px;align-items:center;';

        this.canvas = document.createElement('canvas');
        this.canvas.width = this.params.imgW * SCALE;
        this.canvas.height = this.params.imgH * SCALE;
        this.canvas.style.cssText =
            `width:${this.params.imgW * SCALE}px;max-width:100%;height:auto;` +
            'image-rendering:pixelated;background:#000;display:block;' +
            'border:1px solid rgb(var(--c-border));cursor:crosshair;touch-action:none;';
        this.ctx = this.canvas.getContext('2d')!;
        top.appendChild(this.canvas);

        const side = document.createElement('div');
        side.style.cssText =
            'flex:1;min-width:230px;display:flex;flex-direction:column;gap:10px;';
        this.status = document.createElement('div');
        this.status.style.cssText =
            'font-size:13.5px;line-height:1.6;color:rgb(var(--c-ink));';
        this.status.textContent = 'loading the trained weights...';
        this.rewardEl = document.createElement('div');
        this.rewardEl.style.cssText =
            'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;' +
            'font-size:12.5px;color:rgb(var(--c-gray-500));';
        side.append(this.status, this.rewardEl);
        top.appendChild(side);
        wrap.appendChild(top);

        const tapeLabel = document.createElement('div');
        tapeLabel.textContent =
            'What the model expects next, imagined from here with no further frames';
        tapeLabel.style.cssText =
            'margin:20px 0 8px;font-size:11px;letter-spacing:.04em;' +
            'text-transform:uppercase;color:#ff8a5c;font-weight:600;';
        wrap.appendChild(tapeLabel);

        const strip = document.createElement('div');
        strip.style.cssText = 'display:flex;gap:6px;flex-wrap:wrap;';
        const ticks = document.createElement('div');
        ticks.style.cssText =
            'display:flex;gap:6px;margin-top:5px;font-size:10.5px;' +
            'color:rgb(var(--c-gray-500));font-family:ui-monospace,monospace;';
        for (let i = 0; i < TAPE_LEN; i++) {
            const c = document.createElement('canvas');
            c.width = this.params.imgW * TAPE_SCALE;
            c.height = this.params.imgH * TAPE_SCALE;
            c.style.cssText =
                `width:${this.params.imgW * TAPE_SCALE}px;height:auto;` +
                'image-rendering:pixelated;background:#000;display:block;' +
                'border:1px solid rgb(var(--c-border));';
            this.tape.push(c);
            strip.appendChild(c);

            const t = document.createElement('div');
            t.textContent = `+${(i + 1) * TAPE_STRIDE}`;
            t.style.cssText =
                `width:${this.params.imgW * TAPE_SCALE}px;text-align:center;`;
            ticks.appendChild(t);
        }
        wrap.append(strip, ticks);

        const toCell = (ev: PointerEvent) => {
            const r = this.canvas.getBoundingClientRect();
            this.pointer = {
                gx: ((ev.clientX - r.left) / r.width) * this.params.imgW - 0.5,
                gy: ((ev.clientY - r.top) / r.height) * this.params.imgH - 0.5,
            };
        };
        this.canvas.addEventListener('pointerdown', (e) => {
            this.canvas.setPointerCapture(e.pointerId);
            toCell(e);
            e.preventDefault();
        });
        this.canvas.addEventListener('pointermove', (e) => {
            if (e.buttons || e.pointerType === 'mouse') toCell(e);
        });
        this.canvas.addEventListener('pointerleave', () => { this.pointer = null; });

        this.container.appendChild(wrap);
    }

    private async start(modelPath: string): Promise<void> {
        try {
            this.model = await AgentModel.load(modelPath);
        } catch (err) {
            this.status.textContent = 'could not load the trained weights';
            console.error('ChaseHero: weights failed to load', err);
            return;
        }
        this.env = goalReset(makeGoalRng(11), this.params);
        this.agent = this.model.initialState();
        this.status.innerHTML =
            '<strong>Move your cursor over the panel</strong> to place the red goal. ' +
            'The white ball is driven by a policy that trained entirely inside this ' +
            'world model and never saw a real transition.';
        this.timer = window.setInterval(() => this.tick(), 1000 / FPS);
    }

    private tick(): void {
        const model = this.model;
        if (!model) return;

        const frame = goalRender(this.env, this.params);
        this.agent = model.observeAndAct(this.agent, frame);
        this.env = goalStep(this.env, this.agent.a[0], this.agent.a[1],
                            this.params, this.pointer ?? undefined);
        this.paint(this.ctx, frame, this.canvas);

        const r = goalReward(this.env, this.params);
        this.rewardTrace.push(r);
        if (this.rewardTrace.length > 60) this.rewardTrace.shift();
        const avg = this.rewardTrace.reduce((a, b) => a + b, 0) / this.rewardTrace.length;
        this.rewardEl.textContent =
            `reward now ${r.toFixed(2)}    ·    last 60 steps ${avg.toFixed(2)}`;

        if (++this.sinceRefresh >= REFRESH_EVERY) {
            this.refreshTape(model);
            this.sinceRefresh = 0;
        }
    }

    /**
     * Roll the prior forward from the current filtered state, with the actor
     * choosing its own actions, and paint every TAPE_STRIDE-th decoded frame.
     */
    private refreshTape(model: AgentModel): void {
        let s: AgentState = { ...this.agent };
        for (let i = 0; i < TAPE_LEN; i++) {
            let out = model.imagine(s);
            s = out.state;
            for (let k = 1; k < TAPE_STRIDE; k++) {
                out = model.imagine(s);
                s = out.state;
            }
            this.paint(this.tape[i].getContext('2d')!, out.frame, this.tape[i]);
        }
    }

    /** RGB float frame to a canvas, nearest neighbour, clamped not rescaled. */
    private paint(ctx: CanvasRenderingContext2D, frame: Float32Array,
                  target: HTMLCanvasElement): void {
        const { imgW: W, imgH: H } = this.params;
        const img = ctx.createImageData(W, H);
        for (let i = 0; i < W * H; i++) {
            img.data[i * 4] = Math.max(0, Math.min(1, frame[i * 3])) * 255;
            img.data[i * 4 + 1] = Math.max(0, Math.min(1, frame[i * 3 + 1])) * 255;
            img.data[i * 4 + 2] = Math.max(0, Math.min(1, frame[i * 3 + 2])) * 255;
            img.data[i * 4 + 3] = 255;
        }
        const off = document.createElement('canvas');
        off.width = W;
        off.height = H;
        off.getContext('2d')!.putImageData(img, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(off, 0, 0, target.width, target.height);
    }

    public destroy(): void {
        if (this.timer !== null) window.clearInterval(this.timer);
        this.timer = null;
        this.container.innerHTML = '';
    }
}
