/**
 * DivergenceDemo - why a small state error becomes a total one.
 *
 * Two balls start from the same place with velocities differing by a
 * fraction of a pixel per step. Between walls they stay together. At a
 * bounce the disagreement about *where* becomes a disagreement about
 * *when*, and the trajectories are unrelated from then on.
 *
 * No network is involved. This is the environment from part 1 run twice,
 * and it is the floor under every open-loop drift curve in the post.
 */

import {
    DEFAULT_PARAMS,
    step,
    type EnvParams,
    type EnvState,
} from './BallEnv.js';

export interface DivergenceDemoConfig {
    containerId: string;
}

const SCALE = 6;
const FPS = 20;
const TRAIL = 90;

export class DivergenceDemo {
    private container: HTMLElement;
    private params: EnvParams = { ...DEFAULT_PARAMS };
    private a!: EnvState;
    private b!: EnvState;
    private trailA: Array<[number, number]> = [];
    private trailB: Array<[number, number]> = [];
    private gap: number[] = [];
    private nudge = 0.05;
    private timer: number | null = null;

    private canvas!: HTMLCanvasElement;
    private ctx!: CanvasRenderingContext2D;
    private plot!: HTMLCanvasElement;
    private pctx!: CanvasRenderingContext2D;
    private readout!: HTMLElement;
    private slider!: HTMLInputElement;

    constructor(config: DivergenceDemoConfig) {
        const container = document.getElementById(config.containerId);
        if (!container) throw new Error(`Container #${config.containerId} not found`);
        this.container = container;
        this.build();
        this.reset();
        this.timer = window.setInterval(() => this.advance(), 1000 / FPS);
    }

    private build(): void {
        const wrap = document.createElement('div');
        wrap.style.cssText =
            'display:flex;flex-wrap:wrap;gap:20px;align-items:flex-start;padding:20px;' +
            'border:1px solid rgb(var(--c-border));background:rgb(var(--c-surface));' +
            'border-radius:3px;';

        const left = document.createElement('div');
        left.style.cssText = 'display:flex;flex-direction:column;gap:8px;';
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.params.imgW * SCALE;
        this.canvas.height = this.params.imgH * SCALE;
        this.canvas.style.cssText =
            `width:${this.params.imgW * SCALE}px;max-width:100%;height:auto;` +
            'background:#000;border:1px solid rgb(var(--c-border));display:block;';
        this.ctx = this.canvas.getContext('2d')!;
        left.appendChild(this.canvas);
        wrap.appendChild(left);

        const right = document.createElement('div');
        right.style.cssText =
            'flex:1;min-width:240px;display:flex;flex-direction:column;gap:12px;';

        const label = document.createElement('label');
        label.style.cssText =
            'font-size:13px;color:rgb(var(--c-ink));display:flex;' +
            'flex-direction:column;gap:6px;';
        const labelText = document.createElement('span');
        labelText.textContent = 'starting velocity difference';
        this.slider = document.createElement('input');
        this.slider.type = 'range';
        this.slider.min = '0';
        this.slider.max = '20';
        this.slider.value = '5';
        this.slider.step = '1';
        this.slider.style.width = '100%';
        this.slider.addEventListener('input', () => {
            this.nudge = Number(this.slider.value) / 100;
            this.reset();
        });
        label.appendChild(labelText);
        label.appendChild(this.slider);
        right.appendChild(label);

        this.plot = document.createElement('canvas');
        this.plot.width = 460;
        this.plot.height = 130;
        this.plot.style.cssText =
            'width:100%;height:auto;border:1px solid rgb(var(--c-border));display:block;';
        this.pctx = this.plot.getContext('2d')!;
        right.appendChild(this.plot);

        this.readout = document.createElement('div');
        this.readout.style.cssText =
            'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;' +
            'line-height:1.7;color:rgb(var(--c-ink));';
        right.appendChild(this.readout);

        const btn = document.createElement('button');
        btn.textContent = 'Restart';
        btn.style.cssText =
            'font:inherit;font-size:13px;padding:6px 12px;cursor:pointer;align-self:flex-start;' +
            'color:rgb(var(--c-ink));background:rgb(var(--c-bg));' +
            'border:1px solid rgb(var(--c-border));border-radius:2px;';
        btn.addEventListener('click', () => this.reset());
        right.appendChild(btn);

        wrap.appendChild(right);
        this.container.appendChild(wrap);
    }

    private reset(): void {
        // A start that reaches a wall quickly, so the point lands fast.
        this.a = { x: 8, y: 20, vx: 1.5, vy: -1.2, step: 0 };
        this.b = { ...this.a, vx: this.a.vx + this.nudge };
        this.trailA = [];
        this.trailB = [];
        this.gap = [];
        this.draw();
    }

    private advance(): void {
        // No actions: the divergence is the walls, not the noise.
        this.a = step(this.a, 0, 0, this.params);
        this.b = step(this.b, 0, 0, this.params);
        this.trailA.push([this.a.x, this.a.y]);
        this.trailB.push([this.b.x, this.b.y]);
        if (this.trailA.length > TRAIL) {
            this.trailA.shift();
            this.trailB.shift();
        }
        this.gap.push(Math.hypot(this.a.x - this.b.x, this.a.y - this.b.y));
        if (this.gap.length > 200) this.gap.shift();
        if (this.a.step > 400) this.reset();
        this.draw();
    }

    private draw(): void {
        const ctx = this.ctx;
        const { imgW, imgH } = this.params;
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        const trail = (pts: Array<[number, number]>, color: string) => {
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.globalAlpha = 0.55;
            ctx.beginPath();
            pts.forEach(([x, y], i) => {
                const px = (x + 0.5) * SCALE;
                const py = (y + 0.5) * SCALE;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            });
            ctx.stroke();
            ctx.globalAlpha = 1;
        };
        trail(this.trailA, '#5aa2f0');
        trail(this.trailB, '#ff8a5c');

        const dot = (s: EnvState, color: string) => {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc((s.x + 0.5) * SCALE, (s.y + 0.5) * SCALE, 4, 0, 2 * Math.PI);
            ctx.fill();
        };
        dot(this.a, '#5aa2f0');
        dot(this.b, '#ff8a5c');

        // Reference: pixel size, so "a fraction of a pixel" is legible.
        ctx.strokeStyle = 'rgba(255,255,255,0.18)';
        ctx.lineWidth = 1;
        ctx.strokeRect(0.5, 0.5, imgW * SCALE - 1, imgH * SCALE - 1);

        this.drawPlot();

        const d = this.gap.length ? this.gap[this.gap.length - 1] : 0;
        this.readout.innerHTML =
            `step&nbsp;&nbsp;${this.a.step}<br>` +
            `<span style="color:#5aa2f0">ball A</span> vs ` +
            `<span style="color:#ff8a5c">ball B</span>, started ` +
            `${this.nudge.toFixed(2)} px/step apart<br>` +
            `distance now&nbsp;&nbsp;<strong>${d.toFixed(2)} px</strong>`;
    }

    private drawPlot(): void {
        const c = this.pctx;
        const W = this.plot.width;
        const H = this.plot.height;
        c.clearRect(0, 0, W, H);

        const maxGap = Math.hypot(this.params.imgW, this.params.imgH);
        c.strokeStyle = 'rgba(128,128,128,0.35)';
        c.lineWidth = 1;
        c.beginPath();
        c.moveTo(0, H - 1);
        c.lineTo(W, H - 1);
        c.stroke();

        c.strokeStyle = '#ff8a5c';
        c.lineWidth = 1.6;
        c.beginPath();
        this.gap.forEach((g, i) => {
            const x = (i / 200) * W;
            const y = H - 2 - (g / maxGap) * (H - 8);
            if (i === 0) c.moveTo(x, y);
            else c.lineTo(x, y);
        });
        c.stroke();

        c.fillStyle = 'rgba(128,128,128,0.85)';
        c.font = '11px ui-monospace, monospace';
        c.fillText('distance between the two balls', 6, 14);
    }

    public destroy(): void {
        if (this.timer !== null) window.clearInterval(this.timer);
        this.timer = null;
        this.container.innerHTML = '';
    }
}
