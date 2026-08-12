/**
 * BallEnvDemo - the bouncing ball environment running live in the post.
 *
 * The point it has to make is pedagogical rather than decorative. A reader
 * can pause on any single frame and try to say which way the ball is
 * moving, which is precisely the question the VAE fails at the end of the
 * post. So the ground-truth overlay starts hidden: you should feel the
 * ambiguity before you are shown the answer.
 */

import {
    DEFAULT_PARAMS,
    makeRng,
    randomNudge,
    render,
    reset,
    step,
    type EnvParams,
    type EnvState,
} from './BallEnv.js';

export interface BallEnvDemoConfig {
    containerId: string;
}

const SCALE = 9;              // 32px frame -> 288px canvas
const FPS = 20;

export class BallEnvDemo {
    private container: HTMLElement;
    private params: EnvParams = { ...DEFAULT_PARAMS };
    private rand: () => number;
    private state: EnvState;
    private canvas!: HTMLCanvasElement;
    private ctx!: CanvasRenderingContext2D;
    private frameImage!: ImageData;
    private timer: number | null = null;
    private showTruth = false;
    private playing = true;
    private readout!: HTMLElement;
    private playBtn!: HTMLButtonElement;
    private truthBtn!: HTMLButtonElement;

    constructor(config: BallEnvDemoConfig) {
        const container = document.getElementById(config.containerId);
        if (!container) {
            throw new Error(`Container #${config.containerId} not found`);
        }
        this.container = container;
        this.rand = makeRng(7);
        this.state = reset(this.rand, this.params);
        this.build();
        this.draw();
        this.play();
    }

    private build(): void {
        const wrap = document.createElement('div');
        wrap.style.cssText =
            'display:flex;flex-wrap:wrap;gap:20px;align-items:flex-start;' +
            'padding:20px;border:1px solid rgb(var(--c-border));' +
            'background:rgb(var(--c-surface));border-radius:3px;';

        // Left: the frame itself, pixelated so 32x32 stays legible.
        const stage = document.createElement('div');
        stage.style.cssText = 'display:flex;flex-direction:column;gap:8px;';
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.params.imgW * SCALE;
        this.canvas.height = this.params.imgH * SCALE;
        this.canvas.style.cssText =
            `width:${this.params.imgW * SCALE}px;max-width:100%;height:auto;` +
            'image-rendering:pixelated;background:#000;' +
            'border:1px solid rgb(var(--c-border));display:block;';
        const ctx = this.canvas.getContext('2d');
        if (!ctx) throw new Error('2d context unavailable');
        this.ctx = ctx;
        this.frameImage = ctx.createImageData(this.params.imgW, this.params.imgH);
        stage.appendChild(this.canvas);

        const cap = document.createElement('div');
        cap.textContent = '32 x 32, one channel. This is the model\'s entire input.';
        cap.style.cssText =
            'font-size:12px;color:rgb(var(--c-gray-500));';
        stage.appendChild(cap);
        wrap.appendChild(stage);

        // Right: controls and the ground truth the simulator knows.
        const side = document.createElement('div');
        side.style.cssText =
            'flex:1;min-width:220px;display:flex;flex-direction:column;gap:12px;';

        const controls = document.createElement('div');
        controls.style.cssText = 'display:flex;flex-wrap:wrap;gap:8px;';
        this.playBtn = this.button('Pause', () => this.toggle());
        controls.appendChild(this.playBtn);
        controls.appendChild(this.button('Step', () => {
            this.pause();
            this.advance();
        }));
        controls.appendChild(this.button('New episode', () => {
            this.state = reset(this.rand, this.params);
            this.draw();
        }));
        side.appendChild(controls);

        this.truthBtn = this.button('Show what the simulator knows', () => {
            this.showTruth = !this.showTruth;
            this.truthBtn.textContent = this.showTruth
                ? 'Hide the ground truth'
                : 'Show what the simulator knows';
            this.draw();
        });
        this.truthBtn.style.width = '100%';
        side.appendChild(this.truthBtn);

        this.readout = document.createElement('div');
        this.readout.style.cssText =
            'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;' +
            'font-size:12.5px;line-height:1.8;color:rgb(var(--c-ink));' +
            'border-top:1px solid rgb(var(--c-border));padding-top:10px;';
        side.appendChild(this.readout);

        wrap.appendChild(side);
        this.container.appendChild(wrap);
    }

    private button(label: string, onClick: () => void): HTMLButtonElement {
        const b = document.createElement('button');
        b.textContent = label;
        b.style.cssText =
            'font:inherit;font-size:13px;padding:6px 12px;cursor:pointer;' +
            'color:rgb(var(--c-ink));background:rgb(var(--c-bg));' +
            'border:1px solid rgb(var(--c-border));border-radius:2px;';
        b.addEventListener('click', onClick);
        return b;
    }

    private advance(): void {
        const [ax, ay] = randomNudge(this.rand, this.params.maxNudge);
        this.state = step(this.state, ax, ay, this.params);
        this.draw();
    }

    private draw(): void {
        const { imgW, imgH } = this.params;
        const frame = render(this.state, this.params);
        for (let i = 0; i < imgW * imgH; i++) {
            const v = Math.max(0, Math.min(1, frame[i])) * 255;
            this.frameImage.data[i * 4] = v;
            this.frameImage.data[i * 4 + 1] = v;
            this.frameImage.data[i * 4 + 2] = v;
            this.frameImage.data[i * 4 + 3] = 255;
        }

        // Draw the frame at 1:1 on an offscreen canvas, then blit it scaled
        // with smoothing off, so the reader sees actual pixels.
        const off = document.createElement('canvas');
        off.width = imgW;
        off.height = imgH;
        off.getContext('2d')!.putImageData(this.frameImage, 0, 0);
        this.ctx.imageSmoothingEnabled = false;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(off, 0, 0, this.canvas.width, this.canvas.height);

        if (this.showTruth) this.drawVelocity();
        this.updateReadout();
    }

    /** The arrow exists nowhere in the pixels. That is the whole point. */
    private drawVelocity(): void {
        const { x, y, vx, vy } = this.state;
        const cx = (x + 0.5) * SCALE;
        const cy = (y + 0.5) * SCALE;
        const len = 14 * SCALE * 0.5;
        const speed = Math.hypot(vx, vy) || 1;
        const ex = cx + (vx / speed) * len;
        const ey = cy + (vy / speed) * len;

        const ctx = this.ctx;
        ctx.strokeStyle = '#ff8a5c';
        ctx.fillStyle = '#ff8a5c';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(ex, ey);
        ctx.stroke();

        const a = Math.atan2(ey - cy, ex - cx);
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - 10 * Math.cos(a - 0.4), ey - 10 * Math.sin(a - 0.4));
        ctx.lineTo(ex - 10 * Math.cos(a + 0.4), ey - 10 * Math.sin(a + 0.4));
        ctx.closePath();
        ctx.fill();
    }

    private updateReadout(): void {
        const { x, y, vx, vy, step: t } = this.state;
        const f = (v: number) => (v >= 0 ? ' ' : '') + v.toFixed(2);
        if (this.showTruth) {
            this.readout.innerHTML =
                `frame&nbsp;&nbsp;${t}<br>` +
                `x, y&nbsp;&nbsp;&nbsp;${f(x)}, ${f(y)}<br>` +
                `vx, vy&nbsp;${f(vx)}, ${f(vy)}<br>` +
                `<span style="color:#ff8a5c">the arrow is drawn from vx, vy, ` +
                `which appear nowhere in the pixels</span>`;
        } else {
            this.readout.innerHTML =
                `frame&nbsp;&nbsp;${t}<br>` +
                `<span style="color:rgb(var(--c-gray-500))">Pause, then say ` +
                `which way the ball is moving. The renderer is a function of ` +
                `position alone, so the frame cannot tell you.</span>`;
        }
    }

    private play(): void {
        if (this.timer !== null) return;
        this.playing = true;
        this.playBtn.textContent = 'Pause';
        this.timer = window.setInterval(() => this.advance(), 1000 / FPS);
    }

    private pause(): void {
        if (this.timer === null) return;
        window.clearInterval(this.timer);
        this.timer = null;
        this.playing = false;
        this.playBtn.textContent = 'Play';
    }

    private toggle(): void {
        if (this.playing) this.pause();
        else this.play();
    }

    public destroy(): void {
        this.pause();
        this.container.innerHTML = '';
    }
}
