/**
 * TrainingChart - Interactive Canvas 2D chart showing training curves
 * for the CT segmentation model (loss and dice score views).
 */

import trainingData from '../../data/ct-training-data.json';

export interface TrainingChartConfig {
    containerId: string;
}

interface TrainingDataset {
    epochs: number[];
    train_loss: number[];
    val_loss: number[];
    pseudo_dice: number[];
    pseudo_dice_ema: number[];
}

type ViewMode = 'loss' | 'dice';

export class TrainingChart {
    private container: HTMLElement;
    private canvas: HTMLCanvasElement | null = null;
    private ctx: CanvasRenderingContext2D | null = null;
    private resizeObserver: ResizeObserver | null = null;
    private activeView: ViewMode = 'loss';
    private tooltipEl: HTMLElement | null = null;
    private data: TrainingDataset;

    // Hover state
    private mouseX: number = -1;
    private mouseY: number = -1;

    // Bindings for cleanup
    private boundMouseMove: ((e: MouseEvent) => void) | null = null;
    private boundMouseLeave: (() => void) | null = null;
    private boundThemeChange: (() => void) | null = null;

    // Chart padding
    private static readonly PAD = { top: 20, right: 30, bottom: 50, left: 70 };

    constructor(config: TrainingChartConfig) {
        const container = document.getElementById(config.containerId);
        if (!container) {
            throw new Error(`Container #${config.containerId} not found`);
        }
        this.container = container;
        this.data = trainingData as TrainingDataset;
        this.render();
    }

    private render(): void {
        // Create wrapper
        const wrapper = document.createElement('div');
        wrapper.className = 'training-chart-container';
        wrapper.style.position = 'relative';

        // Tab buttons
        const tabBar = document.createElement('div');
        tabBar.style.cssText = 'display:flex; gap:0; margin-bottom:12px;';

        const btnLoss = this.createTabButton('Loss', true);
        const btnDice = this.createTabButton('Dice Score', false);

        btnLoss.addEventListener('click', () => {
            this.activeView = 'loss';
            this.updateTabStyles(btnLoss, btnDice);
            this.drawChart();
        });
        btnDice.addEventListener('click', () => {
            this.activeView = 'dice';
            this.updateTabStyles(btnDice, btnLoss);
            this.drawChart();
        });

        tabBar.appendChild(btnLoss);
        tabBar.appendChild(btnDice);
        wrapper.appendChild(tabBar);

        // Canvas
        const canvas = document.createElement('canvas');
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.display = 'block';
        wrapper.appendChild(canvas);

        // Tooltip
        const tooltip = document.createElement('div');
        tooltip.style.cssText = `
            position:absolute; display:none; pointer-events:none; z-index:10;
            background:rgb(var(--c-surface)); border:1px solid rgb(var(--c-border)); border-radius:6px;
            padding:8px 12px; font-family:monospace; font-size:11px; color:rgb(var(--c-gray-300));
            box-shadow: 3px 3px 0 rgb(var(--c-ink) / 0.15);
        `;
        wrapper.appendChild(tooltip);

        this.container.appendChild(wrapper);
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.tooltipEl = tooltip;

        // Setup canvas size
        this.setupCanvas();

        // Resize observer
        this.resizeObserver = new ResizeObserver(() => {
            this.setupCanvas();
            this.drawChart();
        });
        this.resizeObserver.observe(wrapper);

        // Mouse interaction
        this.boundMouseMove = (e: MouseEvent) => this.handleMouseMove(e);
        this.boundMouseLeave = () => this.handleMouseLeave();
        canvas.addEventListener('mousemove', this.boundMouseMove);
        canvas.addEventListener('mouseleave', this.boundMouseLeave);

        // Repaint when the palette changes under us
        this.boundThemeChange = () => this.drawChart();
        window.addEventListener('themechange', this.boundThemeChange);

        this.drawChart();
    }

    private createTabButton(text: string, active: boolean): HTMLButtonElement {
        const btn = document.createElement('button');
        btn.textContent = text;
        btn.style.cssText = `
            padding: 6px 16px; font-size: 12px; font-family: monospace;
            border: 1px solid rgb(var(--c-border)); cursor: pointer; transition: all 0.2s;
            ${active
                ? 'background: rgb(var(--c-accent) / 0.12); color: rgb(var(--c-accent)); border-color: rgb(var(--c-accent));'
                : 'background: transparent; color: rgb(var(--c-gray-400)); border-color: rgb(var(--c-border));'
            }
        `;
        btn.style.borderRadius = text === 'Loss' ? '4px 0 0 4px' : '0 4px 4px 0';
        return btn;
    }

    private updateTabStyles(active: HTMLButtonElement, inactive: HTMLButtonElement): void {
        active.style.background = 'rgb(var(--c-accent) / 0.12)';
        active.style.color = 'rgb(var(--c-accent))';
        active.style.borderColor = 'rgb(var(--c-accent))';
        inactive.style.background = 'transparent';
        inactive.style.color = 'rgb(var(--c-gray-400))';
        inactive.style.borderColor = 'rgb(var(--c-border))';
    }

    private setupCanvas(): void {
        if (!this.canvas) return;
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        if (this.ctx) {
            this.ctx.scale(dpr, dpr);
        }
    }

    /**
     * Canvas can't read CSS variables, so resolve the theme tokens here and
     * redraw whenever the theme changes.
     */
    private palette() {
        const styles = getComputedStyle(document.documentElement);
        const token = (name: string, alpha = 1) => {
            const channels = styles.getPropertyValue(name).trim();
            if (!channels) return '#888888';
            return alpha === 1 ? `rgb(${channels})` : `rgb(${channels} / ${alpha})`;
        };

        return {
            grid: token('--c-border'),
            axis: token('--c-gray-400'),
            legend: token('--c-gray-300'),
            accent: token('--c-accent'),
            red: token('--c-red'),
            green: token('--c-green'),
            crosshair: token('--c-ink', 0.25),
        };
    }

    private drawChart(): void {
        if (!this.canvas || !this.ctx) return;

        const C = this.palette();
        const ctx = this.ctx;
        const rect = this.canvas.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;
        const P = TrainingChart.PAD;

        // Clear
        ctx.clearRect(0, 0, w, h);

        const chartW = w - P.left - P.right;
        const chartH = h - P.top - P.bottom;

        const { epochs } = this.data;
        const xMin = 0;
        const xMax = 850;

        let yMin: number, yMax: number;
        let series: { data: number[]; color: string; label: string; dashed?: boolean }[];

        if (this.activeView === 'loss') {
            yMin = -0.9;
            yMax = 0;
            series = [
                { data: this.data.train_loss, color: C.accent, label: 'Train Loss' },
                { data: this.data.val_loss, color: C.red, label: 'Val Loss' },
            ];
        } else {
            yMin = 0.4;
            yMax = 0.9;
            series = [
                { data: this.data.pseudo_dice, color: C.axis, label: 'Raw Pseudo Dice', dashed: true },
                { data: this.data.pseudo_dice_ema, color: C.green, label: 'Dice EMA' },
            ];
        }

        // Helpers
        const toX = (epoch: number) => P.left + ((epoch - xMin) / (xMax - xMin)) * chartW;
        const toY = (val: number) => P.top + ((yMax - val) / (yMax - yMin)) * chartH;

        // Draw grid
        ctx.strokeStyle = C.grid;
        ctx.lineWidth = 1;
        const yTicks = 6;
        for (let i = 0; i <= yTicks; i++) {
            const yVal = yMin + (yMax - yMin) * (i / yTicks);
            const py = toY(yVal);
            ctx.beginPath();
            ctx.moveTo(P.left, py);
            ctx.lineTo(w - P.right, py);
            ctx.stroke();

            // Y axis labels
            ctx.fillStyle = C.axis;
            ctx.font = '10px monospace';
            ctx.textAlign = 'right';
            ctx.fillText(yVal.toFixed(2), P.left - 8, py + 3);
        }

        const xTicks = 8;
        for (let i = 0; i <= xTicks; i++) {
            const xVal = xMin + (xMax - xMin) * (i / xTicks);
            const px = toX(xVal);
            ctx.beginPath();
            ctx.moveTo(px, P.top);
            ctx.lineTo(px, h - P.bottom);
            ctx.stroke();

            // X axis labels
            ctx.fillStyle = C.axis;
            ctx.font = '10px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(Math.round(xVal).toString(), px, h - P.bottom + 16);
        }

        // Axis labels
        ctx.fillStyle = C.axis;
        ctx.font = '11px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Epoch', w / 2, h - 8);

        ctx.save();
        ctx.translate(14, h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(this.activeView === 'loss' ? 'Loss' : 'Dice Score', 0, 0);
        ctx.restore();

        // Draw series
        for (const s of series) {
            ctx.strokeStyle = s.color;
            ctx.lineWidth = s.dashed ? 1 : 2;
            if (s.dashed) {
                ctx.setLineDash([4, 4]);
            } else {
                ctx.setLineDash([]);
            }
            ctx.beginPath();
            for (let i = 0; i < epochs.length; i++) {
                const px = toX(epochs[i]);
                const py = toY(s.data[i]);
                if (i === 0) {
                    ctx.moveTo(px, py);
                } else {
                    ctx.lineTo(px, py);
                }
            }
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Legend
        const legendX = P.left + 10;
        const legendY = P.top + 15;
        for (let i = 0; i < series.length; i++) {
            const s = series[i];
            const ly = legendY + i * 18;
            ctx.strokeStyle = s.color;
            ctx.lineWidth = s.dashed ? 1 : 2;
            if (s.dashed) ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(legendX, ly);
            ctx.lineTo(legendX + 20, ly);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = C.legend;
            ctx.font = '10px monospace';
            ctx.textAlign = 'left';
            ctx.fillText(s.label, legendX + 26, ly + 3);
        }

        // Draw hover line if mouse is in chart area
        if (this.mouseX >= P.left && this.mouseX <= w - P.right) {
            const hoverEpoch = xMin + ((this.mouseX - P.left) / chartW) * (xMax - xMin);
            // Find nearest data point
            let nearest = 0;
            let nearestDist = Infinity;
            for (let i = 0; i < epochs.length; i++) {
                const d = Math.abs(epochs[i] - hoverEpoch);
                if (d < nearestDist) {
                    nearestDist = d;
                    nearest = i;
                }
            }

            const px = toX(epochs[nearest]);

            // Vertical line
            ctx.strokeStyle = C.crosshair;
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(px, P.top);
            ctx.lineTo(px, h - P.bottom);
            ctx.stroke();
            ctx.setLineDash([]);

            // Dots on data points
            for (const s of series) {
                const py = toY(s.data[nearest]);
                ctx.fillStyle = s.color;
                ctx.beginPath();
                ctx.arc(px, py, 4, 0, Math.PI * 2);
                ctx.fill();
            }

            // Update tooltip
            if (this.tooltipEl) {
                this.tooltipEl.style.display = 'block';
                this.tooltipEl.style.left = `${this.mouseX + 12}px`;
                this.tooltipEl.style.top = `${this.mouseY - 40}px`;

                let html = `<div style="color:rgb(var(--c-ink));margin-bottom:4px">Epoch ${epochs[nearest]}</div>`;
                for (const s of series) {
                    html += `<div style="color:${s.color}">${s.label}: ${s.data[nearest].toFixed(4)}</div>`;
                }
                this.tooltipEl.innerHTML = html;
            }
        }
    }

    private handleMouseMove(e: MouseEvent): void {
        if (!this.canvas) return;
        const rect = this.canvas.getBoundingClientRect();
        this.mouseX = e.clientX - rect.left;
        this.mouseY = e.clientY - rect.top;
        this.drawChart();
    }

    private handleMouseLeave(): void {
        this.mouseX = -1;
        this.mouseY = -1;
        if (this.tooltipEl) {
            this.tooltipEl.style.display = 'none';
        }
        this.drawChart();
    }

    destroy(): void {
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }
        if (this.canvas && this.boundMouseMove) {
            this.canvas.removeEventListener('mousemove', this.boundMouseMove);
        }
        if (this.canvas && this.boundMouseLeave) {
            this.canvas.removeEventListener('mouseleave', this.boundMouseLeave);
        }
        if (this.boundThemeChange) {
            window.removeEventListener('themechange', this.boundThemeChange);
        }
        this.boundMouseMove = null;
        this.boundMouseLeave = null;
        this.boundThemeChange = null;

        const wrapper = this.container.querySelector('.training-chart-container');
        if (wrapper) {
            wrapper.remove();
        }
        this.canvas = null;
        this.ctx = null;
        this.tooltipEl = null;
    }
}
