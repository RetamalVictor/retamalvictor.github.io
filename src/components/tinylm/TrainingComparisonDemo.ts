/**
 * TrainingComparisonDemo - Interactive chart comparing LLaMA vs GPT training curves.
 *
 * Displays loss/perplexity over training steps for both architectures,
 * with toggles for metric type and train/val split.
 */

import trainingCurvesData from '../../data/training_curves.json';
import trainingComparisonData from '../../data/training_comparison.json';

interface TrainingComparisonConfig {
    containerId: string;
}

// Data point format: [step, loss]
type DataPoint = [number, number];

interface RawArchData {
    train_loss: DataPoint[];
    val_loss: DataPoint[];
}

interface ArchitectureData {
    name: string;
    config: Record<string, any>;
    final: {
        train_loss: number;
        val_loss: number;
        train_ppl: number;
        val_ppl: number;
    };
}

type MetricType = 'loss' | 'perplexity';
type SplitType = 'val' | 'both';

export class TrainingComparisonDemo {
    private container: HTMLElement;
    private canvas: HTMLCanvasElement | null = null;
    private ctx: CanvasRenderingContext2D | null = null;
    private llamaValLoss: DataPoint[];
    private llamaTrainLoss: DataPoint[];
    private gptValLoss: DataPoint[];
    private gptTrainLoss: DataPoint[];
    private comparison: { llama: ArchitectureData; gpt: ArchitectureData; training: any };
    private animationFrame: number | null = null;

    // Display state
    private metric: MetricType = 'loss';
    private split: SplitType = 'val';
    private ignoreOutliers: boolean = false;

    constructor(config: TrainingComparisonConfig) {
        const container = document.getElementById(config.containerId);
        if (!container) {
            throw new Error(`Container #${config.containerId} not found`);
        }

        this.container = container;

        // Parse the raw data
        const rawData = trainingCurvesData as { llama: RawArchData; gpt: RawArchData };
        this.llamaValLoss = rawData.llama.val_loss;
        this.llamaTrainLoss = rawData.llama.train_loss;
        this.gptValLoss = rawData.gpt.val_loss;
        this.gptTrainLoss = rawData.gpt.train_loss;
        this.comparison = trainingComparisonData as any;

        this.render();
    }

    private render(): void {
        const { llama, gpt, training } = this.comparison;

        this.container.innerHTML = `
            <div class="bg-dark-surface border border-dark-border rounded-lg p-6">
                <!-- Header with controls -->
                <div class="flex flex-wrap justify-between items-start mb-4 gap-4">
                    <div>
                        <h3 class="text-lg font-semibold text-white mb-1">Training Comparison</h3>
                        <p class="text-sm text-gray-400">${training.model_params} model on ${training.dataset} (${training.steps.toLocaleString()} steps)</p>
                    </div>

                    <!-- Controls -->
                    <div class="flex flex-wrap gap-3">
                        <!-- Metric toggle -->
                        <div class="flex rounded-md overflow-hidden border border-dark-border">
                            <button id="btn-loss" class="px-3 py-1.5 text-xs font-medium bg-cyan-500/20 text-cyan-400 border-r border-dark-border">
                                Loss
                            </button>
                            <button id="btn-ppl" class="px-3 py-1.5 text-xs font-medium text-gray-400 hover:text-gray-300 hover:bg-dark-bg/50">
                                Perplexity
                            </button>
                        </div>

                        <!-- Split toggle -->
                        <div class="flex rounded-md overflow-hidden border border-dark-border">
                            <button id="btn-val" class="px-3 py-1.5 text-xs font-medium bg-cyan-500/20 text-cyan-400 border-r border-dark-border">
                                Val
                            </button>
                            <button id="btn-both" class="px-3 py-1.5 text-xs font-medium text-gray-400 hover:text-gray-300 hover:bg-dark-bg/50">
                                Train + Val
                            </button>
                        </div>

                        <!-- Outliers toggle -->
                        <button id="btn-outliers" class="px-3 py-1.5 text-xs font-medium rounded-md border border-dark-border text-gray-400 hover:text-gray-300 hover:bg-dark-bg/50" title="Clip values above 95th percentile">
                            Ignore outliers
                        </button>
                    </div>
                </div>

                <!-- Legend -->
                <div class="flex flex-wrap gap-4 text-xs mb-4">
                    <div class="flex items-center gap-2">
                        <span class="w-4 h-0.5 bg-cyan-400"></span>
                        <span class="text-gray-300">LLaMA</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="w-4 h-0.5 bg-purple-400"></span>
                        <span class="text-gray-300">GPT</span>
                    </div>
                    <div id="legend-train" class="flex items-center gap-2 hidden">
                        <span class="w-4 h-0.5 border-t-2 border-dashed border-gray-400"></span>
                        <span class="text-gray-400">train (dashed)</span>
                    </div>
                </div>

                <!-- Chart -->
                <div class="relative mb-6">
                    <canvas id="training-chart" class="w-full" style="height: 300px;"></canvas>
                    <div id="tooltip" class="absolute hidden bg-dark-bg border border-dark-border rounded px-3 py-2 text-sm pointer-events-none z-10 shadow-lg"></div>
                </div>

                <!-- Final Results -->
                <div class="grid grid-cols-2 gap-4">
                    <div class="bg-dark-bg rounded-lg p-4 border-l-4 border-cyan-400">
                        <div class="text-sm text-gray-400 mb-2">LLaMA-style (Pre-norm, RoPE, SwiGLU)</div>
                        <div class="grid grid-cols-2 gap-3">
                            <div>
                                <div class="text-2xl font-bold text-cyan-400">${llama.final.val_loss.toFixed(2)}</div>
                                <div class="text-xs text-gray-500">Val Loss</div>
                            </div>
                            <div>
                                <div class="text-2xl font-bold text-cyan-400">${llama.final.val_ppl.toFixed(1)}</div>
                                <div class="text-xs text-gray-500">Val PPL</div>
                            </div>
                        </div>
                    </div>
                    <div class="bg-dark-bg rounded-lg p-4 border-l-4 border-purple-400">
                        <div class="text-sm text-gray-400 mb-2">GPT-style (Post-norm, Learned, GELU)</div>
                        <div class="grid grid-cols-2 gap-3">
                            <div>
                                <div class="text-2xl font-bold text-purple-400">${gpt.final.val_loss.toFixed(2)}</div>
                                <div class="text-xs text-gray-500">Val Loss</div>
                            </div>
                            <div>
                                <div class="text-2xl font-bold text-purple-400">${gpt.final.val_ppl.toFixed(1)}</div>
                                <div class="text-xs text-gray-500">Val PPL</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Config Summary -->
                <details class="mt-4">
                    <summary class="text-sm text-gray-400 cursor-pointer hover:text-gray-300">Training config</summary>
                    <div class="mt-2 text-xs text-gray-500 grid grid-cols-2 md:grid-cols-4 gap-2">
                        <span>batch_size: ${training.batch_size}</span>
                        <span>seq_len: ${training.seq_len}</span>
                        <span>lr: ${training.lr}</span>
                        <span>steps: ${training.steps.toLocaleString()}</span>
                    </div>
                </details>
            </div>
        `;

        // Initialize canvas
        this.canvas = document.getElementById('training-chart') as HTMLCanvasElement;
        if (this.canvas) {
            this.setupCanvas();
            this.drawChart();
            this.setupInteraction();
        }

        this.setupControls();
    }

    private setupControls(): void {
        const btnLoss = document.getElementById('btn-loss');
        const btnPpl = document.getElementById('btn-ppl');
        const btnVal = document.getElementById('btn-val');
        const btnBoth = document.getElementById('btn-both');
        const legendTrain = document.getElementById('legend-train');

        const activeClass = 'bg-cyan-500/20 text-cyan-400';
        const inactiveClass = 'text-gray-400 hover:text-gray-300 hover:bg-dark-bg/50';

        const updateButtonStyle = (active: HTMLElement, inactive: HTMLElement) => {
            active.className = `px-3 py-1.5 text-xs font-medium ${activeClass} border-r border-dark-border`;
            inactive.className = `px-3 py-1.5 text-xs font-medium ${inactiveClass}`;
        };

        btnLoss?.addEventListener('click', () => {
            this.metric = 'loss';
            updateButtonStyle(btnLoss, btnPpl!);
            this.drawChart();
        });

        btnPpl?.addEventListener('click', () => {
            this.metric = 'perplexity';
            updateButtonStyle(btnPpl, btnLoss!);
            // Fix border on second button
            btnPpl.className = `px-3 py-1.5 text-xs font-medium ${activeClass}`;
            btnLoss!.className = `px-3 py-1.5 text-xs font-medium ${inactiveClass} border-r border-dark-border`;
            this.drawChart();
        });

        btnVal?.addEventListener('click', () => {
            this.split = 'val';
            updateButtonStyle(btnVal, btnBoth!);
            legendTrain?.classList.add('hidden');
            this.drawChart();
        });

        btnBoth?.addEventListener('click', () => {
            this.split = 'both';
            updateButtonStyle(btnBoth, btnVal!);
            btnBoth.className = `px-3 py-1.5 text-xs font-medium ${activeClass}`;
            btnVal!.className = `px-3 py-1.5 text-xs font-medium ${inactiveClass} border-r border-dark-border`;
            legendTrain?.classList.remove('hidden');
            this.drawChart();
        });

        // Outliers toggle
        const btnOutliers = document.getElementById('btn-outliers');
        btnOutliers?.addEventListener('click', () => {
            this.ignoreOutliers = !this.ignoreOutliers;
            if (this.ignoreOutliers) {
                btnOutliers.className = `px-3 py-1.5 text-xs font-medium rounded-md border border-dark-border ${activeClass}`;
            } else {
                btnOutliers.className = `px-3 py-1.5 text-xs font-medium rounded-md border border-dark-border text-gray-400 hover:text-gray-300 hover:bg-dark-bg/50`;
            }
            this.drawChart();
        });
    }

    private setupCanvas(): void {
        if (!this.canvas) return;

        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();

        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;

        this.ctx = this.canvas.getContext('2d');
        if (this.ctx) {
            this.ctx.scale(dpr, dpr);
        }
    }

    // Convert loss to perplexity: PPL = e^loss
    private toPerplexity(loss: number): number {
        return Math.exp(loss);
    }

    // Calculate percentile value from array
    private getPercentile(values: number[], percentile: number): number {
        const sorted = [...values].sort((a, b) => a - b);
        const index = Math.ceil((percentile / 100) * sorted.length) - 1;
        return sorted[Math.max(0, index)];
    }

    // Transform data points based on current metric
    private transformData(data: DataPoint[]): DataPoint[] {
        if (this.metric === 'loss') {
            return data;
        }
        return data.map(([step, loss]) => [step, this.toPerplexity(loss)]);
    }

    private drawChart(): void {
        if (!this.ctx || !this.canvas) return;

        const ctx = this.ctx;
        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        const margin = { top: 20, right: 20, bottom: 40, left: 55 };
        const chartWidth = width - margin.left - margin.right;
        const chartHeight = height - margin.top - margin.bottom;

        ctx.clearRect(0, 0, width, height);

        // Get transformed data
        const llamaVal = this.transformData(this.llamaValLoss);
        const gptVal = this.transformData(this.gptValLoss);
        const llamaTrain = this.transformData(this.llamaTrainLoss);
        const gptTrain = this.transformData(this.gptTrainLoss);

        // Calculate range
        const valPoints = [...llamaVal, ...gptVal];
        const trainPoints = this.split === 'both' ? [...llamaTrain, ...gptTrain] : [];
        const allPoints = [...valPoints, ...trainPoints];

        const maxStep = Math.max(...allPoints.map(p => p[0]));
        const allValues = allPoints.map(p => p[1]);

        let minValue: number;
        let maxValue: number;

        if (this.ignoreOutliers) {
            // Use 5th-95th percentile range to ignore outliers
            minValue = this.getPercentile(allValues, 5) * 0.98;
            maxValue = this.getPercentile(allValues, 95) * 1.02;
        } else {
            minValue = Math.min(...allValues) * 0.95;
            maxValue = Math.max(...allValues) * 1.05;
        }

        // Scale functions
        const xScale = (step: number) => margin.left + (step / maxStep) * chartWidth;
        const yScale = (value: number) => margin.top + (1 - (value - minValue) / (maxValue - minValue)) * chartHeight;

        // Draw grid
        ctx.strokeStyle = '#374151';
        ctx.lineWidth = 0.5;

        const yTicks = 5;
        for (let i = 0; i <= yTicks; i++) {
            const y = margin.top + (i / yTicks) * chartHeight;
            ctx.beginPath();
            ctx.moveTo(margin.left, y);
            ctx.lineTo(width - margin.right, y);
            ctx.stroke();

            const value = maxValue - (i / yTicks) * (maxValue - minValue);
            ctx.fillStyle = '#9ca3af';
            ctx.font = '11px system-ui';
            ctx.textAlign = 'right';
            // Format based on metric
            const label = this.metric === 'loss' ? value.toFixed(2) : value.toFixed(1);
            ctx.fillText(label, margin.left - 8, y + 4);
        }

        // X-axis labels
        const xTicks = 5;
        ctx.textAlign = 'center';
        for (let i = 0; i <= xTicks; i++) {
            const step = (i / xTicks) * maxStep;
            const x = xScale(step);
            ctx.fillText(`${(step / 1000).toFixed(0)}k`, x, height - margin.bottom + 20);
        }

        // Axis labels
        ctx.fillStyle = '#9ca3af';
        ctx.font = '12px system-ui';
        ctx.textAlign = 'center';
        ctx.fillText('Steps', width / 2, height - 5);

        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(this.metric === 'loss' ? 'Loss' : 'Perplexity', 0, 0);
        ctx.restore();

        // Draw lines
        const drawLine = (data: DataPoint[], color: string, dashed: boolean) => {
            if (data.length === 0) return;

            ctx.strokeStyle = color;
            ctx.lineWidth = dashed ? 1.5 : 2;
            ctx.setLineDash(dashed ? [4, 4] : []);
            ctx.globalAlpha = dashed ? 0.6 : 1;
            ctx.beginPath();

            const sampledData = data.filter((_, i) => i % 5 === 0 || i === data.length - 1);

            sampledData.forEach((point, i) => {
                const x = xScale(point[0]);
                const y = yScale(point[1]);
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.globalAlpha = 1;
        };

        // Draw train first (behind) if enabled
        if (this.split === 'both') {
            drawLine(llamaTrain, '#22d3d3', true);
            drawLine(gptTrain, '#a78bfa', true);
        }

        // Draw validation (on top)
        drawLine(llamaVal, '#22d3d3', false);
        drawLine(gptVal, '#a78bfa', false);
    }

    private setupInteraction(): void {
        if (!this.canvas) return;

        const tooltip = document.getElementById('tooltip');
        if (!tooltip) return;

        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas!.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const margin = { top: 20, right: 20, bottom: 40, left: 55 };
            const chartWidth = rect.width - margin.left - margin.right;
            const chartHeight = rect.height - margin.top - margin.bottom;

            // Get transformed data for current metric
            const llamaVal = this.transformData(this.llamaValLoss);
            const gptVal = this.transformData(this.gptValLoss);

            const allPoints = [...llamaVal, ...gptVal];
            const maxStep = Math.max(...allPoints.map(p => p[0]));
            const minValue = Math.min(...allPoints.map(p => p[1])) * 0.95;
            const maxValue = Math.max(...allPoints.map(p => p[1])) * 1.05;

            const step = ((x - margin.left) / chartWidth) * maxStep;

            const findClosest = (data: DataPoint[]): DataPoint => {
                return data.reduce((closest, point) =>
                    Math.abs(point[0] - step) < Math.abs(closest[0] - step) ? point : closest
                );
            };

            const llamaPoint = findClosest(llamaVal);
            const gptPoint = findClosest(gptVal);

            // Also find train points if showing both
            const llamaTrainPoint = this.split === 'both' ? findClosest(this.transformData(this.llamaTrainLoss)) : null;
            const gptTrainPoint = this.split === 'both' ? findClosest(this.transformData(this.gptTrainLoss)) : null;

            const yScale = (value: number) => margin.top + (1 - (value - minValue) / (maxValue - minValue)) * chartHeight;
            const llamaY = yScale(llamaPoint[1]);
            const gptY = yScale(gptPoint[1]);

            const closest = Math.abs(llamaY - y) < Math.abs(gptY - y)
                ? { arch: 'LLaMA', valPoint: llamaPoint, trainPoint: llamaTrainPoint, color: '#22d3d3' }
                : { arch: 'GPT', valPoint: gptPoint, trainPoint: gptTrainPoint, color: '#a78bfa' };

            if (x > margin.left && x < rect.width - margin.right &&
                y > margin.top && y < rect.height - margin.bottom) {

                const metricLabel = this.metric === 'loss' ? 'Loss' : 'PPL';
                const valValue = this.metric === 'loss'
                    ? closest.valPoint[1].toFixed(4)
                    : closest.valPoint[1].toFixed(2);

                let trainInfo = '';
                if (closest.trainPoint) {
                    const trainValue = this.metric === 'loss'
                        ? closest.trainPoint[1].toFixed(4)
                        : closest.trainPoint[1].toFixed(2);
                    trainInfo = `<div class="text-gray-500">Train ${metricLabel}: ${trainValue}</div>`;
                }

                tooltip.innerHTML = `
                    <div class="font-medium" style="color: ${closest.color}">${closest.arch}</div>
                    <div class="text-gray-400">Step: ${closest.valPoint[0].toLocaleString()}</div>
                    <div class="text-gray-300">Val ${metricLabel}: ${valValue}</div>
                    ${trainInfo}
                `;
                tooltip.style.left = `${x + 10}px`;
                tooltip.style.top = `${y - 10}px`;
                tooltip.classList.remove('hidden');
            } else {
                tooltip.classList.add('hidden');
            }
        });

        this.canvas.addEventListener('mouseleave', () => {
            tooltip?.classList.add('hidden');
        });

        window.addEventListener('resize', () => {
            if (this.animationFrame) {
                cancelAnimationFrame(this.animationFrame);
            }
            this.animationFrame = requestAnimationFrame(() => {
                this.setupCanvas();
                this.drawChart();
            });
        });
    }

    public destroy(): void {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }
}
