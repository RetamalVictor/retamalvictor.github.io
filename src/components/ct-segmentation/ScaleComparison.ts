/**
 * ScaleComparison - Visual comparing 2D image size vs 3D CT volume size
 * for the "What Makes 3D Medical Segmentation Hard" section.
 *
 * Shows a 2D grid vs isometric 3D stack, plus a relative bar chart.
 */

export interface ScaleComparisonConfig {
    containerId: string;
}

export class ScaleComparison {
    private container: HTMLElement;
    private wrapper: HTMLElement | null = null;

    constructor(config: ScaleComparisonConfig) {
        const container = document.getElementById(config.containerId);
        if (!container) {
            throw new Error(`Container #${config.containerId} not found`);
        }
        this.container = container;
        this.render();
    }

    private render(): void {
        const wrapper = document.createElement('div');
        wrapper.className = 'diagram-container';
        wrapper.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:28px;padding:24px;';

        // Title
        const title = document.createElement('h4');
        title.textContent = '2D Image vs 3D CT Volume: Scale Comparison';
        title.style.cssText = 'margin:0;color:rgb(var(--c-ink));font-size:14px;font-weight:600;letter-spacing:0.5px;';
        wrapper.appendChild(title);

        // Panels container
        const panels = document.createElement('div');
        panels.style.cssText = 'display:flex;gap:40px;flex-wrap:wrap;justify-content:center;width:100%;';

        // Left panel - 2D image
        panels.appendChild(this.create2DPanel());
        // Right panel - 3D volume
        panels.appendChild(this.create3DPanel());

        wrapper.appendChild(panels);

        // Bar chart comparison
        wrapper.appendChild(this.createBarChart());

        this.container.appendChild(wrapper);
        this.wrapper = wrapper;
    }

    private create2DPanel(): HTMLElement {
        const panel = document.createElement('div');
        panel.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:12px;flex:1;min-width:180px;max-width:240px;';

        const label = document.createElement('div');
        label.textContent = '2D Image';
        label.style.cssText = 'color:rgb(var(--c-accent));font-size:13px;font-weight:600;';
        panel.appendChild(label);

        // 8x8 grid
        const grid = document.createElement('div');
        grid.style.cssText = `
            display:grid;
            grid-template-columns:repeat(8, 1fr);
            gap:2px;
            width:120px;
            height:120px;
            padding:4px;
            background:rgb(var(--c-bg));
            border:1px solid rgb(var(--c-border));
            border-radius:6px;
        `;

        for (let i = 0; i < 64; i++) {
            const cell = document.createElement('div');
            // Vary brightness slightly for visual interest
            const brightness = 25 + Math.floor(Math.random() * 30);
            cell.style.cssText = `background:rgb(var(--c-accent) / ${(brightness / 100).toFixed(2)});border-radius:2px;`;
            grid.appendChild(cell);
        }

        panel.appendChild(grid);

        const dims = document.createElement('div');
        dims.innerHTML = '<strong style="color:rgb(var(--c-ink));">256 &times; 256</strong> = <span style="color:rgb(var(--c-accent));">65,536</span> pixels';
        dims.style.cssText = 'font-size:12px;color:rgb(var(--c-gray-300));text-align:center;';
        panel.appendChild(dims);

        const sub = document.createElement('div');
        sub.textContent = 'Fits in memory easily';
        sub.style.cssText = 'font-size:11px;color:rgb(var(--c-gray-400));';
        panel.appendChild(sub);

        return panel;
    }

    private create3DPanel(): HTMLElement {
        const panel = document.createElement('div');
        panel.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:12px;flex:1;min-width:220px;max-width:300px;';

        const label = document.createElement('div');
        label.textContent = '3D CT Volume';
        label.style.cssText = 'color:rgb(var(--c-accent-2));font-size:13px;font-weight:600;';
        panel.appendChild(label);

        // Isometric stack using CSS transforms
        const stackContainer = document.createElement('div');
        stackContainer.style.cssText = `
            position:relative;
            width:180px;
            height:160px;
            perspective:600px;
        `;

        // Create 4 layers offset to show depth
        const layers = 4;
        for (let i = 0; i < layers; i++) {
            const layer = document.createElement('div');
            const offset = i * 16;
            const opacity = 0.4 + (i / layers) * 0.6;
            layer.style.cssText = `
                position:absolute;
                left:${offset}px;
                top:${(layers - 1 - i) * 16}px;
                width:100px;
                height:100px;
                display:grid;
                grid-template-columns:repeat(6, 1fr);
                gap:1px;
                padding:3px;
                background:rgb(var(--c-bg) / 0.92);
                border:1px solid rgb(var(--c-accent-2) / ${opacity * 0.6});
                border-radius:4px;
                transform:rotateX(15deg) rotateY(-15deg);
                opacity:${opacity};
            `;

            for (let c = 0; c < 36; c++) {
                const cell = document.createElement('div');
                const brightness = 15 + Math.floor(Math.random() * 25);
                cell.style.cssText = `background:rgb(var(--c-accent-2) / ${(brightness / 60).toFixed(2)});border-radius:1px;`;
                layer.appendChild(cell);
            }

            stackContainer.appendChild(layer);
        }

        // Training patch highlight
        const patch = document.createElement('div');
        patch.style.cssText = `
            position:absolute;
            right:8px;
            bottom:8px;
            width:40px;
            height:40px;
            border:2px dashed rgb(var(--c-yellow));
            border-radius:4px;
            display:flex;
            align-items:center;
            justify-content:center;
        `;
        const patchLabel = document.createElement('span');
        patchLabel.textContent = '160\u00B3';
        patchLabel.style.cssText = 'font-size:8px;color:rgb(var(--c-yellow));font-weight:600;';
        patch.appendChild(patchLabel);
        stackContainer.appendChild(patch);

        panel.appendChild(stackContainer);

        const dims = document.createElement('div');
        dims.innerHTML = '<strong style="color:rgb(var(--c-ink));">512 &times; 512 &times; 400</strong> = <span style="color:rgb(var(--c-accent-2));">105M</span> voxels';
        dims.style.cssText = 'font-size:12px;color:rgb(var(--c-gray-300));text-align:center;';
        panel.appendChild(dims);

        const sub = document.createElement('div');
        sub.textContent = 'Must use patch-based training';
        sub.style.cssText = 'font-size:11px;color:rgb(var(--c-gray-400));';
        panel.appendChild(sub);

        const patchNote = document.createElement('div');
        patchNote.innerHTML = '<span style="color:rgb(var(--c-yellow));">&#9634;</span> training patch: 160\u00B3 voxels';
        patchNote.style.cssText = 'font-size:10px;color:rgb(var(--c-gray-400));margin-top:2px;';
        panel.appendChild(patchNote);

        return panel;
    }

    private createBarChart(): HTMLElement {
        const container = document.createElement('div');
        container.style.cssText = `
            width:100%;
            max-width:400px;
            padding:16px;
            background:rgb(var(--c-bg));
            border:1px solid rgb(var(--c-border));
            border-radius:8px;
        `;

        const barTitle = document.createElement('div');
        barTitle.textContent = 'Relative Size (log scale)';
        barTitle.style.cssText = 'color:rgb(var(--c-gray-400));font-size:11px;margin-bottom:12px;text-align:center;';
        container.appendChild(barTitle);

        // 2D bar
        const bar2d = this.createBar('2D (65K)', 3, 'rgb(var(--c-accent))');
        container.appendChild(bar2d);

        // 3D bar
        const bar3d = this.createBar('3D (105M)', 100, 'rgb(var(--c-accent-2))');
        container.appendChild(bar3d);

        // Factor label
        const factor = document.createElement('div');
        factor.textContent = '\u00D71,600x larger';
        factor.style.cssText = 'color:rgb(var(--c-yellow));font-size:11px;text-align:right;margin-top:6px;font-weight:600;';
        container.appendChild(factor);

        return container;
    }

    private createBar(label: string, widthPercent: number, color: string): HTMLElement {
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:center;gap:10px;margin-bottom:8px;';

        const lbl = document.createElement('span');
        lbl.textContent = label;
        lbl.style.cssText = 'color:rgb(var(--c-gray-300));font-size:11px;min-width:70px;text-align:right;font-family:monospace;';

        const barBg = document.createElement('div');
        barBg.style.cssText = 'flex:1;height:20px;background:rgb(var(--c-gray-700));border-radius:4px;overflow:hidden;';

        const barFill = document.createElement('div');
        barFill.style.cssText = `
            width:${widthPercent}%;
            height:100%;
            background:${color};
            border-radius:4px;
            opacity:0.8;
        `;

        barBg.appendChild(barFill);
        row.appendChild(lbl);
        row.appendChild(barBg);

        return row;
    }

    destroy(): void {
        if (this.wrapper) {
            this.wrapper.remove();
            this.wrapper = null;
        }
    }
}
