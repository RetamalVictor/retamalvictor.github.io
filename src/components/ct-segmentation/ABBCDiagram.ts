/**
 * ABBCDiagram - CSS pixel-grid visualization showing the cross-section
 * of two adjacent bone fragments with ABBC class coloring.
 *
 * Classes: Background, Boundary, Core, Border (fracture contact zone).
 */

export interface ABBCDiagramConfig {
    containerId: string;
}

// 0 = background, 1 = boundary, 2 = core, 3 = border (fracture line)
type CellClass = 0 | 1 | 2 | 3;

export class ABBCDiagram {
    private container: HTMLElement;
    private wrapper: HTMLElement | null = null;

    private static readonly COLORS: Record<CellClass, string> = {
        0: '#1a1a2e',  // background
        1: '#3b82f6',  // boundary (blue)
        2: '#ef4444',  // core (red/orange)
        3: '#eab308',  // border (yellow - fracture contact)
    };

    private static readonly LABELS: Record<CellClass, string> = {
        0: 'Background',
        1: 'Boundary',
        2: 'Core',
        3: 'Border (fracture)',
    };

    // 20 columns x 14 rows grid representing cross-section of two bone fragments
    private static readonly GRID: CellClass[][] = [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0],
        [0,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0],
        [0,0,1,1,1,2,2,1,1,1,0,1,1,1,2,2,1,1,1,0],
        [0,0,1,1,2,2,2,2,1,1,0,1,1,2,2,2,2,1,1,0],
        [0,1,1,1,2,2,2,2,1,1,3,1,1,2,2,2,2,1,1,0],
        [0,1,1,2,2,2,2,2,1,1,3,1,1,2,2,2,2,2,1,0],
        [0,1,1,2,2,2,2,2,1,1,3,1,1,2,2,2,2,2,1,0],
        [0,1,1,1,2,2,2,2,1,1,3,1,1,2,2,2,2,1,1,0],
        [0,0,1,1,2,2,2,1,1,1,3,1,1,1,2,2,1,1,1,0],
        [0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0],
        [0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0],
        [0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ];

    constructor(config: ABBCDiagramConfig) {
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
        wrapper.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:20px;padding:24px;';

        // Title
        const title = document.createElement('h4');
        title.textContent = 'ABBC Cross-Section: Two Adjacent Bone Fragments';
        title.style.cssText = 'margin:0;color:#ffffff;font-size:14px;font-weight:600;letter-spacing:0.5px;';
        wrapper.appendChild(title);

        // Grid container
        const gridEl = document.createElement('div');
        gridEl.style.cssText = `
            display:grid;
            grid-template-columns:repeat(20, 1fr);
            gap:2px;
            max-width:440px;
            width:100%;
            border-radius:8px;
            padding:8px;
            background:#0d0d14;
            border:1px solid #2e2e4a;
        `;

        const grid = ABBCDiagram.GRID;
        const colors = ABBCDiagram.COLORS;

        for (let row = 0; row < grid.length; row++) {
            for (let col = 0; col < grid[row].length; col++) {
                const cell = document.createElement('div');
                const cls = grid[row][col];
                cell.style.cssText = `
                    aspect-ratio:1;
                    border-radius:3px;
                    background:${colors[cls]};
                    transition:transform 0.15s;
                `;
                cell.title = ABBCDiagram.LABELS[cls];
                cell.addEventListener('mouseenter', () => { cell.style.transform = 'scale(1.2)'; });
                cell.addEventListener('mouseleave', () => { cell.style.transform = 'scale(1)'; });
                gridEl.appendChild(cell);
            }
        }

        wrapper.appendChild(gridEl);

        // Legend
        const legend = document.createElement('div');
        legend.style.cssText = 'display:flex;flex-wrap:wrap;gap:16px;justify-content:center;';

        for (const key of [0, 1, 2, 3] as CellClass[]) {
            const item = document.createElement('div');
            item.style.cssText = 'display:flex;align-items:center;gap:6px;';

            const swatch = document.createElement('div');
            swatch.style.cssText = `
                width:14px;height:14px;border-radius:3px;
                background:${colors[key]};
                border:1px solid ${key === 0 ? '#2e2e4a' : colors[key]};
            `;

            const label = document.createElement('span');
            label.textContent = ABBCDiagram.LABELS[key];
            label.style.cssText = 'font-size:12px;color:#d1d5db;font-family:system-ui,sans-serif;';

            item.appendChild(swatch);
            item.appendChild(label);
            legend.appendChild(item);
        }

        wrapper.appendChild(legend);

        // Description
        const desc = document.createElement('p');
        desc.textContent = 'Each cell represents a voxel classified by its anatomical context: cores lie deep inside each fragment, boundary wraps the outer surface, and border marks the fracture contact zone.';
        desc.style.cssText = 'margin:8px 0 0;color:#9ca3af;font-size:11px;text-align:center;max-width:440px;line-height:1.5;';
        wrapper.appendChild(desc);

        this.container.appendChild(wrapper);
        this.wrapper = wrapper;
    }

    destroy(): void {
        if (this.wrapper) {
            this.wrapper.remove();
            this.wrapper = null;
        }
    }
}
