/**
 * ArchDiagram - Publication-quality U-Net architecture figure.
 * Two-column grid layout. arxiv/TikZ style.
 */
export interface ArchDiagramConfig { containerId: string; }

export class ArchDiagram {
    private container: HTMLElement;
    private wrapper: HTMLElement | null = null;
    private styleEl: HTMLStyleElement | null = null;

    constructor(config: ArchDiagramConfig) {
        const el = document.getElementById(config.containerId);
        if (!el) throw new Error(`#${config.containerId} not found`);
        this.container = el;
        this.injectStyles();
        this.render();
    }

    private injectStyles(): void {
        const s = document.createElement('style');
        s.textContent = `
.farch{max-width:720px;margin:0 auto;font-family:system-ui,sans-serif}
.farch-title{text-align:center;margin-bottom:16px}
.farch-title h3{font-size:14px;font-weight:600;color:rgb(var(--c-ink));margin:0}
.farch-title p{font-size:11px;color:rgb(var(--c-gray-500));margin:2px 0 0}
.farch-grid{display:grid;grid-template-columns:1fr 60px 1fr 60px;gap:0;border:1px solid rgb(var(--c-border))}
.farch-hdr{padding:6px 8px;font-size:10px;font-weight:600;color:rgb(var(--c-gray-500));text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid rgb(var(--c-border));background:rgb(var(--c-bg));text-align:center;font-family:'Courier New',monospace}
.farch-enc{padding:8px 10px;border-bottom:1px solid rgb(var(--c-border));background:rgb(var(--c-accent) / 0.06)}
.farch-dec{padding:8px 10px;border-bottom:1px solid rgb(var(--c-border));background:rgb(var(--c-green) / 0.08)}
.farch-skip{display:flex;align-items:center;justify-content:center;border-bottom:1px solid rgb(var(--c-border));background:rgb(var(--c-bg));font-size:10px;color:rgb(var(--c-yellow));font-family:'Courier New',monospace;letter-spacing:1px}
.farch-ds{display:flex;align-items:center;justify-content:center;border-bottom:1px solid rgb(var(--c-border));background:rgb(var(--c-yellow) / 0.08);font-size:9px;color:rgb(var(--c-gray-400));font-family:'Courier New',monospace}
.farch-bn{grid-column:1/-1;padding:8px 10px;background:rgb(var(--c-accent-2) / 0.08);border-bottom:1px solid rgb(var(--c-border));text-align:center}
.farch-out{grid-column:3;padding:8px 10px;background:rgb(var(--c-bg));text-align:center}
.farch-op{font-size:9px;color:rgb(var(--c-gray-500));font-family:'Courier New',monospace;text-align:center;padding:3px 0;border-bottom:1px solid rgb(var(--c-border));background:rgb(var(--c-bg))}
.farch-op.enc-op{grid-column:1}
.farch-op.dec-op{grid-column:3}
.farch-op.skip-op{grid-column:2}
.farch-op.ds-op{grid-column:4}
.farch-cell-t{font-size:11px;font-weight:600;color:rgb(var(--c-ink))}
.farch-cell-d{font-size:9px;color:rgb(var(--c-gray-500));font-family:'Courier New',monospace;margin-top:1px}
.farch-resblock{border:1px dashed rgb(var(--c-border));margin-top:16px;padding:10px 14px}
.farch-resblock-title{font-size:9px;color:rgb(var(--c-gray-500));text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;font-family:'Courier New',monospace}
.farch-rb-flow{display:flex;align-items:center;gap:0;flex-wrap:wrap}
.farch-rb-box{border:1px solid rgb(var(--c-border));padding:4px 8px;font-size:9px;color:rgb(var(--c-ink));background:rgb(var(--c-accent) / 0.06);font-family:'Courier New',monospace;white-space:nowrap}
.farch-rb-arr{padding:0 4px;font-size:10px;color:rgb(var(--c-gray-500))}
.farch-rb-add{width:20px;height:20px;border:1px solid rgb(var(--c-border));border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;color:rgb(var(--c-ink));background:rgb(var(--c-bg));position:relative}
.farch-rb-skip{font-size:8px;color:rgb(var(--c-yellow));text-align:center;margin-top:4px}
.farch-cap{margin-top:10px;font-size:11px;color:rgb(var(--c-gray-500));line-height:1.5}
.farch-cap b{color:rgb(var(--c-gray-400));font-weight:600}
@media(max-width:500px){.farch-grid{grid-template-columns:1fr 40px 1fr 40px}.farch-cell-t{font-size:10px}.farch-cell-d{font-size:8px}}
        `;
        document.head.appendChild(s);
        this.styleEl = s;
    }

    private render(): void {
        const w = document.createElement('div');
        w.className = 'diagram-container';
        const fig = document.createElement('div');
        fig.className = 'farch';

        // Title
        const title = document.createElement('div');
        title.className = 'farch-title';
        title.innerHTML = '<h3>ResEncL U-Net Architecture</h3><p>4-class ABBC output (background, boundary, core, border)</p>';
        fig.appendChild(title);

        // Grid
        const grid = document.createElement('div');
        grid.className = 'farch-grid';

        // Headers
        grid.innerHTML = `
<div class="farch-hdr">Encoder</div>
<div class="farch-hdr">Skip</div>
<div class="farch-hdr">Decoder</div>
<div class="farch-hdr">DS</div>
`;

        const levels = [
            { ch: 32, res: 'D x H x W', ds: '' },
            { ch: 64, res: 'D/2 x H/2 x W/2', ds: 'w=0.5' },
            { ch: 128, res: 'D/4 x H/4 x W/4', ds: 'w=0.25' },
            { ch: 256, res: 'D/8 x H/8 x W/8', ds: 'w=0.125' },
        ];

        levels.forEach((lv, i) => {
            // Encoder cell
            const enc = document.createElement('div');
            enc.className = 'farch-enc';
            enc.innerHTML = `<div class="farch-cell-t">ResBlock ${lv.ch}ch</div><div class="farch-cell-d">${lv.res}</div>`;
            grid.appendChild(enc);

            // Skip
            const skip = document.createElement('div');
            skip.className = 'farch-skip';
            skip.textContent = '- - ->';
            grid.appendChild(skip);

            // Decoder cell
            const dec = document.createElement('div');
            dec.className = 'farch-dec';
            dec.innerHTML = `<div class="farch-cell-t">ConvBlock ${lv.ch}ch</div><div class="farch-cell-d">${lv.res}</div>`;
            grid.appendChild(dec);

            // Deep supervision
            const ds = document.createElement('div');
            ds.className = 'farch-ds';
            ds.textContent = i === 0 ? '-> out (4ch)' : lv.ds;
            grid.appendChild(ds);

            // Operation row (pool/upsample) between levels
            if (i < levels.length - 1) {
                const encOp = document.createElement('div');
                encOp.className = 'farch-op';
                encOp.textContent = `stride 2`;
                grid.appendChild(encOp);

                const skipOp = document.createElement('div');
                skipOp.className = 'farch-op';
                skipOp.textContent = '';
                grid.appendChild(skipOp);

                const decOp = document.createElement('div');
                decOp.className = 'farch-op';
                decOp.textContent = `upsample 2x`;
                grid.appendChild(decOp);

                const dsOp = document.createElement('div');
                dsOp.className = 'farch-op';
                dsOp.textContent = '';
                grid.appendChild(dsOp);
            }
        });

        // Encoder level 5 -> bottleneck
        const encOp5 = document.createElement('div');
        encOp5.className = 'farch-op';
        encOp5.textContent = 'stride 2';
        grid.appendChild(encOp5);
        for (let i = 0; i < 3; i++) {
            const empty = document.createElement('div');
            empty.className = 'farch-op';
            grid.appendChild(empty);
        }

        // Bottleneck
        const bn = document.createElement('div');
        bn.className = 'farch-bn';
        bn.innerHTML = '<div class="farch-cell-t">Bottleneck (320ch)</div><div class="farch-cell-d">ResBlock, D/16 x H/16 x W/16</div>';
        grid.appendChild(bn);

        fig.appendChild(grid);

        // Residual block detail
        const rb = document.createElement('div');
        rb.className = 'farch-resblock';
        rb.innerHTML = '<div class="farch-resblock-title">Residual encoder block detail</div>';

        const rbFlow = document.createElement('div');
        rbFlow.className = 'farch-rb-flow';
        const parts = [
            { type: 'box', text: 'x' },
            { type: 'arr', text: '->' },
            { type: 'box', text: 'Conv3D, IN, ReLU' },
            { type: 'arr', text: '->' },
            { type: 'box', text: 'Conv3D, IN' },
            { type: 'arr', text: '->' },
            { type: 'add', text: '+' },
            { type: 'arr', text: '->' },
            { type: 'box', text: 'ReLU' },
            { type: 'arr', text: '->' },
            { type: 'box', text: 'out' },
        ];
        parts.forEach(p => {
            if (p.type === 'box') {
                const el = document.createElement('div');
                el.className = 'farch-rb-box';
                el.textContent = p.text;
                rbFlow.appendChild(el);
            } else if (p.type === 'arr') {
                const el = document.createElement('div');
                el.className = 'farch-rb-arr';
                el.textContent = p.text;
                rbFlow.appendChild(el);
            } else {
                const el = document.createElement('div');
                el.className = 'farch-rb-add';
                el.textContent = p.text;
                rbFlow.appendChild(el);
            }
        });
        rb.appendChild(rbFlow);

        const skipNote = document.createElement('div');
        skipNote.className = 'farch-rb-skip';
        skipNote.textContent = 'identity shortcut: x added to output before final ReLU';
        rb.appendChild(skipNote);

        fig.appendChild(rb);

        // Caption
        const cap = document.createElement('div');
        cap.className = 'farch-cap';
        cap.innerHTML = '<b>Fig. 2.</b> ResEncL U-Net architecture for ABBC prediction. The encoder uses residual blocks with instance normalization and identity shortcuts. Skip connections concatenate encoder features with upsampled decoder features at each resolution. Deep supervision heads at decoder stages 2-4 compute auxiliary losses with decreasing weights. The network outputs 4 classes at full input resolution.';
        fig.appendChild(cap);

        w.appendChild(fig);
        this.container.appendChild(w);
        this.wrapper = w;
    }

    destroy(): void {
        this.wrapper?.remove(); this.wrapper = null;
        this.styleEl?.remove(); this.styleEl = null;
    }
}
