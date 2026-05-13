/**
 * PipelineDiagram - Publication-quality pipeline figure.
 * arxiv/TikZ-inspired. No animations, no decorative elements.
 */
export interface PipelineDiagramConfig { containerId: string; }

export class PipelineDiagram {
    private container: HTMLElement;
    private wrapper: HTMLElement | null = null;
    private styleEl: HTMLStyleElement | null = null;

    constructor(config: PipelineDiagramConfig) {
        const el = document.getElementById(config.containerId);
        if (!el) throw new Error(`#${config.containerId} not found`);
        this.container = el;
        this.injectStyles();
        this.render();
    }

    private injectStyles(): void {
        const s = document.createElement('style');
        s.textContent = `
.fpipe{max-width:100%;font-family:system-ui,sans-serif}
.fpipe-flow{display:flex;align-items:center;gap:0;padding:8px 0;overflow-x:auto}
.fpipe-box{flex:0 0 auto;border:1px solid #3a4a5a;background:#1a1e24;padding:10px 14px;min-width:110px;text-align:center;position:relative}
.fpipe-box .n{position:absolute;top:2px;left:5px;font-size:9px;color:#00d4ff;font-family:'Courier New',monospace}
.fpipe-box .t{font-size:12px;font-weight:600;color:#e5e5e5;margin-bottom:1px}
.fpipe-box .s{font-size:10px;color:#9ca3af;font-style:italic}
.fpipe-a{flex:0 0 auto;display:flex;flex-direction:column;align-items:center;padding:0 2px}
.fpipe-a .l{width:28px;height:0;border-top:1px solid #666;position:relative}
.fpipe-a .l::after{content:'';position:absolute;right:-1px;top:-4px;border:4px solid transparent;border-left:5px solid #666}
.fpipe-a .al{font-size:8px;color:#666;font-family:'Courier New',monospace;margin-top:2px;white-space:nowrap}
.fpipe-detail{border:1px dashed #3a4a5a;margin-top:14px;padding:10px 14px}
.fpipe-dt{font-size:9px;color:#555;letter-spacing:.5px;text-transform:uppercase;margin-bottom:8px;font-family:'Courier New',monospace}
.fpipe-sf{display:flex;align-items:center;gap:0;flex-wrap:wrap}
.fpipe-sb{border:1px solid #2a3a4a;background:#141820;padding:5px 8px;text-align:center}
.fpipe-sb .st{font-size:10px;color:#e5e5e5;font-weight:500}
.fpipe-sb .sd{font-size:8px;color:#555;font-family:'Courier New',monospace}
.fpipe-sa{padding:0 3px}
.fpipe-sa .l{width:16px;height:0;border-top:1px solid #444;position:relative}
.fpipe-sa .l::after{content:'';position:absolute;right:-1px;top:-3px;border:3px solid transparent;border-left:4px solid #444}
.fpipe-cap{margin-top:10px;font-size:11px;color:#555;line-height:1.5}
.fpipe-cap b{color:#9ca3af;font-weight:600}
@media(max-width:640px){.fpipe-flow{padding:4px 0}.fpipe-box{min-width:90px;padding:8px 10px}}
        `;
        document.head.appendChild(s);
        this.styleEl = s;
    }

    private render(): void {
        const w = document.createElement('div');
        w.className = 'diagram-container';
        const fig = document.createElement('div');
        fig.className = 'fpipe';

        const stages = [
            { n:'', t:'CT Volume', s:'NIfTI input' },
            { n:'1', t:'Anatomical Seg.', s:'TotalSegmentator' },
            { n:'2', t:'ABBC Prediction', s:'nnU-Net ResEncL' },
            { n:'3', t:'Post-processing', s:'CC + FMM + Vote' },
            { n:'', t:'Labels 0-30', s:'per-fragment STL' },
        ];
        const labels = ['','3-class mask','4-class ABBC','seeds',''];

        const flow = document.createElement('div');
        flow.className = 'fpipe-flow';
        stages.forEach((st, i) => {
            const b = document.createElement('div');
            b.className = 'fpipe-box';
            b.innerHTML = `${st.n?`<span class="n">${st.n}</span>`:''}` +
                `<div class="t">${st.t}</div><div class="s">${st.s}</div>`;
            flow.appendChild(b);
            if (i < stages.length-1) {
                const a = document.createElement('div');
                a.className = 'fpipe-a';
                a.innerHTML = `<div class="l"></div>${labels[i+1]?`<div class="al">${labels[i+1]}</div>`:''}`;
                flow.appendChild(a);
            }
        });
        fig.appendChild(flow);

        const detail = document.createElement('div');
        detail.className = 'fpipe-detail';
        detail.innerHTML = '<div class="fpipe-dt">Post-processing detail</div>';
        const subs = [
            {t:'Core voxels',d:'pred == 2'},{t:'CC3D',d:'26-conn'},
            {t:'Seeds',d:'min 50 vox'},{t:'FMM',d:'border=wall'},
            {t:'Instances',d:'Voronoi'},{t:'Anatomy vote',d:'majority'},
            {t:'Labels',d:'0-30'},
        ];
        const sf = document.createElement('div');
        sf.className = 'fpipe-sf';
        subs.forEach((s, i) => {
            const b = document.createElement('div');
            b.className = 'fpipe-sb';
            b.innerHTML = `<div class="st">${s.t}</div><div class="sd">${s.d}</div>`;
            sf.appendChild(b);
            if (i < subs.length-1) {
                const a = document.createElement('div');
                a.className = 'fpipe-sa';
                a.innerHTML = '<div class="l"></div>';
                sf.appendChild(a);
            }
        });
        detail.appendChild(sf);
        fig.appendChild(detail);

        const cap = document.createElement('div');
        cap.className = 'fpipe-cap';
        cap.innerHTML = '<b>Fig. 1.</b> Two-stage pelvic fracture segmentation pipeline. Stage 1 identifies anatomical bone regions using a pretrained TotalSegmentator model. Stage 2 predicts 4 ABBC geometry classes using nnU-Net. Post-processing recovers fragment instances via connected components and Fast Marching Method geodesic expansion.';
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
