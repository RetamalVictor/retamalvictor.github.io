/**
 * PelvisAnatomy - Stylized top-down schematic of the pelvis ring showing
 * three bone regions (Sacrum, Left Hip, Right Hip) with fragment labeling.
 */

export interface PelvisAnatomyConfig {
    containerId: string;
}

export class PelvisAnatomy {
    private container: HTMLElement;
    private wrapper: HTMLElement | null = null;

    private static readonly REGIONS = {
        sacrum: { color: '#e05050', label: 'Sacrum', fragments: '1\u201310' },
        leftHip: { color: '#50c878', label: 'Left Hip', fragments: '11\u201320' },
        rightHip: { color: '#5070e0', label: 'Right Hip', fragments: '21\u201330' },
    };

    constructor(config: PelvisAnatomyConfig) {
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
        title.textContent = 'Pelvic Ring Anatomy & Fragment Labeling';
        title.style.cssText = 'margin:0;color:#ffffff;font-size:14px;font-weight:600;letter-spacing:0.5px;';
        wrapper.appendChild(title);

        // SVG diagram
        const svgNS = 'http://www.w3.org/2000/svg';
        const svg = document.createElementNS(svgNS, 'svg');
        svg.setAttribute('viewBox', '0 0 400 320');
        svg.setAttribute('width', '100%');
        svg.style.cssText = 'max-width:400px;display:block;';

        const R = PelvisAnatomy.REGIONS;

        svg.innerHTML = `
            <!-- Connection lines (ring structure) -->
            <path d="M 200 60 Q 100 100 100 180" stroke="${R.sacrum.color}" stroke-width="2" fill="none" opacity="0.5"/>
            <path d="M 200 60 Q 300 100 300 180" stroke="${R.sacrum.color}" stroke-width="2" fill="none" opacity="0.5"/>
            <path d="M 100 180 Q 120 250 180 270" stroke="${R.leftHip.color}" stroke-width="2" fill="none" opacity="0.5"/>
            <path d="M 300 180 Q 280 250 220 270" stroke="${R.rightHip.color}" stroke-width="2" fill="none" opacity="0.5"/>
            <path d="M 180 270 L 220 270" stroke="#666666" stroke-width="1.5" stroke-dasharray="4 3" fill="none" opacity="0.5"/>

            <!-- Sacrum (top center) -->
            <ellipse cx="200" cy="55" rx="60" ry="35"
                fill="${R.sacrum.color}" fill-opacity="0.2"
                stroke="${R.sacrum.color}" stroke-width="2"/>
            <text x="200" y="52" text-anchor="middle" font-family="system-ui,sans-serif"
                font-size="12" font-weight="600" fill="${R.sacrum.color}">${R.sacrum.label}</text>
            <text x="200" y="68" text-anchor="middle" font-family="monospace"
                font-size="10" fill="#d1d5db">fragments ${R.sacrum.fragments}</text>

            <!-- Left Hip (bottom-left) -->
            <ellipse cx="110" cy="190" rx="50" ry="65"
                fill="${R.leftHip.color}" fill-opacity="0.15"
                stroke="${R.leftHip.color}" stroke-width="2"/>
            <text x="110" y="187" text-anchor="middle" font-family="system-ui,sans-serif"
                font-size="12" font-weight="600" fill="${R.leftHip.color}">${R.leftHip.label}</text>
            <text x="110" y="203" text-anchor="middle" font-family="monospace"
                font-size="10" fill="#d1d5db">fragments ${R.leftHip.fragments}</text>

            <!-- Right Hip (bottom-right) -->
            <ellipse cx="290" cy="190" rx="50" ry="65"
                fill="${R.rightHip.color}" fill-opacity="0.15"
                stroke="${R.rightHip.color}" stroke-width="2"/>
            <text x="290" y="187" text-anchor="middle" font-family="system-ui,sans-serif"
                font-size="12" font-weight="600" fill="${R.rightHip.color}">${R.rightHip.label}</text>
            <text x="290" y="203" text-anchor="middle" font-family="monospace"
                font-size="10" fill="#d1d5db">fragments ${R.rightHip.fragments}</text>

            <!-- Pubic symphysis connection at bottom -->
            <ellipse cx="200" cy="270" rx="25" ry="12"
                fill="none" stroke="#666666" stroke-width="1" stroke-dasharray="4 3"/>
            <text x="200" y="295" text-anchor="middle" font-family="system-ui,sans-serif"
                font-size="9" fill="#9ca3af">pubic symphysis</text>

            <!-- SI joint labels -->
            <text x="145" y="110" text-anchor="middle" font-family="system-ui,sans-serif"
                font-size="9" fill="#9ca3af">SI joint</text>
            <text x="255" y="110" text-anchor="middle" font-family="system-ui,sans-serif"
                font-size="9" fill="#9ca3af">SI joint</text>
        `;

        wrapper.appendChild(svg);

        // Legend / description
        const legend = document.createElement('div');
        legend.style.cssText = 'display:flex;gap:20px;flex-wrap:wrap;justify-content:center;';

        for (const [, region] of Object.entries(R)) {
            const item = document.createElement('div');
            item.style.cssText = 'display:flex;align-items:center;gap:6px;';

            const swatch = document.createElement('div');
            swatch.style.cssText = `width:12px;height:12px;border-radius:50%;background:${region.color};`;

            const text = document.createElement('span');
            text.textContent = `${region.label} (${region.fragments})`;
            text.style.cssText = 'font-size:11px;color:#d1d5db;';

            item.appendChild(swatch);
            item.appendChild(text);
            legend.appendChild(item);
        }

        wrapper.appendChild(legend);

        // Note
        const note = document.createElement('p');
        note.textContent = 'The pelvis forms a bony ring. Each region is independently labeled with up to 10 fragment IDs, yielding a 30-class semantic segmentation problem plus background.';
        note.style.cssText = 'margin:4px 0 0;color:#9ca3af;font-size:11px;text-align:center;max-width:400px;line-height:1.5;';
        wrapper.appendChild(note);

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
