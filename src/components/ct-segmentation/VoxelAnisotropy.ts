/**
 * VoxelAnisotropy - Visualization comparing isotropic vs anisotropic voxels
 * using CSS 3D transforms to render cube/rectangular prism shapes.
 */

export interface VoxelAnisotropyConfig {
    containerId: string;
}

export class VoxelAnisotropy {
    private container: HTMLElement;
    private wrapper: HTMLElement | null = null;

    constructor(config: VoxelAnisotropyConfig) {
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
        wrapper.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:24px;padding:24px;';

        // Title
        const title = document.createElement('h4');
        title.textContent = 'Voxel Spacing: Isotropic vs Anisotropic';
        title.style.cssText = 'margin:0;color:#ffffff;font-size:14px;font-weight:600;letter-spacing:0.5px;';
        wrapper.appendChild(title);

        // Cubes container
        const cubesRow = document.createElement('div');
        cubesRow.style.cssText = 'display:flex;gap:60px;flex-wrap:wrap;justify-content:center;width:100%;';

        cubesRow.appendChild(this.createCube('Isotropic', '1 \u00D7 1 \u00D7 1 mm', 80, 80, 80, '#00d4ff'));
        cubesRow.appendChild(this.createCube('Anisotropic', '0.8 \u00D7 0.8 \u00D7 2.5 mm', 64, 64, 140, '#a855f7'));

        wrapper.appendChild(cubesRow);

        // Explanation note
        const note = document.createElement('div');
        note.style.cssText = `
            max-width:480px;
            padding:12px 16px;
            background:#0d0d14;
            border:1px solid #2e2e4a;
            border-left:3px solid #eab308;
            border-radius:6px;
            color:#d1d5db;
            font-size:12px;
            line-height:1.6;
        `;
        note.textContent = 'Convolutions, distance transforms, and loss functions must account for this spacing difference. A 3\u00D73\u00D73 kernel covers different physical volumes depending on voxel size.';
        wrapper.appendChild(note);

        this.container.appendChild(wrapper);
        this.wrapper = wrapper;
    }

    private createCube(
        title: string,
        dimensions: string,
        width: number,
        depth: number,
        height: number,
        color: string
    ): HTMLElement {
        const panel = document.createElement('div');
        panel.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:12px;';

        const label = document.createElement('div');
        label.textContent = title;
        label.style.cssText = `color:${color};font-size:13px;font-weight:600;`;
        panel.appendChild(label);

        // 3D cube scene
        const scene = document.createElement('div');
        scene.style.cssText = `
            width:160px;
            height:160px;
            perspective:400px;
            display:flex;
            align-items:center;
            justify-content:center;
        `;

        const cube = document.createElement('div');
        cube.style.cssText = `
            position:relative;
            width:${width}px;
            height:${height}px;
            transform-style:preserve-3d;
            transform:rotateX(-20deg) rotateY(30deg);
        `;

        // Front face
        const front = this.createFace(width, height, `translateZ(${depth / 2}px)`, color, 0.3);
        cube.appendChild(front);

        // Back face
        const back = this.createFace(width, height, `translateZ(-${depth / 2}px)`, color, 0.1);
        cube.appendChild(back);

        // Left face
        const left = this.createFace(depth, height, `rotateY(-90deg) translateZ(${width / 2}px)`, color, 0.15);
        cube.appendChild(left);

        // Right face
        const right = this.createFace(depth, height, `rotateY(90deg) translateZ(${width / 2}px)`, color, 0.35);
        cube.appendChild(right);

        // Top face
        const top = this.createFace(width, depth, `rotateX(90deg) translateZ(${height / 2}px)`, color, 0.45);
        cube.appendChild(top);

        // Bottom face
        const bottom = this.createFace(width, depth, `rotateX(-90deg) translateZ(${height / 2}px)`, color, 0.1);
        cube.appendChild(bottom);

        scene.appendChild(cube);
        panel.appendChild(scene);

        // Dimensions label
        const dims = document.createElement('div');
        dims.textContent = dimensions;
        dims.style.cssText = 'color:#d1d5db;font-size:12px;font-family:monospace;';
        panel.appendChild(dims);

        return panel;
    }

    private createFace(
        width: number,
        height: number,
        transform: string,
        color: string,
        opacity: number
    ): HTMLElement {
        const face = document.createElement('div');
        face.style.cssText = `
            position:absolute;
            width:${width}px;
            height:${height}px;
            left:50%;
            top:50%;
            margin-left:-${width / 2}px;
            margin-top:-${height / 2}px;
            background:${color};
            opacity:${opacity};
            border:1px solid ${color};
            transform:${transform};
            backface-visibility:hidden;
        `;
        return face;
    }

    destroy(): void {
        if (this.wrapper) {
            this.wrapper.remove();
            this.wrapper = null;
        }
    }
}
