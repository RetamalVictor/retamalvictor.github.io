/**
 * STLViewer - 3D viewer for pelvic fracture segmentation results.
 * Layout: main panel (70%) + 3 stacked side panels (sacrum, L.hip, R.hip).
 * Each fragment gets a unique color. All 4 views share one animation loop.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { onThemeChange, themeColor } from '../../utils/themeColors.js';

export interface STLViewerConfig {
    containerId: string;
}

interface CaseDef {
    id: string;
    fragments: string[]; // filenames without .stl
}

interface ViewPanel {
    canvas: HTMLCanvasElement;
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    meshes: THREE.Mesh[];
}

const FRAGMENT_PALETTE = [
    0xe05050, 0x50c878, 0x5070e0, 0xf5a623, 0xa855f7,
    0x00d4ff, 0xf472b6, 0x84cc16, 0x06b6d4, 0xfbbf24,
    0xec4899, 0x22d3ee,
];

const CASES: CaseDef[] = [
    { id: '001', fragments: ['left_hip_01','left_hip_02','right_hip_01','sacrum_01'] },
    { id: '002', fragments: ['left_hip_01','right_hip_01','right_hip_02','right_hip_03','right_hip_04','sacrum_01','sacrum_02','sacrum_03'] },
    { id: '003', fragments: ['left_hip_01','left_hip_02','left_hip_03','left_hip_04','right_hip_01','sacrum_01'] },
    { id: '004', fragments: ['left_hip_01','left_hip_02','right_hip_01','sacrum_01','sacrum_02'] },
    { id: '005', fragments: ['left_hip_01','left_hip_02','right_hip_01','right_hip_02','right_hip_03','sacrum_01'] },
];

function boneType(name: string): 'sacrum' | 'left_hip' | 'right_hip' {
    if (name.startsWith('sacrum')) return 'sacrum';
    if (name.startsWith('left_hip')) return 'left_hip';
    return 'right_hip';
}

export class STLViewer {
    private container: HTMLElement;
    private wrapper: HTMLElement | null = null;
    private styleEl: HTMLStyleElement | null = null;
    private mainPanel: ViewPanel | null = null;
    private sidePanels: Map<string, ViewPanel> = new Map();
    private activeCaseIdx = 0;
    private animId = 0;
    private unsubscribeTheme: (() => void) | null = null;
    private observer: IntersectionObserver | null = null;
    private visible = true;
    private abortCtrl: AbortController | null = null;
    private loader = new STLLoader();

    constructor(config: STLViewerConfig) {
        const el = document.getElementById(config.containerId);
        if (!el) throw new Error(`Container #${config.containerId} not found`);
        this.container = el;
        this.injectStyles();
        this.buildUI();
        this.setupVisibility();
        this.loadCase(0);
    }

    private injectStyles(): void {
        const style = document.createElement('style');
        style.textContent = `
            .stl-root { position:relative; width:100%; height:550px; background:#0a0a0f; border:1px solid #1e1e2e; border-radius:6px; overflow:hidden; display:grid; grid-template-columns:1fr 280px; grid-template-rows:1fr 1fr 1fr 40px; }
            @media(max-width:768px) { .stl-root { height:400px; grid-template-columns:1fr; grid-template-rows:300px 100px 100px 100px 36px; } }
            .stl-main { grid-column:1; grid-row:1/4; position:relative; border-right:1px solid #1e1e2e; }
            @media(max-width:768px) { .stl-main { grid-column:1; grid-row:1; border-right:none; border-bottom:1px solid #1e1e2e; } }
            .stl-side { position:relative; border-bottom:1px solid #1e1e2e; overflow:hidden; }
            .stl-side:last-of-type { border-bottom:none; }
            .stl-side-label { position:absolute; top:4px; left:8px; font-size:10px; color:#9ca3af; z-index:2; font-family:system-ui,sans-serif; pointer-events:none; }
            .stl-tabs { grid-column:1/-1; display:flex; align-items:center; gap:2px; padding:0 8px; background:#12121a; border-top:1px solid #1e1e2e; }
            .stl-tab { padding:4px 14px; font-size:11px; color:#888; background:transparent; border:1px solid transparent; border-radius:3px; cursor:pointer; font-family:system-ui,sans-serif; transition:color .15s,border-color .15s; }
            .stl-tab:hover { color:#d1d5db; }
            .stl-tab.active { color:#00d4ff; border-color:#00d4ff; }
            .stl-progress { position:absolute; bottom:0; left:0; height:2px; background:#00d4ff; transition:width .2s; z-index:3; }
            .stl-canvas { width:100%; height:100%; display:block; }
        `;
        document.head.appendChild(style);
        this.styleEl = style;
    }

    private buildUI(): void {
        const root = document.createElement('div');
        root.className = 'stl-root';

        // Main panel
        const mainDiv = document.createElement('div');
        mainDiv.className = 'stl-main';
        const mainCanvas = document.createElement('canvas');
        mainCanvas.className = 'stl-canvas';
        mainDiv.appendChild(mainCanvas);
        const mainProgress = document.createElement('div');
        mainProgress.className = 'stl-progress';
        mainProgress.style.width = '0%';
        mainDiv.appendChild(mainProgress);
        root.appendChild(mainDiv);

        // Side panels
        const boneLabels = [
            { key: 'sacrum', label: 'Sacrum' },
            { key: 'left_hip', label: 'Left Hip' },
            { key: 'right_hip', label: 'Right Hip' },
        ];
        for (const { key, label } of boneLabels) {
            const sideDiv = document.createElement('div');
            sideDiv.className = 'stl-side';
            sideDiv.dataset.bone = key;
            const lbl = document.createElement('div');
            lbl.className = 'stl-side-label';
            lbl.textContent = label;
            lbl.dataset.boneLabel = key;
            sideDiv.appendChild(lbl);
            const canvas = document.createElement('canvas');
            canvas.className = 'stl-canvas';
            sideDiv.appendChild(canvas);
            root.appendChild(sideDiv);
        }

        // Tabs
        const tabBar = document.createElement('div');
        tabBar.className = 'stl-tabs';
        CASES.forEach((c, i) => {
            const tab = document.createElement('button');
            tab.className = 'stl-tab' + (i === 0 ? ' active' : '');
            tab.textContent = `Case ${c.id}`;
            tab.addEventListener('click', () => this.switchCase(i));
            tabBar.appendChild(tab);
        });
        root.appendChild(tabBar);

        this.container.appendChild(root);
        this.wrapper = root;

        // Init renderers
        this.mainPanel = this.createPanel(mainCanvas);
        const sideDivs = root.querySelectorAll<HTMLElement>('.stl-side');
        const bones = ['sacrum', 'left_hip', 'right_hip'];
        sideDivs.forEach((div, i) => {
            const canvas = div.querySelector<HTMLCanvasElement>('.stl-canvas')!;
            this.sidePanels.set(bones[i], this.createPanel(canvas));
        });

        this.startRenderLoop();

        // Every panel shares the page background
        this.unsubscribeTheme = onThemeChange(() => {
            const background = themeColor('--c-scene-bg');
            this.mainPanel?.renderer.setClearColor(background);
            for (const panel of this.sidePanels.values()) panel.renderer.setClearColor(background);
        });
    }

    private createPanel(canvas: HTMLCanvasElement): ViewPanel {
        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setClearColor(themeColor('--c-scene-bg'));

        const scene = new THREE.Scene();
        const ambient = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambient);
        const dir1 = new THREE.DirectionalLight(0xffffff, 0.8);
        dir1.position.set(1, 2, 3);
        scene.add(dir1);
        const dir2 = new THREE.DirectionalLight(0xffffff, 0.3);
        dir2.position.set(-2, -1, -1);
        scene.add(dir2);

        const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10000);
        camera.position.set(0, 0, 300);

        const controls = new OrbitControls(camera, canvas);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.enablePan = true;
        controls.minDistance = 50;
        controls.maxDistance = 1000;

        return { canvas, renderer, scene, camera, controls, meshes: [] };
    }

    private resizePanel(panel: ViewPanel): void {
        const w = panel.canvas.clientWidth;
        const h = panel.canvas.clientHeight;
        if (w === 0 || h === 0) return;
        if (panel.canvas.width !== w || panel.canvas.height !== h) {
            panel.renderer.setSize(w, h, false);
            panel.camera.aspect = w / h;
            panel.camera.updateProjectionMatrix();
        }
    }

    private startRenderLoop(): void {
        const render = () => {
            this.animId = requestAnimationFrame(render);
            if (!this.visible) return;
            if (this.mainPanel) {
                this.resizePanel(this.mainPanel);
                this.mainPanel.controls.update();
                this.mainPanel.renderer.render(this.mainPanel.scene, this.mainPanel.camera);
            }
            for (const panel of this.sidePanels.values()) {
                this.resizePanel(panel);
                panel.controls.update();
                panel.renderer.render(panel.scene, panel.camera);
            }
        };
        render();
    }

    private setupVisibility(): void {
        this.observer = new IntersectionObserver(([entry]) => {
            this.visible = entry.isIntersecting;
        }, { threshold: 0.05 });
        this.observer.observe(this.container);
    }

    private switchCase(idx: number): void {
        if (idx === this.activeCaseIdx) return;
        this.activeCaseIdx = idx;
        // Update tabs
        this.wrapper?.querySelectorAll('.stl-tab').forEach((t, i) => {
            t.classList.toggle('active', i === idx);
        });
        this.loadCase(idx);
    }

    private async loadCase(idx: number): Promise<void> {
        // Abort any in-flight loads
        this.abortCtrl?.abort();
        this.abortCtrl = new AbortController();
        const signal = this.abortCtrl.signal;

        // Clear existing meshes
        this.clearMeshes();

        const caseDef = CASES[idx];
        const progressBar = this.wrapper?.querySelector<HTMLElement>('.stl-progress');
        if (progressBar) progressBar.style.width = '0%';

        const total = caseDef.fragments.length;
        let loaded = 0;

        const geometries: { name: string; geom: THREE.BufferGeometry; color: number }[] = [];

        for (const fragName of caseDef.fragments) {
            if (signal.aborted) return;
            const url = `/assets/models/pelvis/${caseDef.id}/${fragName}.stl`;
            const colorIdx = caseDef.fragments.indexOf(fragName);
            const color = FRAGMENT_PALETTE[colorIdx % FRAGMENT_PALETTE.length];
            try {
                const geom = await this.loadSTL(url);
                if (signal.aborted) { geom.dispose(); return; }
                geometries.push({ name: fragName, geom, color });
                loaded++;
                if (progressBar) progressBar.style.width = `${(loaded / total) * 100}%`;
            } catch (e) {
                console.warn(`Failed to load ${url}:`, e);
                loaded++;
                if (progressBar) progressBar.style.width = `${(loaded / total) * 100}%`;
            }
        }

        if (signal.aborted) return;

        // Add meshes to scenes
        for (const { name, geom, color } of geometries) {
            const mat = new THREE.MeshPhongMaterial({
                color,
                specular: 0x222222,
                shininess: 30,
                flatShading: false,
            });

            // Main panel
            if (this.mainPanel) {
                const mesh = new THREE.Mesh(geom, mat.clone());
                this.mainPanel.scene.add(mesh);
                this.mainPanel.meshes.push(mesh);
            }

            // Side panel
            const bone = boneType(name);
            const sidePanel = this.sidePanels.get(bone);
            if (sidePanel) {
                const mesh = new THREE.Mesh(geom.clone(), mat.clone());
                sidePanel.scene.add(mesh);
                sidePanel.meshes.push(mesh);
            }
        }

        // Update side panel labels with fragment counts
        const counts: Record<string, number> = { sacrum: 0, left_hip: 0, right_hip: 0 };
        for (const f of caseDef.fragments) counts[boneType(f)]++;
        for (const [bone, count] of Object.entries(counts)) {
            const lbl = this.wrapper?.querySelector<HTMLElement>(`[data-bone-label="${bone}"]`);
            if (lbl) {
                const name = bone === 'sacrum' ? 'Sacrum' : bone === 'left_hip' ? 'Left Hip' : 'Right Hip';
                lbl.textContent = `${name} (${count})`;
            }
        }

        // Frame cameras
        if (this.mainPanel) this.frameCamera(this.mainPanel);
        for (const panel of this.sidePanels.values()) {
            if (panel.meshes.length > 0) this.frameCamera(panel);
        }

        // Hide progress
        setTimeout(() => { if (progressBar) progressBar.style.width = '0%'; }, 600);
    }

    private loadSTL(url: string): Promise<THREE.BufferGeometry> {
        return new Promise((resolve, reject) => {
            this.loader.load(url, resolve, undefined, reject);
        });
    }

    private frameCamera(panel: ViewPanel): void {
        const box = new THREE.Box3();
        for (const m of panel.meshes) box.expandByObject(m);
        if (box.isEmpty()) return;
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const dist = maxDim * 1.5;
        panel.camera.position.set(center.x, center.y, center.z + dist);
        panel.camera.lookAt(center);
        panel.controls.target.copy(center);
        panel.controls.update();
    }

    private clearMeshes(): void {
        const clearPanel = (panel: ViewPanel | null) => {
            if (!panel) return;
            for (const m of panel.meshes) {
                panel.scene.remove(m);
                m.geometry.dispose();
                (m.material as THREE.Material).dispose();
            }
            panel.meshes = [];
        };
        clearPanel(this.mainPanel);
        for (const p of this.sidePanels.values()) clearPanel(p);
    }

    destroy(): void {
        if (this.animId) cancelAnimationFrame(this.animId);
        this.unsubscribeTheme?.();
        this.unsubscribeTheme = null;
        this.abortCtrl?.abort();
        this.observer?.disconnect();
        this.clearMeshes();

        const disposePanel = (p: ViewPanel | null) => {
            if (!p) return;
            p.controls.dispose();
            p.renderer.dispose();
        };
        disposePanel(this.mainPanel);
        for (const p of this.sidePanels.values()) disposePanel(p);
        this.mainPanel = null;
        this.sidePanels.clear();

        this.wrapper?.remove();
        this.wrapper = null;
        this.styleEl?.remove();
        this.styleEl = null;
    }
}
