/**
 * A three-dimensional view of a two-dimensional problem.
 *
 * Nothing about the simulation changes here. The planner, the communication
 * graph and the policy are all flat and stay flat; the board is a grid of cells
 * and a robot occupies one of them. The third dimension is decoration — it
 * exists because a hundred robots threading between extruded blocks reads as a
 * warehouse, and a hundred dots on a grid reads as a screensaver.
 *
 * Everything that scales with the fleet is instanced: one draw call for the
 * robots, one for the obstacles, one for the goals, one for the communication
 * links. At two hundred robots that is four draw calls a frame.
 *
 * three.js is imported lazily by `createRenderer3D`, so a reader who never
 * switches to this view never downloads it.
 */

import type * as THREE_NS from 'three';

import type { MapfEnv } from './env';
import type { BoardRenderer } from './types';

type THREE = typeof THREE_NS;

/** Height of an obstacle block, in cells. */
const WALL_HEIGHT = 0.9;
/** Radius of a robot, in cells. */
const ROBOT_RADIUS = 0.36;

/**
 * Resolve a palette token to a three.js colour.
 *
 * `utils/themeColors` does this too, but importing it would pull three into the
 * main bundle for every page — the whole point of loading this module lazily.
 */
function token(name: string, fallback: number): number {
    const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    const [r, g, b] = raw.split(/[\s,/]+/).map(Number);
    if (!raw || [r, g, b].some(c => !Number.isFinite(c))) return fallback;
    return (r << 16) | (g << 8) | b;
}

export async function createRenderer3D(canvas: HTMLCanvasElement): Promise<BoardRenderer> {
    const THREE = await import('three');
    return new Renderer3D(THREE, canvas);
}

class Renderer3D implements BoardRenderer {
    private readonly THREE: THREE;
    private readonly canvas: HTMLCanvasElement;
    private readonly renderer: THREE_NS.WebGLRenderer;
    private readonly scene: THREE_NS.Scene;
    private readonly camera: THREE_NS.PerspectiveCamera;

    private robots: THREE_NS.InstancedMesh | null = null;
    private walls: THREE_NS.InstancedMesh | null = null;
    private goals: THREE_NS.InstancedMesh | null = null;
    private links: THREE_NS.LineSegments | null = null;
    private ground: THREE_NS.Mesh | null = null;
    private grid: THREE_NS.GridHelper | null = null;

    private readonly dummy: THREE_NS.Object3D;
    private readonly colour: THREE_NS.Color;
    private palette = { moving: 0, arrived: 0, stuck: 0, wall: 0, goal: 0, link: 0, ground: 0, grid: 0 };
    private paletteTheme: string | null = null;

    private instanceKey = '';
    private linkVersion = -1;
    private linkFocus = -1;
    private linkPositions: Float32Array = new Float32Array(0);

    /** Camera orbit, driven by dragging. */
    private azimuth = -0.35;
    private elevation = 0.85;
    private board = 1;
    private dragging = false;
    private lastPointer = { x: 0, y: 0 };

    constructor(three: THREE, canvas: HTMLCanvasElement) {
        this.THREE = three;
        this.canvas = canvas;

        this.renderer = new three.WebGLRenderer({ canvas, antialias: true, alpha: false });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

        this.scene = new three.Scene();
        this.camera = new three.PerspectiveCamera(38, 1, 0.5, 4000);

        // Flat shading with one key light and plenty of ambient: the shapes need
        // to read as solid, but a scene of identical blocks does not want drama.
        this.scene.add(new three.AmbientLight(0xffffff, 1.5));
        const key = new three.DirectionalLight(0xffffff, 1.6);
        key.position.set(1, 2.4, 1.2);
        this.scene.add(key);

        this.dummy = new three.Object3D();
        this.colour = new three.Color();

        canvas.style.touchAction = 'none';
        canvas.addEventListener('pointerdown', this.onPointerDown);
        canvas.addEventListener('pointermove', this.onPointerMove);
        canvas.addEventListener('pointerup', this.onPointerUp);
        canvas.addEventListener('pointercancel', this.onPointerUp);

        this.resize();
    }

    // ── interaction ──────────────────────────────────────────────────────

    private onPointerDown = (event: PointerEvent): void => {
        this.dragging = true;
        this.lastPointer = { x: event.clientX, y: event.clientY };
        this.canvas.setPointerCapture(event.pointerId);
    };

    private onPointerMove = (event: PointerEvent): void => {
        if (!this.dragging) return;
        this.azimuth -= (event.clientX - this.lastPointer.x) * 0.006;
        // Clamped short of the horizon and of straight down: past either the
        // board stops being readable.
        this.elevation = Math.min(1.45, Math.max(0.18,
            this.elevation + (event.clientY - this.lastPointer.y) * 0.005));
        this.lastPointer = { x: event.clientX, y: event.clientY };
    };

    private onPointerUp = (event: PointerEvent): void => {
        this.dragging = false;
        if (this.canvas.hasPointerCapture(event.pointerId)) {
            this.canvas.releasePointerCapture(event.pointerId);
        }
    };

    // ── lifecycle ────────────────────────────────────────────────────────

    resize(): void {
        const rect = this.canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        this.renderer.setSize(rect.width, rect.height, false);
        this.camera.aspect = rect.width / rect.height;
        this.camera.updateProjectionMatrix();
    }

    invalidate(): void {
        this.instanceKey = '';
        this.linkVersion = -1;
    }

    dispose(): void {
        this.canvas.removeEventListener('pointerdown', this.onPointerDown);
        this.canvas.removeEventListener('pointermove', this.onPointerMove);
        this.canvas.removeEventListener('pointerup', this.onPointerUp);
        this.canvas.removeEventListener('pointercancel', this.onPointerUp);
        this.scene.traverse(object => {
            const mesh = object as THREE_NS.Mesh;
            mesh.geometry?.dispose();
            const material = mesh.material as THREE_NS.Material | THREE_NS.Material[] | undefined;
            if (Array.isArray(material)) material.forEach(m => m.dispose());
            else material?.dispose();
        });
        this.renderer.dispose();
    }

    // ── drawing ──────────────────────────────────────────────────────────

    draw(env: MapfEnv, alpha: number, focus: number): void {
        this.readPalette();
        this.build(env);
        this.placeRobots(env, alpha);
        this.placeLinks(env, alpha, focus);
        this.aimCamera(env);
        this.renderer.render(this.scene, this.camera);
    }

    private readPalette(): void {
        const theme = document.documentElement.getAttribute('data-theme');
        if (theme === this.paletteTheme) return;
        this.paletteTheme = theme;
        this.palette = {
            moving: token('--c-accent', 0x0b57d0),
            arrived: token('--c-green', 0x15803d),
            stuck: token('--c-amber', 0xb45309),
            wall: token('--c-gray-500', 0x5b7899),
            goal: token('--c-accent-2', 0x123a94),
            link: token('--c-accent', 0x0b57d0),
            ground: token('--c-bg', 0xeaf1fb),
            grid: token('--c-accent', 0x0b57d0),
        };
        this.scene.background = new this.THREE.Color(this.palette.ground);
        this.instanceKey = '';
    }

    /**
     * Rebuild the instanced meshes when the instance changes.
     *
     * Keyed on board, fleet and obstacle layout, so it runs once per episode
     * rather than once per frame.
     */
    private build(env: MapfEnv): void {
        const key = `${env.board}|${env.numAgents}|${env.instanceId}|${this.paletteTheme}`;
        if (key === this.instanceKey) return;
        this.instanceKey = key;
        this.board = env.board;

        const T = this.THREE;
        for (const mesh of [this.robots, this.walls, this.goals]) {
            if (mesh) { this.scene.remove(mesh); mesh.geometry.dispose(); (mesh.material as THREE_NS.Material).dispose(); }
        }
        if (this.ground) { this.scene.remove(this.ground); this.ground.geometry.dispose(); }
        if (this.grid) { this.scene.remove(this.grid); this.grid.dispose(); }

        const size = env.board;

        this.ground = new T.Mesh(
            new T.PlaneGeometry(size, size),
            new T.MeshBasicMaterial({ color: this.palette.ground })
        );
        this.ground.rotation.x = -Math.PI / 2;
        this.ground.position.set(size / 2, -0.01, size / 2);
        this.scene.add(this.ground);

        this.grid = new T.GridHelper(size, size, this.palette.grid, this.palette.grid);
        this.grid.position.set(size / 2, 0, size / 2);
        (this.grid.material as THREE_NS.Material).opacity = 0.22;
        (this.grid.material as THREE_NS.Material).transparent = true;
        this.scene.add(this.grid);

        // Obstacles: fixed for the episode, so their matrices are written once.
        const cells: number[] = [];
        for (let cell = 0; cell < env.obstacles.length; cell++) if (env.obstacles[cell]) cells.push(cell);
        this.walls = new T.InstancedMesh(
            new T.BoxGeometry(0.86, WALL_HEIGHT, 0.86),
            new T.MeshLambertMaterial({ color: this.palette.wall }),
            Math.max(1, cells.length)
        );
        cells.forEach((cell, i) => {
            const x = cell % env.board;
            const y = Math.floor(cell / env.board);
            this.dummy.position.set(x + 0.5, WALL_HEIGHT / 2, this.depth(y));
            this.dummy.rotation.set(0, 0, 0);
            this.dummy.scale.setScalar(1);
            this.dummy.updateMatrix();
            this.walls!.setMatrixAt(i, this.dummy.matrix);
        });
        this.walls.count = cells.length;
        this.walls.instanceMatrix.needsUpdate = true;
        this.scene.add(this.walls);

        // Goals: flat discs on the floor, also fixed for the episode.
        this.goals = new T.InstancedMesh(
            new T.TorusGeometry(0.34, 0.07, 6, 14),
            new T.MeshBasicMaterial({ color: this.palette.goal, transparent: true, opacity: 0.75 }),
            env.numAgents
        );
        for (let i = 0; i < env.numAgents; i++) {
            // Rings laid flat on the floor: a filled disc at this size reads as
            // a robot lying down.
            this.dummy.position.set(env.goalX[i] + 0.5, 0.02, this.depth(env.goalY[i]));
            this.dummy.rotation.set(-Math.PI / 2, 0, 0);
            this.dummy.scale.setScalar(1);
            this.dummy.updateMatrix();
            this.goals.setMatrixAt(i, this.dummy.matrix);
        }
        this.goals.instanceMatrix.needsUpdate = true;
        this.scene.add(this.goals);

        // No vertexColors here. An InstancedMesh gets its per-instance tint from
        // `instanceColor`, which three wires up on the first setColorAt; asking
        // for vertexColors as well makes the shader look for a colour attribute
        // on the geometry, which does not exist, and every robot renders black.
        this.robots = new T.InstancedMesh(
            new T.SphereGeometry(ROBOT_RADIUS, 12, 8),
            new T.MeshLambertMaterial({ color: 0xffffff }),
            env.numAgents
        );
        this.robots.instanceMatrix.setUsage(T.DynamicDrawUsage);
        this.scene.add(this.robots);

        if (this.links) { this.scene.remove(this.links); this.links.geometry.dispose(); }
        this.linkPositions = new Float32Array(0);
        this.links = new T.LineSegments(
            new T.BufferGeometry(),
            new T.LineBasicMaterial({ color: this.palette.link, transparent: true, opacity: 0.4 })
        );
        this.links.frustumCulled = false;
        this.scene.add(this.links);
        this.linkVersion = -1;
    }

    /** Board row to world depth, so the view matches the 2D one. */
    private depth(y: number): number {
        return this.board - 0.5 - y;
    }

    private placeRobots(env: MapfEnv, alpha: number): void {
        const robots = this.robots;
        if (!robots) return;

        for (let i = 0; i < env.numAgents; i++) {
            const x = env.prevX[i] + (env.posX[i] - env.prevX[i]) * alpha;
            const y = env.prevY[i] + (env.posY[i] - env.prevY[i]) * alpha;
            const arrived = env.posX[i] === env.goalX[i] && env.posY[i] === env.goalY[i];
            const stuck = !arrived && env.stalled[i] >= 3;

            // Arrived robots shrink and settle: an episode is won only when every
            // robot is home, so the ones still moving are what matters.
            const scale = arrived ? 0.55 : 1;
            this.dummy.position.set(x + 0.5, ROBOT_RADIUS * scale + 0.02, this.board - 0.5 - y);
            this.dummy.rotation.set(0, 0, 0);
            this.dummy.scale.setScalar(scale);
            this.dummy.updateMatrix();
            robots.setMatrixAt(i, this.dummy.matrix);

            this.colour.setHex(arrived ? this.palette.arrived : stuck ? this.palette.stuck : this.palette.moving);
            if (arrived) this.colour.lerp(new this.THREE.Color(this.palette.ground), 0.45);
            robots.setColorAt(i, this.colour);
        }

        robots.instanceMatrix.needsUpdate = true;
        if (robots.instanceColor) robots.instanceColor.needsUpdate = true;
    }

    /**
     * Communication links, rebuilt only when the graph changes.
     *
     * The endpoints still move every frame while robots slide between cells, so
     * the buffer is rewritten each draw — but the *edge list* is only recomputed
     * when the environment steps.
     */
    private placeLinks(env: MapfEnv, alpha: number, focus: number): void {
        const links = this.links;
        if (!links) return;
        const n = env.numAgents;
        const single = focus >= 0 && focus < n;

        if (env.version !== this.linkVersion || focus !== this.linkFocus) {
            this.linkVersion = env.version;
            this.linkFocus = focus;
            const pairs: number[] = [];
            for (let i = 0; i < n; i++) {
                const row = i * n;
                for (let j = i + 1; j < n; j++) {
                    if (env.adjacency[row + j] === 0 && env.adjacency[j * n + i] === 0) continue;
                    if (single && i !== focus && j !== focus) continue;
                    pairs.push(i, j);
                }
            }
            this.pairs = Int32Array.from(pairs);
            if (this.linkPositions.length < pairs.length * 3) {
                this.linkPositions = new Float32Array(pairs.length * 3);
                links.geometry.setAttribute(
                    'position', new this.THREE.BufferAttribute(this.linkPositions, 3)
                );
            }
        }

        const at = (i: number) => {
            const x = env.prevX[i] + (env.posX[i] - env.prevX[i]) * alpha + 0.5;
            const y = env.prevY[i] + (env.posY[i] - env.prevY[i]) * alpha;
            return [x, ROBOT_RADIUS + 0.02, this.board - 0.5 - y];
        };

        for (let e = 0; e < this.pairs.length; e++) {
            const [x, h, z] = at(this.pairs[e]);
            this.linkPositions[e * 3] = x;
            this.linkPositions[e * 3 + 1] = h;
            this.linkPositions[e * 3 + 2] = z;
        }
        const attribute = links.geometry.getAttribute('position');
        if (attribute) {
            attribute.needsUpdate = true;
            links.geometry.setDrawRange(0, this.pairs.length);
        }
    }

    private pairs: Int32Array = new Int32Array(0);

    /** Frame the whole board, from wherever the reader has dragged the camera. */
    private aimCamera(env: MapfEnv): void {
        const size = env.board;
        const distance = size * 1.35;
        const centre = size / 2;
        this.camera.position.set(
            centre + Math.sin(this.azimuth) * Math.cos(this.elevation) * distance,
            Math.sin(this.elevation) * distance,
            centre + Math.cos(this.azimuth) * Math.cos(this.elevation) * distance
        );
        this.camera.lookAt(centre, 0, centre);
    }
}
