import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Core components
import { RacingDrone } from './core/RacingDrone';
import { MPC } from './control/MPC';
import { ControlCommand } from './types';

// Trajectory system
import {
    Trajectory,
    TrajectoryType,
    TRAJECTORY_INFO,
    createTrajectory,
} from './trajectory';

/**
 * Drone Racing Demo
 *
 * Interactive demonstration of MPC-controlled drone racing.
 * Features multiple trajectory types that can be selected via UI.
 */
export class DroneRacingDemo {
    // DOM elements
    private container: HTMLElement;
    private canvas: HTMLCanvasElement;

    // Three.js
    private renderer: THREE.WebGLRenderer;
    private scene: THREE.Scene;
    private camera: THREE.PerspectiveCamera;
    private controls: OrbitControls;

    // Core components
    private drone: RacingDrone;
    private mpc: MPC;
    private trajectory: Trajectory;

    // Configuration
    private currentTrajectoryType: TrajectoryType = 'circle';
    private readonly defaultSpeed = 15.0;
    private readonly defaultHeight = 4.0;

    // State
    private simulationTime: number = 0;
    private isRunning: boolean = true;
    private lastFrameTime: number = 0;
    private animationFrameId: number = 0;

    // Visualization
    private trajectoryLine: THREE.Line | null = null;
    private droneTrail: THREE.Line | null = null;
    private trailPositions: THREE.Vector3[] = [];
    private readonly maxTrailLength = 1000;

    // UI elements
    private debugOverlay: HTMLElement | null = null;
    private trajectoryButtons: Map<TrajectoryType, HTMLButtonElement> = new Map();

    constructor(containerId: string) {
        // Get container
        const container = document.getElementById(containerId);
        if (!container) {
            throw new Error(`Container ${containerId} not found`);
        }
        this.container = container;
        this.container.innerHTML = '';
        this.container.style.position = 'relative';

        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.container.appendChild(this.canvas);

        // Initialize Three.js
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true,
        });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setClearColor(0x0a0a0f, 1);

        // Create scene
        this.scene = new THREE.Scene();

        // Create camera
        this.camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
        this.camera.position.set(50, 30, 50);
        this.camera.lookAt(0, 4, 0);

        // Create controls
        this.controls = new OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0, 4, 0);

        // Initialize components
        this.drone = new RacingDrone();
        this.mpc = new MPC();
        this.trajectory = createTrajectory(this.currentTrajectoryType, this.defaultSpeed, this.defaultHeight);

        // Setup scene
        this.setupScene();
        this.setupUI();

        // Reset to start position
        this.resetSimulation();

        // Handle resize
        this.handleResize();
        window.addEventListener('resize', () => this.handleResize());

        // Start animation
        this.animate();
    }

    /**
     * Setup 3D scene
     */
    private setupScene(): void {
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 20, 10);
        this.scene.add(directionalLight);

        // Ground grid - larger for bigger tracks
        const gridHelper = new THREE.GridHelper(100, 50, 0x2a2a3e, 0x1a1a2e);
        this.scene.add(gridHelper);

        // Add drone
        this.scene.add(this.drone.mesh);

        // Create trajectory visualization
        this.updateTrajectoryVisualization();

        // Create drone trail
        this.createDroneTrail();
    }

    /**
     * Setup UI elements
     */
    private setupUI(): void {
        // Main controls container (top-left)
        const controlsDiv = document.createElement('div');
        controlsDiv.style.cssText = `
            position: absolute;
            top: 16px;
            left: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        `;

        // Playback controls
        const playbackDiv = document.createElement('div');
        playbackDiv.style.cssText = 'display: flex; gap: 8px;';

        const resetBtn = this.createButton('Reset', () => this.resetSimulation());
        playbackDiv.appendChild(resetBtn);

        const pauseBtn = this.createButton('Pause', () => {
            this.isRunning = !this.isRunning;
            pauseBtn.textContent = this.isRunning ? 'Pause' : 'Play';
        });
        playbackDiv.appendChild(pauseBtn);

        controlsDiv.appendChild(playbackDiv);

        // Trajectory selection panel
        const trajectoryPanel = document.createElement('div');
        trajectoryPanel.style.cssText = `
            background: rgba(10, 10, 15, 0.9);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 8px;
            padding: 12px;
        `;

        const panelTitle = document.createElement('div');
        panelTitle.textContent = 'Trajectory';
        panelTitle.style.cssText = `
            color: #00d4ff;
            font-family: monospace;
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 10px;
        `;
        trajectoryPanel.appendChild(panelTitle);

        const buttonsContainer = document.createElement('div');
        buttonsContainer.style.cssText = `
            display: flex;
            flex-direction: column;
            gap: 6px;
        `;

        for (const info of TRAJECTORY_INFO) {
            const btn = this.createTrajectoryButton(info.type, info.name, info.description);
            this.trajectoryButtons.set(info.type, btn);
            buttonsContainer.appendChild(btn);
        }

        trajectoryPanel.appendChild(buttonsContainer);
        controlsDiv.appendChild(trajectoryPanel);

        this.container.appendChild(controlsDiv);

        // Debug overlay (top-right)
        this.debugOverlay = document.createElement('div');
        this.debugOverlay.style.cssText = `
            position: absolute;
            top: 16px;
            right: 16px;
            background: rgba(0, 0, 0, 0.85);
            border: 1px solid rgba(0, 212, 255, 0.5);
            border-radius: 8px;
            padding: 12px;
            font-family: monospace;
            font-size: 11px;
            color: #00d4ff;
            max-width: 250px;
            line-height: 1.4;
        `;
        this.container.appendChild(this.debugOverlay);

        // Highlight initial trajectory button
        this.updateTrajectoryButtonStyles();
    }

    /**
     * Create styled button
     */
    private createButton(text: string, onClick: () => void): HTMLButtonElement {
        const btn = document.createElement('button');
        btn.textContent = text;
        btn.style.cssText = `
            padding: 8px 16px;
            background: rgba(10, 10, 15, 0.8);
            border: 1px solid rgba(0, 212, 255, 0.5);
            border-radius: 6px;
            color: #00d4ff;
            font-family: monospace;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        `;
        btn.addEventListener('mouseenter', () => {
            btn.style.background = 'rgba(0, 212, 255, 0.2)';
        });
        btn.addEventListener('mouseleave', () => {
            btn.style.background = 'rgba(10, 10, 15, 0.8)';
        });
        btn.addEventListener('click', onClick);
        return btn;
    }

    /**
     * Create trajectory selection button
     */
    private createTrajectoryButton(type: TrajectoryType, name: string, description: string): HTMLButtonElement {
        const btn = document.createElement('button');
        btn.innerHTML = `<strong>${name}</strong><br><span style="font-size: 9px; opacity: 0.7;">${description}</span>`;
        btn.style.cssText = `
            padding: 8px 12px;
            background: rgba(10, 10, 15, 0.6);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 6px;
            color: #00d4ff;
            font-family: monospace;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s;
            text-align: left;
            width: 180px;
        `;
        btn.addEventListener('click', () => this.selectTrajectory(type));
        return btn;
    }

    /**
     * Select a trajectory type
     */
    private selectTrajectory(type: TrajectoryType): void {
        if (type === this.currentTrajectoryType) return;

        this.currentTrajectoryType = type;
        this.trajectory = createTrajectory(type, this.defaultSpeed, this.defaultHeight);

        this.updateTrajectoryButtonStyles();
        this.updateTrajectoryVisualization();
        this.resetSimulation();
    }

    /**
     * Update trajectory button styles
     */
    private updateTrajectoryButtonStyles(): void {
        for (const [type, btn] of this.trajectoryButtons) {
            if (type === this.currentTrajectoryType) {
                btn.style.background = 'rgba(0, 212, 255, 0.3)';
                btn.style.borderColor = 'rgba(0, 212, 255, 0.8)';
            } else {
                btn.style.background = 'rgba(10, 10, 15, 0.6)';
                btn.style.borderColor = 'rgba(0, 212, 255, 0.3)';
            }
        }
    }

    /**
     * Update trajectory visualization
     */
    private updateTrajectoryVisualization(): void {
        // Remove old trajectory line
        if (this.trajectoryLine) {
            this.scene.remove(this.trajectoryLine);
            this.trajectoryLine.geometry.dispose();
            (this.trajectoryLine.material as THREE.Material).dispose();
        }

        // Get trajectory points
        const trajPoints = this.trajectory.getVisualizationPoints(200);
        const points = trajPoints.map(p => new THREE.Vector3(p.x, p.y, p.z));

        // Create new line
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: 0xa855f7,
            opacity: 0.6,
            transparent: true,
        });
        this.trajectoryLine = new THREE.Line(geometry, material);
        this.scene.add(this.trajectoryLine);

        // Update camera position based on trajectory size
        this.adjustCameraForTrajectory();
    }

    /**
     * Adjust camera position based on trajectory
     */
    private adjustCameraForTrajectory(): void {
        const points = this.trajectory.getVisualizationPoints(50);
        let maxDist = 0;
        for (const p of points) {
            const dist = Math.sqrt(p.x * p.x + p.z * p.z);
            if (dist > maxDist) maxDist = dist;
        }

        const cameraDistance = maxDist * 2.5;
        this.camera.position.set(cameraDistance * 0.7, cameraDistance * 0.4, cameraDistance * 0.7);
        this.controls.target.set(0, this.defaultHeight, 0);
        this.controls.update();
    }

    /**
     * Create drone trail line
     */
    private createDroneTrail(): void {
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(this.maxTrailLength * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setDrawRange(0, 0);

        const material = new THREE.LineBasicMaterial({
            color: 0x22c55e,
            opacity: 0.8,
            transparent: true,
        });

        this.droneTrail = new THREE.Line(geometry, material);
        this.scene.add(this.droneTrail);
    }

    /**
     * Update drone trail
     */
    private updateDroneTrail(): void {
        if (!this.droneTrail) return;

        const pos = this.drone.getPosition();
        this.trailPositions.push(pos.clone());

        if (this.trailPositions.length > this.maxTrailLength) {
            this.trailPositions.shift();
        }

        const geometry = this.droneTrail.geometry;
        const positions = geometry.attributes.position.array as Float32Array;

        for (let i = 0; i < this.trailPositions.length; i++) {
            positions[i * 3] = this.trailPositions[i].x;
            positions[i * 3 + 1] = this.trailPositions[i].y;
            positions[i * 3 + 2] = this.trailPositions[i].z;
        }

        geometry.attributes.position.needsUpdate = true;
        geometry.setDrawRange(0, this.trailPositions.length);
    }

    /**
     * Reset simulation
     */
    private resetSimulation(): void {
        this.simulationTime = 0;
        this.mpc.reset();
        this.trailPositions = [];

        // Position drone at trajectory start
        const initialState = this.trajectory.getInitialState();
        const startWp = this.trajectory.getWaypoint(0);

        this.drone.reset(initialState.position);
        this.drone.setHeading(initialState.heading);
        this.drone.setVelocity(
            startWp.velocity.x,
            startWp.velocity.y,
            startWp.velocity.z
        );
    }

    /**
     * Main animation loop
     */
    private animate(): void {
        this.animationFrameId = requestAnimationFrame(() => this.animate());

        const currentTime = performance.now() / 1000;
        const dt = this.lastFrameTime > 0 ? currentTime - this.lastFrameTime : 1 / 60;
        this.lastFrameTime = currentTime;

        // Update simulation
        if (this.isRunning) {
            this.updateSimulation(Math.min(dt, 0.05));
        }

        // Update controls
        this.controls.update();

        // Render
        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Update simulation step
     */
    private updateSimulation(dt: number): void {
        // Get current state
        const droneState = this.drone.getState();

        // Compute MPC control
        const command = this.mpc.computeControl(
            droneState,
            (t) => this.trajectory.getWaypoint(t),
            this.simulationTime
        );

        // Update drone
        this.drone.update(command, dt);

        // Update trail
        this.updateDroneTrail();

        // Update time
        this.simulationTime += dt;

        // Update debug overlay
        this.updateDebugOverlay(droneState, command);
    }

    /**
     * Update debug overlay
     */
    private updateDebugOverlay(
        state: { position: { x: number; y: number; z: number }; velocity: { x: number; y: number; z: number } },
        command: ControlCommand
    ): void {
        if (!this.debugOverlay) return;

        const ref = this.trajectory.getWaypoint(this.simulationTime);
        const posError = Math.sqrt(
            (state.position.x - ref.position.x) ** 2 +
            (state.position.y - ref.position.y) ** 2 +
            (state.position.z - ref.position.z) ** 2
        );
        const speed = Math.sqrt(
            state.velocity.x ** 2 +
            state.velocity.y ** 2 +
            state.velocity.z ** 2
        );
        const period = this.trajectory.getPeriod();
        const lapProgress = ((this.simulationTime % period) / period * 100).toFixed(0);

        this.debugOverlay.innerHTML = `
            <div style="margin-bottom: 8px; font-weight: bold;">${this.trajectory.getName()}</div>
            <div><b>Time:</b> ${this.simulationTime.toFixed(2)}s</div>
            <div><b>Lap:</b> ${lapProgress}%</div>
            <div style="margin-top: 6px;"><b>Speed:</b> ${speed.toFixed(1)} m/s (${(speed * 3.6).toFixed(0)} km/h)</div>
            <div><b>Tracking Error:</b> ${posError.toFixed(3)} m</div>
            <div style="margin-top: 6px;"><b>Thrust:</b> ${command.thrust.toFixed(1)} m/s²</div>
            <div><b>Rates:</b> (${command.rollRate.toFixed(1)}, ${command.pitchRate.toFixed(1)}, ${command.yawRate.toFixed(1)})</div>
        `;
    }

    /**
     * Handle window resize
     */
    private handleResize(): void {
        const rect = this.container.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        cancelAnimationFrame(this.animationFrameId);
        window.removeEventListener('resize', () => this.handleResize());

        this.renderer.dispose();
        this.controls.dispose();
        this.drone.dispose();

        while (this.scene.children.length > 0) {
            this.scene.remove(this.scene.children[0]);
        }

        this.container.innerHTML = '';
    }
}
