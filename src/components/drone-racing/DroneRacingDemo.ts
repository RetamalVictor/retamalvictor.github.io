import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Core components
import { RacingDrone } from './core/RacingDrone';
import { MPC } from './control/MPC';
import { Waypoint, ControlCommand } from './types';

/**
 * Minimal Drone Racing Demo
 *
 * Clean demo using only core components:
 * - RacingDrone (DroneDynamics + DroneVisualization)
 * - MPC (true Model Predictive Control)
 *
 * Demonstrates the drone tracking a circular reference trajectory.
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

    // Trajectory parameters - Racing drone speeds!
    private readonly trajectoryRadius = 25.0;   // Large racing track
    private readonly trajectoryHeight = 4.0;    // Good altitude for visibility
    private readonly trajectorySpeed = 20.0;    // 20 m/s = 72 km/h racing speed

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

    // Debug overlay
    private debugOverlay: HTMLElement | null = null;

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

        // Create camera - positioned for large racing track
        this.camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
        this.camera.position.set(40, 20, 40);
        this.camera.lookAt(0, 4, 0);

        // Create controls
        this.controls = new OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0, 4, 0);

        // Initialize components
        this.drone = new RacingDrone();
        this.mpc = new MPC();

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

        // Ground grid
        const gridHelper = new THREE.GridHelper(40, 40, 0x2a2a3e, 0x1a1a2e);
        this.scene.add(gridHelper);

        // Add drone
        this.scene.add(this.drone.mesh);

        // Create trajectory visualization
        this.createTrajectoryVisualization();

        // Create drone trail
        this.createDroneTrail();
    }

    /**
     * Setup UI elements
     */
    private setupUI(): void {
        // Controls container
        const controlsDiv = document.createElement('div');
        controlsDiv.style.cssText = `
            position: absolute;
            top: 16px;
            left: 16px;
            display: flex;
            gap: 8px;
        `;

        // Reset button
        const resetBtn = this.createButton('Reset', () => this.resetSimulation());
        controlsDiv.appendChild(resetBtn);

        // Pause/Play button
        const pauseBtn = this.createButton('Pause', () => {
            this.isRunning = !this.isRunning;
            pauseBtn.textContent = this.isRunning ? 'Pause' : 'Play';
        });
        controlsDiv.appendChild(pauseBtn);

        this.container.appendChild(controlsDiv);

        // Debug overlay
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
     * Create circular trajectory visualization
     */
    private createTrajectoryVisualization(): void {
        const points: THREE.Vector3[] = [];
        const segments = 100;

        for (let i = 0; i <= segments; i++) {
            const angle = (i / segments) * Math.PI * 2;
            points.push(new THREE.Vector3(
                this.trajectoryRadius * Math.cos(angle),
                this.trajectoryHeight,
                this.trajectoryRadius * Math.sin(angle)
            ));
        }

        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: 0xa855f7,
            opacity: 0.6,
            transparent: true,
        });
        this.trajectoryLine = new THREE.Line(geometry, material);
        this.scene.add(this.trajectoryLine);
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
     * Get reference waypoint at time t (circular trajectory)
     */
    private getReference(t: number): Waypoint {
        const omega = this.trajectorySpeed / this.trajectoryRadius;
        const angle = omega * t;

        // Position
        const x = this.trajectoryRadius * Math.cos(angle);
        const z = this.trajectoryRadius * Math.sin(angle);

        // Velocity (tangent)
        const vx = -this.trajectorySpeed * Math.sin(angle);
        const vz = this.trajectorySpeed * Math.cos(angle);

        // Acceleration (centripetal)
        const centripetalAccel = (this.trajectorySpeed * this.trajectorySpeed) / this.trajectoryRadius;
        const ax = -centripetalAccel * Math.cos(angle);
        const az = -centripetalAccel * Math.sin(angle);

        // Heading (direction of motion) - nose points along velocity
        const heading = Math.atan2(vx, vz);

        return {
            position: { x, y: this.trajectoryHeight, z },
            velocity: { x: vx, y: 0, z: vz },
            acceleration: { x: ax, y: 0, z: az },
            jerk: { x: 0, y: 0, z: 0 },
            heading,
            headingRate: -omega,  // Negative because heading decreases as we go counterclockwise
            time: t,
        };
    }

    /**
     * Reset simulation
     */
    private resetSimulation(): void {
        this.simulationTime = 0;
        this.mpc.reset();
        this.trailPositions = [];

        // Position drone at trajectory start
        const startRef = this.getReference(0);
        this.drone.reset({
            x: startRef.position.x,
            y: startRef.position.y,
            z: startRef.position.z,
        });
        this.drone.setHeading(startRef.heading);
        this.drone.setVelocity(
            startRef.velocity.x,
            startRef.velocity.y,
            startRef.velocity.z
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
            (t) => this.getReference(t),
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
    private updateDebugOverlay(state: { position: { x: number; y: number; z: number }; velocity: { x: number; y: number; z: number } }, command: ControlCommand): void {
        if (!this.debugOverlay) return;

        const ref = this.getReference(this.simulationTime);
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

        this.debugOverlay.innerHTML = `
            <div style="margin-bottom: 8px; font-weight: bold;">MPC Demo</div>
            <div><b>Time:</b> ${this.simulationTime.toFixed(2)}s</div>
            <div style="margin-top: 6px;"><b>Position:</b></div>
            <div>(${state.position.x.toFixed(2)}, ${state.position.y.toFixed(2)}, ${state.position.z.toFixed(2)})</div>
            <div style="margin-top: 6px;"><b>Speed:</b> ${speed.toFixed(2)} m/s</div>
            <div><b>Pos Error:</b> ${posError.toFixed(3)} m</div>
            <div style="margin-top: 6px;"><b>Command:</b></div>
            <div>Thrust: ${command.thrust.toFixed(2)} m/s²</div>
            <div>Rates: (${command.rollRate.toFixed(2)}, ${command.pitchRate.toFixed(2)}, ${command.yawRate.toFixed(2)})</div>
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
