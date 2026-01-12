import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Core components
import { RacingDrone } from './core/RacingDrone';
import { MPC } from './control/MPC';
import { ControlCommand } from './types';

// Trajectory system
import {
    Trajectory,
    createTrajectory,
} from './trajectory';

// Gate visualization
import { GateManager } from './visualization/GateVisualization';

// Utilities
import { VisibilityManager } from '../../utils/VisibilityManager';

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
    private readonly defaultSpeed = 18.0;
    private readonly defaultHeight = 4.0;

    // Camera mode
    private cameraMode: 'overview' | 'follow' = 'overview';
    private followDistance = 15;  // Distance behind drone
    private followHeight = 6;     // Height above drone
    private smoothedHeading = 0;  // Smoothed heading for camera

    // State
    private simulationTime: number = 0;
    private isRunning: boolean = true;
    private lastFrameTime: number = 0;
    private animationFrameId: number = 0;

    // Visibility-based pausing
    private isPaused: boolean = false;
    private visibilityManager: VisibilityManager | null = null;
    private boundHandleResize: () => void;

    // Visualization
    private trajectoryLine: THREE.Line | null = null;
    private droneTrail: THREE.Line | null = null;
    private trailPositions: THREE.Vector3[] = [];
    private readonly maxTrailLength = 1000;
    private gateManager: GateManager | null = null;

    // UI elements
    private debugOverlay: HTMLElement | null = null;
    private cameraButtons: { overview: HTMLButtonElement | null; follow: HTMLButtonElement | null } = { overview: null, follow: null };
    private howItWorksPanel: HTMLElement | null = null;
    private howItWorksOverlay: HTMLElement | null = null;
    private showHowItWorks: boolean = false;

    constructor(containerId: string) {
        // Bind event handlers for proper cleanup
        this.boundHandleResize = this.handleResize.bind(this);

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
        this.trajectory = createTrajectory(this.defaultSpeed, this.defaultHeight);

        // Setup scene
        this.setupScene();
        this.setupUI();

        // Reset to start position
        this.resetSimulation();

        // Handle resize
        this.handleResize();
        window.addEventListener('resize', this.boundHandleResize);

        // Setup visibility-based pausing
        this.setupVisibilityHandling();

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

        // Ground grid - large enough for the track
        const gridHelper = new THREE.GridHelper(150, 75, 0x2a2a3e, 0x1a1a2e);
        gridHelper.position.set(20, 0, 20);  // Center under the track
        this.scene.add(gridHelper);

        // Add drone
        this.scene.add(this.drone.mesh);

        // Initialize gate manager
        this.gateManager = new GateManager(this.scene, { color: 0xff4444 });

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

        // Camera mode panel
        const cameraPanel = document.createElement('div');
        cameraPanel.style.cssText = `
            background: rgba(10, 10, 15, 0.9);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 8px;
            padding: 12px;
        `;

        const cameraTitle = document.createElement('div');
        cameraTitle.textContent = 'Camera';
        cameraTitle.style.cssText = `
            color: #00d4ff;
            font-family: monospace;
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 10px;
        `;
        cameraPanel.appendChild(cameraTitle);

        const cameraButtonsDiv = document.createElement('div');
        cameraButtonsDiv.style.cssText = 'display: flex; gap: 8px;';

        this.cameraButtons.overview = this.createCameraButton('Overview', 'overview');
        this.cameraButtons.follow = this.createCameraButton('Follow', 'follow');

        cameraButtonsDiv.appendChild(this.cameraButtons.overview);
        cameraButtonsDiv.appendChild(this.cameraButtons.follow);
        cameraPanel.appendChild(cameraButtonsDiv);

        controlsDiv.appendChild(cameraPanel);

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

        // Highlight initial camera button
        this.updateCameraButtonStyles();

        // "How it works" button (bottom-right)
        const howItWorksBtn = document.createElement('button');
        howItWorksBtn.innerHTML = `
            <svg class="w-4 h-4" style="width: 16px; height: 16px; margin-right: 6px;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <span>How it works</span>
        `;
        howItWorksBtn.style.cssText = `
            position: absolute;
            bottom: 16px;
            right: 16px;
            display: flex;
            align-items: center;
            padding: 8px 12px;
            background: rgba(168, 85, 247, 0.1);
            border: 1px solid rgba(168, 85, 247, 0.4);
            border-radius: 8px;
            color: #a855f7;
            font-family: system-ui, sans-serif;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
        `;
        howItWorksBtn.addEventListener('mouseenter', () => {
            howItWorksBtn.style.background = 'rgba(168, 85, 247, 0.2)';
            howItWorksBtn.style.borderColor = '#a855f7';
        });
        howItWorksBtn.addEventListener('mouseleave', () => {
            howItWorksBtn.style.background = 'rgba(168, 85, 247, 0.1)';
            howItWorksBtn.style.borderColor = 'rgba(168, 85, 247, 0.4)';
        });
        howItWorksBtn.addEventListener('click', () => this.toggleHowItWorks());
        this.container.appendChild(howItWorksBtn);

        // Create "How it works" panel and overlay
        this.createHowItWorksPanel();
    }

    private createHowItWorksPanel(): void {
        // Overlay
        this.howItWorksOverlay = document.createElement('div');
        this.howItWorksOverlay.style.cssText = `
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.5);
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s;
            z-index: 40;
        `;
        this.howItWorksOverlay.addEventListener('click', () => this.toggleHowItWorks());
        document.body.appendChild(this.howItWorksOverlay);

        // Panel
        this.howItWorksPanel = document.createElement('div');
        this.howItWorksPanel.style.cssText = `
            position: fixed;
            top: 0;
            right: 0;
            height: 100%;
            width: 384px;
            max-width: 90vw;
            background: #0a0a0f;
            border-left: 1px solid #1e1e2e;
            transform: translateX(100%);
            transition: transform 0.3s ease-in-out;
            z-index: 50;
            overflow-y: auto;
        `;
        this.howItWorksPanel.innerHTML = this.renderHowItWorksContent();
        document.body.appendChild(this.howItWorksPanel);

        // Close button handler
        const closeBtn = this.howItWorksPanel.querySelector('#drone-how-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.toggleHowItWorks());
        }

        // Escape key handler
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.showHowItWorks) {
                this.toggleHowItWorks();
            }
        });
    }

    private renderHowItWorksContent(): string {
        const cyan = '#00d4ff';
        const purple = '#a855f7';
        const yellow = '#facc15';
        const darkBg = '#0a0a0f';

        return `
            <!-- Panel Header -->
            <div style="position: sticky; top: 0; background: ${darkBg}; border-bottom: 1px solid #1e1e2e; padding: 16px; display: flex; align-items: center; justify-content: space-between;">
                <h2 style="font-size: 18px; font-weight: 600; color: ${cyan}; margin: 0;">Drone Racing Demo</h2>
                <button id="drone-how-close" style="padding: 4px; border-radius: 4px; background: none; border: none; color: #9ca3af; cursor: pointer;">
                    <svg style="width: 20px; height: 20px;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                </button>
            </div>

            <!-- Panel Content -->
            <div style="padding: 16px; font-family: system-ui, sans-serif;">
                <!-- Overview -->
                <div style="margin-bottom: 24px;">
                    <h3 style="font-weight: 500; margin-bottom: 8px; color: white;">Overview</h3>
                    <p style="color: #9ca3af; line-height: 1.6;">
                        This demo showcases <strong style="color: white;">Model Predictive Control (MPC)</strong> for autonomous drone racing.
                        A quadrotor follows various trajectories while an MPC controller computes optimal control inputs.
                    </p>
                    <p style="color: #6b7280; font-size: 14px; margin-top: 8px;">
                        Work in progress - more details coming soon.
                    </p>
                </div>

                <!-- MPC Controller -->
                <div style="margin-bottom: 24px;">
                    <h3 style="font-weight: 500; margin-bottom: 8px; color: white;">MPC Controller</h3>
                    <p style="color: #9ca3af; margin-bottom: 12px;">
                        Model Predictive Control optimizes future control inputs by predicting system behavior:
                    </p>
                    <div style="background: ${darkBg}; border-radius: 8px; padding: 12px; font-family: monospace; font-size: 14px;">
                        <div style="color: #d1d5db;"><span style="color: ${cyan};">min</span> Σ (x - x<sub>ref</sub>)ᵀQ(x - x<sub>ref</sub>) + uᵀRu</div>
                        <div style="color: #6b7280; font-size: 12px; margin-top: 8px;">subject to: dynamics, input constraints</div>
                    </div>
                </div>

                <!-- Trajectories -->
                <div style="margin-bottom: 24px;">
                    <h3 style="font-weight: 500; margin-bottom: 8px; color: white;">Trajectories</h3>
                    <p style="color: #9ca3af; margin-bottom: 12px;">
                        Multiple trajectory types demonstrate different racing scenarios:
                    </p>
                    <div style="font-size: 14px; color: #9ca3af;">
                        <div style="margin-bottom: 4px;"><span style="color: ${cyan};">●</span> Circle - Basic circular path</div>
                        <div style="margin-bottom: 4px;"><span style="color: ${cyan};">●</span> Figure 8 - Smooth transitions</div>
                        <div style="margin-bottom: 4px;"><span style="color: ${cyan};">●</span> Race Track - Multi-gate course</div>
                        <div style="margin-bottom: 4px;"><span style="color: ${cyan};">●</span> Hairpin - Sharp turns</div>
                        <div><span style="color: ${cyan};">●</span> 3D Racing - Vertical maneuvers</div>
                    </div>
                </div>

                <!-- Dynamics -->
                <div style="margin-bottom: 24px;">
                    <h3 style="font-weight: 500; margin-bottom: 8px; color: white;">Quadrotor Dynamics</h3>
                    <p style="color: #9ca3af; margin-bottom: 12px;">
                        6-DOF rigid body model with rate control:
                    </p>
                    <div style="background: ${darkBg}; border-radius: 8px; padding: 12px; font-family: monospace; font-size: 13px;">
                        <div style="color: #d1d5db;"><span style="color: ${purple};">Inputs:</span> thrust, roll rate, pitch rate, yaw rate</div>
                        <div style="color: #d1d5db; margin-top: 4px;"><span style="color: ${purple};">State:</span> position (x,y,z), velocity, orientation</div>
                    </div>
                </div>

                <!-- Controls -->
                <div>
                    <h3 style="font-weight: 500; margin-bottom: 8px; color: white;">Controls</h3>
                    <div style="font-size: 14px; color: #9ca3af;">
                        <div style="margin-bottom: 4px;"><span style="color: ${yellow};">Mouse drag</span> - Orbit camera</div>
                        <div style="margin-bottom: 4px;"><span style="color: ${yellow};">Scroll</span> - Zoom in/out</div>
                        <div style="margin-bottom: 4px;"><span style="color: ${yellow};">Reset</span> - Restart simulation</div>
                        <div><span style="color: ${yellow};">Pause/Play</span> - Toggle simulation</div>
                    </div>
                </div>
            </div>
        `;
    }

    private toggleHowItWorks(): void {
        this.showHowItWorks = !this.showHowItWorks;

        if (this.howItWorksPanel && this.howItWorksOverlay) {
            if (this.showHowItWorks) {
                this.howItWorksPanel.style.transform = 'translateX(0)';
                this.howItWorksOverlay.style.opacity = '1';
                this.howItWorksOverlay.style.pointerEvents = 'auto';
            } else {
                this.howItWorksPanel.style.transform = 'translateX(100%)';
                this.howItWorksOverlay.style.opacity = '0';
                this.howItWorksOverlay.style.pointerEvents = 'none';
            }
        }
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
     * Create camera mode button
     */
    private createCameraButton(label: string, mode: 'overview' | 'follow'): HTMLButtonElement {
        const btn = document.createElement('button');
        btn.textContent = label;
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
        `;
        btn.addEventListener('click', () => this.setCameraMode(mode));
        return btn;
    }

    /**
     * Set camera mode
     */
    private setCameraMode(mode: 'overview' | 'follow'): void {
        if (mode === this.cameraMode) return;

        this.cameraMode = mode;
        this.updateCameraButtonStyles();

        if (mode === 'overview') {
            // Re-enable orbit controls and adjust camera for overview
            this.controls.enabled = true;
            this.adjustCameraForTrajectory();
        } else {
            // Disable orbit controls for follow mode
            this.controls.enabled = false;
        }
    }

    /**
     * Update camera button styles
     */
    private updateCameraButtonStyles(): void {
        const modes: ('overview' | 'follow')[] = ['overview', 'follow'];
        for (const mode of modes) {
            const btn = this.cameraButtons[mode];
            if (!btn) continue;

            if (mode === this.cameraMode) {
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

        // Update gates
        this.updateGates();

        // Update camera position based on trajectory size
        this.adjustCameraForTrajectory();
    }

    /**
     * Update gate visualization for current trajectory
     */
    private updateGates(): void {
        if (!this.gateManager) return;

        const gatePositions = this.trajectory.getGatePositions();
        this.gateManager.setGates(gatePositions, 3.0, 3.0);  // 3m x 3m gates
    }

    /**
     * Adjust camera position - spectator view from stadium seating
     */
    private adjustCameraForTrajectory(): void {
        const gatePositions = this.trajectory.getGatePositions();

        if (gatePositions.length === 0) {
            this.camera.position.set(50, 15, 50);
            this.controls.target.set(0, this.defaultHeight, 0);
            this.controls.update();
            return;
        }

        // Compute center of all gates
        let sumX = 0, sumY = 0, sumZ = 0;
        for (const gate of gatePositions) {
            sumX += gate.position.x;
            sumY += gate.position.y;
            sumZ += gate.position.z;
        }
        const centerX = sumX / gatePositions.length;
        const centerY = sumY / gatePositions.length;
        const centerZ = sumZ / gatePositions.length;

        // Spectator position: on the side of the track, elevated like bleachers
        // Positioned to the left side (negative X), looking at the action
        this.camera.position.set(
            -25,    // Side of track (spectator stands)
            18,     // Elevated like stadium seating
            25      // Midway along the track
        );
        this.controls.target.set(centerX, centerY, centerZ);
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

        // Reset gate states
        if (this.gateManager) {
            this.gateManager.resetGates();
        }

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
     * Setup visibility-based pausing using VisibilityManager
     */
    private setupVisibilityHandling(): void {
        this.visibilityManager = new VisibilityManager(
            this.container,
            (paused) => {
                if (paused) {
                    this.isPaused = true;
                    if (this.animationFrameId) {
                        cancelAnimationFrame(this.animationFrameId);
                        this.animationFrameId = 0;
                    }
                } else {
                    this.isPaused = false;
                    this.lastFrameTime = performance.now() / 1000;
                    this.animate();
                }
            }
        );
    }

    /**
     * Main animation loop
     */
    private animate(): void {
        if (this.isPaused) return;

        this.animationFrameId = requestAnimationFrame(() => this.animate());

        const currentTime = performance.now() / 1000;
        const dt = this.lastFrameTime > 0 ? currentTime - this.lastFrameTime : 1 / 60;
        this.lastFrameTime = currentTime;

        // Update simulation
        if (this.isRunning) {
            this.updateSimulation(Math.min(dt, 0.05));
        }

        // Update camera based on mode
        if (this.cameraMode === 'follow') {
            this.updateFollowCamera();
        } else {
            this.controls.update();
        }

        // Render
        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Update camera to follow the drone from behind
     */
    private updateFollowCamera(): void {
        const dronePos = this.drone.getPosition();
        const droneState = this.drone.getState();

        // Compute target heading from velocity direction
        const vx = droneState.velocity.x;
        const vz = droneState.velocity.z;
        const speed = Math.sqrt(vx * vx + vz * vz);

        let targetHeading: number;
        if (speed > 1.0) {
            // Use velocity direction when moving fast enough
            targetHeading = Math.atan2(vx, vz);
        } else {
            // Keep current heading when slow/stationary
            targetHeading = this.smoothedHeading;
        }

        // Smooth the heading change (handles angle wrapping)
        let headingDiff = targetHeading - this.smoothedHeading;
        // Wrap to [-PI, PI]
        while (headingDiff > Math.PI) headingDiff -= 2 * Math.PI;
        while (headingDiff < -Math.PI) headingDiff += 2 * Math.PI;

        // Very smooth heading interpolation to avoid jitter
        const headingSmoothness = 0.03;
        this.smoothedHeading += headingDiff * headingSmoothness;

        // Calculate camera position BEHIND the drone (opposite to smoothed heading)
        const targetCamPos = new THREE.Vector3(
            dronePos.x - Math.sin(this.smoothedHeading) * this.followDistance,
            dronePos.y + this.followHeight,
            dronePos.z - Math.cos(this.smoothedHeading) * this.followDistance
        );

        // Smooth camera position transition
        const positionSmoothness = 0.06;
        this.camera.position.lerp(targetCamPos, positionSmoothness);

        // Look at the drone
        this.camera.lookAt(dronePos);
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
        // Stop animation
        this.isPaused = true;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = 0;
        }

        // Clean up visibility manager and resize listener
        if (this.visibilityManager) {
            this.visibilityManager.destroy();
            this.visibilityManager = null;
        }
        window.removeEventListener('resize', this.boundHandleResize);

        this.renderer.dispose();
        this.controls.dispose();
        this.drone.dispose();

        // Dispose gate manager
        if (this.gateManager) {
            this.gateManager.dispose();
        }

        // Clean up "How it works" panel and overlay
        if (this.howItWorksPanel && this.howItWorksPanel.parentNode) {
            this.howItWorksPanel.parentNode.removeChild(this.howItWorksPanel);
        }
        if (this.howItWorksOverlay && this.howItWorksOverlay.parentNode) {
            this.howItWorksOverlay.parentNode.removeChild(this.howItWorksOverlay);
        }

        while (this.scene.children.length > 0) {
            this.scene.remove(this.scene.children[0]);
        }

        this.container.innerHTML = '';
    }
}
