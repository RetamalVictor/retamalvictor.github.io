import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Core components
import { RacingDrone } from './core/RacingDrone';
import { MPC } from './control/MPC';

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
    private readonly defaultSpeed = 25.0;  // 90 km/h - testing
    private readonly defaultHeight = 4.0;
    private readonly showTrajectoryLine = false;  // Set to true for debugging

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

    // Lap timing and gate tracking
    private currentLapStartTime: number = 0;
    private bestLapTime: number = Infinity;
    private nextGateIndex: number = 0;
    private lastDroneZ: number = 0;

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
    private startOverlay: HTMLElement | null = null;
    private hasStarted: boolean = false;

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

        // Show start overlay instead of auto-starting
        this.showStartOverlay();

        // Render one frame for preview
        this.renderer.render(this.scene, this.camera);
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
        // Pause button (bottom-right, positioned to left of the unified reset button from hero.html)
        const pauseIcon = `<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;
        const playIcon = `<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><polygon points="5,3 19,12 5,21"/></svg>`;

        const pauseBtn = document.createElement('button');
        pauseBtn.className = 'absolute bottom-3 right-14 p-2 rounded-lg bg-dark-surface/80 border border-dark-border hover:border-accent-cyan hover:text-accent-cyan transition-colors text-gray-400 z-10';
        pauseBtn.title = 'Pause';
        pauseBtn.innerHTML = pauseIcon;
        pauseBtn.addEventListener('click', () => {
            this.isRunning = !this.isRunning;
            pauseBtn.innerHTML = this.isRunning ? pauseIcon : playIcon;
            pauseBtn.title = this.isRunning ? 'Pause' : 'Play';
        });
        this.container.appendChild(pauseBtn);

        // Camera mode buttons (top-left)
        const cameraDiv = document.createElement('div');
        cameraDiv.className = 'absolute top-3 left-3 flex gap-2 z-10';

        // Overview: wide view icon, Follow: crosshair/target icon
        const overviewIcon = `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>`;
        const followIcon = `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="12" cy="12" r="3"/></svg>`;

        this.cameraButtons.overview = document.createElement('button');
        this.cameraButtons.overview.className = 'p-2 rounded-lg bg-dark-surface/80 border border-dark-border hover:border-accent-cyan hover:text-accent-cyan transition-colors text-gray-400';
        this.cameraButtons.overview.title = 'Overview';
        this.cameraButtons.overview.innerHTML = overviewIcon;
        this.cameraButtons.overview.addEventListener('click', () => this.setCameraMode('overview'));
        cameraDiv.appendChild(this.cameraButtons.overview);

        this.cameraButtons.follow = document.createElement('button');
        this.cameraButtons.follow.className = 'p-2 rounded-lg bg-dark-surface/80 border border-dark-border hover:border-accent-cyan hover:text-accent-cyan transition-colors text-gray-400';
        this.cameraButtons.follow.title = 'Follow';
        this.cameraButtons.follow.innerHTML = followIcon;
        this.cameraButtons.follow.addEventListener('click', () => this.setCameraMode('follow'));
        cameraDiv.appendChild(this.cameraButtons.follow);

        this.container.appendChild(cameraDiv);

        // Info overlay (top-right) - only speed and lap
        this.debugOverlay = document.createElement('div');
        this.debugOverlay.className = 'absolute top-3 right-3 bg-dark-surface/90 border border-dark-border rounded-lg px-4 py-3 font-mono text-sm text-accent-cyan z-10';
        this.container.appendChild(this.debugOverlay);

        // Highlight initial camera button
        this.updateCameraButtonStyles();
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
                btn.className = 'p-2 rounded-lg bg-accent-cyan/20 border border-accent-cyan text-accent-cyan transition-colors';
            } else {
                btn.className = 'p-2 rounded-lg bg-dark-surface/80 border border-dark-border hover:border-accent-cyan hover:text-accent-cyan transition-colors text-gray-400';
            }
        }
    }

    /**
     * Show start overlay with play button and CPU warning
     */
    private showStartOverlay(): void {
        this.startOverlay = document.createElement('div');
        this.startOverlay.className = 'absolute inset-0 flex flex-col items-center justify-center bg-dark-bg/80 backdrop-blur-sm z-20 cursor-pointer transition-opacity duration-300';

        // Play button
        const playButton = document.createElement('div');
        playButton.className = 'w-20 h-20 rounded-full bg-accent-cyan/20 border-2 border-accent-cyan flex items-center justify-center mb-4 hover:bg-accent-cyan/30 transition-colors';
        playButton.innerHTML = `
            <svg class="w-10 h-10 text-accent-cyan ml-1" fill="currentColor" viewBox="0 0 24 24">
                <polygon points="5,3 19,12 5,21"/>
            </svg>
        `;

        // Warning text
        const warning = document.createElement('div');
        warning.className = 'text-center px-4';
        warning.innerHTML = `
            <p class="text-gray-300 text-sm mb-1">Click to start simulation</p>
            <p class="text-gray-500 text-xs">This demo uses CPU for real-time MPC computation</p>
        `;

        this.startOverlay.appendChild(playButton);
        this.startOverlay.appendChild(warning);
        this.container.appendChild(this.startOverlay);

        // Start on click
        this.startOverlay.addEventListener('click', () => this.startSimulation());
    }

    /**
     * Start the simulation after user confirms
     */
    private startSimulation(): void {
        if (this.hasStarted) return;
        this.hasStarted = true;

        // Fade out and remove overlay
        if (this.startOverlay) {
            this.startOverlay.style.opacity = '0';
            setTimeout(() => {
                if (this.startOverlay) {
                    this.startOverlay.remove();
                    this.startOverlay = null;
                }
            }, 300);
        }

        // Start animation loop
        this.lastFrameTime = performance.now() / 1000;
        this.animate();
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
            this.trajectoryLine = null;
        }

        // Only show trajectory line if debugging is enabled
        if (this.showTrajectoryLine) {
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
        }

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
            opacity: 0.6,
            transparent: true,
            depthWrite: false,  // Prevents z-fighting glitches
        });

        this.droneTrail = new THREE.Line(geometry, material);
        this.droneTrail.renderOrder = 1;  // Render after other objects
        this.droneTrail.frustumCulled = false;  // Prevent disappearing when camera moves
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
    public resetSimulation(): void {
        this.simulationTime = 0;
        this.mpc.reset();
        this.trailPositions = [];

        // Reset lap timing and gate tracking (keep best lap time)
        this.currentLapStartTime = 0;
        this.nextGateIndex = 0;
        this.lastDroneZ = 0;

        // Reset gate visualization
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
                    // Only resume if simulation has been started by user
                    if (this.hasStarted) {
                        this.lastFrameTime = performance.now() / 1000;
                        this.animate();
                    }
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

        // Check gate crossing
        this.checkGateCrossing(droneState.position);

        // Update time
        this.simulationTime += dt;

        // Update debug overlay
        this.updateDebugOverlay(droneState);
    }

    /**
     * Check if drone has crossed through a gate
     */
    private checkGateCrossing(dronePos: { x: number; y: number; z: number }): void {
        const gatePositions = this.trajectory.getGatePositions();
        if (gatePositions.length === 0 || !this.gateManager) return;

        const nextGate = gatePositions[this.nextGateIndex];
        if (!nextGate) return;

        const gateZ = nextGate.position.z;
        const gateX = nextGate.position.x;
        const gateY = nextGate.position.y;

        // Check if drone crossed the gate plane
        const crossedZ = (this.lastDroneZ < gateZ && dronePos.z >= gateZ) ||
                         (this.lastDroneZ > gateZ && dronePos.z <= gateZ);

        if (crossedZ) {
            const dx = Math.abs(dronePos.x - gateX);
            const dy = Math.abs(dronePos.y - gateY);
            const gateHalfWidth = 2.0;
            const gateHalfHeight = 2.0;

            if (dx < gateHalfWidth && dy < gateHalfHeight) {
                // Reset gates when starting new lap (crossing gate 0), not when finishing
                if (this.nextGateIndex === 0) {
                    this.gateManager.resetGates();
                }

                this.gateManager.markGatePassed(this.nextGateIndex);
                this.nextGateIndex++;

                if (this.nextGateIndex >= gatePositions.length) {
                    // Lap completed - measure from last gate to last gate
                    const lapTime = this.simulationTime - this.currentLapStartTime;

                    // Record if valid (skip first lap where currentLapStartTime was 0)
                    if (this.currentLapStartTime > 0 && lapTime > 1.0 && lapTime < this.bestLapTime) {
                        this.bestLapTime = lapTime;
                    }

                    // Start timer for next lap from this moment
                    this.currentLapStartTime = this.simulationTime;
                    this.nextGateIndex = 0;
                    // Don't reset gates here - they stay green until next lap starts
                }
            }
        }

        this.lastDroneZ = dronePos.z;
    }

    /**
     * Update debug overlay
     */
    private updateDebugOverlay(
        state: { velocity: { x: number; y: number; z: number } }
    ): void {
        if (!this.debugOverlay) return;

        const speed = Math.sqrt(
            state.velocity.x ** 2 +
            state.velocity.y ** 2 +
            state.velocity.z ** 2
        );
        const currentLapTime = this.simulationTime - this.currentLapStartTime;
        const bestLapStr = this.bestLapTime < Infinity ? `${this.bestLapTime.toFixed(1)}s` : '--';

        this.debugOverlay.innerHTML = `
            <div><span class="text-gray-400">Speed</span> ${(speed * 3.6).toFixed(0)} km/h</div>
            <div><span class="text-gray-400">Lap</span> ${currentLapTime.toFixed(1)}s</div>
            <div><span class="text-gray-400">Best</span> ${bestLapStr}</div>
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

        // Clean up start overlay
        if (this.startOverlay) {
            this.startOverlay.remove();
            this.startOverlay = null;
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

        while (this.scene.children.length > 0) {
            this.scene.remove(this.scene.children[0]);
        }

        this.container.innerHTML = '';
    }
}
