import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Core components
import { RacingDrone } from './core/RacingDrone';
import { RaceTrack } from './core/RaceTrack';

// Perception
import { PinholeCamera } from '../ibvs/Camera';
import { GateDetector } from './perception/GateDetector';

// Planning & Control
import { TrajectoryGenerator } from './planning/TrajectoryGenerator';
import { SimplifiedMPC } from './control/SimplifiedMPC';

// Types
import { RacingConfig, DEFAULT_CONFIG, Trajectory, PipelineStatus, GateDetection, Waypoint } from './types';

/**
 * Drone Racing Demo - Main Orchestrator
 *
 * Interactive demo showcasing the autonomous drone racing pipeline:
 * 1. Gate Detection - Simulated corner keypoint detection
 * 2. State Estimation - PnP pose estimation
 * 3. Trajectory Generation - Minimum-jerk paths through gates
 * 4. MPC Control - Trajectory tracking with prediction
 *
 * Features:
 * - 3D visualization with Three.js
 * - Camera view panel showing detections
 * - Pipeline status bar
 * - Interactive controls (speed, pause, reset)
 */
export class DroneRacingDemo {
    // DOM elements
    private container: HTMLElement;
    private canvas: HTMLCanvasElement;
    private cameraCanvas: HTMLCanvasElement;
    private statusBar: HTMLElement | null = null;
    private debugOverlay: HTMLElement | null = null;
    private logPanel: HTMLElement | null = null;
    private logHistory: string[] = [];

    // Three.js
    private renderer: THREE.WebGLRenderer;
    private scene: THREE.Scene;
    private camera: THREE.PerspectiveCamera;
    private controls: OrbitControls;

    // Racing components
    private drone: RacingDrone;
    private track: RaceTrack;
    private droneCamera: PinholeCamera;
    private gateDetector: GateDetector;
    private trajectoryGenerator: TrajectoryGenerator;
    private mpc: SimplifiedMPC;

    // State
    private config: RacingConfig;
    private trajectory: Trajectory | null = null;
    private simulationTime: number = 0;
    private isRunning: boolean = true;
    private lastFrameTime: number = 0;

    // Visualization
    private trajectoryLine: THREE.Line | null = null;
    private predictionPoints: THREE.Points | null = null;
    private lastDetections: GateDetection[] = [];

    // Drone trail visualization
    private droneTrail: THREE.Line | null = null;
    private trailPositions: THREE.Vector3[] = [];
    private readonly maxTrailLength = 50000;  // Keep long trail for trajectory study

    // Animation
    private animationFrameId: number = 0;

    // Test mode: simple line trajectory
    private testMode: boolean = true;  // Set to true for simple line test
    private targetSpeed: number = 3.0;  // Target speed for test trajectory (controlled by slider)

    // Plotting data
    private plotCanvas: HTMLCanvasElement | null = null;
    private plotData: {
        time: number[];
        posError: number[];
        velocity: number[];
        tilt: number[];
        pitch: number[];
        thrust: number[];
    } = { time: [], posError: [], velocity: [], tilt: [], pitch: [], thrust: [] };
    private readonly maxPlotPoints = 300;  // ~5 seconds at 60fps

    constructor(containerId: string, config: Partial<RacingConfig> = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };

        // Get container
        const container = document.getElementById(containerId);
        if (!container) {
            throw new Error(`Container ${containerId} not found`);
        }
        this.container = container;
        this.container.innerHTML = '';
        this.container.style.position = 'relative';

        // Create main canvas
        this.canvas = document.createElement('canvas');
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.container.appendChild(this.canvas);

        // Create camera view canvas
        this.cameraCanvas = document.createElement('canvas');
        this.cameraCanvas.width = 200;
        this.cameraCanvas.height = 150;
        this.cameraCanvas.style.position = 'absolute';
        this.cameraCanvas.style.left = '16px';
        this.cameraCanvas.style.bottom = '60px';
        this.cameraCanvas.style.border = '2px solid rgba(0, 212, 255, 0.5)';
        this.cameraCanvas.style.borderRadius = '8px';
        this.cameraCanvas.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
        this.container.appendChild(this.cameraCanvas);

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
        this.camera.position.set(15, 10, 15);
        this.camera.lookAt(0, 2, 0);

        // Create controls
        this.controls = new OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0, 2, 0);

        // Initialize components
        this.drone = new RacingDrone();
        this.track = new RaceTrack(this.config);
        this.droneCamera = new PinholeCamera({
            focalLength: 4,
            sensorWidth: 200,
            sensorHeight: 150,
            sensorSizeMM: 6,
        });
        this.gateDetector = new GateDetector(this.droneCamera, this.config);
        this.trajectoryGenerator = new TrajectoryGenerator(this.config);
        this.mpc = new SimplifiedMPC(this.config);

        // Setup scene
        this.setupScene();
        this.setupUI();

        // Generate initial trajectory
        this.generateTrajectory();

        // Position drone at first gate
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

        // Add track (but hide gates for trajectory study)
        this.scene.add(this.track.trackGroup);
        // Hide all gate meshes - only show reference trajectory
        for (const gate of this.track.gates) {
            gate.mesh.visible = false;
        }

        // Add drone
        this.scene.add(this.drone.mesh);

        // Create drone trail line
        this.createDroneTrail();

        // Create trajectory line
        this.createTrajectoryVisualization();
    }

    /**
     * Setup UI elements
     */
    private setupUI(): void {
        // Status bar
        this.statusBar = document.createElement('div');
        this.statusBar.style.cssText = `
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 48px;
            background: rgba(10, 10, 15, 0.9);
            border-top: 1px solid rgba(0, 212, 255, 0.3);
            display: flex;
            align-items: center;
            justify-content: space-around;
            padding: 0 16px;
            font-family: monospace;
            font-size: 12px;
            color: #e0e0e0;
        `;
        this.container.appendChild(this.statusBar);

        // Control buttons
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

        // Speed control
        const speedControl = document.createElement('div');
        speedControl.style.cssText = `
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(10, 10, 15, 0.8);
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid rgba(0, 212, 255, 0.3);
        `;
        const speedLabel = document.createElement('span');
        speedLabel.textContent = 'Speed:';
        speedLabel.style.color = '#aaa';
        speedControl.appendChild(speedLabel);

        const speedSlider = document.createElement('input');
        speedSlider.type = 'range';
        speedSlider.min = '2';
        speedSlider.max = '10';
        speedSlider.value = String(this.config.maxSpeed);
        speedSlider.style.width = '80px';
        speedSlider.addEventListener('input', () => {
            const speed = parseFloat(speedSlider.value);
            this.targetSpeed = speed;  // Update target speed for test mode
            this.trajectoryGenerator.setMaxSpeed(speed);
            this.generateTrajectory();
            speedValue.textContent = speed.toFixed(1) + ' m/s';
        });
        speedControl.appendChild(speedSlider);

        const speedValue = document.createElement('span');
        speedValue.textContent = this.config.maxSpeed.toFixed(1) + ' m/s';
        speedValue.style.color = '#00d4ff';
        speedValue.style.minWidth = '50px';
        speedControl.appendChild(speedValue);

        controlsDiv.appendChild(speedControl);
        this.container.appendChild(controlsDiv);

        // Camera view label
        const cameraLabel = document.createElement('div');
        cameraLabel.textContent = 'Drone Camera';
        cameraLabel.style.cssText = `
            position: absolute;
            left: 16px;
            bottom: 215px;
            font-family: monospace;
            font-size: 11px;
            color: #888;
        `;
        this.container.appendChild(cameraLabel);

        // Debug overlay (top right)
        this.debugOverlay = document.createElement('div');
        this.debugOverlay.style.cssText = `
            position: absolute;
            top: 16px;
            right: 16px;
            background: rgba(0, 0, 0, 0.85);
            border: 1px solid rgba(255, 100, 100, 0.5);
            border-radius: 8px;
            padding: 12px;
            font-family: monospace;
            font-size: 11px;
            color: #ff6666;
            max-width: 300px;
            line-height: 1.4;
        `;
        this.debugOverlay.innerHTML = 'DEBUG: Initializing...';
        this.container.appendChild(this.debugOverlay);

        // Log panel (bottom right, scrollable history)
        this.logPanel = document.createElement('div');
        this.logPanel.style.cssText = `
            position: absolute;
            bottom: 60px;
            right: 16px;
            width: 350px;
            height: 200px;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid rgba(100, 255, 100, 0.5);
            border-radius: 8px;
            padding: 8px;
            font-family: monospace;
            font-size: 10px;
            color: #88ff88;
            overflow-y: auto;
            line-height: 1.3;
        `;
        this.logPanel.innerHTML = '<div style="color: #ffaa00;">LOG HISTORY:</div>';
        this.container.appendChild(this.logPanel);

        // Real-time plots canvas
        this.plotCanvas = document.createElement('canvas');
        this.plotCanvas.width = 400;
        this.plotCanvas.height = 300;
        this.plotCanvas.style.cssText = `
            position: absolute;
            top: 60px;
            right: 16px;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid rgba(0, 212, 255, 0.5);
            border-radius: 8px;
        `;
        this.container.appendChild(this.plotCanvas);
    }

    /**
     * Add a log entry (visible in the log panel)
     */
    private log(message: string): void {
        const timestamp = this.simulationTime.toFixed(2);
        const entry = `[${timestamp}s] ${message}`;
        this.logHistory.push(entry);

        // Keep last 50 entries
        if (this.logHistory.length > 50) {
            this.logHistory.shift();
        }

        // Update log panel
        if (this.logPanel) {
            this.logPanel.innerHTML = '<div style="color: #ffaa00; margin-bottom: 4px;">LOG HISTORY:</div>' +
                this.logHistory.map(e => `<div>${e}</div>`).join('');
            this.logPanel.scrollTop = this.logPanel.scrollHeight;
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
     * Create trajectory visualization
     */
    private createTrajectoryVisualization(): void {
        // Will be updated when trajectory is generated
    }

    /**
     * Create drone trail line for visualizing actual path
     */
    private createDroneTrail(): void {
        const geometry = new THREE.BufferGeometry();
        // Pre-allocate buffer for trail positions
        const positions = new Float32Array(this.maxTrailLength * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setDrawRange(0, 0);  // Start with nothing visible

        const material = new THREE.LineBasicMaterial({
            color: 0x22c55e,  // Green for actual path
            opacity: 0.9,
            transparent: true,
            linewidth: 2,
        });

        this.droneTrail = new THREE.Line(geometry, material);
        this.scene.add(this.droneTrail);
    }

    /**
     * Update drone trail with current position
     */
    private updateDroneTrail(): void {
        if (!this.droneTrail) return;

        const pos = this.drone.getPosition();
        this.trailPositions.push(pos.clone());

        // Update geometry - reallocate if needed
        const geometry = this.droneTrail.geometry;
        const currentBuffer = geometry.attributes.position.array as Float32Array;

        // If buffer is too small, create a larger one
        if (this.trailPositions.length * 3 > currentBuffer.length) {
            const newSize = Math.min(this.trailPositions.length * 2, this.maxTrailLength) * 3;
            const newPositions = new Float32Array(newSize);
            newPositions.set(currentBuffer);
            geometry.setAttribute('position', new THREE.BufferAttribute(newPositions, 3));
        }

        const positions = geometry.attributes.position.array as Float32Array;

        // Only update the latest position (optimization)
        const idx = this.trailPositions.length - 1;
        positions[idx * 3] = this.trailPositions[idx].x;
        positions[idx * 3 + 1] = this.trailPositions[idx].y;
        positions[idx * 3 + 2] = this.trailPositions[idx].z;

        geometry.attributes.position.needsUpdate = true;
        geometry.setDrawRange(0, this.trailPositions.length);
    }

    /**
     * Clear drone trail
     */
    private clearDroneTrail(): void {
        this.trailPositions = [];
        if (this.droneTrail) {
            this.droneTrail.geometry.setDrawRange(0, 0);
        }
    }

    /**
     * Generate a circular trajectory for testing
     * Drone flies in a circle in the XZ plane at constant height
     */
    private generateTestTrajectory(): void {
        const speed = this.targetSpeed;  // Use slider-controlled speed

        // Calculate radius to keep centripetal acceleration under 3 m/s²
        // a_centripetal = v²/r, so r = v²/a_max
        const maxCentripetalAccel = 3.0;  // m/s² - well within tilt limits
        const minRadius = (speed * speed) / maxCentripetalAccel;
        const radius = Math.max(10.0, minRadius);  // At least 10m, more if needed

        const height = 2.0;  // meters
        const sampleRate = 50;  // Hz

        // Angular velocity: ω = v / r
        const omega = speed / radius;

        // One full circle duration: T = 2πr / v
        const duration = (2 * Math.PI * radius) / speed;

        // Centripetal acceleration: a = v² / r
        const centripetalAccel = (speed * speed) / radius;

        const waypoints: Waypoint[] = [];
        const dt = 1 / sampleRate;

        for (let t = 0; t <= duration + dt; t += dt) {
            const angle = omega * t;  // Current angle in radians

            // Position on circle (starting at +X, going counterclockwise when viewed from above)
            const x = radius * Math.cos(angle);
            const z = radius * Math.sin(angle);

            // Velocity (tangent to circle)
            const vx = -speed * Math.sin(angle);
            const vz = speed * Math.cos(angle);

            // Acceleration (centripetal, pointing toward center)
            const ax = -centripetalAccel * Math.cos(angle);
            const az = -centripetalAccel * Math.sin(angle);

            // Heading: direction of motion (tangent)
            // atan2(vz, vx) but we want heading where +Z is forward
            const heading = Math.atan2(vx, vz);

            waypoints.push({
                position: { x, y: height, z },
                velocity: { x: vx, y: 0, z: vz },
                acceleration: { x: ax, y: 0, z: az },
                jerk: { x: 0, y: 0, z: 0 },
                heading: heading,
                headingRate: omega,  // Constant turn rate
                time: t,
            });
        }

        this.trajectory = {
            segments: [{
                startWaypoint: waypoints[0],
                endWaypoint: waypoints[waypoints.length - 1],
                duration: duration,
                gateId: -1,  // Test mode indicator
            }],
            totalDuration: duration,
            waypoints: waypoints,
        };

        this.log(`<span style="color:#00ff00">TEST MODE: Circle trajectory</span>`);
        this.log(`Speed: ${speed.toFixed(1)} m/s, Radius: ${radius.toFixed(1)}m, Duration: ${duration.toFixed(1)}s`);
        this.log(`Centripetal accel: ${centripetalAccel.toFixed(2)} m/s² (max 3.0)`);

        // Debug: log first few waypoints
        console.log('=== CIRCLE TRAJECTORY DEBUG ===');
        console.log(`Duration: ${duration.toFixed(2)}s, Waypoints: ${waypoints.length}`);
        for (let i = 0; i < Math.min(5, waypoints.length); i++) {
            const wp = waypoints[i];
            console.log(`WP[${i}] t=${wp.time.toFixed(2)}: pos=(${wp.position.x.toFixed(2)}, ${wp.position.y.toFixed(2)}, ${wp.position.z.toFixed(2)}) vel=(${wp.velocity.x.toFixed(2)}, ${wp.velocity.y.toFixed(2)}, ${wp.velocity.z.toFixed(2)}) acc=(${wp.acceleration.x.toFixed(2)}, ${wp.acceleration.y.toFixed(2)}, ${wp.acceleration.z.toFixed(2)})`);
        }
    }

    /**
     * Generate trajectory through track
     */
    private generateTrajectory(): void {
        if (this.testMode) {
            this.generateTestTrajectory();
            return;
        }

        this.trajectory = this.trajectoryGenerator.generateRacingTrajectory(this.track);

        // DEBUG: Log trajectory info
        console.log('=== TRAJECTORY DEBUG ===');
        console.log('Total duration:', this.trajectory.totalDuration);
        console.log('Num waypoints:', this.trajectory.waypoints.length);
        console.log('First 5 waypoints:');
        this.trajectory.waypoints.slice(0, 5).forEach((wp, i) => {
            console.log(`  [${i}] pos:`, wp.position, 'vel:', wp.velocity);
        });
        console.log('Gate positions:');
        this.track.gates.forEach((g, i) => {
            const pos = g.getCenter();
            console.log(`  Gate ${i}: [${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)}]`);
        });

        // Update visualization
        if (this.trajectoryLine) {
            this.scene.remove(this.trajectoryLine);
        }

        const points = this.trajectory.waypoints.map(wp =>
            new THREE.Vector3(wp.position.x, wp.position.y, wp.position.z)
        );

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
     * Reset simulation
     */
    private resetSimulation(): void {
        // Reset time
        this.simulationTime = 0;

        // Reset track
        this.track.reset();

        // Reset MPC
        this.mpc.reset();

        // Clear plot data
        this.plotData = { time: [], posError: [], velocity: [], tilt: [], pitch: [], thrust: [] };

        // Clear drone trail
        this.clearDroneTrail();

        // Regenerate trajectory first
        this.generateTrajectory();

        // CRITICAL: Position drone AT trajectory start to match initial conditions
        if (!this.trajectory) return;

        // Get start waypoint based on mode
        const startWaypoint = this.testMode
            ? this.trajectory.waypoints[0]
            : this.trajectoryGenerator.getWaypointAtTime(this.trajectory, 0);

        this.drone.reset({
            x: startWaypoint.position.x,
            y: startWaypoint.position.y,
            z: startWaypoint.position.z
        });
        this.drone.setHeading(startWaypoint.heading);
        this.drone.setVelocity(
            startWaypoint.velocity.x,
            startWaypoint.velocity.y,
            startWaypoint.velocity.z
        );

        // Clear log and add reset info
        this.logHistory = [];
        this.log(`<span style="color:#00ffff">RESET - Matched to trajectory start</span>`);
        this.log(`Drone: (${startWaypoint.position.x.toFixed(1)}, ${startWaypoint.position.y.toFixed(1)}, ${startWaypoint.position.z.toFixed(1)})`);
        this.log(`Velocity: (${startWaypoint.velocity.x.toFixed(1)}, ${startWaypoint.velocity.y.toFixed(1)}, ${startWaypoint.velocity.z.toFixed(1)})`);
        this.log(`Accel: (${startWaypoint.acceleration.x.toFixed(2)}, ${startWaypoint.acceleration.y.toFixed(2)}, ${startWaypoint.acceleration.z.toFixed(2)})`);
        this.log(`Trajectory: ${this.trajectory.segments.length} segments, ${this.trajectory.totalDuration.toFixed(1)}s total`);

        // Debug: verify drone state matches
        console.log('=== RESET DEBUG ===');
        console.log('Start waypoint:', startWaypoint);
        const postResetState = this.drone.getState();
        console.log('Drone state after reset:', postResetState);

        // SANITY CHECK: Verify quaternion after setHeading (per plan Step 1)
        const state = this.drone.getState();
        const q = state.orientation;
        const threeQuat = new THREE.Quaternion(q.x, q.y, q.z, q.w);
        const euler = new THREE.Euler().setFromQuaternion(threeQuat, 'YXZ');
        const rollDeg = euler.z * 180 / Math.PI;
        const pitchDeg = euler.x * 180 / Math.PI;
        const yawDeg = euler.y * 180 / Math.PI;
        this.log(`Quat: w=${q.w.toFixed(3)} x=${q.x.toFixed(3)} y=${q.y.toFixed(3)} z=${q.z.toFixed(3)}`);
        this.log(`Euler(YXZ): roll=${rollDeg.toFixed(1)}° pitch=${pitchDeg.toFixed(1)}° yaw=${yawDeg.toFixed(1)}°`);
        if (Math.abs(rollDeg) > 1 || Math.abs(pitchDeg) > 1) {
            this.log(`<span style="color:#ff0000">BUG: Expected roll≈0 pitch≈0 at reset!</span>`);
        }
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
        if (this.isRunning && this.trajectory) {
            this.updateSimulation(dt);
        }

        // Update controls
        this.controls.update();

        // Update visualizations
        this.updateCameraView();
        this.updateStatusBar();
        this.updatePredictionVisualization();
        this.drawPlots();

        // Render
        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Draw real-time plots
     */
    private drawPlots(): void {
        if (!this.plotCanvas) return;
        const ctx = this.plotCanvas.getContext('2d');
        if (!ctx) return;

        const w = this.plotCanvas.width;
        const h = this.plotCanvas.height;

        // Clear
        ctx.fillStyle = 'rgba(0, 0, 0, 0.95)';
        ctx.fillRect(0, 0, w, h);

        // Title
        ctx.fillStyle = '#00d4ff';
        ctx.font = 'bold 12px monospace';
        ctx.fillText('REAL-TIME PLOTS', 10, 15);

        // Plot areas (3 rows)
        const plotH = 80;
        const plotY = [30, 120, 210];
        const plotLabels = ['Pos Error (m)', 'Speed (m/s)', 'Pitch (deg)'];
        const plotColors = ['#ff6666', '#66ff66', '#6666ff'];
        const plotData = [this.plotData.posError, this.plotData.velocity, this.plotData.pitch];
        const plotRanges = [[0, 2], [0, 12], [-30, 30]];  // [min, max] for each plot

        for (let p = 0; p < 3; p++) {
            const y0 = plotY[p];
            const data = plotData[p];
            const [minV, maxV] = plotRanges[p];

            // Background
            ctx.fillStyle = 'rgba(30, 30, 40, 0.8)';
            ctx.fillRect(50, y0, w - 60, plotH);

            // Border
            ctx.strokeStyle = 'rgba(100, 100, 100, 0.5)';
            ctx.strokeRect(50, y0, w - 60, plotH);

            // Label
            ctx.fillStyle = plotColors[p];
            ctx.font = '10px monospace';
            ctx.fillText(plotLabels[p], 5, y0 + plotH / 2);

            // Zero line for pitch
            if (p === 2) {
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
                ctx.beginPath();
                const zeroY = y0 + plotH * (1 - (0 - minV) / (maxV - minV));
                ctx.moveTo(50, zeroY);
                ctx.lineTo(w - 10, zeroY);
                ctx.stroke();
            }

            // Target line for velocity (only if within plot bounds)
            if (p === 1 && this.targetSpeed >= minV && this.targetSpeed <= maxV) {
                ctx.strokeStyle = 'rgba(255, 255, 0, 0.5)';
                ctx.beginPath();
                const targetY = y0 + plotH * (1 - (this.targetSpeed - minV) / (maxV - minV));
                ctx.moveTo(50, targetY);
                ctx.lineTo(w - 10, targetY);
                ctx.stroke();
                ctx.fillStyle = '#ffff00';
                ctx.fillText(`${this.targetSpeed.toFixed(0)}`, w - 45, targetY - 2);
            }

            // Draw data
            if (data.length > 1) {
                ctx.strokeStyle = plotColors[p];
                ctx.lineWidth = 1.5;
                ctx.beginPath();

                const xScale = (w - 60) / this.maxPlotPoints;

                for (let i = 0; i < data.length; i++) {
                    const x = 50 + i * xScale;
                    const v = Math.max(minV, Math.min(maxV, data[i]));
                    const y = y0 + plotH * (1 - (v - minV) / (maxV - minV));

                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
                ctx.lineWidth = 1;

                // Current value
                const currentVal = data[data.length - 1];
                ctx.fillStyle = '#ffffff';
                ctx.fillText(currentVal.toFixed(2), w - 45, y0 + 12);
            }

            // Y-axis labels
            ctx.fillStyle = '#888888';
            ctx.font = '9px monospace';
            ctx.fillText(maxV.toString(), 50, y0 + 10);
            ctx.fillText(minV.toString(), 50, y0 + plotH - 2);
        }
    }

    // DEBUG: frame counter for logging
    private debugFrameCount = 0;

    /**
     * Update simulation step
     */
    private updateSimulation(dt: number): void {
        if (!this.trajectory) return;

        // Store previous position for gate detection
        const prevPosition = this.drone.getPosition();

        // Get reference waypoint
        const getReference = (t: number): Waypoint => {
            if (this.testMode) {
                // Simple interpolation for test trajectory
                const traj = this.trajectory!;
                const wrappedT = t % traj.totalDuration;
                const idx = Math.floor(wrappedT * 50);  // 50 Hz sample rate
                const maxIdx = traj.waypoints.length - 1;
                return traj.waypoints[Math.min(idx, maxIdx)];
            }
            return this.trajectoryGenerator.getWaypointAtTime(this.trajectory!, t);
        };

        // Compute MPC control
        const droneState = this.drone.getState();
        const command = this.mpc.computeControl(droneState, getReference, this.simulationTime);

        // Compute tracking errors
        const ref = getReference(this.simulationTime);
        const posErr = {
            x: ref.position.x - droneState.position.x,
            y: ref.position.y - droneState.position.y,
            z: ref.position.z - droneState.position.z,
        };
        const posErrMag = Math.sqrt(posErr.x*posErr.x + posErr.y*posErr.y + posErr.z*posErr.z);
        const droneSpeed = Math.sqrt(droneState.velocity.x**2 + droneState.velocity.y**2 + droneState.velocity.z**2);

        // Compute Euler angles from quaternion using YXZ order (Three.js Y-up convention)
        // CRITICAL: Must use same order as MPC for consistency
        const q = droneState.orientation;
        const threeQuat = new THREE.Quaternion(q.x, q.y, q.z, q.w);
        const euler = new THREE.Euler().setFromQuaternion(threeQuat, 'YXZ');
        const roll = euler.z;   // Roll around Z
        const pitch = euler.x;  // Pitch around X
        // const yaw = euler.y;    // Yaw around Y (not used in tilt calculation)
        const tiltDeg = Math.sqrt(roll*roll + pitch*pitch) * 180 / Math.PI;
        const pitchDeg = pitch * 180 / Math.PI;

        // Record data for plots
        this.plotData.time.push(this.simulationTime);
        this.plotData.posError.push(posErrMag);
        this.plotData.velocity.push(droneSpeed);  // Total speed magnitude
        this.plotData.tilt.push(tiltDeg);
        this.plotData.pitch.push(pitchDeg);
        this.plotData.thrust.push(command.thrust);

        // Keep only last N points
        if (this.plotData.time.length > this.maxPlotPoints) {
            this.plotData.time.shift();
            this.plotData.posError.shift();
            this.plotData.velocity.shift();
            this.plotData.tilt.shift();
            this.plotData.pitch.shift();
            this.plotData.thrust.shift();
        }

        // DEBUG: Detailed logging for first 60 frames (~1 second)
        this.debugFrameCount++;
        if (this.debugFrameCount <= 60 && this.debugFrameCount % 10 === 0) {
            console.log(`=== FRAME ${this.debugFrameCount} (t=${this.simulationTime.toFixed(3)}s) ===`);
            console.log(`Drone: pos=(${droneState.position.x.toFixed(2)}, ${droneState.position.y.toFixed(2)}, ${droneState.position.z.toFixed(2)})`);
            console.log(`       vel=(${droneState.velocity.x.toFixed(2)}, ${droneState.velocity.y.toFixed(2)}, ${droneState.velocity.z.toFixed(2)})`);
            console.log(`Ref:   pos=(${ref.position.x.toFixed(2)}, ${ref.position.y.toFixed(2)}, ${ref.position.z.toFixed(2)})`);
            console.log(`       vel=(${ref.velocity.x.toFixed(2)}, ${ref.velocity.y.toFixed(2)}, ${ref.velocity.z.toFixed(2)})`);
            console.log(`       acc=(${ref.acceleration.x.toFixed(2)}, ${ref.acceleration.y.toFixed(2)}, ${ref.acceleration.z.toFixed(2)})`);
            console.log(`Error: ${posErrMag.toFixed(3)}m`);
            console.log(`Cmd: thrust=${command.thrust.toFixed(2)}, roll=${command.rollRate.toFixed(3)}, pitch=${command.pitchRate.toFixed(3)}, yaw=${command.yawRate.toFixed(3)}`);
            console.log(`Euler: roll=${(roll*180/Math.PI).toFixed(1)}°, pitch=${(pitch*180/Math.PI).toFixed(1)}°`);
        }

        // DEBUG: Update overlay every 10 frames
        if (this.debugFrameCount % 10 === 0 && this.debugOverlay) {
            this.debugOverlay.innerHTML = `
                <div style="color: #ffaa00; margin-bottom: 8px; font-weight: bold;">DEBUG PANEL</div>
                <div><b>Time:</b> ${this.simulationTime.toFixed(2)}s</div>
                <div style="margin-top: 6px; color: #88ff88;"><b>DRONE STATE:</b></div>
                <div>Pos: (${droneState.position.x.toFixed(2)}, ${droneState.position.y.toFixed(2)}, ${droneState.position.z.toFixed(2)})</div>
                <div>Vel: (${droneState.velocity.x.toFixed(2)}, ${droneState.velocity.y.toFixed(2)}, ${droneState.velocity.z.toFixed(2)})</div>
                <div>Speed: ${droneSpeed.toFixed(2)} m/s</div>
                <div>Tilt: ${tiltDeg.toFixed(1)}° (roll=${(roll*180/Math.PI).toFixed(1)}° pitch=${(pitch*180/Math.PI).toFixed(1)}°)</div>
                <div style="margin-top: 6px; color: #88aaff;"><b>REFERENCE:</b></div>
                <div>Pos: (${ref.position.x.toFixed(2)}, ${ref.position.y.toFixed(2)}, ${ref.position.z.toFixed(2)})</div>
                <div>Vel: (${ref.velocity.x.toFixed(2)}, ${ref.velocity.y.toFixed(2)}, ${ref.velocity.z.toFixed(2)})</div>
                <div>Accel: (${ref.acceleration.x.toFixed(2)}, ${ref.acceleration.y.toFixed(2)}, ${ref.acceleration.z.toFixed(2)})</div>
                <div style="margin-top: 6px; color: #ff8888;"><b>ERRORS:</b></div>
                <div>Pos Error: ${posErrMag.toFixed(2)}m</div>
                <div style="margin-top: 6px; color: #ffff88;"><b>COMMAND:</b></div>
                <div>Thrust: ${command.thrust.toFixed(2)} m/s² (gravity=9.81)</div>
                <div>Roll/Pitch/Yaw: ${command.rollRate.toFixed(2)} / ${command.pitchRate.toFixed(2)} / ${command.yawRate.toFixed(2)} rad/s</div>
                <div style="margin-top: 6px; color: #aaaaaa;"><b>dt:</b> ${(dt*1000).toFixed(1)}ms</div>
            `;
        }

        // Log important events every 0.5s
        if (this.debugFrameCount % 30 === 0) {
            this.log(`pos=(${droneState.position.x.toFixed(1)},${droneState.position.y.toFixed(1)},${droneState.position.z.toFixed(1)}) vel=(${droneState.velocity.x.toFixed(1)},${droneState.velocity.y.toFixed(1)},${droneState.velocity.z.toFixed(1)})`);
            this.log(`  ref=(${ref.position.x.toFixed(1)},${ref.position.y.toFixed(1)},${ref.position.z.toFixed(1)}) err=${posErrMag.toFixed(1)}m tilt=${tiltDeg.toFixed(0)}°`);
            this.log(`  cmd: T=${command.thrust.toFixed(1)} r=${command.rollRate.toFixed(2)} p=${command.pitchRate.toFixed(2)}`);
        }

        // Log warnings - less frequent
        if (posErrMag > 5.0 && this.debugFrameCount % 30 === 0) {
            this.log(`<span style="color:#ff6666">WARNING: Large error ${posErrMag.toFixed(1)}m</span>`);
        }
        if (droneState.position.y < 0.5 && this.debugFrameCount % 30 === 0) {
            this.log(`<span style="color:#ff0000">CRASH: y=${droneState.position.y.toFixed(2)} tilt=${tiltDeg.toFixed(0)}° thrust=${command.thrust.toFixed(1)}</span>`);
        }
        if (tiltDeg > 20 && this.debugFrameCount % 30 === 0) {
            this.log(`<span style="color:#ffaa00">High tilt: ${tiltDeg.toFixed(1)}° roll=${(roll*180/Math.PI).toFixed(1)}° pitch=${(pitch*180/Math.PI).toFixed(1)}°</span>`);
        }

        // Update drone dynamics
        this.drone.update(command, dt);

        // Update drone trail visualization
        this.updateDroneTrail();

        // Update simulation time
        this.simulationTime += dt;

        // Check gate passage
        const currentPosition = this.drone.getPosition();
        if (this.track.checkGatePassage(prevPosition, currentPosition)) {
            const passedGate = this.track.getCurrentGateIndex();
            this.track.advanceGate();
            this.log(`<span style="color:#00ff00">PASSED GATE ${passedGate}! Next: ${this.track.getCurrentGateIndex()}</span>`);
        }

        // Run detection
        const cameraPose = this.drone.getCameraPose();
        this.lastDetections = this.gateDetector.detectGates(this.track.gates, cameraPose);
    }

    /**
     * Update camera view panel
     */
    private updateCameraView(): void {
        const ctx = this.cameraCanvas.getContext('2d');
        if (!ctx) return;

        const w = this.cameraCanvas.width;
        const h = this.cameraCanvas.height;

        // Clear
        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        ctx.fillRect(0, 0, w, h);

        // Draw crosshair
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(w / 2, 0);
        ctx.lineTo(w / 2, h);
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();

        // Draw detected gates
        for (const detection of this.lastDetections) {
            const corners = detection.keypoints.corners;
            const confidence = detection.keypoints.confidence;

            // Gate outline
            ctx.strokeStyle = `rgba(0, 212, 255, ${confidence})`;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(corners[0].u, corners[0].v);
            for (let i = 1; i <= 4; i++) {
                ctx.lineTo(corners[i % 4].u, corners[i % 4].v);
            }
            ctx.stroke();

            // Corner markers
            ctx.fillStyle = `rgba(255, 255, 255, ${confidence})`;
            for (const corner of corners) {
                ctx.beginPath();
                ctx.arc(corner.u, corner.v, 3, 0, Math.PI * 2);
                ctx.fill();
            }

            // Gate ID
            const centerU = (corners[0].u + corners[2].u) / 2;
            const centerV = (corners[0].v + corners[2].v) / 2;
            ctx.fillStyle = '#a855f7';
            ctx.font = 'bold 14px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(String(detection.gateId + 1), centerU, centerV);
        }

        // Detection count
        ctx.fillStyle = '#00d4ff';
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`Detected: ${this.lastDetections.length}`, 8, 14);
    }

    /**
     * Update status bar
     */
    private updateStatusBar(): void {
        if (!this.statusBar) return;

        const status = this.getPipelineStatus();

        this.statusBar.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #888;">Detection:</span>
                <span style="color: #00d4ff;">${status.detection.gatesDetected} gates</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #888;">Next Gate:</span>
                <span style="color: #22c55e;">${status.estimation.distanceToNextGate.toFixed(1)}m</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #888;">Speed:</span>
                <span style="color: #a855f7;">${status.trajectory.currentSpeed.toFixed(1)} m/s</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #888;">Thrust:</span>
                <span style="color: #f59e0b;">${status.control.thrust.toFixed(1)} m/s²</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #888;">Lap:</span>
                <span style="color: #fff;">${this.track.getLapCount()}</span>
            </div>
        `;
    }

    /**
     * Get current pipeline status
     */
    private getPipelineStatus(): PipelineStatus {
        const dronePos = this.drone.getPosition();

        return {
            detection: {
                gatesDetected: this.lastDetections.length,
                activeGate: this.track.getCurrentGateIndex(),
            },
            estimation: {
                distanceToNextGate: this.track.getDistanceToCurrentGate(dronePos),
                estimatedPosition: this.drone.getState().position,
            },
            trajectory: {
                currentSpeed: this.drone.getSpeed(),
                maxSpeed: this.trajectoryGenerator.getMaxSpeed(),
                progressPercent: this.trajectory
                    ? (this.simulationTime % this.trajectory.totalDuration) / this.trajectory.totalDuration * 100
                    : 0,
            },
            control: {
                thrust: this.mpc.getReferenceWaypoints()[0]?.acceleration.y ?? 9.81,
                trackingError: 0,
            },
        };
    }

    /**
     * Update MPC prediction visualization
     */
    private updatePredictionVisualization(): void {
        // Remove old points
        if (this.predictionPoints) {
            this.scene.remove(this.predictionPoints);
        }

        const predictedStates = this.mpc.getPredictedStates();
        if (predictedStates.length === 0) return;

        const positions = new Float32Array(predictedStates.length * 3);
        for (let i = 0; i < predictedStates.length; i++) {
            const s = predictedStates[i];
            positions[i * 3] = s.position.x;
            positions[i * 3 + 1] = s.position.y;
            positions[i * 3 + 2] = s.position.z;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const material = new THREE.PointsMaterial({
            color: 0x22c55e,
            size: 0.1,
            opacity: 0.8,
            transparent: true,
        });

        this.predictionPoints = new THREE.Points(geometry, material);
        this.scene.add(this.predictionPoints);
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

        // Clear scene
        while (this.scene.children.length > 0) {
            this.scene.remove(this.scene.children[0]);
        }

        // Clear container
        this.container.innerHTML = '';
    }
}
