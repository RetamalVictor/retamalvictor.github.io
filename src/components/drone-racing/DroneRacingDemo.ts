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
import { RacingConfig, DEFAULT_CONFIG, Trajectory, PipelineStatus, GateDetection } from './types';

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

    // Animation
    private animationFrameId: number = 0;

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

        // Add track
        this.scene.add(this.track.trackGroup);

        // Add drone
        this.scene.add(this.drone.mesh);

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
     * Generate trajectory through track
     */
    private generateTrajectory(): void {
        this.trajectory = this.trajectoryGenerator.generateRacingTrajectory(this.track);

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

        // Position drone at first gate approach
        const firstGate = this.track.gates[0];
        const startPos = firstGate.getApproachPosition(2);
        this.drone.reset({ x: startPos.x, y: startPos.y, z: startPos.z });

        // Set initial heading toward first gate
        const toGate = firstGate.getCenter().sub(startPos);
        const heading = Math.atan2(toGate.x, toGate.z);
        this.drone.setHeading(heading);

        // Reset MPC
        this.mpc.reset();

        // Regenerate trajectory
        this.generateTrajectory();
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

        // Render
        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Update simulation step
     */
    private updateSimulation(dt: number): void {
        if (!this.trajectory) return;

        // Store previous position for gate detection
        const prevPosition = this.drone.getPosition();

        // Get reference waypoint
        const getReference = (t: number) =>
            this.trajectoryGenerator.getWaypointAtTime(this.trajectory!, t);

        // Compute MPC control
        const droneState = this.drone.getState();
        const command = this.mpc.computeControl(droneState, getReference, this.simulationTime);

        // Update drone dynamics
        this.drone.update(command, dt);

        // Update simulation time
        this.simulationTime += dt;

        // Check gate passage
        const currentPosition = this.drone.getPosition();
        if (this.track.checkGatePassage(prevPosition, currentPosition)) {
            this.track.advanceGate();
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
