import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PinholeCamera } from './ibvs/Camera';
import { IBVSController } from './ibvs/Controller';
import { QuadrotorModel } from './ibvs/QuadrotorModel';
import { QuadrotorDynamics } from './ibvs/QuadrotorDynamics';

export interface VisualServoDemoConfig {
    containerId: string;
    backgroundColor?: number;
    debug?: boolean;
}

/**
 * Interactive Visual Servoing Demo
 * Shows a quadrotor using IBVS to track a draggable target
 */
export class VisualServoDemo {
    private container: HTMLElement;
    private backgroundColor: number;

    // Three.js
    private renderer!: THREE.WebGLRenderer;
    private scene!: THREE.Scene;
    private viewCamera!: THREE.PerspectiveCamera;
    private orbitControls!: OrbitControls;
    private animationId: number | null = null;

    // Objects
    private quadrotor!: QuadrotorModel;
    private targetGroup!: THREE.Group;
    private targetCorners: THREE.Vector3[] = [];

    // Trajectory visualization
    private trajectoryLine!: THREE.Line;
    private trajectoryPoints: THREE.Vector3[] = [];

    // IBVS components
    private droneCamera!: PinholeCamera;
    private controller!: IBVSController;
    private desiredFeatures!: Float32Array;
    private dynamics!: QuadrotorDynamics;

    // Camera sub-view
    private cameraCanvas!: HTMLCanvasElement;
    private cameraCtx!: CanvasRenderingContext2D;
    private showSimulatedView: boolean = false;
    private cameraPipContainer!: HTMLElement;


    // Dragging
    private raycaster: THREE.Raycaster;
    private mouse: THREE.Vector2;
    private isDragging: boolean = false;
    private dragPlane: THREE.Plane;

    // State
    private isInitialized: boolean = false;
    private lastTime: number = 0;
    private controlEnabled: boolean = true;

    // Constants
    private readonly TARGET_SIZE = 1.0;
    private cameraWidth: number = 192;
    private cameraHeight: number = 144;

    constructor(config: VisualServoDemoConfig) {
        this.container = document.getElementById(config.containerId)!;
        this.backgroundColor = config.backgroundColor || 0x0a0a0f;

        if (!this.container) {
            throw new Error(`Container with id ${config.containerId} not found`);
        }

        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        this.dragPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);

        this.init();
    }

    private async init(): Promise<void> {
        this.container.innerHTML = '';

        // Set camera panel size based on screen width (responsive)
        const isMobile = window.innerWidth < 640;
        this.cameraWidth = isMobile ? 120 : 192;
        this.cameraHeight = isMobile ? 90 : 144;

        try {
            this.setupRenderer();
            this.setupScene();
            this.setupViewCamera();
            this.createWorld();
            this.createTarget();
            this.createQuadrotor();
            this.createTrajectoryLine();
            this.setupIBVS();
            this.createCameraView();
            this.setupInteraction();
            this.setupResize();
            this.computeDesiredFeatures();
            this.isInitialized = true;
            this.lastTime = performance.now() / 1000;
            this.animate();
        } catch (error) {
            console.error('VisualServoDemo initialization failed:', error);
            this.showFallback();
        }
    }

    private setupRenderer(): void {
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: false
        });

        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setClearColor(this.backgroundColor);

        this.container.appendChild(this.renderer.domElement);
    }

    private setupScene(): void {
        this.scene = new THREE.Scene();

        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        // Add directional light for better depth perception
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.4);
        dirLight.position.set(5, 5, -5);
        this.scene.add(dirLight);
    }

    private setupViewCamera(): void {
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.viewCamera = new THREE.PerspectiveCamera(50, aspect, 0.1, 100);

        // Position camera to see both drone and target from the side
        this.viewCamera.position.set(6, 4, 0);
        this.viewCamera.lookAt(0, 0, 2);

        // Add orbit controls
        this.orbitControls = new OrbitControls(this.viewCamera, this.renderer.domElement);
        this.orbitControls.enableDamping = true;
        this.orbitControls.dampingFactor = 0.05;
        this.orbitControls.target.set(0, 0, 2);
        // Enable middle mouse button for panning
        this.orbitControls.mouseButtons = {
            LEFT: THREE.MOUSE.ROTATE,
            MIDDLE: THREE.MOUSE.PAN,
            RIGHT: THREE.MOUSE.ROTATE
        };
        this.orbitControls.update();
    }

    private createWorld(): void {
        // Ground plane - horizontal floor at Y = -1 (below the drone)
        const groundSize = 16;
        const floorY = -1;

        // Grid on floor (GridHelper is already horizontal by default in XZ plane)
        const gridHelper = new THREE.GridHelper(groundSize, 20, 0x2a2a3e, 0x1e1e2e);
        gridHelper.position.y = floorY;
        this.scene.add(gridHelper);

        // Semi-transparent floor surface
        const groundGeometry = new THREE.PlaneGeometry(groundSize, groundSize);
        const groundMaterial = new THREE.MeshBasicMaterial({
            color: 0x1a1a2e,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.3
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;  // Rotate to horizontal (XZ plane)
        ground.position.y = floorY - 0.01;  // Slightly below grid
        this.scene.add(ground);
    }

    private createTarget(): void {
        this.targetGroup = new THREE.Group();

        const halfSize = this.TARGET_SIZE / 2;

        // Define corner positions in target's local frame
        this.targetCorners = [
            new THREE.Vector3(-halfSize, -halfSize, 0),
            new THREE.Vector3(halfSize, -halfSize, 0),
            new THREE.Vector3(halfSize, halfSize, 0),
            new THREE.Vector3(-halfSize, halfSize, 0)
        ];

        // Create target square outline (thicker)
        const squareMaterial = new THREE.LineBasicMaterial({
            color: 0x00d4ff,
            linewidth: 3
        });

        const squarePoints = [
            ...this.targetCorners,
            this.targetCorners[0]
        ];
        const squareGeometry = new THREE.BufferGeometry().setFromPoints(squarePoints);
        const squareLine = new THREE.Line(squareGeometry, squareMaterial);
        this.targetGroup.add(squareLine);

        // Add corner spheres for better visibility
        const cornerMaterial = new THREE.MeshBasicMaterial({ color: 0x00d4ff });
        for (const corner of this.targetCorners) {
            const sphere = new THREE.Mesh(
                new THREE.SphereGeometry(0.05, 8, 8),
                cornerMaterial
            );
            sphere.position.copy(corner);
            this.targetGroup.add(sphere);
        }

        // Add fill for dragging
        const fillGeometry = new THREE.PlaneGeometry(this.TARGET_SIZE, this.TARGET_SIZE);
        const fillMaterial = new THREE.MeshBasicMaterial({
            color: 0x00d4ff,
            transparent: true,
            opacity: 0.15,
            side: THREE.DoubleSide
        });
        const fillMesh = new THREE.Mesh(fillGeometry, fillMaterial);
        fillMesh.name = 'targetFill';
        this.targetGroup.add(fillMesh);

        // Add "TARGET" label
        this.addTargetLabel();

        // Position target in front of drone
        this.targetGroup.position.set(0, 0, 4);

        this.scene.add(this.targetGroup);
    }

    private addTargetLabel(): void {
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 32;
        const ctx = canvas.getContext('2d')!;
        ctx.fillStyle = '#00d4ff';
        ctx.font = 'bold 20px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('TARGET', 64, 22);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(material);
        sprite.position.set(0, 0.8, 0);
        sprite.scale.set(1, 0.25, 1);
        this.targetGroup.add(sprite);
    }

    // Visual aid: line from drone showing camera look direction
    private lookDirectionLine!: THREE.Line;

    private createQuadrotor(): void {
        this.quadrotor = new QuadrotorModel();
        this.dynamics = new QuadrotorDynamics();

        // Initial position: at origin, facing +Z (toward target)
        this.dynamics.reset(0, 0, 0);
        this.syncDroneFromDynamics();

        this.scene.add(this.quadrotor.mesh);

        // Add a line showing the camera look direction (from drone toward target)
        const lineMaterial = new THREE.LineBasicMaterial({
            color: 0xa855f7,
            transparent: true,
            opacity: 0.5
        });
        const lineGeometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, 0, 2)  // Points 2 units in +Z
        ]);
        this.lookDirectionLine = new THREE.Line(lineGeometry, lineMaterial);
        this.scene.add(this.lookDirectionLine);
    }

    private createTrajectoryLine(): void {
        const material = new THREE.LineBasicMaterial({
            color: 0xa855f7,
            transparent: true,
            opacity: 0.6
        });
        const geometry = new THREE.BufferGeometry();
        this.trajectoryLine = new THREE.Line(geometry, material);
        this.scene.add(this.trajectoryLine);
    }

    private updateTrajectory(): void {
        // Add current position to trajectory
        this.trajectoryPoints.push(this.quadrotor.mesh.position.clone());

        // Keep only last 200 points
        if (this.trajectoryPoints.length > 200) {
            this.trajectoryPoints.shift();
        }

        // Update line geometry
        this.trajectoryLine.geometry.setFromPoints(this.trajectoryPoints);
    }

    private setupIBVS(): void {
        // Create drone's camera
        this.droneCamera = new PinholeCamera({
            focalLength: 8,
            sensorWidth: this.cameraWidth,
            sensorHeight: this.cameraHeight,
            sensorSizeMM: 36
        });

        // Create IBVS controller
        this.controller = new IBVSController(
            this.droneCamera.focalLengthPx,
            0.5,   // gain (low for smooth motion with dynamics)
            0.3    // clip threshold
        );
    }

    private computeDesiredFeatures(): void {
        // Desired pose: 2m in front of target, centered
        const desiredDistance = 2.0;

        const desiredDronePos = this.targetGroup.position.clone();
        desiredDronePos.z -= desiredDistance;

        // Set camera to desired position to compute what features should look like
        this.droneCamera.setPose(
            desiredDronePos.x,
            desiredDronePos.y,
            desiredDronePos.z,
            0, 0, 0
        );

        // Project target corners from desired position
        const worldCorners = this.getTargetWorldCorners();
        const { imagePoints } = this.droneCamera.projectPoints(worldCorners);
        this.desiredFeatures = imagePoints;

        // Reset camera to drone pose
        this.syncCameraToDrone();
    }

    private getTargetWorldCorners(): THREE.Vector3[] {
        // Make sure world matrix is updated
        this.targetGroup.updateMatrixWorld(true);

        return this.targetCorners.map(corner => {
            const worldCorner = corner.clone();
            worldCorner.applyMatrix4(this.targetGroup.matrixWorld);
            return worldCorner;
        });
    }

    private syncCameraToDrone(): void {
        const pos = this.quadrotor.mesh.position;
        const rot = this.quadrotor.mesh.rotation;
        this.droneCamera.setPose(pos.x, pos.y, pos.z, rot.x, rot.y, rot.z);
    }

    private createCameraView(): void {
        this.cameraPipContainer = document.createElement('div');
        this.cameraPipContainer.className = 'absolute bottom-2 left-2 sm:bottom-3 sm:left-3 rounded-lg overflow-hidden border border-dark-border bg-dark-surface/90 shadow-lg';
        this.cameraPipContainer.style.width = `${this.cameraWidth}px`;
        this.cameraPipContainer.style.zIndex = '10';

        this.cameraCanvas = document.createElement('canvas');
        this.cameraCanvas.width = this.cameraWidth;
        this.cameraCanvas.height = this.cameraHeight;
        this.cameraCanvas.className = 'block';
        this.cameraPipContainer.appendChild(this.cameraCanvas);

        this.cameraCtx = this.cameraCanvas.getContext('2d')!;

        // Control bar (smaller on mobile)
        const isMobile = window.innerWidth < 640;
        const controlBar = document.createElement('div');
        controlBar.className = `flex items-center justify-between px-1.5 py-0.5 sm:px-2 sm:py-1 bg-dark-bg/50 ${isMobile ? 'text-[10px]' : 'text-xs'}`;

        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'text-gray-400 hover:text-accent-cyan transition-colors';
        toggleBtn.textContent = 'Features';
        toggleBtn.addEventListener('click', () => {
            this.showSimulatedView = !this.showSimulatedView;
            toggleBtn.textContent = this.showSimulatedView ? 'Simulated' : 'Features';
        });
        controlBar.appendChild(toggleBtn);

        const errorDisplay = document.createElement('span');
        errorDisplay.id = 'error-display';
        errorDisplay.className = 'text-gray-500';
        errorDisplay.textContent = 'Err: 0.0';
        controlBar.appendChild(errorDisplay);

        this.cameraPipContainer.appendChild(controlBar);
        this.container.appendChild(this.cameraPipContainer);
    }

    private renderCameraView(): void {
        const ctx = this.cameraCtx;
        const w = this.cameraWidth;
        const h = this.cameraHeight;

        ctx.fillStyle = '#12121a';
        ctx.fillRect(0, 0, w, h);

        const worldCorners = this.getTargetWorldCorners();
        const { imagePoints: currentFeatures, visible } = this.droneCamera.projectPoints(worldCorners);

        if (this.showSimulatedView) {
            this.renderSimulatedView(ctx, currentFeatures, visible);
        } else {
            this.renderFeatureView(ctx, currentFeatures, visible);
        }

        const error = this.controller.computeError(this.desiredFeatures, currentFeatures);

        const errorDisplay = document.getElementById('error-display');
        if (errorDisplay) {
            errorDisplay.textContent = `Err: ${error.toFixed(1)}`;
            errorDisplay.className = error < 10 ? 'text-green-400' : 'text-gray-500';
        }
    }

    private renderFeatureView(
        ctx: CanvasRenderingContext2D,
        currentFeatures: Float32Array,
        visible: boolean[]
    ): void {
        const w = this.cameraWidth;
        const h = this.cameraHeight;

        // Crosshair
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(w / 2, 0);
        ctx.lineTo(w / 2, h);
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();

        // Desired features (cyan)
        ctx.strokeStyle = '#00d4ff';
        ctx.lineWidth = 2;
        for (let i = 0; i < 4; i++) {
            const x = this.desiredFeatures[i * 2];
            const y = this.desiredFeatures[i * 2 + 1];
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.stroke();
        }

        // Desired square
        ctx.beginPath();
        ctx.moveTo(this.desiredFeatures[0], this.desiredFeatures[1]);
        for (let i = 1; i < 4; i++) {
            ctx.lineTo(this.desiredFeatures[i * 2], this.desiredFeatures[i * 2 + 1]);
        }
        ctx.closePath();
        ctx.stroke();

        // Current features (purple)
        ctx.fillStyle = '#a855f7';
        for (let i = 0; i < 4; i++) {
            if (!visible[i]) continue;
            const x = currentFeatures[i * 2];
            const y = currentFeatures[i * 2 + 1];
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fill();
        }

        // Current square
        ctx.strokeStyle = '#a855f7';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < 4; i++) {
            if (!visible[i]) continue;
            if (!started) {
                ctx.moveTo(currentFeatures[i * 2], currentFeatures[i * 2 + 1]);
                started = true;
            } else {
                ctx.lineTo(currentFeatures[i * 2], currentFeatures[i * 2 + 1]);
            }
        }
        if (started && visible[0]) {
            ctx.lineTo(currentFeatures[0], currentFeatures[1]);
        }
        ctx.stroke();

        // Error lines
        ctx.strokeStyle = '#ff4444';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        for (let i = 0; i < 4; i++) {
            if (!visible[i]) continue;
            ctx.beginPath();
            ctx.moveTo(this.desiredFeatures[i * 2], this.desiredFeatures[i * 2 + 1]);
            ctx.lineTo(currentFeatures[i * 2], currentFeatures[i * 2 + 1]);
            ctx.stroke();
        }
        ctx.setLineDash([]);

        // Legend
        ctx.font = '10px monospace';
        ctx.fillStyle = '#00d4ff';
        ctx.fillText('○ Desired', 5, 12);
        ctx.fillStyle = '#a855f7';
        ctx.fillText('● Current', 5, 24);
    }

    private renderSimulatedView(
        ctx: CanvasRenderingContext2D,
        currentFeatures: Float32Array,
        visible: boolean[]
    ): void {
        ctx.strokeStyle = '#00d4ff';
        ctx.lineWidth = 2;
        ctx.fillStyle = 'rgba(0, 212, 255, 0.1)';

        ctx.beginPath();
        let started = false;
        for (let i = 0; i < 4; i++) {
            if (!visible[i]) continue;
            if (!started) {
                ctx.moveTo(currentFeatures[i * 2], currentFeatures[i * 2 + 1]);
                started = true;
            } else {
                ctx.lineTo(currentFeatures[i * 2], currentFeatures[i * 2 + 1]);
            }
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // Corner markers
        for (let i = 0; i < 4; i++) {
            if (!visible[i]) continue;
            const x = currentFeatures[i * 2];
            const y = currentFeatures[i * 2 + 1];

            ctx.strokeStyle = '#ff4444';
            ctx.lineWidth = 2;
            const size = 8;
            ctx.beginPath();
            ctx.moveTo(x - size, y - size);
            ctx.lineTo(x + size, y + size);
            ctx.moveTo(x + size, y - size);
            ctx.lineTo(x - size, y + size);
            ctx.stroke();
        }

        ctx.font = '10px monospace';
        ctx.fillStyle = '#888';
        ctx.fillText('Camera View', 5, 12);
    }

    private setupInteraction(): void {
        const canvas = this.renderer.domElement;

        canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        canvas.addEventListener('mouseleave', this.onMouseUp.bind(this));

        canvas.addEventListener('touchstart', this.onTouchStart.bind(this), { passive: false });
        canvas.addEventListener('touchmove', this.onTouchMove.bind(this), { passive: false });
        canvas.addEventListener('touchend', this.onMouseUp.bind(this));
    }

    private updateMousePosition(clientX: number, clientY: number): void {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    }

    private onMouseDown(event: MouseEvent): void {
        this.updateMousePosition(event.clientX, event.clientY);
        this.raycaster.setFromCamera(this.mouse, this.viewCamera);

        const targetMesh = this.targetGroup.getObjectByName('targetFill');
        if (targetMesh) {
            const intersects = this.raycaster.intersectObject(targetMesh);
            if (intersects.length > 0) {
                this.isDragging = true;
                this.orbitControls.enabled = false;
                this.dragPlane.setFromNormalAndCoplanarPoint(
                    this.viewCamera.getWorldDirection(new THREE.Vector3()).negate(),
                    this.targetGroup.position
                );
            }
        }
    }

    private onMouseMove(event: MouseEvent): void {
        if (!this.isDragging) return;

        this.updateMousePosition(event.clientX, event.clientY);
        this.raycaster.setFromCamera(this.mouse, this.viewCamera);

        const intersection = new THREE.Vector3();
        if (this.raycaster.ray.intersectPlane(this.dragPlane, intersection)) {
            intersection.x = Math.max(-4, Math.min(4, intersection.x));
            intersection.y = Math.max(-4, Math.min(4, intersection.y));
            intersection.z = Math.max(1, Math.min(8, intersection.z));

            this.targetGroup.position.copy(intersection);
            this.computeDesiredFeatures();
        }
    }

    private onMouseUp(): void {
        this.isDragging = false;
        this.orbitControls.enabled = true;
    }

    private onTouchStart(event: TouchEvent): void {
        event.preventDefault();
        if (event.touches.length > 0) {
            const touch = event.touches[0];
            this.updateMousePosition(touch.clientX, touch.clientY);
            this.raycaster.setFromCamera(this.mouse, this.viewCamera);

            const targetMesh = this.targetGroup.getObjectByName('targetFill');
            if (targetMesh) {
                const intersects = this.raycaster.intersectObject(targetMesh);
                if (intersects.length > 0) {
                    this.isDragging = true;
                    this.orbitControls.enabled = false;
                    this.dragPlane.setFromNormalAndCoplanarPoint(
                        this.viewCamera.getWorldDirection(new THREE.Vector3()).negate(),
                        this.targetGroup.position
                    );
                }
            }
        }
    }

    private onTouchMove(event: TouchEvent): void {
        event.preventDefault();
        if (!this.isDragging || event.touches.length === 0) return;

        const touch = event.touches[0];
        this.updateMousePosition(touch.clientX, touch.clientY);
        this.raycaster.setFromCamera(this.mouse, this.viewCamera);

        const intersection = new THREE.Vector3();
        if (this.raycaster.ray.intersectPlane(this.dragPlane, intersection)) {
            intersection.x = Math.max(-4, Math.min(4, intersection.x));
            intersection.y = Math.max(-4, Math.min(4, intersection.y));
            intersection.z = Math.max(1, Math.min(8, intersection.z));

            this.targetGroup.position.copy(intersection);
            this.computeDesiredFeatures();
        }
    }

    private setupResize(): void {
        const resizeObserver = new ResizeObserver(() => {
            this.handleResize();
        });
        resizeObserver.observe(this.container);
    }

    private handleResize(): void {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.viewCamera.aspect = width / height;
        this.viewCamera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    private animate = (): void => {
        this.animationId = requestAnimationFrame(this.animate);

        const now = performance.now() / 1000;
        const dt = Math.min(now - this.lastTime, 0.05);
        this.lastTime = now;

        if (this.isInitialized && this.controlEnabled) {
            this.updateIBVS(dt);
            this.updateTrajectory();
        }

        // Update look direction line to follow drone
        if (this.lookDirectionLine) {
            const dronePos = this.quadrotor.mesh.position;
            const forward = this.quadrotor.getForwardVector().multiplyScalar(2);
            const linePoints = [
                dronePos.clone(),
                dronePos.clone().add(forward)
            ];
            this.lookDirectionLine.geometry.setFromPoints(linePoints);
        }

        this.quadrotor.animateRotors(1);
        this.orbitControls.update();
        this.renderCameraView();

        this.renderer.render(this.scene, this.viewCamera);
    };

    private updateIBVS(dt: number): void {
        this.syncCameraToDrone();

        const worldCorners = this.getTargetWorldCorners();
        const { imagePoints: currentFeatures, cameraPoints } = this.droneCamera.projectPoints(worldCorners);

        const depths = new Float32Array(cameraPoints.map(p => p.z));

        // Check if target is visible
        const allVisible = depths.every(d => d > 0.1);
        if (!allVisible) {
            // Target behind camera - command zero velocity but still update dynamics
            // This lets the drone slow down naturally via drag
            const zeroVelocity = new Float32Array(6);
            this.dynamics.update(zeroVelocity, dt);
            this.syncDroneFromDynamics();
            return;
        }

        const velocity = this.controller.computeControl(
            this.desiredFeatures,
            currentFeatures,
            depths,
            this.droneCamera.cx,
            this.droneCamera.cy
        );

        // Update underactuated quadrotor dynamics
        // velocity is in camera frame: [vx, vy, vz, wx, wy, wz]
        this.dynamics.update(velocity, dt);

        // Sync mesh position and orientation from dynamics
        this.syncDroneFromDynamics();
    }

    /**
     * Sync quadrotor mesh from dynamics state
     */
    private syncDroneFromDynamics(): void {
        const [x, y, z] = this.dynamics.getPosition();
        const [roll, pitch, yaw] = this.dynamics.getOrientation();

        this.quadrotor.mesh.position.set(x, y, z);
        this.quadrotor.mesh.rotation.set(roll, pitch, yaw, 'XYZ');
    }

    public reset(): void {
        this.dynamics.reset(0, 0, 0);
        this.syncDroneFromDynamics();
        this.targetGroup.position.set(0, 0, 4);
        this.trajectoryPoints = [];
        this.computeDesiredFeatures();
    }

    public destroy(): void {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }

        if (this.renderer) {
            this.renderer.dispose();
        }

        this.orbitControls?.dispose();

        if (this.container && this.renderer?.domElement?.parentNode === this.container) {
            this.container.removeChild(this.renderer.domElement);
        }

        if (this.cameraPipContainer?.parentNode === this.container) {
            this.container.removeChild(this.cameraPipContainer);
        }
    }

    private showFallback(): void {
        this.container.innerHTML = `
            <div class="w-full h-full flex items-center justify-center relative overflow-hidden bg-dark-surface">
                <div class="text-center text-gray-500">
                    <div class="w-16 h-16 mx-auto mb-4 rounded-full bg-accent-cyan/20 animate-pulse"></div>
                    <p class="text-sm">Visual Servoing Demo</p>
                    <p class="text-xs mt-2 text-gray-600">WebGL required</p>
                </div>
            </div>
        `;
    }
}
