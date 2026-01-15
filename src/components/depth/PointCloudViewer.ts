/**
 * PointCloudViewer - Three.js point cloud visualization for depth maps
 *
 * Features:
 * - Back-projects depth to 3D using approximate camera intrinsics
 * - Turbo colormap for depth visualization
 * - Temporal smoothing (EMA) for stability
 * - Subsampling for performance
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export interface PointCloudConfig {
    subsample: number;      // Sample every N pixels (2-4 recommended)
    depthScale: number;     // Scale factor for depth values
    nearClip: number;       // Near clipping distance
    farClip: number;        // Far clipping distance
    smoothingFactor: number; // EMA smoothing (0 = no smoothing, 0.9 = heavy)
    fov: number;            // Approximate camera FOV in degrees
    backgroundRemoval: number; // 0 = show all, 0.5 = remove far 50%, 1 = remove all background
}

const DEFAULT_CONFIG: PointCloudConfig = {
    subsample: 2,
    depthScale: 1.0,
    nearClip: 0.1,
    farClip: 10.0,
    smoothingFactor: 0.5,
    fov: 60,
    backgroundRemoval: 0.4  // Remove far 40% by default
};

// Turbo colormap (approximation)
function turboColormap(t: number): THREE.Color {
    // Clamp t to [0, 1]
    t = Math.max(0, Math.min(1, t));

    // Approximate turbo colormap
    const r = Math.max(0, Math.min(1,
        0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))))
    ));
    const g = Math.max(0, Math.min(1,
        0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))))
    ));
    const b = Math.max(0, Math.min(1,
        0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))))
    ));

    return new THREE.Color(r, g, b);
}

export class PointCloudViewer {
    private container: HTMLElement;
    private config: PointCloudConfig;

    private scene: THREE.Scene;
    private camera: THREE.PerspectiveCamera;
    private renderer: THREE.WebGLRenderer;
    private controls: OrbitControls;
    private pointCloud: THREE.Points | null = null;

    // Geometry buffers
    private positions: Float32Array | null = null;
    private colors: Float32Array | null = null;
    private smoothedDepth: Float32Array | null = null;

    private depthWidth = 0;
    private depthHeight = 0;
    private animationId: number | null = null;
    private isDestroyed = false;

    constructor(container: HTMLElement, config: Partial<PointCloudConfig> = {}) {
        this.container = container;
        this.config = { ...DEFAULT_CONFIG, ...config };

        // Initialize Three.js
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);

        // Camera - on negative Z side looking at origin
        const aspect = container.clientWidth / container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
        this.camera.position.set(0, 0, -15);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(this.renderer.domElement);

        // Controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.rotateSpeed = 0.5;

        // Add axes helper (small, for orientation)
        const axesHelper = new THREE.AxesHelper(0.2);
        this.scene.add(axesHelper);

        // Handle resize
        this.handleResize = this.handleResize.bind(this);
        window.addEventListener('resize', this.handleResize);

        // Start render loop
        this.animate();
    }

    /**
     * Update point cloud with new depth data
     */
    update(depthMap: Float32Array, width: number, height: number, _rgbData?: Uint8ClampedArray): void {
        if (this.isDestroyed) return;

        const { subsample, depthScale, nearClip, farClip, smoothingFactor, fov, backgroundRemoval } = this.config;

        // Initialize or resize buffers if needed
        if (this.depthWidth !== width || this.depthHeight !== height) {
            this.initBuffers(width, height);
        }

        // Apply temporal smoothing (EMA)
        if (this.smoothedDepth && smoothingFactor > 0) {
            for (let i = 0; i < depthMap.length; i++) {
                this.smoothedDepth[i] = smoothingFactor * this.smoothedDepth[i] + (1 - smoothingFactor) * depthMap[i];
            }
        } else if (this.smoothedDepth) {
            this.smoothedDepth.set(depthMap);
        }

        const depth = this.smoothedDepth || depthMap;

        // Calculate camera intrinsics (approximate)
        const fovRad = (fov * Math.PI) / 180;
        const fx = width / (2 * Math.tan(fovRad / 2));
        const fy = fx; // Assume square pixels
        const cx = width / 2;
        const cy = height / 2;

        // Find depth range for normalization
        let minDepth = Infinity;
        let maxDepth = -Infinity;
        for (let i = 0; i < depth.length; i++) {
            if (depth[i] < minDepth) minDepth = depth[i];
            if (depth[i] > maxDepth) maxDepth = depth[i];
        }
        const depthRange = maxDepth - minDepth || 1;

        // Update point positions and colors
        let pointIdx = 0;

        for (let y = 0; y < height; y += subsample) {
            for (let x = 0; x < width; x += subsample) {
                const depthIdx = y * width + x;
                const depthValue = depth[depthIdx];

                // Normalize depth to [0, 1] for coloring
                const normalizedDepth = (depthValue - minDepth) / depthRange;

                // Background removal: skip far points (low normalizedDepth = far from camera)
                if (backgroundRemoval > 0 && normalizedDepth < backgroundRemoval) {
                    this.positions![pointIdx * 3] = 0;
                    this.positions![pointIdx * 3 + 1] = 0;
                    this.positions![pointIdx * 3 + 2] = -1000;
                    pointIdx++;
                    continue;
                }

                // Map normalized depth directly to z distance
                const z = normalizedDepth * (farClip - nearClip) + nearClip;

                // Skip points outside clipping range
                if (z < nearClip || z > farClip) {
                    // Set position far away (effectively hide)
                    this.positions![pointIdx * 3] = 0;
                    this.positions![pointIdx * 3 + 1] = 0;
                    this.positions![pointIdx * 3 + 2] = -1000;
                    pointIdx++;
                    continue;
                }

                // Back-project to 3D (mirror X for natural webcam view)
                const X = -((x - cx) * z) / fx * depthScale; // Mirrored
                const Y = -((y - cy) * z) / fy * depthScale; // Flip Y
                const Z = -z * depthScale; // Negative Z (into screen)

                this.positions![pointIdx * 3] = X;
                this.positions![pointIdx * 3 + 1] = Y;
                this.positions![pointIdx * 3 + 2] = Z;

                // Color from turbo colormap (close = warm, far = cool)
                const color = turboColormap(normalizedDepth);
                this.colors![pointIdx * 3] = color.r;
                this.colors![pointIdx * 3 + 1] = color.g;
                this.colors![pointIdx * 3 + 2] = color.b;

                pointIdx++;
            }
        }

        // Update geometry
        if (this.pointCloud) {
            const geometry = this.pointCloud.geometry as THREE.BufferGeometry;
            geometry.attributes.position.needsUpdate = true;
            geometry.attributes.color.needsUpdate = true;
        }
    }

    private initBuffers(width: number, height: number): void {
        this.depthWidth = width;
        this.depthHeight = height;

        const { subsample } = this.config;
        const sampledWidth = Math.floor(width / subsample);
        const sampledHeight = Math.floor(height / subsample);
        const numPoints = sampledWidth * sampledHeight;

        // Allocate buffers
        this.positions = new Float32Array(numPoints * 3);
        this.colors = new Float32Array(numPoints * 3);
        this.smoothedDepth = new Float32Array(width * height);

        // Create point cloud geometry
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(this.colors, 3));

        // Point material
        const material = new THREE.PointsMaterial({
            size: 0.01,
            vertexColors: true,
            sizeAttenuation: true
        });

        // Remove old point cloud if exists
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            (this.pointCloud.material as THREE.Material).dispose();
        }

        // Create new point cloud
        this.pointCloud = new THREE.Points(geometry, material);
        this.scene.add(this.pointCloud);
    }

    private animate = (): void => {
        if (this.isDestroyed) return;

        this.animationId = requestAnimationFrame(this.animate);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    };

    private handleResize(): void {
        if (this.isDestroyed) return;

        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    /**
     * Update configuration
     */
    setConfig(config: Partial<PointCloudConfig>): void {
        this.config = { ...this.config, ...config };
    }

    /**
     * Get current configuration
     */
    getConfig(): PointCloudConfig {
        return { ...this.config };
    }

    /**
     * Reset camera to default position
     */
    resetCamera(): void {
        this.camera.position.set(0, 0, -15);
        this.camera.lookAt(0, 0, 0);
        this.controls.reset();
    }

    /**
     * Cleanup resources
     */
    destroy(): void {
        this.isDestroyed = true;

        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }

        window.removeEventListener('resize', this.handleResize);

        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            (this.pointCloud.material as THREE.Material).dispose();
        }

        this.controls.dispose();
        this.renderer.dispose();

        if (this.renderer.domElement.parentNode) {
            this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
        }
    }
}
