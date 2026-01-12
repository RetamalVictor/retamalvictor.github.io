/**
 * Camera Controller
 *
 * Manages camera modes for drone racing visualization.
 * Supports overview (spectator) and follow (chase) camera modes.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { lerpAngle } from '../../utils/mathUtils';
import { GatePosition, Vector3 } from './types';

export type CameraMode = 'overview' | 'follow';

export interface CameraConfig {
    followDistance: number;      // Distance behind drone (meters)
    followHeight: number;        // Height above drone (meters)
    headingSmoothness: number;   // Smoothing factor for heading (0-1)
    positionSmoothness: number;  // Smoothing factor for position (0-1)
    spectatorOffset: Vector3;    // Fixed spectator position offset
}

export const DEFAULT_CAMERA_CONFIG: CameraConfig = {
    followDistance: 15,
    followHeight: 6,
    headingSmoothness: 0.03,
    positionSmoothness: 0.06,
    spectatorOffset: { x: 30, y: 26, z: 30 },  // Stadium view offset from track center
};

export class CameraController {
    private camera: THREE.PerspectiveCamera;
    private controls: OrbitControls;
    private config: CameraConfig;

    private mode: CameraMode = 'overview';
    private smoothedHeading: number = 0;
    private smoothedPosition: THREE.Vector3 = new THREE.Vector3();
    private isInitialized: boolean = false;

    constructor(
        camera: THREE.PerspectiveCamera,
        controls: OrbitControls,
        config: Partial<CameraConfig> = {}
    ) {
        this.camera = camera;
        this.controls = controls;
        this.config = { ...DEFAULT_CAMERA_CONFIG, ...config };
    }

    /**
     * Get current camera mode
     */
    public getMode(): CameraMode {
        return this.mode;
    }

    /**
     * Set camera mode
     */
    public setMode(mode: CameraMode): void {
        if (mode === this.mode) return;

        this.mode = mode;

        if (mode === 'overview') {
            // Re-enable orbit controls for manual camera manipulation
            this.controls.enabled = true;
        } else {
            // Disable orbit controls - camera follows drone
            this.controls.enabled = false;
        }
    }

    /**
     * Update camera for follow mode
     * Should be called every frame when in follow mode
     */
    public updateFollowCamera(
        targetPos: THREE.Vector3,
        targetHeading: number
    ): void {
        if (this.mode !== 'follow') return;

        // Initialize smoothed values on first update
        if (!this.isInitialized) {
            this.smoothedHeading = targetHeading;
            this.smoothedPosition.copy(targetPos);
            this.isInitialized = true;
        }

        // Smooth the heading to prevent jitter
        this.smoothedHeading = lerpAngle(
            this.smoothedHeading,
            targetHeading,
            this.config.headingSmoothness
        );

        // Calculate camera position behind the drone
        const offsetX = -Math.sin(this.smoothedHeading) * this.config.followDistance;
        const offsetZ = -Math.cos(this.smoothedHeading) * this.config.followDistance;

        const targetCameraPos = new THREE.Vector3(
            targetPos.x + offsetX,
            targetPos.y + this.config.followHeight,
            targetPos.z + offsetZ
        );

        // Smooth camera position
        this.smoothedPosition.lerp(targetCameraPos, this.config.positionSmoothness);

        // Apply to camera
        this.camera.position.copy(this.smoothedPosition);
        this.camera.lookAt(targetPos);
    }

    /**
     * Set up spectator camera position based on gate layout
     */
    public setSpectatorPosition(gates: GatePosition[]): void {
        if (gates.length === 0) return;

        // Calculate center of all gates
        let centerX = 0, centerY = 0, centerZ = 0;
        for (const gate of gates) {
            centerX += gate.position.x;
            centerY += gate.position.y;
            centerZ += gate.position.z;
        }
        centerX /= gates.length;
        centerY /= gates.length;
        centerZ /= gates.length;

        // Position camera at fixed spectator location
        const offset = this.config.spectatorOffset;
        this.camera.position.set(
            centerX + offset.x,
            centerY + offset.y,
            centerZ + offset.z
        );

        // Look at center of track
        this.controls.target.set(centerX, centerY, centerZ);
        this.controls.update();
    }

    /**
     * Adjust camera position based on trajectory size (stadium view)
     * Dynamically positions camera to see entire trajectory
     */
    public adjustForTrajectory(
        trajectoryPoints: Vector3[],
        defaultHeight: number
    ): void {
        // Find max distance from origin to determine track size
        let maxDist = 0;
        for (const p of trajectoryPoints) {
            const dist = Math.sqrt(p.x * p.x + p.z * p.z);
            if (dist > maxDist) maxDist = dist;
        }

        // Position camera at distance proportional to track size
        const cameraDistance = maxDist * 2.5;
        this.camera.position.set(
            cameraDistance * 0.7,
            cameraDistance * 0.4,
            cameraDistance * 0.7
        );

        // Look at track center at default flight height
        this.controls.target.set(0, defaultHeight, 0);
        this.controls.update();
    }

    /**
     * Reset camera state (e.g., when simulation resets)
     */
    public reset(): void {
        this.isInitialized = false;
        this.smoothedHeading = 0;
        this.smoothedPosition.set(0, 0, 0);
    }

    /**
     * Get camera configuration
     */
    public getConfig(): CameraConfig {
        return { ...this.config };
    }

    /**
     * Update camera configuration
     */
    public setConfig(config: Partial<CameraConfig>): void {
        this.config = { ...this.config, ...config };
    }
}
