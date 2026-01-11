import * as THREE from 'three';
import { Vector3, Quaternion, GatePose } from '../types';

/**
 * Race Gate Model
 *
 * Represents a racing gate with:
 * - 4 corner keypoints (TL, TR, BR, BL) for detection
 * - 3D mesh for visualization
 * - Pose (position + orientation) in world frame
 *
 * Gate local frame:
 * - Origin at gate center
 * - X axis: right
 * - Y axis: up
 * - Z axis: forward (through the gate opening)
 *
 * Based on drone-racing-msgs GatePoses message format.
 */
export class RaceGate {
    public readonly id: number;
    public readonly innerSize: number;  // gate opening size (meters)

    // 3D mesh group
    public readonly mesh: THREE.Group;

    // Gate pose in world frame
    private position: THREE.Vector3;
    private quaternion: THREE.Quaternion;

    // Corner positions in local frame (fixed)
    private readonly localCorners: THREE.Vector3[];

    // Colors
    private readonly frameColor = 0x00d4ff;     // Cyan (accent)
    private readonly cornerColor = 0xffffff;    // White
    private readonly labelColor = 0xa855f7;     // Purple

    constructor(id: number, innerSize: number = 1.52) {
        this.id = id;
        this.innerSize = innerSize;

        // Initialize pose at origin
        this.position = new THREE.Vector3(0, 0, 0);
        this.quaternion = new THREE.Quaternion(0, 0, 0, 1);

        // Define corners in local frame: TL, TR, BR, BL
        const halfSize = innerSize / 2;
        this.localCorners = [
            new THREE.Vector3(-halfSize, halfSize, 0),   // Top-Left
            new THREE.Vector3(halfSize, halfSize, 0),    // Top-Right
            new THREE.Vector3(halfSize, -halfSize, 0),   // Bottom-Right
            new THREE.Vector3(-halfSize, -halfSize, 0),  // Bottom-Left
        ];

        // Create 3D mesh
        this.mesh = this.createMesh();
    }

    /**
     * Create gate 3D mesh
     */
    private createMesh(): THREE.Group {
        const group = new THREE.Group();
        const halfSize = this.innerSize / 2;
        const frameWidth = 0.08;  // frame tube thickness

        // Frame material
        const frameMaterial = new THREE.MeshBasicMaterial({
            color: this.frameColor,
            transparent: true,
            opacity: 0.9,
        });

        // Create frame as 4 box segments
        const segments = [
            // Top
            { pos: [0, halfSize, 0], size: [this.innerSize + frameWidth, frameWidth, frameWidth] },
            // Bottom
            { pos: [0, -halfSize, 0], size: [this.innerSize + frameWidth, frameWidth, frameWidth] },
            // Left
            { pos: [-halfSize, 0, 0], size: [frameWidth, this.innerSize, frameWidth] },
            // Right
            { pos: [halfSize, 0, 0], size: [frameWidth, this.innerSize, frameWidth] },
        ];

        for (const seg of segments) {
            const geometry = new THREE.BoxGeometry(seg.size[0], seg.size[1], seg.size[2]);
            const mesh = new THREE.Mesh(geometry, frameMaterial);
            mesh.position.set(seg.pos[0], seg.pos[1], seg.pos[2]);
            group.add(mesh);
        }

        // Add corner markers (small spheres)
        const cornerMaterial = new THREE.MeshBasicMaterial({ color: this.cornerColor });
        const cornerGeometry = new THREE.SphereGeometry(0.06, 8, 8);

        for (const corner of this.localCorners) {
            const sphere = new THREE.Mesh(cornerGeometry, cornerMaterial);
            sphere.position.copy(corner);
            group.add(sphere);
        }

        // Add gate ID label
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        const ctx = canvas.getContext('2d')!;
        ctx.fillStyle = '#a855f7';
        ctx.font = 'bold 48px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(this.id + 1), 32, 32);

        const texture = new THREE.CanvasTexture(canvas);
        const labelMaterial = new THREE.SpriteMaterial({ map: texture });
        const label = new THREE.Sprite(labelMaterial);
        label.scale.set(0.5, 0.5, 1);
        label.position.set(0, halfSize + 0.4, 0);
        group.add(label);

        // Add direction indicator (arrow showing which way to fly through)
        const arrowGeometry = new THREE.ConeGeometry(0.1, 0.3, 8);
        const arrowMaterial = new THREE.MeshBasicMaterial({ color: this.labelColor });
        const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);
        arrow.rotation.x = Math.PI / 2;  // Point along +Z
        arrow.position.set(0, 0, 0.3);
        group.add(arrow);

        return group;
    }

    /**
     * Set gate pose in world frame
     */
    public setPose(position: Vector3, orientation: Quaternion): void {
        this.position.set(position.x, position.y, position.z);
        this.quaternion.set(orientation.x, orientation.y, orientation.z, orientation.w);

        // Update mesh
        this.mesh.position.copy(this.position);
        this.mesh.quaternion.copy(this.quaternion);
    }

    /**
     * Set gate position (keep current orientation)
     */
    public setPosition(x: number, y: number, z: number): void {
        this.position.set(x, y, z);
        this.mesh.position.copy(this.position);
    }

    /**
     * Set gate orientation from yaw angle (rotation around Y axis)
     */
    public setYaw(yaw: number): void {
        this.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
        this.mesh.quaternion.copy(this.quaternion);
    }

    /**
     * Get gate pose
     */
    public getPose(): GatePose {
        return {
            gateId: this.id,
            position: { x: this.position.x, y: this.position.y, z: this.position.z },
            orientation: {
                w: this.quaternion.w,
                x: this.quaternion.x,
                y: this.quaternion.y,
                z: this.quaternion.z,
            },
            innerSize: this.innerSize,
        };
    }

    /**
     * Get corner positions in world frame
     * Returns: [TL, TR, BR, BL]
     */
    public getWorldCorners(): THREE.Vector3[] {
        const worldCorners: THREE.Vector3[] = [];

        for (const localCorner of this.localCorners) {
            const worldCorner = localCorner.clone();
            // Apply gate rotation
            worldCorner.applyQuaternion(this.quaternion);
            // Apply gate translation
            worldCorner.add(this.position);
            worldCorners.push(worldCorner);
        }

        return worldCorners;
    }

    /**
     * Get gate center position
     */
    public getCenter(): THREE.Vector3 {
        return this.position.clone();
    }

    /**
     * Get gate forward direction (normal to gate plane, direction to fly through)
     */
    public getForwardDirection(): THREE.Vector3 {
        const forward = new THREE.Vector3(0, 0, 1);
        forward.applyQuaternion(this.quaternion);
        return forward;
    }

    /**
     * Get distance from a point to gate center
     */
    public distanceToPoint(point: THREE.Vector3): number {
        return this.position.distanceTo(point);
    }

    /**
     * Check if a point is in front of the gate (positive Z in gate frame)
     */
    public isPointInFront(point: THREE.Vector3): boolean {
        const toPoint = point.clone().sub(this.position);
        const forward = this.getForwardDirection();
        return toPoint.dot(forward) > 0;
    }

    /**
     * Get approach waypoint position (offset in front of gate)
     */
    public getApproachPosition(distance: number = 1.0): THREE.Vector3 {
        const forward = this.getForwardDirection();
        return this.position.clone().sub(forward.multiplyScalar(distance));
    }

    /**
     * Get exit waypoint position (offset behind gate)
     */
    public getExitPosition(distance: number = 1.0): THREE.Vector3 {
        const forward = this.getForwardDirection();
        return this.position.clone().add(forward.multiplyScalar(distance));
    }

    /**
     * Highlight gate (when it's the next target)
     */
    public setHighlight(active: boolean): void {
        const color = active ? 0x22c55e : this.frameColor;  // Green when active

        this.mesh.traverse((child) => {
            if (child instanceof THREE.Mesh && child.material instanceof THREE.MeshBasicMaterial) {
                // Don't change corner markers
                if (child.geometry instanceof THREE.SphereGeometry) return;
                if (child.geometry instanceof THREE.ConeGeometry) return;
                child.material.color.setHex(color);
            }
        });
    }
}
