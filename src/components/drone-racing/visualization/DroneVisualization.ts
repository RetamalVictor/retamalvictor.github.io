import * as THREE from 'three';
import { DroneState } from '../types';

/**
 * Pure Drone Visualization
 *
 * This class handles ONLY the 3D rendering - no physics.
 * - Creates and manages Three.js mesh
 * - Updates mesh position/orientation from state
 * - Handles rotor animation
 * - Provides camera pose for FPV view
 */
export interface DroneVisualizationConfig {
    frameColor: number;
    rotorColor: number;
    cameraColor: number;
    armLength: number;
    rotorRadius: number;
    bodySize: number;
    cameraPitch: number;
}

export const DEFAULT_VISUALIZATION_CONFIG: DroneVisualizationConfig = {
    frameColor: 0x00d4ff,    // Cyan
    rotorColor: 0xffffff,    // White
    cameraColor: 0xa855f7,   // Purple
    armLength: 0.25,
    rotorRadius: 0.1,
    bodySize: 0.08,
    cameraPitch: -0.2,       // Slightly looking down
};

export class DroneVisualization {
    public readonly mesh: THREE.Group;
    private config: DroneVisualizationConfig;

    // Rotor meshes for animation
    private rotors: THREE.Mesh[] = [];

    // Camera offset from drone center
    private readonly cameraOffset = new THREE.Vector3(0, -0.05, 0.1);

    constructor(config: Partial<DroneVisualizationConfig> = {}) {
        this.config = { ...DEFAULT_VISUALIZATION_CONFIG, ...config };
        this.mesh = this.createMesh();
    }

    /**
     * Create the drone mesh
     */
    private createMesh(): THREE.Group {
        const group = new THREE.Group();
        const { frameColor, rotorColor, cameraColor, armLength, rotorRadius, bodySize } = this.config;

        // Create X-frame arms
        const armMaterial = new THREE.LineBasicMaterial({ color: frameColor, linewidth: 2 });

        const arm1 = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(-armLength, 0, armLength),
                new THREE.Vector3(armLength, 0, -armLength)
            ]),
            armMaterial
        );
        group.add(arm1);

        const arm2 = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(armLength, 0, armLength),
                new THREE.Vector3(-armLength, 0, -armLength)
            ]),
            armMaterial
        );
        group.add(arm2);

        // Create rotors
        const rotorPositions = [
            new THREE.Vector3(-armLength, 0, armLength),
            new THREE.Vector3(armLength, 0, armLength),
            new THREE.Vector3(armLength, 0, -armLength),
            new THREE.Vector3(-armLength, 0, -armLength)
        ];

        const rotorMaterial = new THREE.MeshBasicMaterial({
            color: rotorColor,
            transparent: true,
            opacity: 0.8
        });
        const rotorGeometry = new THREE.CircleGeometry(rotorRadius, 16);

        for (const pos of rotorPositions) {
            const rotor = new THREE.Mesh(rotorGeometry, rotorMaterial);
            rotor.rotation.x = -Math.PI / 2;  // Horizontal
            rotor.position.copy(pos);
            group.add(rotor);
            this.rotors.push(rotor);
        }

        // Center body
        const bodyGeometry = new THREE.BoxGeometry(bodySize * 2, bodySize, bodySize * 2);
        const bodyMaterial = new THREE.MeshBasicMaterial({
            color: frameColor,
            transparent: true,
            opacity: 0.9
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        group.add(body);

        // Camera indicator (cone pointing forward-down)
        const cameraGeometry = new THREE.ConeGeometry(0.04, 0.15, 8);
        const cameraMaterial = new THREE.MeshBasicMaterial({ color: cameraColor });
        const camera = new THREE.Mesh(cameraGeometry, cameraMaterial);
        camera.rotation.x = Math.PI / 2 + this.config.cameraPitch;
        camera.position.set(0, -bodySize * 0.3, 0.08);
        group.add(camera);

        // Velocity indicator arrow (hidden by default)
        const arrowGroup = new THREE.Group();
        arrowGroup.name = 'velocityArrow';
        const arrowMaterial = new THREE.MeshBasicMaterial({ color: 0x22c55e });
        const arrowShaft = new THREE.Mesh(
            new THREE.CylinderGeometry(0.02, 0.02, 0.3, 8),
            arrowMaterial
        );
        arrowShaft.rotation.x = Math.PI / 2;
        arrowShaft.position.z = 0.15;
        arrowGroup.add(arrowShaft);
        const arrowHead = new THREE.Mesh(
            new THREE.ConeGeometry(0.04, 0.1, 8),
            arrowMaterial
        );
        arrowHead.rotation.x = Math.PI / 2;
        arrowHead.position.z = 0.35;
        arrowGroup.add(arrowHead);
        arrowGroup.visible = false;
        group.add(arrowGroup);

        return group;
    }

    /**
     * Update visualization from state
     *
     * @param state - Current drone state
     * @param thrustNormalized - Thrust normalized by gravity (for rotor animation)
     */
    public update(state: DroneState, thrustNormalized: number = 1.0): void {
        // Update position
        this.mesh.position.set(
            state.position.x,
            state.position.y,
            state.position.z
        );

        // Update orientation
        this.mesh.quaternion.set(
            state.orientation.x,
            state.orientation.y,
            state.orientation.z,
            state.orientation.w
        );

        // Animate rotors
        this.animateRotors(thrustNormalized);
    }

    /**
     * Animate rotor spinning based on thrust
     */
    private animateRotors(thrustFactor: number): void {
        const speed = Math.sqrt(Math.max(0, thrustFactor)) * 0.5;
        for (let i = 0; i < this.rotors.length; i++) {
            const dir = (i % 2 === 0) ? 1 : -1;
            this.rotors[i].rotation.z += dir * speed;
        }
    }

    /**
     * Get camera pose for FPV view
     */
    public getCameraPose(state: DroneState): { position: THREE.Vector3; rotation: THREE.Euler } {
        const orientation = new THREE.Quaternion(
            state.orientation.x,
            state.orientation.y,
            state.orientation.z,
            state.orientation.w
        );

        // Camera position: offset from drone center, rotated by drone orientation
        const camOffset = this.cameraOffset.clone().applyQuaternion(orientation);
        const camPos = new THREE.Vector3(
            state.position.x,
            state.position.y,
            state.position.z
        ).add(camOffset);

        // Camera rotation: drone orientation + camera pitch
        const camQuat = orientation.clone();
        const pitchQuat = new THREE.Quaternion().setFromAxisAngle(
            new THREE.Vector3(1, 0, 0),
            this.config.cameraPitch
        );
        camQuat.multiply(pitchQuat);

        const camEuler = new THREE.Euler().setFromQuaternion(camQuat, 'XYZ');

        return { position: camPos, rotation: camEuler };
    }

    /**
     * Show/hide velocity arrow
     */
    public showVelocityArrow(show: boolean): void {
        const arrow = this.mesh.getObjectByName('velocityArrow');
        if (arrow) arrow.visible = show;
    }

    /**
     * Set frame color
     */
    public setFrameColor(color: number): void {
        this.mesh.traverse((child) => {
            if (child instanceof THREE.Line) {
                (child.material as THREE.LineBasicMaterial).color.setHex(color);
            }
        });
    }

    /**
     * Dispose of Three.js resources
     */
    public dispose(): void {
        this.mesh.traverse((child) => {
            if (child instanceof THREE.Mesh) {
                child.geometry.dispose();
                if (child.material instanceof THREE.Material) {
                    child.material.dispose();
                }
            }
            if (child instanceof THREE.Line) {
                child.geometry.dispose();
                if (child.material instanceof THREE.Material) {
                    child.material.dispose();
                }
            }
        });
    }
}
