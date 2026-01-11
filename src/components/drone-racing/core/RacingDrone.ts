import * as THREE from 'three';
import { DroneState, ControlCommand, Vector3 } from '../types';

/**
 * Racing Drone Model and Dynamics
 *
 * Combines:
 * - 3D mesh (quadrotor visualization)
 * - 10-state dynamics (position, orientation, velocity)
 * - Control interface (thrust + body rates)
 *
 * Based on the drone-racing-control dynamics model:
 * - Quaternion-based orientation
 * - First-order actuator dynamics
 * - Drag modeling
 */
export class RacingDrone {
    public readonly mesh: THREE.Group;

    // Physical parameters
    private readonly mass = 1.0;           // kg
    private readonly gravity = 9.81;       // m/s²
    private readonly linearDrag = 0.3;     // Linear drag coefficient

    // Actuator dynamics (first-order time constants)
    private readonly tauThrust = 0.04;     // 40ms
    private readonly tauRate = 0.03;       // 30ms

    // State
    private position: THREE.Vector3;
    private velocity: THREE.Vector3;
    private orientation: THREE.Quaternion;
    private angularVelocity: THREE.Vector3;

    // Actuator states (for smooth transitions)
    private currentThrust: number;
    private currentRates: THREE.Vector3;

    // Rotor meshes for animation
    private rotors: THREE.Mesh[] = [];

    // Camera attached to drone
    private readonly cameraOffset = new THREE.Vector3(0, -0.05, 0.1);
    private readonly cameraPitch = -0.2;  // Slightly looking down

    constructor() {
        this.mesh = this.createMesh();

        // Initialize state
        this.position = new THREE.Vector3(0, 2, 0);
        this.velocity = new THREE.Vector3(0, 0, 0);
        this.orientation = new THREE.Quaternion(0, 0, 0, 1);
        this.angularVelocity = new THREE.Vector3(0, 0, 0);

        this.currentThrust = this.gravity;
        this.currentRates = new THREE.Vector3(0, 0, 0);

        this.syncMesh();
    }

    /**
     * Create racing drone mesh
     */
    private createMesh(): THREE.Group {
        const group = new THREE.Group();

        // Colors
        const frameColor = 0x00d4ff;    // Cyan
        const rotorColor = 0xffffff;    // White
        const cameraColor = 0xa855f7;   // Purple

        // Dimensions
        const armLength = 0.25;
        const rotorRadius = 0.1;
        const bodySize = 0.08;

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

        const rotorMaterial = new THREE.MeshBasicMaterial({ color: rotorColor, transparent: true, opacity: 0.8 });
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
        const bodyMaterial = new THREE.MeshBasicMaterial({ color: frameColor, transparent: true, opacity: 0.9 });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        group.add(body);

        // Camera indicator (cone pointing forward-down)
        const cameraGeometry = new THREE.ConeGeometry(0.04, 0.15, 8);
        const cameraMaterial = new THREE.MeshBasicMaterial({ color: cameraColor });
        const camera = new THREE.Mesh(cameraGeometry, cameraMaterial);
        camera.rotation.x = Math.PI / 2 + this.cameraPitch;
        camera.position.set(0, -bodySize * 0.3, 0.08);
        group.add(camera);

        // Velocity indicator (arrow)
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
     * Update dynamics with control command
     */
    public update(command: ControlCommand, dt: number): void {
        // Limit dt to prevent instability
        dt = Math.min(dt, 0.05);

        // Apply first-order dynamics to actuators
        const alphaThrust = 1 - Math.exp(-dt / this.tauThrust);
        const alphaRate = 1 - Math.exp(-dt / this.tauRate);

        this.currentThrust += alphaThrust * (command.thrust - this.currentThrust);
        this.currentRates.x += alphaRate * (command.rollRate - this.currentRates.x);
        this.currentRates.y += alphaRate * (command.yawRate - this.currentRates.y);
        this.currentRates.z += alphaRate * (command.pitchRate - this.currentRates.z);

        // Update angular velocity (simplified: direct from rates)
        this.angularVelocity.copy(this.currentRates);

        // Update orientation using quaternion integration
        this.integrateQuaternion(dt);

        // Compute thrust in world frame
        const thrustBody = new THREE.Vector3(0, this.currentThrust, 0);
        const thrustWorld = thrustBody.clone().applyQuaternion(this.orientation);

        // Acceleration (thrust - gravity - drag)
        const acceleration = new THREE.Vector3(
            thrustWorld.x / this.mass - this.linearDrag * this.velocity.x,
            thrustWorld.y / this.mass - this.gravity - this.linearDrag * this.velocity.y,
            thrustWorld.z / this.mass - this.linearDrag * this.velocity.z
        );

        // Update velocity
        this.velocity.add(acceleration.multiplyScalar(dt));

        // Update position
        this.position.add(this.velocity.clone().multiplyScalar(dt));

        // Sync mesh
        this.syncMesh();

        // Animate rotors (speed based on thrust)
        this.animateRotors(this.currentThrust / this.gravity);
    }

    /**
     * Integrate quaternion using angular velocity
     */
    private integrateQuaternion(dt: number): void {
        const w = this.angularVelocity;
        const wMag = w.length();

        if (wMag < 1e-10) return;

        // Quaternion derivative: dq/dt = 0.5 * q * [0, w]
        const halfAngle = wMag * dt / 2;
        const sinHalf = Math.sin(halfAngle);
        const cosHalf = Math.cos(halfAngle);

        // Rotation quaternion for this timestep
        const dq = new THREE.Quaternion(
            w.x / wMag * sinHalf,
            w.y / wMag * sinHalf,
            w.z / wMag * sinHalf,
            cosHalf
        );

        // Apply rotation
        this.orientation.multiply(dq);
        this.orientation.normalize();
    }

    /**
     * Sync mesh with state
     */
    private syncMesh(): void {
        this.mesh.position.copy(this.position);
        this.mesh.quaternion.copy(this.orientation);
    }

    /**
     * Animate rotor spinning
     */
    private animateRotors(thrustFactor: number): void {
        const speed = Math.sqrt(thrustFactor) * 0.5;
        for (let i = 0; i < this.rotors.length; i++) {
            const dir = (i % 2 === 0) ? 1 : -1;
            this.rotors[i].rotation.z += dir * speed;
        }
    }

    /**
     * Get current state
     */
    public getState(): DroneState {
        return {
            position: { x: this.position.x, y: this.position.y, z: this.position.z },
            orientation: {
                w: this.orientation.w,
                x: this.orientation.x,
                y: this.orientation.y,
                z: this.orientation.z
            },
            velocity: { x: this.velocity.x, y: this.velocity.y, z: this.velocity.z },
            timestamp: Date.now() * 1000,
        };
    }

    /**
     * Get camera pose (world frame)
     */
    public getCameraPose(): { position: THREE.Vector3; rotation: THREE.Euler } {
        // Camera position: offset from drone center, rotated by drone orientation
        const camOffset = this.cameraOffset.clone().applyQuaternion(this.orientation);
        const camPos = this.position.clone().add(camOffset);

        // Camera rotation: drone orientation + camera pitch
        const camQuat = this.orientation.clone();
        const pitchQuat = new THREE.Quaternion().setFromAxisAngle(
            new THREE.Vector3(1, 0, 0),
            this.cameraPitch
        );
        camQuat.multiply(pitchQuat);

        const camEuler = new THREE.Euler().setFromQuaternion(camQuat, 'XYZ');

        return { position: camPos, rotation: camEuler };
    }

    /**
     * Get position as THREE.Vector3
     */
    public getPosition(): THREE.Vector3 {
        return this.position.clone();
    }

    /**
     * Get velocity magnitude
     */
    public getSpeed(): number {
        return this.velocity.length();
    }

    /**
     * Reset to initial state
     */
    public reset(position?: Vector3): void {
        this.position.set(
            position?.x ?? 0,
            position?.y ?? 2,
            position?.z ?? 0
        );
        this.velocity.set(0, 0, 0);
        this.orientation.set(0, 0, 0, 1);
        this.angularVelocity.set(0, 0, 0);
        this.currentThrust = this.gravity;
        this.currentRates.set(0, 0, 0);
        this.syncMesh();
    }

    /**
     * Set position directly (for initialization)
     */
    public setPosition(x: number, y: number, z: number): void {
        this.position.set(x, y, z);
        this.syncMesh();
    }

    /**
     * Set heading (yaw) directly
     */
    public setHeading(yaw: number): void {
        this.orientation.setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
        this.syncMesh();
    }

    /**
     * Show/hide velocity arrow
     */
    public showVelocityArrow(show: boolean): void {
        const arrow = this.mesh.getObjectByName('velocityArrow');
        if (arrow) arrow.visible = show;
    }
}
