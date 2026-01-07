import * as THREE from 'three';

/**
 * Simple wireframe quadrotor model for Three.js
 * X-frame design with 4 rotors and camera indicator
 * Body is horizontal (rotors in XZ plane), camera points forward (+Z)
 */
export class QuadrotorModel {
    public readonly mesh: THREE.Group;

    // Quadrotor dimensions
    private readonly armLength: number = 0.4;
    private readonly rotorRadius: number = 0.15;
    private readonly bodySize: number = 0.1;

    // Colors
    private readonly armColor = 0x00d4ff;      // Cyan (accent)
    private readonly rotorColor = 0xffffff;    // White
    private readonly cameraColor = 0xa855f7;   // Purple

    // Rotor meshes for animation
    private rotors: THREE.Line[] = [];

    constructor() {
        this.mesh = this.createGeometry();
    }

    private createGeometry(): THREE.Group {
        const group = new THREE.Group();

        // Create X-frame arms in XZ plane (horizontal)
        const armMaterial = new THREE.LineBasicMaterial({
            color: this.armColor,
            linewidth: 2
        });

        // Arm 1: diagonal in XZ plane
        const arm1Geometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(-this.armLength, 0, this.armLength),
            new THREE.Vector3(this.armLength, 0, -this.armLength)
        ]);
        const arm1 = new THREE.Line(arm1Geometry, armMaterial);
        group.add(arm1);

        // Arm 2: other diagonal in XZ plane
        const arm2Geometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(this.armLength, 0, this.armLength),
            new THREE.Vector3(-this.armLength, 0, -this.armLength)
        ]);
        const arm2 = new THREE.Line(arm2Geometry, armMaterial);
        group.add(arm2);

        // Create rotors at arm ends (in XZ plane)
        const rotorMaterial = new THREE.LineBasicMaterial({
            color: this.rotorColor,
            linewidth: 1
        });

        const rotorPositions = [
            new THREE.Vector3(-this.armLength, 0, this.armLength),   // front-left
            new THREE.Vector3(this.armLength, 0, this.armLength),    // front-right
            new THREE.Vector3(this.armLength, 0, -this.armLength),   // back-right
            new THREE.Vector3(-this.armLength, 0, -this.armLength)   // back-left
        ];

        for (const pos of rotorPositions) {
            const rotorGeometry = this.createCircleGeometry(this.rotorRadius, 16);
            const rotor = new THREE.Line(rotorGeometry, rotorMaterial);
            rotor.position.copy(pos);
            // Rotor circles are already in XY, rotate to be in XZ (horizontal)
            rotor.rotation.x = Math.PI / 2;
            group.add(rotor);
            this.rotors.push(rotor);
        }

        // Create center body (small box wireframe)
        const bodyGeometry = new THREE.BoxGeometry(
            this.bodySize * 2,
            this.bodySize,
            this.bodySize * 2
        );
        const bodyEdges = new THREE.EdgesGeometry(bodyGeometry);
        const bodyWireframe = new THREE.LineSegments(
            bodyEdges,
            new THREE.LineBasicMaterial({ color: this.armColor })
        );
        group.add(bodyWireframe);

        // Create camera frustum indicator (pointing in +Z direction)
        const cameraMaterial = new THREE.LineBasicMaterial({
            color: this.cameraColor,
            linewidth: 2
        });

        // Camera frustum lines - pointing forward (+Z)
        const frustumSize = 0.15;
        const frustumDepth = 0.4;
        const frustumPoints = [
            // Apex at camera position (slightly below body center)
            new THREE.Vector3(0, -this.bodySize * 0.3, 0),
            new THREE.Vector3(-frustumSize, -frustumSize * 0.5, frustumDepth),

            new THREE.Vector3(0, -this.bodySize * 0.3, 0),
            new THREE.Vector3(frustumSize, -frustumSize * 0.5, frustumDepth),

            new THREE.Vector3(0, -this.bodySize * 0.3, 0),
            new THREE.Vector3(frustumSize, frustumSize * 0.5, frustumDepth),

            new THREE.Vector3(0, -this.bodySize * 0.3, 0),
            new THREE.Vector3(-frustumSize, frustumSize * 0.5, frustumDepth),

            // Base rectangle
            new THREE.Vector3(-frustumSize, -frustumSize * 0.5, frustumDepth),
            new THREE.Vector3(frustumSize, -frustumSize * 0.5, frustumDepth),

            new THREE.Vector3(frustumSize, -frustumSize * 0.5, frustumDepth),
            new THREE.Vector3(frustumSize, frustumSize * 0.5, frustumDepth),

            new THREE.Vector3(frustumSize, frustumSize * 0.5, frustumDepth),
            new THREE.Vector3(-frustumSize, frustumSize * 0.5, frustumDepth),

            new THREE.Vector3(-frustumSize, frustumSize * 0.5, frustumDepth),
            new THREE.Vector3(-frustumSize, -frustumSize * 0.5, frustumDepth)
        ];

        const frustumGeometry = new THREE.BufferGeometry().setFromPoints(frustumPoints);
        const frustum = new THREE.LineSegments(frustumGeometry, cameraMaterial);
        group.add(frustum);

        return group;
    }

    private createCircleGeometry(radius: number, segments: number): THREE.BufferGeometry {
        const points: THREE.Vector3[] = [];
        for (let i = 0; i <= segments; i++) {
            const angle = (i / segments) * Math.PI * 2;
            points.push(new THREE.Vector3(
                Math.cos(angle) * radius,
                Math.sin(angle) * radius,
                0
            ));
        }
        return new THREE.BufferGeometry().setFromPoints(points);
    }

    /**
     * Set quadrotor pose
     */
    public setPose(position: THREE.Vector3, rotation: THREE.Euler): void {
        this.mesh.position.copy(position);
        this.mesh.rotation.copy(rotation);
    }

    /**
     * Set quadrotor pose from array
     */
    public setPoseFromArray(pose: Float32Array): void {
        this.mesh.position.set(pose[0], pose[1], pose[2]);
        this.mesh.rotation.set(pose[3], pose[4], pose[5], 'XYZ');
    }

    /**
     * Get current position
     */
    public getPosition(): THREE.Vector3 {
        return this.mesh.position.clone();
    }

    /**
     * Get current rotation
     */
    public getRotation(): THREE.Euler {
        return this.mesh.rotation.clone();
    }

    /**
     * Get forward direction vector (camera pointing direction)
     */
    public getForwardVector(): THREE.Vector3 {
        const forward = new THREE.Vector3(0, 0, 1);
        forward.applyEuler(this.mesh.rotation);
        return forward;
    }

    /**
     * Animate rotors spinning
     */
    public animateRotors(speed: number = 1): void {
        for (let i = 0; i < this.rotors.length; i++) {
            // Alternate spin direction for adjacent rotors
            const direction = (i % 2 === 0) ? 1 : -1;
            this.rotors[i].rotation.z += direction * speed * 0.5;
        }
    }

    /**
     * Update pose from velocity (integration)
     */
    public updateFromVelocity(velocity: Float32Array, dt: number): void {
        this.mesh.position.x += velocity[0] * dt;
        this.mesh.position.y += velocity[1] * dt;
        this.mesh.position.z += velocity[2] * dt;

        this.mesh.rotation.x += velocity[3] * dt;
        this.mesh.rotation.y += velocity[4] * dt;
        this.mesh.rotation.z += velocity[5] * dt;
    }
}
