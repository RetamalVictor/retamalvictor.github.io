import * as THREE from 'three';
import { Gate, GatePosition, Quaternion } from '../types';

/**
 * Gate Visualization
 *
 * Renders a racing gate as a rectangular frame using Three.js.
 * Gates are the checkpoints that drones must fly through in racing.
 *
 * Visual style: Glowing neon rectangular frame (like real FPV racing gates)
 */
export interface GateVisualizationConfig {
    color: number;          // Frame color (hex)
    emissiveIntensity: number;
    frameThickness: number; // Thickness of frame bars
    opacity: number;
}

export const DEFAULT_GATE_CONFIG: GateVisualizationConfig = {
    color: 0xff4444,        // Bright red
    emissiveIntensity: 0.5,
    frameThickness: 0.1,
    opacity: 0.9,
};

/**
 * Creates and manages a single gate mesh
 */
export class GateVisualization {
    public readonly mesh: THREE.Group;
    private config: GateVisualizationConfig;
    private gate: Gate;

    constructor(gate: Gate, config: Partial<GateVisualizationConfig> = {}) {
        this.config = { ...DEFAULT_GATE_CONFIG, ...config };
        this.gate = gate;
        this.mesh = this.createMesh();
        this.updatePose();
    }

    /**
     * Create the gate mesh (rectangular frame)
     */
    private createMesh(): THREE.Group {
        const group = new THREE.Group();
        const { width, height } = this.gate;
        const { color, frameThickness, opacity } = this.config;

        const halfW = width / 2;
        const halfH = height / 2;

        // Create frame using line segments for a clean look
        const frameMaterial = new THREE.LineBasicMaterial({
            color: color,
            linewidth: 2,
            transparent: true,
            opacity: opacity,
        });

        // Frame outline (rectangle)
        const framePoints = [
            // Bottom
            new THREE.Vector3(-halfW, -halfH, 0),
            new THREE.Vector3(halfW, -halfH, 0),
            // Right
            new THREE.Vector3(halfW, -halfH, 0),
            new THREE.Vector3(halfW, halfH, 0),
            // Top
            new THREE.Vector3(halfW, halfH, 0),
            new THREE.Vector3(-halfW, halfH, 0),
            // Left
            new THREE.Vector3(-halfW, halfH, 0),
            new THREE.Vector3(-halfW, -halfH, 0),
        ];

        const frameGeometry = new THREE.BufferGeometry().setFromPoints(framePoints);
        const frameLine = new THREE.LineSegments(frameGeometry, frameMaterial);
        group.add(frameLine);

        // Add corner posts for 3D depth
        const postMaterial = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: opacity * 0.8,
        });
        const postGeometry = new THREE.CylinderGeometry(
            frameThickness / 2,
            frameThickness / 2,
            height,
            8
        );

        // Left post
        const leftPost = new THREE.Mesh(postGeometry, postMaterial);
        leftPost.position.set(-halfW, 0, 0);
        group.add(leftPost);

        // Right post
        const rightPost = new THREE.Mesh(postGeometry, postMaterial);
        rightPost.position.set(halfW, 0, 0);
        group.add(rightPost);

        // Add top bar
        const barGeometry = new THREE.CylinderGeometry(
            frameThickness / 2,
            frameThickness / 2,
            width,
            8
        );
        const topBar = new THREE.Mesh(barGeometry, postMaterial);
        topBar.rotation.z = Math.PI / 2;
        topBar.position.set(0, halfH, 0);
        group.add(topBar);

        // Add bottom bar
        const bottomBar = new THREE.Mesh(barGeometry, postMaterial);
        bottomBar.rotation.z = Math.PI / 2;
        bottomBar.position.set(0, -halfH, 0);
        group.add(bottomBar);

        // Add gate number indicator
        this.addGateNumber(group);

        return group;
    }

    /**
     * Add gate number indicator
     */
    private addGateNumber(group: THREE.Group): void {
        // Simple number indicator using a small sphere with color coding
        const indicatorGeometry = new THREE.SphereGeometry(0.15, 8, 8);
        const indicatorMaterial = new THREE.MeshBasicMaterial({
            color: this.config.color,
        });
        const indicator = new THREE.Mesh(indicatorGeometry, indicatorMaterial);
        indicator.position.set(0, this.gate.height / 2 + 0.3, 0);
        indicator.name = 'gateIndicator';
        group.add(indicator);
    }

    /**
     * Update mesh position and orientation from gate data
     */
    private updatePose(): void {
        this.mesh.position.set(
            this.gate.position.x,
            this.gate.position.y,
            this.gate.position.z
        );

        this.mesh.quaternion.set(
            this.gate.orientation.x,
            this.gate.orientation.y,
            this.gate.orientation.z,
            this.gate.orientation.w
        );
    }

    /**
     * Update gate color (e.g., when passed)
     */
    public setColor(color: number): void {
        this.mesh.traverse((child) => {
            if (child instanceof THREE.Line) {
                (child.material as THREE.LineBasicMaterial).color.setHex(color);
            }
            if (child instanceof THREE.Mesh) {
                (child.material as THREE.MeshBasicMaterial).color.setHex(color);
            }
        });
    }

    /**
     * Highlight gate (e.g., as next target)
     */
    public setHighlighted(highlighted: boolean): void {
        const scale = highlighted ? 1.1 : 1.0;
        this.mesh.scale.set(scale, scale, scale);
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
            if (child instanceof THREE.Line || child instanceof THREE.LineSegments) {
                child.geometry.dispose();
                if (child.material instanceof THREE.Material) {
                    child.material.dispose();
                }
            }
        });
    }
}

/**
 * Create a Gate from GatePosition data
 */
export function createGateFromPosition(
    gatePos: GatePosition,
    index: number,
    width: number = 3.0,
    height: number = 3.0
): Gate {
    // Convert heading to quaternion (rotation around Y axis)
    const halfAngle = gatePos.heading / 2;
    const orientation: Quaternion = {
        w: Math.cos(halfAngle),
        x: 0,
        y: Math.sin(halfAngle),
        z: 0,
    };

    return {
        position: { ...gatePos.position },
        orientation,
        width,
        height,
        index,
    };
}

/**
 * Manages multiple gates in the scene
 */
export class GateManager {
    private gates: GateVisualization[] = [];
    private scene: THREE.Scene;
    private config: Partial<GateVisualizationConfig>;

    constructor(scene: THREE.Scene, config: Partial<GateVisualizationConfig> = {}) {
        this.scene = scene;
        this.config = config;
    }

    /**
     * Set gates from gate position data
     */
    public setGates(
        gatePositions: GatePosition[],
        width: number = 3.0,
        height: number = 3.0
    ): void {
        // Clear existing gates
        this.clearGates();

        // Create new gates
        for (let i = 0; i < gatePositions.length; i++) {
            const gate = createGateFromPosition(gatePositions[i], i, width, height);
            const visualization = new GateVisualization(gate, this.config);
            this.gates.push(visualization);
            this.scene.add(visualization.mesh);
        }
    }

    /**
     * Get number of gates
     */
    public getGateCount(): number {
        return this.gates.length;
    }

    /**
     * Highlight a specific gate (e.g., next gate to pass)
     */
    public highlightGate(index: number): void {
        for (let i = 0; i < this.gates.length; i++) {
            this.gates[i].setHighlighted(i === index);
        }
    }

    /**
     * Mark a gate as passed (change color to green)
     */
    public markGatePassed(index: number): void {
        if (index >= 0 && index < this.gates.length) {
            this.gates[index].setColor(0x22c55e); // Green
        }
    }

    /**
     * Reset all gates to default state
     */
    public resetGates(): void {
        for (const gate of this.gates) {
            gate.setColor(this.config.color ?? DEFAULT_GATE_CONFIG.color);
            gate.setHighlighted(false);
        }
    }

    /**
     * Clear all gates from scene
     */
    public clearGates(): void {
        for (const gate of this.gates) {
            this.scene.remove(gate.mesh);
            gate.dispose();
        }
        this.gates = [];
    }

    /**
     * Dispose of all resources
     */
    public dispose(): void {
        this.clearGates();
    }
}
