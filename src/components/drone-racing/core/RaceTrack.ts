import * as THREE from 'three';
import { RaceGate } from './RaceGate';
import { RacingConfig, DEFAULT_CONFIG, GatePose } from '../types';

/**
 * Race Track Configuration
 *
 * Manages a sequence of gates forming a racing circuit.
 * Provides:
 * - Gate placement in a loop configuration
 * - Next gate tracking
 * - Lap counting
 * - Track visualization (connecting lines)
 */
export class RaceTrack {
    public readonly gates: RaceGate[];
    public readonly config: RacingConfig;

    // Track state
    private currentGateIndex: number = 0;
    private lapCount: number = 0;

    // Visualization
    public readonly trackGroup: THREE.Group;
    private trackLine: THREE.Line | null = null;

    constructor(config: Partial<RacingConfig> = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.gates = [];
        this.trackGroup = new THREE.Group();

        this.createTrack();
    }

    /**
     * Create gates arranged in an oval/figure-8 pattern
     */
    private createTrack(): void {
        const numGates = this.config.numGates;
        const radius = this.config.trackRadius;
        const gateSize = this.config.gateSize;
        const height = 2.0;  // gate center height above ground

        // Create gates in an oval pattern
        for (let i = 0; i < numGates; i++) {
            const gate = new RaceGate(i, gateSize);

            // Position gates in an oval
            const angle = (i / numGates) * Math.PI * 2;

            // Oval shape: wider in X, narrower in Z
            const x = Math.cos(angle) * radius;
            const z = Math.sin(angle) * radius * 0.6;
            const y = height;

            gate.setPosition(x, y, z);

            // Orient gate to face the next gate (tangent to oval)
            const nextAngle = ((i + 1) / numGates) * Math.PI * 2;
            const nextX = Math.cos(nextAngle) * radius;
            const nextZ = Math.sin(nextAngle) * radius * 0.6;

            const direction = Math.atan2(nextX - x, nextZ - z);
            gate.setYaw(direction);

            this.gates.push(gate);
            this.trackGroup.add(gate.mesh);
        }

        // Create track visualization line
        this.createTrackLine();

        // Highlight first gate
        this.gates[0].setHighlight(true);
    }

    /**
     * Create dashed line showing track path
     */
    private createTrackLine(): void {
        const points: THREE.Vector3[] = [];

        // Add gate centers and approach/exit points for smooth curve
        for (let i = 0; i <= this.gates.length; i++) {
            const gate = this.gates[i % this.gates.length];
            points.push(gate.getCenter().clone());
        }

        // Create smooth curve through points
        const curve = new THREE.CatmullRomCurve3(points, true);
        const curvePoints = curve.getPoints(100);

        const geometry = new THREE.BufferGeometry().setFromPoints(curvePoints);
        const material = new THREE.LineDashedMaterial({
            color: 0x00d4ff,
            dashSize: 0.3,
            gapSize: 0.15,
            opacity: 0.4,
            transparent: true,
        });

        this.trackLine = new THREE.Line(geometry, material);
        this.trackLine.computeLineDistances();  // Required for dashed lines
        this.trackLine.position.y = 0.1;  // Slightly above ground
        this.trackGroup.add(this.trackLine);
    }

    /**
     * Get current target gate
     */
    public getCurrentGate(): RaceGate {
        return this.gates[this.currentGateIndex];
    }

    /**
     * Get next gate after current
     */
    public getNextGate(): RaceGate {
        const nextIndex = (this.currentGateIndex + 1) % this.gates.length;
        return this.gates[nextIndex];
    }

    /**
     * Get current gate index
     */
    public getCurrentGateIndex(): number {
        return this.currentGateIndex;
    }

    /**
     * Get lap count
     */
    public getLapCount(): number {
        return this.lapCount;
    }

    /**
     * Advance to next gate (called when drone passes through current gate)
     */
    public advanceGate(): void {
        // Remove highlight from current gate
        this.gates[this.currentGateIndex].setHighlight(false);

        // Advance to next gate
        this.currentGateIndex = (this.currentGateIndex + 1) % this.gates.length;

        // Check for lap completion
        if (this.currentGateIndex === 0) {
            this.lapCount++;
        }

        // Highlight new current gate
        this.gates[this.currentGateIndex].setHighlight(true);
    }

    /**
     * Check if drone has passed through the current gate
     * Uses a simple plane intersection test
     */
    public checkGatePassage(
        prevPosition: THREE.Vector3,
        currentPosition: THREE.Vector3
    ): boolean {
        const gate = this.getCurrentGate();
        const gatePos = gate.getCenter();
        const gateForward = gate.getForwardDirection();

        // Check if we crossed the gate plane
        const prevDist = prevPosition.clone().sub(gatePos).dot(gateForward);
        const currDist = currentPosition.clone().sub(gatePos).dot(gateForward);

        // Crossed if sign changed and we're close enough
        if (prevDist < 0 && currDist >= 0) {
            // Check if within gate bounds (rough check)
            const distToCenter = currentPosition.distanceTo(gatePos);
            if (distToCenter < gate.innerSize * 1.5) {
                return true;
            }
        }

        return false;
    }

    /**
     * Get all gate poses
     */
    public getGatePoses(): GatePose[] {
        return this.gates.map(gate => gate.getPose());
    }

    /**
     * Get distance from position to next gate
     */
    public getDistanceToCurrentGate(position: THREE.Vector3): number {
        return this.getCurrentGate().distanceToPoint(position);
    }

    /**
     * Reset track state
     */
    public reset(): void {
        // Remove all highlights
        for (const gate of this.gates) {
            gate.setHighlight(false);
        }

        // Reset to first gate
        this.currentGateIndex = 0;
        this.lapCount = 0;

        // Highlight first gate
        this.gates[0].setHighlight(true);
    }

    /**
     * Get total track length (approximate)
     */
    public getTotalLength(): number {
        let length = 0;
        for (let i = 0; i < this.gates.length; i++) {
            const current = this.gates[i].getCenter();
            const next = this.gates[(i + 1) % this.gates.length].getCenter();
            length += current.distanceTo(next);
        }
        return length;
    }

    /**
     * Get waypoints for trajectory generation
     * Returns gate centers with approach/exit offsets
     */
    public getTrajectoryWaypoints(): THREE.Vector3[] {
        const waypoints: THREE.Vector3[] = [];

        for (const gate of this.gates) {
            // Add approach point
            waypoints.push(gate.getApproachPosition(0.5));
            // Add gate center
            waypoints.push(gate.getCenter());
            // Add exit point
            waypoints.push(gate.getExitPosition(0.5));
        }

        return waypoints;
    }

    /**
     * Move a gate (for interactive editing)
     */
    public moveGate(gateId: number, position: THREE.Vector3): void {
        if (gateId >= 0 && gateId < this.gates.length) {
            this.gates[gateId].setPosition(position.x, position.y, position.z);
            this.updateTrackLine();
        }
    }

    /**
     * Update track line after gate movement
     */
    private updateTrackLine(): void {
        if (this.trackLine) {
            this.trackGroup.remove(this.trackLine);
            this.trackLine.geometry.dispose();
        }
        this.createTrackLine();
    }
}
