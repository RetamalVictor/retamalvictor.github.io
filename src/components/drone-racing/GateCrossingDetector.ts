/**
 * Gate Crossing Detector
 *
 * Detects when a drone passes through racing gates.
 * Uses plane-crossing detection with position tolerance checking.
 */

import { GatePosition, Vector3 } from './types';

export interface GateTolerance {
    halfWidth: number;   // Half-width for X tolerance (meters)
    halfHeight: number;  // Half-height for Y tolerance (meters)
}

export interface GateCrossingResult {
    crossed: boolean;       // Whether a gate was crossed this frame
    gateIndex: number;      // Index of the crossed gate (-1 if none)
    lapCompleted: boolean;  // Whether this crossing completed a lap
}

export const DEFAULT_GATE_TOLERANCE: GateTolerance = {
    halfWidth: 2.0,   // 2m tolerance for 3m wide gate
    halfHeight: 2.0,  // 2m tolerance for 3m tall gate
};

export class GateCrossingDetector {
    private gates: GatePosition[];
    private tolerance: GateTolerance;
    private nextGateIndex: number = 0;
    private lastZ: number = 0;

    constructor(
        gates: GatePosition[],
        tolerance: Partial<GateTolerance> = {}
    ) {
        this.gates = gates;
        this.tolerance = { ...DEFAULT_GATE_TOLERANCE, ...tolerance };
    }

    /**
     * Update gates (e.g., when trajectory changes)
     */
    public setGates(gates: GatePosition[]): void {
        this.gates = gates;
        this.reset();
    }

    /**
     * Check if the drone crossed through the next gate
     * @param currentPos Current drone position
     * @returns Result indicating if a gate was crossed
     */
    public check(currentPos: Vector3): GateCrossingResult {
        const result: GateCrossingResult = {
            crossed: false,
            gateIndex: -1,
            lapCompleted: false,
        };

        // No gates to check
        if (this.gates.length === 0) {
            this.lastZ = currentPos.z;
            return result;
        }

        const nextGate = this.gates[this.nextGateIndex];
        if (!nextGate) {
            this.lastZ = currentPos.z;
            return result;
        }

        // Check if drone crossed the gate's Z plane
        const gateZ = nextGate.position.z;
        const crossedZ = this.checkPlaneCrossing(this.lastZ, currentPos.z, gateZ);

        if (crossedZ) {
            // Verify drone is within gate bounds
            const withinBounds = this.checkWithinBounds(currentPos, nextGate.position);

            if (withinBounds) {
                result.crossed = true;
                result.gateIndex = this.nextGateIndex;

                // Advance to next gate
                this.nextGateIndex++;

                // Check if lap completed
                if (this.nextGateIndex >= this.gates.length) {
                    result.lapCompleted = true;
                    this.nextGateIndex = 0;
                }
            }
        }

        this.lastZ = currentPos.z;
        return result;
    }

    /**
     * Check if a value crossed a threshold between two frames
     */
    private checkPlaneCrossing(lastZ: number, currentZ: number, planeZ: number): boolean {
        return (lastZ < planeZ && currentZ >= planeZ) ||
               (lastZ > planeZ && currentZ <= planeZ);
    }

    /**
     * Check if position is within gate bounds (X and Y tolerance)
     */
    private checkWithinBounds(pos: Vector3, gatePos: Vector3): boolean {
        const dx = Math.abs(pos.x - gatePos.x);
        const dy = Math.abs(pos.y - gatePos.y);

        return dx < this.tolerance.halfWidth &&
               dy < this.tolerance.halfHeight;
    }

    /**
     * Get the index of the next gate to pass through
     */
    public getNextGateIndex(): number {
        return this.nextGateIndex;
    }

    /**
     * Get total number of gates
     */
    public getGateCount(): number {
        return this.gates.length;
    }

    /**
     * Reset detector state for new lap/session
     */
    public reset(): void {
        this.nextGateIndex = 0;
        this.lastZ = 0;
    }

    /**
     * Get the position of the next gate
     */
    public getNextGatePosition(): Vector3 | null {
        if (this.gates.length === 0) return null;
        return this.gates[this.nextGateIndex]?.position ?? null;
    }
}
