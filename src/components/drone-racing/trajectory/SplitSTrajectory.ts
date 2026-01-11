/**
 * Split-S / Power Loop Trajectory
 *
 * Demonstrates aggressive 3D maneuvers required to enter gates
 * from the "wrong" side - flying over/past then diving back through.
 *
 * The split-S maneuver:
 * 1. Approach gate from front
 * 2. Pull up and over (half loop up)
 * 3. Roll 180° to invert
 * 4. Pull through the gate from behind/above
 *
 * Track profile (side view showing one gate maneuver):
 *
 *                  ___
 *                 /   \
 *                /     |
 *       ____    /      |
 *      /    \__/       |  (gate here, entered from behind)
 *     /                |
 *    |                 v
 *    |
 *    \________________/
 *
 * The trajectory includes multiple gates requiring different approach angles.
 */

import { Trajectory, TrajectoryParams } from './Trajectory';
import { GatePosition } from '../types';

export interface SplitSParams extends TrajectoryParams {
    loopRadius: number;     // Radius of the vertical loops
    gateSpacing: number;    // Horizontal distance between gates
    numGates: number;       // Number of gates/maneuvers
}

export const DEFAULT_SPLITS_PARAMS: SplitSParams = {
    speed: 12.0,            // Moderate speed for control
    height: 5.0,            // Base height
    loopRadius: 6.0,        // 6m loop radius
    gateSpacing: 30.0,      // 30m between gates
    numGates: 3,            // 3 split-S maneuvers
};

export class SplitSTrajectory extends Trajectory {
    private loopRadius: number;
    private gateSpacing: number;
    private numGates: number;
    private totalLength: number = 0;

    // Segment definitions
    private segments: {
        type: 'straight' | 'loop_up' | 'loop_down' | 'dive';
        length: number;
        startZ: number;
        startY: number;
    }[] = [];

    constructor(params: Partial<SplitSParams> = {}) {
        const fullParams = { ...DEFAULT_SPLITS_PARAMS, ...params };
        super(fullParams);
        this.loopRadius = fullParams.loopRadius;
        this.gateSpacing = fullParams.gateSpacing;
        this.numGates = fullParams.numGates;

        this.buildTrack();
    }

    public getName(): string {
        return 'Split-S';
    }

    public getPeriod(): number {
        return this.totalLength / this.speed;
    }

    /**
     * Build the split-S track
     */
    private buildTrack(): void {
        const R = this.loopRadius;
        const baseY = this.height;

        // For each gate, we do:
        // 1. Straight approach
        // 2. Half-loop up (semicircle going up and back)
        // 3. Short inverted section
        // 4. Half-loop down (dive back through gate)
        // 5. Exit straight

        const straightLen = this.gateSpacing * 0.4;
        const loopArc = Math.PI * R;  // Semicircle arc length

        this.segments = [];
        let z = 0;
        let y = baseY;

        for (let gate = 0; gate < this.numGates; gate++) {
            // Approach straight
            this.segments.push({
                type: 'straight',
                length: straightLen,
                startZ: z,
                startY: y,
            });
            z += straightLen;
            this.totalLength += straightLen;

            // Half loop UP (pull up and over)
            this.segments.push({
                type: 'loop_up',
                length: loopArc,
                startZ: z,
                startY: y,
            });
            // After half loop up: y increases by 2*R, z stays same (we went up and back)
            y += 2 * R;
            this.totalLength += loopArc;

            // Dive section (coming back down through the gate)
            this.segments.push({
                type: 'dive',
                length: loopArc,
                startZ: z,
                startY: y,
            });
            // After dive: y decreases by 2*R, z increases by 2*R
            y -= 2 * R;
            z += 2 * R;
            this.totalLength += loopArc;
        }

        // Final straight to close the loop (return to start)
        const returnLen = z;  // Go back to start
        this.segments.push({
            type: 'straight',
            length: returnLen,
            startZ: z,
            startY: y,
        });
        this.totalLength += returnLen;
    }

    protected getPositionAtPhase(phase: number): { x: number; y: number; z: number } {
        const distance = phase * this.totalLength;
        const R = this.loopRadius;

        let accumulated = 0;
        for (const seg of this.segments) {
            if (accumulated + seg.length >= distance) {
                const localDist = distance - accumulated;
                const progress = localDist / seg.length;

                switch (seg.type) {
                    case 'straight': {
                        // Straight section - check if forward or return
                        const isReturn = seg.startZ > this.gateSpacing;
                        if (isReturn) {
                            // Going backward (return to start)
                            return {
                                x: 0,
                                y: seg.startY,
                                z: seg.startZ - progress * seg.length,
                            };
                        } else {
                            // Going forward
                            return {
                                x: 0,
                                y: seg.startY,
                                z: seg.startZ + progress * seg.length,
                            };
                        }
                    }

                    case 'loop_up': {
                        // Semicircle going up: starts at (0, startY, startZ)
                        // Center is at (0, startY + R, startZ)
                        // Goes from angle -π/2 (bottom) to π/2 (top)
                        const angle = -Math.PI / 2 + progress * Math.PI;
                        return {
                            x: 0,
                            y: seg.startY + R + R * Math.sin(angle),
                            z: seg.startZ - R * Math.cos(angle),
                        };
                    }

                    case 'dive': {
                        // Quarter circle dive: from top, diving forward and down
                        // Starts at top of loop (high), ends at gate level going forward
                        const angle = progress * Math.PI / 2;
                        return {
                            x: 0,
                            y: seg.startY - R * Math.sin(angle) - R * (1 - Math.cos(angle)),
                            z: seg.startZ + 2 * R * progress,
                        };
                    }

                    default:
                        return { x: 0, y: this.height, z: 0 };
                }
            }
            accumulated += seg.length;
        }

        // Should not reach here
        return { x: 0, y: this.height, z: 0 };
    }

    /**
     * Place gates at the apex of each loop (top of the split-S maneuver)
     * This is where the drone does the sudden change of direction
     */
    public override getGatePositions(): GatePosition[] {
        const gates: GatePosition[] = [];
        const R = this.loopRadius;
        const straightLen = this.gateSpacing * 0.4;

        // For each gate, calculate the apex position (top of the loop)
        // Each gate sequence moves: straightLen (approach) + 0 (loop_up stays at same z) + 2R (dive)
        // So accumulated z before gate n: n * (straightLen + 2R) + straightLen
        for (let gate = 0; gate < this.numGates; gate++) {
            // Z position: accumulated from previous gates + current approach
            // Previous gates contribute: gate * (straightLen + 2*R)
            // Current approach: straightLen
            const gateZ = gate * (straightLen + 2 * R) + straightLen;

            // Y position: top of the loop (base height + 2 * radius)
            const gateY = this.height + 2 * R;

            gates.push({
                position: {
                    x: 0,
                    y: gateY,  // Gate at apex height
                    z: gateZ,
                },
                heading: Math.PI,  // Gate faces -Z (drone comes from +Z side, inverted)
            });
        }

        return gates;
    }
}
