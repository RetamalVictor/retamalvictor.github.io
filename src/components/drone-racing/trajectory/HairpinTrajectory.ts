/**
 * Hairpin Trajectory
 *
 * Racing-style trajectory with tight 180° turns at each end.
 * Simulates a drag strip with hairpin turns - tests aggressive banking.
 *
 * Shape: Two semicircles connected by straight sections
 *        ___
 *       /   \
 *      |     |  <- tight turn (radius r)
 *      |     |
 *      |     |  <- straight section (length L)
 *      |     |
 *       \___/   <- tight turn (radius r)
 */

import { Trajectory, TrajectoryParams } from './Trajectory';

export interface HairpinParams extends TrajectoryParams {
    turnRadius: number;     // Radius of the hairpin turns
    straightLength: number; // Length of straight sections
}

export const DEFAULT_HAIRPIN_PARAMS: HairpinParams = {
    speed: 15.0,
    height: 4.0,
    turnRadius: 8.0,
    straightLength: 30.0,
};

export class HairpinTrajectory extends Trajectory {
    private turnRadius: number;
    private straightLength: number;

    // Segment lengths for parameterization
    private turnArc: number;
    private totalLength: number;

    constructor(params: Partial<HairpinParams> = {}) {
        const fullParams = { ...DEFAULT_HAIRPIN_PARAMS, ...params };
        super(fullParams);
        this.turnRadius = fullParams.turnRadius;
        this.straightLength = fullParams.straightLength;

        // Compute segment lengths
        this.turnArc = Math.PI * this.turnRadius;  // Semicircle arc
        this.totalLength = 2 * this.straightLength + 2 * this.turnArc;
    }

    public getName(): string {
        return 'Hairpin';
    }

    public getPeriod(): number {
        return this.totalLength / this.speed;
    }

    protected getPositionAtPhase(phase: number): { x: number; z: number } {
        // Convert phase to distance along track
        const distance = phase * this.totalLength;

        // Segment boundaries
        const seg1End = this.straightLength;                    // End of first straight
        const seg2End = seg1End + this.turnArc;                 // End of first turn
        const seg3End = seg2End + this.straightLength;          // End of second straight
        // seg4End = totalLength (end of second turn)

        const r = this.turnRadius;
        const L = this.straightLength;
        const spacing = 2 * r;  // Distance between the two straights

        if (distance < seg1End) {
            // Segment 1: Straight going +Z
            const progress = distance / this.straightLength;
            return {
                x: spacing / 2,
                z: -L / 2 + progress * L,
            };
        } else if (distance < seg2End) {
            // Segment 2: Top hairpin turn (180° right)
            const arcDistance = distance - seg1End;
            const angle = (arcDistance / this.turnArc) * Math.PI;
            return {
                x: r * Math.cos(angle),
                z: L / 2 + r * Math.sin(angle),
            };
        } else if (distance < seg3End) {
            // Segment 3: Straight going -Z
            const progress = (distance - seg2End) / this.straightLength;
            return {
                x: -spacing / 2,
                z: L / 2 - progress * L,
            };
        } else {
            // Segment 4: Bottom hairpin turn (180° right)
            const arcDistance = distance - seg3End;
            const angle = (arcDistance / this.turnArc) * Math.PI;
            return {
                x: -r * Math.cos(angle),
                z: -L / 2 - r * Math.sin(angle),
            };
        }
    }
}
