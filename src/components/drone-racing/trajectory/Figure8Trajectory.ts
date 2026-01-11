/**
 * Figure-8 (Lemniscate) Trajectory
 *
 * Infinity-shaped curve that tests the full range of heading angles.
 * The drone must handle smooth transitions through 360° of heading.
 *
 * Uses a simplified lemniscate parameterization:
 *   x(t) = a * sin(t)
 *   z(t) = a * sin(t) * cos(t) = (a/2) * sin(2t)
 */

import { Trajectory, TrajectoryParams } from './Trajectory';
import { GatePosition } from '../types';

export interface Figure8Params extends TrajectoryParams {
    size: number;       // Size of the figure-8 (half-width)
}

export const DEFAULT_FIGURE8_PARAMS: Figure8Params = {
    speed: 8.0,         // Lower speed for tight curves (centripetal accel ~ v²/(size/4))
    height: 4.0,
    size: 20.0,         // Larger size for gentler curvature
};

export class Figure8Trajectory extends Trajectory {
    private size: number;

    constructor(params: Partial<Figure8Params> = {}) {
        const fullParams = { ...DEFAULT_FIGURE8_PARAMS, ...params };
        super(fullParams);
        this.size = fullParams.size;
    }

    public getName(): string {
        return 'Figure-8';
    }

    public getPeriod(): number {
        // Arc length of lemniscate ≈ 5.24 * size
        const arcLength = 5.24 * this.size;
        return arcLength / this.speed;
    }

    protected getPositionAtPhase(phase: number): { x: number; z: number } {
        const t = phase * 2 * Math.PI;

        // Lemniscate parameterization
        // x = a * sin(t)
        // z = a * sin(t) * cos(t) = (a/2) * sin(2t)
        return {
            x: this.size * Math.sin(t),
            z: (this.size / 2) * Math.sin(2 * t),
        };
    }

    /**
     * Place gates at key points: center crossing and extremes of both loops
     */
    public override getGatePositions(): GatePosition[] {
        const period = this.getPeriod();

        // Gate phases:
        // 0.0 = center crossing (start)
        // 0.25 = far end of first loop (x = +size)
        // 0.5 = center crossing again
        // 0.75 = far end of second loop (x = -size)
        const gatePhases = [0.0, 0.25, 0.5, 0.75];

        return gatePhases.map(phase => {
            const wp = this.getWaypoint(phase * period);
            return {
                position: { ...wp.position },
                heading: wp.heading,
            };
        });
    }
}
