/**
 * Circle Trajectory
 *
 * Simple circular path - the baseline trajectory for testing.
 * Good for validating MPC tracking before more complex maneuvers.
 */

import { Trajectory, TrajectoryParams } from './Trajectory';

export interface CircleParams extends TrajectoryParams {
    radius: number;     // Circle radius in meters
}

export const DEFAULT_CIRCLE_PARAMS: CircleParams = {
    speed: 15.0,
    height: 4.0,
    radius: 20.0,
};

export class CircleTrajectory extends Trajectory {
    private radius: number;

    constructor(params: Partial<CircleParams> = {}) {
        const fullParams = { ...DEFAULT_CIRCLE_PARAMS, ...params };
        super(fullParams);
        this.radius = fullParams.radius;
    }

    public getName(): string {
        return 'Circle';
    }

    public getPeriod(): number {
        return (2 * Math.PI * this.radius) / this.speed;
    }

    protected getPositionAtPhase(phase: number): { x: number; z: number } {
        const angle = phase * 2 * Math.PI;
        return {
            x: this.radius * Math.cos(angle),
            z: this.radius * Math.sin(angle),
        };
    }
}
