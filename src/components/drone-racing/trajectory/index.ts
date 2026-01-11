/**
 * Trajectory Module
 *
 * Provides various trajectory types for drone racing demonstrations.
 */

export { Trajectory } from './Trajectory';
export type { TrajectoryParams } from './Trajectory';
export { CircleTrajectory, DEFAULT_CIRCLE_PARAMS } from './CircleTrajectory';
export type { CircleParams } from './CircleTrajectory';
export { Figure8Trajectory, DEFAULT_FIGURE8_PARAMS } from './Figure8Trajectory';
export type { Figure8Params } from './Figure8Trajectory';
export { HairpinTrajectory, DEFAULT_HAIRPIN_PARAMS } from './HairpinTrajectory';
export type { HairpinParams } from './HairpinTrajectory';
export { SnakeTrajectory, DEFAULT_SNAKE_PARAMS } from './SnakeTrajectory';
export type { SnakeParams } from './SnakeTrajectory';
export { RaceTrackTrajectory, DEFAULT_RACETRACK_PARAMS } from './RaceTrackTrajectory';
export type { RaceTrackParams } from './RaceTrackTrajectory';
export { Racing3DTrajectory, DEFAULT_RACING3D_PARAMS } from './Racing3DTrajectory';
export type { Racing3DParams } from './Racing3DTrajectory';
export { SplitSTrajectory, DEFAULT_SPLITS_PARAMS } from './SplitSTrajectory';
export type { SplitSParams } from './SplitSTrajectory';

import { Trajectory } from './Trajectory';
import { CircleTrajectory } from './CircleTrajectory';
import { Figure8Trajectory } from './Figure8Trajectory';
import { HairpinTrajectory } from './HairpinTrajectory';
import { SnakeTrajectory } from './SnakeTrajectory';
import { RaceTrackTrajectory } from './RaceTrackTrajectory';
import { Racing3DTrajectory } from './Racing3DTrajectory';
import { SplitSTrajectory } from './SplitSTrajectory';

/**
 * Available trajectory types
 */
export type TrajectoryType = 'circle' | 'figure8' | 'hairpin' | 'snake' | 'racetrack' | 'racing3d' | 'splits';

/**
 * Trajectory metadata for UI
 */
export interface TrajectoryInfo {
    type: TrajectoryType;
    name: string;
    description: string;
}

export const TRAJECTORY_INFO: TrajectoryInfo[] = [
    {
        type: 'circle',
        name: 'Circle',
        description: 'Simple circular path - baseline trajectory',
    },
    {
        type: 'figure8',
        name: 'Figure-8',
        description: 'Infinity-shaped curve with full heading range',
    },
    {
        type: 'hairpin',
        name: 'Hairpin',
        description: 'Racing-style tight 180° turns',
    },
    {
        type: 'snake',
        name: 'Snake',
        description: 'Serpentine weaving pattern',
    },
    {
        type: 'racetrack',
        name: 'Race Track',
        description: 'Multi-gate racing course',
    },
    {
        type: 'racing3d',
        name: '3D Racing',
        description: 'Full 3D with dives and climbs',
    },
    {
        type: 'splits',
        name: 'Split-S',
        description: 'Power loops through gates',
    },
];

/**
 * Factory function to create demo trajectories
 *
 * Note: These are pre-defined trajectories for demonstration.
 * The trajectory generator will compute optimal speeds based on
 * gate positions and path curvature at runtime.
 */
export function createTrajectory(
    type: TrajectoryType,
    speed: number = 18.0,
    height: number = 4.0
): Trajectory {
    switch (type) {
        case 'circle':
            return new CircleTrajectory({ speed: 20.0, height, radius: 25.0 });
        case 'figure8':
            return new Figure8Trajectory({ speed: 12.0, height, size: 25.0 });
        case 'hairpin':
            return new HairpinTrajectory({ speed: 16.0, height, turnRadius: 12.0, straightLength: 40.0 });
        case 'snake':
            return new SnakeTrajectory({ speed: 14.0, height });
        case 'racetrack':
            return new RaceTrackTrajectory({ speed: 18.0, height, gateSpacing: 30.0, turnRadius: 15.0 });
        case 'racing3d':
            return new Racing3DTrajectory({ speed: 18.0, height, trackLength: 100.0, minHeight: 2.0, maxHeight: 12.0 });
        case 'splits':
            return new SplitSTrajectory({ speed: 14.0, height, loopRadius: 8.0, gateSpacing: 35.0, numGates: 3 });
        default:
            return new CircleTrajectory({ speed, height, radius: 25.0 });
    }
}
