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
 * Factory function to create trajectories
 * Speed is limited based on trajectory curvature to ensure drone can track it
 */
export function createTrajectory(
    type: TrajectoryType,
    speed: number = 12.0,
    height: number = 4.0
): Trajectory {
    switch (type) {
        case 'circle':
            // Circle has constant curvature, can handle higher speeds
            return new CircleTrajectory({ speed: Math.min(speed, 15.0), height, radius: 20.0 });
        case 'figure8':
            // Figure-8 has tight curvature at center, needs lower speed
            // Max speed ~ sqrt(5 * size) for centripetal accel < 20 m/s²
            return new Figure8Trajectory({ speed: Math.min(speed, 8.0), height, size: 20.0 });
        case 'hairpin':
            // Hairpin has tight 180° turns, moderate speed
            return new HairpinTrajectory({ speed: Math.min(speed, 12.0), height, turnRadius: 8.0, straightLength: 30.0 });
        case 'snake':
            // Snake with smooth turnarounds
            return new SnakeTrajectory({ speed: Math.min(speed, 10.0), height });
        case 'racetrack':
            // Race track with various turn radii
            return new RaceTrackTrajectory({ speed: Math.min(speed, 12.0), height, gateSpacing: 25.0, turnRadius: 10.0 });
        case 'racing3d':
            // Full 3D racing with altitude changes
            return new Racing3DTrajectory({ speed: Math.min(speed, 15.0), height, trackLength: 80.0, minHeight: 2.0, maxHeight: 10.0 });
        case 'splits':
            // Split-S maneuvers through gates
            return new SplitSTrajectory({ speed: Math.min(speed, 12.0), height, loopRadius: 6.0, gateSpacing: 25.0, numGates: 2 });
        default:
            return new CircleTrajectory({ speed, height, radius: 20.0 });
    }
}
