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
 * Centripetal acceleration limit: v²/r < 25 m/s²
 */
export function createTrajectory(
    type: TrajectoryType,
    speed: number = 18.0,
    height: number = 4.0
): Trajectory {
    switch (type) {
        case 'circle':
            // Circle r=25m: max v = sqrt(25*25) = 25 m/s
            return new CircleTrajectory({ speed: Math.min(speed, 22.0), height, radius: 25.0 });
        case 'figure8':
            // Figure-8 size=25m, tightest radius ~6m: max v = sqrt(25*6) = 12 m/s
            return new Figure8Trajectory({ speed: Math.min(speed, 12.0), height, size: 25.0 });
        case 'hairpin':
            // Hairpin r=12m: max v = sqrt(25*12) = 17 m/s
            return new HairpinTrajectory({ speed: Math.min(speed, 16.0), height, turnRadius: 12.0, straightLength: 40.0 });
        case 'snake':
            // Snake with wider turns
            return new SnakeTrajectory({ speed: Math.min(speed, 14.0), height });
        case 'racetrack':
            // Race track r=15m: max v = sqrt(25*15) = 19 m/s
            return new RaceTrackTrajectory({ speed: Math.min(speed, 18.0), height, gateSpacing: 30.0, turnRadius: 15.0 });
        case 'racing3d':
            // Full 3D racing with altitude changes, larger track
            return new Racing3DTrajectory({ speed: Math.min(speed, 20.0), height, trackLength: 120.0, minHeight: 2.0, maxHeight: 12.0 });
        case 'splits':
            // Split-S r=8m: max v = sqrt(25*8) = 14 m/s
            return new SplitSTrajectory({ speed: Math.min(speed, 14.0), height, loopRadius: 8.0, gateSpacing: 35.0, numGates: 3 });
        default:
            return new CircleTrajectory({ speed, height, radius: 25.0 });
    }
}
