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

// Gate-based trajectory generation
export { GateTrajectoryGenerator, GeneratedTrajectory } from './GateTrajectoryGenerator';
export type { GeneratorConfig } from './GateTrajectoryGenerator';
export { SmoothLineSegment } from './segments/SmoothLineSegment';
export { ArcSegment } from './segments/ArcSegment';
export type { ArcParams } from './segments/ArcSegment';

import { GateTrajectoryGenerator } from './GateTrajectoryGenerator';
import { GateWaypoint } from '../types';
import { Trajectory } from './Trajectory';

/**
 * Factory function to create the racing trajectory
 *
 * Single trajectory with stacked gates for power loop demonstration.
 */
export function createTrajectory(
    _speed: number = 18.0,
    height: number = 4.0
): Trajectory {
    const generator = new GateTrajectoryGenerator();
    return generator.generate(createRacingGates(height));
}

// =====================================================
// Racing Gate Layout
// =====================================================

/**
 * Racing track with one stacked gate pair (power loop)
 *
 * Simple oval-ish track with:
 * - Start gate
 * - One stacked gate pair (high + low) for power loop
 * - Turn gates
 */
function createRacingGates(height: number): GateWaypoint[] {
    const lowHeight = height;       // 4m
    const highHeight = height + 8;  // 12m - stacked gate

    return [
        // Gate 1: Start - going forward
        {
            position: { x: 0, y: lowHeight, z: 0 },
            entranceDir: { x: 0, y: 0, z: 1 },
        },

        // Gate 2: High gate (top of stacked pair) - climb up
        {
            position: { x: 0, y: highHeight, z: 30 },
            entranceDir: { x: 0, y: 0.3, z: 0.954 },
        },

        // Gate 3: Low gate (bottom of stacked pair) - dive down
        // Same X/Z as high gate, directly below
        {
            position: { x: 0, y: lowHeight, z: 30 },
            entranceDir: { x: 0, y: -0.7, z: 0.714 },  // 45° dive
        },

        // Gate 4: Far turn - going right
        {
            position: { x: 25, y: lowHeight, z: 50 },
            entranceDir: { x: 1, y: 0, z: 0 },
        },

        // Gate 5: Return path - going back
        {
            position: { x: 35, y: lowHeight, z: 25 },
            entranceDir: { x: 0, y: 0, z: -1 },
        },

        // Gate 6: Final turn - back to start
        {
            position: { x: 20, y: lowHeight, z: -10 },
            entranceDir: { x: -1, y: 0, z: 0 },
        },
    ];
}
