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
import { SplitSTrajectory } from './SplitSTrajectory';

/**
 * Available trajectory types
 */
export type TrajectoryType = 'figure8' | 'splits' | 'splits-classic' | 'dive' | 'crazy';

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
        type: 'figure8',
        name: 'Figure-8',
        description: 'Infinity-shaped curve with gates',
    },
    {
        type: 'splits',
        name: 'Split-S',
        description: 'Stacked gates - dive through pairs',
    },
    {
        type: 'splits-classic',
        name: 'Split-S Classic',
        description: 'Original power loop trajectory',
    },
    {
        type: 'dive',
        name: 'Dive',
        description: 'Gates stacked vertically - power loops',
    },
    {
        type: 'crazy',
        name: 'Crazy',
        description: 'Stacked gates, dives, and chaos',
    },
];

/**
 * Factory function to create trajectories
 *
 * Gate-based trajectories use the generator with automatic speed computation.
 * Classic trajectories use the original class-based approach.
 */
export function createTrajectory(
    type: TrajectoryType,
    _speed: number = 18.0,
    height: number = 4.0
): Trajectory {
    const generator = new GateTrajectoryGenerator();

    switch (type) {
        case 'figure8':
            return generator.generate(createFigure8Gates(height));
        case 'splits':
            return generator.generate(createSplitSGates(height));
        case 'splits-classic':
            // Original Split-S with predefined trajectory (no gates)
            return new SplitSTrajectory({ speed: 14.0, height, loopRadius: 8.0, gateSpacing: 35.0, numGates: 3 });
        case 'dive':
            return generator.generate(createDiveGates(height));
        case 'crazy':
            return generator.generate(createCrazyGates(height));
        default:
            return generator.generate(createFigure8Gates(height));
    }
}

// =====================================================
// Gate Definitions for Each Trajectory Type
// =====================================================

/**
 * Figure-8: Racing scale lemniscate with smooth curves
 *
 * For racing at ~70 km/h (20 m/s) with 15m turn radius,
 * gates need to be spaced ~60m apart.
 */
function createFigure8Gates(height: number): GateWaypoint[] {
    const size = 50;  // Half-width of figure-8 (100m total width)

    return [
        // Center crossing (going right) - slight forward angle for smooth entry
        {
            position: { x: 0, y: height, z: 0 },
            entranceDir: { x: 0.866, y: 0, z: 0.5 },  // 30° off center
        },
        // Right loop - far right, going backward
        {
            position: { x: size, y: height, z: -20 },
            entranceDir: { x: 0, y: 0, z: -1 },
        },
        // Center crossing (going left) - opposite direction
        {
            position: { x: 0, y: height, z: 0 },
            entranceDir: { x: -0.866, y: 0, z: -0.5 },
        },
        // Left loop - far left, going forward
        {
            position: { x: -size, y: height, z: 20 },
            entranceDir: { x: 0, y: 0, z: 1 },
        },
    ];
}

/**
 * Split-S: Stacked gate pairs for power loop maneuvers
 *
 * Racing scale with larger spacing for high-speed flight.
 * Each split has gates offset in Z to allow for the dive trajectory.
 */
function createSplitSGates(height: number): GateWaypoint[] {
    const gateSpacing = 80;  // Large spacing for racing speeds
    const numSplits = 3;
    const highHeight = height + 12;  // High gate
    const lowHeight = height;        // Low gate at base height

    const gates: GateWaypoint[] = [];

    for (let i = 0; i < numSplits; i++) {
        const z = i * gateSpacing;

        // High gate - drone climbs up, enters going forward
        gates.push({
            position: { x: 0, y: highHeight, z },
            entranceDir: { x: 0, y: 0.3, z: 0.954 },  // Slight climb
        });

        // Low gate - offset forward for dive trajectory
        gates.push({
            position: { x: 0, y: lowHeight, z: z + 25 },
            entranceDir: { x: 0, y: -0.3, z: 0.954 },  // Diving forward
        });
    }

    return gates;
}

/**
 * Dive: Gates stacked immediately on top of each other
 *
 * Each stack has two gates at the SAME X/Z position:
 * - High gate: drone enters going forward
 * - Low gate: directly below, drone dives down and forward
 *
 * The low gate has an angled entrance (down + forward) so the
 * exit trajectory continues forward, not into the ground.
 */
function createDiveGates(height: number): GateWaypoint[] {
    const numStacks = 3;
    const stackSpacing = 50;  // Distance between stacks along Z
    const highHeight = height + 10;  // Top gate (y = 14)
    const lowHeight = height;        // Bottom gate at same height as base

    const gates: GateWaypoint[] = [];

    for (let i = 0; i < numStacks; i++) {
        const z = i * stackSpacing;

        // High gate - enter going forward
        gates.push({
            position: { x: 0, y: highHeight, z },
            entranceDir: { x: 0, y: 0, z: 1 },
        });

        // Low gate - SAME X/Z, directly below!
        // Enter diving down at 60° angle (mostly down, some forward)
        // This ensures exit point stays above ground
        gates.push({
            position: { x: 0, y: lowHeight, z },
            entranceDir: { x: 0, y: -0.866, z: 0.5 },  // 60° dive angle
        });
    }

    return gates;
}

/**
 * Crazy: Chaotic track with stacked gates, dives, and challenging maneuvers
 *
 * Features:
 * - Stacked vertical gates
 * - Sharp altitude changes
 * - Gates at various angles
 * - Tight turns
 */
function createCrazyGates(height: number): GateWaypoint[] {
    const minHeight = 2;
    const maxHeight = 14;

    return [
        // Start - low gate
        {
            position: { x: 0, y: minHeight, z: 0 },
            entranceDir: { x: 0, y: 0.3, z: 1 },  // Climb out
        },

        // Stacked pair 1 - climb to high then dive to low
        {
            position: { x: 0, y: maxHeight, z: 20 },
            entranceDir: { x: 0.5, y: 0, z: 0.866 },
        },
        {
            position: { x: 5, y: minHeight, z: 30 },
            entranceDir: { x: 0, y: -0.5, z: 0.866 },  // Steep dive
        },

        // Sharp turn at ground level
        {
            position: { x: 20, y: minHeight + 1, z: 40 },
            entranceDir: { x: 1, y: 0.2, z: 0 },
        },

        // Climb while turning
        {
            position: { x: 35, y: height, z: 35 },
            entranceDir: { x: 0.707, y: 0.3, z: -0.707 },
        },

        // High gate - inverted approach
        {
            position: { x: 40, y: maxHeight, z: 20 },
            entranceDir: { x: 0, y: 0, z: -1 },
        },

        // Stacked pair 2 - two gates on top of each other
        {
            position: { x: 35, y: maxHeight - 2, z: 5 },
            entranceDir: { x: -0.5, y: -0.2, z: -0.866 },
        },
        {
            position: { x: 30, y: minHeight + 2, z: 0 },
            entranceDir: { x: -0.866, y: 0, z: -0.5 },
        },

        // Low swooping turn back
        {
            position: { x: 15, y: minHeight, z: -5 },
            entranceDir: { x: -1, y: 0.1, z: 0 },
        },

        // Final climb back to start
        {
            position: { x: 0, y: height, z: -5 },
            entranceDir: { x: 0, y: 0.2, z: 1 },
        },
    ];
}
