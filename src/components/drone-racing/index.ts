/**
 * Drone Racing Demo - Module Exports
 *
 * Interactive demo showcasing autonomous drone racing pipeline:
 * - Gate Detection (PnP pose estimation)
 * - Trajectory Generation (minimum-jerk profiles)
 * - MPC Control (trajectory tracking)
 */

// Main demo
export { DroneRacingDemo } from './DroneRacingDemo';

// Core components
export { RacingDrone } from './core/RacingDrone';
export { RaceGate } from './core/RaceGate';
export { RaceTrack } from './core/RaceTrack';

// Perception
export { GateDetector } from './perception/GateDetector';
export { PnPSolver } from './perception/PnPSolver';

// Planning
export { TrajectoryGenerator } from './planning/TrajectoryGenerator';

// Control
export { SimplifiedMPC } from './control/SimplifiedMPC';

// Types
export * from './types';
