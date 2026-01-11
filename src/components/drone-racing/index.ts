/**
 * Drone Racing - Module Exports
 *
 * Clean architecture with separation of concerns:
 * - DroneDynamics: Pure physics simulation
 * - DroneVisualization: Pure 3D rendering
 * - MPC: True Model Predictive Control with Sequential QP
 */

// Demo
export { DroneRacingDemo } from './DroneRacingDemo';

// Core - Dynamics
export { DroneDynamics, DEFAULT_DYNAMICS_PARAMS } from './core/DroneDynamics';
export type { DynamicsParams, DynamicsState } from './core/DroneDynamics';

// Core - Composed drone (dynamics + visualization)
export { RacingDrone } from './core/RacingDrone';
export type { RacingDroneConfig } from './core/RacingDrone';

// Visualization
export { DroneVisualization, DEFAULT_VISUALIZATION_CONFIG } from './visualization/DroneVisualization';
export type { DroneVisualizationConfig } from './visualization/DroneVisualization';

// Control - MPC
export { MPC, DEFAULT_MPC_CONFIG } from './control/MPC';
export type { MPCConfig, MPCReference } from './control/MPC';

export { MPCModel } from './control/MPCModel';
export type { MPCState, MPCInput, LinearizedDynamics } from './control/MPCModel';

export { QPSolver, MatrixUtils } from './control/QPSolver';
export type { QPProblem, QPSolution, QPOptions } from './control/QPSolver';

// Types
export * from './types';
