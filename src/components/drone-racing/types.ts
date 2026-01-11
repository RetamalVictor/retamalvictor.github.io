/**
 * Drone Racing - Type Definitions
 *
 * Core types for drone simulation and MPC:
 * - 10-state drone state vector
 * - 4-input control commands
 * - Trajectory waypoint format
 */

// ============================================
// Core Math Types
// ============================================

export interface Vector3 {
    x: number;
    y: number;
    z: number;
}

export interface Quaternion {
    w: number;
    x: number;
    y: number;
    z: number;
}

// ============================================
// Drone State (10-state vector from MPC)
// ============================================

export interface DroneState {
    position: Vector3;          // [px, py, pz] in meters
    orientation: Quaternion;    // [qw, qx, qy, qz] normalized
    velocity: Vector3;          // [vx, vy, vz] in m/s
    timestamp: number;          // microseconds
}

// ============================================
// Control Commands (4-input vector)
// ============================================

export interface ControlCommand {
    thrust: number;      // normalized thrust (m/s^2)
    rollRate: number;    // rad/s
    pitchRate: number;   // rad/s
    yawRate: number;     // rad/s
    timestamp: number;   // microseconds
}

// ============================================
// Trajectory
// ============================================

/** Single trajectory waypoint with full state */
export interface Waypoint {
    position: Vector3;
    velocity: Vector3;
    acceleration: Vector3;
    jerk: Vector3;
    heading: number;       // yaw angle (rad)
    headingRate: number;   // yaw rate (rad/s)
    time: number;          // time from trajectory start (seconds)
}

// ============================================
// Configuration
// ============================================

export interface RacingConfig {
    // Drone
    maxSpeed: number;           // m/s
    maxAcceleration: number;    // m/s^2

    // MPC
    predictionHorizon: number;  // seconds
    controlRate: number;        // Hz
    commandDelay: number;       // seconds
}

export const DEFAULT_CONFIG: RacingConfig = {
    maxSpeed: 6.0,
    maxAcceleration: 8.0,
    predictionHorizon: 0.5,
    controlRate: 50,
    commandDelay: 0.05,
};

// ============================================
// Racing Gates
// ============================================

/** Racing gate that the drone must fly through */
export interface Gate {
    position: Vector3;      // Center of gate
    orientation: Quaternion; // Gate facing direction
    width: number;          // Gate width (meters)
    height: number;         // Gate height (meters)
    index: number;          // Gate number in sequence
}

/** Gate position data from trajectory (before full orientation is computed) */
export interface GatePosition {
    position: Vector3;
    heading: number;        // Yaw angle the gate faces (rad)
}

// ============================================
// Trajectory Generation
// ============================================

/**
 * Gate waypoint for trajectory generation
 *
 * Used as input to TrajectoryGenerator to define where gates are
 * and what direction the drone should enter them from.
 */
export interface GateWaypoint {
    position: Vector3;      // Gate center position
    entranceDir: Vector3;   // Unit vector - direction drone enters FROM
    speed?: number;         // Optional target speed at gate (auto-computed if not set)
}

/**
 * Trajectory segment interface
 *
 * Represents a single segment of a generated trajectory (line or arc).
 * The generator chains these together to form the full path.
 */
export interface TrajectorySegment {
    /** Get position at normalized parameter t ∈ [0, 1] */
    getPosition(t: number): Vector3;

    /** Get velocity at normalized parameter t ∈ [0, 1] */
    getVelocity(t: number): Vector3;

    /** Get acceleration at normalized parameter t ∈ [0, 1] */
    getAcceleration(t: number): Vector3;

    /** Get total duration of this segment in seconds */
    getDuration(): number;

    /** Get total arc length of this segment in meters */
    getLength(): number;
}
