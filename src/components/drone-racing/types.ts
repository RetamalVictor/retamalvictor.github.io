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
// Demo Configuration (DroneRacingDemo)
// ============================================

export interface SimulationConfig {
    defaultSpeed: number;       // Default trajectory speed (m/s)
    defaultHeight: number;      // Default flight height (m)
}

export interface TrailConfig {
    maxLength: number;          // Maximum trail points
    color: number;              // Trail color (hex)
    opacity: number;            // Trail opacity (0-1)
}

export interface GateToleranceConfig {
    halfWidth: number;          // Half-width for crossing detection (m)
    halfHeight: number;         // Half-height for crossing detection (m)
}

export interface DroneRacingDemoConfig {
    simulation: SimulationConfig;
    trail: TrailConfig;
    gateTolerance: GateToleranceConfig;
    showTrajectoryLine: boolean; // Debug: show trajectory path
}

export const DEFAULT_DEMO_CONFIG: DroneRacingDemoConfig = {
    simulation: {
        defaultSpeed: 18.0,     // 18 m/s (~65 km/h)
        defaultHeight: 4.0,     // 4 meters
    },
    trail: {
        maxLength: 1000,        // 1000 points max
        color: 0x22c55e,        // Green
        opacity: 0.6,
    },
    gateTolerance: {
        halfWidth: 2.0,         // 2m (for 3m wide gate)
        halfHeight: 2.0,        // 2m (for 3m tall gate)
    },
    showTrajectoryLine: false,  // Hidden by default
};

// ============================================
// MPC State Indices (for type-safe array access)
// ============================================

export enum MPCStateIndex {
    PX = 0,
    PY = 1,
    PZ = 2,
    VX = 3,
    VY = 4,
    VZ = 5,
    QW = 6,
    QX = 7,
    QY = 8,
    QZ = 9,
}

export enum MPCInputIndex {
    THRUST = 0,
    ROLL_RATE = 1,
    PITCH_RATE = 2,
    YAW_RATE = 3,
}

// State and input dimensions
export const MPC_STATE_DIM = 10;
export const MPC_INPUT_DIM = 4;

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
