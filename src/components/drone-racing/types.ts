/**
 * Drone Racing Pipeline - Type Definitions
 *
 * Based on drone-racing-msgs from the real system:
 * - 10-state drone state vector
 * - 4-input control commands
 * - Gate keypoint definitions
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

export interface Point2D {
    u: number;  // image x coordinate (pixels)
    v: number;  // image y coordinate (pixels)
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
// Gate Detection (from drone-racing-msgs)
// ============================================

/** Gate corner keypoints in image space */
export interface GateKeypoints {
    corners: [Point2D, Point2D, Point2D, Point2D];  // TL, TR, BR, BL
    confidence: number;  // 0-1
}

/** Gate detection result */
export interface GateDetection {
    gateId: number;
    keypoints: GateKeypoints;
    reprojectionError: number;
    visible: boolean;
}

/** Gate pose in world frame */
export interface GatePose {
    gateId: number;
    position: Vector3;
    orientation: Quaternion;
    innerSize: number;  // gate opening size in meters
}

// ============================================
// Trajectory (from trajectory-generation)
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

/** Trajectory segment between waypoints */
export interface TrajectorySegment {
    startWaypoint: Waypoint;
    endWaypoint: Waypoint;
    duration: number;
    gateId?: number;  // associated gate if passing through
}

/** Full trajectory through gates */
export interface Trajectory {
    segments: TrajectorySegment[];
    totalDuration: number;
    waypoints: Waypoint[];  // sampled at fixed rate for visualization
}

// ============================================
// MPC State (for visualization)
// ============================================

export interface MPCState {
    predictedStates: DroneState[];   // prediction horizon
    referenceStates: Waypoint[];     // reference trajectory
    currentCommand: ControlCommand;
    horizonTime: number;             // seconds
    trackingError: number;           // position error magnitude
}

// ============================================
// Pipeline Status (for UI)
// ============================================

export interface PipelineStatus {
    detection: {
        gatesDetected: number;
        activeGate: number | null;
    };
    estimation: {
        distanceToNextGate: number;
        estimatedPosition: Vector3;
    };
    trajectory: {
        currentSpeed: number;
        maxSpeed: number;
        progressPercent: number;
    };
    control: {
        thrust: number;
        trackingError: number;
    };
}

// ============================================
// Configuration
// ============================================

export interface RacingConfig {
    // Track
    gateSize: number;           // meters (default 1.52)
    numGates: number;           // number of gates
    trackRadius: number;        // approximate track size

    // Drone
    maxSpeed: number;           // m/s
    maxAcceleration: number;    // m/s^2

    // MPC
    predictionHorizon: number;  // seconds (default 0.5)
    controlRate: number;        // Hz
    commandDelay: number;       // seconds (default 0.05)

    // Detection
    cameraFov: number;          // degrees
    detectionNoise: number;     // pixels
}

export const DEFAULT_CONFIG: RacingConfig = {
    gateSize: 1.52,
    numGates: 4,
    trackRadius: 8,
    maxSpeed: 6.0,
    maxAcceleration: 8.0,
    predictionHorizon: 0.5,
    controlRate: 50,
    commandDelay: 0.05,
    cameraFov: 90,
    detectionNoise: 2.0,
};
