/**
 * Math Utilities
 *
 * Shared mathematical operations used across the drone racing demo.
 * Centralizes angle wrapping, vector operations, and numerical differentiation.
 */

// ============================================================================
// Constants
// ============================================================================

/** Standard time step for numerical differentiation */
export const NUMERICAL_DIFF_DT = 0.001;

/** Small epsilon for floating point comparisons */
export const EPSILON = 1e-10;

// ============================================================================
// Types
// ============================================================================

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

// ============================================================================
// Angle Operations
// ============================================================================

/**
 * Wrap angle to [-PI, PI] range
 */
export function wrapAngle(angle: number): number {
    while (angle > Math.PI) angle -= 2 * Math.PI;
    while (angle < -Math.PI) angle += 2 * Math.PI;
    return angle;
}

/**
 * Wrap angular rate to [-PI/dt, PI/dt] range
 * Used to prevent discontinuities in heading rate calculations
 */
export function wrapAngleRate(rate: number, dt: number): number {
    const limit = Math.PI / dt;
    while (rate > limit) rate -= 2 * limit;
    while (rate < -limit) rate += 2 * limit;
    return rate;
}

/**
 * Compute shortest angular difference between two angles
 * Result is in [-PI, PI]
 */
export function angleDifference(target: number, current: number): number {
    return wrapAngle(target - current);
}

/**
 * Linearly interpolate between two angles, handling wrap-around
 */
export function lerpAngle(a: number, b: number, t: number): number {
    const diff = angleDifference(b, a);
    return wrapAngle(a + diff * t);
}

// ============================================================================
// Vector Operations
// ============================================================================

/**
 * Compute magnitude (length) of a 3D vector
 */
export function magnitude(v: Vector3): number {
    return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

/**
 * Compute magnitude of a 2D vector (x, z components only)
 */
export function magnitude2D(v: Vector3): number {
    return Math.sqrt(v.x * v.x + v.z * v.z);
}

/**
 * Normalize a 3D vector to unit length
 * Returns zero vector if input magnitude is too small
 */
export function normalize(v: Vector3): Vector3 {
    const mag = magnitude(v);
    if (mag < EPSILON) {
        return { x: 0, y: 0, z: 0 };
    }
    return {
        x: v.x / mag,
        y: v.y / mag,
        z: v.z / mag,
    };
}

/**
 * Compute dot product of two 3D vectors
 */
export function dot(a: Vector3, b: Vector3): number {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
 * Compute cross product of two 3D vectors
 */
export function cross(a: Vector3, b: Vector3): Vector3 {
    return {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    };
}

/**
 * Add two 3D vectors
 */
export function add(a: Vector3, b: Vector3): Vector3 {
    return {
        x: a.x + b.x,
        y: a.y + b.y,
        z: a.z + b.z,
    };
}

/**
 * Subtract two 3D vectors (a - b)
 */
export function subtract(a: Vector3, b: Vector3): Vector3 {
    return {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    };
}

/**
 * Scale a 3D vector by a scalar
 */
export function scale(v: Vector3, s: number): Vector3 {
    return {
        x: v.x * s,
        y: v.y * s,
        z: v.z * s,
    };
}

/**
 * Linearly interpolate between two 3D vectors
 */
export function lerp(a: Vector3, b: Vector3, t: number): Vector3 {
    return {
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
        z: a.z + (b.z - a.z) * t,
    };
}

// ============================================================================
// Quaternion Operations
// ============================================================================

/**
 * Normalize a quaternion to unit length
 */
export function normalizeQuaternion(q: Quaternion): Quaternion {
    const mag = Math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (mag < EPSILON) {
        // Return identity quaternion if magnitude too small
        return { w: 1, x: 0, y: 0, z: 0 };
    }
    return {
        w: q.w / mag,
        x: q.x / mag,
        y: q.y / mag,
        z: q.z / mag,
    };
}

/**
 * Create a quaternion from a heading angle (rotation around Y axis)
 */
export function quaternionFromHeading(heading: number): Quaternion {
    const halfAngle = heading / 2;
    return {
        w: Math.cos(halfAngle),
        x: 0,
        y: Math.sin(halfAngle),
        z: 0,
    };
}

/**
 * Extract heading (yaw) angle from a quaternion
 */
export function headingFromQuaternion(q: Quaternion): number {
    // Yaw from quaternion: atan2(2(qw*qy + qx*qz), 1 - 2(qy² + qz²))
    // Simplified for Y-up convention
    const siny_cosp = 2 * (q.w * q.y + q.x * q.z);
    const cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    return Math.atan2(siny_cosp, cosy_cosp);
}

// ============================================================================
// Numerical Differentiation
// ============================================================================

/**
 * Compute numerical derivative using central difference
 * @param fn Function to differentiate
 * @param t Point at which to compute derivative
 * @param dt Time step (defaults to NUMERICAL_DIFF_DT)
 */
export function numericalDerivative(
    fn: (t: number) => number,
    t: number,
    dt: number = NUMERICAL_DIFF_DT
): number {
    const f1 = fn(t + dt);
    const f0 = fn(t - dt);
    return (f1 - f0) / (2 * dt);
}

/**
 * Compute numerical derivative of a vector function
 */
export function numericalDerivativeVector(
    fn: (t: number) => Vector3,
    t: number,
    dt: number = NUMERICAL_DIFF_DT
): Vector3 {
    const p1 = fn(t + dt);
    const p0 = fn(t - dt);
    const scale = 1 / (2 * dt);
    return {
        x: (p1.x - p0.x) * scale,
        y: (p1.y - p0.y) * scale,
        z: (p1.z - p0.z) * scale,
    };
}

/**
 * Compute forward difference derivative (for boundary cases)
 */
export function forwardDerivative(
    fn: (t: number) => number,
    t: number,
    dt: number = NUMERICAL_DIFF_DT
): number {
    return (fn(t + dt) - fn(t)) / dt;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Clamp a value between min and max
 */
export function clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value));
}

/**
 * Safe array access with fallback
 */
export function safeArrayAccess<T>(arr: T[], index: number, fallback: T): T {
    return index >= 0 && index < arr.length ? arr[index] : fallback;
}

/**
 * Linear interpolation between two numbers
 */
export function lerpScalar(a: number, b: number, t: number): number {
    return a + (b - a) * t;
}
