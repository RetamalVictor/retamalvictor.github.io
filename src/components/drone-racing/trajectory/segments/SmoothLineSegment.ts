/**
 * Smooth Line Segment
 *
 * A straight-line trajectory segment with jerk-limited velocity profile.
 * Based on the reference trajectory generation approach.
 *
 * The jerk-limited profile ensures smooth acceleration transitions:
 * - First half: constant positive jerk (velocity increasing smoothly)
 * - Second half: constant negative jerk (velocity reaching v1 smoothly)
 *
 * Key formulas:
 * - Total time: T = 2 * length / (v0 + v1)
 * - Jerk magnitude: |j| = 4 * |v1 - v0| / T²
 */

import { Vector3, TrajectorySegment } from '../../types';

export class SmoothLineSegment implements TrajectorySegment {
    private p0: Vector3;
    private p1: Vector3;
    private v0: number;  // Initial speed (scalar, along line direction)
    private v1: number;  // Final speed (scalar, along line direction)

    private direction: Vector3;  // Unit vector from p0 to p1
    private length: number;      // Distance from p0 to p1
    private duration: number;    // Total time to traverse segment
    private jerk: number;        // Jerk magnitude (signed)

    constructor(p0: Vector3, p1: Vector3, v0: number, v1: number) {
        this.p0 = { ...p0 };
        this.p1 = { ...p1 };
        this.v0 = Math.max(v0, 0.1);  // Avoid zero velocity (causes infinite time)
        this.v1 = Math.max(v1, 0.1);

        // Compute direction and length
        const dx = p1.x - p0.x;
        const dy = p1.y - p0.y;
        const dz = p1.z - p0.z;
        this.length = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (this.length < 1e-6) {
            // Degenerate segment (start = end)
            this.direction = { x: 1, y: 0, z: 0 };
            this.length = 0;
            this.duration = 0;
            this.jerk = 0;
        } else {
            this.direction = {
                x: dx / this.length,
                y: dy / this.length,
                z: dz / this.length,
            };

            // Jerk-limited profile: T = 2L / (v0 + v1)
            this.duration = (2 * this.length) / (this.v0 + this.v1);

            // Jerk: j = 4(v1 - v0) / T²
            this.jerk = (4 * (this.v1 - this.v0)) / (this.duration * this.duration);
        }
    }

    /**
     * Get position at normalized parameter t ∈ [0, 1]
     */
    getPosition(t: number): Vector3 {
        t = Math.max(0, Math.min(1, t));

        if (this.duration === 0) {
            return { ...this.p0 };
        }

        // Convert to actual time
        const time = t * this.duration;

        // Compute distance traveled using jerk-limited profile
        const s = this.computeDistance(time);

        return {
            x: this.p0.x + s * this.direction.x,
            y: this.p0.y + s * this.direction.y,
            z: this.p0.z + s * this.direction.z,
        };
    }

    /**
     * Get velocity at normalized parameter t ∈ [0, 1]
     */
    getVelocity(t: number): Vector3 {
        t = Math.max(0, Math.min(1, t));

        if (this.duration === 0) {
            return { x: 0, y: 0, z: 0 };
        }

        const time = t * this.duration;
        const speed = this.computeSpeed(time);

        return {
            x: speed * this.direction.x,
            y: speed * this.direction.y,
            z: speed * this.direction.z,
        };
    }

    /**
     * Get acceleration at normalized parameter t ∈ [0, 1]
     */
    getAcceleration(t: number): Vector3 {
        t = Math.max(0, Math.min(1, t));

        if (this.duration === 0) {
            return { x: 0, y: 0, z: 0 };
        }

        const time = t * this.duration;
        const accel = this.computeAcceleration(time);

        return {
            x: accel * this.direction.x,
            y: accel * this.direction.y,
            z: accel * this.direction.z,
        };
    }

    /**
     * Get total duration in seconds
     */
    getDuration(): number {
        return this.duration;
    }

    /**
     * Get total arc length in meters
     */
    getLength(): number {
        return this.length;
    }

    /**
     * Get start position
     */
    getStartPosition(): Vector3 {
        return { ...this.p0 };
    }

    /**
     * Get end position
     */
    getEndPosition(): Vector3 {
        return { ...this.p1 };
    }

    /**
     * Get start velocity (scalar)
     */
    getStartSpeed(): number {
        return this.v0;
    }

    /**
     * Get end velocity (scalar)
     */
    getEndSpeed(): number {
        return this.v1;
    }

    /**
     * Compute speed at time t using jerk-limited profile
     *
     * For t ∈ [0, T/2]: v(t) = v0 + (j/2) * t²
     * For t ∈ [T/2, T]: v(t) = v1 - (j/2) * (T-t)²
     *
     * This creates a smooth S-curve velocity transition.
     */
    private computeSpeed(time: number): number {
        const T = this.duration;
        const halfT = T / 2;

        if (time <= halfT) {
            // First half: accelerating with increasing rate
            return this.v0 + (this.jerk / 2) * time * time;
        } else {
            // Second half: decelerating to final velocity
            const dt = T - time;
            return this.v1 - (this.jerk / 2) * dt * dt;
        }
    }

    /**
     * Compute acceleration at time t
     *
     * a(t) = dv/dt
     * For t ∈ [0, T/2]: a(t) = j * t
     * For t ∈ [T/2, T]: a(t) = j * (T - t)
     */
    private computeAcceleration(time: number): number {
        const T = this.duration;
        const halfT = T / 2;

        if (time <= halfT) {
            return this.jerk * time;
        } else {
            return this.jerk * (T - time);
        }
    }

    /**
     * Compute distance traveled at time t by integrating velocity
     *
     * For t ∈ [0, T/2]:
     *   s(t) = v0*t + (j/6) * t³
     *
     * For t ∈ [T/2, T]:
     *   s(t) = s(T/2) + v1*(t - T/2) - (j/6) * (T-t)³ + (j/6) * (T/2)³
     */
    private computeDistance(time: number): number {
        const T = this.duration;
        const halfT = T / 2;

        if (time <= halfT) {
            // First half: s(t) = v0*t + (j/6)*t³
            return this.v0 * time + (this.jerk / 6) * time * time * time;
        } else {
            // Distance at midpoint
            const sMid = this.v0 * halfT + (this.jerk / 6) * halfT * halfT * halfT;

            // Second half: integrate from midpoint
            // s(t) = sMid + ∫[T/2 to t] v(τ) dτ
            // where v(τ) = v1 - (j/2)(T-τ)²
            // = sMid + v1*(t - T/2) + (j/6)*[(T-t)³ - (T/2)³]
            const dt = time - halfT;
            const dtEnd = T - time;

            return sMid + this.v1 * dt + (this.jerk / 6) * (dtEnd * dtEnd * dtEnd - halfT * halfT * halfT);
        }
    }
}
