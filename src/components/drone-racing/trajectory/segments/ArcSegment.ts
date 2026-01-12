/**
 * Arc Segment
 *
 * A circular arc trajectory segment with jerk-limited velocity profile.
 * Used for turns between gates when direction change is needed.
 *
 * The arc is defined in a plane with:
 * - center: Center point of the arc
 * - radius: Arc radius
 * - startAngle, endAngle: Angles in the arc plane
 * - normal: Normal vector to the arc plane (default Y-up for horizontal arcs)
 */

import { Vector3, TrajectorySegment } from '../../types';

export interface ArcParams {
    center: Vector3;
    radius: number;
    startAngle: number;  // Radians, angle from reference axis
    endAngle: number;    // Radians
    v0: number;          // Initial speed
    v1: number;          // Final speed
    planeNormal?: Vector3;  // Normal to arc plane (default: Y-up)
    height?: number;     // Y coordinate if horizontal arc
}

export class ArcSegment implements TrajectorySegment {
    private center: Vector3;
    private radius: number;
    private startAngle: number;
    private endAngle: number;
    private v0: number;
    private v1: number;
    private height: number;

    private arcLength: number;
    private duration: number;
    private jerk: number;
    private angleSpan: number;

    constructor(params: ArcParams) {
        this.center = { ...params.center };
        this.radius = params.radius;
        this.startAngle = params.startAngle;
        this.endAngle = params.endAngle;
        this.v0 = Math.max(params.v0, 0.1);
        this.v1 = Math.max(params.v1, 0.1);
        this.height = params.height ?? params.center.y;

        // Compute angle span (always go the short way unless specified)
        this.angleSpan = this.endAngle - this.startAngle;

        // Normalize to [-π, π] for shortest path
        while (this.angleSpan > Math.PI) this.angleSpan -= 2 * Math.PI;
        while (this.angleSpan < -Math.PI) this.angleSpan += 2 * Math.PI;

        // Arc length = radius * |angle|
        this.arcLength = this.radius * Math.abs(this.angleSpan);

        if (this.arcLength < 1e-6) {
            this.duration = 0;
            this.jerk = 0;
        } else {
            // Jerk-limited profile: T = 2L / (v0 + v1)
            this.duration = (2 * this.arcLength) / (this.v0 + this.v1);

            // Jerk: j = 4(v1 - v0) / T²
            this.jerk = (4 * (this.v1 - this.v0)) / (this.duration * this.duration);
        }
    }

    /**
     * Get position at normalized parameter t ∈ [0, 1]
     * Arc is in XZ plane at constant Y height
     */
    getPosition(t: number): Vector3 {
        t = Math.max(0, Math.min(1, t));

        if (this.duration === 0) {
            return this.getPointAtAngle(this.startAngle);
        }

        // Convert to actual time and compute distance traveled
        const time = t * this.duration;
        const s = this.computeDistance(time);

        // Convert distance to angle
        const angleTraveled = (s / this.arcLength) * this.angleSpan;
        const currentAngle = this.startAngle + angleTraveled;

        return this.getPointAtAngle(currentAngle);
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
        const s = this.computeDistance(time);

        // Convert distance to angle
        const angleTraveled = (s / this.arcLength) * this.angleSpan;
        const currentAngle = this.startAngle + angleTraveled;

        // Velocity is tangent to arc
        // For arc in XZ plane: tangent = (-sin(θ), 0, cos(θ)) * sign(angleSpan)
        const sign = this.angleSpan >= 0 ? 1 : -1;
        return {
            x: -Math.sin(currentAngle) * speed * sign,
            y: 0,
            z: Math.cos(currentAngle) * speed * sign,
        };
    }

    /**
     * Get acceleration at normalized parameter t ∈ [0, 1]
     * Has both tangential (from jerk profile) and centripetal components
     */
    getAcceleration(t: number): Vector3 {
        t = Math.max(0, Math.min(1, t));

        if (this.duration === 0) {
            return { x: 0, y: 0, z: 0 };
        }

        const time = t * this.duration;
        const speed = this.computeSpeed(time);
        const tangentialAccel = this.computeTangentialAcceleration(time);
        const s = this.computeDistance(time);

        // Current angle
        const angleTraveled = (s / this.arcLength) * this.angleSpan;
        const currentAngle = this.startAngle + angleTraveled;

        // Centripetal acceleration: v²/r, pointing toward center
        const centripetalMag = (speed * speed) / this.radius;

        // Centripetal direction (toward center)
        const pos = this.getPointAtAngle(currentAngle);
        const toCenterX = this.center.x - pos.x;
        const toCenterZ = this.center.z - pos.z;
        const toCenterMag = Math.sqrt(toCenterX * toCenterX + toCenterZ * toCenterZ);

        let centripetalX = 0, centripetalZ = 0;
        if (toCenterMag > 1e-6) {
            centripetalX = (toCenterX / toCenterMag) * centripetalMag;
            centripetalZ = (toCenterZ / toCenterMag) * centripetalMag;
        }

        // Tangential direction
        const sign = this.angleSpan >= 0 ? 1 : -1;
        const tangentX = -Math.sin(currentAngle) * sign;
        const tangentZ = Math.cos(currentAngle) * sign;

        return {
            x: centripetalX + tangentX * tangentialAccel,
            y: 0,
            z: centripetalZ + tangentZ * tangentialAccel,
        };
    }

    getDuration(): number {
        return this.duration;
    }

    getLength(): number {
        return this.arcLength;
    }

    getStartPosition(): Vector3 {
        return this.getPointAtAngle(this.startAngle);
    }

    getEndPosition(): Vector3 {
        return this.getPointAtAngle(this.startAngle + this.angleSpan);
    }

    getStartSpeed(): number {
        return this.v0;
    }

    getEndSpeed(): number {
        return this.v1;
    }

    /**
     * Get point on arc at given angle
     */
    private getPointAtAngle(angle: number): Vector3 {
        return {
            x: this.center.x + this.radius * Math.cos(angle),
            y: this.height,
            z: this.center.z + this.radius * Math.sin(angle),
        };
    }

    /**
     * Compute speed at time t using jerk-limited profile
     */
    private computeSpeed(time: number): number {
        const T = this.duration;
        const halfT = T / 2;

        if (time <= halfT) {
            return this.v0 + (this.jerk / 2) * time * time;
        } else {
            const dt = T - time;
            return this.v1 - (this.jerk / 2) * dt * dt;
        }
    }

    /**
     * Compute tangential acceleration at time t
     */
    private computeTangentialAcceleration(time: number): number {
        const T = this.duration;
        const halfT = T / 2;

        if (time <= halfT) {
            return this.jerk * time;
        } else {
            return this.jerk * (T - time);
        }
    }

    /**
     * Compute distance traveled at time t
     */
    private computeDistance(time: number): number {
        const T = this.duration;
        const halfT = T / 2;

        if (time <= halfT) {
            return this.v0 * time + (this.jerk / 6) * time * time * time;
        } else {
            const sMid = this.v0 * halfT + (this.jerk / 6) * halfT * halfT * halfT;
            const dt = time - halfT;
            const dtEnd = T - time;
            const vMid = this.v0 + (this.jerk / 2) * halfT * halfT;

            return sMid + vMid * dt + (this.jerk / 6) * (halfT * halfT * halfT - dtEnd * dtEnd * dtEnd);
        }
    }
}
