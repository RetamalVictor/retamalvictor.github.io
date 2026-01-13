/**
 * Gate-Based Trajectory Generator
 *
 * Automatically generates trajectories through a sequence of gates.
 * This is an enhancement over the reference approach where trajectories
 * are manually constructed - here we auto-compute the path and speeds.
 *
 * Algorithm:
 * 1. For each pair of gates, determine if straight line or arc is needed
 * 2. Compute safe speeds based on turn geometry
 * 3. Chain segments with jerk-limited velocity profiles
 *
 * The result is a closed-loop trajectory that the MPC can track.
 */

import { Vector3, GateWaypoint, TrajectorySegment, Waypoint, GatePosition } from '../types';
import { Trajectory } from './Trajectory';
import { SmoothLineSegment } from './segments/SmoothLineSegment';

export interface GeneratorConfig {
    maxAccel: number;       // Max centripetal acceleration (m/s²)
    maxSpeed: number;       // Max allowed speed (m/s)
    minSpeed: number;       // Min speed (m/s)
    defaultSpeed: number;   // Speed when no constraints
}

const DEFAULT_CONFIG: GeneratorConfig = {
    maxAccel: 25.0,     // 25 m/s² centripetal limit (aggressive racing)
    maxSpeed: 22.0,     // 22 m/s max (~80 km/h)
    minSpeed: 8.0,      // 8 m/s min (~30 km/h)
    defaultSpeed: 18.0, // 18 m/s default (~65 km/h)
};

/**
 * Generated trajectory from gate sequence
 */
export class GeneratedTrajectory extends Trajectory {
    private segments: TrajectorySegment[] = [];
    private segmentStartTimes: number[] = [];
    private totalDuration: number = 0;
    private gates: GateWaypoint[];

    constructor(segments: TrajectorySegment[], gates: GateWaypoint[]) {
        // Use default speed (will be overridden by segments)
        super({ speed: DEFAULT_CONFIG.defaultSpeed, height: 5.0 });

        this.segments = segments;
        this.gates = gates;

        // Compute segment start times
        let t = 0;
        for (const seg of segments) {
            this.segmentStartTimes.push(t);
            t += seg.getDuration();
        }
        this.totalDuration = t;
    }

    getName(): string {
        return 'Generated';
    }

    getPeriod(): number {
        return this.totalDuration;
    }

    /**
     * Get position at phase (used by base class)
     */
    protected getPositionAtPhase(phase: number): { x: number; y: number; z: number } {
        const time = phase * this.totalDuration;
        const { segment, localT } = this.findSegment(time);

        if (!segment) {
            return { x: 0, y: this.height, z: 0 };
        }

        return segment.getPosition(localT);
    }

    /**
     * Override getWaypoint to use curvature-based variable speed
     * - Position: from segment (smooth path)
     * - Velocity: segment direction scaled to curvature-based speed
     * - Acceleration/Jerk: from base class (consistent with variable speed)
     */
    public override getWaypoint(t: number): Waypoint {
        // Wrap time to trajectory period
        const period = this.getPeriod();
        const wrappedTime = ((t % period) + period) % period;
        const phase = wrappedTime / period;

        const { segment, localT } = this.findSegment(wrappedTime);

        if (!segment) {
            return {
                position: { x: 0, y: this.height, z: 0 },
                velocity: { x: 0, y: 0, z: 0 },
                acceleration: { x: 0, y: 0, z: 0 },
                jerk: { x: 0, y: 0, z: 0 },
                heading: 0,
                headingRate: 0,
                time: t,
            };
        }

        // Get position from segment (smooth path)
        const pos = segment.getPosition(localT);

        // Get velocity direction from segment, but scale to curvature-based speed
        const segVel = segment.getVelocity(localT);
        const segSpeed = Math.sqrt(segVel.x ** 2 + segVel.y ** 2 + segVel.z ** 2);
        const targetSpeed = this.getSpeedAtPhase(phase);

        let vel = segVel;
        if (segSpeed > 1e-6) {
            const scale = targetSpeed / segSpeed;
            vel = {
                x: segVel.x * scale,
                y: segVel.y * scale,
                z: segVel.z * scale,
            };
        }

        // Compute acceleration using numerical differentiation on scaled velocity
        const dt2 = 0.001;
        const v1 = this.getScaledVelocityAtTime(wrappedTime - dt2);
        const v2 = this.getScaledVelocityAtTime(wrappedTime + dt2);
        const acc = {
            x: (v2.x - v1.x) / (2 * dt2),
            y: (v2.y - v1.y) / (2 * dt2),
            z: (v2.z - v1.z) / (2 * dt2),
        };

        // Compute jerk using numerical differentiation on acceleration
        const dt3 = 0.002;
        let jerk = { x: 0, y: 0, z: 0 };
        if (wrappedTime > dt3 && wrappedTime < period - dt3) {
            const a1 = this.getScaledAccelerationAtTime(wrappedTime - dt3);
            const a2 = this.getScaledAccelerationAtTime(wrappedTime + dt3);
            jerk = {
                x: (a2.x - a1.x) / (2 * dt3),
                y: (a2.y - a1.y) / (2 * dt3),
                z: (a2.z - a1.z) / (2 * dt3),
            };
        }

        // Clamp jerk to reasonable bounds
        const maxJerk = 100;
        const jerkMag = Math.sqrt(jerk.x ** 2 + jerk.y ** 2 + jerk.z ** 2);
        if (jerkMag > maxJerk) {
            const scale = maxJerk / jerkMag;
            jerk = { x: jerk.x * scale, y: jerk.y * scale, z: jerk.z * scale };
        }

        // Compute heading from velocity
        const heading = Math.atan2(vel.x, vel.z);

        // Compute heading rate
        const heading1 = Math.atan2(v1.x, v1.z);
        const heading2 = Math.atan2(v2.x, v2.z);
        let headingRate = this.wrapAngle(heading2 - heading1) / (2 * dt2);
        const maxHeadingRate = 5.0;
        headingRate = Math.max(-maxHeadingRate, Math.min(maxHeadingRate, headingRate));

        return {
            position: pos,
            velocity: vel,
            acceleration: acc,
            jerk,
            heading,
            headingRate,
            time: t,
        };
    }

    /**
     * Helper: get velocity at time, scaled to curvature-based speed
     */
    private getScaledVelocityAtTime(t: number): Vector3 {
        const period = this.getPeriod();
        const wrappedTime = ((t % period) + period) % period;
        const phase = wrappedTime / period;

        const { segment, localT } = this.findSegment(wrappedTime);
        if (!segment) return { x: 0, y: 0, z: 0 };

        const segVel = segment.getVelocity(localT);
        const segSpeed = Math.sqrt(segVel.x ** 2 + segVel.y ** 2 + segVel.z ** 2);
        const targetSpeed = this.getSpeedAtPhase(phase);

        if (segSpeed > 1e-6) {
            const scale = targetSpeed / segSpeed;
            return { x: segVel.x * scale, y: segVel.y * scale, z: segVel.z * scale };
        }
        return segVel;
    }

    /**
     * Helper: get acceleration at time (from scaled velocity)
     */
    private getScaledAccelerationAtTime(t: number): Vector3 {
        const dt = 0.001;
        const v1 = this.getScaledVelocityAtTime(t - dt);
        const v2 = this.getScaledVelocityAtTime(t + dt);
        return {
            x: (v2.x - v1.x) / (2 * dt),
            y: (v2.y - v1.y) / (2 * dt),
            z: (v2.z - v1.z) / (2 * dt),
        };
    }

    /**
     * Find segment and local parameter for given time
     */
    private findSegment(time: number): { segment: TrajectorySegment | null; localT: number } {
        if (this.segments.length === 0) {
            return { segment: null, localT: 0 };
        }

        // Find which segment contains this time
        for (let i = 0; i < this.segments.length; i++) {
            const startTime = this.segmentStartTimes[i];
            const duration = this.segments[i].getDuration();

            if (time >= startTime && time < startTime + duration) {
                const localT = (time - startTime) / duration;
                return { segment: this.segments[i], localT };
            }
        }

        // Past end - return last segment at t=1
        const lastSeg = this.segments[this.segments.length - 1];
        return { segment: lastSeg, localT: 1 };
    }

    /**
     * Get gate positions for visualization
     */
    public override getGatePositions(): GatePosition[] {
        return this.gates.map(gate => ({
            position: { ...gate.position },
            heading: Math.atan2(gate.entranceDir.x, gate.entranceDir.z),
        }));
    }

    /**
     * Get segments for debugging
     */
    public getSegments(): TrajectorySegment[] {
        return this.segments;
    }
}

/**
 * Gate-Based Trajectory Generator
 *
 * Takes gate positions and generates optimal trajectory through them.
 */
export class GateTrajectoryGenerator {
    private config: GeneratorConfig;

    constructor(config: Partial<GeneratorConfig> = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
    }

    /**
     * Generate closed-loop trajectory through gates
     *
     * Simple approach: straight segments through gates at constant speed.
     * The MPC will handle speed adaptation at corners.
     */
    generate(gates: GateWaypoint[]): GeneratedTrajectory {
        if (gates.length < 2) {
            throw new Error('Need at least 2 gates to generate trajectory');
        }

        const minHeight = 1.5;
        const approachDist = 5.0;
        const speed = this.config.defaultSpeed;  // MPC will adapt

        const segments: TrajectorySegment[] = [];

        for (let i = 0; i < gates.length; i++) {
            const gate = gates[i];
            const nextGate = gates[(i + 1) % gates.length];
            const dir = this.normalize(gate.entranceDir);

            // Approach point
            const approach: Vector3 = {
                x: gate.position.x - dir.x * approachDist,
                y: Math.max(minHeight, gate.position.y - dir.y * approachDist),
                z: gate.position.z - dir.z * approachDist,
            };

            // Exit point
            const exit: Vector3 = {
                x: gate.position.x + dir.x * approachDist,
                y: Math.max(minHeight, gate.position.y + dir.y * approachDist),
                z: gate.position.z + dir.z * approachDist,
            };

            // Next approach point
            const nextDir = this.normalize(nextGate.entranceDir);
            const nextApproach: Vector3 = {
                x: nextGate.position.x - nextDir.x * approachDist,
                y: Math.max(minHeight, nextGate.position.y - nextDir.y * approachDist),
                z: nextGate.position.z - nextDir.z * approachDist,
            };

            // Segments: approach → gate → exit → next approach
            segments.push(new SmoothLineSegment(approach, gate.position, speed, speed));
            segments.push(new SmoothLineSegment(gate.position, exit, speed, speed));
            segments.push(new SmoothLineSegment(exit, nextApproach, speed, speed));
        }

        return new GeneratedTrajectory(segments, gates);
    }

    /**
     * Normalize a vector
     */
    private normalize(v: Vector3): Vector3 {
        const mag = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        if (mag < 1e-6) {
            return { x: 0, y: 0, z: 1 };
        }
        return {
            x: v.x / mag,
            y: v.y / mag,
            z: v.z / mag,
        };
    }
}
