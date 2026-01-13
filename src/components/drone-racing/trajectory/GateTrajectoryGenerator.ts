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
    minSpeed: number;       // Min speed (m/s) - for tight corners
    defaultSpeed: number;   // Speed when no constraints
}

const DEFAULT_CONFIG: GeneratorConfig = {
    maxAccel: 30.0,     // 30 m/s² centripetal limit (~3g)
    maxSpeed: 32.0,     // 32 m/s max (~115 km/h)
    minSpeed: 7.0,      // 7 m/s for tight turns (~25 km/h)
    defaultSpeed: 26.0, // 26 m/s default (~94 km/h)
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
     * Override getSpeedAtPhase to use segment-based speeds
     * This replaces the base class curvature-based speed with direction-change based speed
     */
    protected override getSpeedAtPhase(phase: number): number {
        const time = phase * this.totalDuration;
        const { segment, localT } = this.findSegment(time);

        if (!segment) {
            return this.speed;
        }

        // Get velocity magnitude from segment (which interpolates entry/exit speeds)
        const vel = segment.getVelocity(localT);
        return Math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2);
    }

    /**
     * Override getWaypoint to use segment-based variable speed
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
     * Simple approach: straight segments with direction-based speeds.
     */
    generate(gates: GateWaypoint[]): GeneratedTrajectory {
        if (gates.length < 2) {
            throw new Error('Need at least 2 gates to generate trajectory');
        }

        const minHeight = 1.5;
        const approachDist = 5.0;
        const maxSpeed = this.config.maxSpeed;
        const minSpeed = this.config.minSpeed;

        // Generate waypoints
        interface WaypointInfo {
            pos: Vector3;
            gateIndex: number;
            type: 'approach' | 'gate' | 'exit';
        }

        const waypoints: WaypointInfo[] = [];

        for (let i = 0; i < gates.length; i++) {
            const gate = gates[i];
            const dir = this.normalize(gate.entranceDir);

            const approach: Vector3 = {
                x: gate.position.x - dir.x * approachDist,
                y: Math.max(minHeight, gate.position.y - dir.y * approachDist),
                z: gate.position.z - dir.z * approachDist,
            };

            const exit: Vector3 = {
                x: gate.position.x + dir.x * approachDist,
                y: Math.max(minHeight, gate.position.y + dir.y * approachDist),
                z: gate.position.z + dir.z * approachDist,
            };

            waypoints.push({ pos: approach, gateIndex: i, type: 'approach' });
            waypoints.push({ pos: gate.position, gateIndex: i, type: 'gate' });
            waypoints.push({ pos: exit, gateIndex: i, type: 'exit' });
        }

        // Compute speeds based on direction change
        const waypointSpeeds: number[] = [];

        for (let i = 0; i < waypoints.length; i++) {
            const prev = waypoints[(i - 1 + waypoints.length) % waypoints.length];
            const curr = waypoints[i];
            const next = waypoints[(i + 1) % waypoints.length];

            const dirIn = this.normalize(this.sub(curr.pos, prev.pos));
            const dirOut = this.normalize(this.sub(next.pos, curr.pos));
            const dot = dirIn.x * dirOut.x + dirIn.y * dirOut.y + dirIn.z * dirOut.z;
            const angle = Math.acos(Math.max(-1, Math.min(1, dot)));

            // Speed based on angle - threshold + sqrt for aggressive slowdown
            const threshold = Math.PI / 9; // 20 degrees - no slowdown below this
            let speed: number;
            if (angle < threshold) {
                speed = maxSpeed; // Full speed for gentle turns
            } else {
                // Sqrt slowdown - aggressive for moderate angles, ensures safe power loop
                const sharpness = (angle - threshold) / (Math.PI - threshold);
                speed = maxSpeed - (maxSpeed - minSpeed) * Math.sqrt(sharpness);
            }

            waypointSpeeds.push(speed);
        }

        // Smooth speeds with acceleration limits (forward + backward pass)
        // This ensures speed changes are physically achievable
        const maxTangentialAccel = this.config.maxAccel * 0.7; // 70% for safe deceleration

        // Forward pass: limit acceleration
        for (let i = 0; i < waypoints.length; i++) {
            const next = (i + 1) % waypoints.length;
            const dist = this.distance(waypoints[i].pos, waypoints[next].pos);
            // v_next² ≤ v_curr² + 2*a*d  →  v_next ≤ √(v_curr² + 2*a*d)
            const maxNextSpeed = Math.sqrt(waypointSpeeds[i] ** 2 + 2 * maxTangentialAccel * dist);
            waypointSpeeds[next] = Math.min(waypointSpeeds[next], maxNextSpeed);
        }

        // Backward pass: limit braking (iterate multiple times for closed loop)
        for (let pass = 0; pass < 5; pass++) {
            for (let i = waypoints.length - 1; i >= 0; i--) {
                const next = (i + 1) % waypoints.length;
                const dist = this.distance(waypoints[i].pos, waypoints[next].pos);
                // v_curr² ≤ v_next² + 2*a*d  →  v_curr ≤ √(v_next² + 2*a*d)
                const maxCurrSpeed = Math.sqrt(waypointSpeeds[next] ** 2 + 2 * maxTangentialAccel * dist);
                waypointSpeeds[i] = Math.min(waypointSpeeds[i], maxCurrSpeed);
            }
        }

        // Create segments
        const segments: TrajectorySegment[] = [];

        for (let i = 0; i < waypoints.length; i++) {
            const curr = waypoints[i];
            const next = waypoints[(i + 1) % waypoints.length];
            const currSpeed = waypointSpeeds[i];
            const nextSpeed = waypointSpeeds[(i + 1) % waypoints.length];

            segments.push(new SmoothLineSegment(curr.pos, next.pos, currSpeed, nextSpeed));
        }

        return new GeneratedTrajectory(segments, gates);
    }

    // ============================================
    // Vector utilities
    // ============================================

    private sub(a: Vector3, b: Vector3): Vector3 {
        return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
    }

    private distance(a: Vector3, b: Vector3): number {
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dz = a.z - b.z;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
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
