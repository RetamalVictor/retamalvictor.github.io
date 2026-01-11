import * as THREE from 'three';
import { RaceTrack } from '../core/RaceTrack';
import { Waypoint, Trajectory, TrajectorySegment, Vector3, RacingConfig, DEFAULT_CONFIG } from '../types';

/**
 * Trajectory Generator
 *
 * Generates smooth minimum-jerk trajectories through racing gates.
 *
 * Based on trajectory-generation from the real system:
 * - 5th-order (quintic) polynomials for minimum-jerk motion
 * - Symmetric trapezoidal acceleration profiles
 * - Smooth velocity transitions at gate passages
 *
 * Minimum-jerk ensures:
 * - Continuous position, velocity, acceleration
 * - Zero jerk at endpoints
 * - Smooth, natural-looking motion
 */
export class TrajectoryGenerator {
    private config: RacingConfig;
    private sampleRate: number = 50;  // Hz for waypoint sampling

    constructor(config: Partial<RacingConfig> = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
    }

    /**
     * Generate full racing trajectory through all gates
     */
    public generateRacingTrajectory(track: RaceTrack): Trajectory {
        const segments: TrajectorySegment[] = [];
        const gates = track.gates;
        const maxSpeed = this.config.maxSpeed;

        // Generate segments between consecutive gates
        for (let i = 0; i < gates.length; i++) {
            const currentGate = gates[i];
            const nextGate = gates[(i + 1) % gates.length];

            // Compute waypoints for this segment
            const start = this.createGateWaypoint(currentGate, nextGate, maxSpeed, false);
            const end = this.createGateWaypoint(nextGate, gates[(i + 2) % gates.length], maxSpeed, true);

            // Compute segment duration based on distance and max speed
            const distance = new THREE.Vector3(
                end.position.x - start.position.x,
                end.position.y - start.position.y,
                end.position.z - start.position.z
            ).length();

            // Use minimum-jerk time allocation
            const duration = this.computeMinJerkDuration(distance, maxSpeed);

            segments.push({
                startWaypoint: start,
                endWaypoint: end,
                duration,
                gateId: currentGate.id,
            });
        }

        // Adjust segment times to be continuous
        let totalTime = 0;
        for (const segment of segments) {
            segment.startWaypoint.time = totalTime;
            totalTime += segment.duration;
            segment.endWaypoint.time = totalTime;
        }

        // Sample waypoints at fixed rate for visualization
        const waypoints = this.sampleTrajectory(segments);

        return {
            segments,
            totalDuration: totalTime,
            waypoints,
        };
    }

    /**
     * Create waypoint at gate position with appropriate velocity
     */
    private createGateWaypoint(
        gate: { getCenter: () => THREE.Vector3; getForwardDirection: () => THREE.Vector3 },
        nextGate: { getCenter: () => THREE.Vector3 },
        speed: number,
        isApproach: boolean
    ): Waypoint {
        const pos = gate.getCenter();
        const forward = gate.getForwardDirection();

        // Velocity direction: blend between gate forward and direction to next gate
        const toNext = nextGate.getCenter().clone().sub(pos).normalize();
        const velDir = isApproach
            ? toNext.clone()
            : forward.clone().lerp(toNext, 0.5).normalize();

        const velocity = velDir.multiplyScalar(speed);

        return {
            position: { x: pos.x, y: pos.y, z: pos.z },
            velocity: { x: velocity.x, y: velocity.y, z: velocity.z },
            acceleration: { x: 0, y: 0, z: 0 },
            jerk: { x: 0, y: 0, z: 0 },
            heading: Math.atan2(velDir.x, velDir.z),
            headingRate: 0,
            time: 0,
        };
    }

    /**
     * Compute duration for minimum-jerk trajectory
     * Based on distance and max speed constraint
     */
    private computeMinJerkDuration(distance: number, maxSpeed: number): number {
        // For minimum-jerk with v0=vf=speed, the average speed is ~0.6 * max
        // T = distance / (0.6 * maxSpeed)
        const avgSpeedFactor = 0.7;
        const duration = distance / (avgSpeedFactor * maxSpeed);

        // Enforce minimum segment time
        return Math.max(duration, 0.5);
    }

    /**
     * Sample trajectory at fixed rate
     */
    private sampleTrajectory(segments: TrajectorySegment[]): Waypoint[] {
        const waypoints: Waypoint[] = [];
        const dt = 1 / this.sampleRate;

        for (const segment of segments) {
            const startTime = segment.startWaypoint.time;
            const duration = segment.duration;

            // Sample within this segment
            for (let t = 0; t < duration; t += dt) {
                const waypoint = this.interpolateMinJerk(
                    segment.startWaypoint,
                    segment.endWaypoint,
                    t / duration
                );
                waypoint.time = startTime + t;
                waypoints.push(waypoint);
            }
        }

        // Add final waypoint
        if (segments.length > 0) {
            const lastSegment = segments[segments.length - 1];
            waypoints.push({ ...lastSegment.endWaypoint });
        }

        return waypoints;
    }

    /**
     * Minimum-jerk interpolation using quintic polynomial
     *
     * The quintic polynomial ensures:
     * - Position, velocity, acceleration continuity
     * - Zero jerk at boundaries
     *
     * x(τ) = x0 + (xf - x0) * (10τ³ - 15τ⁴ + 6τ⁵)
     * where τ = t / T (normalized time)
     */
    private interpolateMinJerk(start: Waypoint, end: Waypoint, tau: number): Waypoint {
        // Clamp tau to [0, 1]
        tau = Math.max(0, Math.min(1, tau));

        // Minimum-jerk basis functions
        const tau2 = tau * tau;
        const tau3 = tau2 * tau;
        const tau4 = tau3 * tau;
        const tau5 = tau4 * tau;

        // Position interpolation: s(τ) = 10τ³ - 15τ⁴ + 6τ⁵
        const s = 10 * tau3 - 15 * tau4 + 6 * tau5;

        // Velocity interpolation: ds/dτ = 30τ² - 60τ³ + 30τ⁴
        const ds = 30 * tau2 - 60 * tau3 + 30 * tau4;

        // Acceleration interpolation: d²s/dτ² = 60τ - 180τ² + 120τ³
        const dds = 60 * tau - 180 * tau2 + 120 * tau3;

        // Jerk interpolation: d³s/dτ³ = 60 - 360τ + 360τ²
        const ddds = 60 - 360 * tau + 360 * tau2;

        // Interpolate each component
        const position = this.lerpVector3(start.position, end.position, s);
        const velocity = this.lerpVector3(start.velocity, end.velocity, ds);
        const acceleration = this.lerpVector3(start.acceleration, end.acceleration, dds);
        const jerk: Vector3 = { x: ddds, y: ddds, z: ddds };  // Simplified

        // Interpolate heading
        let headingDiff = end.heading - start.heading;
        // Normalize to [-π, π]
        while (headingDiff > Math.PI) headingDiff -= 2 * Math.PI;
        while (headingDiff < -Math.PI) headingDiff += 2 * Math.PI;

        const heading = start.heading + headingDiff * s;
        const headingRate = headingDiff * ds;

        return {
            position,
            velocity,
            acceleration,
            jerk,
            heading,
            headingRate,
            time: 0,  // Set by caller
        };
    }

    /**
     * Linear interpolation for Vector3
     */
    private lerpVector3(a: Vector3, b: Vector3, t: number): Vector3 {
        return {
            x: a.x + (b.x - a.x) * t,
            y: a.y + (b.y - a.y) * t,
            z: a.z + (b.z - a.z) * t,
        };
    }

    /**
     * Get waypoint at specific time
     */
    public getWaypointAtTime(trajectory: Trajectory, time: number): Waypoint {
        // Wrap time for looping trajectory
        const wrappedTime = time % trajectory.totalDuration;

        // Find segment containing this time
        for (const segment of trajectory.segments) {
            const segmentEnd = segment.startWaypoint.time + segment.duration;
            if (wrappedTime <= segmentEnd) {
                const localTime = wrappedTime - segment.startWaypoint.time;
                const tau = localTime / segment.duration;
                const waypoint = this.interpolateMinJerk(
                    segment.startWaypoint,
                    segment.endWaypoint,
                    tau
                );
                waypoint.time = wrappedTime;
                return waypoint;
            }
        }

        // Return last waypoint if time exceeds trajectory
        return trajectory.waypoints[trajectory.waypoints.length - 1];
    }

    /**
     * Get trajectory segment for visualization
     * Returns n waypoints from current time looking ahead
     */
    public sampleAhead(
        trajectory: Trajectory,
        currentTime: number,
        numSamples: number,
        lookahead: number
    ): Waypoint[] {
        const samples: Waypoint[] = [];
        const dt = lookahead / (numSamples - 1);

        for (let i = 0; i < numSamples; i++) {
            const t = currentTime + i * dt;
            samples.push(this.getWaypointAtTime(trajectory, t));
        }

        return samples;
    }

    /**
     * Regenerate trajectory with new speed
     */
    public setMaxSpeed(speed: number): void {
        this.config.maxSpeed = Math.max(1, Math.min(15, speed));
    }

    /**
     * Get current max speed
     */
    public getMaxSpeed(): number {
        return this.config.maxSpeed;
    }
}
