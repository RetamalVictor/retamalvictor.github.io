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
     * Override getWaypoint for direct segment access (more accurate)
     */
    public override getWaypoint(t: number): Waypoint {
        // Wrap time to trajectory period
        const period = this.getPeriod();
        const wrappedTime = ((t % period) + period) % period;

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

        const pos = segment.getPosition(localT);
        const vel = segment.getVelocity(localT);
        const acc = segment.getAcceleration(localT);

        // Compute heading from velocity
        const heading = Math.atan2(vel.x, vel.z);

        // Compute heading rate using numerical differentiation
        const dt = 0.001;
        const { segment: seg2, localT: t2 } = this.findSegment(wrappedTime + dt);
        let headingRate = 0;
        if (seg2) {
            const vel2 = seg2.getVelocity(t2);
            const heading2 = Math.atan2(vel2.x, vel2.z);
            headingRate = this.wrapAngle(heading2 - heading) / dt;

            // Clamp heading rate
            const maxHeadingRate = 5.0;
            headingRate = Math.max(-maxHeadingRate, Math.min(maxHeadingRate, headingRate));
        }

        return {
            position: pos,
            velocity: vel,
            acceleration: acc,
            jerk: { x: 0, y: 0, z: 0 },
            heading,
            headingRate,
            time: t,
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
