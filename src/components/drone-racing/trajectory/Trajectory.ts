/**
 * Base Trajectory Interface and Abstract Class
 *
 * Defines the contract for all trajectory types and provides
 * common utility methods for trajectory generation.
 */

import { Waypoint } from '../types';

/**
 * Trajectory configuration shared by all trajectory types
 */
export interface TrajectoryParams {
    speed: number;      // Base speed in m/s
    height: number;     // Flight altitude in m
}

/**
 * Abstract base class for all trajectory types
 *
 * Provides:
 * - Common interface for waypoint generation
 * - Numerical differentiation utilities
 * - Trajectory visualization helpers
 */
export abstract class Trajectory {
    protected speed: number;
    protected height: number;

    constructor(params: TrajectoryParams) {
        this.speed = params.speed;
        this.height = params.height;
    }

    /**
     * Get the period (lap time) of the trajectory in seconds
     */
    abstract getPeriod(): number;

    /**
     * Get the name of this trajectory type
     */
    abstract getName(): string;

    /**
     * Get position at normalized time t ∈ [0, 1]
     * Returns { x, y, z } - subclasses can override y for 3D trajectories
     * Default y = this.height for 2D trajectories
     */
    protected abstract getPositionAtPhase(phase: number): { x: number; y?: number; z: number };

    /**
     * Get waypoint at time t
     */
    public getWaypoint(t: number): Waypoint {
        const period = this.getPeriod();
        const phase = (t % period) / period;

        // Get position (y defaults to this.height for 2D trajectories)
        const pos = this.getPositionAtPhase(phase);
        const y = pos.y !== undefined ? pos.y : this.height;

        // Compute velocity using central difference
        const dt = 0.0001;
        const phase1 = ((t - dt) % period + period) / period % 1;
        const phase2 = ((t + dt) % period) / period;
        const pos1 = this.getPositionAtPhase(phase1);
        const pos2 = this.getPositionAtPhase(phase2);

        const y1 = pos1.y !== undefined ? pos1.y : this.height;
        const y2 = pos2.y !== undefined ? pos2.y : this.height;

        let vx = (pos2.x - pos1.x) / (2 * dt);
        let vy = (y2 - y1) / (2 * dt);
        let vz = (pos2.z - pos1.z) / (2 * dt);

        // Normalize velocity to target speed (3D magnitude)
        const vMag = Math.sqrt(vx * vx + vy * vy + vz * vz);
        if (vMag > 1e-6) {
            const scale = this.speed / vMag;
            vx *= scale;
            vy *= scale;
            vz *= scale;
        }

        // Compute acceleration using central difference on velocity
        const dt2 = 0.001;
        const wp1 = this.getVelocityAtTime(t - dt2);
        const wp2 = this.getVelocityAtTime(t + dt2);
        const ax = (wp2.vx - wp1.vx) / (2 * dt2);
        const ay = (wp2.vy - wp1.vy) / (2 * dt2);
        const az = (wp2.vz - wp1.vz) / (2 * dt2);

        // Compute heading and heading rate (heading is in XZ plane)
        const heading = Math.atan2(vx, vz);
        const heading1 = Math.atan2(wp1.vx, wp1.vz);
        const heading2 = Math.atan2(wp2.vx, wp2.vz);
        let headingRate = this.wrapAngle(heading2 - heading1) / (2 * dt2);

        // Clamp heading rate to physically reasonable limits (max ~5 rad/s = 286°/s)
        const maxHeadingRate = 5.0;  // rad/s
        headingRate = Math.max(-maxHeadingRate, Math.min(maxHeadingRate, headingRate));

        return {
            position: { x: pos.x, y, z: pos.z },
            velocity: { x: vx, y: vy, z: vz },
            acceleration: { x: ax, y: ay, z: az },
            jerk: { x: 0, y: 0, z: 0 },
            heading,
            headingRate,
            time: t,
        };
    }

    /**
     * Helper to get velocity at a specific time (for numerical differentiation)
     */
    private getVelocityAtTime(t: number): { vx: number; vy: number; vz: number } {
        const period = this.getPeriod();
        const dt = 0.0001;
        const phase1 = ((t - dt) % period + period) / period % 1;
        const phase2 = ((t + dt) % period) / period;
        const pos1 = this.getPositionAtPhase(phase1);
        const pos2 = this.getPositionAtPhase(phase2);

        const y1 = pos1.y !== undefined ? pos1.y : this.height;
        const y2 = pos2.y !== undefined ? pos2.y : this.height;

        let vx = (pos2.x - pos1.x) / (2 * dt);
        let vy = (y2 - y1) / (2 * dt);
        let vz = (pos2.z - pos1.z) / (2 * dt);

        const vMag = Math.sqrt(vx * vx + vy * vy + vz * vz);
        if (vMag > 1e-6) {
            const scale = this.speed / vMag;
            vx *= scale;
            vy *= scale;
            vz *= scale;
        }

        return { vx, vy, vz };
    }

    /**
     * Wrap angle to [-π, π]
     */
    protected wrapAngle(angle: number): number {
        while (angle > Math.PI) angle -= 2 * Math.PI;
        while (angle < -Math.PI) angle += 2 * Math.PI;
        return angle;
    }

    /**
     * Get trajectory points for visualization
     */
    public getVisualizationPoints(numPoints: number = 200): { x: number; y: number; z: number }[] {
        const points: { x: number; y: number; z: number }[] = [];

        for (let i = 0; i <= numPoints; i++) {
            const phase = i / numPoints;
            const pos = this.getPositionAtPhase(phase);
            const y = pos.y !== undefined ? pos.y : this.height;
            points.push({ x: pos.x, y, z: pos.z });
        }

        return points;
    }

    /**
     * Get initial state for starting the trajectory
     */
    public getInitialState(): { position: { x: number; y: number; z: number }; heading: number } {
        const wp = this.getWaypoint(0);
        return {
            position: wp.position,
            heading: wp.heading,
        };
    }
}
