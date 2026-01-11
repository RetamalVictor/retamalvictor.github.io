/**
 * 3D Racing Trajectory
 *
 * Full 3D racing course with altitude changes, dives, and climbs.
 * Simulates real drone racing with gates at different heights.
 *
 * Features:
 * - Dive sections (descending while moving forward)
 * - Climb sections (ascending while moving forward)
 * - Split-S style maneuvers (over-under obstacles)
 * - Varying altitude gates
 *
 * Track profile (side view):
 *
 *           /\
 *          /  \____
 *         /        \
 *    ____/          \____
 *                        \
 *                         \___/
 */

import { Trajectory, TrajectoryParams } from './Trajectory';
import { GatePosition } from '../types';

export interface Racing3DParams extends TrajectoryParams {
    trackLength: number;    // Total XZ track length
    minHeight: number;      // Minimum altitude
    maxHeight: number;      // Maximum altitude
    numGates: number;       // Number of gates/sections
}

export const DEFAULT_RACING3D_PARAMS: Racing3DParams = {
    speed: 15.0,            // Racing speed
    height: 5.0,            // Base/average height
    trackLength: 100.0,     // 100m track
    minHeight: 2.0,         // Low gates at 2m
    maxHeight: 10.0,        // High gates at 10m
    numGates: 6,            // 6 gate sections
};

/**
 * 3D point for track definition
 */
interface TrackPoint {
    x: number;
    y: number;
    z: number;
    t: number;  // Normalized parameter [0, 1]
}

export class Racing3DTrajectory extends Trajectory {
    private trackLength: number;
    private minHeight: number;
    private maxHeight: number;
    private trackPoints: TrackPoint[] = [];
    private totalArcLength: number = 0;

    constructor(params: Partial<Racing3DParams> = {}) {
        const fullParams = { ...DEFAULT_RACING3D_PARAMS, ...params };
        super(fullParams);
        this.trackLength = fullParams.trackLength;
        this.minHeight = fullParams.minHeight;
        this.maxHeight = fullParams.maxHeight;

        this.buildTrack();
    }

    public getName(): string {
        return '3D Racing';
    }

    public getPeriod(): number {
        return this.totalArcLength / this.speed;
    }

    /**
     * Build the 3D track with varying altitude
     */
    private buildTrack(): void {
        // Create a closed loop track with altitude variations
        const points: TrackPoint[] = [];
        const segments = 100;  // High resolution for smooth interpolation

        // Track layout: oval with altitude profile
        // The track goes: start -> climb -> high section -> dive -> low section -> climb back

        for (let i = 0; i <= segments; i++) {
            const t = i / segments;
            const angle = t * 2 * Math.PI;

            // Oval shape in XZ plane
            const radiusX = this.trackLength / 3;
            const radiusZ = this.trackLength / 4;
            const x = radiusX * Math.sin(angle);
            const z = radiusZ * Math.cos(angle);

            // Altitude profile: combination of sine waves for interesting 3D path
            // Main altitude variation (one full cycle per lap)
            const altBase = (this.maxHeight + this.minHeight) / 2;
            const altAmp = (this.maxHeight - this.minHeight) / 2;

            // Primary wave: goes high on one side, low on the other
            const primaryWave = Math.sin(angle) * altAmp * 0.7;

            // Secondary wave: adds a "hill" and "valley" for gates
            const secondaryWave = Math.sin(2 * angle) * altAmp * 0.3;

            // Dive section (sharp descent between 60% and 75% of track)
            let diveBonus = 0;
            if (t > 0.6 && t < 0.75) {
                const diveProgress = (t - 0.6) / 0.15;
                diveBonus = -Math.sin(diveProgress * Math.PI) * altAmp * 0.5;
            }

            const y = altBase + primaryWave + secondaryWave + diveBonus;

            points.push({ x, y, z, t });
        }

        // Compute arc lengths and total length
        this.totalArcLength = 0;
        for (let i = 1; i < points.length; i++) {
            const dx = points[i].x - points[i - 1].x;
            const dy = points[i].y - points[i - 1].y;
            const dz = points[i].z - points[i - 1].z;
            this.totalArcLength += Math.sqrt(dx * dx + dy * dy + dz * dz);
        }

        this.trackPoints = points;
    }

    protected getPositionAtPhase(phase: number): { x: number; y: number; z: number } {
        // Interpolate between track points
        // Handle phase values at or near 1.0
        let t = phase % 1;
        if (t < 0) t += 1;

        const numPoints = this.trackPoints.length;
        if (numPoints === 0) {
            return { x: 0, y: this.height, z: 0 };
        }

        const idx = t * (numPoints - 1);
        const i0 = Math.min(Math.floor(idx), numPoints - 1);
        const i1 = Math.min(i0 + 1, numPoints - 1);
        const frac = idx - Math.floor(idx);

        const p0 = this.trackPoints[i0];
        const p1 = this.trackPoints[i1];

        return {
            x: p0.x + frac * (p1.x - p0.x),
            y: p0.y + frac * (p1.y - p0.y),
            z: p0.z + frac * (p1.z - p0.z),
        };
    }

    /**
     * Place gates at key altitude change points around the 3D track
     */
    public override getGatePositions(): GatePosition[] {
        const period = this.getPeriod();

        // Place 6 gates at even intervals around the track
        // This includes points at different altitudes
        const numGates = 6;
        const gates: GatePosition[] = [];

        for (let i = 0; i < numGates; i++) {
            const phase = i / numGates;
            const wp = this.getWaypoint(phase * period);
            gates.push({
                position: { ...wp.position },
                heading: wp.heading,
            });
        }

        return gates;
    }
}
