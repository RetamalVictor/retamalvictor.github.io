/**
 * Race Track Trajectory
 *
 * Multi-gate racing course combining straights, turns, and chicanes.
 * Tests all aspects of drone control in a realistic racing scenario.
 *
 * Track layout (top view):
 *
 *     START
 *        |
 *        v
 *    +-------+
 *    |       |
 *    |   1   |----+
 *    |       |    |
 *    +-------+    |
 *                 |
 *        +--------+
 *        |
 *    +---+---+
 *    |       |
 *    |   2   |
 *    |       |
 *    +---+---+
 *        |
 *        +--------+
 *                 |
 *    +-------+    |
 *    |       |----+
 *    |   3   |
 *    |       |
 *    +---+---+
 *        |
 *        v
 *     FINISH (= START)
 */

import { Trajectory, TrajectoryParams } from './Trajectory';

export interface RaceTrackParams extends TrajectoryParams {
    gateSpacing: number;    // Distance between gates
    turnRadius: number;     // Radius of turns
}

export const DEFAULT_RACETRACK_PARAMS: RaceTrackParams = {
    speed: 18.0,
    height: 4.0,
    gateSpacing: 25.0,
    turnRadius: 10.0,
};

/**
 * Segment types for track definition
 */
type SegmentType = 'straight' | 'turn';

interface Segment {
    type: SegmentType;
    length: number;         // Arc length of segment
    startPos: { x: number; z: number };
    startHeading: number;
    turnDirection?: 1 | -1; // 1 = right, -1 = left
    turnAngle?: number;     // Total turn angle in radians
    turnRadius?: number;
}

export class RaceTrackTrajectory extends Trajectory {
    private gateSpacing: number;
    private turnRadius: number;
    private segments: Segment[] = [];
    private totalLength: number = 0;

    constructor(params: Partial<RaceTrackParams> = {}) {
        const fullParams = { ...DEFAULT_RACETRACK_PARAMS, ...params };
        super(fullParams);
        this.gateSpacing = fullParams.gateSpacing;
        this.turnRadius = fullParams.turnRadius;

        this.buildTrack();
    }

    public getName(): string {
        return 'Race Track';
    }

    public getPeriod(): number {
        return this.totalLength / this.speed;
    }

    /**
     * Build the race track from segments
     */
    private buildTrack(): void {
        const G = this.gateSpacing;
        const R = this.turnRadius;
        const segments: Segment[] = [];

        let x = 0, z = 0, heading = 0;

        // Helper to add straight segment
        const addStraight = (length: number) => {
            segments.push({
                type: 'straight',
                length,
                startPos: { x, z },
                startHeading: heading,
            });
            x += length * Math.sin(heading);
            z += length * Math.cos(heading);
            this.totalLength += length;
        };

        // Helper to add turn segment
        const addTurn = (angle: number, direction: 1 | -1) => {
            const arcLength = Math.abs(angle) * R;
            segments.push({
                type: 'turn',
                length: arcLength,
                startPos: { x, z },
                startHeading: heading,
                turnDirection: direction,
                turnAngle: angle,
                turnRadius: R,
            });

            // Update position: center of turn is R units perpendicular to heading
            const centerX = x + R * direction * Math.cos(heading);
            const centerZ = z - R * direction * Math.sin(heading);

            // End position
            heading += angle * direction;
            x = centerX - R * direction * Math.cos(heading);
            z = centerZ + R * direction * Math.sin(heading);

            this.totalLength += arcLength;
        };

        // Build the track
        // Start: going +Z
        heading = 0;
        x = G / 2;
        z = -G;

        // Segment 1: Straight to first gate
        addStraight(G);

        // Turn right 90°
        addTurn(Math.PI / 2, 1);

        // Straight to side
        addStraight(G);

        // Turn right 90°
        addTurn(Math.PI / 2, 1);

        // Straight back
        addStraight(G * 1.5);

        // Turn left 90°
        addTurn(Math.PI / 2, -1);

        // Straight across
        addStraight(G);

        // Turn left 90°
        addTurn(Math.PI / 2, -1);

        // Straight forward
        addStraight(G * 1.5);

        // Turn right 90°
        addTurn(Math.PI / 2, 1);

        // Straight to side
        addStraight(G);

        // Turn right 90°
        addTurn(Math.PI / 2, 1);

        // Final straight back to start
        addStraight(G);

        this.segments = segments;
    }

    protected getPositionAtPhase(phase: number): { x: number; z: number } {
        const distance = phase * this.totalLength;

        // Find which segment we're in
        let accumulated = 0;
        for (const segment of this.segments) {
            if (accumulated + segment.length >= distance) {
                const localDist = distance - accumulated;
                return this.getPositionInSegment(segment, localDist);
            }
            accumulated += segment.length;
        }

        // Should not reach here, but return start position
        return this.segments[0].startPos;
    }

    private getPositionInSegment(segment: Segment, localDistance: number): { x: number; z: number } {
        if (segment.type === 'straight') {
            const { startPos, startHeading } = segment;
            return {
                x: startPos.x + localDistance * Math.sin(startHeading),
                z: startPos.z + localDistance * Math.cos(startHeading),
            };
        } else {
            // Turn segment
            const { startPos, startHeading, turnDirection, turnRadius } = segment;
            const R = turnRadius!;
            const dir = turnDirection!;

            // Center of turn circle
            const centerX = startPos.x + R * dir * Math.cos(startHeading);
            const centerZ = startPos.z - R * dir * Math.sin(startHeading);

            // Angle traveled
            const angleTraveled = localDistance / R;
            const currentHeading = startHeading + angleTraveled * dir;

            // Position on circle
            return {
                x: centerX - R * dir * Math.cos(currentHeading),
                z: centerZ + R * dir * Math.sin(currentHeading),
            };
        }
    }
}
