/**
 * Snake (Serpentine) Trajectory
 *
 * Sinusoidal weaving pattern - tests continuous direction changes.
 * The drone weaves left and right while progressing forward,
 * then makes a smooth U-turn and returns.
 *
 * Shape: Sine wave with smooth semicircular turnarounds at each end
 *
 *     /\    /\    /\__
 *    /  \  /  \  /    \  (turnaround)
 *   /    \/    \/      |
 *                      |
 *     /\    /\    /\__/
 *    /  \  /  \  /
 *   /    \/    \/
 */

import { Trajectory, TrajectoryParams } from './Trajectory';

export interface SnakeParams extends TrajectoryParams {
    amplitude: number;      // Lateral amplitude of weaving
    wavelength: number;     // Distance between wave peaks
    numWaves: number;       // Number of complete sine waves
    turnRadius: number;     // Radius of turnaround at ends
}

export const DEFAULT_SNAKE_PARAMS: SnakeParams = {
    speed: 10.0,            // Moderate speed
    height: 4.0,
    amplitude: 6.0,         // Gentler weaving
    wavelength: 25.0,       // Longer wavelength for smoother path
    numWaves: 2,
    turnRadius: 8.0,        // Smooth turnaround radius
};

export class SnakeTrajectory extends Trajectory {
    private amplitude: number;
    private wavelength: number;
    private numWaves: number;
    private turnRadius: number;
    private pathLength: number;
    private arcLength: number;
    private turnArc: number;
    private totalLength: number;

    constructor(params: Partial<SnakeParams> = {}) {
        const fullParams = { ...DEFAULT_SNAKE_PARAMS, ...params };
        super(fullParams);
        this.amplitude = fullParams.amplitude;
        this.wavelength = fullParams.wavelength;
        this.numWaves = fullParams.numWaves;
        this.turnRadius = fullParams.turnRadius;

        // Straight path length
        this.pathLength = this.numWaves * this.wavelength;

        // Arc length of sine wave portion
        const k = 2 * Math.PI / this.wavelength;
        this.arcLength = this.pathLength * Math.sqrt(1 + 0.5 * Math.pow(k * this.amplitude, 2));

        // Semicircle turnaround arc
        this.turnArc = Math.PI * this.turnRadius;

        // Total: forward snake + turn + backward snake + turn
        this.totalLength = 2 * this.arcLength + 2 * this.turnArc;
    }

    public getName(): string {
        return 'Snake';
    }

    public getPeriod(): number {
        return this.totalLength / this.speed;
    }

    protected getPositionAtPhase(phase: number): { x: number; z: number } {
        // Segment boundaries (as fractions of total length)
        const seg1 = this.arcLength / this.totalLength;              // End of forward snake
        const seg2 = (this.arcLength + this.turnArc) / this.totalLength;  // End of first turn
        const seg3 = (2 * this.arcLength + this.turnArc) / this.totalLength;  // End of backward snake
        // seg4 = 1.0 (end of second turn)

        const halfPath = this.pathLength / 2;
        const k = 2 * Math.PI / this.wavelength;

        if (phase < seg1) {
            // Segment 1: Forward snake (going +Z)
            const progress = phase / seg1;  // 0 to 1
            const z = -halfPath + progress * this.pathLength;
            const x = this.amplitude * Math.sin(k * z);
            return { x, z };
        } else if (phase < seg2) {
            // Segment 2: Turnaround at +Z end (semicircle going right then back)
            const turnProgress = (phase - seg1) / (seg2 - seg1);  // 0 to 1
            const angle = turnProgress * Math.PI;  // 0 to π
            // End of snake is at z = halfPath, x = amplitude * sin(k * halfPath)
            const endX = this.amplitude * Math.sin(k * halfPath);
            const centerX = endX + this.turnRadius;  // Center of turn to the right
            const centerZ = halfPath;
            return {
                x: centerX - this.turnRadius * Math.cos(angle),
                z: centerZ + this.turnRadius * Math.sin(angle),
            };
        } else if (phase < seg3) {
            // Segment 3: Backward snake (going -Z), offset to the right
            const progress = (phase - seg2) / (seg3 - seg2);
            const z = halfPath - progress * this.pathLength;
            // Mirror the x position around the turn center
            const endX = this.amplitude * Math.sin(k * halfPath);
            const offsetX = 2 * this.turnRadius + 2 * endX;
            return {
                x: offsetX - this.amplitude * Math.sin(k * z),
                z,
            };
        } else {
            // Segment 4: Turnaround at -Z end (semicircle going left then forward)
            const turnProgress = (phase - seg3) / (1 - seg3);  // 0 to 1
            const angle = turnProgress * Math.PI;  // 0 to π
            // End position calculation
            const endX = this.amplitude * Math.sin(k * halfPath);
            const offsetX = 2 * this.turnRadius + 2 * endX;
            const returnEndX = offsetX - this.amplitude * Math.sin(k * (-halfPath));
            const centerX = returnEndX - this.turnRadius;
            const centerZ = -halfPath;
            return {
                x: centerX + this.turnRadius * Math.cos(angle),
                z: centerZ - this.turnRadius * Math.sin(angle),
            };
        }
    }
}
